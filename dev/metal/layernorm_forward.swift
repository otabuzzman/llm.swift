// swiftlint:disable:next blanket_disable_command
// swiftlint:disable identifier_name

/// Kernel benchmark for the positional layernorm forward pass in GPT-2.
///
/// Kernels (Metal shaders) are in `DefaultLibrary.swift´
///
/// Compile:
/// xcodebuild -scheme layer_pass -configuration Release \
///   SWIFT_ACTIVE_COMPILATION_CONDITIONS="$SWIFT_ACTIVE_COMPILATION_CONDITIONS LAYER_PASS_STANDALONE"
///
/// version 1 naive drag and drop implementation into kernel, parallelize over B,T, loop over C
/// ./layernorm_forward 1
///
/// version 2 parallelizes over all of B,T,C
/// ./layernorm_forward 2
///
/// version 3 uses cooperative groups to parallelize over all of B,T,C
/// ./layernorm_forward 3
///
/// version 4 uses a more clever way to estimate variance, var(x) = mean(x**2) - mean(x)**2
///           (allowing us to do a single pass over x on load)
/// ./layernorm_forward 4
///
/// version 5 allocates blocks per row instead of warps per row, same alg as 4 otherwise
/// ./layernorm_forward 5
///
/// version 6 inspired by `fused_residual_forward_kernel5´ in `fused_residual_forward.cu´
/// ./layernorm_forward 6

import Metal

// known kernel (Metal shader) versions
private let versions = 1...6

// shader specific launch stub
// swiftlint:disable:next function_parameter_count
func layernorm_forward1(
    _ out: UnsafeMutablePointer<Float>,
    _ mean: UnsafeMutablePointer<Float>,
    _ rstd: UnsafeMutablePointer<Float>,
    _ inp: UnsafePointer<Float>,
    _ weight: UnsafePointer<Float>,
    _ bias: UnsafePointer<Float>,
    _ B: Int, _ T: Int, _ C: Int,
    _ block_size: Int = 0) throws {
    let context = KernelContext(threadsPerGrid: B * T, threadsPerGroup: block_size)

    let params: [KernelParam] = [
        UnsafeMutableRawPointer(out),
        UnsafeMutableRawPointer(mean),
        UnsafeMutableRawPointer(rstd),
        UnsafeMutableRawPointer(mutating: inp),
        UnsafeMutableRawPointer(mutating: weight),
        UnsafeMutableRawPointer(mutating: bias),
        Int32(B), Int32(T), Int32(C)]

    try launchPad?.dispatchKernel(
        name: "layernorm_forward_kernel1",
        context: context,
        params: params)
}

// version dispatcher
// swiftlint:disable:next function_parameter_count
private func layernorm_forward(
    _ version: Int,
    _ out: UnsafeMutablePointer<Float>,
    _ mean: UnsafeMutablePointer<Float>,
    _ rstd: UnsafeMutablePointer<Float>,
    _ inp: UnsafePointer<Float>,
    _ weight: UnsafePointer<Float>,
    _ bias: UnsafePointer<Float>,
    _ B: Int, _ T: Int, _ C: Int,
    _ block_size: Int = 0) throws {
    guard
        versions.contains(version) == true
    else { throw LlmSwiftError.wrongApiUsage(api: "\(#function) version \(version) unknown") }

    switch version {
    case 1:
        try layernorm_forward1(out, mean, rstd, inp, weight, bias, B, T, C, block_size)
    case 2, 3, 4, 5, 6:
        fatalError("layer-pass function \(#function) version \(version) not implemented")
    default:
        break
    }
}

// standalone runner
// swiftlint:disable:next function_body_length
func layernorm_forward(_ argc: Int, _ argv: [String]) throws {
    let B = 8
    let T = 1024
    let C = 768

    try launchPad?.registerKernel(name: "layernorm_forward_kernel1")

    // create memory of random numbers
    let out_cpu = UnsafeMutablePointer<Float>.allocate(capacity: B * T * C)
    let mean_cpu = UnsafeMutablePointer<Float>.allocate(capacity: B * T)
    let rstd_cpu = UnsafeMutablePointer<Float>.allocate(capacity: B * T)

    let out_gpu = UnsafeMutablePointer<Float>.allocate(capacity: B * T * C)
    let out_gpu_length = B * T * C * MemoryLayout<Float>.size
    try launchPad?.registerBuffer(address: out_gpu, length: out_gpu_length)

    let mean_gpu = UnsafeMutablePointer<Float>.allocate(capacity: B * T)
    let mean_gpu_length = B * T * MemoryLayout<Float>.size
    try launchPad?.registerBuffer(address: mean_gpu, length: mean_gpu_length)

    let rstd_gpu = UnsafeMutablePointer<Float>.allocate(capacity: B * T)
    let rstd_gpu_length = B * T * MemoryLayout<Float>.size
    try launchPad?.registerBuffer(address: rstd_gpu, length: rstd_gpu_length)

    let inp = UnsafeMutablePointer<Float>.allocate(capacity: B * T * C)
    for i in 0..<B * T * C { inp[i] = Float.random(in: -1.0...1.0) }
    let inp_length = B * T * C * MemoryLayout<Float>.size
    try launchPad?.registerBuffer(address: inp, length: inp_length)

    let weight = UnsafeMutablePointer<Float>.allocate(capacity: C)
    for i in 0..<C { weight[i] = Float.random(in: -1.0...1.0) }
    let weight_length = C * MemoryLayout<Float>.size
    try launchPad?.registerBuffer(address: weight, length: weight_length)

    let bias = UnsafeMutablePointer<Float>.allocate(capacity: C)
    for i in 0..<C { bias[i] = Float.random(in: -1.0...1.0) }
    let bias_length = C * MemoryLayout<Float>.size
    try launchPad?.registerBuffer(address: bias, length: bias_length)

    defer {
        out_cpu.deallocate()
        mean_cpu.deallocate()
        rstd_cpu.deallocate()
        launchPad?.unregisterBuffer(address: out_gpu)
        out_gpu.deallocate()
        launchPad?.unregisterBuffer(address: mean_gpu)
        mean_gpu.deallocate()
        launchPad?.unregisterBuffer(address: rstd_gpu)
        rstd_gpu.deallocate()
        launchPad?.unregisterBuffer(address: inp)
        inp.deallocate()
        launchPad?.unregisterBuffer(address: weight)
        weight.deallocate()
        launchPad?.unregisterBuffer(address: bias)
        bias.deallocate()
    }

    // read kernel_num from command line
    var kernel_num = 1
    if argv.count > 1 {
        kernel_num = Int(argv[1]) ?? 2
    }
    print("Using kernel \(kernel_num)")

    // first check the correctness of the kernel
    layernorm_forward(out_cpu, mean_cpu, rstd_cpu, inp, weight, bias, B, T, C)

    // time the kernel at different block sizes
    let block_sizes = [0, 64, 128, 256, 512, 1024]
    for block_size in block_sizes {
        print("Checking block size \(block_size)\(block_size == 0 ? " (computed)" : "")")
        try layernorm_forward(kernel_num, out_gpu, mean_gpu, rstd_gpu, inp, weight, bias, B, T, C, block_size)
        try launchPad?.commit(wait: true)
        let tol: Float = 1e-5
        try validate_result(out_gpu, out_cpu, "out", B * T * C, tol)
        try validate_result(mean_gpu, mean_cpu, "mean", B * T, tol)
        try validate_result(rstd_gpu, rstd_cpu, "rstd", B * T, tol)
    }

    print("All results match. Starting benchmarks.\n")

    let repeat_times = 2000
    var elapsed_time: Double = 0
    // CPU for comparison
    for _ in 0..<repeat_times {
        let start = Date()
        layernorm_forward(out_cpu, mean_cpu, rstd_cpu, inp, weight, bias, B, T, C)
        let end = Date()
        elapsed_time += end.timeIntervalSince(start)
    }
    elapsed_time /= Double(repeat_times)
    elapsed_time *= 1e3 // ms

    print("CPU time \(String(format: "%.4f", elapsed_time)) ms")

    for block_size in block_sizes {
        // omitted generic `benchmark_kernel´ in dev/cuda/common.h
        elapsed_time = 0
        for _ in 0..<repeat_times {
            // clear L2
            // TODO: if necessary and applicable

            let start = Date()
            try layernorm_forward(kernel_num, out_gpu, mean_gpu, rstd_gpu, inp, weight, bias, B, T, C, block_size)
            let end = Date()
            elapsed_time += end.timeIntervalSince(start)
        }
        elapsed_time *= 1e3 // ms
        elapsed_time /= Double(repeat_times)

        // napkin math: estimate the memory bandwidth achieved
        // and e.g. A100 40GB PCIe is advertised at 1,555GB/s
        let memory_ops = 2 * B * T * C * 4 // *4 for float
        let memory_bandwidth = Double(memory_ops) / elapsed_time / 1e6

        print("block_size \(String(format: "%4d", block_size)) | time \(String(format: "%.4f", elapsed_time)) ms | bandwidth n/a (unified)")
    }
}
