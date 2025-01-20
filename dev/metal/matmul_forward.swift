// swiftlint:disable:next blanket_disable_command
// swiftlint:disable identifier_name

/// Kernel benchmark for the positional matmul forward pass in GPT-2.
///
/// Kernels (Metal shaders) are in `DefaultLibrary.swift´
///
/// Compile:
/// xcodebuild -scheme layer_pass -configuration Release \
///   SWIFT_ACTIVE_COMPILATION_CONDITIONS="$SWIFT_ACTIVE_COMPILATION_CONDITIONS LAYER_PASS_STANDALONE"
///
/// version 1 is naive port from CPU code to kernel: parallelizes over B,T, loops over C
/// ./matmul_forward 1
///
/// version 2 calls cuBLAS, very fast
/// ./matmul_forward 2
///
/// version 3 calls cuBLASLt, should be even faster
/// ./matmul_forward 3
///
/// version 4 handwritten, relatively efficient non-tensorcore matmul kernel
/// ./matmul_forward 4

import Metal

// known kernel (Metal shader) versions
private let versions = 1...4

// overwrite async CPU version in `train_gpt2.swift´
// swiftlint:disable:next function_parameter_count
func matmul_forward(
    _ out: UnsafeMutablePointer<Float>,
    _ inp: UnsafePointer<Float>,
    _ weight: UnsafePointer<Float>,
    _ bias: UnsafePointer<Float>?,
    _ B: Int, _ T: Int, _ C: Int, _ OC: Int) {
    // the most naive implementation of matrix multiplication
    // this serves as an algorithmic reference, and as a fallback for
    // unfriendly input shapes inside matmul_forward(), below.
    // #pragma omp parallel for collapse(2)
    for bt in 0..<B * T {
        for o in 0..<OC {
            var val = bias?[o] ?? 0
            for i in 0..<C {
                val += inp[bt * C + i] * weight[o * C + i]
            }
            out[bt * OC + o] = val
        }
    }
}

// swiftlint:disable:next function_parameter_count
func matmul_forward1(
    _ out: UnsafeMutablePointer<Float>,
    _ inp: UnsafePointer<Float>,
    _ weight: UnsafePointer<Float>,
//    _ bias: UnsafePointer<Float>?,
    _ bias: UnsafePointer<Float>,
    _ B: Int, _ T: Int, _ C: Int, _ OC: Int,
    _ block_size: Int = 0) throws {
        let context = KernelContext(threadsPerGrid: B * T * OC, threadsPerGroup: block_size)

    let params: [KernelParam] = [
        UnsafeMutableRawPointer(out),
        UnsafeMutableRawPointer(mutating: inp),
        UnsafeMutableRawPointer(mutating: weight),
        UnsafeMutableRawPointer(mutating: bias),
        Int32(B * T), Int32(C), Int32(OC)]

    try launchPad?.dispatchKernel(
        name: "matmul_forward_kernel1",
        context: context,
        params: params)
}

// swiftlint:disable:next function_parameter_count
private func matmul_forward(
    _ version: Int,
    _ out: UnsafeMutablePointer<Float>,
    _ inp: UnsafePointer<Float>,
    _ weight: UnsafePointer<Float>,
    _ bias: UnsafePointer<Float>,
    _ B: Int, _ T: Int, _ C: Int, _ OC: Int,
    _ block_size: Int = 0) throws {
    guard
        versions.contains(version) == true
    else { throw LlmSwiftError.wrongApiUsage(api: "\(#function) version \(version) unknown") }

    switch version {
    case 1:
        try matmul_forward1(out, inp, weight, bias, B, T, C, OC, block_size)
    case 2, 3, 4:
        fatalError("layer-pass function \(#function) version \(version) not implemented")
    default:
        break
    }
}

// swiftlint:disable:next function_body_length
func matmul_forward(_ argc: Int, _ argv: [String]) throws {
    let B = 32
    let T = 1024
    let C = 768
    let OC = 768 * 4 // expansion of 4, e.g. in the MLP

    try launchPad?.registerKernel(name: "matmul_forward_kernel1")

    // create memory of random numbers
    let out_cpu = UnsafeMutablePointer<Float>.allocate(capacity: B * T * OC)

    let out_gpu = UnsafeMutablePointer<Float>.allocate(capacity: B * T * OC)
    let out_gpu_length = B * T * OC * MemoryLayout<Float>.size
    try launchPad?.registerBuffer(address: out_gpu, length: out_gpu_length)

    let inp = UnsafeMutablePointer<Float>.allocate(capacity: B * T * C)
    for i in 0..<B * T * C { inp[i] = Float.random(in: -1.0...1.0) }
    let inp_length = B * T * C * MemoryLayout<Float>.size
    try launchPad?.registerBuffer(address: inp, length: inp_length)

    let weight = UnsafeMutablePointer<Float>.allocate(capacity: C * OC)
    for i in 0..<C * OC { weight[i] = Float.random(in: -1.0...1.0) }
    let weight_length = C * OC * MemoryLayout<Float>.size
    try launchPad?.registerBuffer(address: weight, length: weight_length)

    let bias = UnsafeMutablePointer<Float>.allocate(capacity: OC)
    for i in 0..<OC { bias[i] = Float.random(in: -1.0...1.0) }
    let bias_length = OC * MemoryLayout<Float>.size
    try launchPad?.registerBuffer(address: bias, length: bias_length)

    defer {
        out_cpu.deallocate()
        launchPad?.unregisterBuffer(address: out_gpu)
        out_gpu.deallocate()
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
    matmul_forward(out_cpu, inp, weight, bias, B, T, C, OC)

    // time the kernel at different block sizes
    let sqrt_block_sizes = [0, 4, 8, 16, 32]
    for sqrt_block_size in sqrt_block_sizes {
        print("Checking block size \(sqrt_block_size)\(sqrt_block_size == 0 ? " (computed)" : "")")
        try matmul_forward(kernel_num, out_gpu, inp, weight, bias, B, T, C, OC, sqrt_block_size)
        try launchPad?.commit(wait: true)
        let tol: Float = 1e-5
        try validate_result(out_gpu, out_cpu, "out", B * T * OC, tol)
    }

    print("All results match. Starting benchmarks.\n")

    var elapsed_time: Double = 0
    // CPU for comparison
    let start = Date()
    matmul_forward(out_cpu, inp, weight, bias, B, T, C, OC)
    let end = Date()
    elapsed_time = end.timeIntervalSince(start)
    elapsed_time *= 1e3 // ms

    print("CPU time \(String(format: "%.4f", elapsed_time)) ms")

    for sqrt_block_size in sqrt_block_sizes {
        let repeat_times = 100
        // omitted generic `benchmark_kernel´ in dev/cuda/common.h
        elapsed_time = 0
        for _ in 0..<repeat_times {
            // clear L2
            // TODO: if necessary and applicable

            let start = Date()
            try matmul_forward(kernel_num, out_gpu, inp, weight, bias, B, T, C, OC, sqrt_block_size)
            let end = Date()
            elapsed_time += end.timeIntervalSince(start)
        }
        elapsed_time /= Double(repeat_times)
        elapsed_time *= 1e3 // ms

        // napkin math: estimate the flops achieved
        // e.g. A100 40GB PCIe is advertised at 19.5 TFLOPS fp32
        let tflops = Double(B * T * C * OC * 2) / elapsed_time * 1e3 / 1e12

        print("block_size \(sqrt_block_size) | time \(String(format: "%.4f", elapsed_time)) ms | tflops \(String(format: "%.2f", tflops))")
    }
}
