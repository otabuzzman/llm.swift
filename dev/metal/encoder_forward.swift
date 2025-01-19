// swiftlint:disable:next blanket_disable_command
// swiftlint:disable identifier_name

/// Kernel benchmark for the positional encoder forward pass in GPT-2.
///
/// Kernels (Metal shaders) are in `DefaultLibrary.swift´
///
/// Compile:
/// xcodebuild -scheme encoder_forward -configuration Release \
///   SWIFT_ACTIVE_COMPILATION_CONDITIONS="$SWIFT_ACTIVE_COMPILATION_CONDITIONS LAYER_PASS_STANDALONE"
///
/// version 1 is naive port from CPU code to kernel: parallelizes over B,T, loops over C
/// ./encoder_forward 1
///
/// version 2 is more optimized, parallelizes over all of B,T,C
/// ./encoder_forward 2
///
/// version 3 is like version 2 but uses float4 reads/writes
/// ./encoder_forward 3

import Metal

// known kernel (Metal shader) versions
private let versions = 1...3

// swiftlint:disable:next function_parameter_count
func encoder_forward1(
    _ out: UnsafeMutablePointer<Float>,
    _ inp: UnsafePointer<Int32>,
    _ wte: UnsafePointer<Float>,
    _ wpe: UnsafePointer<Float>,
    _ B: Int, _ T: Int, _ C: Int,
    _ block_size: Int = 0) throws {
    let context = KernelContext(threadsPerGrid: B * T, threadsPerGroup: block_size)

    let params: [KernelParam] = [
        UnsafeMutableRawPointer(out),
        UnsafeMutableRawPointer(mutating: inp),
        UnsafeMutableRawPointer(mutating: wte),
        UnsafeMutableRawPointer(mutating: wpe),
        Int32(B), Int32(T), Int32(C)]

    try launchPad?.dispatchKernel(
        name: "encoder_forward_kernel1",
        context: context,
        params: params)
}

// swiftlint:disable:next function_parameter_count
func encoder_forward2(
    _ out: UnsafeMutablePointer<Float>,
    _ inp: UnsafePointer<Int32>,
    _ wte: UnsafePointer<Float>,
    _ wpe: UnsafePointer<Float>,
    _ B: Int, _ T: Int, _ C: Int,
    _ block_size: Int = 0) throws {
    let context = KernelContext(threadsPerGrid: (B * T * C) / 4, threadsPerGroup: block_size)

    let params: [KernelParam] = [
        UnsafeMutableRawPointer(out),
        UnsafeMutableRawPointer(mutating: inp),
        UnsafeMutableRawPointer(mutating: wte),
        UnsafeMutableRawPointer(mutating: wpe),
        Int32(B), Int32(T), Int32(C)]

    try launchPad?.dispatchKernel(
        name: "encoder_forward_kernel2",
        context: context,
        params: params)
}

// swiftlint:disable:next function_parameter_count
func encoder_forward3(
    _ out: UnsafeMutablePointer<Float>,
    _ inp: UnsafePointer<Int32>,
    _ wte: UnsafePointer<Float>,
    _ wpe: UnsafePointer<Float>,
    _ B: Int, _ T: Int, _ C: Int,
    _ block_size: Int = 0) throws {
    let context = KernelContext(threadsPerGrid: B * T * C, threadsPerGroup: block_size)

    let params: [KernelParam] = [
        UnsafeMutableRawPointer(out),
        UnsafeMutableRawPointer(mutating: inp),
        UnsafeMutableRawPointer(mutating: wte),
        UnsafeMutableRawPointer(mutating: wpe),
        Int32(B), Int32(T), Int32(C)]

    try launchPad?.dispatchKernel(
        name: "encoder_forward_kernel3",
        context: context,
        params: params)
}

// swiftlint:disable:next function_parameter_count
private func encoder_forward(
    _ version: Int,
    _ out: UnsafeMutablePointer<Float>,
    _ inp: UnsafePointer<Int32>,
    _ wte: UnsafePointer<Float>,
    _ wpe: UnsafePointer<Float>,
    _ B: Int, _ T: Int, _ C: Int,
    _ block_size: Int = 0) throws {
    guard
        versions.contains(version) == true
    else { throw LlmSwiftError.wrongApiUsage(api: "\(#function) version \(version)") }

    switch version {
    case 1:
        try encoder_forward1(out, inp, wte, wpe, B, T, C, block_size)
    case 2:
        try encoder_forward2(out, inp, wte, wpe, B, T, C, block_size)
    case 3:
        try encoder_forward3(out, inp, wte, wpe, B, T, C, block_size)
    default:
        break
    }
}

// swiftlint:disable:next function_body_length
func encoder_forward(_ argc: Int, _ argv: [String]) throws {
    let B = 8
    let T = 1024
    let C = 768
    let V = 50257

    try launchPad?.registerKernel(name: "encoder_forward_kernel1")
    try launchPad?.registerKernel(name: "encoder_forward_kernel2")
    try launchPad?.registerKernel(name: "encoder_forward_kernel3")

    // create memory of random numbers
    let out_cpu = UnsafeMutablePointer<Float>.allocate(capacity: B * T * C)

    let out_gpu = UnsafeMutablePointer<Float>.allocate(capacity: B * T * C)
    let out_gpu_length = B * T * C * MemoryLayout<Float>.size
    try launchPad?.registerBuffer(address: out_gpu, length: out_gpu_length)

    let inp = UnsafeMutablePointer<Int32>.allocate(capacity: B * T)
    for i in 0..<B * T { inp[i] = Int32(Int.random(in: 0..<V)) }
    let inp_length = B * T * MemoryLayout<Int32>.size
    try launchPad?.registerBuffer(address: inp, length: inp_length)

    let wte = UnsafeMutablePointer<Float>.allocate(capacity: V * C)
    for i in 0..<V * C { wte[i] = Float.random(in: -1.0...1.0) }
    let wte_length = V * C * MemoryLayout<Float>.size
    try launchPad?.registerBuffer(address: wte, length: wte_length)

    let wpe = UnsafeMutablePointer<Float>.allocate(capacity: T * C)
    for i in 0..<T * C { wpe[i] = Float.random(in: -1.0...1.0) }
    let wpe_length = T * C * MemoryLayout<Float>.size
    try launchPad?.registerBuffer(address: wpe, length: wpe_length)

    defer {
        out_cpu.deallocate()
        launchPad?.unregisterBuffer(address: out_gpu)
        out_gpu.deallocate()
        launchPad?.unregisterBuffer(address: inp)
        inp.deallocate()
        launchPad?.unregisterBuffer(address: wte)
        wte.deallocate()
        launchPad?.unregisterBuffer(address: wpe)
        wpe.deallocate()
    }

    // read kernel_num from command line
    var kernel_num = 2
    if argv.count > 0 {
        kernel_num = Int(argv[1]) ?? 2
    }
    print("Using kernel \(kernel_num)")

    // first check the correctness of the kernel
    encoder_forward(out_cpu, inp, wte, wpe, B, T, C)

    // time the kernel at different block sizes
    let block_sizes = [0, 64, 128, 256, 512, 1024]
    for block_size in block_sizes {
        print("Checking block size \(block_size)\(block_size == 0 ? " (computed)" : "")")
        try encoder_forward(kernel_num, out_gpu, inp, wte, wpe, B, T, C, block_size)
        try launchPad?.commit(wait: true)
        let tol: Float = 1e-5
        try validate_result(out_gpu, out_cpu, "out", B * T * C, tol)
    }

    print("All results match. Starting benchmarks.\n")

    let repeat_times = 1000
    var elapsed_time: Double = 0
    // CPU for comparison
    for _ in 0..<repeat_times {
        let start = Date()
        encoder_forward(out_cpu, inp, wte, wpe, B, T, C)
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
            try encoder_forward(kernel_num, out_gpu, inp, wte, wpe, B, T, C, block_size)
            let end = Date()
            elapsed_time += end.timeIntervalSince(start)
        }
        elapsed_time /= Double(repeat_times)
        elapsed_time *= 1e3 // ms

        // napkin math: estimate the memory bandwidth achieved
        // for each (B,T,C) output element, we do 3 reads and 1 write, 4 bytes each
        // and e.g. A100 40GB PCIe is advertised at 1,555GB/s
        let memory_ops = B * T * C * 4 * 4
        let memory_bandwidth = Double(memory_ops) / elapsed_time / 1e9

        print("block_size \(block_size) | time \(String(format: "%.4f", elapsed_time)) ms | bandwidth \(String(format: "%.2f", memory_bandwidth)) GB/s")
    }
}
