// swiftlint:disable:next blanket_disable_command
// swiftlint:disable identifier_name

/// Kernel benchmark for the positional gelu forward pass in GPT-2.
///
/// Kernels (Metal shaders) are in `DefaultLibrary.swift´
///
/// Compile:
/// xcodebuild -scheme layer_pass -configuration Release \
///   SWIFT_ACTIVE_COMPILATION_CONDITIONS="$SWIFT_ACTIVE_COMPILATION_CONDITIONS LAYER_PASS_STANDALONE"
///
/// version 1 is naive CPU port
/// ./gelu_forward 1
/// 
/// version 2 is bfloat16 with the Packed128 data structure
/// ./gelu_forward 2

import Metal

// known kernel (Metal shader) versions
private let versions = 1...2

// shader specific launch stub
func gelu_forward1(
    _ out: UnsafeMutablePointer<Float>,
    _ inp: UnsafePointer<Float>,
    _ N: Int,
    _ block_size: Int = 0) throws {
    let context = KernelContext(threadsPerGrid: N, threadsPerGroup: block_size)

    let params: [KernelParam] = [
        UnsafeMutableRawPointer(out),
        UnsafeMutableRawPointer(mutating: inp),
        Int32(N)]

    try launchPad?.dispatchKernel(
        name: "gelu_forward_kernel1",
        context: context,
        params: params)
}

// shader specific launch stub
func gelu_forward2(
    _ out: UnsafeMutablePointer<Float>,
    _ inp: UnsafePointer<Float>,
    _ N: Int,
    _ block_size: Int = 0) throws {
    let context = KernelContext(threadsPerGrid: N / 4, threadsPerGroup: block_size)

    let params: [KernelParam] = [
        UnsafeMutableRawPointer(out),
        UnsafeMutableRawPointer(mutating: inp),
        Int32(N)]

    try launchPad?.dispatchKernel(
        name: "gelu_forward_kernel2",
        context: context,
        params: params)
}

// version dispatcher
private func gelu_forward(
    _ version: Int,
    _ out: UnsafeMutablePointer<Float>,
    _ inp: UnsafePointer<Float>,
    _ N: Int,
    _ block_size: Int = 0) throws {
    guard
        versions.contains(version) == true
    else { throw LlmDotSwiftError.wrongApiUsage(api: "\(#function) version \(version) unknown") }

    switch version {
    case 1:
        try gelu_forward1(out, inp, N, block_size)
    case 2:
        try gelu_forward2(out, inp, N, block_size)
    default:
        break
    }
}

// standalone runner
// swiftlint:disable:next function_body_length
func gelu_forward(_ argc: Int, _ argv: [String]) throws {
    let B = 8
    let T = 1024
    let C = 768

    try launchPad?.registerKernel(name: "gelu_forward_kernel1")
    try launchPad?.registerKernel(name: "gelu_forward_kernel2")

    // create memory of random numbers
    let out_cpu = UnsafeMutablePointer<Float>.allocate(capacity: B * T * C)

    let out_gpu = UnsafeMutablePointer<Float>.allocate(capacity: B * T * C)
    let out_gpu_length = B * T * C * MemoryLayout<Float>.size
    try launchPad?.registerBuffer(address: out_gpu, length: out_gpu_length)

    let inp = UnsafeMutablePointer<Float>.allocate(capacity: B * T * C)
    for i in 0..<B * T * C { inp[i] = Float.random(in: -1.0...1.0) }
    let inp_length = B * T * C * MemoryLayout<Float>.size
    try launchPad?.registerBuffer(address: inp, length: inp_length)

    defer {
        out_cpu.deallocate()
        launchPad?.unregisterBuffer(address: out_gpu)
        out_gpu.deallocate()
        launchPad?.unregisterBuffer(address: inp)
        inp.deallocate()
    }

    // defaults
    var kernel_num = 2
    var repeat_times = 1000
    var block_sizes = [0, 32, 64, 128, 256, 512, 1024]

    // command line arguments
    var argNoCheck = false
    var argBlockSize = false
    var argRepeatNum = false
    for arg in argv[1..<argv.count] {
        switch arg {
        case "nocheck":
            argNoCheck = true
        case "blocksize":
            argBlockSize = true
        case "repeats":
            argRepeatNum = true
        default:
            let argNum = Int(arg) ?? 0
            if argBlockSize { block_sizes = [argNum] ; argBlockSize = false ; continue }
            if argRepeatNum { repeat_times = argNum ; argRepeatNum = false ; continue }

            kernel_num = argNum
        }
    }
    print("Using kernel \(kernel_num)")

    // first check the correctness of the kernel
    if !argNoCheck {
        let start = Date()
        gelu_forward(out_cpu, inp, B * T * C)
        let end = Date()
        print("CPU version took \(String(format: "%.2f", end.timeIntervalSince(start) * 1e3)) ms\n")

        // time the kernel at different block sizes
        for block_size in block_sizes {
            print("Checking block size \(block_size)\(block_size == 0 ? " (computed)" : "")")
            try gelu_forward(kernel_num, out_gpu, inp, B * T * C, block_size)
            try launchPad?.commit(wait: true)
            // #if !defined(ENABLE_BF16) && !defined(ENABLE_FP16)
            let tol: Float = 1e-5
            // #else
            // let tol: Float16 = 1e-2
            // #endif
            try validate_result(out_gpu, out_cpu, "out", B * T * C, tol)
        }
        print("All results match. ")
    }

    print("Starting benchmarks.\n")
    Task {
        try? await Task.sleep(for: .seconds(15))
        print("still busy. consider less repeats (\(repeat_times))?")
    }

    var elapsed_time: Double = 0
    for block_size in block_sizes {
        // omitted generic `benchmark_kernel´ in dev/cuda/common.h
        let start = Date()
        for _ in 0..<repeat_times {
            // clear L2
            // TODO: if necessary and applicable

            try gelu_forward(kernel_num, out_gpu, inp, B * T * C, block_size)
            try launchPad?.commit(wait: true)
        }
        let end = Date()
        elapsed_time = end.timeIntervalSince(start)
        elapsed_time /= Double(repeat_times)
        elapsed_time *= 1e3 // ms

        // napkin math: estimate the memory bandwidth achieved
        // and e.g. A100 40GB PCIe is advertised at 1,555GB/s
        // using MTLStorageMode.shared (unified storage accessible by CPU and GPU)
        // let memory_ops = 2 * B * T * C * 4 // *4 for float
        // let memory_bandwidth = Double(memory_ops) / elapsed_time / 1e6

        print("block_size \(String(format: "%4d", block_size)) | time \(String(format: "%.4f", elapsed_time)) ms | bandwidth n/a (unified)")
    }
}
