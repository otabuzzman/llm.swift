// swiftlint:disable:next blanket_disable_command
// swiftlint:disable identifier_name

/// Kernel benchmark for the positional softmax forward pass in GPT-2.
///
/// Kernels (Metal shaders) are in `DefaultLibrary.swift´
///
/// Compile:
/// xcodebuild -scheme layer_pass -configuration Release \
///   SWIFT_ACTIVE_COMPILATION_CONDITIONS="$SWIFT_ACTIVE_COMPILATION_CONDITIONS LAYER_PASS_STANDALONE"
///
/// version 1 is naive port from CPU code to kernel: parallelizes over B,T, loops over C
/// ./softmax_forward 1
///
/// version 2 is a fused kernel that parallelizes over all of B,T,C
/// ./softmax_forward 2
///
/// version 3 uses intra-warp reductions for maxval and sumval, must use block_size=32
/// ./softmax_forward 3
///
/// version 4 uses both intra-warp reductions and shared memory for inter-warp reductions
/// so it can tolerate any block_size % 32 == 0. this is hopefully the most efficient version
/// ./softmax_forward 4
///
/// version 5 is naive port from CPU code (softmax_online) to kernel: parallelizes over B,T, loops over C
/// ./softmax_forward 5
///
/// version 6 is softmax_online that parallelizes over all of B,T,C
/// ./softmax_forward 6
///
/// version 7 is softmax optimized for very large C.
/// ./softmax_forward 7
///
/// version 8 calculates sumval and maxval in one loop (http://arxiv.org/abs/1805.02867)
/// ./softmax_forward 8

import Metal

// known kernel (Metal shader) versions
private let versions = 1...8

// shader specific launch stub
// swiftlint:disable:next function_parameter_count
func softmax_forward1(
    _ out: UnsafeMutablePointer<Float>,
    _ inp: UnsafePointer<Float>,
    _ B: Int, _ T: Int, _ V: Int, _ Vp: Int,
    _ block_size: Int = 0) throws {
    let context = KernelContext(threadsPerGrid: B * T, threadsPerGroup: block_size)

    let params: [KernelParam] = [
        UnsafeMutableRawPointer(out),
        UnsafeMutableRawPointer(mutating: inp),
        Int32(B * T), Int32(V), Int32(Vp)]

    try launchPad?.dispatchKernel(
        name: "softmax_forward_kernel1",
        context: context,
        params: params)
}

// shader specific launch stub
// swiftlint:disable:next function_parameter_count
func softmax_forward7(
    _ out: UnsafeMutablePointer<Float>,
    _ inp: UnsafePointer<Float>,
    _ B: Int, _ T: Int, _ V: Int, _ Vp: Int,
    _ block_size: Int = 0) throws {
    let context = KernelContext(
        threadsPerGrid: B * T,
        threadsPerGroup: block_size,
        threadgroupMemory: ThreadgroupMemoryDescriptor(
            scope: .threadgroup,
            units: 1, type: Float.self))

    let params: [KernelParam] = [
        UnsafeMutableRawPointer(out),
        UnsafeMutableRawPointer(mutating: inp),
        Int32(B * T), Int32(V), Int32(Vp)]

    try launchPad?.dispatchKernel(
        name: "softmax_forward_kernel7",
        context: context,
        params: params)
}

// shader specific launch stub
// swiftlint:disable:next function_parameter_count
func softmax_forward8(
    _ out: UnsafeMutablePointer<Float>,
    _ inp: UnsafePointer<Float>,
    _ B: Int, _ T: Int, _ V: Int, _ Vp: Int,
    _ block_size: Int = 0) throws {
    let context = KernelContext(threadsPerGrid: B * T, threadsPerGroup: block_size)

    let params: [KernelParam] = [
        UnsafeMutableRawPointer(out),
        UnsafeMutableRawPointer(mutating: inp),
        Int32(B * T), Int32(V), Int32(Vp)]

    try launchPad?.dispatchKernel(
        name: "softmax_forward_kernel8",
        context: context,
        params: params)
}

// version dispatcher
// swiftlint:disable:next function_parameter_count
private func softmax_forward(
    _ version: Int,
    _ out: UnsafeMutablePointer<Float>,
    _ inp: UnsafePointer<Float>,
    _ B: Int, _ T: Int, _ V: Int, _ Vp: Int,
    _ block_size: Int = 0) throws {
    guard
        versions.contains(version) == true
    else { throw LlmSwiftError.wrongApiUsage(api: "\(#function) version \(version) unknown") }

    switch version {
    case 1:
        try softmax_forward1(out, inp, B, T, V, Vp, block_size)
    case 7:
        try softmax_forward7(out, inp, B, T, V, Vp, block_size)
    case 8:
        try softmax_forward8(out, inp, B, T, V, Vp, block_size)
    case 2, 3, 4, 5, 6:
        fatalError("layer-pass function \(#function) version \(version) not implemented")
    default:
        break
    }
}

// standalone runner
// swiftlint:disable:next function_body_length
func softmax_forward(_ argc: Int, _ argv: [String]) async throws {
    let B = 8
    let T = 1024
    let V = 50257

    try launchPad?.registerKernel(name: "softmax_forward_kernel1")

    // create memory of random numbers
    let out_cpu = UnsafeMutablePointer<Float>.allocate(capacity: B * T * V)

    let out_gpu = UnsafeMutablePointer<Float>.allocate(capacity: B * T * V)
    let out_gpu_length = B * T * V * MemoryLayout<Float>.size
    try launchPad?.registerBuffer(address: out_gpu, length: out_gpu_length)

    let inp = UnsafeMutablePointer<Float>.allocate(capacity: B * T * V)
    for i in 0..<B * T * V { inp[i] = Float.random(in: -1.0...1.0) }
    let inp_length = B * T * V * MemoryLayout<Float>.size
    try launchPad?.registerBuffer(address: inp, length: inp_length)

    defer {
        out_cpu.deallocate()
        launchPad?.unregisterBuffer(address: out_gpu)
        out_gpu.deallocate()
        launchPad?.unregisterBuffer(address: inp)
        inp.deallocate()
    }

    // defaults
    var kernel_num = 1
    var repeat_times = 100
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
        await softmax_forward(out_cpu, inp, B, T, V, V)
        let end = Date()
        print("CPU version took \(String(format: "%.2f", end.timeIntervalSince(start) * 1e3)) ms\n")

        // time the kernel at different block sizes
        for block_size in block_sizes {
            print("Checking block size \(block_size)\(block_size == 0 ? " (computed)" : "")")
            try softmax_forward(kernel_num, out_gpu, inp, B, T, V, V, block_size)
            try launchPad?.commit(wait: true)
            let tol: Float = 1e-4
            try validate_result(out_gpu, out_cpu, "out", B * T * V, tol)
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

            try softmax_forward(kernel_num, out_gpu, inp, B, T, V, V, block_size)
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
