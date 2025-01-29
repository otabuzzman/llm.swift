// swiftlint:disable:next blanket_disable_command
// swiftlint:disable identifier_name

/// Kernel benchmark for the positional crossentropy forward pass in GPT-2.
///
/// Kernels (Metal shaders) are in `DefaultLibrary.swift´
///
/// Compile:
/// xcodebuild -scheme layer_pass -configuration Release \
///   SWIFT_ACTIVE_COMPILATION_CONDITIONS="$SWIFT_ACTIVE_COMPILATION_CONDITIONS LAYER_PASS_STANDALONE"
///
/// version 1 is a straight-forward port from CPU code to kernel, parallel over B,T
/// ./crossentropy_forward 1

import Metal

// known kernel (Metal shader) versions
private let versions = 1...1

// shader specific launch stub
// swiftlint:disable:next function_parameter_count
func crossentropy_forward1(
    _ losses: UnsafeMutablePointer<Float>,
    _ probs: UnsafePointer<Float>,
    _ targets: UnsafePointer<Int32>,
    _ B: Int, _ T: Int, _ V: Int,
    _ block_size: Int = 0) throws {
    let context = KernelContext(threadsPerGrid: B * T, threadsPerGroup: block_size)

    let params: [KernelParam] = [
        UnsafeMutableRawPointer(losses),
        UnsafeMutableRawPointer(mutating: probs),
        UnsafeMutableRawPointer(mutating: targets),
        Int32(B), Int32(T), Int32(V)]

    try launchPad?.dispatchKernel(
        name: "crossentropy_forward_kernel1",
        context: context,
        params: params)
}

// version dispatcher
// swiftlint:disable:next function_parameter_count
private func crossentropy_forward(
    _ version: Int,
    _ losses: UnsafeMutablePointer<Float>,
    _ probs: UnsafePointer<Float>,
    _ targets: UnsafePointer<Int32>,
    _ B: Int, _ T: Int, _ V: Int,
    _ block_size: Int = 0) throws {
    guard
        versions.contains(version) == true
    else { throw LlmSwiftError.wrongApiUsage(api: "\(#function) version \(version) unknown") }

    switch version {
    case 1:
        try crossentropy_forward1(losses, probs, targets, B, T, V, block_size)
    default:
        break
    }
}

// standalone runner
// swiftlint:disable:next function_body_length
func crossentropy_forward(_ argc: Int, _ argv: [String]) throws {
    let B = 8
    let T = 1024
    let V = 50257

    try launchPad?.registerKernel(name: "crossentropy_forward_kernel1")

    // create memory of random numbers
    let out_cpu = UnsafeMutablePointer<Float>.allocate(capacity: B * T)

    let out_gpu = UnsafeMutablePointer<Float>.allocate(capacity: B * T)
    let out_gpu_length = B * T * MemoryLayout<Float>.size
    try launchPad?.registerBuffer(address: out_gpu, length: out_gpu_length)

    let losses = UnsafeMutablePointer<Float>.allocate(capacity: B * T * V)
    for i in 0..<B * T * V { losses[i] = Float.random(in: -1.0...1.0) }
    let losses_length = B * T * V * MemoryLayout<Float>.size
    try launchPad?.registerBuffer(address: losses, length: losses_length)

    let targets = UnsafeMutablePointer<Int32>.allocate(capacity: B * T)
    for i in 0..<B * T { targets[i] = Int32(Int.random(in: 0..<V)) }
    let targets_length = B * T * MemoryLayout<Int32>.size
    try launchPad?.registerBuffer(address: targets, length: targets_length)

    defer {
        out_cpu.deallocate()
        launchPad?.unregisterBuffer(address: out_gpu)
        out_gpu.deallocate()
        launchPad?.unregisterBuffer(address: losses)
        losses.deallocate()
        launchPad?.unregisterBuffer(address: targets)
        targets.deallocate()
    }

    // defaults
    var kernel_num = 1
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
        crossentropy_forward(out_cpu, losses, targets, B, T, V)
        let end = Date()
        print("CPU version took \(end.timeIntervalSince(start) * 1e3) ms\n")

        // time the kernel at different block sizes
        for block_size in block_sizes {
            print("Checking block size \(block_size)\(block_size == 0 ? " (computed)" : "")")
            try crossentropy_forward(kernel_num, out_gpu, losses, targets, B, T, V, block_size)
            try launchPad?.commit(wait: true)
            let tol: Float = 1e-5
            try validate_result(out_gpu, out_cpu, "out", B * T, tol)
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

            try crossentropy_forward(kernel_num, out_gpu, losses, targets, B, T, V, block_size)
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
