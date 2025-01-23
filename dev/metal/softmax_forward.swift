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

import Metal

// known kernel (Metal shader) versions
private let versions = 1...8

// overwrite async CPU version in `train_gpt2.swift´
// swiftlint:disable:next function_parameter_count
func softmax_forward(
    _ probs: UnsafeMutablePointer<Float>,
    _ logits: UnsafeMutablePointer<Float>,
    _ B: Int, _ T: Int, _ V: Int, _ Vp: Int) {
    // output: probs are (B,T,Vp) of the probabilities (sums to 1.0 in each b,t position)
    // input: logits is (B,T,Vp) of the unnormalized log probabilities
    // Vp is the padded vocab size (for efficiency), V is the "real" vocab size
    // example: Vp is 50304 and V is 50257
    // #pragma omp parallel for collapse(2)
    for b in 0..<B {
        for t in 0..<T {
            // probs <- softmax(logits)
            let logits_bt = logits + b * T * Vp + t * Vp
            let probs_bt = probs + b * T * Vp + t * Vp

            // maxval is only calculated and subtracted for numerical stability
            var maxval: Float = -10000 // TODO something better // swiftlint:disable:this todo
            for i in 0..<V where logits_bt[i] > maxval {
                maxval = logits_bt[i]
            }
            var sum: Float = 0
            for i in 0..<V {
                probs_bt[i] = expf(logits_bt[i] - maxval)
                sum += probs_bt[i]
            }
            // note we only loop to V, leaving the padded dimensions
            for i in 0..<V {
                probs_bt[i] /= sum
            }
            // for extra super safety we may wish to include this too,
            // forcing the probabilities here to be zero, but it shouldn't matter
            for i in V..<Vp {
                probs_bt[i] = 0
            }
        }
    }
}

// shader specific launch stub
func softmax_forward(
    _ out: UnsafeMutablePointer<Float>,
    _ inp: UnsafePointer<Float>,
    _ N: Int, _ C: Int,
    _ block_size: Int = 0) throws {
    let context = KernelContext(threadsPerGrid: N, threadsPerGroup: block_size)

    let params: [KernelParam] = [
        UnsafeMutableRawPointer(out),
        UnsafeMutableRawPointer(mutating: inp),
        Int32(N), Int32(C)]

    try launchPad?.dispatchKernel(
        name: "softmax_forward_kernel1",
        context: context,
        params: params)
}

// version dispatcher
private func softmax_forward(
    _ version: Int,
    _ out: UnsafeMutablePointer<Float>,
    _ inp: UnsafePointer<Float>,
    _ N: Int, _ C: Int,
    _ block_size: Int = 0) throws {
    guard
        versions.contains(version) == true
    else { throw LlmSwiftError.wrongApiUsage(api: "\(#function) version \(version) unknown") }

    switch version {
    case 1:
        try softmax_forward1(out, inp, N, C, block_size)
    case 2, 3, 4, 5, 6, 7, 8:
        fatalError("layer-pass function \(#function) version \(version) not implemented")
    default:
        break
    }
}

// standalone runner
// swiftlint:disable:next function_body_length
func softmax_forward(_ argc: Int, _ argv: [String]) throws {
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

    // read kernel_num from command line
    var kernel_num = 1
    if argv.count > 1 {
        kernel_num = Int(argv[1]) ?? 2
    }
    print("Using kernel \(kernel_num)")

    // first check the correctness of the kernel
    softmax_forward(out_cpu, inp, B, T, V, V)

    // time the kernel at different block sizes
    let block_sizes = [0, 32, 64, 128, 256, 512, 1024]
    for block_size in block_sizes {
        print("Checking block size \(block_size)\(block_size == 0 ? " (computed)" : "")")
        try softmax_forward(kernel_num, out_gpu, inp, B, T, V, block_size)
        try launchPad?.commit(wait: true)
        // #if !defined(ENABLE_BF16) && !defined(ENABLE_FP16)
        let tol: Float = 1e-4
        // #else
        // let tol: Float16 = 1e-2
        // #endif
        try validate_result(out_gpu, out_cpu, "out", B * T * V, tol)
    }

    print("All results match. Starting benchmarks.\n")

    let repeat_times = 1000
    var elapsed_time: Double = 0
    // CPU for comparison
    for _ in 0..<repeat_times {
        let start = Date()
        softmax_forward(out_cpu, inp, B, T, V, V)
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
            try softmax_forward(kernel_num, out_gpu, inp, B, T, V, block_size)
            let end = Date()
            elapsed_time += end.timeIntervalSince(start)
        }
        elapsed_time *= 1e3 // ms
        elapsed_time /= Double(repeat_times)

        // napkin math: estimate the memory bandwidth achieved
        // and e.g. A100 40GB PCIe is advertised at 1,555GB/s
        // using MTLStorageMode.shared (unified storage accessible by CPU and GPU)
        // let memory_ops = 2 * B * T * C * 4 // *4 for float
        // let memory_bandwidth = Double(memory_ops) / elapsed_time / 1e6

        print("block_size \(String(format: "%4d", block_size)) | time \(String(format: "%.4f", elapsed_time)) ms | bandwidth n/a (unified)")
    }
}
