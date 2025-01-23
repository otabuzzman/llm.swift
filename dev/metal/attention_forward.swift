// swiftlint:disable:next blanket_disable_command
// swiftlint:disable identifier_name

/// Kernel benchmark for the positional attention forward pass in GPT-2.
///
/// Kernels (Metal shaders) are in `DefaultLibrary.swift´
///
/// Compile:
/// xcodebuild -scheme layer_pass -configuration Release \
///   SWIFT_ACTIVE_COMPILATION_CONDITIONS="$SWIFT_ACTIVE_COMPILATION_CONDITIONS LAYER_PASS_STANDALONE"
///
/// version 1 is naive port from CPU code to kernel, parallelize over batch, time, heads only
/// ./attention_forward 1
///
/// version 2 is a naive implementation of flash attention, taken, adapted from
///           https://github.com/tspeterkim/flash-attention-minimal and with help from
///           https://github.com/leloykun/flash-hyperbolic-attention-minimal
///           sadly, this flash attention version seems about 3X slower than the naive version
/// ./attention_forward 2
///
/// version 3 is a cuBLAS + softmax version, similar to the PyTorch implementation
///           cuBLAS is used both to calculate the QK^T and the final weighted sum
///           the softmax is calculated using a custom, efficient kernel as well
///           this turns out to be ~20X faster than (1) nice
/// ./attention_forward 3
///
/// version 4 is a further optimized kernel that fuses the scale operation,
///           uses a directly autoregressive softmax, and uses the online softmax algorithm.
/// ./attention_forward 4
///
/// version 5 is a FP16 version of kernel 4
/// ./attention_forward 5
///
/// version 6 is kernel 5 skipping (un)permute (unrealistic but useful comparison point)
///
/// version 10 is using cuDNN Flash Attention using FP16 or BF16, see:
///            https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/operations/Attention.md
/// ./attention_forward 10

import Metal

// known kernel (Metal shader) versions
private let versions = 1...10
private let excludeVersions = 7...9

// TODO: check if applicable for macOS
// CUDA & cuDNN setup
private var first_run_validation = true

// overwrite async CPU version in `train_gpt2.swift´
// swiftlint:disable:next function_parameter_count function_body_length
func attention_forward(
    _ out: UnsafeMutablePointer<Float>,
    _ preatt: UnsafeMutablePointer<Float>,
    _ att: UnsafeMutablePointer<Float>,
    _ inp: UnsafePointer<Float>,
    _ B: Int, _ T: Int, _ C: Int, _ NH: Int) {
    // input is (B, T, 3C) holding the query, key, value (Q, K, V) vectors
    // preatt, att are (B, NH, T, T). NH = number of heads, T = sequence length
    // that holds the pre-attention and post-attention scores (used in backward)
    // output is (B, T, C)
    // attention is the only layer that mixes information across time
    // every other operation is applied at every (b,t) position independently
    // (and of course, no layer mixes information across batch)
    let C3 = C * 3
    let hs = C / NH // head size
    let scale = 1 / sqrtf(Float(hs))

    // #pragma omp parallel for collapse(3)
    for b in 0..<B {
        for t in 0..<T {
            for h in 0..<NH {
                let query_t = inp + b * T * C3 + t * C3 + h * hs
                let preatt_bth = preatt + b * NH * T * T + h * T * T + t * T
                let att_bth = att + b * NH * T * T + h * T * T + t * T

                // pass 1: calculate query dot key and maxval
                var maxval: Float = -10000 // TODO something better // swiftlint:disable:this todo
                for t2 in 0...t {
                    let key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C // +C because it's key

                    // (query_t) dot (key_t2)
                    var val: Float = 0
                    for i in 0..<hs {
                        val += query_t[i] * key_t2[i]
                    }
                    val *= scale
                    if val > maxval {
                        maxval = val
                    }

                    preatt_bth[t2] = val
                }
                // pad with -INFINITY outside of autoregressive region for debugging comparisons
                for t2 in t+1..<T {
                    preatt_bth[t2] = Float.infinity
                }

                // pass 2: calculate the exp and keep track of sum
                // maxval is being calculated and subtracted only for numerical stability
                var expsum: Float = 0
                for t2 in 0...t {
                    let expv = expf(preatt_bth[t2] - maxval)
                    expsum += expv
                    att_bth[t2] = expv
                }
                let expsum_inv = expsum == 0 ? 0 : 1 / expsum

                // pass 3: normalize to get the softmax
                for t2 in 0..<T {
                    if t2 <= t {
                        att_bth[t2] *= expsum_inv
                    } else {
                        // causal attention mask. not strictly necessary to set to zero here
                        // only doing this explicitly for debugging and checking to PyTorch
                        att_bth[t2] = 0
                    }
                }

                // pass 4: accumulate weighted values into the output of attention
                let out_bth = out + b * T * C + t * C + h * hs
                for i in 0..<hs { out_bth[i] = 0 }
                for t2 in 0...t {
                    let value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C * 2 // +C*2 because it's value
                    let att_btht2 = att_bth[t2]
                    for i in 0..<hs {
                        out_bth[i] += att_btht2 * value_t2[i]
                    }
                }
            }
        }
    }
}

// shader specific launch stub
// swiftlint:disable:next function_parameter_count
func attention_forward1(
    _ out: UnsafeMutablePointer<Float>,
    _ preatt: UnsafeMutablePointer<Float>,
    _ att: UnsafeMutablePointer<Float>,
    _ inp: UnsafePointer<Float>,
    _ B: Int, _ T: Int, _ C: Int, _ NH: Int,
    _ block_size: Int = 0) throws {
    var context = KernelContext(threadsPerGrid: B * NH * T * T, threadsPerGroup: block_size)

    let param_preatt = UnsafeMutableRawPointer(preatt)
    let param_inp = UnsafeMutableRawPointer(mutating: inp)
    var params: [KernelParam] = [
        param_preatt, param_inp,
        Int32(B), Int32(T), Int32(C), Int32(NH)]

    try launchPad?.dispatchKernel(
        name: "attention_query_key_kernel1",
        context: context,
        params: params)

    context = KernelContext(threadsPerGrid: B * T * NH, threadsPerGroup: block_size)

    let param_att = UnsafeMutableRawPointer(att)
    params = [
        param_att, param_preatt,
        Int32(B), Int32(T), Int32(NH)]

    try launchPad?.dispatchKernel(
        name: "attention_softmax_kernel1",
        context: context,
        params: params)

    params = [
        UnsafeMutableRawPointer(out), param_att, param_inp,
        Int32(B), Int32(T), Int32(C), Int32(NH)]

    try launchPad?.dispatchKernel(
        name: "attention_value_kernel1",
        context: context,
        params: params)
}

// version dispatcher
// swiftlint:disable:next function_parameter_count
private func attention_forward(
    _ version: Int,
    _ out: UnsafeMutablePointer<Float>,
    _ preatt: UnsafeMutablePointer<Float>,
    _ att: UnsafeMutablePointer<Float>,
    _ inp: UnsafePointer<Float>,
    _ B: Int, _ T: Int, _ C: Int, _ NH: Int,
    _ block_size: Int = 0) throws {
    guard
        versions.contains(version) == true,
        excludeVersions.contains(version) == false
    else { throw LlmSwiftError.wrongApiUsage(api: "\(#function) version \(version) unknown") }

    switch version {
    case 1:
        try attention_forward1(out, preatt, att, inp, B, T, C, NH, block_size)
    case 2, 3, 4, 5, 6, 10:
        fatalError("layer-pass function \(#function) version \(version) not implemented")
    default:
        break
    }
}

// standalone runner
// swiftlint:disable:next function_body_length
func attention_forward(_ argc: Int, _ argv: [String]) throws {
    let B = 8
    let T = 1024
    let C = 768
    let NH = 12

    try launchPad?.registerKernel(name: "attention_query_key_kernel1")
    try launchPad?.registerKernel(name: "attention_softmax_kernel1")
    try launchPad?.registerKernel(name: "attention_value_kernel1")

    // create memory of random numbers
    let out_cpu = UnsafeMutablePointer<Float>.allocate(capacity: B * T * C)
    let preatt_cpu = UnsafeMutablePointer<Float>.allocate(capacity: B * NH * T * T)
    let att_cpu = UnsafeMutablePointer<Float>.allocate(capacity: B * NH * T * T)

    let out_gpu = UnsafeMutablePointer<Float>.allocate(capacity: B * T * C)
    let out_gpu_length = B * T * C * MemoryLayout<Float>.size
    try launchPad?.registerBuffer(address: out_gpu, length: out_gpu_length)

    let preatt_gpu = UnsafeMutablePointer<Float>.allocate(capacity: B * NH * T * T)
    let preatt_gpu_length = B * NH * T * T * MemoryLayout<Float>.size
    try launchPad?.registerBuffer(address: preatt_gpu, length: preatt_gpu_length)

    let att_gpu = UnsafeMutablePointer<Float>.allocate(capacity: B * NH * T * T)
    let att_gpu_length = B * NH * T * T * MemoryLayout<Float>.size
    try launchPad?.registerBuffer(address: att_gpu, length: att_gpu_length)

    let inp = UnsafeMutablePointer<Float>.allocate(capacity: B * T * 3 * C)
    for i in 0..<B * T * 3 * C { inp[i] = Float.random(in: -1.0...1.0) }
    let inp_length = B * T * 3 * C * MemoryLayout<Float>.size
    try launchPad?.registerBuffer(address: inp, length: inp_length)

    defer {
        out_cpu.deallocate()
        preatt_cpu.deallocate()
        att_cpu.deallocate()
        launchPad?.unregisterBuffer(address: out_gpu)
        out_gpu.deallocate()
        launchPad?.unregisterBuffer(address: preatt_gpu)
        preatt_gpu.deallocate()
        launchPad?.unregisterBuffer(address: att_gpu)
        att_gpu.deallocate()
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
    attention_forward(out_cpu, preatt_cpu, att_cpu, inp, B, T, C, NH)

    // time the kernel at different block sizes
    let block_sizes = [0, 32, 64, 128, 256, 512]
    for block_size in block_sizes {
        print("Checking block size \(block_size)\(block_size == 0 ? " (computed)" : "")")
        try attention_forward(kernel_num, out_gpu, preatt_gpu, att_gpu, inp, B, T, C, NH, block_size)
        try launchPad?.commit(wait: true)
        let accuracy_threshold: Float = kernel_num <= 4 ? 1e-3 : 1e-2
        try validate_result(out_gpu, out_cpu, "out", B * T * C, accuracy_threshold)
        if kernel_num != 2 && kernel_num < 5 {
            try validate_result(att_gpu, att_cpu, "att", B * NH * T * T, accuracy_threshold)
        }
        if kernel_num != 2 && kernel_num < 4 {
            try validate_result(preatt_gpu, preatt_cpu, "preatt", B * NH * T * T, accuracy_threshold)
        }
    }

    print("All results match. Starting benchmarks.\n")
    first_run_validation = false

    let repeat_times = 100
    var elapsed_time: Double = 0
    // CPU for comparison
    for _ in 0..<repeat_times {
        let start = Date()
        attention_forward(out_cpu, preatt_cpu, att_cpu, inp, B, T, C, NH)
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
            try attention_forward(kernel_num, out_gpu, preatt_gpu, att_gpu, inp, B, T, C, NH, block_size)
            let end = Date()
            elapsed_time += end.timeIntervalSince(start)
        }
        elapsed_time *= 1e3 // ms
        elapsed_time /= Double(repeat_times)

        print("block_size \(String(format: "%4d", block_size)) | time \(String(format: "%.4f", elapsed_time)) ms")
    }
}
