// swiftlint:disable:next blanket_disable_command
// swiftlint:disable identifier_name

import Foundation
import System

// #define TESTING

// poor man's tensor checker
func check_tensor(
    _ a: UnsafePointer<Float>, _ b: UnsafePointer<Float>, _ n: Int,
    _ label: String, _ stdlog: ((String) -> Void)?) -> Bool {
    let print_upto = 5
    var ok = true
    var maxdiff: Float = 0
    let tol: Float = 2e-2
    stdlog?("\(label)\n")
    for i in 0..<n {
        // look at the diffence at position i of these two tensors
        let diff = fabsf(a[i] - b[i])

        // keep track of the overall error
        ok = ok && (diff <= tol)
        if diff > maxdiff { maxdiff = diff }

        // for the first few elements of each tensor, pretty print
        // the actual numbers, so we can do a visual, qualitative proof/assessment
        if i < print_upto {
            if diff <= tol {
                stdlog?("OK \(a[i]) \(b[i])\n")
            } else {
                stdlog?("NOT OK \(a[i]) \(b[i])\n")
            }
        }
    }
    // print the final result for this tensor
    if ok {
        stdlog?("TENSOR OK, maxdiff = \(maxdiff)\n")
    } else {
        stdlog?("TENSOR NOT OK, maxdiff = \(maxdiff)\n")
    }
    return ok
}

// swiftlint:disable:next function_body_length
func test_gpt2(_ folder: URL?, _ stdlog: ((String) -> Void)? = nil) async throws {
    let cwd = FileManager.default.currentDirectoryPath
    defer { FileManager.default.changeCurrentDirectoryPath(cwd) }
    if let folder = folder {
        FileManager.default.changeCurrentDirectoryPath(folder.path)
    }

    // build the GPT-2 model from a checkpoint
    var model = GPT2()
    let model_filename = "gpt2_124M.bin"
    let model_handle = try FileHandle(forReadingFrom: URL(string: model_filename)!)
    try gpt2_build_from_checkpoint(&model, model_handle, stdlog)

    let C = model.config.channels
    let V = model.config.vocab_size
    let Vp = model.config.padded_vocab_size
    let maxT = model.config.max_seq_len
    let L = model.config.num_layers

    // load additional information that we will use for debugging and error checking
    let state_filename = "gpt2_124M_debug_state.bin"
    let state_file = try FileHandle(forReadingFrom: URL(string: state_filename)!)
    guard
        let header_data = try state_file.read(upToCount: 256 * MemoryLayout<Int32>.size)
    else { throw LlmSwiftError.apiReturnedNil(api: "read (in \(#function)") }
    let state_header = header_data.withUnsafeBytes { (state_header: UnsafeRawBufferPointer) -> [Int] in
        state_header.bindMemory(to: Int32.self).map { Int($0) }
    }
    assert(state_header[0] == 20240327, "Bad magic in state file (try `python train_gpt2.py`)")
    assert(state_header[1] == 2, "Bad version in state file (try `python train_gpt2.py`)")
    let B = state_header[2] // batch size, e.g. 4
    let T = state_header[3] // time / sequence length (e.g. 64, up to maxT)
    stdlog?("[State]\n")
    stdlog?("batch_size: \(B)\n")
    stdlog?("seq_len: \(T)\n")

    var expected_grads = ParameterTensors()
    let expected_grads_memory = malloc_and_point_parameters(&expected_grads, model.param_sizes)
    let expected_grads_memory_buffer = UnsafeMutableRawBufferPointer(expected_grads_memory)

    defer {
        // free on leaving
        expected_grads_memory.deallocate()
        gpt2_free(&model)
    }

    // read reference information from Python
    guard
        let x_data = try state_file.read(upToCount: B * T * MemoryLayout<Int32>.size),
        let y_data = try state_file.read(upToCount: B * T * MemoryLayout<Int32>.size),
        let expected_logits_data = try state_file.read(upToCount: B * T * V * MemoryLayout<Float>.size),
        let expected_loss_data = try state_file.read(upToCount: MemoryLayout<Float>.size)
    else { throw LlmSwiftError.apiReturnedNil(api: "read (in \(#function))") }
    _ = try FileDescriptor(rawValue: state_file.fileDescriptor).read(into: expected_grads_memory_buffer)
    // inputs and expected outputs, only used for error checking
    let x = x_data.withUnsafeBytes { $0.bindMemory(to: Int32.self) }
    let y = y_data.withUnsafeBytes { $0.bindMemory(to: Int32.self) }
    let expected_logits = expected_logits_data.withUnsafeBytes { $0.bindMemory(to: Float.self) }
    let expected_loss = expected_loss_data.withUnsafeBytes { $0.load(as: Float.self) }
    try? state_file.close()

    // register inputs/targets for Metal
    let buffer_length = B * T * MemoryLayout<Int32>.size
    let inputs_memory = UnsafeMutableRawPointer(mutating: x.baseAddress!)
    let targets_memory = UnsafeMutableRawPointer(mutating: y.baseAddress!)
    try launchPad?.registerBuffer(address: inputs_memory, length: buffer_length)
    try launchPad?.registerBuffer(address: targets_memory, length: buffer_length)
    defer {
        launchPad?.unregisterBuffer(address: inputs_memory)
        launchPad?.unregisterBuffer(address: targets_memory)
    }

    // overall OK signal for the test
    var allok = true

    // let's do 10 training iterations, following the pytorch code
    let expected_losses: [Float] = [
        5.270007133483887,
        4.059706687927246,
        3.3751230239868164,
        2.8007826805114746,
        2.315382242202759,
        1.8490285873413086,
        1.3946564197540283,
        0.9991465210914612,
        0.6240804195404053,
        0.37651097774505615
    ]

    for step in 0..<10 {
        let start = Date.timeIntervalSinceReferenceDate

        try await gpt2_forward(&model, x.baseAddress!, y.baseAddress!, B, T, stdlog)
        gpt2_zero_grad(&model)
        try await gpt2_backward(&model)

        let end = Date.timeIntervalSinceReferenceDate

        if step == 0 {
            // error checking at step 0 for reference activations/gradients
            // at this point, target should be equal to expected_logits, let's compare
            var logits_ok = true
            let calculated_logits = model.acts.logits
            var max_diff: Float = 0
            for bt in 0..<B * T {
                for v in 0..<V { // note we only loop to V (ignoring padding)
                    let i = bt * Vp + v // linearized index, using Vp
                    if i < 10 {
                        stdlog?("\(expected_logits[i]) \(calculated_logits![i])\n")
                    }
                    let diff = fabsf(expected_logits[bt * V + v] - calculated_logits![i])
                    max_diff = fmaxf(max_diff, diff)
                    if diff >= 1e-2 {
                        stdlog?("MISMATCH AT INDEX \(bt),\(v): \(expected_logits[bt * V + v]) \(calculated_logits![i])\n")
                        logits_ok = false
                        break // break out of both loops
                    }
                }
                if !logits_ok { break }
            }
            stdlog?("\(logits_ok ? "" : "NOT ")OK (LOGITS), max_diff = \(max_diff)\n")
            allok = allok && logits_ok

            // compare the achieved loss
            if fabsf(model.mean_loss - expected_loss) >= 1e-2 {
                stdlog?("LOSS MISMATCH: \(model.mean_loss) \(expected_loss)\n")
                allok = false
            } else {
                stdlog?("LOSS OK: \(model.mean_loss) \(expected_loss)\n")
            }

            // finally check all the gradients
            var gradoks = [Bool](repeating: false, count: 16)
            let grads = model.grads
            gradoks[0] = check_tensor(grads.wte, expected_grads.wte, V * C, "dwte", stdlog)
            gradoks[1] = check_tensor(grads.wpe, expected_grads.wpe, maxT * C, "dwpe", stdlog)
            gradoks[2] = check_tensor(grads.ln1w, expected_grads.ln1w, L * C, "dln1w", stdlog)
            gradoks[3] = check_tensor(grads.ln1b, expected_grads.ln1b, L * C, "dln1b", stdlog)
            gradoks[4] = check_tensor(grads.qkvw, expected_grads.qkvw, L * 3 * C * C, "dqkvw", stdlog)
            gradoks[5] = check_tensor(grads.qkvb, expected_grads.qkvb, L * 3 * C, "dqkvb", stdlog)
            gradoks[6] = check_tensor(grads.attprojw, expected_grads.attprojw, L * C * C, "dattprojw", stdlog)
            gradoks[7] = check_tensor(grads.attprojb, expected_grads.attprojb, L * C, "dattprojb", stdlog)
            gradoks[8] = check_tensor(grads.ln2w, expected_grads.ln2w, L * C, "dln2w", stdlog)
            gradoks[9] = check_tensor(grads.ln2b, expected_grads.ln2b, L * C, "dln2b", stdlog)
            gradoks[10] = check_tensor(grads.fcw, expected_grads.fcw, L * 4 * C * C, "dfcw", stdlog)
            gradoks[11] = check_tensor(grads.fcb, expected_grads.fcb, L * 4 * C, "dfcb", stdlog)
            gradoks[12] = check_tensor(grads.fcprojw, expected_grads.fcprojw, L * C * 4 * C, "dfcprojw", stdlog)
            gradoks[13] = check_tensor(grads.fcprojb, expected_grads.fcprojb, L * C, "dfcprojb", stdlog)
            gradoks[14] = check_tensor(grads.lnfw, expected_grads.lnfw, C, "dlnfw", stdlog)
            gradoks[15] = check_tensor(grads.lnfb, expected_grads.lnfb, C, "dlnfb", stdlog)
            for i in 0..<16 {
                allok = allok && gradoks[i]
            }
        }

        gpt2_update(&model, 1e-4, 0.9, 0.999, 1e-8, 0.01, step + 1)

        // compare the losses
        let expected_loss = expected_losses[step]
        let actual_loss = model.mean_loss
        let step_loss_ok = fabsf(expected_loss - actual_loss) < 1e-2
        allok = allok && step_loss_ok

        // print the timing information at the end
        stdlog?("step \(step): loss \(model.mean_loss) (took \(String(format: "%1.2f", (end - start) * 1000)) ms) OK = \(step_loss_ok)\n")
    }

    // final judgement
    stdlog?("overall okay: \(allok)\n")
}
