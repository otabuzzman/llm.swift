// swiftlint:disable:next blanket_disable_command
// swiftlint:disable identifier_name

import Foundation
import System

// #define TESTING

// poor man's tensor checker
func check_tensor(
    _ a: UnsafePointer<Float>, _ b: UnsafePointer<Float>, _ n: Int,
    _ label: String, _ info: ((String) -> Void)?) -> Bool {
    let print_upto = 5
    var ok = true
    var maxdiff: Float = 0
    let tol: Float = 2e-2
    info?("\(label)\n")
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
                info?("OK \(a[i]) \(b[i])\n")
            } else {
                info?("NOT OK \(a[i]) \(b[i])\n")
            }
        }
    }
    // print the final result for this tensor
    if ok {
        info?("TENSOR OK, maxdiff = \(maxdiff)\n")
    } else {
        info?("TENSOR NOT OK, maxdiff = \(maxdiff)\n")
    }
    return ok
}

// swiftlint:disable:next function_body_length
func test_gpt2(_ folder: URL?, _ info: ((String) -> Void)? = nil) async {
    let cwd = FileManager.default.currentDirectoryPath
    defer { FileManager.default.changeCurrentDirectoryPath(cwd) }
    if let folder = folder {
        FileManager.default.changeCurrentDirectoryPath(folder.path)
    }

    // build the GPT-2 model from a checkpoint
    var model = GPT2()
    gpt2_build_from_checkpoint(&model, "gpt2_124M.bin", info)

    let C = model.config.channels
    let V = model.config.vocab_size
    let Vp = model.config.padded_vocab_size
    let maxT = model.config.max_seq_len
    let L = model.config.num_layers

    // load additional information that we will use for debugging and error checking
    let state_filename = "gpt2_124M_debug_state.bin"
    guard
        let state_file = FileHandle(forReadingAtPath: state_filename)
    else { fatalError("Error opening state file \(state_filename)") }
    let state_fd = state_file.fileDescriptor
    guard
        let header_data = try? state_file.read(upToCount: 256 * MemoryLayout<Int32>.size)
    else { fatalError("Error reading header from state file \(state_filename)") }
    let state_header = header_data.withUnsafeBytes { (state_header: UnsafeRawBufferPointer) -> [Int] in
        state_header.bindMemory(to: Int32.self).map { Int($0) }
    }
    assert(state_header[0] == 20240327, "Bad magic state file \(state_filename)")
    if state_header[1] != 2 {
        fatalError("Wrong version \(state_header[1]) instead of 2 found in state file \(state_filename) (try `python train_gpt2.py`)")
    }
    let B = state_header[2] // batch size, e.g. 4
    let T = state_header[3] // time / sequence length (e.g. 64, up to maxT)
    info?("[State]\n")
    info?("batch_size: \(B)\n")
    info?("seq_len: \(T)\n")

    var expected_grads = ParameterTensors()
    let expected_grads_memory = malloc_and_point_parameters(&expected_grads, model.param_sizes)

    // read reference information from Python
    guard
        let x_data = try? state_file.read(upToCount: B * T * MemoryLayout<Int32>.size),
        let y_data = try? state_file.read(upToCount: B * T * MemoryLayout<Int32>.size),
        let expected_logits_data = try? state_file.read(upToCount: B * T * V * MemoryLayout<Float>.size),
        let expected_loss_data = try? state_file.read(upToCount: MemoryLayout<Float>.size),
        (try? FileDescriptor(rawValue: state_fd).read(into: UnsafeMutableRawBufferPointer(expected_grads_memory))) != nil
    else { fatalError("Error reading state file\(state_filename)") }
    // inputs and expected outputs, only used for error checking
    let x = x_data.withUnsafeBytes { $0.bindMemory(to: Int32.self) }
    let y = y_data.withUnsafeBytes { $0.bindMemory(to: Int32.self) }
    let expected_logits = expected_logits_data.withUnsafeBytes { $0.bindMemory(to: Float.self) }
    let expected_loss = expected_loss_data.withUnsafeBytes { $0.load(as: Float.self) }
    try? state_file.close()

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

        await gpt2_forward(&model, x.baseAddress!, y.baseAddress!, B, T, info)
        gpt2_zero_grad(&model)
        await gpt2_backward(&model)

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
                        info?("\(expected_logits[i]) \(calculated_logits![i])\n")
                    }
                    let diff = fabsf(expected_logits[bt * V + v] - calculated_logits![i])
                    max_diff = fmaxf(max_diff, diff)
                    if diff >= 1e-2 {
                        info?("MISMATCH AT INDEX \(bt),\(v): \(expected_logits[bt*V + v]) \(calculated_logits![i])\n")
                        logits_ok = false
                        break // break out of both loops
                    }
                }
                if !logits_ok { break }
            }
            info?("\(logits_ok ? "" : "NOT ")OK (LOGITS), max_diff = \(max_diff)\n")
            allok = allok && logits_ok

            // compare the achieved loss
            if fabsf(model.mean_loss - expected_loss) >= 1e-2 {
                info?("LOSS MISMATCH: \(model.mean_loss) \(expected_loss)\n")
                allok = false
            } else {
                info?("LOSS OK: \(model.mean_loss) \(expected_loss)\n")
            }

            // finally check all the gradients
            var gradoks = [Bool](repeating: false, count: 16)
            let grads = model.grads
            gradoks[0] = check_tensor(grads.wte, expected_grads.wte, V * C, "dwte", info)
            gradoks[1] = check_tensor(grads.wpe, expected_grads.wpe, maxT * C, "dwpe", info)
            gradoks[2] = check_tensor(grads.ln1w, expected_grads.ln1w, L * C, "dln1w", info)
            gradoks[3] = check_tensor(grads.ln1b, expected_grads.ln1b, L * C, "dln1b", info)
            gradoks[4] = check_tensor(grads.qkvw, expected_grads.qkvw, L * 3 * C * C, "dqkvw", info)
            gradoks[5] = check_tensor(grads.qkvb, expected_grads.qkvb, L * 3 * C, "dqkvb", info)
            gradoks[6] = check_tensor(grads.attprojw, expected_grads.attprojw, L * C * C, "dattprojw", info)
            gradoks[7] = check_tensor(grads.attprojb, expected_grads.attprojb, L * C, "dattprojb", info)
            gradoks[8] = check_tensor(grads.ln2w, expected_grads.ln2w, L * C, "dln2w", info)
            gradoks[9] = check_tensor(grads.ln2b, expected_grads.ln2b, L * C, "dln2b", info)
            gradoks[10] = check_tensor(grads.fcw, expected_grads.fcw, L * 4 * C * C, "dfcw", info)
            gradoks[11] = check_tensor(grads.fcb, expected_grads.fcb, L * 4 * C, "dfcb", info)
            gradoks[12] = check_tensor(grads.fcprojw, expected_grads.fcprojw, L * C * 4 * C, "dfcprojw", info)
            gradoks[13] = check_tensor(grads.fcprojb, expected_grads.fcprojb, L * C, "dfcprojb", info)
            gradoks[14] = check_tensor(grads.lnfw, expected_grads.lnfw, C, "dlnfw", info)
            gradoks[15] = check_tensor(grads.lnfb, expected_grads.lnfb, C, "dlnfb", info)
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
        info?("step \(step): loss \(model.mean_loss) (took \(String(format: "%1.2f", (end - start) * 1000)) ms) OK = \(step_loss_ok)\n")
    }

    // final judgement
    info?("overall okay: \(allok)\n")

    // free everything
    expected_grads_memory.deallocate()
    gpt2_free(&model)
}
