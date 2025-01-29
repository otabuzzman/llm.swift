#if LAYER_PASS_STANDALONE

import Foundation

let argv = CommandLine.arguments
let argc = argv.count

let release = argv[0].range(of: "layer_pass$", options: [.regularExpression]) == nil

let layerPassNameArgvIndex = release ? 0 : 1

let layers: [String: (Int, [String]) async throws -> Void] = [
    "encoder_forward": encoder_forward,
    "layernorm_forward": layernorm_forward,
    "matmul_forward": matmul_forward,
    "attention_forward": attention_forward,
    "residual_forward": residual_forward,
    "gelu_forward": gelu_forward,
    "softmax_forward": softmax_forward,
    "crossentropy_forward": crossentropy_forward
]

guard
    let layerPassName = URL(string: argv[layerPassNameArgvIndex])?.lastPathComponent
else { fatalError("\(argv[layerPassNameArgvIndex]): invalid") }
guard
    let layerPassFunc = layers[layerPassName]
else { fatalError("\(layerPassName): unknown") }

do {
    if launchPad == nil { launchPad = try LaunchPad() }
    try launchPad?.makeCommandBuffer()

    try await layerPassFunc(argc, release ? argv : Array(argv[1..<argc]))
} catch let error as LlmSwiftError {
    fatalError("\(error.localizedDescription)")
} catch let error as LaunchPadError {
    fatalError("\(error.localizedDescription)")
} catch {
    fatalError("caught exception: \(error)")
}

#endif // LAYER_PASS_STANDALONE
