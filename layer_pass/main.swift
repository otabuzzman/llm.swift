#if LAYER_PASS_STANDALONE

import Foundation

let argv = CommandLine.arguments
let argc = argv.count

let layers: [String: (Int, [String]) throws -> Void] = [
    "encoder_forward": encoder_forward,
    "layernorm_forward": layernorm_forward,
    "matmul_forward": matmul_forward,
    "attention_forward": attention_forward,
    "residual_forward": residual_forward,
    "gelu_forward": gelu_forward,
    "softmax_forward": softmax_forward
]

guard
    let layerPassName = URL(string: argv[0])?.lastPathComponent
else { fatalError("\(argv[0]): invalid") }
guard
    let layerPassFunc = layers[layerPassName]
else { fatalError("\(layerPassName): unknown") }

do {
    if launchPad == nil { launchPad = try LaunchPad() }

    try layerPassFunc(argc, argv)
} catch let error as LlmSwiftError {
    fatalError("\(error.localizedDescription)")
} catch let error as LaunchPadError {
    fatalError("\(error.localizedDescription)")
} catch {
    fatalError("caught exception: \(error)")
}

#endif // LAYER_PASS_STANDALONE
