#if LLMSWIFT_STANDALONE

import Foundation
import System

let argv = CommandLine.arguments

let test = argv[0].range(of: "train[^/]+$", options: [.regularExpression]) == nil
let data = argv.count > 1 ? URL(fileURLWithPath: argv[1], isDirectory: true) : nil

do {
    if launchPad == nil { launchPad = try LaunchPad() }

    if test {
        try await test_gpt2(data, { print($0, terminator: "") })
    } else {
        try await train_gpt2(data, { print($0, terminator: "") })
    }
} catch { fatalError("\(error)") }

#endif // LLMSWIFT_STANDALONE
