import Foundation
import System

let argv = CommandLine.arguments

let test = argv[0].range(of: "train[^/]+$", options: [.regularExpression]) == nil
let data = argv.count > 1 ? URL(fileURLWithPath: argv[1], isDirectory: true) : nil

// swiftlint:disable:next identifier_name
let matmul_forward: (UnsafeMutablePointer<Float>, UnsafePointer<Float>, UnsafePointer<Float>, UnsafePointer<Float>?, Int, Int, Int, Int) async -> Void = matmul_forward_default(_:_:_:_:_:_:_:_:)

do {
    if test {
        try await test_gpt2(data, { print($0, terminator: "") })
    } else {
        try await train_gpt2(data, { print($0, terminator: "") })
    }
} catch { fatalError("\(error)") }
