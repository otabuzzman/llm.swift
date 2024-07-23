import Foundation
import System

let argv = CommandLine.arguments

let test = argv[0].range(of: "train[^/]+$", options: [.regularExpression]) == nil
let data = argv.count > 1 ? URL(fileURLWithPath: argv[1], isDirectory: true) : nil

if test {
    await test_gpt2(data, { print($0, terminator: "") })
} else {
    await train_gpt2(data, { print($0, terminator: "") })
}
