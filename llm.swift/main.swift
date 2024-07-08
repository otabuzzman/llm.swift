//
//  main.swift
//  llm.swift
//
//  Created by Jürgen Schuck on 10.05.24.
//

import Foundation
import System

let argv = CommandLine.arguments

let test = argv[0].range(of: "train[^/]+$", options: [.regularExpression]) == nil
let data = argv.count > 1 ? URL(fileURLWithPath: argv[1], isDirectory: true) : nil

await test ? test_gpt2(data) : train_gpt2(data)
