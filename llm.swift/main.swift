//
//  main.swift
//  llm.swift
//
//  Created by Jürgen Schuck on 10.05.24.
//

import Foundation
import System

let argv = CommandLine.arguments

let test = argv[0].range(of: "/train[^/]+$", options: [.regularExpression]) == nil
let data = URL(fileURLWithPath: argv.count > 1 ? argv[1] : ".", isDirectory: true)
test_mt19937()
//await test ? test_gpt2(data) : train_gpt2(data)
