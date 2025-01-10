#if ENCODER_FORWARD_STANDALONE

let argv = CommandLine.arguments
let argc = argv.count

do {
    if launchPad == nil { launchPad = try LaunchPad() }

    try main(argc, argv)
} catch let error as LlmSwiftError {
    fatalError("\(error.localizedDescription)")
} catch let error as LaunchPadError {
    fatalError("\(error.localizedDescription)")
} catch {
    fatalError("caught exception: \(error)")
}

#endif // ENCODER_FORWARD_STANDALONE
