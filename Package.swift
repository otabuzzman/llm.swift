// swift-tools-version: 5.9


import PackageDescription
 
let package = Package(
    name: "LlmDotSwift",
    platforms: [
        .iOS("18.1"), .macOS("15.2"),
    ],
    products: [
        .library(
            name: "LlmDotSwift",
            targets: ["LlmDotSwift"]),
    ],
    dependencies: [
        .package(url: "https://github.com/Bouke/Glob.git", "1.0.5"..<"2.0.0")
    ],
    targets: [
        .target(
            name: "LlmDotSwift",
            dependencies: [
                "Glob",
            ],
            path: ".",
            exclude: [
                "dev/metal/DefaultLibrary.metal",
            ],
            sources: [
                "llmc",
                "dev/metal",
                "LaunchPad.swift",
                "test_gpt2.swift",
                "train_gpt2.swift",
            ]),
    ]
)
