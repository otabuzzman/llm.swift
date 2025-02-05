// swift-tools-version: 5.9


import PackageDescription
 
let package = Package(
    name: "llm.swift",
    platforms: [
        .iOS("17"), .macOS("14")
    ],
    products: [
        .library(
            name: "llm.swift",
            targets: ["llm.swift"]),
    ],
    dependencies: [
        .package(url: "https://github.com/Bouke/Glob.git", "1.0.5"..<"2.0.0")
    ],
    targets: [
        .target(
            name: "llm.swift",
            dependencies: [
                "Glob",
            ],
            path: ".",
            #if os(iOS)
            exclude: [
                "dev/metal/DefaultLibrary.metal",
            ],
            #endif
            sources: [
                "llmc",
                "dev/metal",
                "LaunchPad.swift",
                "test_gpt2.swift",
                "train_gpt2.swift",
            ]),
    ]
)
