// swiftlint:disable:next blanket_disable_command
// swiftlint:disable identifier_name

import Foundation

func ceil_div<T: FloatingPoint>(_ dividend: T, _ divisor: T) -> T {
    (dividend + divisor - 1) / divisor
}

func ceil_div<T: BinaryInteger>(_ dividend: T, _ divisor: T) -> T {
    (dividend + divisor - 1) / divisor
}

func validate_result(
    _ device_result: UnsafePointer<Float>,
    _ cpu_reference: UnsafePointer<Float>,
    _ name: String,
    _ num_elements: Int,
    _ tolerance: Float = 1e-4) throws {
    var nfaults = 0

    let epsilon: Float = 0.079

    for i in 0..<num_elements {
        // Skip masked elements
        if !cpu_reference[i].isFinite { continue }

        // print the first few comparisons
        if i < 5 {
            print("\(cpu_reference[i]) \(device_result[i])")
        }
        // effective tolerance is based on expected rounding error (epsilon)
        // plus any specified additional tolerance
        let t_eff = tolerance + abs(cpu_reference[i]) * epsilon
        // ensure correctnes for all elements
        if abs(cpu_reference[i] - device_result[i]) > t_eff {
            print("Mismatch of \(name) at \(i): CPU_ref: \(cpu_reference[i]) vs GPU: \(device_result[i])\n---")
            nfaults += 1
            if nfaults > 10 { fatalError("too many mismatched elements") }
        }
    }
}
