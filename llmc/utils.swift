// swiftlint:disable:next blanket_disable_command
// swiftlint:disable identifier_name
// swiftlint:disable:next blanket_disable_command
// swiftlint:disable large_tuple

import Foundation

func indicesOf(combined: Int, _ X: Int, _ Y: Int, _ Z: Int = 1) -> (Int, Int, Int) {
    assert(X * Y * Z > combined, "Combined index out of bounds")
    let x, y, z: Int
    if Z == 1 {
         z = 0
         y = combined / X
         x = combined % X
    } else {
         z = combined / (X * Y)
         y = (combined % (X * Y)) / X
         x = combined % X
    }
    return (x, y, z)
}
