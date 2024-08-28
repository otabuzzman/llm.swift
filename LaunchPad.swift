import Metal

enum LaunchPadError: Error {
    case apiReturnedNil(api: String)
    case miscellaneous(info: String)
}

extension LaunchPadError: LocalizedError {
    var errorDescription: String? {
        switch self {
        case .apiReturnedNil(let api):
            return NSLocalizedString("API \(api) returned nil", comment: "")
        case .miscellaneous(let info):
            return NSLocalizedString("internal error: \(info)", comment: "")
        }
    }
}

struct KernelContext {
    let threadsPerGrid: MTLSize
    let threadsPerGroup: MTLSize
}

protocol KernelParam {}
extension UnsafeMutableRawPointer: KernelParam {}
extension Float: KernelParam {}
extension Int32: KernelParam {}

struct LaunchPadDescriptor {}

struct LaunchPad {
    private var descriptor = LaunchPadDescriptor()

    private let device: MTLDevice
    private let queue: MTLCommandQueue

    private let library: MTLLibrary

    private var kernel = [String: MTLComputePipelineState]()
    private var buffer = [MTLBuffer?]()

    // transient objects
    private var command: MTLCommandBuffer?
    private var encoder: MTLComputeCommandEncoder?
}

extension LaunchPad {
    init(descriptor: LaunchPadDescriptor? = nil) throws {
        if let descriptor = descriptor { self.descriptor = descriptor }

        guard let device = MTLCreateSystemDefaultDevice() else {
            throw LaunchPadError.apiReturnedNil(api: "MTLCreateSystemDefaultDevice")
        }
        self.device = device
        guard let queue = device.makeCommandQueue() else {
            throw LaunchPadError.apiReturnedNil(api: "makeCommandQueue")
        }
        self.queue = queue
        // guard let library = device.makeDefaultLibrary() else {
        guard let library = try? device.makeLibrary(source: defaultLibrary, options: nil) else {
            throw LaunchPadError.apiReturnedNil(api: "makeDefaultLibrary")
        }
        self.library = library

        try makeTransientObjects()
    }

    private mutating func makeTransientObjects() throws {
        guard let command = queue.makeCommandBuffer() else {
            throw LaunchPadError.apiReturnedNil(api: "makeCommandBuffer")
        }
        self.command = command
        guard let encoder = command.makeComputeCommandEncoder() else {
            throw LaunchPadError.apiReturnedNil(api: "makeComputeCommandEncoder")
        }
        self.encoder = encoder
    }

    mutating func registerKernel(name: String, _ preserve: Bool = true) throws {
        if preserve {
            if kernel.contains(where: { $0.0 == name }) { return }
        }
        guard let function = library.makeFunction(name: name) else {
            throw LaunchPadError.apiReturnedNil(api: "makeFunction \(name)")
        }
        let pipeline = try device.makeComputePipelineState(function: function)
        kernel[name] = pipeline
    }

    mutating func registerBuffer(address: UnsafeMutableRawPointer, length: Int, _ preserve: Bool = true) throws {
        if preserve {
            if (try? lookupBuffer(for: address)) != nil { return }
        }
        guard
            let buffer = device.makeBuffer(bytesNoCopy: address, length: length, options: [.storageModeShared])
        else { throw LaunchPadError.apiReturnedNil(api: "makeBuffer") }
        if let index = self.buffer.firstIndex(where: { $0 == nil }) {
            self.buffer[index] = buffer
        } else {
            self.buffer.append(buffer)
        }
    }

    mutating func unregisterBuffer(address: UnsafeMutableRawPointer) {
        if let (index, _) = try? lookupBuffer(for: address) { buffer[index] = nil }
    }

    private func lookupBuffer(for address: UnsafeMutableRawPointer) throws -> (Int, UnsafeMutableRawPointer) {
        for index in 0..<buffer.count {
            if let buffer = self.buffer[index] {
                let bufferBaseAddress = buffer.contents()
                let bufferLastAddress = bufferBaseAddress + buffer.length
                if (bufferBaseAddress..<bufferLastAddress).contains(address) {
                    return (index, bufferBaseAddress)
                }
            }
        }
        throw LaunchPadError.miscellaneous(info: "no buffer found")
    }

    func dispatchKernel(name: String, context: KernelContext, params: [KernelParam]) throws {
        guard
            let kernel = self.kernel[name]
        else { throw LaunchPadError.miscellaneous(info: "kernel \(name) not registered") }
        encoder?.setComputePipelineState(kernel)

        var index = 0
        for param in params {
            switch param {
            case is UnsafeMutableRawPointer:
                let address = (param as? UnsafeMutableRawPointer)!
                let (bufferIndex, bufferAddress) = try lookupBuffer(for: address)
                let offset = address - bufferAddress
                encoder?.setBuffer(buffer[bufferIndex], offset: offset, index: index)
                index += 1
            case is Float:
                var scalar = (param as? Float)!
                encoder?.setBytes(&scalar, length: MemoryLayout<Float>.stride, index: index)
                index += 1
            case is Int32:
                var scalar = (param as? Int32)!
                encoder?.setBytes(&scalar, length: MemoryLayout<Int32>.stride, index: index)
                index += 1
            default:
                break
            }
        }

        encoder?.dispatchThreadgroups(context.threadsPerGrid, threadsPerThreadgroup: context.threadsPerGroup)
    }

    mutating func commit(wait: Bool = false) throws {
        encoder?.endEncoding()

        command?.commit()
        if wait { command?.waitUntilCompleted() }

        try makeTransientObjects()
    }
}
