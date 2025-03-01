// swiftlint:disable:next blanket_disable_command
// swiftlint:disable identifier_name

import Metal

public enum LaunchPadError: Error {
    case apiException(api: String, error: Error)
    case apiReturnedNil(api: String)
    case miscellaneous(info: String)
}

extension LaunchPadError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .apiException(let api, let error):
            return NSLocalizedString("API \(api) threw error:\n\(error)", comment: "")
        case .apiReturnedNil(let api):
            return NSLocalizedString("API \(api) returned nil", comment: "")
        case .miscellaneous(let info):
            return NSLocalizedString("internal error: \(info)", comment: "")
        }
    }
}

struct ThreadgroupMemoryDescriptor {
    enum Scope {
        case threadgroup // `threads_per_threadgroup´ * self.units * MemoryLayout.size(ofValue: self.type)
        case simdgroup // `threads_per_simdgroup´ * ...
        case simdstake // `simdgroups_per_threadgroup´ * ...
    }

    let scope: Scope
    let units: Int
    let type: Any.Type
}

struct KernelContext {
    let threadsPerGrid: Int // CUDA grid size
    let threadsPerGroup: Int // CUDA block size
    private(set) var threadgroupMemory: ThreadgroupMemoryDescriptor?
}

protocol KernelParam {}
extension UnsafeMutableRawPointer: KernelParam {}
extension Float: KernelParam {}
extension Int32: KernelParam {}

public struct LaunchPadDescriptor {}

public struct LaunchPad {
    private var descriptor = LaunchPadDescriptor()

    public let device: MTLDevice
    private let queue: MTLCommandQueue

    private let library: MTLLibrary

    private var kernel = [String: MTLComputePipelineState]()
    private var buffer = [MTLBuffer?]()

    // transient objects
    private var command = [MTLCommandBuffer]()
    private var encoder: MTLComputeCommandEncoder? // of command.last

    // capture GPU workload
    private var manager: MTLCaptureManager?
}

extension LaunchPad {
    public init(descriptor: LaunchPadDescriptor? = nil) throws {
        if let descriptor = descriptor { self.descriptor = descriptor }

        guard let device = MTLCreateSystemDefaultDevice() else {
            throw LaunchPadError.apiReturnedNil(api: "MTLCreateSystemDefaultDevice")
        }
        self.device = device
        // to few command buffers here let makeCommandBuffer() freeze
        // https://stackoverflow.com/q/41206620 (warrenm's comment on Q)
        guard let queue = device.makeCommandQueue(maxCommandBufferCount: 128) else {
            throw LaunchPadError.apiReturnedNil(api: "makeCommandQueue")
        }
        self.queue = queue
        do {
            #if os(macOS)
            self.library = try device.makeDefaultLibrary(bundle: Bundle.main)
            #else
            self.library = try device.makeLibrary(source: defaultLibrary, options: nil)
            #endif
        } catch { throw LaunchPadError.apiException(api: "makeLibrary", error: error)}
    }

    @discardableResult
    mutating func makeCommandBuffer(createEncoder: Bool = true) throws -> (MTLCommandBuffer, MTLComputeCommandEncoder?) {
        if let encoder = encoder { encoder.endEncoding() } // reset to nil in self.commit()
        guard let command = queue.makeCommandBuffer() else {
            throw LaunchPadError.apiReturnedNil(api: "makeCommandBuffer")
        }
        self.command.append(command)

        if !createEncoder { encoder = nil ; return (command, nil) }

        guard let encoder = command.makeComputeCommandEncoder() else {
            throw LaunchPadError.apiReturnedNil(api: "makeComputeCommandEncoder")
        }
        self.encoder = encoder
        return (command, encoder)
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

    func lookupBuffer(for address: UnsafeMutableRawPointer) throws -> (Int, MTLBuffer) {
        for index in 0..<buffer.count {
            if let buffer = self.buffer[index] {
                let bufferBaseAddress = buffer.contents()
                let bufferLastAddress = bufferBaseAddress + buffer.length
                if (bufferBaseAddress..<bufferLastAddress).contains(address) {
                    return (index, buffer)
                }
            }
        }
        throw LaunchPadError.miscellaneous(info: "\(#function): no buffer found")
    }

    // swiftlint:disable:next function_body_length
    func dispatchKernel<each KernelParam>(name: String, context: KernelContext, params: repeat each KernelParam) throws {
        guard
            let kernel = self.kernel[name]
        else { throw LaunchPadError.miscellaneous(info: "\(#function): kernel \(name) not registered") }
        encoder?.setComputePipelineState(kernel)

        var index = 0 // argument location index, for MSL kernel functions
        for param in repeat each params {
            switch param {
            case is UnsafeMutableRawPointer:
                let address = (param as? UnsafeMutableRawPointer)!
                let (bufferIndex, bufferObject) = try lookupBuffer(for: address)
                let offset = address - bufferObject.contents()
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

        // switch to MSL specification names
        let threads_per_grid = MTLSize(context.threadsPerGrid)
        var threads_per_threadgroup: MTLSize // CUDA blockDim
        let threads_per_simdgroup = kernel.threadExecutionWidth // CUDA warp size
        var simdgroups_per_threadgroup: Int // CUDA blockDim / 32

        // using 1D grid and threadgroups
        if context.threadsPerGroup > 0 {
            simdgroups_per_threadgroup = context.threadsPerGroup / threads_per_simdgroup
            threads_per_threadgroup = MTLSize(context.threadsPerGroup)
        } else {
            simdgroups_per_threadgroup = kernel.maxTotalThreadsPerThreadgroup / threads_per_simdgroup
            threads_per_threadgroup = MTLSize(simdgroups_per_threadgroup * threads_per_simdgroup)
        }

        assert(threads_per_threadgroup.width % threads_per_simdgroup == 0)

        index = 0 // argument location index, for MSL threadgroup arguments
        if let threadgroupMemory = context.threadgroupMemory {
            var threadgroupMemoryLength = MemoryLayout.size(ofValue: threadgroupMemory.type)
            switch threadgroupMemory.scope {
            case .threadgroup:
                threadgroupMemoryLength *= threads_per_threadgroup.width * threadgroupMemory.units
            case .simdgroup:
                threadgroupMemoryLength *= kernel.threadExecutionWidth * threadgroupMemory.units
            case .simdstake:
                threadgroupMemoryLength *= simdgroups_per_threadgroup * threadgroupMemory.units
            }
            encoder?.setThreadgroupMemoryLength(threadgroupMemoryLength, index: index)
        }

        encoder?.dispatchThreads(threads_per_grid, threadsPerThreadgroup: threads_per_threadgroup)
    }

    mutating func commit(wait: Bool = false) throws {
        guard let latest = command.last else { return }

        encoder?.endEncoding()
        encoder = nil // checked in makeCommandBuffer()

        for buffer in command { buffer.commit() }
        command.removeAll(keepingCapacity: true)
        try makeCommandBuffer()

        if wait { latest.waitUntilCompleted() }
    }

    mutating func startCapture(_ gpuTraceFile: URL? = nil) throws {
        let control = MTLCaptureDescriptor()
        control.captureObject = device

        if let gpuTraceFile = gpuTraceFile {
            control.destination = .gpuTraceDocument
            control.outputURL = gpuTraceFile
        } else {
            control.destination = .developerTools
        }

        if manager == nil { manager = MTLCaptureManager.shared() }

        try manager?.startCapture(with: control)
    }

    func closeCapture() {
        guard let manager = manager, manager.isCapturing else { return }
        manager.stopCapture()
    }
}

// for brevity
extension MTLSize {
    init(_ width: Int, _ height: Int = 1, _ depth: Int = 1) {
        self.init(width: width, height: height, depth: depth)
    }
}
