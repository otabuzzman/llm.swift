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
        case threadgroup // `threads_per_threadgroup' * self.units * MemoryLayout.size(ofValue: self.type)
        case simdgroup // `threads_per_simdgroup' * ...
        case simdstake // `simdgroups_per_threadgroup' * ...
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

protocol Constant {}
extension Float: Constant {}
extension Int32: Constant {}
extension UInt32: Constant {}

protocol KernelParam {}
extension UnsafeMutableRawPointer: KernelParam {}
extension Float: KernelParam {}
extension Int32: KernelParam {}

public struct LaunchPadDescriptor {
    var shaderPixelFormat: MTLPixelFormat = .rgba8Uint
}

public struct LaunchPad {
    private var descriptor = LaunchPadDescriptor()

    public let device: MTLDevice
    private let queue: MTLCommandQueue

    private let library: MTLLibrary
    private var function = [String: MTLFunction]()

    private var kernel = [String: MTLComputePipelineState]()
    private var shader = [String: MTLRenderPipelineState]()
    private var buffer = [MTLBuffer?]()

    // transient objects
    private var command = [MTLCommandBuffer]()
    private var encoder: MTLCommandEncoder? // of command.last

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
        } catch { throw LaunchPadError.apiException(api: "makeLibrary", error: error) }
    }

    mutating func makeCommandBuffer(computePassDescriptor descriptor: MTLComputePassDescriptor?, customize: ((MTLCommandBuffer, MTLComputeCommandEncoder?) -> Void)? = nil) throws {
        if let encoder = encoder { encoder.endEncoding() } // reset to nil in self.commit()
        guard let command = queue.makeCommandBuffer() else {
            throw LaunchPadError.apiReturnedNil(api: "makeCommandBuffer")
        }
        self.command.append(command)
        if let descriptor = descriptor {
            guard let encoder = command.makeComputeCommandEncoder(descriptor: descriptor) else {
                throw LaunchPadError.apiReturnedNil(api: "makeComputeCommandEncoder")
            }
            self.encoder = encoder
        } else {
            encoder = nil
        }
        customize?(command, encoder as? MTLComputeCommandEncoder)
    }

    mutating func makeCommandBuffer(renderPassDescriptor descriptor: MTLRenderPassDescriptor, customize: ((MTLCommandBuffer, MTLRenderCommandEncoder) -> Void)? = nil) throws {
        if let encoder = encoder { encoder.endEncoding() } // reset to nil in self.commit()
        guard let command = queue.makeCommandBuffer() else {
            throw LaunchPadError.apiReturnedNil(api: "makeCommandBuffer")
        }
        self.command.append(command)
        guard let encoder = command.makeRenderCommandEncoder(descriptor: descriptor) else {
            throw LaunchPadError.apiReturnedNil(api: "makeRenderCommandEncoder")
        }
        self.encoder = encoder
        customize?(command, encoder as MTLRenderCommandEncoder)
    }

    mutating func registerFunction<each Constant>(name: String, constants: repeat each Constant, preserve: Bool = true) throws {
        if preserve {
            if function.contains(where: { $0.0 == name }) { return }
        }

        let constantValues = MTLFunctionConstantValues()

        var index = -1 // constant location index, for MSL functions
        for constant in repeat each constants {
            switch constant {
            case is Float:
                // swiftlint:disable:next force_cast
                var constant: Float = constant as! Float // avoid compiler warning
                constantValues.setConstantValue(&constant, type: .float, index: index + 1)
            case is Int32:
                // swiftlint:disable:next force_cast
                var constant: Int32 = constant as! Int32
                constantValues.setConstantValue(&constant, type: .int, index: index + 1)
            case is UInt32:
                // swiftlint:disable:next force_cast
                var constant: UInt32 = constant as! UInt32
                constantValues.setConstantValue(&constant, type: .uint, index: index + 1)
            default:
                continue
            }
            index += 1
        }

        let function: MTLFunction?
        if index == -1 { // no constants given
            function = library.makeFunction(name: name)
            if function == nil {
                throw LaunchPadError.apiReturnedNil(api: "makeFunction \(name)")
            }
        } else {
            do {
                function = try library.makeFunction(name: name, constantValues: constantValues)
            } catch { throw LaunchPadError.apiException(api: "makeFunction", error: error) }
        }
        self.function[name] = function
    }

    mutating func registerKernel(name: String, functions: String..., customize: ((MTLComputePipelineState) -> Void)? = nil, preserve: Bool = true) throws {
        if preserve {
            if kernel.contains(where: { $0.0 == name }) { return }
        }

        try registerFunction(name: name, constants: ())

        var linkedFunctions: MTLLinkedFunctions?
        if functions.count > 0 {
            linkedFunctions = MTLLinkedFunctions()
            linkedFunctions!.functions = []
            for name in functions {
                guard
                    let function = self.function[name]
                else { throw LaunchPadError.miscellaneous(info: "\(#function): function \(name) not registered") }
                linkedFunctions!.functions?.append(function)
            }
        }

        let descriptor = MTLComputePipelineDescriptor()
        descriptor.computeFunction = function[name]
        descriptor.linkedFunctions = linkedFunctions

        let pipeline = try device.makeComputePipelineState(descriptor: descriptor, options: MTLPipelineOption(), reflection: nil)

        customize?(pipeline)

        kernel[name] = pipeline
    }

    mutating func registerShader(vertex: String, fragment: String, customize: ((MTLRenderPipelineState) -> Void)? = nil) throws {
        if !shader.contains(where: { $0.0 == vertex }) {
            try registerFunction(name: vertex, constants: ())
        }
        if !shader.contains(where: { $0.0 == fragment }) {
            try registerFunction(name: fragment, constants: ())
        }

        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.vertexFunction = function[vertex]
        descriptor.fragmentFunction = function[fragment]
        descriptor.colorAttachments[0].pixelFormat = self.descriptor.shaderPixelFormat

        let pipeline = try device.makeRenderPipelineState(descriptor: descriptor, options: MTLPipelineOption(), reflection: nil)

        customize?(pipeline)

        shader[vertex] = pipeline
    }

    mutating func registerBuffer(address: UnsafeMutableRawPointer, length: Int, preserve: Bool = true) throws {
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
    func dispatchKernel<each KernelParam>(name: String, context: KernelContext, params: repeat each KernelParam, customize: ((MTLComputeCommandEncoder) -> Void)? = nil) throws {
        guard
            let kernel = self.kernel[name]
        else { throw LaunchPadError.miscellaneous(info: "\(#function): kernel \(name) not registered") }
        guard
            let encoder = encoder as? MTLComputeCommandEncoder
        else { throw LaunchPadError.miscellaneous(info: "\(#function): encoder not set") }
        encoder.setComputePipelineState(kernel)

        var index = 0 // argument location index, for MSL kernel functions
        for param in repeat each params {
            switch param {
            case is UnsafeMutableRawPointer:
                let address = (param as? UnsafeMutableRawPointer)!
                let (bufferIndex, bufferObject) = try lookupBuffer(for: address)
                let offset = address - bufferObject.contents()
                encoder.setBuffer(buffer[bufferIndex], offset: offset, index: index)
            case is Float:
                var scalar = (param as? Float)!
                encoder.setBytes(&scalar, length: MemoryLayout<Float>.stride, index: index)
            case is Int32:
                var scalar = (param as? Int32)!
                encoder.setBytes(&scalar, length: MemoryLayout<Int32>.stride, index: index)
            default:
                continue
            }
            index += 1
        }

        customize?(encoder)

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
            encoder.setThreadgroupMemoryLength(threadgroupMemoryLength, index: index)
        }

        encoder.dispatchThreads(threads_per_grid, threadsPerThreadgroup: threads_per_threadgroup)
    }

    func dispatchShader(name: String, customize: ((MTLRenderCommandEncoder) -> Void)? = nil) throws {
        guard
            let shader = self.shader[name]
        else { throw LaunchPadError.miscellaneous(info: "\(#function): shader \(name) not registered") }
        guard
            let encoder = encoder as? MTLRenderCommandEncoder
        else { throw LaunchPadError.miscellaneous(info: "\(#function): encoder not set") }
        encoder.setRenderPipelineState(shader)

        customize?(encoder)
    }

    mutating func commit(wait: Bool = false) throws {
        guard let latest = command.last else { return }

        encoder?.endEncoding()
        encoder = nil // checked in makeCommandBuffer()

        for buffer in command { buffer.commit() }
        command.removeAll(keepingCapacity: true)

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
