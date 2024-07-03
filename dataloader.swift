// ----------------------------------------------------------------------------
// data loader lite
// returns random batches of data from a file of integers

import Foundation

struct DataLoader {
    // hyperparameters
    var B = 0 // batch size
    var T = 0 // sequence length
    // input handling and its state
    var tokens_file: FileHandle?
    var file_size = 0
    var current_position = 0
    // output memory
    var batch = UnsafeMutableBufferPointer<Int32>.allocate(capacity: 0)
    var inputs = UnsafeMutablePointer<Int32>.allocate(capacity: 0)
    var targets = UnsafeMutablePointer<Int32>.allocate(capacity: 0)
    // convenience variables
    var num_batches = 0
}

func dataloader_init(_ loader: UnsafeMutablePointer<DataLoader>, _ filename: URL, _ B: Int, _ T: Int) -> Void {
    loader.pointee.B = B
    loader.pointee.T = T
    
    // open the input file for reading
    guard
        let tokens_file = try? FileHandle(forReadingFrom: filename)
    else { fatalError("Error opening tokens file") }
    loader.pointee.tokens_file = tokens_file
    
    // determine the file size
    loader.pointee.file_size = Int((try? tokens_file.seekToEnd()) ?? 0)
    try? tokens_file.seek(toOffset: 0)
    if loader.pointee.file_size < (B * T + 1) * MemoryLayout<Int32>.size {
        fatalError("File size too small for batch size and sequence length")
    }
    loader.pointee.current_position = 0 // start at the beginning
    
    // allocate space for B * T + 1 integers to store the inputs and targets
    // loader.pointee.batch = UnsafeMutablePointer<Int32>.allocate(capacity: B * T + 1)
    // loader.pointee.inputs = loader.pointee.batch
    // loader.pointee.targets = loader.pointee.batch + 1 // targets are shifted by one
    loader.pointee.num_batches = loader.pointee.file_size / (B * T * MemoryLayout<Int32>.size)
}

func dataloader_reset(_ loader: UnsafeMutablePointer<DataLoader>) -> Void {
    loader.pointee.current_position = 0
}

func dataloader_next_batch(_ loader: UnsafeMutablePointer<DataLoader>) -> Void {
    let B = loader.pointee.B
    let T = loader.pointee.T
    // if we are at the end of the file, loop back to the beginning
    if loader.pointee.current_position + (B * T + 1) * MemoryLayout<Int32>.size > loader.pointee.file_size {
        loader.pointee.current_position = 0
    }
    // read the B * T + 1 integers from the file into batch
    try? loader.pointee.tokens_file!.seek(toOffset: UInt64(loader.pointee.current_position))
    guard
        let file_data = try? loader.pointee.tokens_file!.read(upToCount: (B * T + 1) * MemoryLayout<Int32>.size)
    else { fatalError("Error reading tokens file") }
    var batch_data = file_data
    loader.pointee.batch = batch_data.withUnsafeMutableBytes { $0.bindMemory(to: Int32.self) }
    loader.pointee.inputs = loader.pointee.batch.baseAddress!
    loader.pointee.targets = loader.pointee.batch.baseAddress! + 1
    // advance the current position by B * T integers
    loader.pointee.current_position += B * T * MemoryLayout<Int32>.size
}

func dataloader_free(_ loader: UnsafeMutablePointer<DataLoader>) -> Void {
    try? loader.pointee.tokens_file!.close()
    // loader.pointee.batch.deallocate()
}
