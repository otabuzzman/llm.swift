// swiftlint:disable:next blanket_disable_command
// swiftlint:disable identifier_name

/*
 Implements:
 - DataLoader for model training. Reads and serves data shards.
 - EvalLoader for multiple-choice evaluation datasets, e.g. HellaSwag.
 */

import Foundation
import Glob

enum DataLoaderError: Error {
    case corrupted
}

// Distributed Data Loader
let HEADER_SIZE = 256

struct DataLoader {
    // variables related to distributed training
    // each process/worker has to access different parts of the data
    var process_rank = 0
    var num_processes = 0
    // batch and token information
    var B = 0 // batch size
    var T = 0 // sequence length
    var num_tokens = 0 // total number of tokens
    var shard_num_samples = 0  // total number of samples in the current shard per process
    // shards and current position
    var glob_result: Glob! // stores the result of glob, for all shards we want to iterate
    var current_shard_idx = 0 // the current shard we are reading from
    var current_sample_idx = 0 // the current sample we are reading from
    // file handle
    var tokens_file: FileHandle?
    // data buffers
    // we fread data from file into this buffer
    var buffer: UnsafeMutablePointer<UInt16>!
    // input tokens into transformer
    var inputs: UnsafeMutablePointer<Int32>!
    // target tokens for the transformer
    var targets: UnsafeMutablePointer<Int32>!
    // random shuffle related variables
    var shuffle_rng: mt19937_state!
    var should_shuffle = false
    var shard_indices: UnsafeMutablePointer<Int32>!
    var intra_shard_indices: UnsafeMutablePointer<Int32>?
    // sizes in bytes
    var total_batch_size_bytes = 0  // total across all processes
    var local_batch_offset_bytes = 0  // inner-sample offset for this process
    var header_bytes = 0  // header size in bytes
    var file_size_bytes = 0
}

@discardableResult
private func dataloader_load_shard(_ loader: UnsafeMutablePointer<DataLoader>, _ shard_index: Int) throws -> Int {
    var file_index = shard_index
    if loader.pointee.should_shuffle {
        file_index = Int(loader.pointee.shard_indices[shard_index])
    }
    // use the first glob match as the filename for now
    let filename = loader.pointee.glob_result[file_index]
    // open the input file for reading. also only a single file can be opened at a time
    try? loader.pointee.tokens_file?.close()
    do {
        loader.pointee.tokens_file = try FileHandle(forReadingFrom: URL(string: filename)!)
    } catch { throw LlmSwiftError.apiReturnedNil }
    let tokens_file = loader.pointee.tokens_file! // brevity
    // validate the header
    guard
        let header_data = try tokens_file.read(upToCount: HEADER_SIZE * MemoryLayout<Int32>.size)
    else { throw LlmSwiftError.apiReturnedNil }
    let header = header_data.withUnsafeBytes { (header_data: UnsafeRawBufferPointer) -> [Int] in
        header_data.bindMemory(to: Int32.self).map { Int($0) }
    }
    assert(header[0] == 20240520, "Bad magic in data file (retry preprocessing or refer to README)")
    assert(header[1] == 1, "Wrong version in data file (retry preprocessing or refer to README)")

    let ntok = header[2] // number of tokens in the file
    if ntok == 0 { throw DataLoaderError.corrupted } // we expect some tokens in the file. this should never trip, right?
    // determine the file size and make sure it is consistent with the number of tokens
    loader.pointee.file_size_bytes = Int((try? tokens_file.seekToEnd()) ?? 0)
    try? tokens_file.seek(toOffset: 0)
    // we expect ntok in the file to be consistent with filesize, assert that is the case
    let expected_file_size = HEADER_SIZE * MemoryLayout<Int32>.size + ntok * MemoryLayout<UInt16>.size
    if loader.pointee.file_size_bytes != expected_file_size {
        throw DataLoaderError.corrupted
    }
    // -1 uint16_t due to us taking B*T+1 tokens but moving by B*T tokens
    loader.pointee.shard_num_samples = (ntok * MemoryLayout<UInt16>.size - MemoryLayout<UInt16>.size) / loader.pointee.total_batch_size_bytes
    return ntok
}

private func prepare_intra_shard_indices(_ loader: UnsafeMutablePointer<DataLoader>) {
    // shuffle the examples inside the shards
    loader.pointee.intra_shard_indices?.deallocate()
    loader.pointee.intra_shard_indices = UnsafeMutablePointer<Int32>.allocate(capacity: loader.pointee.shard_num_samples)
    init_identity_permutation(loader.pointee.intra_shard_indices!, loader.pointee.shard_num_samples)
    random_permutation(loader.pointee.intra_shard_indices!, loader.pointee.shard_num_samples, &loader.pointee.shuffle_rng)
}

func dataloader_reset(_ loader: UnsafeMutablePointer<DataLoader>) throws {
    loader.pointee.current_shard_idx = 0
    loader.pointee.current_sample_idx = 0

    if loader.pointee.should_shuffle {  // shuffle the shards
        random_permutation(loader.pointee.shard_indices, loader.pointee.glob_result.count, &loader.pointee.shuffle_rng)
    }

    try dataloader_load_shard(loader, loader.pointee.current_shard_idx)

    if loader.pointee.should_shuffle {
        prepare_intra_shard_indices(loader)
    }
}

private func dataloader_advance(_ loader: UnsafeMutablePointer<DataLoader>) throws {
    if loader.pointee.current_shard_idx == loader.pointee.glob_result.count - 1 {
        // if we are at the last shard, we reset the loader and start a new epoch
        try dataloader_reset(loader)
        return
    }

    // advance the loader by loading the next data shard and resetting the position
    loader.pointee.current_shard_idx = (loader.pointee.current_shard_idx + 1) % loader.pointee.glob_result.count
    loader.pointee.current_sample_idx = 0
    try dataloader_load_shard(loader, loader.pointee.current_shard_idx)

    if loader.pointee.should_shuffle {
        prepare_intra_shard_indices(loader)
    }
}

// swiftlint:disable:next function_parameter_count
func dataloader_init(_ loader: UnsafeMutablePointer<DataLoader>,
                     _ filename_pattern: String,
                     _ B: Int,
                     _ T: Int,
                     _ process_rank: Int,
                     _ num_processes: Int,
                     _ should_shuffle: Bool) throws {
    loader.pointee.process_rank = process_rank
    loader.pointee.num_processes = num_processes
    loader.pointee.B = B
    loader.pointee.T = T
    loader.pointee.should_shuffle = should_shuffle
    loader.pointee.header_bytes = HEADER_SIZE * MemoryLayout<Int32>.size
    loader.pointee.total_batch_size_bytes = ((loader.pointee.num_processes * (loader.pointee.B * loader.pointee.T)) * MemoryLayout<UInt16>.size)
    loader.pointee.local_batch_offset_bytes = loader.pointee.process_rank * loader.pointee.B * loader.pointee.T * MemoryLayout<UInt16>.size
    // glob to get the list of files matching the pattern, these are our data shards
    loader.pointee.glob_result = Glob(pattern: filename_pattern)
    if loader.pointee.glob_result.count == 0 {
        throw LlmSwiftError.wrongApiUsage
    }

    if should_shuffle {
        var shuffle_rng = mt19937_state()
        manual_seed(&shuffle_rng, 42 + process_rank)
        loader.pointee.shuffle_rng = shuffle_rng
        loader.pointee.shard_indices = UnsafeMutablePointer<Int32>.allocate(capacity: loader.pointee.glob_result.count)
        init_identity_permutation(loader.pointee.shard_indices, loader.pointee.glob_result.count)
    }

    // inspect and validate all shards so we don't get any runtime errors later
    // if too slow / too many shards, may wish to revisit later
    var ntok_total = 0
    for shard_index in 0..<loader.pointee.glob_result.count {
        let shard_ntok = try dataloader_load_shard(loader, shard_index)
        // we need at least one batch/shard, the way things are written right now.
        // can be relaxed a lot later.
        if shard_ntok < num_processes * B * T + 1 { throw LlmSwiftError.wrongApiUsage } // at least one batch per shard needed
        ntok_total += shard_ntok
    }
    // debugging prints
    // print("DataLoader: filename_pattern \(filename_pattern)")
    // print("DataLoader: Found \(ntok_total) tokens across \(loader.pointee.glob_result.count) shards")

    // allocate all the space we'll need
    loader.pointee.inputs = UnsafeMutablePointer<Int32>.allocate(capacity: B * T)
    loader.pointee.targets = UnsafeMutablePointer<Int32>.allocate(capacity: B * T)
    loader.pointee.num_tokens = ntok_total

    // reset the loader, to initialize it
    try dataloader_reset(loader)
}

func dataloader_load_batch(_ loader: UnsafeMutablePointer<DataLoader>) throws {
    if loader.pointee.should_shuffle && loader.pointee.intra_shard_indices == nil { throw LlmSwiftError.wrongApiUsage } // no shards to shuffle
    if loader.pointee.current_sample_idx >= loader.pointee.shard_num_samples { throw LlmSwiftError.wrongApiUsage } // sample index out of bounds
    let idx = loader.pointee.should_shuffle ? Int(loader.pointee.intra_shard_indices![loader.pointee.current_sample_idx]) : loader.pointee.current_sample_idx
    let global_batch_offset_bytes = idx * loader.pointee.total_batch_size_bytes
    let current_offset = loader.pointee.header_bytes + global_batch_offset_bytes + loader.pointee.local_batch_offset_bytes

    let B = loader.pointee.B
    let T = loader.pointee.T
    // read B*T+1 uint16_t tokens from the file into buffer
    try loader.pointee.tokens_file!.seek(toOffset: UInt64(current_offset))
    guard
        let file_data = try loader.pointee.tokens_file!.read(upToCount: (B * T + 1) * MemoryLayout<UInt16>.size)
    else { throw LlmSwiftError.apiReturnedNil }
    var token_data = file_data
    loader.pointee.buffer = token_data.withUnsafeMutableBytes { $0.bindMemory(to: UInt16.self) }.baseAddress
    // decode the buffer into inputs and targets (cast to int)
    for i in 0..<B * T {
        loader.pointee.inputs[i] = Int32(loader.pointee.buffer[i])
        loader.pointee.targets[i] = Int32(loader.pointee.buffer[i+1])
    }
}

func dataloader_next_batch(_ loader: UnsafeMutablePointer<DataLoader>) throws {
    // if the next batch would go past the end of the file, advance the loader
    if loader.pointee.current_sample_idx >= loader.pointee.shard_num_samples {
        try dataloader_advance(loader)
    }
    try dataloader_load_batch(loader)
    loader.pointee.current_sample_idx += 1
}

func dataloader_resume(_ loader: UnsafeMutablePointer<DataLoader>, _ current_shard_idx: Int, _ current_sample_idx: Int) throws {
    // used during model resumption (-y 1) flag
    loader.pointee.current_shard_idx = current_shard_idx
    loader.pointee.current_sample_idx = current_sample_idx
    try dataloader_load_shard(loader, loader.pointee.current_shard_idx)
}

func dataloader_free(_ loader: UnsafeMutablePointer<DataLoader>) {
    loader.pointee.inputs.deallocate()
    loader.pointee.targets.deallocate()
    if loader.pointee.should_shuffle {
        loader.pointee.shard_indices.deallocate()
        loader.pointee.intra_shard_indices?.deallocate()
    }
    try? loader.pointee.tokens_file!.close()
}

// ----------------------------------------------------------------------------
// Distributed Eval Loader
// Swift port saved for later...
