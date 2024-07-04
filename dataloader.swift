/*
Implements:
- DataLoader for model training. Reads and serves data shards.
- EvalLoader for multiple-choice evaluation datasets, e.g. HellaSwag.
*/

import Foundation
import Glob

// Distributed Data Loader
//#define HEADER_SIZE 256
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
    var buffer = UnsafeMutableBufferPointer<UInt16>.allocate(capacity: 0)
    // input tokens into transformer
    var inputs = UnsafeMutablePointer<Int32>.allocate(capacity: 0)
    // target tokens for the transformer
    var targets = UnsafeMutablePointer<Int32>.allocate(capacity: 0)
    // random shuffle related variables
    var shuffle_rng: mt19937_state!
    var should_shuffle = false
    var shard_indices = UnsafeMutablePointer<Int32>.allocate(capacity: 0)
    var intra_shard_indices: UnsafeMutablePointer<Int32>?
    // sizes in bytes
    var total_batch_size_bytes = 0  // total across all processes
    var local_batch_offset_bytes = 0  // inner-sample offset for this process
    var header_bytes = 0  // header size in bytes
    var file_size_bytes = 0
}

@discardableResult
fileprivate func dataloader_load_shard(_ loader: UnsafeMutablePointer<DataLoader>, _ shard_index: Int) -> Int {
    var file_index = shard_index
    if loader.pointee.should_shuffle {
        file_index = Int(loader.pointee.shard_indices[shard_index])
    }
    // use the first glob match as the filename for now
    let filename = loader.pointee.glob_result[file_index]
    // open the input file for reading. also only a single file can be opened at a time
    try? loader.pointee.tokens_file?.close()
    guard
        let tokens_file = try? FileHandle(forReadingFrom: URL(string: filename)!)
    else { fatalError("Error opening tokens file") }
    loader.pointee.tokens_file = tokens_file
    // validate the header
    guard
        let header_data = try? tokens_file.read(upToCount: HEADER_SIZE * MemoryLayout<Int32>.size)
    else { fatalError("Error reading header from model file") }
    let header = header_data.withUnsafeBytes { (header_data: UnsafeRawBufferPointer) -> [Int] in
        header_data.bindMemory(to: Int32.self).map { Int($0) }
    }
    if header[0] != 20240520 { fatalError("Bad magic in data file (data encoding may have changed, re-run data prepro or refer again to README)") }
    if header[1] != 1 { fatalError("Bad version in data file") }

    let ntok = header[2] // number of tokens in the file
    assert(ntok > 0, "No tokens in file") // we expect some tokens in the file. this should never trip, right?
    // determine the file size and make sure it is consistent with the number of tokens
    loader.pointee.file_size_bytes = Int((try? tokens_file.seekToEnd()) ?? 0)
    try? tokens_file.seek(toOffset: 0)
    // we expect ntok in the file to be consistent with filesize, assert that is the case
    let expected_file_size = HEADER_SIZE * MemoryLayout<Int32>.size + ntok * MemoryLayout<UInt16>.size
    if loader.pointee.file_size_bytes != expected_file_size {
        fatalError("File size not as expected")
    }
    // -1 uint16_t due to us taking B*T+1 tokens but moving by B*T tokens
    loader.pointee.shard_num_samples = (ntok * MemoryLayout<UInt16>.size - MemoryLayout<UInt16>.size) / loader.pointee.total_batch_size_bytes
    return ntok
}

fileprivate func prepare_intra_shard_indices(_ loader: UnsafeMutablePointer<DataLoader>) -> Void {
    // shuffle the examples inside the shards
    loader.pointee.intra_shard_indices?.deallocate()
    loader.pointee.intra_shard_indices = UnsafeMutablePointer<Int32>.allocate(capacity: loader.pointee.shard_num_samples)
    init_identity_permutation(loader.pointee.intra_shard_indices!, loader.pointee.shard_num_samples)
    random_permutation(loader.pointee.intra_shard_indices!, loader.pointee.shard_num_samples, &loader.pointee.shuffle_rng)
}

func dataloader_reset(_ loader: UnsafeMutablePointer<DataLoader>) -> Void {
    loader.pointee.current_shard_idx = 0
    loader.pointee.current_sample_idx = 0

    if loader.pointee.should_shuffle {  // shuffle the shards
        random_permutation(loader.pointee.shard_indices, loader.pointee.glob_result.count, &loader.pointee.shuffle_rng)
    }

    dataloader_load_shard(loader, loader.pointee.current_shard_idx)

    if loader.pointee.should_shuffle {
        prepare_intra_shard_indices(loader)
    }
}

fileprivate func dataloader_advance(_ loader: UnsafeMutablePointer<DataLoader>) -> Void {
    if loader.pointee.current_shard_idx == loader.pointee.glob_result.count - 1 {
        // if we are at the last shard, we reset the loader and start a new epoch
        dataloader_reset(loader)
        return
    }

    // advance the loader by loading the next data shard and resetting the position
    loader.pointee.current_shard_idx = (loader.pointee.current_shard_idx + 1) % loader.pointee.glob_result.count
    loader.pointee.current_sample_idx = 0
    dataloader_load_shard(loader, loader.pointee.current_shard_idx)

    if loader.pointee.should_shuffle {
        prepare_intra_shard_indices(loader)
    }
}

func dataloader_init(_ loader: UnsafeMutablePointer<DataLoader>,
                     _ filename_pattern: String,
                     _ B: Int,
                     _ T: Int,
                     _ process_rank: Int,
                     _ num_processes: Int,
                     _ should_shuffle: Bool) -> Void {
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
        fatalError("No files matching pattern \(filename_pattern)")
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
        let shard_ntok = dataloader_load_shard(loader, shard_index)
        // we need at least one batch/shard, the way things are written right now.
        // can be relaxed a lot later.
        assert(shard_ntok >= num_processes * B * T + 1, "At least one batch per shard needed")
        ntok_total += shard_ntok
    }
    // debugging prints
    // print("DataLoader: filename_pattern \(filename_pattern)")
    // print("DataLoader: Found \(ntok_total) tokens across \(loader.pointee.glob_result.count) shards")

    // allocate all the space we'll need
    loader.pointee.buffer = UnsafeMutableBufferPointer<UInt16>.allocate(capacity: B * T + 1)
    loader.pointee.inputs = UnsafeMutablePointer<Int32>.allocate(capacity: B * T)
    loader.pointee.targets = UnsafeMutablePointer<Int32>.allocate(capacity: B * T)
    loader.pointee.num_tokens = ntok_total

    // reset the loader, to initialize it
    dataloader_reset(loader)
}

func dataloader_load_batch(_ loader: UnsafeMutablePointer<DataLoader>) -> Void {
    assert(!loader.pointee.should_shuffle || (loader.pointee.should_shuffle && loader.pointee.intra_shard_indices != nil), "No indices to shuffle")
    assert(loader.pointee.current_sample_idx < loader.pointee.shard_num_samples, "Sample index out of bounds")
    let idx = loader.pointee.should_shuffle ? Int(loader.pointee.intra_shard_indices![loader.pointee.current_sample_idx]) : loader.pointee.current_sample_idx
    let global_batch_offset_bytes = idx * loader.pointee.total_batch_size_bytes
    let current_offset = loader.pointee.header_bytes + global_batch_offset_bytes + loader.pointee.local_batch_offset_bytes

    let B = loader.pointee.B
    let T = loader.pointee.T
    // read B*T+1 uint16_t tokens from the file into buffer
    try? loader.pointee.tokens_file!.seek(toOffset: UInt64(current_offset))
    guard
        let file_data = try? loader.pointee.tokens_file!.read(upToCount: (B * T + 1) * MemoryLayout<UInt16>.size)
    else { fatalError("Error reading tokens file") }
    var token_data = file_data
    loader.pointee.buffer = token_data.withUnsafeMutableBytes { $0.bindMemory(to: UInt16.self) }
    // decode the buffer into inputs and targets (cast to int)
    for i in 0..<B * T {
        loader.pointee.inputs[i] = Int32(loader.pointee.buffer[i])
        loader.pointee.targets[i] = Int32(loader.pointee.buffer[i+1])
    }
}

func dataloader_next_batch(_ loader: UnsafeMutablePointer<DataLoader>) -> Void {
    // if the next batch would go past the end of the file, advance the loader
    if loader.pointee.current_sample_idx >= loader.pointee.shard_num_samples {
        dataloader_advance(loader)
    }
    dataloader_load_batch(loader)
    loader.pointee.current_sample_idx += 1
}

func dataloader_resume(_ loader: UnsafeMutablePointer<DataLoader>, _ current_shard_idx: Int, _ current_sample_idx: Int) -> Void {
    // used during model resumption (-y 1) flag
    loader.pointee.current_shard_idx = current_shard_idx
    loader.pointee.current_sample_idx = current_sample_idx
    dataloader_load_shard(loader, loader.pointee.current_shard_idx)
}

func dataloader_free(_ loader: UnsafeMutablePointer<DataLoader>) -> Void {
    loader.pointee.buffer.deallocate()
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
