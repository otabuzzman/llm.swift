// swiftlint:disable:next blanket_disable_command
// swiftlint:disable identifier_name

/*
 Defines the GPT-2 Tokenizer.
 Only supports decoding, i.e.: tokens (integers) -> strings
 This is all we need for unconditional generation.
 If we wanted to later prompt the model, we'd have to add decoding.
 Which could be tricky in C because of the regex involved, to look into later.
 */

import Foundation
import System

enum TokenizerError: Error {
    case corrupted
}

struct Tokenizer {
    var vocab_size = 0
    var token_table: UnsafeMutablePointer<UnsafeMutablePointer<UInt8>>!
    var init_ok = false
    var eot_token: Int32 = 0 // <|endoftext|> token id
}

func isprint(_ piece: UnsafeMutablePointer<UInt8>) -> Bool {
    // the tokens are raw bytes, and we we only want to print the printable ones
    // many bytes can be various control codes, backspace, etc.
    if piece[0] == 0 { return false }
    // handle individual byte tokens
    // every token is asserted to be at least one byte so doing piece[1] is ok
    if piece[1] == 0 {
        let byte_val = piece[0]
        if !(isprint(byte_val) || isspace(byte_val)) {
            return false // weird byte, don't print it
        }
    }
    return true
}

private func isprint(_ byte: UInt8) -> Bool {
    if byte > 0x20 && byte != 0x7F {
        return true
    }
    return false
}

private func isspace(_ byte: UInt8) -> Bool {
    if byte == 0x20
        || byte == 0x09
        || byte == 0x0A
        || byte == 0x0B
        || byte == 0x0C
        || byte == 0x0D {
        return true
    }
    return false
}

func tokenizer_init(_ tokenizer: UnsafeMutablePointer<Tokenizer>, _ handle: FileHandle) throws {
    // read in the header
    guard
        let header_data = try handle.read(upToCount: 256 * MemoryLayout<Int32>.size)
    else { throw LlmSwiftError.apiReturnedNil }
    let header = header_data.withUnsafeBytes { (header_data: UnsafeRawBufferPointer) -> [Int] in
        header_data.bindMemory(to: Int32.self).map { Int($0) }
    }
    assert(header[0] == 20240328, "Bad magic in tokenizer file")
    assert(header[1] ~= 1...2, "Wrong version in tokenizer file")
    let version = header[1]
    tokenizer.pointee.vocab_size = header[2]
    let vocab_size = tokenizer.pointee.vocab_size // for brevity
    if version == 1 {
        // version 1 didn't include the EOT token id
        // so we assume it is 50256, the EOT in GPT-2
        if vocab_size != 50257 { throw TokenizerError.corrupted } // let's be defensive here
        tokenizer.pointee.eot_token = 50256
    } else { // version == 2
        tokenizer.pointee.eot_token = Int32(header[3])
    }
    // read in all the tokens
    tokenizer.pointee.token_table = UnsafeMutablePointer<UnsafeMutablePointer<UInt8>>.allocate(capacity: vocab_size)
    for i in 0..<vocab_size {
        guard
            let length_data = try handle.read(upToCount: 1 * MemoryLayout<UInt8>.size)
        else { throw LlmSwiftError.apiReturnedNil }
        let length = Int(length_data.withUnsafeBytes { $0.bindMemory(to: UInt8.self)[0] })
        if length == 0 { throw TokenizerError.corrupted }
        let token_bytes = UnsafeMutablePointer<UInt8>.allocate(capacity: length + 1)
        let token_bytes_buffer = UnsafeMutableRawBufferPointer(start: token_bytes, count: length)
        _ = try FileDescriptor(rawValue: handle.fileDescriptor).read(into: token_bytes_buffer)
        token_bytes[length] = 0 // add null terminator for printing
        tokenizer.pointee.token_table[i] = token_bytes
    }
    // cleanups
    try? handle.close()
    tokenizer.pointee.init_ok = true
}

func tokenizer_decode(_ tokenizer: UnsafePointer<Tokenizer>, _ token_id: Int) -> UnsafeMutablePointer<UInt8>? {
    if !tokenizer.pointee.init_ok { return nil }
    if token_id < tokenizer.pointee.vocab_size {
        return tokenizer.pointee.token_table[token_id]
    } else {
        return nil
    }
}

func tokenizer_free(_ tokenizer: UnsafePointer<Tokenizer>) {
    let vocab_size = tokenizer.pointee.vocab_size // for brevity
    let token_table = tokenizer.pointee.token_table
    if tokenizer.pointee.init_ok {
        for i in 0..<vocab_size {
            token_table![i].deallocate()
        }
        token_table!.deallocate()
    }
}
