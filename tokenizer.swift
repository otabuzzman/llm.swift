//
//  tokenizer.swift
//  llm.swift
//
//  Created by Jürgen Schuck on 10.05.24.
//

import Foundation
import System

/*
 Defines the GPT-2 Tokenizer.
 Only supports decoding, i.e.: tokens (integers) -> strings
 This is all we need for unconditional generation.
 If we wanted to later prompt the model, we'd have to add decoding.
 Which could be tricky in C because of the regex involved, to look into later.
 */

// ----------------------------------------------------------------------------

struct Tokenizer {
    var vocab_size = 0
    var token_table = UnsafeMutableBufferPointer<UnsafeMutableBufferPointer<UInt8>>.allocate(capacity: 0)
    var init_ok = 0
    var eot_token = 0 // <|endoftext|> token id
}

func safe_printf(_ piece: UnsafeMutablePointer<UInt8>?) -> Void {
    // the tokens are raw bytes, and we we only want to print the printable ones
    // many bytes can be various control codes, backspace, etc.
    guard let piece = piece else { return }
    if piece[0] == 0 { return }
    // handle individual byte tokens
    // every token is asserted to be at least one byte so doing piece[1] is ok
    if piece[1] == 0 {
        let byte_val = piece[0]
        if !(isprint(byte_val) || isspace(byte_val)) {
            return // weird byte, don't print it
        }
    }
    print("\(String(cString: piece))", terminator: "")
}

func isprint(_ byte: UInt8) -> Bool {
    if byte > 0x20 && byte != 0x7F {
        return true
    }
    return false
}

func isspace(_ byte: UInt8) -> Bool {
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

func tokenizer_init(_ tokenizer: UnsafeMutablePointer<Tokenizer>, _ filename: URL) -> Void {
    guard
        let file = try? FileHandle(forReadingFrom: filename)
    else {
        tokenizer.pointee.init_ok = 0
        fatalError("Error opening tokens file (try `python train_gpt2.py`)")
    }
    // read in the header
    guard
        let header_data = try? file.read(upToCount: 256 * MemoryLayout<Int32>.size)
    else { fatalError("Error reading header from tokens file") }
    let header = header_data.withUnsafeBytes { (header_data: UnsafeRawBufferPointer) -> [Int] in
        header_data.bindMemory(to: Int32.self).map { Int($0) }
    }
    assert(header[0] == 20240328, "Bad magic tokens file")
    let version = header[1]
    tokenizer.pointee.vocab_size = header[2]
    let vocab_size = tokenizer.pointee.vocab_size // for brevity
    if version == 1 {
        // version 1 didn't include the EOT token id
        // so we assume it is 50256, the EOT in GPT-2
        assert(vocab_size == 50257, "Wrong tokenizer vocabulary size") // let's be defensive here
        tokenizer.pointee.eot_token = 50256;
    } else if version == 2 {
        tokenizer.pointee.eot_token = header[3]
    } else {
        fatalError("Wrong version \(version) of tokenizer file \(filename)")
    }
    // read in all the tokens
    tokenizer.pointee.token_table = UnsafeMutableBufferPointer<UnsafeMutableBufferPointer<UInt8>>.allocate(capacity: vocab_size)
    for i in 0..<vocab_size {
        guard
            let length_data = try? file.read(upToCount: 1 * MemoryLayout<UInt8>.size)
        else { fatalError("Error reading token length from tokens file") }
        let length = Int(length_data.withUnsafeBytes { $0.bindMemory(to: UInt8.self)[0] })
        assert(length > 0, "Every token should be at least one character")
        let token_bytes = UnsafeMutableBufferPointer<UInt8>.allocate(capacity: length + 1)
//        guard
//            let token_bytes_data = try? file.read(upToCount: length * MemoryLayout<UInt8>.size)
//        else { fatalError("Error reading token bytes from tokens file") }
//        _ = token_bytes_data.withUnsafeBytes { $0.copyBytes(to: token_bytes) }
        do {
            let fd = file.fileDescriptor
            _ = try FileDescriptor(rawValue: fd).read(into: UnsafeMutableRawBufferPointer(start: token_bytes.baseAddress!, count: length))
        } catch { fatalError("Error reading token bytes from tokens file") }
        token_bytes[length] = 0 // add null terminator for printing
        tokenizer.pointee.token_table[i] = token_bytes
    }
    // cleanups
    try? file.close()
    tokenizer.pointee.init_ok = 1
}

func tokenizer_decode(_ tokenizer: UnsafePointer<Tokenizer>, _ token_id: Int) -> UnsafeMutablePointer<UInt8>? {
    if tokenizer.pointee.init_ok == 0 { return nil }
    if token_id < tokenizer.pointee.vocab_size {
        return tokenizer.pointee.token_table[token_id].baseAddress
    } else {
        print("Invalid token id \(token_id)")
        return nil
    }
}

func tokenizer_free(_ tokenizer: UnsafePointer<Tokenizer>) -> Void {
    let vocab_size = tokenizer.pointee.vocab_size // for brevity
    let token_table = tokenizer.pointee.token_table
    if tokenizer.pointee.init_ok == 1 {
        for i in 0..<vocab_size {
            token_table[i].deallocate()
        }
        token_table.deallocate()
    }
}
