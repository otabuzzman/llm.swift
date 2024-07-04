/*
Mersenne Twisters implementation, numerically identical to torch.

Example usage:

    var state = mt19937_state()
    manual_seed(&state, 137)
    print("\(randint32(&state))")
    print("\(randint32(&state))")
    print("\(randint32(&state))")
    print("\(randint32(&state))")
    print("\(randint32(&state))")

    var a8 = Array<Float>(repeating: 0, count 8)
    a8.withUnsafeBufferPointer { t8 in
        normal(t8, 8, 0, 1, &state)
        for i in 0..<8 {
            print("\(t8[i])")
        }
        print("\(randint32(&state))")
    }

    var a16 = Array<Float>(repeating: 0, count 16)
    a16.withUnsafeBufferPointer { t16 in
        normal(t16, 16, 0, 1, &state)
        for i in 0..<16 {
            print("\(t16[i])")
        }
        print("\(randint32(&state))")
    }

PyTorch reference (producing identical results):

    import torch
    torch.manual_seed(137)
    print(torch.randint(0, 0xFFFFFFFF, [1]).item())
    print(torch.randint(0, 0xFFFFFFFF, [1]).item())
    print(torch.randint(0, 0xFFFFFFFF, [1]).item())
    print(torch.randint(0, 0xFFFFFFFF, [1]).item())
    print(torch.randint(0, 0xFFFFFFFF, [1]).item())
    t = torch.zeros(8);
    t.normal_()
    for i in range(len(t)) :
        print(t[i].item())
    print(torch.randint(0, 0xFFFFFFFF, [1]).item())
    t = torch.zeros(16);
    t.normal_()
    for i in range(len(t)) :
        print(t[i].item())
    print(torch.randint(0, 0xFFFFFFFF, [1]).item())

Both output:

    4053805790
    2173880614
    380293709
    1237255315
    2986595568
    0.7947664260864258
    1.4369317293167114
    - 0.2292192131280899
    0.47556325793266296
    - 0.6334410905838013
    - 0.5791953802108765
    - 0.0925704762339592
    - 0.8659197092056274
    2186503452
    - 1.2813878059387207
    - 2.646395683288574
    - 0.06569503247737885
    0.2180829495191574
    - 0.46536165475845337
    - 0.33108410239219666
    2.5485482215881348
    0.10425379872322083
    0.8460659980773926
    0.9462448358535767
    - 0.2913765013217926
    0.34313806891441345
    - 1.1186704635620117
    - 0.18305328488349915
    - 2.3153159618377686
    0.3961987793445587
    2756748748
*/

import Accelerate

// #define MERSENNE_STATE_M 397u
// #define MERSENNE_STATE_N 624u
let MERSENNE_STATE_M = 397
let MERSENNE_STATE_N = 624

// #define LMASK 0x7ffffffful
// #define UMASK 0x80000000ul
let LMASK = 0x7fffffff
let UMASK = 0x80000000

// Copyright(c) Makoto Matsumoto and Takuji Nishimura

// This implementation follows PyTorch so that we are numerically identical when running verification tests.

struct mt19937_state {
    /* private */ var seed = 0
    /* private */ var left = 0
    /* private */ var next = 0
    /* private */ var state = Array<Int>(repeating: 0, count: MERSENNE_STATE_N)
    var MATRIX_A = Array<Int>(repeating: 0, count: 2)
}

func manual_seed(_ state: UnsafeMutablePointer<mt19937_state>, _ seed: Int) -> Void {
    state.pointee.MATRIX_A[0] = 0x0
    state.pointee.MATRIX_A[1] = 0x9908b0d
    state.pointee.state[0] = seed & 0xffffffff
    for j in 1..<MERSENNE_STATE_N {
        state.pointee.state[j] = 1812433253 * (state.pointee.state[j - 1] ^ (state.pointee.state[j - 1] >> 30)) + j
        state.pointee.state[j] &= 0xffffffff
    }
    state.pointee.left = 1
    state.pointee.next = 0
}

func nextstate(_ state: UnsafeMutablePointer<mt19937_state>) -> Void {
    state.pointee.left = MERSENNE_STATE_N
    state.pointee.next = 0
    var y = 0, j = 0
    for i in 0..<MERSENNE_STATE_N - MERSENNE_STATE_M {
		j = i
        y = (state.pointee.state[j] & UMASK) | (state.pointee.state[j + 1] & LMASK)
        state.pointee.state[j] = state.pointee.state[j + MERSENNE_STATE_M] ^ (y >> 1) ^ state.pointee.MATRIX_A[y & 0x1]
    }
    for i in j..<MERSENNE_STATE_N - 1 {
        j = i
        y = (state.pointee.state[j] & UMASK) | (state.pointee.state[j + 1] & LMASK)
        state.pointee.state[j] = state.pointee.state[j + (MERSENNE_STATE_M - MERSENNE_STATE_N)] ^ (y >> 1) ^ state.pointee.MATRIX_A[y & 0x1]
    }
    y = (state.pointee.state[MERSENNE_STATE_N - 1] & UMASK) | (state.pointee.state[0] & LMASK)
    state.pointee.state[MERSENNE_STATE_N - 1] = state.pointee.state[MERSENNE_STATE_M - 1] ^ (y >> 1) ^ state.pointee.MATRIX_A[y & 0x1]
}

func randint32(_ state: UnsafeMutablePointer<mt19937_state>) -> UInt32 {
    if state.pointee.MATRIX_A[0] != 0 || state.pointee.MATRIX_A[1] != 0x9908b0df { manual_seed(state, 5489) } // auto-initialize
	state.pointee.left -= 1
    if state.pointee.left <= 0 {
        nextstate(state)
    }
    var y = state.pointee.state[state.pointee.next]
	state.pointee.next += 1
    y ^= y >> 11
    y ^= (y << 7) & 0x9d2c5680
    y ^= (y << 15) & 0xefc60000
    y ^= y >> 18
    return UInt32(y)
}

@inline(__always) // https://forums.swift.org/t/when-should-both-inlinable-and-inline-always-be-used/37375/2
func randint64(_ state: UnsafeMutablePointer<mt19937_state>) -> UInt64 {
    (UInt64(randint32(state)) << 32) | UInt64(randint32(state))
}

@inline(__always)
func randfloat32(_ state: UnsafeMutablePointer<mt19937_state>) -> Float {
    Float(randint32(state) & ((1 << 24) - 1)) * (1.0 / Float(1 << 24))
}

@inline(__always)
func randfloat64(_ state: UnsafeMutablePointer<mt19937_state>) -> Double {
    Double(randint64(state) & ((1 << 53) - 1)) * (1.0 / Double(1 << 53))
}

fileprivate func uniform(_ data: UnsafeMutablePointer<Float>, _ numel: Int, _ from: Float, _ to: Float, _ state: UnsafeMutablePointer<mt19937_state>) -> Void {
    for t in 0..<numel {
        data[t] = randfloat32(state) * (to - from) + from
    }
}

// Box-Muller transform: maps uniform random numbers to Gaussian distributed numbers
// https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
func normal_fill_16(_ data: UnsafeMutablePointer<Float>, _ mean: Float, _ std: Float) -> Void {
    // #define EPSILONE 1e-12f
    let EPSILONE: Float = 1e-12
    for t in 0..<8 {
        let u1: Float = 1 - data[t]
        let u2: Float = data[t + 8]
        let radius: Float = sqrt(-2 * log(u1 + EPSILONE))
        let theta: Float = 2 * Float.pi * u2
        data[t] = radius * cos(theta) * std + mean
        data[t + 8] = radius * sin(theta) * std + mean
    }
}

func normal_fill(_ data: UnsafeMutableBufferPointer<Float>, _ numel: Int, _ mean: Float, _ std: Float, _ state: UnsafeMutablePointer<mt19937_state>) -> Void {
    for t in 0..<numel {
        data[t] = Float(randfloat32(state))
    }
	var i = 0
	while i < numel - 15 {
        let data = (data.baseAddress?.advanced(by: i))!
        normal_fill_16(data, mean, std)
		i += 16
	}
    if numel % 16 != 0 {
        // recompute the last 16 values
        let data = (data.baseAddress?.advanced(by: numel - 16))!
        for i in 0..<16 {
            data[i] = Float(randfloat32(state))
        }
        normal_fill_16(data, mean, std)
    }
}

fileprivate func normal(_ data: UnsafeMutableBufferPointer<Float>, _ numel: Int, _ mean: Float, _ std: Float, _ state: UnsafeMutablePointer<mt19937_state>) -> Void {
    // #define EPSILONE 1e-12f
    let EPSILONE: Float = 1e-12
    if numel >= 16 {
        normal_fill(data, numel, mean, std, state)
    } else {
        var nextdouble_normal_sample: Double = 0 // make compiler warning happy, won't be used
        var has_nextdouble_normal_sample = false
        for t in 0..<numel {
            if has_nextdouble_normal_sample {
                data[t] = Float(nextdouble_normal_sample) * std + mean
                has_nextdouble_normal_sample = false
                continue
            }
            // for numel < 16 we draw a double (float64)
            let u1 = Float(randfloat64(state))
            let u2 = Float(randfloat64(state))
            let radius = sqrtf(-2 * logf(1 - u2 + EPSILONE))
            let theta = 2 * Float.pi * u1
            nextdouble_normal_sample = Double(radius * sinf(theta))
            has_nextdouble_normal_sample = true
            data[t] = radius * cosf(theta) * std + mean
        }
    }
}

func init_identity_permutation(_ data: UnsafeMutablePointer<Int32>, _ numel: Int) -> Void {
    for i in 0..<numel {
        data[i] = Int32(i)
    }
}

func random_permutation(_ data: UnsafeMutablePointer<Int32>, _ numel: Int, _ state: UnsafeMutablePointer<mt19937_state>) -> Void {
    for i in (0..<numel).reversed() {
        // pick an index j in [0, i] with equal probability
        let j = Int(randint32(state) % UInt32(i + 1))
        // swap i <-> j
        let tmp = data[i]
        data[i] = data[j]
        data[j] = tmp
    }
}

func test_mt19937() -> Void {
    var state = mt19937_state()
    manual_seed(&state, 137)
    print("\(randint32(&state))")
    print("\(randint32(&state))")
    print("\(randint32(&state))")
    print("\(randint32(&state))")
    print("\(randint32(&state))")

    var a8 = Array<Float>(repeating: 0, count: 8)
    a8.withUnsafeMutableBufferPointer { t8 in
        normal(t8, 8, 0, 1, &state)
        for i in 0..<8 {
            print("\(t8[i])")
        }
        print("\(randint32(&state))")
    }

    var a16 = Array<Float>(repeating: 0, count: 16)
    a16.withUnsafeMutableBufferPointer { t16 in
        normal(t16, 16, 0, 1, &state)
        for i in 0..<16 {
            print("\(t16[i])")
        }
        print("\(randint32(&state))");
    }
}
