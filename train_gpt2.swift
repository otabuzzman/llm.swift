// swiftlint:disable:next blanket_disable_command
// swiftlint:disable identifier_name

import Foundation
import System

enum LlmSwiftError: Error {
    case wrongApiUsage
    case apiReturnedNil
    case outOfBounds
}

// ----------------------------------------------------------------------------
// all the individual layers' forward and backward passes
// B = batch_size, T = sequence_length, C = channels, V = vocab_size

// swiftlint:disable:next function_parameter_count
func encoder_forward(
    _ out: UnsafeMutablePointer<Float>,
    _ inp: UnsafePointer<Int32>,
    _ wte: UnsafePointer<Float>,
    _ wpe: UnsafePointer<Float>,
    _ B: Int, _ T: Int, _ C: Int) {
    // out is (B,T,C). At each position (b,t), a C-dimensional vector summarizing token & position
    // inp is (B,T) of integers, holding the token ids at each (b,t) position
    // wte is (V,C) of token embeddings, short for "weight token embeddings"
    // wpe is (maxT,C) of position embeddings, short for "weight positional embedding"
    for b in 0..<B {
        for t in 0..<T {
            // seek to the output position in out[b,t,:]
            let out_bt = out + b * T * C + t * C
            // get the index of the token at inp[b, t]
            let ix = Int(inp[b * T + t])
            // seek to the position in wte corresponding to the token
            let wte_ix = wte + ix * C
            // seek to the position in wpe corresponding to the position
            let wpe_t = wpe + t * C
            // add the two vectors and store the result in out[b,t,:]
            for i in 0..<C {
                out_bt[i] = wte_ix[i] + wpe_t[i]
            }
        }
    }
}

// swiftlint:disable:next function_parameter_count
func encoder_backward(
    _ dwte: UnsafeMutablePointer<Float>,
    _ dwpe: UnsafeMutablePointer<Float>,
    _ dout: UnsafePointer<Float>,
    _ inp: UnsafePointer<Int32>,
    _ B: Int, _ T: Int, _ C: Int) {
    for b in 0..<B {
        for t in 0..<T {
            let dout_bt = dout + b * T * C + t * C
            let ix = Int(inp[b * T + t])
            let dwte_ix = dwte + ix * C
            let dwpe_t = dwpe + t * C
            for i in 0..<C {
                let d = dout_bt[i]
                dwte_ix[i] += d
                dwpe_t[i] += d
            }
        }
    }
}

// swiftlint:disable:next function_parameter_count
func layernorm_forward(
    _ out: UnsafeMutablePointer<Float>,
    _ mean: UnsafeMutablePointer<Float>,
    _ rstd: UnsafeMutablePointer<Float>,
    _ inp: UnsafePointer<Float>,
    _ weight: UnsafePointer<Float>,
    _ bias: UnsafePointer<Float>, _ B: Int, _ T: Int, _ C: Int) {
    // reference: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
    // both inp and out are (B,T,C) of the activations
    // mean and rstd are (B,T) buffers, to be used later in backward pass
    // at each position (b,t) of the input, the C-dimensional vector
    // of activations gets normalized, then scaled and shifted
    let eps: Float = 1e-5
    let fC = Float(C)
    for b in 0..<B {
        for t in 0..<T {
            // seek to the input position inp[b,t,:]
            let x = inp + b * T * C + t * C
            // calculate the mean
            var m: Float = 0
            for i in 0..<C {
                m += x[i]
            }
            m /= fC
            // calculate the variance (without any bias correction)
            var v: Float = 0
            for i in 0..<C {
                let xshift = x[i] - m
                v += xshift * xshift
            }
            v /= fC
            // calculate the rstd (reciprocal standard deviation)
            let s: Float = 1 / sqrtf(v + eps)
            // seek to the output position in out[b,t,:]
            let out_bt = out + b * T * C + t * C
            for i in 0..<C {
                let n = (s * (x[i] - m)) // normalize
                let o = n * weight[i] + bias[i] // scale and shift
                out_bt[i] = o // write
            }
            // cache the mean and rstd for the backward pass later
            mean[b * T + t] = m
            rstd[b * T + t] = s
        }
    }
}

// swiftlint:disable:next function_parameter_count
func layernorm_backward(
    _ dinp: UnsafeMutablePointer<Float>,
    _ dweight: UnsafeMutablePointer<Float>,
    _ dbias: UnsafeMutablePointer<Float>,
    _ dout: UnsafePointer<Float>,
    _ inp: UnsafePointer<Float>,
    _ weight: UnsafePointer<Float>,
    _ mean: UnsafePointer<Float>,
    _ rstd: UnsafePointer<Float>,
    _ B: Int, _ T: Int, _ C: Int) {
    let fC = Float(C)
    for b in 0..<B {
        for t in 0..<T {
            let dout_bt = dout + b * T * C + t * C
            let inp_bt = inp + b * T * C + t * C
            let dinp_bt = dinp + b * T * C + t * C
            let mean_bt = mean[b * T + t]
            let rstd_bt = rstd[b * T + t]

            // first: two reduce operations
            var dnorm_mean: Float = 0
            var dnorm_norm_mean: Float = 0
            for i in 0..<C {
                let norm_bti = (inp_bt[i] - mean_bt) * rstd_bt
                let dnorm_i = weight[i] * dout_bt[i]
                dnorm_mean += dnorm_i
                dnorm_norm_mean += dnorm_i * norm_bti
            }
            dnorm_mean /= fC
            dnorm_norm_mean /= fC

            // now iterate again and accumulate all the gradients
            for i in 0..<C {
                let norm_bti = (inp_bt[i] - mean_bt) * rstd_bt
                let dnorm_i = weight[i] * dout_bt[i]
                // gradient contribution to bias
                dbias[i] += dout_bt[i]
                // gradient contribution to weight
                dweight[i] += norm_bti * dout_bt[i]
                // gradient contribution to input
                var dval: Float = 0
                dval += dnorm_i // term 1
                dval -= dnorm_mean // term 2
                dval -= norm_bti * dnorm_norm_mean // term 3
                dval *= rstd_bt // final scale
                dinp_bt[i] += dval
            }
        }
    }
}

// swiftlint:disable:next function_parameter_count
func matmul_forward_naive(
    _ out: UnsafeMutablePointer<Float>,
    _ inp: UnsafePointer<Float>,
    _ weight: UnsafePointer<Float>,
    _ bias: UnsafePointer<Float>?,
    _ B: Int, _ T: Int, _ C: Int, _ OC: Int) async {
    // the most naive implementation of matrix multiplication
    // this serves as an algorithmic reference, and as a fallback for
    // unfriendly input shapes inside matmul_forward(), below.
    // #pragma omp parallel for collapse(2)
    DispatchQueue.global(qos: .userInteractive).sync {
        DispatchQueue.concurrentPerform(iterations: B * T) { bt in
            for o in 0..<OC {
                var val = bias?[o] ?? 0
                for i in 0..<C {
                    val += inp[bt * C + i] * weight[o * C + i]
                }
                out[bt * OC + o] = val
            }
        }
    }
//    await withTaskGroup(of: Void.self) {
//        for b in 0..<B {
//            for t in 0..<T {
//                $0.addTask {
//                    let bt = b * T + t
//                    ...
//                }
//            }
//        }
//    }
}

// swiftlint:disable:next function_parameter_count
func matmul_forward(
    _ out: UnsafeMutablePointer<Float>,
    _ inp: UnsafePointer<Float>,
    _ weight: UnsafePointer<Float>,
    _ bias: UnsafePointer<Float>?,
    _ B: Int, _ T: Int, _ C: Int, _ OC: Int) async {
    // most of the running time is spent here and in matmul_backward
    // therefore, the implementation below is very mildly optimized
    // this function is otherwise identical to that of matmul_forward_naive()
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)

//    let t0 = Date.timeIntervalSinceReferenceDate
    // make sure the tiled loop will be correct or fallback to naive version
    let LOOP_UNROLL = 8
    if (B * T) % LOOP_UNROLL != 0 {
        await matmul_forward_naive(out, inp, weight, bias, B, T, C, OC)
        return
    }

    // collapse the B and T loops into one and turn it into a strided loop.
    // then we can tile the inner loop, and reuse the loaded weight LOOP_UNROLL many times
    // #pragma omp parallel for
    DispatchQueue.global(qos: .userInteractive).sync {
        DispatchQueue.concurrentPerform(iterations: B * T / LOOP_UNROLL) {
            let obt = $0 * LOOP_UNROLL
            for o in 0..<OC {
                // we'll keep LOOP_UNROLL many results in registers
                var result = [Float](repeating: 0, count: LOOP_UNROLL)
                // initialize the bias, if it exists
                for ibt in 0..<LOOP_UNROLL {
                    result[ibt] = bias?[o] ?? 0
                }
                // inner loops. Because we do LOOP_UNROLL steps of inner bt, we can cache
                // the value of weight[i + o * C] and reuse it.
                // we compile with -Ofast, so the compiler will turn the inner loop into FMAs
                for i in 0..<C {
                    let w = weight[i + o * C]
                    for ibt in 0..<LOOP_UNROLL {
                        let bt = obt + ibt
                        result[ibt] += inp[bt * C + i] * w
                    }
                }
                // write back results to main memory
                for ibt in 0..<LOOP_UNROLL {
                    let bt = obt + ibt
                    out[bt * OC + o] = result[ibt]
                }
            }
        }
    }
//    let t1 = Date.timeIntervalSinceReferenceDate
//    print("matmul took \((t1 - t0) * 1000) ms")
}

// swiftlint:disable:next function_parameter_count
func matmul_backward(
    _ dinp: UnsafeMutablePointer<Float>,
    _ dweight: UnsafeMutablePointer<Float>,
    _ dbias: UnsafeMutablePointer<Float>?,
    _ dout: UnsafePointer<Float>,
    _ inp: UnsafePointer<Float>,
    _ weight: UnsafePointer<Float>,
    _ B: Int, _ T: Int, _ C: Int, _ OC: Int) async {
    // most of the running time is spent here and in matmul_forward
    // this backward could be done in a single "round" of loops
    // but that doesn't afford an efficient parallelization strategy

    // backward into inp first, parallelize over B,T
    // #pragma omp parallel for collapse(2)
    DispatchQueue.global(qos: .userInteractive).sync {
        DispatchQueue.concurrentPerform(iterations: B * T) {
            let (t, b, _) = indicesOf(combined: $0, T, B)

            let dout_bt = dout + b * T * OC + t * OC
            let dinp_bt = dinp + b * T * C + t * C
            for o in 0..<OC {
                let wrow = weight + o * C
                let d = dout_bt[o]
                for i in 0..<C {
                    dinp_bt[i] += wrow[i] * d
                }
            }
        }
    }
//    await withTaskGroup(of: Void.self) {
//        for b in 0..<B {
//            for t in 0..<T {
//                $0.addTask {
//                    let dout_bt = dout + b * T * OC + t * OC
//                    ...
//                }
//            }
//        }
//    }

    // backward into weight/bias, parallelize over output channels OC
    // #pragma omp parallel for
    DispatchQueue.global(qos: .userInteractive).sync {
        DispatchQueue.concurrentPerform(iterations: OC) { o in
            for b in 0..<B {
                for t in 0..<T {
                    let dout_bt = dout + b * T * OC + t * OC
                    let inp_bt = inp + b * T * C + t * C
                    let dwrow = dweight + o * C
                    let d = dout_bt[o]
                    if let dbias = dbias { dbias[o] += d }
                    for i in 0..<C {
                        dwrow[i] += inp_bt[i] * d
                    }
                }
            }
        }
    }
//    await withTaskGroup(of: Void.self) {
//        for o in 0..<OC {
//            $0.addTask {
//                for b in 0..<B {
//                ...
//            }
//        }
//    }
}

// swiftlint:disable:next function_parameter_count
func attention_forward(
    _ out: UnsafeMutablePointer<Float>,
    _ preatt: UnsafeMutablePointer<Float>,
    _ att: UnsafeMutablePointer<Float>,
    _ inp: UnsafePointer<Float>,
    _ B: Int, _ T: Int, _ C: Int, _ NH: Int) async {
    // input is (B, T, 3C) holding the query, key, value (Q, K, V) vectors
    // preatt, att are (B, NH, T, T). NH = number of heads, T = sequence length
    // that holds the pre-attention and post-attention scores (used in backward)
    // output is (B, T, C)
    // attention is the only layer that mixes information across time
    // every other operation is applied at every (b,t) position independently
    // (and of course, no layer mixes information across batch)
    let C3 = C * 3
    let hs = C / NH // head size
    let scale = 1 / sqrtf(Float(hs))

    // #pragma omp parallel for collapse(3)
    DispatchQueue.global(qos: .userInteractive).sync {
        DispatchQueue.concurrentPerform(iterations: B * T * NH) {
            let (h, t, b) = indicesOf(combined: $0, NH, T, B)

            let query_t = inp + b * T * C3 + t * C3 + h * hs
            let preatt_bth = preatt + b * NH * T * T + h * T * T + t * T
            let att_bth = att + b * NH * T * T + h * T * T + t * T

            // pass 1: calculate query dot key and maxval
            var maxval: Float = -10000 // TODO something better // swiftlint:disable:this todo
            for t2 in 0...t {
                let key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C // +C because it's key

                // (query_t) dot (key_t2)
                var val: Float = 0
                for i in 0..<hs {
                    val += query_t[i] * key_t2[i]
                }
                val *= scale
                if val > maxval {
                    maxval = val
                }

                preatt_bth[t2] = val
            }

            // pass 2: calculate the exp and keep track of sum
            // maxval is being calculated and subtracted only for numerical stability
            var expsum: Float = 0
            for t2 in 0...t {
                let expv = expf(preatt_bth[t2] - maxval)
                expsum += expv
                att_bth[t2] = expv
            }
            let expsum_inv = expsum == 0 ? 0 : 1 / expsum

            // pass 3: normalize to get the softmax
            for t2 in 0..<T {
                if t2 <= t {
                    att_bth[t2] *= expsum_inv
                } else {
                    // causal attention mask. not strictly necessary to set to zero here
                    // only doing this explicitly for debugging and checking to PyTorch
                    att_bth[t2] = 0
                }
            }

            // pass 4: accumulate weighted values into the output of attention
            let out_bth = out + b * T * C + t * C + h * hs
            for i in 0..<hs { out_bth[i] = 0 }
            for t2 in 0...t {
                let value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C * 2 // +C*2 because it's value
                let att_btht2 = att_bth[t2]
                for i in 0..<hs {
                    out_bth[i] += att_btht2 * value_t2[i]
                }
            }
        }
    }
//    await withTaskGroup(of: Void.self) {
//        for b in 0..<B {
//            for t in 0..<T {
//                for h in 0..<NH {
//                    $0.addTask {
//                        let query_t = inp + b * T * C3 + t * C3 + h * hs
//                        ...
//                    }
//                }
//            }
//        }
//    }
}

// swiftlint:disable:next function_parameter_count
func attention_backward(
    _ dinp: UnsafeMutablePointer<Float>,
    _ dpreatt: UnsafeMutablePointer<Float>,
    _ datt: UnsafeMutablePointer<Float>,
    _ dout: UnsafePointer<Float>,
    _ inp: UnsafePointer<Float>,
    _ att: UnsafePointer<Float>,
    _ B: Int, _ T: Int, _ C: Int, _ NH: Int) {
    // inp/dinp are (B, T, 3C) Q,K,V
    // att/datt/dpreatt are (B, NH, T, T)
    // dout is (B, T, C)
    let C3 = C * 3
    let hs = C / NH // head size
    let scale = 1 / sqrtf(Float(hs))

    for b in 0..<B {
        for t in 0..<T {
            for h in 0..<NH {
                let att_bth = att + b * NH * T * T + h * T * T + t * T
                let datt_bth = datt + b * NH * T * T + h * T * T + t * T
                let dpreatt_bth = dpreatt + b * NH * T * T + h * T * T + t * T
                let dquery_t = dinp + b * T * C3 + t * C3 + h * hs
                let query_t = inp + b * T * C3 + t * C3 + h * hs

                // backward pass 4, through the value accumulation
                let dout_bth = dout + b * T * C + t * C + h * hs
                for t2 in 0...t {
                    let value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C*2 // +C*2 because it's value
                    let dvalue_t2 = dinp + b * T * C3 + t2 * C3 + h * hs + C*2
                    for i in 0..<hs {
                        // in the forward pass this was:
                        // out_bth[i] += att_bth[t2] * value_t2[i]
                        // so now we have:
                        datt_bth[t2] += value_t2[i] * dout_bth[i]
                        dvalue_t2[i] += att_bth[t2] * dout_bth[i]
                    }
                }

                // backward pass 2 & 3, the softmax
                // note that softmax (like e.g. tanh) doesn't need the input (preatt) to backward
                for t2 in 0...t {
                    for t3 in 0...t {
                        let indicator: Float = t2 == t3 ? 1 : 0
                        let local_derivative = att_bth[t2] * (indicator - att_bth[t3])
                        dpreatt_bth[t3] += local_derivative * datt_bth[t2]
                    }
                }

                // backward pass 1, the query @ key matmul
                for t2 in 0...t {
                    let key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C // +C because it's key
                    let dkey_t2 = dinp + b * T * C3 + t2 * C3 + h * hs + C // +C because it's key
                    for i in 0..<hs {
                        // in the forward pass this was:
                        // preatt_bth[t2] += (query_t[i] * key_t2[i]) * scale
                        // so now we have:
                        dquery_t[i] += key_t2[i] * dpreatt_bth[t2] * scale
                        dkey_t2[i] += query_t[i] * dpreatt_bth[t2] * scale
                    }
                }
            }
        }
    }
}

let GELU_SCALING_FACTOR = sqrtf(2 / Float.pi)
func gelu_forward(_ out: UnsafeMutablePointer<Float>, _ inp: UnsafePointer<Float>, _ N: Int) {
    // (approximate) GeLU elementwise non-linearity in the MLP block of Transformer
    for i in 0..<N {
        let x = inp[i]
        let cube = 0.044715 * x * x * x
        out[i] = 0.5 * x * (1 + tanhf(GELU_SCALING_FACTOR * (x + cube)))
    }
}

// we want to use -Ofast optimization, but sadly GeLU breaks, so disable this flag just for it (#168)
// #pragma float_control(precise, on, push)
// #if defined(__GNUC__) && !defined(__clang__)
// __attribute__((optimize("no-finite-math-only")))
// #endif
@_optimize(none)
func gelu_backward(
    _ dinp: UnsafeMutablePointer<Float>,
    _ inp: UnsafePointer<Float>,
    _ dout: UnsafePointer<Float>,
    _ N: Int) {
    for i in 0..<N {
        let x = inp[i]
        let cube = 0.044715 * x * x * x
        let tanh_arg = GELU_SCALING_FACTOR * (x + cube)
        let tanh_out = tanhf(tanh_arg)
        let coshf_out = coshf(tanh_arg)
        let sech_out = 1 / (coshf_out * coshf_out)
        let local_grad = 0.5 * (1 + tanh_out) + x * 0.5 * sech_out * GELU_SCALING_FACTOR * (1 + 3 * 0.044715 * x * x)
        dinp[i] += local_grad * dout[i]
    }
}
// #pragma float_control(pop)

func residual_forward(
    _ out: UnsafeMutablePointer<Float>,
    _ inp1: UnsafePointer<Float>,
    _ inp2: UnsafePointer<Float>,
    _ N: Int) {
    for i in 0..<N {
        out[i] = inp1[i] + inp2[i]
    }
}

func residual_backward(
    _ dinp1: UnsafeMutablePointer<Float>,
    _ dinp2: UnsafeMutablePointer<Float>,
    _ dout: UnsafePointer<Float>,
    _ N: Int) {
    for i in 0..<N {
        dinp1[i] += dout[i]
        dinp2[i] += dout[i]
    }
}

// swiftlint:disable:next function_parameter_count
func softmax_forward(
    _ probs: UnsafeMutablePointer<Float>,
    _ logits: UnsafeMutablePointer<Float>,
    _ B: Int, _ T: Int, _ V: Int, _ Vp: Int) async {
    // output: probs are (B,T,Vp) of the probabilities (sums to 1.0 in each b,t position)
    // input: logits is (B,T,Vp) of the unnormalized log probabilities
    // Vp is the padded vocab size (for efficiency), V is the "real" vocab size
    // example: Vp is 50304 and V is 50257
    // #pragma omp parallel for collapse(2)
    DispatchQueue.global(qos: .userInteractive).sync {
        DispatchQueue.concurrentPerform(iterations: B * T) {
            let (t, b, _) = indicesOf(combined: $0, T, B)

            // probs <- softmax(logits)
            let logits_bt = logits + b * T * Vp + t * Vp
            let probs_bt = probs + b * T * Vp + t * Vp

            // maxval is only calculated and subtracted for numerical stability
            var maxval: Float = -10000 // TODO something better // swiftlint:disable:this todo
            for i in 0..<V where logits_bt[i] > maxval {
                maxval = logits_bt[i]
            }
            var sum: Float = 0
            for i in 0..<V {
                probs_bt[i] = expf(logits_bt[i] - maxval)
                sum += probs_bt[i]
            }
            // note we only loop to V, leaving the padded dimensions
            for i in 0..<V {
                probs_bt[i] /= sum
            }
            // for extra super safety we may wish to include this too,
            // forcing the probabilities here to be zero, but it shouldn't matter
            for i in V..<Vp {
                probs_bt[i] = 0
            }
        }
    }
//    await withTaskGroup(of: Void.self) {
//        for b in 0..<B {
//            for t in 0..<T {
//                $0.addTask {
//                    // probs <- softmax(logits)
//                    let logits_bt = logits + b * T * Vp + t * Vp
//                    ...
//                }
//            }
//        }
//    }
}

// swiftlint:disable:next function_parameter_count
func crossentropy_forward(
    _ losses: UnsafeMutablePointer<Float>,
    _ probs: UnsafePointer<Float>,
    _ targets: UnsafePointer<Int32>,
    _ B: Int, _ T: Int, _ Vp: Int) {
    // output: losses is (B,T) of the individual losses at each position
    // input: probs are (B,T,Vp) of the probabilities
    // input: targets is (B,T) of integers giving the correct index in logits
    for b in 0..<B {
        for t in 0..<T {
            // loss = -log(probs[target])
            let probs_bt = probs + b * T * Vp + t * Vp
            let ix = Int(targets[b * T + t])
            losses[b * T + t] = -logf(probs_bt[ix])
        }
    }
}

// swiftlint:disable:next function_parameter_count
func crossentropy_softmax_backward(
    _ dlogits: UnsafeMutablePointer<Float>,
    _ dlosses: UnsafePointer<Float>,
    _ probs: UnsafePointer<Float>,
    _ targets: UnsafePointer<Int32>,
    _ B: Int, _ T: Int, _ V: Int, _ Vp: Int) {
    // backwards through both softmax and crossentropy
    for b in 0..<B {
        for t in 0..<T {
            let dlogits_bt = dlogits + b * T * Vp + t * Vp
            let probs_bt = probs + b * T * Vp + t * Vp
            let dloss = dlosses[b * T + t]
            let ix = Int(targets[b * T + t])
            // note we only loop to V, leaving the padded dimensions
            // of dlogits untouched, so gradient there stays at zero
            for i in 0..<V {
                let p = probs_bt[i]
                let indicator: Float = i == ix ? 1 : 0
                dlogits_bt[i] += (p - indicator) * dloss
            }
        }
    }
}

// ----------------------------------------------------------------------------
// GPT-2 model definition

struct GPT2Config {
    var max_seq_len = 0 // max sequence length, e.g. 1024
    var vocab_size = 0 // vocab size, e.g. 50257
    var padded_vocab_size = 0 // padded to e.g. %128==0, 50304
    var num_layers = 0 // number of layers, e.g. 12
    var num_heads = 0 // number of Int in attention, e.g. 12
    var channels = 0 // number of channels, e.g. 768
}

// the parameters of the model
let NUM_PARAMETER_TENSORS = 16
struct ParameterTensors {
    var wte: UnsafeMutablePointer<Float>! // (V, C)
    var wpe: UnsafeMutablePointer<Float>! // (maxT, C)
    var ln1w: UnsafeMutablePointer<Float>! // (L, C)
    var ln1b: UnsafeMutablePointer<Float>! // (L, C)
    var qkvw: UnsafeMutablePointer<Float>! // (L, 3*C, C)
    var qkvb: UnsafeMutablePointer<Float>! // (L, 3*C)
    var attprojw: UnsafeMutablePointer<Float>! // (L, C, C)
    var attprojb: UnsafeMutablePointer<Float>! // (L, C)
    var ln2w: UnsafeMutablePointer<Float>! // (L, C)
    var ln2b: UnsafeMutablePointer<Float>! // (L, C)
    var fcw: UnsafeMutablePointer<Float>! // (L, 4*C, C)
    var fcb: UnsafeMutablePointer<Float>! // (L, 4*C)
    var fcprojw: UnsafeMutablePointer<Float>! // (L, C, 4*C)
    var fcprojb: UnsafeMutablePointer<Float>! // (L, C)
    var lnfw: UnsafeMutablePointer<Float>! // (C)
    var lnfb: UnsafeMutablePointer<Float>! // (C)
}

func fill_in_parameter_sizes(_ param_sizes: UnsafeMutablePointer<Int>, _ config: GPT2Config) {
    let Vp = config.padded_vocab_size
    let C = config.channels
    let maxT = config.max_seq_len
    let L = config.num_layers
    param_sizes[0] = Vp * C // wte
    param_sizes[1] = maxT * C // wpe
    param_sizes[2] = L * C // ln1w
    param_sizes[3] = L * C // ln1b
    param_sizes[4] = L * (3 * C) * C // qkvw
    param_sizes[5] = L * (3 * C) // qkvb
    param_sizes[6] = L * C * C // attprojw
    param_sizes[7] = L * C // attprojb
    param_sizes[8] = L * C // ln2w
    param_sizes[9] = L * C // ln2b
    param_sizes[10] = L * (4 * C) * C // fcw
    param_sizes[11] = L * (4 * C) // fcb
    param_sizes[12] = L * C * (4 * C) // fcprojw
    param_sizes[13] = L * C // fcprojb
    param_sizes[14] = C // lnfw
    param_sizes[15] = C // lnfb
}

// allocate memory for the parameters and point the individual tensors to the right places
func malloc_and_point_parameters(
    _ params: UnsafeMutablePointer<ParameterTensors>,
    _ param_sizes: UnsafePointer<Int>) -> UnsafeMutableBufferPointer<Float> {
    var num_parameters = 0
    for i in 0..<NUM_PARAMETER_TENSORS {
        num_parameters += param_sizes[i]
    }
    // malloc all parameters all at once (https://stackoverflow.com/a/74021402)
    let params_memory = UnsafeMutableBufferPointer<Float>.allocate(capacity: num_parameters)
    // assign all the tensors
    var params_memory_iterator = params_memory.baseAddress!
    // Pointer initialization in Swift
    params.pointee.wte = params_memory_iterator
    params_memory_iterator += param_sizes[0]
    params.pointee.wpe = params_memory_iterator
    params_memory_iterator += param_sizes[1]
    params.pointee.ln1w = params_memory_iterator
    params_memory_iterator += param_sizes[2]
    params.pointee.ln1b = params_memory_iterator
    params_memory_iterator += param_sizes[3]
    params.pointee.qkvw = params_memory_iterator
    params_memory_iterator += param_sizes[4]
    params.pointee.qkvb = params_memory_iterator
    params_memory_iterator += param_sizes[5]
    params.pointee.attprojw = params_memory_iterator
    params_memory_iterator += param_sizes[6]
    params.pointee.attprojb = params_memory_iterator
    params_memory_iterator += param_sizes[7]
    params.pointee.ln2w = params_memory_iterator
    params_memory_iterator += param_sizes[8]
    params.pointee.ln2b = params_memory_iterator
    params_memory_iterator += param_sizes[9]
    params.pointee.fcw = params_memory_iterator
    params_memory_iterator += param_sizes[10]
    params.pointee.fcb = params_memory_iterator
    params_memory_iterator += param_sizes[11]
    params.pointee.fcprojw = params_memory_iterator
    params_memory_iterator += param_sizes[12]
    params.pointee.fcprojb = params_memory_iterator
    params_memory_iterator += param_sizes[13]
    params.pointee.lnfw = params_memory_iterator
    params_memory_iterator += param_sizes[14]
    params.pointee.lnfb = params_memory_iterator
    params_memory_iterator += param_sizes[15]
/*    
 A 1:1 port of the C implementation for pointer initialization.
 Quite verbose in Swift and also far too difficult to read and understand.
    let ptrs: [UnsafeMutablePointer<UnsafeMutablePointer<Float>>] = [
        withUnsafeMutablePointer(to: &params.pointee.wte) { $0 },
        withUnsafeMutablePointer(to: &params.pointee.wpe) { $0 },
        withUnsafeMutablePointer(to: &params.pointee.ln1w) { $0 },
        withUnsafeMutablePointer(to: &params.pointee.ln1b) { $0 },
        withUnsafeMutablePointer(to: &params.pointee.qkvw) { $0 },
        withUnsafeMutablePointer(to: &params.pointee.qkvb) { $0 },
        withUnsafeMutablePointer(to: &params.pointee.attprojw) { $0 },
        withUnsafeMutablePointer(to: &params.pointee.attprojb) { $0 },
        withUnsafeMutablePointer(to: &params.pointee.ln2w) { $0 },
        withUnsafeMutablePointer(to: &params.pointee.ln2b) { $0 },
        withUnsafeMutablePointer(to: &params.pointee.fcw) { $0 },
        withUnsafeMutablePointer(to: &params.pointee.fcb) { $0 },
        withUnsafeMutablePointer(to: &params.pointee.fcprojw) { $0 },
        withUnsafeMutablePointer(to: &params.pointee.fcprojb) { $0 },
        withUnsafeMutablePointer(to: &params.pointee.lnfw) { $0 },
        withUnsafeMutablePointer(to: &params.pointee.lnfb) { $0 }
    ]
    for i in 0..<NUM_PARAMETER_TENSORS {
        // ptrs[i][0] = params_memory_iterator
        // wordy variant of short form in previous line
        UnsafeMutableRawPointer(ptrs[i]).storeBytes(of: params_memory_iterator, as: UnsafeMutablePointer<Float>.self)
        params_memory_iterator += param_sizes[i]
    }
 */
    return params_memory
}

let NUM_ACTIVATION_TENSORS = 23
struct ActivationTensors {
    var encoded: UnsafeMutablePointer<Float>! // (B, T, C)
    var ln1: UnsafeMutablePointer<Float>! // (L, B, T, C)
    var ln1_mean: UnsafeMutablePointer<Float>! // (L, B, T)
    var ln1_rstd: UnsafeMutablePointer<Float>! // (L, B, T)
    var qkv: UnsafeMutablePointer<Float>! // (L, B, T, 3*C)
    var atty: UnsafeMutablePointer<Float>! // (L, B, T, C)
    var preatt: UnsafeMutablePointer<Float>! // (L, B, NH, T, T)
    var att: UnsafeMutablePointer<Float>! // (L, B, NH, T, T)
    var attproj: UnsafeMutablePointer<Float>! // (L, B, T, C)
    var residual2: UnsafeMutablePointer<Float>! // (L, B, T, C)
    var ln2: UnsafeMutablePointer<Float>! // (L, B, T, C)
    var ln2_mean: UnsafeMutablePointer<Float>! // (L, B, T)
    var ln2_rstd: UnsafeMutablePointer<Float>! // (L, B, T)
    var fch: UnsafeMutablePointer<Float>! // (L, B, T, 4*C)
    var fch_gelu: UnsafeMutablePointer<Float>! // (L, B, T, 4*C)
    var fcproj: UnsafeMutablePointer<Float>! // (L, B, T, C)
    var residual3: UnsafeMutablePointer<Float>! // (L, B, T, C)
    var lnf: UnsafeMutablePointer<Float>! // (B, T, C)
    var lnf_mean: UnsafeMutablePointer<Float>! // (B, T)
    var lnf_rstd: UnsafeMutablePointer<Float>! // (B, T)
    var logits: UnsafeMutablePointer<Float>! // (B, T, V)
    var probs: UnsafeMutablePointer<Float>! // (B, T, V)
    var losses: UnsafeMutablePointer<Float>! // (B, T)
}

// swiftlint:disable:next function_body_length
func malloc_and_point_activations(
    _ acts: UnsafeMutablePointer<ActivationTensors>,
    _ act_sizes: UnsafePointer<Int>) -> UnsafeMutableBufferPointer<Float> {
    var num_activations = 0
    for i in 0..<NUM_ACTIVATION_TENSORS {
        num_activations += act_sizes[i]
    }
    let acts_memory = UnsafeMutableBufferPointer<Float>.allocate(capacity: num_activations)
    var acts_memory_iterator = acts_memory.baseAddress!
    // Pointer initialization in Swift
    acts.pointee.encoded = acts_memory_iterator
    acts_memory_iterator += act_sizes[0]
    acts.pointee.ln1 = acts_memory_iterator
    acts_memory_iterator += act_sizes[1]
    acts.pointee.ln1_mean = acts_memory_iterator
    acts_memory_iterator += act_sizes[2]
    acts.pointee.ln1_rstd = acts_memory_iterator
    acts_memory_iterator += act_sizes[3]
    acts.pointee.qkv = acts_memory_iterator
    acts_memory_iterator += act_sizes[4]
    acts.pointee.atty = acts_memory_iterator
    acts_memory_iterator += act_sizes[5]
    acts.pointee.preatt = acts_memory_iterator
    acts_memory_iterator += act_sizes[6]
    acts.pointee.att = acts_memory_iterator
    acts_memory_iterator += act_sizes[7]
    acts.pointee.attproj = acts_memory_iterator
    acts_memory_iterator += act_sizes[8]
    acts.pointee.residual2 = acts_memory_iterator
    acts_memory_iterator += act_sizes[9]
    acts.pointee.ln2 = acts_memory_iterator
    acts_memory_iterator += act_sizes[10]
    acts.pointee.ln2_mean = acts_memory_iterator
    acts_memory_iterator += act_sizes[11]
    acts.pointee.ln2_rstd = acts_memory_iterator
    acts_memory_iterator += act_sizes[12]
    acts.pointee.fch = acts_memory_iterator
    acts_memory_iterator += act_sizes[13]
    acts.pointee.fch_gelu = acts_memory_iterator
    acts_memory_iterator += act_sizes[14]
    acts.pointee.fcproj = acts_memory_iterator
    acts_memory_iterator += act_sizes[15]
    acts.pointee.residual3 = acts_memory_iterator
    acts_memory_iterator += act_sizes[16]
    acts.pointee.lnf = acts_memory_iterator
    acts_memory_iterator += act_sizes[17]
    acts.pointee.lnf_mean = acts_memory_iterator
    acts_memory_iterator += act_sizes[18]
    acts.pointee.lnf_rstd = acts_memory_iterator
    acts_memory_iterator += act_sizes[19]
    acts.pointee.logits = acts_memory_iterator
    acts_memory_iterator += act_sizes[20]
    acts.pointee.probs = acts_memory_iterator
    acts_memory_iterator += act_sizes[21]
    acts.pointee.losses = acts_memory_iterator
    acts_memory_iterator += act_sizes[22]
/*
 A 1:1 port of the C implementation for pointer initialization.
 Quite verbose in Swift and also difficult to read and understand.
    let ptrs: [UnsafeMutablePointer<UnsafeMutablePointer<Float>>] = [
        withUnsafeMutablePointer(to: &acts.pointee.encoded) { $0 },
        withUnsafeMutablePointer(to: &acts.pointee.ln1) { $0 },
        withUnsafeMutablePointer(to: &acts.pointee.ln1_mean) { $0 },
        withUnsafeMutablePointer(to: &acts.pointee.ln1_rstd) { $0 },
        withUnsafeMutablePointer(to: &acts.pointee.qkv) { $0 },
        withUnsafeMutablePointer(to: &acts.pointee.atty) { $0 },
        withUnsafeMutablePointer(to: &acts.pointee.preatt) { $0 },
        withUnsafeMutablePointer(to: &acts.pointee.att) { $0 },
        withUnsafeMutablePointer(to: &acts.pointee.attproj) { $0 },
        withUnsafeMutablePointer(to: &acts.pointee.residual2) { $0 },
        withUnsafeMutablePointer(to: &acts.pointee.ln2) { $0 },
        withUnsafeMutablePointer(to: &acts.pointee.ln2_mean) { $0 },
        withUnsafeMutablePointer(to: &acts.pointee.ln2_rstd) { $0 },
        withUnsafeMutablePointer(to: &acts.pointee.fch) { $0 },
        withUnsafeMutablePointer(to: &acts.pointee.fch_gelu) { $0 },
        withUnsafeMutablePointer(to: &acts.pointee.fcproj) { $0 },
        withUnsafeMutablePointer(to: &acts.pointee.residual3) { $0 },
        withUnsafeMutablePointer(to: &acts.pointee.lnf) { $0 },
        withUnsafeMutablePointer(to: &acts.pointee.lnf_mean) { $0 },
        withUnsafeMutablePointer(to: &acts.pointee.lnf_rstd) { $0 },
        withUnsafeMutablePointer(to: &acts.pointee.logits) { $0 },
        withUnsafeMutablePointer(to: &acts.pointee.probs) { $0 },
        withUnsafeMutablePointer(to: &acts.pointee.losses) { $0 }
    ]
    for i in 0..<NUM_ACTIVATION_TENSORS {
        // ptrs[i][0] = acts_memory_iterator
        // wordy variant of short form in previous line
        UnsafeMutableRawPointer(ptrs[i]).storeBytes(of: acts_memory_iterator, as: UnsafeMutablePointer<Float>.self)
        acts_memory_iterator += act_sizes[i]
    }
 */
    return acts_memory
}

struct GPT2 {
    var config = GPT2Config()
    // the weights (parameters) of the model, and their sizes
    var params = ParameterTensors()
    var param_sizes = [Int](repeating: 0, count: NUM_PARAMETER_TENSORS)
    var params_memory: UnsafeMutablePointer<Float>?
    var num_parameters = 0
    // gradients of the weights
    var grads = ParameterTensors()
    var grads_memory: UnsafeMutablePointer<Float>?
    // buffers for the AdamW optimizer
    var m_memory: UnsafeMutablePointer<Float>?
    var v_memory: UnsafeMutablePointer<Float>?
    // the activations of the model, and their sizes
    var acts = ActivationTensors()
    var act_sizes = [Int](repeating: 0, count: NUM_ACTIVATION_TENSORS)
    var acts_memory: UnsafeMutablePointer<Float>?
    var num_activations = 0
    // gradients of the activations
    var grads_acts = ActivationTensors()
    var grads_acts_memory: UnsafeMutablePointer<Float>?
    // other run state configuration
    var batch_size = 0 // the batch size (B) of current forward pass
    var seq_len = 0 // the sequence length (T) of current forward pass
    var inputs: UnsafeMutablePointer<Int32>? // the input tokens for the current forward pass
    var targets: UnsafeMutablePointer<Int32>? // the target tokens for the current forward pass
    var mean_loss: Float = 0 // after a forward pass with targets, will be populated with the mean loss
}

// swiftlint:disable:next function_body_length
func gpt2_build_from_checkpoint(
    _ model: UnsafeMutablePointer<GPT2>,
    _ handle: FileHandle,
    _ stdlog: ((String) -> Void)?) throws {
    // read in model from a checkpoint file
    guard
        let header_data = try handle.read(upToCount: 256 * MemoryLayout<Int32>.size)
    else { throw LlmSwiftError.apiReturnedNil }
    let model_header = header_data.withUnsafeBytes { (header_data: UnsafeRawBufferPointer) -> [Int] in
        header_data.bindMemory(to: Int32.self).map { Int($0) }
    }
    assert(model_header[0] == 20240326, "Bad magic in model file (try `python train_gpt2.py`)")
    assert(model_header[1] == 3, "Wrong version in model file (try `python train_gpt2.py`)")

    // read in hyperparameters
    let maxT = model_header[2]
    let V = model_header[3]
    let L = model_header[4]
    let NH = model_header[5]
    let C = model_header[6]
    let Vp = model_header[7]
    model.pointee.config.max_seq_len = maxT
    model.pointee.config.vocab_size = V
    model.pointee.config.num_layers = L
    model.pointee.config.num_heads = NH
    model.pointee.config.channels = C
    model.pointee.config.padded_vocab_size = Vp
    stdlog?("[GPT-2]\n")
    stdlog?("max_seq_len: \(maxT)\n")
    stdlog?("vocab_size: \(V)\n")
    stdlog?("padded_vocab_size: \(Vp)\n")
    stdlog?("num_layers: \(L)\n")
    stdlog?("num_heads: \(NH)\n")
    stdlog?("channels: \(C)\n")

    // allocate space for all the parameters and read them in
    fill_in_parameter_sizes(&model.pointee.param_sizes, model.pointee.config)

    // count the number of parameters
    var num_parameters = 0
    for i in 0..<NUM_PARAMETER_TENSORS {
        num_parameters += model.pointee.param_sizes[i]
    }
    stdlog?("num_parameters: \(num_parameters)\n")
    model.pointee.num_parameters = num_parameters

    // read in all the parameters from file
    let params_memory = malloc_and_point_parameters(&model.pointee.params, model.pointee.param_sizes)
    let params_memory_buffer = UnsafeMutableRawBufferPointer(params_memory)
    _ = try FileDescriptor(rawValue: handle.fileDescriptor).read(into: params_memory_buffer)
    model.pointee.params_memory = params_memory.baseAddress
    try? handle.close()

    // other inits
    model.pointee.acts_memory = nil
    model.pointee.grads_memory = nil
    model.pointee.m_memory = nil
    model.pointee.v_memory = nil
    model.pointee.grads_acts_memory = nil
    model.pointee.inputs = nil
    model.pointee.targets = nil
    model.pointee.batch_size = 0
    model.pointee.seq_len = 0
    model.pointee.mean_loss = -1 // -1 will designate no loss
}

// swiftlint:disable:next function_parameter_count
func gpt2_forward( // swiftlint:disable:this function_body_length
    _ model: UnsafeMutablePointer<GPT2>,
    _ inputs: UnsafePointer<Int32>,
    _ targets: UnsafePointer<Int32>?,
    _ B: Int, _ T: Int, _ stdlog: ((String) -> Void)?) async throws {

    // ensure the model was initialized or error out
    guard
        model.pointee.params_memory != nil
    else { throw LlmSwiftError.wrongApiUsage }

    // convenience parameters
    let V = model.pointee.config.vocab_size
    let Vp = model.pointee.config.padded_vocab_size
    let L = model.pointee.config.num_layers
    let NH = model.pointee.config.num_heads
    let C = model.pointee.config.channels

    // validate inputs, all indices must be in the range [0, V]
    for i in 0..<B * T {
        if !(inputs[i] ~= 0..<V) { throw LlmSwiftError.outOfBounds }
        if let targets = targets {
            if !(targets[i] ~= 0..<V) { throw LlmSwiftError.outOfBounds }
        }
    }

    // allocate space for all the activations if needed (done here, lazily)
    if model.pointee.acts_memory == nil {
        // record the current B,T as well
        model.pointee.batch_size = B
        model.pointee.seq_len = T
        // and now allocate the space
        model.pointee.act_sizes[0] = B * T * C // encoded
        model.pointee.act_sizes[1] = L * B * T * C // ln1
        model.pointee.act_sizes[2] = L * B * T  // ln1_mean
        model.pointee.act_sizes[3] = L * B * T  // ln1_rstd
        model.pointee.act_sizes[4] = L * B * T * 3 * C // qkv
        model.pointee.act_sizes[5] = L * B * T * C  // atty
        model.pointee.act_sizes[6] = L * B * NH * T * T  // preatt
        model.pointee.act_sizes[7] = L * B * NH * T * T  // att
        model.pointee.act_sizes[8] = L * B * T * C // attproj
        model.pointee.act_sizes[9] = L * B * T * C // residual2
        model.pointee.act_sizes[10] = L * B * T * C // ln2
        model.pointee.act_sizes[11] = L * B * T // ln2_mean
        model.pointee.act_sizes[12] = L * B * T // ln2_rstd
        model.pointee.act_sizes[13] = L * B * T * 4 * C // fch
        model.pointee.act_sizes[14] = L * B * T * 4 * C // fch_gelu
        model.pointee.act_sizes[15] = L * B * T * C // fcproj
        model.pointee.act_sizes[16] = L * B * T * C // residual3
        model.pointee.act_sizes[17] = B * T * C // lnf
        model.pointee.act_sizes[18] = B * T // lnf_mean
        model.pointee.act_sizes[19] = B * T // lnf_rstd
        model.pointee.act_sizes[20] = B * T * Vp // logits
        model.pointee.act_sizes[21] = B * T * Vp // probs
        model.pointee.act_sizes[22] = B * T // losses
        var num_activations = 0
        for i in 0..<NUM_ACTIVATION_TENSORS {
            num_activations += model.pointee.act_sizes[i]
        }
        stdlog?("num_activations: \(num_activations)\n")
        model.pointee.num_activations = num_activations
        let acts_memory = malloc_and_point_activations(&model.pointee.acts, model.pointee.act_sizes)
        model.pointee.acts_memory = acts_memory.baseAddress
        // also create memory for caching inputs and targets
        model.pointee.inputs = UnsafeMutablePointer<Int32>.allocate(capacity: B * T)
        // might be unused if we never have targets but it's small
        model.pointee.targets = UnsafeMutablePointer<Int32>.allocate(capacity: B * T)
    } else {
        // validate B,T are not larger than the values used at initialisation
        // (smaller B,T are okay for inference only)
        if B > model.pointee.batch_size || T > model.pointee.seq_len {
            throw LlmSwiftError.wrongApiUsage
        }
    }

    // cache the inputs/targets
    model.pointee.inputs!.update(from: inputs, count: B * T)
    if let targets = targets {
        model.pointee.targets!.update(from: targets, count: B * T)
    }

    // forward pass
    let params = model.pointee.params // for brevity
    let acts = model.pointee.acts
    var residual: UnsafeMutablePointer<Float>
    encoder_forward(acts.encoded, inputs, params.wte, params.wpe, B, T, C) // encoding goes into residual[0]
    for l in 0..<L {
        residual = l == 0 ? acts.encoded : acts.residual3 + (l - 1) * B * T * C

        // get the pointers of the weights for this layer
        let l_ln1w = params.ln1w + l * C
        let l_ln1b = params.ln1b + l * C
        let l_qkvw = params.qkvw + l * 3 * C * C
        let l_qkvb = params.qkvb + l * 3 * C
        let l_attprojw = params.attprojw + l * C * C
        let l_attprojb = params.attprojb + l * C
        let l_ln2w = params.ln2w + l * C
        let l_ln2b = params.ln2b + l * C
        let l_fcw = params.fcw + l * 4 * C * C
        let l_fcb = params.fcb + l * 4 * C
        let l_fcprojw = params.fcprojw + l * C * 4 * C
        let l_fcprojb = params.fcprojb + l * C

        // get the pointers of the activations for this layer
        let l_ln1 = acts.ln1 + l * B * T * C
        let l_ln1_mean = acts.ln1_mean + l * B * T
        let l_ln1_rstd = acts.ln1_rstd + l * B * T
        let l_qkv = acts.qkv + l * B * T * 3 * C
        let l_atty = acts.atty + l * B * T * C
        let l_preatt = acts.preatt + l * B * NH * T * T
        let l_att = acts.att + l * B * NH * T * T
        let l_attproj = acts.attproj + l * B * T * C
        let l_residual2 = acts.residual2 + l * B * T * C
        let l_ln2 = acts.ln2 + l * B * T * C
        let l_ln2_mean = acts.ln2_mean + l * B * T
        let l_ln2_rstd = acts.ln2_rstd + l * B * T
        let l_fch = acts.fch + l * B * T * 4 * C
        let l_fch_gelu = acts.fch_gelu + l * B * T * 4 * C
        let l_fcproj = acts.fcproj + l * B * T * C
        let l_residual3 = acts.residual3 + l * B * T * C

        // now do the forward pass
        layernorm_forward(l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, B, T, C)
        await matmul_forward(l_qkv, l_ln1, l_qkvw, l_qkvb, B, T, C, 3 * C)
        await attention_forward(l_atty, l_preatt, l_att, l_qkv, B, T, C, NH)
        await matmul_forward(l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C)
        residual_forward(l_residual2, residual, l_attproj, B * T * C)
        layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, B, T, C)
        await matmul_forward(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4 * C)
        gelu_forward(l_fch_gelu, l_fch, B * T * 4 * C)
        await matmul_forward(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4 * C, C)
        residual_forward(l_residual3, l_residual2, l_fcproj, B * T * C)
    }
    residual = acts.residual3 + (L - 1) * B * T * C // last residual is in residual3
    layernorm_forward(acts.lnf, acts.lnf_mean, acts.lnf_rstd, residual, params.lnfw, params.lnfb, B, T, C)
    await matmul_forward(acts.logits, acts.lnf, params.wte, nil, B, T, C, Vp)
    await softmax_forward(acts.probs, acts.logits, B, T, V, Vp)

    // also forward the cross-entropy loss function if we have the targets
    if let targets = targets {
        crossentropy_forward(model.pointee.acts.losses, model.pointee.acts.probs, targets, B, T, Vp)
        // for convenience also evaluate the mean loss
        var mean_loss: Float = 0
        for i in 0..<B * T { mean_loss += model.pointee.acts.losses[i] }
        mean_loss /= Float(B * T)
        model.pointee.mean_loss = mean_loss
    } else {
        // if we don't have targets, we don't have a loss
        model.pointee.mean_loss = -1
    }
}

func gpt2_zero_grad(_ model: UnsafeMutablePointer<GPT2>) {
    if let grads_memory = model.pointee.grads_memory {
        grads_memory.update(repeating: 0, count: model.pointee.num_parameters)
    }
    if let grads_acts_memory = model.pointee.grads_acts_memory {
        grads_acts_memory.update(repeating: 0, count: model.pointee.num_activations)
    }
}

// swiftlint:disable:next function_body_length
func gpt2_backward(_ model: UnsafeMutablePointer<GPT2>) async throws {
    // double check we forwarded previously, with targets
    if model.pointee.mean_loss == -1 {
        throw LlmSwiftError.wrongApiUsage // must call gpt2_forward with `targets´ before this API
    }

    // lazily allocate the memory for gradients of the weights and activations, if needed
    if model.pointee.grads_memory == nil {
        let grads_memory = malloc_and_point_parameters(&model.pointee.grads, model.pointee.param_sizes)
        model.pointee.grads_memory = grads_memory.baseAddress
        let grads_acts_memory = malloc_and_point_activations(&model.pointee.grads_acts, model.pointee.act_sizes)
        model.pointee.grads_acts_memory = grads_acts_memory.baseAddress
        gpt2_zero_grad(model)
    }

    // convenience shortcuts
    let B = model.pointee.batch_size
    let T = model.pointee.seq_len
    let V = model.pointee.config.vocab_size
    let Vp = model.pointee.config.padded_vocab_size
    let L = model.pointee.config.num_layers
    let NH = model.pointee.config.num_heads
    let C = model.pointee.config.channels

    // backward pass: go in the reverse order of the forward pass, and call backward() functions
    let params = model.pointee.params // for brevity
    let grads = model.pointee.grads
    let acts = model.pointee.acts
    let grads_acts = model.pointee.grads_acts

    // we kick off the chain rule by filling in dlosses with 1/(B*T)
    // technically this is a small, inline backward() pass of calculating
    // total, final loss as the mean over all losses over all (B,T) positions in the batch
    let dloss_mean = 1 / Float(B * T)
    for i in 0..<B * T { grads_acts.losses[i] = dloss_mean }

    crossentropy_softmax_backward(grads_acts.logits, grads_acts.losses, acts.probs, model.pointee.targets!, B, T, V, Vp)
    await matmul_backward(grads_acts.lnf, grads.wte, nil, grads_acts.logits, acts.lnf, params.wte, B, T, C, Vp)
    var residual = acts.residual3 + (L - 1) * B * T * C // last layer's residual
    var dresidual = grads_acts.residual3 + (L - 1) * B * T * C // write to last layer's residual
    layernorm_backward(dresidual, grads.lnfw, grads.lnfb, grads_acts.lnf, residual, params.lnfw, acts.lnf_mean, acts.lnf_rstd, B, T, C)

    for l in (0..<L).reversed() {
        residual = l == 0 ? acts.encoded : acts.residual3 + (l - 1) * B * T * C
        dresidual = l == 0 ? grads_acts.encoded : grads_acts.residual3 + (l - 1) * B * T * C

        // get the pointers of the weights for this layer
        let l_ln1w = params.ln1w + l * C
        let l_qkvw = params.qkvw + l * 3 * C * C
        let l_attprojw = params.attprojw + l * C * C
        let l_ln2w = params.ln2w + l * C
        let l_fcw = params.fcw + l * 4 * C * C
        let l_fcprojw = params.fcprojw + l * C * 4 * C
        // get the pointers of the gradients of the weights for this layer
        let dl_ln1w = grads.ln1w + l * C
        let dl_ln1b = grads.ln1b + l * C
        let dl_qkvw = grads.qkvw + l * 3 * C * C
        let dl_qkvb = grads.qkvb + l * 3 * C
        let dl_attprojw = grads.attprojw + l * C * C
        let dl_attprojb = grads.attprojb + l * C
        let dl_ln2w = grads.ln2w + l * C
        let dl_ln2b = grads.ln2b + l * C
        let dl_fcw = grads.fcw + l * 4 * C * C
        let dl_fcb = grads.fcb + l * 4 * C
        let dl_fcprojw = grads.fcprojw + l * C * 4 * C
        let dl_fcprojb = grads.fcprojb + l * C
        // get the pointers of the activations for this layer
        let l_ln1 = acts.ln1 + l * B * T * C
        let l_ln1_mean = acts.ln1_mean + l * B * T
        let l_ln1_rstd = acts.ln1_rstd + l * B * T
        let l_qkv = acts.qkv + l * B * T * 3 * C
        let l_atty = acts.atty + l * B * T * C
        let l_att = acts.att + l * B * NH * T * T
        let l_residual2 = acts.residual2 + l * B * T * C
        let l_ln2 = acts.ln2 + l * B * T * C
        let l_ln2_mean = acts.ln2_mean + l * B * T
        let l_ln2_rstd = acts.ln2_rstd + l * B * T
        let l_fch = acts.fch + l * B * T * 4 * C
        let l_fch_gelu = acts.fch_gelu + l * B * T * 4 * C
        // get the pointers of the gradients of the activations for this layer
        let dl_ln1 = grads_acts.ln1 + l * B * T * C
        let dl_qkv = grads_acts.qkv + l * B * T * 3 * C
        let dl_atty = grads_acts.atty + l * B * T * C
        let dl_preatt = grads_acts.preatt + l * B * NH * T * T
        let dl_att = grads_acts.att + l * B * NH * T * T
        let dl_attproj = grads_acts.attproj + l * B * T * C
        let dl_residual2 = grads_acts.residual2 + l * B * T * C
        let dl_ln2 = grads_acts.ln2 + l * B * T * C
        let dl_fch = grads_acts.fch + l * B * T * 4 * C
        let dl_fch_gelu = grads_acts.fch_gelu + l * B * T * 4 * C
        let dl_fcproj = grads_acts.fcproj + l * B * T * C
        let dl_residual3 = grads_acts.residual3 + l * B * T * C

        // backprop this layer
        residual_backward(dl_residual2, dl_fcproj, dl_residual3, B * T * C)
        await matmul_backward(dl_fch_gelu, dl_fcprojw, dl_fcprojb, dl_fcproj, l_fch_gelu, l_fcprojw, B, T, 4 * C, C)
        gelu_backward(dl_fch, l_fch, dl_fch_gelu, B * T * 4 * C)
        await matmul_backward(dl_ln2, dl_fcw, dl_fcb, dl_fch, l_ln2, l_fcw, B, T, C, 4 * C)
        layernorm_backward(dl_residual2, dl_ln2w, dl_ln2b, dl_ln2, l_residual2, l_ln2w, l_ln2_mean, l_ln2_rstd, B, T, C)
        residual_backward(dresidual, dl_attproj, dl_residual2, B * T * C)
        await matmul_backward(dl_atty, dl_attprojw, dl_attprojb, dl_attproj, l_atty, l_attprojw, B, T, C, C)
        attention_backward(dl_qkv, dl_preatt, dl_att, dl_atty, l_qkv, l_att, B, T, C, NH)
        await matmul_backward(dl_ln1, dl_qkvw, dl_qkvb, dl_qkv, l_ln1, l_qkvw, B, T, C, 3 * C)
        layernorm_backward(dresidual, dl_ln1w, dl_ln1b, dl_ln1, residual, l_ln1w, l_ln1_mean, l_ln1_rstd, B, T, C)
    }
    encoder_backward(grads.wte, grads.wpe, grads_acts.encoded, model.pointee.inputs!, B, T, C)
}

// swiftlint:disable:next function_parameter_count
func gpt2_update(
    _ model: UnsafeMutablePointer<GPT2>,
    _ learning_rate: Float,
    _ beta1: Float, _ beta2: Float,
    _ eps: Float, _ weight_decay: Float, _ t: Int) {
    // reference: https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html

    // lazily allocate the memory for m_memory and v_memory
    if model.pointee.m_memory == nil {
        let m_memory = UnsafeMutableBufferPointer<Float>.allocate(capacity: model.pointee.num_parameters)
        model.pointee.m_memory = m_memory.baseAddress
        model.pointee.m_memory!.initialize(repeating: 0, count: model.pointee.num_parameters)
        let v_memory = UnsafeMutableBufferPointer<Float>.allocate(capacity: model.pointee.num_parameters)
        model.pointee.v_memory = v_memory.baseAddress
        model.pointee.v_memory!.initialize(repeating: 0, count: model.pointee.num_parameters)
    }
    let m_memory = model.pointee.m_memory!
    let v_memory = model.pointee.v_memory!
    let params_memory = model.pointee.params_memory!
    let grads_memory = model.pointee.grads_memory!

    for i in 0..<model.pointee.num_parameters {
        let param = (params_memory + i).pointee
        let grad = (grads_memory + i).pointee

        // update the first moment (momentum)
        let m = beta1 * (m_memory + i).pointee + (1 - beta1) * grad
        // update the second moment (RMSprop)
        let v = beta2 * (v_memory + i).pointee + (1 - beta2) * grad * grad
        // bias-correct both moments
        let m_hat = m / (1 - powf(beta1, Float(t)))
        let v_hat = v / (1 - powf(beta2, Float(t)))

        // update
        (m_memory + i).pointee = m
        (v_memory + i).pointee = v
        (params_memory + i).pointee -= learning_rate * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * param)
    }
}

func gpt2_free(_ model: UnsafeMutablePointer<GPT2>) {
    model.pointee.params_memory?.deallocate()
    model.pointee.grads_memory?.deallocate()
    model.pointee.m_memory?.deallocate()
    model.pointee.v_memory?.deallocate()
    model.pointee.acts_memory?.deallocate()
    model.pointee.grads_acts_memory?.deallocate()
    model.pointee.inputs?.deallocate()
    model.pointee.targets?.deallocate()
}

// #ifndef TESTING
// if we are TESTING (see test_gpt2.c), we'll skip the int main below
// ----------------------------------------------------------------------------
// sampler

func random_u32(_ state: UnsafeMutablePointer<UInt64>) -> UInt32 {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    state.pointee ^= state.pointee >> 12
    state.pointee ^= state.pointee << 25
    state.pointee ^= state.pointee >> 27
    return UInt32((state.pointee &* 0x2545F4914F6CDD1D) >> 32)
}

func random_f32(_ state: UnsafeMutablePointer<UInt64>) -> Float { // random Float in [0,1]
    return Float(random_u32(state) >> 8) / 16777216
}

func sample_mult(_ probabilities: UnsafeMutablePointer<Float>, _ n: Int, _ coin: Float) -> Int {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    var cdf: Float = 0
    for i in 0..<n {
        cdf += probabilities[i]
        if coin < cdf {
            return i
        }
    }
    return n - 1 // in case of rounding errors
}

// ----------------------------------------------------------------------------
// main training loop
// swiftlint:disable:next function_body_length
func train_gpt2(_ folder: URL?, _ stdlog: ((String) -> Void)? = nil) async throws {
    let cwd = FileManager.default.currentDirectoryPath
    defer { FileManager.default.changeCurrentDirectoryPath(cwd) }
    if let folder = folder {
        FileManager.default.changeCurrentDirectoryPath(folder.path)
    }

    // build the GPT-2 model from a checkpoint
    var model = GPT2()
    let model_filename = "gpt2_124M.bin"
    guard
        let model_handle = FileHandle(forReadingAtPath: model_filename)
    else { throw LlmSwiftError.apiReturnedNil }
    try gpt2_build_from_checkpoint(&model, model_handle, stdlog)

    // build the DataLoaders from tokens files. for now use tiny_shakespeare if available, else tiny_stories
    let tiny_stories_train = "dev/data/tinystories/TinyStories_train.bin"
    let tiny_stories_val = "dev/data/tinystories/TinyStories_val.bin"
    let tiny_shakespeare_train = "dev/data/tinyshakespeare/tiny_shakespeare_train.bin"
    let tiny_shakespeare_val = "dev/data/tinyshakespeare/tiny_shakespeare_val.bin"
    let train_tokens = FileManager.default.fileExists(atPath: tiny_shakespeare_train) ? tiny_shakespeare_train : tiny_stories_train
    let val_tokens = FileManager.default.fileExists(atPath: tiny_shakespeare_val) ? tiny_shakespeare_val : tiny_stories_val
    let B = 4 // batch size 4 (i.e. 4 independent token sequences will be trained on)
    let T = 64 // sequence length 64 (i.e. each sequence is 64 tokens long). must be <= maxT, which is 1024 for GPT-2
    var train_loader = DataLoader()
    var val_loader = DataLoader()
    try dataloader_init(&train_loader, train_tokens, B, T, 0, 1, true)
    try dataloader_init(&val_loader, val_tokens, B, T, 0, 1, false)
    stdlog?("train dataset num_batches: \(train_loader.num_tokens / (B * T))\n")
    stdlog?("val dataset num_batches: \(val_loader.num_tokens / (B * T))\n")
    let val_num_batches = 5

    // build the Tokenizer
    var tokenizer = Tokenizer()
    let tokenizer_filename = "gpt2_tokenizer.bin"
    guard
        let tokenizer_handle = FileHandle(forReadingAtPath: tokenizer_filename)
    else { throw LlmSwiftError.apiReturnedNil }
    try tokenizer_init(&tokenizer, tokenizer_handle)

    // some memory for generating samples from the model
    let rng_state = UnsafeMutablePointer<UInt64>.allocate(capacity: 1)
    let gen_tokens = UnsafeMutablePointer<Int32>.allocate(capacity: B * T)
    defer {
        rng_state.deallocate()
        gen_tokens.deallocate()
    }
    rng_state.initialize(to: 1337)
    let genT = 64 // number of steps of inference we will do

    // train
    for step in 0...40 {
        // once in a while estimate the validation loss
        if step % 10 == 0 {
            var val_loss: Float = 0
            try dataloader_reset(&val_loader)
            for _ in 0..<val_num_batches {
                try dataloader_next_batch(&val_loader)
                try await gpt2_forward(&model, val_loader.inputs, val_loader.targets, B, T, stdlog)
                val_loss += model.mean_loss
            }
            val_loss /= Float(val_num_batches)
            stdlog?("val loss \(val_loss)\n")
        }

        // once in a while do model inference to print generated text
        if step > 0 && step % 20 == 0 {
            // fill up gen_tokens with the GPT2_EOT, which kicks off the generation
            for i in 0..<B * T {
                gen_tokens[i] = tokenizer.eot_token
            }
            // now sample from the model autoregressively
            stdlog?("generating:\n---\n")
            for t in 1..<genT {
                // note that inference is very wasteful here because for each token
                // we re-calculate the forward pass for all of (B,T) positions from scratch
                // but the inference here is just for sanity checking anyway
                // and we can maybe optimize a bit more later, with careful tests
                try await gpt2_forward(&model, gen_tokens, nil, B, T, stdlog)
                // furthermore, below we're only using b=0 (i.e. the first row) of all B rows
                // we're in principle running B "inference streams" in parallel here
                // but only using position 0
                // get the Vp-dimensional vector probs[0, t-1, :]
                let probs = model.acts.probs + (t - 1) * model.config.padded_vocab_size
                let coin = random_f32(rng_state)
                // note we're only sampling from the first V elements, ignoring padding
                // (the probabilities in the padded region should be zero anyway)
                let next_token = sample_mult(probs, model.config.vocab_size, coin)
                gen_tokens[t] = Int32(next_token)
                // print the generated token, either using the Tokenizer or a fallback
                if tokenizer.init_ok {
                    if let token_str = tokenizer_decode(&tokenizer, next_token) {
                        if isprint(token_str) {
                            stdlog?(String(cString: token_str))
                        }
                    } else {
                        stdlog?("Invalid token id \(next_token)\n")
                    }
                } else {
                    // fall back to printing the token id
                    stdlog?("\(next_token) ")
                }
                fflush(stdout)
            }
            stdlog?("\n---\n")
        }

        // do a training step
        let start = Date.timeIntervalSinceReferenceDate
        try dataloader_next_batch(&train_loader)
        try await gpt2_forward(&model, train_loader.inputs, train_loader.targets, B, T, stdlog)
        gpt2_zero_grad(&model)
        try await gpt2_backward(&model)
        gpt2_update(&model, 1e-4, 0.9, 0.999, 1e-8, 0, step + 1)
        let end = Date.timeIntervalSinceReferenceDate
        stdlog?("step \(step): train loss \(model.mean_loss) (took \(String(format: "%1.2f", (end - start) * 1000)) ms)\n")
    }

    // free
    dataloader_free(&train_loader)
    dataloader_free(&val_loader)
    tokenizer_free(&tokenizer)
    gpt2_free(&model)
}
// #endif
// swiftlint:disable:this file_length
