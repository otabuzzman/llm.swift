#include <metal_stdlib>
using namespace metal;

// --- crossentropy_forward.metal
// #include <metal_stdlib>
// using namespace metal;

kernel void crossentropy_forward_kernel1(device float* losses [[ buffer(0) ]],
                                device float* probs [[ buffer(1) ]],
                                device int* targets [[ buffer(2) ]],
                                constant int& B [[ buffer(3) ]],
                                constant int& T [[ buffer(4) ]],
                                constant int& V [[ buffer(5) ]],
                                uint idx [[ thread_position_in_grid ]]) {
    // uncomment if nonuniform threadgroups not available
    // if (idx >= B * T) { return; }

    int b = idx / T;
    int t = idx % T;
    const device float* probs_bt = probs + b * T * V + t * V;
    int ix = targets[b * T + t];
    losses[b * T + t] = -log(probs_bt[ix]);
}

// --- softmax_forward.metal
// #include <metal_stdlib>
// using namespace metal;

kernel void softmax_forward_kernel1(device float* out [[ buffer(0) ]],
                                device float* inp [[ buffer(1) ]],
                                constant int& BT [[ buffer(2) ]],
                                constant int& V  [[ buffer(3) ]],
                                constant int& Vp [[ buffer(4) ]],
                                uint idx [[ thread_position_in_grid ]]) {
    // uncomment if nonuniform threadgroups not available
    // if (idx >= BT) { return; }

    const device float* inp_row = inp + idx * Vp;
    device float* out_row = out + idx * Vp;

    float maxval = -INFINITY;
    for (int j = 0; j < V; j++) {
        if (inp_row[j] > maxval) {
            maxval = inp_row[j];
        }
    }
    float sum = 0.0f;
    for (int j = 0; j < V; j++) {
        out_row[j] = precise::exp(inp_row[j] - maxval);
        sum += out_row[j];
    }
    for (int j = 0; j < V; j++) {
        out_row[j] /= sum;
    }
    // set probabilities for filled tokens to zero (as in CPU layer function)
    for (int j = V; j < Vp; j++) {
        out_row[j] = 0.0f;
    }
}

// simdgroup-level max reduction (MSL: simd_max)
inline float simdReduceMax(float val, int threads_per_simdgroup) {
    for (int offset = threads_per_simdgroup / 2; offset > 0; offset /= 2) {
        val = fmax(val, simd_shuffle_down(val, offset));
    }
    return val;
}

// simdgroup-level sum reduction (MSL: simd_sum)
inline float simdReduceSum(float val, int threads_per_simdgroup) {
    for (int offset = threads_per_simdgroup / 2; offset > 0; offset /= 2) {
        val += fmax(val, simd_shuffle_down(val, offset));
    }
    return val;
}

kernel void softmax_forward_kernel4(device float* out [[ buffer(0) ]],
                                device float* inp [[ buffer(1) ]],
                                constant int& BT [[ buffer(2) ]],
                                constant int& V [[ buffer(3) ]],
                                constant int& Vp [[ buffer(4) ]],
                                // for max thread ID check if nonuniform threadgroups not available
                                // uint idx [[ thread_position_in_grid ]], // CUDA blockIdx * blockDim + threadIdx
                                uint tgid [[ threadgroup_position_in_grid ]], // CUDA blockIdx
                                uint tid [[ thread_position_in_threadgroup ]], // CUDA threadIdx
                                uint tgSize [[ threads_per_threadgroup ]], // CUDA blockDim
                                uint laneId [[ thread_index_in_simdgroup ]], // CUDA threadIdx % 32
                                // for simdReduceMax and simdReduceSum
                                // uint sgSize [[ threads_per_simdgroup ]], // CUDA warp size
                                uint warpId [[ simdgroup_index_in_threadgroup ]], // CUDA threadIdx / 32
                                uint sgInTg [[ simdgroups_per_threadgroup ]], // CUDA blockDim / 32
                                threadgroup float* shared [[ threadgroup(0) ]]) {
    // uncomment if nonuniform threadgroups not available
    // if (idx >= BT * tgSize) { return; }

    // out is (BT, Vp) just like inp. Each row of inp will get softmaxed.
    // same as kernel3, but can handle any block size (multiple of 32)
    // each row of V elements is handled by block_size threads
    // furthermore, each block_size threads get executed in warps of 32 threads

    // special reduction operations simd_max/simd_sum are used for intra-warp reductions
    // shared memory is used for inter-warp reduction

    // shared[] must be allocated to have sgInTg elements
    // those will be used for max and sum values
    threadgroup float* max_or_sum_storage = shared;

    // one row of inp, i.e. inp[tgid, :] of shape (Vp,)
    const device float* x = inp + tgid * Vp;

    // first, thread coarsening by directly accessing global memory in series
    float maxval = -INFINITY;
    for (int i = tid; i < V; i += tgSize) {
        maxval = fmax(maxval, x[i]);
    }
    // now within-warp reductions for maxval
    maxval = simd_max(maxval); // instead of own simdReduceMax()

    // the 0th thread of each warp writes the maxval of that warp to shared memory
    if (laneId == 0) max_or_sum_storage[warpId] = maxval;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // now the 0th thread of the block reduces the max values in shared memory, i.e. across warps
    if (tid == 0) {
        float val = max_or_sum_storage[tid];
        for (uint i = 1; i < sgInTg; i++) {
            val = fmax(val, max_or_sum_storage[i]);
        }
        // store the final max in the first position
        max_or_sum_storage[0] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // broadcast the max to all threads
    float offset = max_or_sum_storage[0];

    // compute expf and write the result to global memory
    for (int i = tid; i < V; i += tgSize) {
        out[tgid * Vp + i] = precise::exp(x[i] - offset);
    }

    // okay now we calculated exp(x - max(x))
    // step 2: sum all the values and divide by the sum

    // thread coarsening for sum
    x = out + tgid * Vp;
    float sumval = 0.0f;
    for (int i = tid; i < V; i += tgSize) {
        sumval += x[i];
    }
    // within-warp reduction for sumval
    sumval = simd_sum(sumval); // instead of own simdReduceSum()

    // write sumval to shared memory
    if (laneId == 0) max_or_sum_storage[warpId] = sumval;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // inter-thread reduction of sum
    if (tid == 0) {
        float val = max_or_sum_storage[tid];
        for (uint i = 1; i < sgInTg; ++i) {
            val += max_or_sum_storage[i];
        }
        max_or_sum_storage[0] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // broadcast the sum to all threads
    float sum = max_or_sum_storage[0];

    // divide the whole row by the sum
    for (int i = tid; i < V; i += tgSize) {
        out[tgid * Vp + i] = x[i] / sum;
    }
}

// missing std::remove_reference in MSL, ChatGPT implementation
template <typename T>
struct std_remove_reference {
    using type = T;
};

template <typename T>
struct std_remove_reference<thread T&> {
    using type = T;
};

template <typename T>
struct std_remove_reference<thread T&&> {
    using type = T;
};

// missing std::forward in MSL, ChatGPT implementation
template <typename T>
constexpr thread T&& std_forward(thread typename std_remove_reference<T>::type& t) {
    return static_cast<thread T&&>(t);
}

template <typename T>
constexpr thread T&& std_forward(thread typename std_remove_reference<T>::type&& t) {
    return static_cast<thread T&&>(t);
}

// C++ template for missing #pragma unroll in MSL https://stackoverflow.com/a/28232338
template <unsigned N>
struct pragma_unroll {
    template <typename F, typename... Args>
    static void call(thread F&& f, thread Args&&... args) {
        f(std_forward<Args>(args)...);
        return pragma_unroll<N - 1>::call(std_forward<F>(f), std_forward<Args>(args)...);
    }
};

template <>
struct pragma_unroll<0u> {
    template <typename F, typename... Args>
    static void call(thread F&&, thread Args&&...) {}
};

// proxy functions for pragma_unroll template
inline void loop0(thread uint& u, thread uint& i, constant uint& V, thread uint& tgSize, const device float* x, thread float& maxval) {
    maxval = fmax(maxval, x[min(V - 1, i + u++ * tgSize)]);
}

inline void loop1(thread uint& i, thread float& val, threadgroup float* maxvals) {
    val = fmax(val, maxvals[i++]);
}

inline void loop2(thread uint& u, thread uint& i, constant uint& V, thread uint& tgSize, const device float* x, thread float* reg_array) {
    float output = x[min(V - 1, i + u * tgSize)];
    reg_array[u++] = output;
}

inline void loop3(thread uint& u, thread uint& i, constant uint& V, thread uint& tgSize, device float* y, thread float* reg_array, thread float& offset, thread float& sumval) {
    if (i + u * tgSize < V) {
        float output = precise::exp(reg_array[u] - offset);
        y[min(V - 1, i + u++ * tgSize)] = output; // compiler likes redundant min()?!
        sumval += output; // combined into the same loop unlike kernel3
    }
}

inline void loop4(thread uint i, thread float& val, threadgroup float* sumvals) {
    val += sumvals[i++];
}

inline void loop5(thread uint& u, thread uint& i, constant uint& V, thread uint& tgSize, device float* y, thread float* reg_array) {
    float output = y[min(V - 1, i + u * tgSize)];
    reg_array[u++] = output;
}

inline void loop6(thread uint& u, thread uint& i, constant uint& V, thread uint& tgSize, device float* y, thread float* reg_array, thread float& sum) {
    if (i + u * tgSize < V) {
        float output = reg_array[u] / sum;
        y[i + u++ * tgSize] = output;
    }
}

kernel void softmax_forward_kernel7(device float* out [[ buffer(0) ]],
                                device float* inp [[ buffer(1) ]],
                                constant int& BT [[ buffer(2) ]],
                                constant uint& V [[ buffer(3) ]],
                                constant uint& Vp [[ buffer(4) ]],
                                // for max thread ID check if nonuniform threadgroups not available
                                // uint idx [[ thread_position_in_grid ]], // CUDA blockIdx * blockDim + threadIdx
                                uint tgid [[ threadgroup_position_in_grid ]], // CUDA blockIdx
                                uint tid [[ thread_position_in_threadgroup ]], // CUDA threadIdx
                                uint tgSize [[ threads_per_threadgroup ]], // CUDA blockDim
                                uint laneId [[ thread_index_in_simdgroup ]], // CUDA threadIdx % 32
                                // for simdReduceMax and simdReduceSum
                                // uint sgSize [[ threads_per_simdgroup ]], // CUDA warp size
                                uint warpId [[ simdgroup_index_in_threadgroup ]], // CUDA threadIdx / 32
                                uint sgInTg [[ simdgroups_per_threadgroup ]], // CUDA blockDim / 32
                                threadgroup float* shared [[ threadgroup(0) ]]) {
    // uncomment if nonuniform threadgroups not available
    // if (idx >= BT * tgSize) { return; }

    // out is (BT, Vp) just like inp. Each row of inp will get softmaxed.
    // same as kernel4, but optimised for very large Vs with advanced unrolling

    // The trick is to read into a register array (all indices known at compile time)
    // and always read UNROLL_FACTOR values to maximise memory level parallelism
    // even if we would be out of bounds, we set the index to min(V-1, tgid)
    // so we just do some unnecessary reads (obviously bad for small V)
    // the writes are in a separate loop with a conditional check for out of bounds
    // making it separate is necessary to convince the compiler to do the right thing

    const int UNROLL_FACTOR = 8;

    // shared[] must be allocated to have 2 * sgInTg elements
    // first half for max values, the second half for sum values
    threadgroup float* maxvals = shared;
    threadgroup float* sumvals = &shared[sgInTg];

    if (tid >= V) {
        maxvals[warpId] = -INFINITY;
        sumvals[warpId] = 0.0f;
        return;
    }

    const device float* x = inp + tgid * Vp; // input
    device float* y = out + tgid * Vp; // output

    // first, thread coarsening by directly accessing global memory in series
    float maxval = -INFINITY;
    for (uint i = tid; i < V; i += tgSize * UNROLL_FACTOR) {
        // #pragma unroll
        uint u = 0;
        pragma_unroll<UNROLL_FACTOR>::call(loop0, u, i, V, tgSize, x, maxval);
//        for (int u = 0; u < UNROLL_FACTOR; u++) {
//            maxval = fmax(maxval, x[min(V - 1, i + u * tgSize)]);
//        }
    }

    // now within-warp reductions for maxval
    maxval = simd_max(maxval);
    // the 0th thread of each warp writes the maxval of that warp to shared memory
    if (laneId == 0) maxvals[warpId] = maxval;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // now the 0th thread reduces the maxvals in shared memory, i.e. across warps
    if (tid == 0) {
        float val = maxvals[tid];
        // #pragma unroll
        // MSL compiler cannot pragma_unroll<sgInTg>: unknown sgInTg used as constant
        // to make use of anyway set to fixed, e.g. 15/ 31 for block sizes 512/ 1024
        // and call kernel with respective block size.
        // uint i = 1;
        // pragma_unroll<sgInTg>::call(loop1, i, val, maxvals);
        for (uint i = 1; i < sgInTg; i++) {
            val = fmax(val, maxvals[i]);
        }
        // store the final max in the first position
        maxvals[0] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // broadcast the max to all threads
    float offset = maxvals[0];

    // compute expf and write the result to global memory
    // + thread coarsening for sum
    float sumval = 0.0f;
    for (uint i = tid; i < V; i += tgSize * UNROLL_FACTOR) {
        float reg_array[UNROLL_FACTOR];
        // #pragma unroll
        uint u = 0;
        pragma_unroll<UNROLL_FACTOR>::call(loop2, u, i, V, tgSize, x, reg_array);
//        for (int u = 0; u < UNROLL_FACTOR; u++) {
//            reg_array[u] = x[min(V - 1, i + u * tgSize)];
//        }
        // #pragma unroll
        u = 0;
        pragma_unroll<UNROLL_FACTOR>::call(loop3, u, i, V, tgSize, y, reg_array, offset, sumval);
//        for (int u = 0; u < UNROLL_FACTOR; u++) {
//            if (i + u * tgSize < V) {
//                float output = precise::exp(reg_array[u] - offset);
//                y[min(V - 1, i + u * tgSize)] = output; // compiler likes redundant min()?!
//                sumval += output; // combined into the same loop unlike kernel3
//            }
//        }
    }

    // okay now we calculated exp(x - max(x))
    // step 2: sum all the values and divide by the sum

    // within-warp reduction for sumval
    sumval = simd_sum(sumval);
    // write sumval to shared memory
    if (laneId == 0) sumvals[warpId] = sumval;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // inter-thread reduction of sum
    if (tid == 0) {
        float val = sumvals[tid];
        // #pragma unroll
        // MSL compiler cannot pragma_unroll<sgInTg>: unknown sgInTg used as constant
        // uint i = 1;
        // pragma_unroll<sgInTg>::call(loop4, i, val, sumvals);
        for (uint i = 1; i < sgInTg; ++i) {
            val += sumvals[i];
        }
        sumvals[0] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // broadcast the sum to all threads
    float sum = sumvals[0];

    // divide the whole row by the sum
    for (uint i = tid; i < V; i += tgSize * UNROLL_FACTOR) {
        float reg_array[UNROLL_FACTOR];
        // #pragma unroll
        uint u = 0;
        pragma_unroll<UNROLL_FACTOR>::call(loop5, u, i, V, tgSize, y, reg_array);
//        for (int u = 0; u < UNROLL_FACTOR; u++) {
//            reg_array[u] = y[min(V - 1, i + u * tgSize)];
//        }
        // #pragma unroll
        u = 0;
        pragma_unroll<UNROLL_FACTOR>::call(loop6, u, i, V, tgSize, y, reg_array, sum);
//        for (int u = 0; u < UNROLL_FACTOR; u++) {
//            if (i + u * tgSize < V) {
//                y[i + u * tgSize] = reg_array[u] / sum;
//            }
//        }
    }
}

kernel void softmax_forward_kernel8(device float* out [[ buffer(0) ]],
                                device float* inp [[ buffer(1) ]],
                                constant int& BT [[ buffer(2) ]],
                                constant uint& V [[ buffer(3) ]],
                                constant uint& Vp [[ buffer(4) ]],
                                // for max thread ID check if nonuniform threadgroups not available
                                // uint idx [[ thread_position_in_grid ]], // CUDA blockIdx * blockDim + threadIdx
                                uint tgid [[ threadgroup_position_in_grid ]], // CUDA blockIdx
                                uint tid [[ thread_position_in_threadgroup ]], // CUDA threadIdx
                                uint laneId [[ thread_index_in_simdgroup ]], // CUDA threadIdx % 32
                                // for simdReduceMax and simdReduceSum
                                uint sgSize [[ threads_per_simdgroup ]], // CUDA warp size
                                uint warpId [[ simdgroup_index_in_threadgroup ]], // CUDA threadIdx / 32
                                uint sgInTg [[ simdgroups_per_threadgroup ]]) { // CUDA blockDim / 32
    // uncomment if nonuniform threadgroups not available
    // if (idx >= BT * 32) { return; }

    // online softmax paper: http://arxiv.org/abs/1805.02867
    // online softmax reduces loops from 3 to 2
    // which is done by calculating sumval and maxval in one loop

    if (tid >= V) {
        return;
    }

    // one warp one row
    int row = tgid * sgInTg + warpId;

    if (row >= BT) {
        return;
    }

    const device float* x = inp + row * Vp;
    device float* const y = out + row * Vp;

    // merge calculating maxval and sumval in one loop
    // which is an arithmetic improvment from online softmax over normal softmax
    float maxval = -INFINITY, sumval = 0.0f, bigger;
    for (uint i = laneId; i < V; i += sgSize) {
        // when updating the maxval, dynamically updates the previous sumval by
        // multiplying e^{previous_maxval - current_maxval}
        bigger = fmax(maxval, x[i]);
        sumval = sumval * precise::exp(maxval - bigger) + precise::exp(x[i] - bigger);
        maxval = bigger;
    }

    // use warp functions instead of cooperative groups for better readibility
    // calculate the warp wised maxval and sumval
    float offsetMaxval, offsetSumval;
    for (int offset = sgSize / 2; offset > 0; offset >>= 1) {
        simdgroup_barrier(mem_flags::mem_none);
        offsetMaxval = simd_shuffle_down(maxval, offset);
        offsetSumval = simd_shuffle_down(sumval, offset);
        if (offsetMaxval > maxval) {
            sumval *= precise::exp(maxval - offsetMaxval);
            maxval = offsetMaxval;
        } else {
            offsetSumval *= precise::exp(offsetMaxval - maxval);
        }
        sumval += offsetSumval;
    }

    // sync the warp wised maxval and sumval
    // which are also the maxval and sumval of one row in V
    maxval = simd_shuffle(maxval, 0);
    sumval = simd_shuffle(sumval, 0);

    for (uint i = laneId; i < V; i += sgSize) {
        y[i] = precise::exp(x[i] - maxval) / sumval;
    }
}

// --- gelu_forward.metal
// #include <metal_stdlib>
// using namespace metal;

#define GELU_SCALING_FACTOR precise::sqrt(2.0f / M_PI_F)
kernel void gelu_forward_kernel1(device float* out [[ buffer(0) ]],
                                device float* inp [[ buffer(1) ]],
                                constant int& N [[ buffer(2) ]],
                                uint idx [[ thread_position_in_grid ]]) {
    // uncomment if nonuniform threadgroups not available
    // if (idx >= N) { return; }

    float xi = inp[idx];
    float cube = 0.044715f * xi * xi * xi;
    // precise due to fast math. without yields NANs in `gpt2_forward´ (but standalone ok).
    out[idx] = 0.5f * xi * (1.0f + precise::tanh(GELU_SCALING_FACTOR * (xi + cube)));
}

kernel void gelu_forward_kernel2(device float* out [[ buffer(0) ]],
                                device float* inp [[ buffer(1) ]],
                                constant int& N [[ buffer(2) ]],
                                uint idx [[ thread_position_in_grid ]]) {
    int idx_packed_float4 = idx * 4; // packed_float4::size == 4
    // uncomment if nonuniform threadgroups not available
    // if (idx_packed_float4 >= N) { return; }

    packed_float4 packed_out;
    const packed_float4 packed_inp(((device packed_float4*)(inp + idx_packed_float4))[0]);

    float xi = packed_inp[0];
    float cube = 0.044715f * xi * xi * xi;
    packed_out[0] = 0.5f * xi * (1.0f + precise::tanh(GELU_SCALING_FACTOR * (xi + cube)));
    xi = packed_inp[1];
    cube = 0.044715f * xi * xi * xi;
    packed_out[1] = 0.5f * xi * (1.0f + precise::tanh(GELU_SCALING_FACTOR * (xi + cube)));
    xi = packed_inp[2];
    cube = 0.044715f * xi * xi * xi;
    packed_out[2] = 0.5f * xi * (1.0f + precise::tanh(GELU_SCALING_FACTOR * (xi + cube)));
    xi = packed_inp[3];
    cube = 0.044715f * xi * xi * xi;
    packed_out[3] = 0.5f * xi * (1.0f + precise::tanh(GELU_SCALING_FACTOR * (xi + cube)));

    ((device packed_float4*)(out + idx_packed_float4))[0] = packed_out;
}

// --- residual_forward.metal
// #include <metal_stdlib>
// using namespace metal;

kernel void residual_forward_kernel1(device float* out  [[ buffer(0) ]],
                                device float* inp1 [[ buffer(1) ]],
                                device float* inp2 [[ buffer(2) ]],
                                constant int& N [[ buffer(3) ]],
                                uint idx [[ thread_position_in_grid ]]) {
    // uncomment if nonuniform threadgroups not available
    // if (idx >= N) { return; }

    out[idx] = inp1[idx] + inp2[idx];
}

kernel void residual_forward_kernel2(device float* out  [[ buffer(0) ]],
                                device float* inp1 [[ buffer(1) ]],
                                device float* inp2 [[ buffer(2) ]],
                                constant int& N [[ buffer(3) ]],
                                uint idx [[ thread_position_in_grid ]]) {
    int idx_packed_float4 = idx * 4; // packed_float4::size == 4
    // uncomment if nonuniform threadgroups not available
    // if (idx_packed_float4 >= N) { return; }

    packed_float4 packed_out;
    const packed_float4 packed_inp1(((device packed_float4*)(inp1 + idx_packed_float4))[0]);
    const packed_float4 packed_inp2(((device packed_float4*)(inp2 + idx_packed_float4))[0]);

    packed_out[0] = packed_inp1[0] + packed_inp2[0];
    packed_out[1] = packed_inp1[1] + packed_inp2[1];
    packed_out[2] = packed_inp1[2] + packed_inp2[2];
    packed_out[3] = packed_inp1[3] + packed_inp2[3];

    ((device packed_float4*)(out + idx_packed_float4))[0] = packed_out;
}

// --- attention_forward.metal
// #include <metal_stdlib>
// using namespace metal;

kernel void attention_query_key_kernel1(device float* preatt  [[ buffer(0) ]],
                                device float* inp [[ buffer(1) ]],
                                constant int& B  [[ buffer(2) ]],
                                constant int& T  [[ buffer(3) ]],
                                constant int& C  [[ buffer(4) ]],
                                constant int& NH [[ buffer(5) ]],
                                uint idx [[ thread_position_in_grid ]]) {
    // uncomment if nonuniform threadgroups not available
    // if (idx >= B * NH * T * T) { return; }

    int t2 = idx % T;
    int t = (idx / T) % T;
    if (t2 > t) {
        // autoregressive mask
        preatt[idx] = -INFINITY;
        return;
    }
    int h = (idx / (T * T)) % NH;
    int b = idx / (NH * T * T);

    int C3 = C*3;
    int hs = C / NH; // head size
    const device float* query_t = inp + b * T * C3 + t * C3 + h * hs;
    const device float* key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key

    // (query_t) dot (key_t2)
    float val = 0.0f;
    for (int i = 0; i < hs; i++) {
        val += query_t[i] * key_t2[i];
    }
    val *= 1.0f / precise::sqrt((float)hs);

    preatt[idx] = val;
}

kernel void attention_softmax_kernel1(device float* att  [[ buffer(0) ]],
                                device float* preatt [[ buffer(1) ]],
                                constant int& B  [[ buffer(2) ]],
                                constant int& T  [[ buffer(3) ]],
                                constant int& NH [[ buffer(4) ]],
                                uint idx [[ thread_position_in_grid ]]) {
    // uncomment if nonuniform threadgroups not available
    // if (idx >= B * T * NH) { return; }

    int h = idx % NH;
    int t = (idx / NH) % T;
    int b = idx / (NH * T);

    const device float* preatt_bth = preatt + b*NH*T*T + h*T*T + t*T;
    device float* att_bth = att + b*NH*T*T + h*T*T + t*T;

    // find maxval
    float maxval = -FLT_MAX;
    for (int t2 = 0; t2 <= t; t2++) {
        if (preatt_bth[t2] > maxval) {
            maxval = preatt_bth[t2];
        }
    }

    // calculate the exp and keep track of sum
    float expsum = 0.0f;
    for (int t2 = 0; t2 <= t; t2++) {
        float expv = precise::exp(preatt_bth[t2] - maxval);
        expsum += expv;
        att_bth[t2] = expv;
    }
    float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

    // normalize to get the softmax
    for (int t2 = 0; t2 < T; t2++) {
        if (t2 <= t) {
            att_bth[t2] *= expsum_inv;
        } else {
            // causal attention mask. not strictly necessary to set to zero here
            // only doing this explicitly for debugging and checking to PyTorch
            att_bth[t2] = 0.0f;
        }
    }
}

kernel void attention_value_kernel1(device float* out [[ buffer(0) ]],
                                device float* att [[ buffer(1) ]],
                                device float* inp [[ buffer(2) ]],
                                constant int& B  [[ buffer(3) ]],
                                constant int& T  [[ buffer(4) ]],
                                constant int& C  [[ buffer(5) ]],
                                constant int& NH [[ buffer(6) ]],
                                uint idx [[ thread_position_in_grid ]]) {
    // uncomment if nonuniform threadgroups not available
    // if (idx >= B * T * NH) { return; }

    int h = idx % NH;
    int t = (idx / NH) % T;
    int b = idx / (NH * T);

    int C3 = C*3;
    int hs = C / NH; // head size

    device float* out_bth = out + b * T * C + t * C + h * hs;
    const device float* att_bth = att + b*NH*T*T + h*T*T + t*T;

    for (int i = 0; i < hs; i++) { out_bth[i] = 0.0f; }
    for (int t2 = 0; t2 <= t; t2++) {
       const device float* value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C*2; // +C*2 because it's value
        float att_btht2 = att_bth[t2];
        for (int i = 0; i < hs; i++) {
            out_bth[i] += att_btht2 * value_t2[i];
        }
    }
}

// --- matmul_forward.metal
// #include <metal_stdlib>
// using namespace metal;

kernel void matmul_forward_kernel1(device float* out [[ buffer(0) ]],
                                device float* inp  [[ buffer(1) ]],
                                device float* weight [[ buffer(2) ]],
                                constant int& BT [[ buffer(3) ]],
                                constant int& C  [[ buffer(4) ]],
                                constant int& OC [[ buffer(5) ]],
                                uint idx [[ thread_position_in_grid ]]) {
    // uncomment if nonuniform threadgroups not available
    // if (idx >= BT * OC) { return; }

    int bt = idx / OC;
    int oc = idx % OC;
    float val = 0.0f;
    for (int i = 0; i < C; i++) {
        val += inp[bt * C + i] * weight[oc * C + i];
    }
    out[bt * OC + oc] = val;
}

kernel void add_bias_kernel1(device float* out [[ buffer(0) ]],
                                device float* bias  [[ buffer(1) ]],
                                constant int& BT [[ buffer(2) ]],
                                constant int& OC [[ buffer(3) ]],
                                uint idx [[ thread_position_in_grid ]]) {
    // uncomment if nonuniform threadgroups not available
    // if (idx >= BT * OC) { return; }

    out[idx] += bias[idx % OC];
}

// --- layernorm_forward.metal
// #include <metal_stdlib>
// using namespace metal;

kernel void layernorm_forward_kernel1(device float* out [[ buffer(0) ]],
                                device float* mean [[ buffer(1) ]],
                                device float* rstd [[ buffer(2) ]],
                                device float* inp  [[ buffer(3) ]],
                                device float* weight [[ buffer(4) ]],
                                device float* bias [[ buffer(5) ]],
                                constant int& BT [[ buffer(6) ]],
                                constant int& C  [[ buffer(7) ]],
                                uint idx [[ thread_position_in_grid ]]) {
    // uncomment if nonuniform threadgroups not available
    // if (idx >= BT) { return; }

    float eps = 1e-5f;

    // seek to the input position inp[idx,:]
    const device float* x = inp + idx * C;
    // calculate the mean
    float m = 0.0f;
    for (int i = 0; i < C; i++) {
        m += x[i];
    }
    m = m / C;
    // calculate the variance (without any bias correction)
    float v = 0.0f;
    for (int i = 0; i < C; i++) {
        float xshift = x[i] - m;
        v += xshift * xshift;
    }
    v = v / C;
    // calculate the rstd
    float s = 1.0f / sqrt(v + eps);
    // seek to the output position in out[idx,:]
    device float* out_idx = out + idx * C;
    for (int i = 0; i < C; i++) {
        float n = s * (x[i] - m); // normalized output
        float o = n * weight[i] + bias[i]; // scale and shift it
        out_idx[i] = o; // write
    }
    // cache the mean and rstd for the backward pass later
    mean[idx] = m;
    rstd[idx] = s;
}

// --- encoder_forward.metal
// #include <metal_stdlib>
// using namespace metal;

kernel void encoder_forward_kernel1(device float* out [[ buffer(0) ]],
                                device int* inp [[ buffer(1) ]],
                                device float* wte [[ buffer(2) ]],
                                device float* wpe [[ buffer(3) ]],
                                constant int& B [[ buffer(4) ]],
                                constant int& T [[ buffer(5) ]],
                                constant int& C [[ buffer(6) ]],
                                uint idx [[ thread_position_in_grid ]]) {
    // uncomment if nonuniform threadgroups not available
    // if (idx >= B * T) { return; }

    int b = idx / T;
    int t = idx % T;
    device float* out_bt = out + b * T * C + t * C;
    int ix = inp[b * T + t];
    const device float* wte_ix = wte + ix * C;
    const device float* wpe_t = wpe + t * C;
    for (int i = 0; i < C; i++) {
        out_bt[i] = wte_ix[i] + wpe_t[i];
    }
}

kernel void encoder_forward_kernel2(device float* out [[ buffer(0) ]],
                                device int* inp [[ buffer(1) ]],
                                device float* wte [[ buffer(2) ]],
                                device float* wpe [[ buffer(3) ]],
                                constant int& B [[ buffer(4) ]],
                                constant int& T [[ buffer(5) ]],
                                constant int& C [[ buffer(6) ]],
                                uint idx [[ thread_position_in_grid ]]) {
    // uncomment if nonuniform threadgroups not available
    // if (idx >= B * T * C) { return; }

    int bt = idx / C;
    int b = bt / T;
    int t = bt % T;
    int c = idx % C;

    int ix = inp[b * T + t];
    device float* out_btc = out + b * T * C + t * C + c;
    const device float* wte_ix = wte + ix * C + c;
    const device float* wpe_tc = wpe + t * C + c;
    *out_btc = *wte_ix + *wpe_tc;
}

kernel void encoder_forward_kernel3(device float* out [[ buffer(0) ]],
                                device int* inp [[ buffer(1) ]],
                                device float* wte [[ buffer(2) ]],
                                device float* wpe [[ buffer(3) ]],
                                constant int& B [[ buffer(4) ]],
                                constant int& T [[ buffer(5) ]],
                                constant int& C [[ buffer(6) ]],
                                uint idx [[ thread_position_in_grid ]]) {
    int idx_packed_float4 = idx * 4; // packed_float4::size == 4
    // uncomment if nonuniform threadgroups not available
    // if (idx_packed_float4 >= B * T * C) { return; }

    int bt = idx_packed_float4 / C;
    int b = bt / T;
    int t = bt % T;
    int c = idx_packed_float4 % C;

    int ix = inp[b * T + t];

    device float* out_btc = out + b * T * C + t * C + c;
    const device float* wte_ix = wte + ix * C + c;
    const device float* wpe_tc = wpe + t * C + c;

    packed_float4 packed_out;
    const packed_float4 wte2(((device packed_float4*)(wte_ix))[0]);
    const packed_float4 wpe2(((device packed_float4*)(wpe_tc))[0]);

    packed_out[0] = wte2[0] + wpe2[0];
    packed_out[1] = wte2[1] + wpe2[1];
    packed_out[2] = wte2[2] + wpe2[2];
    packed_out[3] = wte2[3] + wpe2[3];

    ((device packed_float4*)(out_btc))[0] = packed_out;
}
