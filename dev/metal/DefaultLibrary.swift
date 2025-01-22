let defaultLibrary = """
#include <metal_stdlib>
using namespace metal;

// --- matmul_forward.metal
// #include <metal_stdlib>
// using namespace metal;

kernel void matmul_forward_kernel1(device float* out [[ buffer(0) ]],
                                device float* inp  [[ buffer(1) ]],
                                device float* weight [[ buffer(2) ]],
                                device float* bias [[ buffer(3) ]],
                                constant uint& BT [[ buffer(4) ]],
                                constant uint& C  [[ buffer(5) ]],
                                constant uint& OC [[ buffer(6) ]],
                                uint idx [[ thread_position_in_grid ]]) {
    // uncomment if nonuniform threadgroups not available
    // if (idx >= BT * OC) { return; }

    int bt = idx / OC;
    int oc = idx % OC;
    float val = bias[oc];
    for (int i = 0; i < C; i++) {
        val += inp[bt * C + i] * weight[oc * C + i];
    }
    out[bt * OC + oc] = val;
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
                                constant uint& B [[ buffer(6) ]],
                                constant uint& T [[ buffer(7) ]],
                                constant uint& C [[ buffer(8) ]],
                                uint idx [[ thread_position_in_grid ]]) {
    // uncomment if nonuniform threadgroups not available
    // if (idx >= B * T) { return; }

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
                                constant uint& B [[ buffer(4) ]],
                                constant uint& T [[ buffer(5) ]],
                                constant uint& C [[ buffer(6) ]],
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
                                constant uint& B [[ buffer(4) ]],
                                constant uint& T [[ buffer(5) ]],
                                constant uint& C [[ buffer(6) ]],
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
                                constant uint& B [[ buffer(4) ]],
                                constant uint& T [[ buffer(5) ]],
                                constant uint& C [[ buffer(6) ]],
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
"""
