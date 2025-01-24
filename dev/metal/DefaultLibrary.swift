let defaultLibrary = """
#include <metal_stdlib>
using namespace metal;

// --- crossentropy_forward.metal
// #include <metal_stdlib>
// using namespace metal;

kernel void crossentropy_forward_kernel1(device float* losses [[ buffer(0) ]],
                                device float* probs [[ buffer(1) ]],
                                device int* targets [[ buffer(2) ]],
                                constant uint& B [[ buffer(3) ]],
                                constant uint& T [[ buffer(4) ]],
                                constant uint& V [[ buffer(5) ]],
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
                                constant uint& N [[ buffer(2) ]],
                                constant uint& C [[ buffer(3) ]],
                                uint idx [[ thread_position_in_grid ]]) {
    // uncomment if nonuniform threadgroups not available
    // if (idx >= N) { return; }

    const device float* inp_row = inp + idx * C;
    device float* out_row = out + idx * C;

    float maxval = -INFINITY;
    for (int j = 0; j < C; j++) {
        if (inp_row[j] > maxval) {
            maxval = inp_row[j];
        }
    }
    float sum = 0.0;
    for (int j = 0; j < C; j++) {
        out_row[j] = exp(inp_row[j] - maxval);
        sum += out_row[j];
    }
    for (int j = 0; j < C; j++) {
        out_row[j] /= sum;
    }
}

// --- gelu_forward.metal
// #include <metal_stdlib>
// using namespace metal;

#define GELU_SCALING_FACTOR sqrt(2.0f / M_PI_F)
kernel void gelu_forward_kernel1(device float* out [[ buffer(0) ]],
                                device float* inp [[ buffer(1) ]],
                                constant uint& N [[ buffer(2) ]],
                                uint idx [[ thread_position_in_grid ]]) {
    // uncomment if nonuniform threadgroups not available
    // if (idx >= N) { return; }

    float xi = inp[idx];
    float cube = 0.044715f * xi * xi * xi;
    // precise due to -ffast-math. without yields NANs in `gpt2_forward´ (but standalone ok).
    out[idx] = 0.5f * xi * (1.0f + precise::tanh(GELU_SCALING_FACTOR * (xi + cube)));
}

kernel void gelu_forward_kernel2(device float* out [[ buffer(0) ]],
                                device float* inp [[ buffer(1) ]],
                                constant uint& N [[ buffer(2) ]],
                                uint idx [[ thread_position_in_grid ]]) {
    int idx_packed_float4 = idx * 4; // packed_float4::size == 4
    // uncomment if nonuniform threadgroups not available
    // if (idx_packed_float4 >= N) { return; }


    packed_float4 packed_out;
    const packed_float4 packed_inp(((device packed_float4*)(inp + idx_packed_float4))[0]);

    float xi = packed_inp[0];
    float cube = 0.044715f * xi * xi * xi;
    packed_out[0] = 0.5f * xi * (1.0f + tanh(GELU_SCALING_FACTOR * (xi + cube)));
    xi = packed_inp[1];
    cube = 0.044715f * xi * xi * xi;
    packed_out[1] = 0.5f * xi * (1.0f + tanh(GELU_SCALING_FACTOR * (xi + cube)));
    xi = packed_inp[2];
    cube = 0.044715f * xi * xi * xi;
    packed_out[2] = 0.5f * xi * (1.0f + tanh(GELU_SCALING_FACTOR * (xi + cube)));
    xi = packed_inp[3];
    cube = 0.044715f * xi * xi * xi;
    packed_out[3] = 0.5f * xi * (1.0f + tanh(GELU_SCALING_FACTOR * (xi + cube)));

    ((device packed_float4*)(out + idx_packed_float4))[0] = packed_out;
}

// --- residual_forward.metal
// #include <metal_stdlib>
// using namespace metal;

kernel void residual_forward_kernel1(device float* out  [[ buffer(0) ]],
                                device float* inp1 [[ buffer(1) ]],
                                device float* inp2 [[ buffer(2) ]],
                                constant uint& N [[ buffer(3) ]],
                                uint idx [[ thread_position_in_grid ]]) {
    // uncomment if nonuniform threadgroups not available
    // if (idx >= N) { return; }

    out[idx] = inp1[idx] + inp2[idx];
}

kernel void residual_forward_kernel2(device float* out  [[ buffer(0) ]],
                                device float* inp1 [[ buffer(1) ]],
                                device float* inp2 [[ buffer(2) ]],
                                constant uint& N [[ buffer(3) ]],
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
                                constant uint& B  [[ buffer(2) ]],
                                constant uint& T  [[ buffer(3) ]],
                                constant uint& C  [[ buffer(4) ]],
                                constant uint& NH [[ buffer(5) ]],
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
    val *= 1.0 / sqrt((float)hs);

    preatt[idx] = val;
}

kernel void attention_softmax_kernel1(device float* att  [[ buffer(0) ]],
                                device float* preatt [[ buffer(1) ]],
                                constant uint& B  [[ buffer(2) ]],
                                constant uint& T  [[ buffer(3) ]],
                                constant uint& NH [[ buffer(4) ]],
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
        float expv = exp(preatt_bth[t2] - maxval);
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
                                constant uint& B  [[ buffer(3) ]],
                                constant uint& T  [[ buffer(4) ]],
                                constant uint& C  [[ buffer(5) ]],
                                constant uint& NH [[ buffer(6) ]],
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
// swiftlint:disable:this file_length
