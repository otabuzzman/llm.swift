let defaultLibrary = """
#include <metal_stdlib>
using namespace metal;

kernel void encoder_forward_kernel1(device float* out [[ buffer(0) ]],
                                device int* inp [[ buffer(1) ]],
                                device float* wte [[ buffer(2) ]],
                                device float* wpe [[ buffer(3) ]],
                                constant uint& B [[ buffer(4) ]],
                                constant uint& T [[ buffer(5) ]],
                                constant uint& C [[ buffer(6) ]],
                                uint xid [[ thread_position_in_grid ]]) {
    if (xid >= B * T) { return; }

    int b = xid / T;
    int t = xid % T;
    device float* out_bt = out + b * T * C + t * C;
    int ix = inp[b * T + t];
    const device float* wte_ix = wte + ix * C;
    const device float* wpe_t = wpe + t * C;
    for (int i = 0; i < C; i++) {
        out_bt[i] = (float)((float)wte_ix[i] + (float)wpe_t[i]);
    }
}

kernel void encoder_forward_kernel2(device float* out [[ buffer(0) ]],
                                device int* inp [[ buffer(1) ]],
                                device float* wte [[ buffer(2) ]],
                                device float* wpe [[ buffer(3) ]],
                                constant uint& B [[ buffer(4) ]],
                                constant uint& T [[ buffer(5) ]],
                                constant uint& C [[ buffer(6) ]],
                                uint xid [[ thread_position_in_grid ]]) {
    if (xid >= B * T * C) { return; }

    int bt = xid / C;
    int b = bt / T;
    int t = bt % T;
    int c = xid % C;

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
                                uint xid [[ thread_position_in_grid ]]) {
    int xid_packed_float4 = xid * 4; // packed_float4::size == 4
    if (xid_packed_float4 >= B * T * C) { return; }

    int bt = xid_packed_float4 / C;
    int b = bt / T;
    int t = bt % T;
    int c = xid_packed_float4 % C;

    int ix = inp[b * T + t];

    device float* out_btc = out + b * T * C + t * C + c;
    const device float* wte_ix = wte + ix * C + c;
    const device float* wpe_tc = wpe + t * C + c;

    packed_float4 packed_out;
    const packed_float4 wte2(((device packed_float4*)(wte_ix))[0]);
    const packed_float4 wpe2(((device packed_float4*)(wpe_tc))[0]);

    packed_out[0] = ((float)wte2[0] + (float)wpe2[0]);
    packed_out[1] = (float)((float)wte2[1] + (float)wpe2[1]);
    packed_out[2] = (float)((float)wte2[2] + (float)wpe2[2]);
    packed_out[3] = (float)((float)wte2[3] + (float)wpe2[3]);

    ((device packed_float4*)(out_btc))[0] = packed_out;
}
"""
