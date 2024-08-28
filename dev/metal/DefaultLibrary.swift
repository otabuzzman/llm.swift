let defaultLibrary = """
#include <metal_stdlib>
using namespace metal;

kernel void encoder_forward(device float* out [[ buffer(0) ]],
                                device int* inp [[ buffer(1) ]],
                                device float* wte [[ buffer(2) ]],
                                device float* wpe [[ buffer(3) ]],
                                constant uint& B [[ buffer(4) ]],
                                constant uint& T [[ buffer(5) ]],
                                constant uint& C [[ buffer(6) ]],
                                uint tid [[ thread_position_in_grid ]]) {
    if (tid >= B * T * C) { return; }

    int bt = tid / C;
    int b = bt / T;
    int t = bt % T;
    int c = tid % C;

    int ix = inp[b * T + t];
    device float* out_btc = out + b * T * C + t * C + c;
    device float* wte_ix = wte + ix * C + c;
    device float* wpe_tc = wpe + t * C + c;
    *out_btc = *wte_ix + *wpe_tc;
}
"""
