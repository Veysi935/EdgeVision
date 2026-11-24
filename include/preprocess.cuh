#pragma once

#include <cuda_runtime.h>
#include <cstdint>

void launch_letterbox_kernel(
    const uint8_t* d_src, int src_w, int src_h,
    float* d_dst, int dst_w, int dst_h,
    float scale, int pad_x, int pad_y, int new_w, int new_h,
    cudaStream_t stream
);