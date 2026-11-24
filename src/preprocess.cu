#include "preprocess.cuh"
#include <cmath>

__global__ void letterbox_bilinear_kernel(
    const uint8_t* __restrict__ src, int src_w, int src_h,
    float* __restrict__ dst, int dst_w, int dst_h,
    float scale, int pad_x, int pad_y, int new_w, int new_h
) {
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;

    if (dx >= dst_w || dy >= dst_h) return;

    float r = 114.0f / 255.0f;
    float g = 114.0f / 255.0f;
    float b = 114.0f / 255.0f;

    if (dx >= pad_x && dx < pad_x + new_w && dy >= pad_y && dy < pad_y + new_h) {
        float fx = (dx - pad_x + 0.5f) / scale - 0.5f;
        float fy = (dy - pad_y + 0.5f) / scale - 0.5f;

        fx = fmaxf(0.0f, fminf(fx, static_cast<float>(src_w - 1)));
        fy = fmaxf(0.0f, fminf(fy, static_cast<float>(src_h - 1)));

        int x0 = static_cast<int>(floorf(fx));
        int y0 = static_cast<int>(floorf(fy));
        int x1 = min(x0 + 1, src_w - 1);
        int y1 = min(y0 + 1, src_h - 1);

        float wx = fx - x0;
        float wy = fy - y0;

        float w00 = (1.0f - wx) * (1.0f - wy);
        float w01 = wx * (1.0f - wy);
        float w10 = (1.0f - wx) * wy;
        float w11 = wx * wy;

        int idx00 = (y0 * src_w + x0) * 3;
        int idx01 = (y0 * src_w + x1) * 3;
        int idx10 = (y1 * src_w + x0) * 3;
        int idx11 = (y1 * src_w + x1) * 3;

        // OpenCV BGR -> RGB donusumu ve 0-1 normalizasyonu
        b = (w00 * src[idx00 + 0] + w01 * src[idx01 + 0] + w10 * src[idx10 + 0] + w11 * src[idx11 + 0]) / 255.0f;
        g = (w00 * src[idx00 + 1] + w01 * src[idx01 + 1] + w10 * src[idx10 + 1] + w11 * src[idx11 + 1]) / 255.0f;
        r = (w00 * src[idx00 + 2] + w01 * src[idx01 + 2] + w10 * src[idx10 + 2] + w11 * src[idx11 + 2]) / 255.0f;
    }

    int area = dst_w * dst_h;
    int out_idx = dy * dst_w + dx;
    dst[0 * area + out_idx] = r;
    dst[1 * area + out_idx] = g;
    dst[2 * area + out_idx] = b;
}

void launch_letterbox_kernel(
    const uint8_t* d_src, int src_w, int src_h,
    float* d_dst, int dst_w, int dst_h,
    float scale, int pad_x, int pad_y, int new_w, int new_h,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y);
    letterbox_bilinear_kernel<<<grid, block, 0, stream>>>(
        d_src, src_w, src_h, d_dst, dst_w, dst_h, scale, pad_x, pad_y, new_w, new_h
    );
}