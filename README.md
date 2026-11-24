# EdgeVision: C++17, CUDA & TensorRT Real-Time Object Detection & Tracking

**EdgeVision** is a high-performance, zero-Python-dependency C++17 inference and multi-object tracking pipeline built with **NVIDIA TensorRT 10** and **CUDA**. It migrates a standard Python/PyTorch YOLOv8 + ByteTrack workflow to production-grade C++, profiles each pipeline stage to uncover hardware bottlenecks, and eliminates CPU overhead using custom CUDA kernels and pinned host memory.

## Tech Stack
- **Language & Build:** C++17 (`-O3` Release), CUDA C++, CMake
- **Inference & GPU:** NVIDIA TensorRT 10.9 (FP16), CUDA 12.8 (`enqueueV3`, CUDA Streams, CUDA Events)
- **GPU Optimization (V2):** Custom CUDA Bilinear Letterbox Kernel (BGR→RGB, Normalize, HWC→CHW), Pinned Host Memory (`cudaMallocHost`)
- **Vision & Tracking:** OpenCV 4.6, Pure C++ ByteTrack (OpenCV `KalmanFilter` + Two-Stage IoU Matching)
- **Model:** YOLOv8n (`.pt` -> `.onnx` -> `.engine`)

## Hardware & Environment
- **GPU:** NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM)
- **CPU / RAM:** AMD Ryzen 7 / 32 GB RAM
- **OS:** Ubuntu 24.04 (WSL2)

## Benchmark Results (581 Frames, 15 Warmup Frames)

| Configuration | Mean Latency | P50 Latency | P95 Latency | Mean FPS |
| :--- | :---: | :---: | :---: | :---: |
| Python + PyTorch + ByteTrack (Baseline) | 12.16 ms | 11.98 ms | 14.98 ms | 82.22 FPS |
| C++17 V1 (CPU Preprocess) + TensorRT FP16 | 6.62 ms | 6.46 ms | 7.75 ms | 151.12 FPS |
| Python + TensorRT FP16 + ByteTrack | 5.14 ms | 5.21 ms | 6.96 ms | 194.41 FPS |
| **C++17 V2 (CUDA Kernel + Pinned Mem) + TRT FP16** | **3.55 ms** | **3.34 ms** | **4.86 ms** | **281.45 FPS** |

### C++17 Stage-by-Stage Profiling (V1 vs. V2)
- **Preprocess:** `2.26 ms` (CPU OpenCV) → **`0.30 ms`** (Custom CUDA Kernel + Pinned Mem) — *7.5x faster*
- **Inference + Memcpy:** `3.63 ms` → **`2.45 ms`** (TensorRT FP16 + Async `uint8` Transfer)
- **Postprocess (Decode + NMS):** `0.71 ms` → **`0.78 ms`**
- **ByteTrack (Pure C++ Kalman + IoU):** `0.02 ms` → **`0.03 ms`**

> **Engineering Takeaway:** Simply rewriting a pipeline in C++ (V1: `151.12 FPS`) does not automatically outperform a GPU-backed Python wrapper (`194.41 FPS`) if preprocessing remains on the CPU (`2.26 ms`). Moving letterbox resizing, color conversion, and normalization into a single custom CUDA kernel with pinned host memory (V2) eliminated the bottleneck, reaching **`281.45 FPS`** (`3.42x` over PyTorch baseline and `1.45x` over Python + TensorRT) with zero Python dependencies.

## Build & Run
```bash
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
./edge_vision