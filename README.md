# EdgeVision: C++17 & TensorRT Real-Time Object Detection & Tracking

**EdgeVision** is a high-performance, zero-Python-dependency C++17 inference and multi-object tracking pipeline built with **NVIDIA TensorRT 10** and **CUDA**. It migrates a standard Python/PyTorch YOLOv8 + ByteTrack workflow to production-grade C++ and profiles each pipeline stage to identify real hardware bottlenecks.

## Tech Stack
* **Language & Build:** C++17 (`-O3` Release), CMake
* **Inference & GPU:** NVIDIA TensorRT 10.9 (FP16), CUDA 12.8 (`enqueueV3`, CUDA Streams)
* **Vision & Tracking:** OpenCV 4.6, Pure C++ ByteTrack (OpenCV `KalmanFilter` + Two-Stage IoU Matching)
* **Model:** YOLOv8n (`.pt` -> `.onnx` -> `.engine`)

## Hardware & Environment
* **GPU:** NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM)
* **CPU / RAM:** AMD Ryzen 7 / 32 GB RAM
* **OS:** Ubuntu 24.04 (WSL2)

## Benchmark Results (581 Frames, 15 Warmup Frames)

| Configuration | Mean Latency | P50 Latency | P95 Latency | Mean FPS |
| :--- | :---: | :---: | :---: | :---: |
| Python + PyTorch + ByteTrack (Baseline) | 12.16 ms | 11.98 ms | 14.98 ms | 82.22 FPS |
| **C++17 (-O3) + TensorRT FP16 + C++ ByteTrack** | **6.62 ms** | **6.46 ms** | **7.75 ms** | **151.12 FPS** |
| Python + TensorRT FP16 + ByteTrack | 5.14 ms | 5.21 ms | 6.96 ms | 194.41 FPS |

### C++17 Stage-by-Stage Profiling
* **Preprocess (CPU Letterbox + Blob):** `2.26 ms`
* **Inference (GPU TensorRT FP16 + Memcpy):** `3.63 ms`
* **Postprocess (Decode + NMS):** `0.71 ms`
* **ByteTrack (Pure C++ Kalman + IoU):** `0.02 ms`

> **Engineering Takeaway:** Rewriting a pipeline in C++ does not automatically guarantee the fastest throughput. While C++ postprocessing (`0.71 ms`) and tracking (`0.02 ms`) run at near-zero overhead, CPU-based preprocessing (`2.26 ms`) acts as the primary bottleneck compared to Ultralytics' GPU-accelerated tensor preprocessing.

## Build & Run
```bash
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
./edge_vision
