# EdgeVision: C++17 & TensorRT Real-Time Object Detection & Tracking

**EdgeVision**, Python/PyTorch ortamında çalışan YOLOv8 + ByteTrack nesne tespit ve takip hattının, üretim (production) ortamına uygun, sıfır Python bağımlılıklı **C++17, CUDA ve NVIDIA TensorRT** mimarisine taşınması ve bilimsel olarak profillenmesi projesidir.

## Teknoloji Yığını
* **Dil & Derleme:** C++17 (`-O3` Release), CMake
* **Inference & GPU:** NVIDIA TensorRT 10.9 (FP16), CUDA 12.8 (`enqueueV3`, CUDA Streams)
* **Görüntü İşleme & Takip:** OpenCV 4.6, Saf C++ ByteTrack (OpenCV `KalmanFilter` + İki Aşamalı IoU Eşleştirme)
* **Model:** YOLOv8n (`.pt` -> `.onnx` -> `.engine`)

## Donanım & Test Ortamı
* **GPU:** NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM)
* **CPU / RAM:** AMD Ryzen 7 / 32 GB RAM
* **OS:** Ubuntu 24.04 (WSL2)

## Benchmark Sonuçları (581 Kare, 15 Kare Warmup)

| Konfigürasyon | Ortalama Süre | P50 Latency | P95 Latency | Ortalama FPS |
| :--- | :---: | :---: | :---: | :---: |
| Python + PyTorch + ByteTrack | 12.16 ms | 11.98 ms | 14.98 ms | 82.22 FPS |
| **C++17 (-O3) + TensorRT FP16 + C++ ByteTrack** | **6.62 ms** | **6.46 ms** | **7.75 ms** | **151.12 FPS** |
| Python + TensorRT FP16 + ByteTrack | 5.14 ms | 5.21 ms | 6.96 ms | 194.41 FPS |

### C++17 Aşama Bazlı Darboğaz Analizi
* **Preprocess (CPU Letterbox + Blob):** `2.26 ms`
* **Inference (GPU TensorRT FP16 + Memcpy):** `3.63 ms`
* **Postprocess (Decode + NMS):** `0.71 ms`
* **ByteTrack (C++ Kalman + IoU):** `0.02 ms`

> **Mühendislik Çıkarımı:** Sadece C++'a geçmek otomatik olarak en hızlı sonucu vermez. C++ hattında `Postprocess` (0.71 ms) ve `ByteTrack` (0.02 ms) neredeyse sıfır maliyetle çalışırken, CPU tabanlı ön işleme (`2.26 ms`) ana darboğaz olarak tespit edilmiştir. V2 sürümünde CUDA Kernel tabanlı GPU Preprocess ve Pinned Memory ile 200+ FPS hedeflenmektedir.

## Derleme ve Çalıştırma
```bash
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
./edge_vision