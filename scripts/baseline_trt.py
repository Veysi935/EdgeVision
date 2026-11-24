import time
import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = "/home/veysi/EdgeVision/models/yolov8n_fp16.engine"
VIDEO_PATH = "/home/veysi/EdgeVision/benchmarks/test.mp4"

def main():
    print("TensorRT Engine Python uzerinden yukleniyor...")
    model = YOLO(MODEL_PATH, task="detect")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"HATA: Video acilamadi -> {VIDEO_PATH}")
        return

    latencies_ms = []
    frame_count = 0
    warmup_frames = 15

    print("Benchmark basliyor (Python + TensorRT FP16 + ByteTrack)...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t_start = time.perf_counter()

        results = model.track(
            frame,
            tracker="bytetrack.yaml",
            persist=True,
            device=0,
            verbose=False
        )

        t_end = time.perf_counter()
        elapsed_ms = (t_end - t_start) * 1000.0

        frame_count += 1
        if frame_count > warmup_frames:
            latencies_ms.append(elapsed_ms)

    cap.release()

    arr = np.array(latencies_ms)
    avg_ms = np.mean(arr)
    p50_ms = np.percentile(arr, 50)
    p95_ms = np.percentile(arr, 95)
    avg_fps = 1000.0 / avg_ms

    print("\n================ BENCHMARK SONUCU ================")
    print(f"Konfigurasyon : Python + TensorRT FP16 + ByteTrack")
    print(f"Olculen Kare  : {len(latencies_ms)} (Ilk {warmup_frames} kare warmup)")
    print(f"Ortalama Sure : {avg_ms:.2f} ms")
    print(f"P50 Latency   : {p50_ms:.2f} ms")
    print(f"P95 Latency   : {p95_ms:.2f} ms")
    print(f"Ortalama FPS  : {avg_fps:.2f} FPS")
    print("==================================================\n")

if __name__ == "__main__":
    main()