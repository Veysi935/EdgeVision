import time
import cv2
import numpy as np
from ultralytics import YOLO

# Dosya yolları
MODEL_PATH = "/home/veysi/EdgeVision/models/yolov8n.pt"
VIDEO_PATH = "/home/veysi/EdgeVision/benchmarks/test.mp4"

def main():
    print("Model yükleniyor...")
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"HATA: Video açılamadı -> {VIDEO_PATH}")
        return

    latencies_ms = []
    frame_count = 0
    warmup_frames = 15  # İlk 15 kareyi GPU ısınması için ölçüme katmıyoruz

    print("Benchmark başlıyor (Python + PyTorch + ByteTrack)...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t_start = time.perf_counter()

        # YOLOv8 + ByteTrack uçtan uca çıkarım ve takip
        results = model.track(
            frame,
            tracker="bytetrack.yaml",
            persist=True,
            device=0,      # RTX 4060 GPU
            verbose=False
        )

        t_end = time.perf_counter()
        elapsed_ms = (t_end - t_start) * 1000.0

        frame_count += 1
        if frame_count > warmup_frames:
            latencies_ms.append(elapsed_ms)

    cap.release()

    if not latencies_ms:
        print("Yeterli kare işlenemedi!")
        return

    arr = np.array(latencies_ms)
    avg_ms = np.mean(arr)
    p50_ms = np.percentile(arr, 50)
    p95_ms = np.percentile(arr, 95)
    avg_fps = 1000.0 / avg_ms

    print("\n================ BENCHMARK SONUCU ================")
    print(f"Konfigürasyon : Python + PyTorch + ByteTrack")
    print(f"Ölçülen Kare  : {len(latencies_ms)} (İlk {warmup_frames} kare warmup)")
    print(f"Ortalama Süre : {avg_ms:.2f} ms")
    print(f"P50 Latency   : {p50_ms:.2f} ms")
    print(f"P95 Latency   : {p95_ms:.2f} ms")
    print(f"Ortalama FPS  : {avg_fps:.2f} FPS")
    print("==================================================\n")

if __name__ == "__main__":
    main()