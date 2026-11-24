#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "yolo_trt.hpp"
#include "bytetrack.hpp"

int main() {
    const std::string engine_path = "/home/veysi/EdgeVision/models/yolov8n_fp16.engine";
    const std::string video_path  = "/home/veysi/EdgeVision/benchmarks/test.mp4";
    const std::string output_path = "/home/veysi/EdgeVision/benchmarks/output_cpp.mp4";

    // ByteTrack dusuk skorlu tespitleri de kullandigi icin conf_thresh=0.10
    YoloTRT detector(engine_path, 0.10f, 0.45f);
    ByteTracker tracker(0.45f, 0.30f, 30);

    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "HATA: Video acilamadi -> " << video_path << std::endl;
        return -1;
    }

    int width  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    cv::VideoWriter writer(output_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(width, height));

    std::vector<double> total_times, pre_times, inf_times, post_times, track_times;
    int frame_count = 0;
    const int warmup_frames = 15;

    std::cout << "Benchmark basliyor (C++17 + TensorRT FP16 + C++ ByteTrack)..." << std::endl;

    cv::Mat frame;
    while (cap.read(frame)) {
        StageTimings t;
        auto detections = detector.detect(frame, t);

        auto t_trk0 = std::chrono::high_resolution_clock::now();
        auto tracks = tracker.update(detections);
        auto t_trk1 = std::chrono::high_resolution_clock::now();
        double track_ms = std::chrono::duration<double, std::milli>(t_trk1 - t_trk0).count();

        double total_ms = t.preprocess_ms + t.inference_ms + t.postprocess_ms + track_ms;
        frame_count++;

        if (frame_count > warmup_frames) {
            total_times.push_back(total_ms);
            pre_times.push_back(t.preprocess_ms);
            inf_times.push_back(t.inference_ms);
            post_times.push_back(t.postprocess_ms);
            track_times.push_back(track_ms);
        }

        // Demo videosu icin kutulari ve ID'leri ciz (benchmark suresine dahil degil)
        for (const auto& obj : tracks) {
            cv::rectangle(frame, obj.box, cv::Scalar(0, 255, 0), 2);
            std::string label = "ID:" + std::to_string(obj.track_id);
            cv::putText(frame, label, cv::Point(obj.box.x, std::max(15, obj.box.y - 5)),
                        cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 255, 0), 2);
        }

        std::string hud = "EdgeVision C++ | TRT FP16 + ByteTrack | " +
                          std::to_string(static_cast<int>(1000.0 / total_ms)) + " FPS (" +
                          std::to_string(total_ms).substr(0, 4) + " ms)";
        cv::putText(frame, hud, cv::Point(15, 30), cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0, 255, 255), 2);
        writer.write(frame);
    }

    cap.release();
    writer.release();

    auto calc_mean = [](const std::vector<double>& v) {
        return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    };

    std::vector<double> sorted_total = total_times;
    std::sort(sorted_total.begin(), sorted_total.end());
    double avg_total = calc_mean(total_times);
    double p50 = sorted_total[sorted_total.size() * 50 / 100];
    double p95 = sorted_total[sorted_total.size() * 95 / 100];
    double avg_fps = 1000.0 / avg_total;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\n================ C++ FULL PIPELINE SONUCU ================" << std::endl;
    std::cout << "Konfigurasyon : C++17 (-O3) + TensorRT FP16 + ByteTrack" << std::endl;
    std::cout << "Olculen Kare  : " << total_times.size() << " (Ilk " << warmup_frames << " kare warmup)" << std::endl;
    std::cout << "----------------------------------------------------------" << std::endl;
    std::cout << "  Preprocess  : " << calc_mean(pre_times) << " ms" << std::endl;
    std::cout << "  Inference   : " << calc_mean(inf_times) << " ms" << std::endl;
    std::cout << "  Postprocess : " << calc_mean(post_times) << " ms" << std::endl;
    std::cout << "  ByteTrack   : " << calc_mean(track_times) << " ms" << std::endl;
    std::cout << "----------------------------------------------------------" << std::endl;
    std::cout << "Ortalama Sure : " << avg_total << " ms" << std::endl;
    std::cout << "P50 Latency   : " << p50 << " ms" << std::endl;
    std::cout << "P95 Latency   : " << p95 << " ms" << std::endl;
    std::cout << "Ortalama FPS  : " << avg_fps << " FPS" << std::endl;
    std::cout << "Demo Videosu  : " << output_path << std::endl;
    std::cout << "==========================================================\n" << std::endl;

    return 0;
}