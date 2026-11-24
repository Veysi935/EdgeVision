#pragma once

#include <string>
#include <vector>
#include <memory>
#include <cstdint>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>

struct Detection {
    cv::Rect box;
    float confidence;
    int class_id;
};

struct StageTimings {
    double preprocess_ms = 0.0;
    double inference_ms = 0.0;
    double postprocess_ms = 0.0;
};

class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
};

class YoloTRT {
public:
    YoloTRT(const std::string& engine_path, float conf_thresh = 0.25f, float nms_thresh = 0.45f);
    ~YoloTRT();

    std::vector<Detection> detect(const cv::Mat& frame, StageTimings& timings);

private:
    TrtLogger logger_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    cudaStream_t stream_ = nullptr;

    // CUDA Events (GPU surelerini pipeline'i durdurmadan olcmek icin)
    cudaEvent_t ev_start_ = nullptr;
    cudaEvent_t ev_pre_end_ = nullptr;
    cudaEvent_t ev_inf_end_ = nullptr;

    // Pinned Host & Device bellekleri
    uint8_t* h_input_pinned_ = nullptr;
    uint8_t* d_raw_img_ = nullptr;
    size_t raw_img_capacity_ = 0;

    void* d_input_ = nullptr;
    void* d_output_ = nullptr;
    float* h_output_pinned_ = nullptr;

    const int input_w_ = 640;
    const int input_h_ = 640;
    const int num_classes_ = 80;
    const int num_anchors_ = 8400;

    float conf_thresh_;
    float nms_thresh_;
};