#include "yolo_trt.hpp"
#include <fstream>
#include <iostream>
#include <chrono>
#include <opencv2/dnn.hpp>

YoloTRT::YoloTRT(const std::string& engine_path, float conf_thresh, float nms_thresh)
    : conf_thresh_(conf_thresh), nms_thresh_(nms_thresh) {

    std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
    if (!file.good()) {
        throw std::runtime_error("Engine dosyasi acilamadi: " + engine_path);
    }

    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> engine_data(size);
    file.read(engine_data.data(), size);
    file.close();

    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    engine_.reset(runtime_->deserializeCudaEngine(engine_data.data(), size));
    context_.reset(engine_->createExecutionContext());

    cudaStreamCreate(&stream_);

    // GPU bellek tahsisi (bir kez yapilir, her karede tekrar kullanilir)
    size_t input_bytes = 1 * 3 * input_h_ * input_w_ * sizeof(float);
    size_t output_elements = 1 * (num_classes_ + 4) * num_anchors_;
    size_t output_bytes = output_elements * sizeof(float);

    cudaMalloc(&d_input_, input_bytes);
    cudaMalloc(&d_output_, output_bytes);
    h_output_.resize(output_elements);

    // TensorRT 10 API: Giris ve cikis bellek adreslerini bagla
    context_->setTensorAddress("images", d_input_);
    context_->setTensorAddress("output0", d_output_);
}

YoloTRT::~YoloTRT() {
    if (d_input_) cudaFree(d_input_);
    if (d_output_) cudaFree(d_output_);
    if (stream_) cudaStreamDestroy(stream_);
}

cv::Mat YoloTRT::letterbox(const cv::Mat& src, float& scale, int& pad_x, int& pad_y) {
    scale = std::min((float)input_w_ / src.cols, (float)input_h_ / src.rows);
    int new_w = static_cast<int>(std::round(src.cols * scale));
    int new_h = static_cast<int>(std::round(src.rows * scale));

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(new_w, new_h));

    pad_x = (input_w_ - new_w) / 2;
    pad_y = (input_h_ - new_h) / 2;

    cv::Mat padded(input_h_, input_w_, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(padded(cv::Rect(pad_x, pad_y, new_w, new_h)));
    return padded;
}

std::vector<Detection> YoloTRT::detect(const cv::Mat& frame, StageTimings& timings) {
    using Clock = std::chrono::high_resolution_clock;

    // 1. PREPROCESS
    auto t0 = Clock::now();
    float scale = 1.0f;
    int pad_x = 0, pad_y = 0;
    cv::Mat padded = letterbox(frame, scale, pad_x, pad_y);

    // BGR -> RGB, 0-1 normalize, HWC -> CHW
    cv::Mat blob = cv::dnn::blobFromImage(padded, 1.0 / 255.0, cv::Size(input_w_, input_h_), cv::Scalar(), true, false);
    auto t1 = Clock::now();

    // 2. INFERENCE (Host -> GPU -> TensorRT -> Host)
    size_t input_bytes = 1 * 3 * input_h_ * input_w_ * sizeof(float);
    size_t output_bytes = h_output_.size() * sizeof(float);

    cudaMemcpyAsync(d_input_, blob.ptr<float>(), input_bytes, cudaMemcpyHostToDevice, stream_);
    context_->enqueueV3(stream_);
    cudaMemcpyAsync(h_output_.data(), d_output_, output_bytes, cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);
    auto t2 = Clock::now();

    // 3. POSTPROCESS (Decode [84, 8400] + NMS)
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;

    for (int i = 0; i < num_anchors_; ++i) {
        float max_score = 0.0f;
        int best_class = -1;

        for (int c = 0; c < num_classes_; ++c) {
            float score = h_output_[(4 + c) * num_anchors_ + i];
            if (score > max_score) {
                max_score = score;
                best_class = c;
            }
        }

        if (max_score >= conf_thresh_) {
            float cx = h_output_[0 * num_anchors_ + i];
            float cy = h_output_[1 * num_anchors_ + i];
            float w  = h_output_[2 * num_anchors_ + i];
            float h  = h_output_[3 * num_anchors_ + i];

            int x = static_cast<int>((cx - 0.5f * w - pad_x) / scale);
            int y = static_cast<int>((cy - 0.5f * h - pad_y) / scale);
            int width  = static_cast<int>(w / scale);
            int height = static_cast<int>(h / scale);

            boxes.emplace_back(x, y, width, height);
            confidences.push_back(max_score);
            class_ids.push_back(best_class);
        }
    }

    std::vector<int> nms_indices;
    cv::dnn::NMSBoxes(boxes, confidences, conf_thresh_, nms_thresh_, nms_indices);

    std::vector<Detection> detections;
    detections.reserve(nms_indices.size());
    for (int idx : nms_indices) {
        detections.push_back({boxes[idx], confidences[idx], class_ids[idx]});
    }
    auto t3 = Clock::now();

    timings.preprocess_ms  = std::chrono::duration<double, std::milli>(t1 - t0).count();
    timings.inference_ms   = std::chrono::duration<double, std::milli>(t2 - t1).count();
    timings.postprocess_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    return detections;
}