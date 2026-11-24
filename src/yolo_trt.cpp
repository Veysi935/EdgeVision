#include "yolo_trt.hpp"
#include "preprocess.cuh"
#include <fstream>
#include <iostream>
#include <chrono>
#include <cstring>
#include <cmath>
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
    cudaEventCreate(&ev_start_);
    cudaEventCreate(&ev_pre_end_);
    cudaEventCreate(&ev_inf_end_);

    size_t input_bytes = 1 * 3 * input_h_ * input_w_ * sizeof(float);
    size_t output_elements = 1 * (num_classes_ + 4) * num_anchors_;
    size_t output_bytes = output_elements * sizeof(float);

    cudaMalloc(&d_input_, input_bytes);
    cudaMalloc(&d_output_, output_bytes);
    cudaMallocHost(&h_output_pinned_, output_bytes); // Pinned Memory

    context_->setTensorAddress("images", d_input_);
    context_->setTensorAddress("output0", d_output_);
}

YoloTRT::~YoloTRT() {
    if (h_input_pinned_) cudaFreeHost(h_input_pinned_);
    if (d_raw_img_) cudaFree(d_raw_img_);
    if (h_output_pinned_) cudaFreeHost(h_output_pinned_);
    if (d_input_) cudaFree(d_input_);
    if (d_output_) cudaFree(d_output_);
    if (ev_start_) cudaEventDestroy(ev_start_);
    if (ev_pre_end_) cudaEventDestroy(ev_pre_end_);
    if (ev_inf_end_) cudaEventDestroy(ev_inf_end_);
    if (stream_) cudaStreamDestroy(stream_);
}

std::vector<Detection> YoloTRT::detect(const cv::Mat& frame, StageTimings& timings) {
    using Clock = std::chrono::high_resolution_clock;

    auto t0 = Clock::now();
    size_t img_bytes = static_cast<size_t>(frame.cols) * frame.rows * 3;

    // Pinned bellegi ilk karede (veya cozunurluk degisirse) bir kez ayir
    if (img_bytes > raw_img_capacity_) {
        if (h_input_pinned_) cudaFreeHost(h_input_pinned_);
        if (d_raw_img_) cudaFree(d_raw_img_);
        cudaMallocHost(&h_input_pinned_, img_bytes);
        cudaMalloc(&d_raw_img_, img_bytes);
        raw_img_capacity_ = img_bytes;
    }

    // Ham goruntuyu Pinned bellege kopyala
    if (frame.isContinuous()) {
        std::memcpy(h_input_pinned_, frame.data, img_bytes);
    } else {
        cv::Mat cont = frame.clone();
        std::memcpy(h_input_pinned_, cont.data, img_bytes);
    }
    auto t_copy_end = Clock::now();

    float scale = std::min((float)input_w_ / frame.cols, (float)input_h_ / frame.rows);
    int new_w = static_cast<int>(std::round(frame.cols * scale));
    int new_h = static_cast<int>(std::round(frame.rows * scale));
    int pad_x = (input_w_ - new_w) / 2;
    int pad_y = (input_h_ - new_h) / 2;

    size_t output_bytes = 1 * (num_classes_ + 4) * num_anchors_ * sizeof(float);

    // 1 & 2. GPU PREPROCESS + TENSORRT INFERENCE (Tek CUDA Stream uzerinde asenkron)
    cudaEventRecord(ev_start_, stream_);
    cudaMemcpyAsync(d_raw_img_, h_input_pinned_, img_bytes, cudaMemcpyHostToDevice, stream_);
    launch_letterbox_kernel(
        d_raw_img_, frame.cols, frame.rows,
        static_cast<float*>(d_input_), input_w_, input_h_,
        scale, pad_x, pad_y, new_w, new_h,
        stream_
    );
    cudaEventRecord(ev_pre_end_, stream_);

    context_->enqueueV3(stream_);
    cudaMemcpyAsync(h_output_pinned_, d_output_, output_bytes, cudaMemcpyDeviceToHost, stream_);
    cudaEventRecord(ev_inf_end_, stream_);

    cudaStreamSynchronize(stream_);
    auto t_gpu_done = Clock::now();

    // 3. POSTPROCESS (Decode [84, 8400] + NMS)
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;

    for (int i = 0; i < num_anchors_; ++i) {
        float max_score = 0.0f;
        int best_class = -1;

        for (int c = 0; c < num_classes_; ++c) {
            float score = h_output_pinned_[(4 + c) * num_anchors_ + i];
            if (score > max_score) {
                max_score = score;
                best_class = c;
            }
        }

        if (max_score >= conf_thresh_) {
            float cx = h_output_pinned_[0 * num_anchors_ + i];
            float cy = h_output_pinned_[1 * num_anchors_ + i];
            float w  = h_output_pinned_[2 * num_anchors_ + i];
            float h  = h_output_pinned_[3 * num_anchors_ + i];

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
    auto t_post_end = Clock::now();

    float gpu_pre_ms = 0.0f, gpu_inf_ms = 0.0f;
    cudaEventElapsedTime(&gpu_pre_ms, ev_start_, ev_pre_end_);
    cudaEventElapsedTime(&gpu_inf_ms, ev_pre_end_, ev_inf_end_);

    double copy_ms = std::chrono::duration<double, std::milli>(t_copy_end - t0).count();
    timings.preprocess_ms  = copy_ms + gpu_pre_ms;
    timings.inference_ms   = gpu_inf_ms;
    timings.postprocess_ms = std::chrono::duration<double, std::milli>(t_post_end - t_gpu_done).count();

    return detections;
}