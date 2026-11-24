#pragma once

#include <vector>
#include <opencv2/opencv.hpp>
#include "yolo_trt.hpp"

struct TrackedObject {
    int track_id;
    cv::Rect box;
    float confidence;
    int class_id;
};

class ByteTracker {
public:
    ByteTracker(float high_thresh = 0.5f, float iou_thresh = 0.3f, int max_lost = 30);
    std::vector<TrackedObject> update(const std::vector<Detection>& detections);

private:
    struct InternalTrack {
        int id;
        cv::KalmanFilter kf;
        cv::Rect box;
        float confidence;
        int class_id;
        int lost_frames;
    };

    float computeIoU(const cv::Rect& a, const cv::Rect& b);
    InternalTrack createTrack(const Detection& det);
    cv::Rect predictTrack(InternalTrack& track);
    void updateTrack(InternalTrack& track, const Detection& det);

    float high_thresh_;
    float iou_thresh_;
    int max_lost_;
    int next_id_ = 1;
    std::vector<InternalTrack> tracks_;
};