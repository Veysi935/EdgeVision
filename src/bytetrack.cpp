#include "bytetrack.hpp"
#include <algorithm>

ByteTracker::ByteTracker(float high_thresh, float iou_thresh, int max_lost)
    : high_thresh_(high_thresh), iou_thresh_(iou_thresh), max_lost_(max_lost) {}

float ByteTracker::computeIoU(const cv::Rect& a, const cv::Rect& b) {
    int inter_area = (a & b).area();
    int union_area = a.area() + b.area() - inter_area;
    return union_area > 0 ? static_cast<float>(inter_area) / union_area : 0.0f;
}

ByteTracker::InternalTrack ByteTracker::createTrack(const Detection& det) {
    InternalTrack t;
    t.id = next_id_++;
    t.box = det.box;
    t.confidence = det.confidence;
    t.class_id = det.class_id;
    t.lost_frames = 0;

    // 8 durum [cx, cy, w, h, vx, vy, vw, vh], 4 olcum [cx, cy, w, h]
    t.kf = cv::KalmanFilter(8, 4, 0, CV_32F);
    cv::setIdentity(t.kf.transitionMatrix);
    for (int i = 0; i < 4; ++i) t.kf.transitionMatrix.at<float>(i, i + 4) = 1.0f;

    cv::setIdentity(t.kf.measurementMatrix);
    cv::setIdentity(t.kf.processNoiseCov, cv::Scalar::all(1e-2));
    cv::setIdentity(t.kf.measurementNoiseCov, cv::Scalar::all(1e-1));
    cv::setIdentity(t.kf.errorCovPost, cv::Scalar::all(1.0));

    float cx = det.box.x + det.box.width * 0.5f;
    float cy = det.box.y + det.box.height * 0.5f;
    t.kf.statePost = (cv::Mat_<float>(8, 1) << cx, cy, (float)det.box.width, (float)det.box.height, 0, 0, 0, 0);
    return t;
}

cv::Rect ByteTracker::predictTrack(InternalTrack& track) {
    cv::Mat pred = track.kf.predict();
    float cx = pred.at<float>(0);
    float cy = pred.at<float>(1);
    float w  = std::max(1.0f, pred.at<float>(2));
    float h  = std::max(1.0f, pred.at<float>(3));
    track.box = cv::Rect(static_cast<int>(cx - w * 0.5f), static_cast<int>(cy - h * 0.5f),
                         static_cast<int>(w), static_cast<int>(h));
    return track.box;
}

void ByteTracker::updateTrack(InternalTrack& track, const Detection& det) {
    float cx = det.box.x + det.box.width * 0.5f;
    float cy = det.box.y + det.box.height * 0.5f;
    cv::Mat meas = (cv::Mat_<float>(4, 1) << cx, cy, (float)det.box.width, (float)det.box.height);
    track.kf.correct(meas);
    track.box = det.box;
    track.confidence = det.confidence;
    track.class_id = det.class_id;
    track.lost_frames = 0;
}

std::vector<TrackedObject> ByteTracker::update(const std::vector<Detection>& detections) {
    // 1. Tum aktif takipleri Kalman filtresi ile tahmin et
    for (auto& t : tracks_) {
        predictTrack(t);
    }

    // 2. ByteTrack mantigi: Tespitleri Yuksek ve Dusuk guven skorlu olarak ikiye ayir
    std::vector<Detection> high_dets, low_dets;
    for (const auto& d : detections) {
        if (d.confidence >= high_thresh_) high_dets.push_back(d);
        else low_dets.push_back(d);
    }

    std::vector<bool> track_matched(tracks_.size(), false);
    std::vector<bool> high_matched(high_dets.size(), false);

    // 3. ASAMA 1: Yuksek skorlu tespitlerle IoU eslestirmesi
    for (size_t i = 0; i < tracks_.size(); ++i) {
        float best_iou = iou_thresh_;
        int best_det = -1;
        for (size_t j = 0; j < high_dets.size(); ++j) {
            if (high_matched[j]) continue;
            float iou = computeIoU(tracks_[i].box, high_dets[j].box);
            if (iou > best_iou) {
                best_iou = iou;
                best_det = static_cast<int>(j);
            }
        }
        if (best_det != -1) {
            updateTrack(tracks_[i], high_dets[best_det]);
            track_matched[i] = true;
            high_matched[best_det] = true;
        }
    }

    // 4. ASAMA 2: Eslesmeyen takipleri dusuk skorlu tespitlerle kurtar (ByteTrack farki)
    std::vector<bool> low_matched(low_dets.size(), false);
    for (size_t i = 0; i < tracks_.size(); ++i) {
        if (track_matched[i]) continue;
        float best_iou = iou_thresh_;
        int best_det = -1;
        for (size_t j = 0; j < low_dets.size(); ++j) {
            if (low_matched[j]) continue;
            float iou = computeIoU(tracks_[i].box, low_dets[j].box);
            if (iou > best_iou) {
                best_iou = iou;
                best_det = static_cast<int>(j);
            }
        }
        if (best_det != -1) {
            updateTrack(tracks_[i], low_dets[best_det]);
            track_matched[i] = true;
            low_matched[best_det] = true;
        }
    }

    // 5. Eslesmeyen takipleri yaslandir, suresi dolanlari sil
    for (size_t i = 0; i < tracks_.size(); ++i) {
        if (!track_matched[i]) tracks_[i].lost_frames++;
    }
    tracks_.erase(
        std::remove_if(tracks_.begin(), tracks_.end(),
                       [this](const InternalTrack& t) { return t.lost_frames > max_lost_; }),
        tracks_.end()
    );

    // 6. Eslesmeyen yuksek skorlu tespitlerden yeni takip baslat
    for (size_t j = 0; j < high_dets.size(); ++j) {
        if (!high_matched[j]) {
            tracks_.push_back(createTrack(high_dets[j]));
        }
    }

    // Sadece o karede aktif guncellenen nesneleri dondur
    std::vector<TrackedObject> output;
    for (const auto& t : tracks_) {
        if (t.lost_frames == 0) {
            output.push_back({t.id, t.box, t.confidence, t.class_id});
        }
    }
    return output;
}