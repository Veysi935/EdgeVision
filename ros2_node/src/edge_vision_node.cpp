#include <memory>
#include <string>
#include <chrono>
#include <iomanip>
#include <sstream>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>

#include "yolo_trt.hpp"
#include "bytetrack.hpp"

class EdgeVisionNode : public rclcpp::Node {
public:
    EdgeVisionNode() : Node("edge_vision_node") {
        this->declare_parameter<std::string>(
            "engine_path", "/home/veysi/EdgeVision/models/yolov8n_fp16.engine");
        this->declare_parameter<std::string>("input_topic", "/camera/image_raw");

        std::string engine_path = this->get_parameter("engine_path").as_string();
        std::string input_topic = this->get_parameter("input_topic").as_string();

        RCLCPP_INFO(this->get_logger(), "TensorRT Engine yukleniyor: %s", engine_path.c_str());
        detector_ = std::make_unique<YoloTRT>(engine_path, 0.10f, 0.45f);
        tracker_ = std::make_unique<ByteTracker>(0.45f, 0.30f, 30);

        sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            input_topic, 10,
            std::bind(&EdgeVisionNode::imageCallback, this, std::placeholders::_1));

        img_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
            "/edge_vision/image_annotated", 10);
        det_pub_ = this->create_publisher<vision_msgs::msg::Detection2DArray>(
            "/edge_vision/detections", 10);

        RCLCPP_INFO(this->get_logger(),
            "EdgeVision V2 ROS 2 Node hazir! Dinlenen topic: %s", input_topic.c_str());
    }

private:
    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        } catch (const cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge hatasi: %s", e.what());
            return;
        }

        cv::Mat& frame = cv_ptr->image;
        StageTimings t;
        auto detections = detector_->detect(frame, t);

        auto t_trk0 = std::chrono::high_resolution_clock::now();
        auto tracks = tracker_->update(detections);
        auto t_trk1 = std::chrono::high_resolution_clock::now();
        double track_ms = std::chrono::duration<double, std::milli>(t_trk1 - t_trk0).count();
        double total_ms = t.preprocess_ms + t.inference_ms + t.postprocess_ms + track_ms;
        double fps = (total_ms > 0.0) ? (1000.0 / total_ms) : 0.0;

        // 1. Detection2DArray mesajini hazirla
        vision_msgs::msg::Detection2DArray det_array;
        det_array.header = msg->header;

        for (const auto& obj : tracks) {
            vision_msgs::msg::Detection2D det_msg;
            det_msg.header = msg->header;
            det_msg.id = std::to_string(obj.track_id);
            det_msg.bbox.center.position.x = obj.box.x + obj.box.width * 0.5;
            det_msg.bbox.center.position.y = obj.box.y + obj.box.height * 0.5;
            det_msg.bbox.size_x = obj.box.width;
            det_msg.bbox.size_y = obj.box.height;

            vision_msgs::msg::ObjectHypothesisWithPose hyp;
            hyp.hypothesis.class_id = std::to_string(obj.class_id);
            hyp.hypothesis.score = obj.confidence;
            det_msg.results.push_back(hyp);
            det_array.detections.push_back(det_msg);

            // Goruntu uzerine kutu ve Track ID ciz
            cv::rectangle(frame, obj.box, cv::Scalar(0, 255, 0), 2);
            std::string label = "ID:" + std::to_string(obj.track_id);
            cv::putText(frame, label, cv::Point(obj.box.x, std::max(15, obj.box.y - 5)),
                        cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 255, 0), 2);
        }

        // HUD bilgisi
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2)
            << "ROS 2 EdgeVision V2 | " << static_cast<int>(fps) << " FPS (" << total_ms << " ms)";
        cv::putText(frame, oss.str(), cv::Point(15, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0, 255, 255), 2);

        det_pub_->publish(det_array);
        img_pub_->publish(*cv_ptr->toImageMsg());

        frame_counter_++;
        if (frame_counter_ % 60 == 0) {
            RCLCPP_INFO(this->get_logger(),
                "Pipeline: %.2f ms (%.1f FPS) | Pre: %.2f | Inf: %.2f | Post: %.2f | Track: %.2f | Aktif ID: %zu",
                total_ms, fps, t.preprocess_ms, t.inference_ms, t.postprocess_ms, track_ms, tracks.size());
        }
    }

    std::unique_ptr<YoloTRT> detector_;
    std::unique_ptr<ByteTracker> tracker_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr img_pub_;
    rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr det_pub_;
    size_t frame_counter_ = 0;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<EdgeVisionNode>());
    rclcpp::shutdown();
    return 0;
}