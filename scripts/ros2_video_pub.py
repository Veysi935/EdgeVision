import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class VideoPublisher(Node):
    def __init__(self):
        super().__init__("video_publisher")
        self.pub = self.create_publisher(Image, "/camera/image_raw", 10)
        self.bridge = CvBridge()

        self.video_path = "/home/veysi/EdgeVision/benchmarks/test.mp4"
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            self.get_logger().error(f"Video acilamadi: {self.video_path}")
            return

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        self.timer = self.create_timer(1.0 / fps, self.publish_frame)
        self.get_logger().info(f"Yayin basladi ({fps:.1f} FPS) -> /camera/image_raw")

    def publish_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            # Video bitince basa sar (dongu)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return

        msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera_link"
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = VideoPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.cap.release()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()