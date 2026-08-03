#!/usr/bin/env python3
"""
aruco_publisher_node.py
------------------------
ROS2 node THUẦN (không chứa logic OpenCV) — chỉ làm nhiệm vụ:
  1. Mở webcam
  2. Gọi lib_aruco_pose.ArucoPoseEstimator để detect
  3. Publish mã GẦN NHẤT lên /vision/aruco_pose dạng [ID, X_cm, Y_cm, Z_cm]

Toàn bộ logic detect/tính pose nằm trong lib_aruco_pose.py — sửa thuật toán
detect thì sửa bên đó, không đụng vào node này.
"""

import argparse
import time

import cv2
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from lib_aruco_pose import ArucoPoseEstimator


class ArucoPublisherNode(Node):
    def __init__(self, cam_id: int, marker_size_cm: float, dictionary_name: str,
                 fx: float, cam_width: int, cam_height: int, show_debug: bool,
                 camera_matrix_file: str = None, dist_coeffs_file: str = None):
        super().__init__('aruco_publisher_node')

        self.show_debug = show_debug
        self.estimator = ArucoPoseEstimator(
            marker_size_cm=marker_size_cm,
            dictionary_name=dictionary_name,
            fx=fx,
            cam_width=cam_width,
            cam_height=cam_height,
            camera_matrix_file=camera_matrix_file,
            dist_coeffs_file=dist_coeffs_file,
        )

        self.cap = cv2.VideoCapture(cam_id, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)
        if not self.cap.isOpened():
            self.get_logger().error(f"Không mở được webcam id={cam_id}")
            raise RuntimeError("Camera open failed")

        self.pub = self.create_publisher(Float32MultiArray, '/vision/aruco_pose', 10)
        self.timer = self.create_timer(0.05, self.loop)  # ~20Hz

        # --- Hysteresis chống chớp tắt ---
        # Giữ lại kết quả detect gần nhất trong tối đa hold_duration_sec dù frame hiện
        # tại không thấy mã (rung/nhòe tức thời), tránh publish bị đứt quãng liên tục.
        self.hold_duration_sec = 0.3
        self.last_detection = None
        self.last_detection_time = 0.0

        # Chỉ tin 1 ID là "thật" sau khi thấy liên tiếp đủ số frame -> lọc false positive thoáng qua
        self.confirm_frames_required = 3
        self.candidate_id = None
        self.candidate_count = 0

        self.get_logger().info(">>> ARUCO PUBLISHER NODE ĐÃ KHỞI ĐỘNG <<<")

    def loop(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("Không đọc được frame từ webcam")
            return

        detections = self.estimator.detect(frame)
        now = time.time()

        if detections:
            nearest = detections[0]  # đã sort theo z_cm tăng dần trong lib

            # Xác nhận ID qua vài frame liên tiếp trước khi tin
            if nearest['id'] == self.candidate_id:
                self.candidate_count += 1
            else:
                self.candidate_id = nearest['id']
                self.candidate_count = 1

            if self.candidate_count >= self.confirm_frames_required:
                self.last_detection = nearest
                self.last_detection_time = now
        else:
            # Không thấy gì frame này -> reset bộ đếm xác nhận, nhưng KHÔNG xoá last_detection ngay
            self.candidate_id = None
            self.candidate_count = 0

        # Publish: ưu tiên detect hiện tại nếu đã xác nhận, không thì dùng detection giữ lại
        # trong hold_duration_sec để tránh chớp tắt
        if self.last_detection is not None and (now - self.last_detection_time) < self.hold_duration_sec:
            d = self.last_detection
            msg = Float32MultiArray()
            msg.data = [float(d['id']), d['x_cm'], d['y_cm'], d['z_cm']]
            self.pub.publish(msg)

        if self.show_debug:
            self.estimator.draw_debug(frame, detections)
            cv2.imshow("aruco_publisher_node", frame)
            cv2.waitKey(1)

    def destroy_node(self):
        self.cap.release()
        if self.show_debug:
            cv2.destroyAllWindows()
        super().destroy_node()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cam-id', type=int, default=0)
    parser.add_argument('--marker-size', type=float, required=True,
                         help='Kích thước cạnh marker thật, đơn vị cm')
    parser.add_argument('--dictionary', type=str, default='DICT_6X6_250')
    parser.add_argument('--fx', type=float, default=800.0)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--camera-matrix', type=str, default='cameraMatrix_webcam.txt',
                         help='Đường dẫn file camera matrix calib thật (để trống/None nếu chưa calib)')
    parser.add_argument('--dist-coeffs', type=str, default='cameraDistortion_webcam.txt',
                         help='Đường dẫn file distortion coefficients calib thật')
    parser.add_argument('--no-debug', action='store_true')
    args, unknown = parser.parse_known_args()

    rclpy.init()
    node = ArucoPublisherNode(
        cam_id=args.cam_id,
        marker_size_cm=args.marker_size,
        dictionary_name=args.dictionary,
        fx=args.fx,
        cam_width=args.width,
        cam_height=args.height,
        show_debug=not args.no_debug,
        camera_matrix_file=args.camera_matrix,
        dist_coeffs_file=args.dist_coeffs,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
