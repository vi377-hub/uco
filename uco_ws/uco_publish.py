#!/usr/bin/env python3
import argparse
import time
import cv2
import cv2.aruco as aruco
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from picamera2 import Picamera2

# =====================================================================
# LỚP NHẬN DIỆN VÀ ƯỚC LƯỢNG VỊ TRÍ ARUCO (Hệ Mét)
# =====================================================================
class ArucoPoseEstimator:
    def __init__(self, marker_size_m: float, dictionary_name: str = 'DICT_6X6_250',
                 fx: float = 800.0, cam_width: int = 640, cam_height: int = 480,
                 camera_matrix_file: str = None, dist_coeffs_file: str = None):
        self.marker_size_m = marker_size_m
        self.aruco_dict = aruco.getPredefinedDictionary(getattr(aruco, dictionary_name))
        self.parameters = self._make_parameters()

        # ==========================================================
        # LOAD CAMERA CALIBRATION (BẮT BUỘC)
        # ==========================================================
        if not camera_matrix_file or not dist_coeffs_file:
            raise RuntimeError(
                "Không truyền đường dẫn cameraMatrix hoặc cameraDistortion."
            )

        try:
            self.camera_matrix = np.loadtxt(
                camera_matrix_file,
                delimiter=',',
                dtype=np.float64
            )

            self.dist_coeffs = np.loadtxt(
                dist_coeffs_file,
                delimiter=',',
                dtype=np.float64
            ).reshape(-1, 1)

        except Exception as e:
            raise RuntimeError(
                f"Không thể load calibration: {e}"
            )


        # ==========================================================
        # VERIFY CAMERA MATRIX
        # ==========================================================

        if self.camera_matrix.shape != (3, 3):
            raise RuntimeError(
                f"cameraMatrix phải có kích thước (3,3), nhận được {self.camera_matrix.shape}"
            )

        if self.dist_coeffs.ndim != 2:
            raise RuntimeError(
                "distCoeffs sai định dạng."
            )

        if np.isnan(self.camera_matrix).any():
            raise RuntimeError(
                "cameraMatrix chứa NaN."
            )

        if np.isnan(self.dist_coeffs).any():
            raise RuntimeError(
                "distCoeffs chứa NaN."
            )
        if np.isinf(self.camera_matrix).any():
            raise RuntimeError("cameraMatrix chứa Inf")

        if np.isinf(self.dist_coeffs).any():
            raise RuntimeError("distCoeffs chứa Inf")

        fx_loaded = self.camera_matrix[0, 0]
        fy_loaded = self.camera_matrix[1, 1]

        if fx_loaded <= 0 or fy_loaded <= 0:
            raise RuntimeError(
                "fx hoặc fy không hợp lệ."
            )

        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]

        if not (0 <= cx <= cam_width):
            raise RuntimeError("cx nằm ngoài ảnh.")

        if not (0 <= cy <= cam_height):
            raise RuntimeError("cy nằm ngoài ảnh.")

        print("✓ Camera calibration hợp lệ.")
        half = marker_size_m / 2.0
        self.obj_points = np.array([
            [-half,  half, 0],
            [ half,  half, 0],
            [ half, -half, 0],
            [-half, -half, 0],
        ], dtype=np.float64)
 #haven't used now
    def _use_fallback_calib(self, fx, w, h):
        cx, cy = w / 2.0, h / 2.0
        self.camera_matrix = np.array([
            [fx, 0, cx],
            [0, fx, cy],
            [0, 0, 1]
        ], dtype=np.float64)
        self.dist_coeffs = np.zeros((5, 1))

    def _make_parameters(self):
        if hasattr(aruco, 'DetectorParameters_create'):
            params = aruco.DetectorParameters_create()
        else:
            params = aruco.DetectorParameters()

        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 23
        params.adaptiveThreshWinSizeStep = 10
        params.adaptiveThreshConstant = 7
        params.minMarkerPerimeterRate = 0.02   
        params.maxMarkerPerimeterRate = 4.0
        params.polygonalApproxAccuracyRate = 0.05
        if hasattr(aruco, 'CORNER_REFINE_SUBPIX'):
            params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
            params.cornerRefinementWinSize = 5
            params.cornerRefinementMaxIterations = 30
            params.cornerRefinementMinAccuracy = 0.1
        return params

    def detect(self, frame_rgb):
        # YÊU CẦU ĐỊNH DẠNG: Đổi sang trắng đen từ RGB (vì bạn cấu hình Picamera2 là RGB888)
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)

        detections = []
        if ids is None:
            return detections

        for marker_corners, marker_id in zip(corners, ids):
            pts = marker_corners.reshape(-1, 2).astype(np.float32)
            obj_pts = self.obj_points.astype(np.float32)
            ok, rvec, tvec = cv2.solvePnP(
                obj_pts,
                pts,
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            if not ok:
                continue
            detections.append({
                'id': int(marker_id[0]),
                'x_m': float(tvec[0][0]),
                'y_m': float(tvec[1][0]),
                'z_m': float(tvec[2][0]),
                'corners': marker_corners,
                'rvec': rvec,
                'tvec': tvec,
            })

        detections.sort(key=lambda d: d['z_m'])
        return detections

    def draw_debug(self, frame_rgb, detections):
        for det in detections:
            aruco.drawDetectedMarkers(frame_rgb, [det['corners']], np.array([[det['id']]]))
            cv2.drawFrameAxes(frame_rgb, self.camera_matrix, self.dist_coeffs,
                               det['rvec'], det['tvec'], self.marker_size_m / 2)
            corner0 = det['corners'].reshape(-1, 2)[0]
            text = f"ID:{det['id']} X:{det['x_m']:.2f} Y:{det['y_m']:.2f} Z:{det['z_m']:.2f}m"
            cv2.putText(frame_rgb, text, (int(corner0[0]), int(corner0[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
        return frame_rgb

# =====================================================================
# ROS 2 PUBLISHER NODE
# =====================================================================
class ArucoPublisherNode(Node):
    def __init__(self, marker_size_m: float, dictionary_name: str,
                 fx: float, cam_width: int, cam_height: int, show_debug: bool,
                 camera_matrix_file: str = None, dist_coeffs_file: str = None):
        super().__init__('aruco_publisher_node')

        self.show_debug = show_debug
        
        # Khởi tạo class Estimator
        try:
            self.estimator = ArucoPoseEstimator(
                marker_size_m=marker_size_m,
                dictionary_name=dictionary_name,
                fx=fx,
                cam_width=cam_width,
                cam_height=cam_height,
                camera_matrix_file=camera_matrix_file,
                dist_coeffs_file=dist_coeffs_file,
            )

        except RuntimeError as e:
            self.get_logger().fatal(str(e))
            raise

        # 1. Khởi tạo Picamera2 (Thay thế cv2.VideoCapture)
        self.get_logger().info("Đang khởi tạo Picamera2...")
        self.picam2 = Picamera2()
        
        # Cấu hình RGB888 đúng như yêu cầu
        config = self.picam2.create_preview_configuration(
            main={"size": (cam_width, cam_height), "format": "RGB888"}
        )
        self.picam2.configure(config)
        self.picam2.start()

        self.get_logger().info("Thiết lập thông số phơi sáng và lấy nét...")
        self.picam2.set_controls({
            "AeEnable": False,
            "ExposureTimeMode": 1,
            "ExposureTime": 5000,
            "Sharpness": 1.5,
            "Contrast": 1.2,
            "AnalogueGainMode": 1,
            "NoiseReductionMode": 0,
            "AnalogueGain": 3.0,
            "FrameDurationLimits": (15000, 15000), 
            "AfMode": 0, 
            "LensPosition": 1.0, 
            "AwbEnable": True,   
        })
        time.sleep(0.5)

        self.pub = self.create_publisher(Float32MultiArray, '/vision/aruco_pose', 10)
        self.timer = self.create_timer(0.05, self.loop)  # Chạy 20Hz

        # Các biến chống chớp tắt và lọc nhiễu 
        self.hold_duration_sec = 0.3
        self.last_detection = None
        self.last_detection_time = 0.0

        self.confirm_frames_required = 3
        self.candidate_id = None
        self.candidate_count = 0

        self.missed_frames = 0
        self.max_missed_frames = 4

        self.get_logger().info(">>> ARUCO PUBLISHER NODE (PICAMERA2) ĐÃ KHỞI ĐỘNG <<<")

    def loop(self):
        try:
            # Bắt frame trực tiếp từ RAM
            frame = self.picam2.capture_array()
        except Exception as e:
            self.get_logger().warn(f"Không lấy được frame từ Picamera2: {e}")
            return

        detections = self.estimator.detect(frame)
        now = time.time()

        if detections:
            self.missed_frames = 0  
            nearest = detections[0]  

            if nearest['id'] == self.candidate_id:
                self.candidate_count += 1
            else:
                self.candidate_id = nearest['id']
                self.candidate_count = 1

            if self.candidate_count >= self.confirm_frames_required:
                self.last_detection = nearest
                self.last_detection_time = now
        else:
            self.missed_frames += 1
            if self.missed_frames > self.max_missed_frames:
                self.candidate_id = None
                self.candidate_count = 0

        # Publish dữ liệu nếu thỏa mãn điều kiện
        if self.last_detection is not None and (now - self.last_detection_time) < self.hold_duration_sec:
            d = self.last_detection
            msg = Float32MultiArray()
            msg.data = [
                float(d['id']),
                d['x_m'],
                d['y_m'],
                d['z_m']
            ]
            self.pub.publish(msg)

        if self.show_debug:
            self.estimator.draw_debug(frame, detections)
            
            # Khung hình gốc là RGB, khi hiển thị bằng imshow OpenCV sẽ bị ngược màu (xanh <-> đỏ). 
            # Đoạn này đổi màu hiển thị trên màn hình thành BGR để không bị đau mắt, 
            # dữ liệu gốc ở trên vẫn là RGB không ảnh hưởng.
            display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("aruco_publisher_node", display_frame)
            cv2.waitKey(1)

    def destroy_node(self):
        self.picam2.stop()
        if self.show_debug:
            cv2.destroyAllWindows()
        super().destroy_node()


def main():
    parser = argparse.ArgumentParser()
    # Cam-id bị loại bỏ vì đã sử dụng Picamera2 thay cho USB Webcam
    parser.add_argument('--marker-size', type=float, default=0.27,
                         help='Kích thước cạnh marker thật, đơn vị MÉT (VD: 0.27)')
    parser.add_argument('--dictionary', type=str, default='DICT_6X6_250')
    parser.add_argument('--fx', type=float, default=800.0)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--camera-matrix', type=str, default='cameraMatrix.txt',
                         help='Đường dẫn file camera matrix')
    parser.add_argument('--dist-coeffs', type=str, default='cameraDistortion.txt',
                         help='Đường dẫn file distortion coefficients')
    parser.add_argument('--no-debug', action='store_true')
    args, unknown = parser.parse_known_args()

    rclpy.init()
    node = ArucoPublisherNode(
        marker_size_m=args.marker_size,
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
