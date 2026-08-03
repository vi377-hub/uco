import cv2
import time
import cv2.aruco as aruco
import numpy as np
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

        if camera_matrix_file and dist_coeffs_file:
            try:
                self.camera_matrix = np.loadtxt(camera_matrix_file, delimiter=',', dtype=np.float64)
                self.dist_coeffs = np.loadtxt(dist_coeffs_file, delimiter=',', dtype=np.float64).reshape(-1, 1)
                print("Đã load file calibration thành công.")
            except Exception as e:
                print(f"Lỗi load calibration, dùng ước lượng. Chi tiết: {e}")
                self._use_fallback_calib(fx, cam_width, cam_height)
        else:
            self._use_fallback_calib(fx, cam_width, cam_height)

        half = marker_size_m / 2.0
        self.obj_points = np.array([
            [-half,  half, 0],
            [ half,  half, 0],
            [ half, -half, 0],
            [-half, -half, 0],
        ], dtype=np.float64)

    def _use_fallback_calib(self, fx, w, h):
        cx, cy = w / 2.0, h / 2.0
        self.camera_matrix = np.array([
            [fx, 0, cx],
            [0, fx, cy],
            [0, 0, 1]
        ], dtype=np.float64)
        self.dist_coeffs = np.zeros((5, 1))

    @staticmethod
    def _make_parameters():
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

    def detect(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)

        detections = []
        if ids is None:
            return detections

        for marker_corners, marker_id in zip(corners, ids):
            pts = marker_corners.reshape(-1, 2).astype(np.float32)
            obj_pts = self.obj_points.astype(np.float32)
            ok, rvec, tvec = cv2.solvePnP(obj_pts, pts, self.camera_matrix, self.dist_coeffs)
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

    def draw_debug(self, frame_bgr, detections):
        for det in detections:
            aruco.drawDetectedMarkers(frame_bgr, [det['corners']], np.array([[det['id']]]))
            cv2.drawFrameAxes(frame_bgr, self.camera_matrix, self.dist_coeffs,
                               det['rvec'], det['tvec'], self.marker_size_m / 2)
            corner0 = det['corners'].reshape(-1, 2)[0]
            text = f"ID:{det['id']} X:{det['x_m']:.2f} Y:{det['y_m']:.2f} Z:{det['z_m']:.2f}m"
            cv2.putText(frame_bgr, text, (int(corner0[0]), int(corner0[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
        return frame_bgr


# =====================================================================
# CHƯƠNG TRÌNH CHÍNH (CAMERA LÔNG GHÉP NHẬN DIỆN)
# =====================================================================
if __name__ == "__main__":
    # 1. Khởi tạo Camera
    picam2 = Picamera2()
    
    # Dùng BGR888 để OpenCV hiển thị màu chuẩn (không bị mặt người xanh lè)
    config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
    picam2.configure(config)
    picam2.start()

    print("Thiết lập control camera (focus/exposure/AWB khoá cứng)...")
    picam2.set_controls({
        "AeEnable": False,
        "ExposureTimeMode": 1,
        "ExposureTime": 16000,
        "Sharpness": 2,
        "Contrast": 0.7,
        "AnalogueGainMode": 1,
        "NoiseReductionMode": 0,
        "AnalogueGain": 2.0,
        "FrameDurationLimits": (15000, 15000), # Fix max FPS ở mức ổn định ~66fps
        "AfMode": 0, 
        "LensPosition": 1.0, # Khoá nét thủ công ở 1 mét
        "AwbEnable": True,   
    })
    time.sleep(0.5)  # Đợi control áp dụng ổn định

    # 2. Khởi tạo bộ nhận diện ArUco
    # Thay đổi kích thước này (marker_size_m) đúng với cạnh của mã ArUco(VD: 0.1m = 10cm)
    estimator = ArucoPoseEstimator(
        marker_size_m=0.27,  
        dictionary_name='DICT_6X6_250',
        cam_width=640,
        cam_height=480,
        camera_matrix_file='cameraMatrix.txt', 
        dist_coeffs_file='cameraDistortion.txt'
    )

    prev_time = time.time()
    scanned_ids = set()

    print("Bắt đầu luồng camera realtime. Bấm phím 'q' trên cửa sổ để thoát.")
    try:
        while True:
            # Chụp ảnh thẳng từ RAM
            frame = picam2.capture_array()
            
            # Tính toán FPS
            current_time = time.time()
            fps = 1.0 / (current_time - prev_time)
            prev_time = current_time

            # Phát hiện ArUco
            detections = estimator.detect(frame)
            for det in detections:
                if det['id'] not in scanned_ids:
                    scanned_ids.add(det['id'])
                    print(f"Đã phát hiện mã mới: {det['id']} ở khoảng cách {det['z_m']:.2f}m")

            # Vẽ bounding box, trục tọa độ và text
            estimator.draw_debug(frame, detections)

            # Vẽ FPS lên góc trái khung hình
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Hiển thị
            cv2.imshow("Pi5 - ArUco Workspace Realtime", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Đã nhận lệnh thoát.")
                break
    finally:
        # Giải phóng tài nguyên camera dù có lỗi văng code hay tắt bằng phím Q
        picam2.stop()
        cv2.destroyAllWindows()
        print("Đã đóng camera thành công.")
        print(f"Tổng số mã ArUco đã thấy trong phiên: {len(scanned_ids)} -> {sorted(scanned_ids)}")
