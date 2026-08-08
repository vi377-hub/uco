"""Nhận diện ArUco và ước lượng pose tương đối (mét)."""

import cv2
import cv2.aruco as aruco
import numpy as np


class ArucoPoseEstimator:

    def __init__(
        self,
        marker_size_m: float,
        dictionary_name: str = "DICT_6X6_250",
        cam_width: int = 640,
        cam_height: int = 480,
        camera_matrix_file: str | None = None,
        dist_coeffs_file: str | None = None,
        input_color: str = "BGR",
        max_reprojection_error_px: float = 5.0,
    ):
        if marker_size_m <= 0:
            raise ValueError("marker_size_m phải lớn hơn 0")
        if input_color not in {"RGB", "BGR"}:
            raise ValueError("input_color chỉ được là 'RGB' hoặc 'BGR'")
        if not camera_matrix_file or not dist_coeffs_file:
            raise RuntimeError(" thiếu camera matrix và distortion coefficients")

        try:
            dictionary_id = getattr(aruco, dictionary_name)
        except AttributeError as exc:
            raise ValueError(f"Không tồn tại ArUco dictionary: {dictionary_name}") from exc

        self.marker_size_m = float(marker_size_m)
        self.input_color = input_color
        self.max_reprojection_error_px = float(max_reprojection_error_px)
        self.aruco_dict = aruco.getPredefinedDictionary(dictionary_id)
        self.parameters = self._make_parameters()

        try:
            self.camera_matrix = np.loadtxt(
                camera_matrix_file, delimiter=",", dtype=np.float64
            )
            self.dist_coeffs = np.loadtxt(
                dist_coeffs_file, delimiter=",", dtype=np.float64
            ).reshape(-1, 1)
        except Exception as exc:
            raise RuntimeError(f"Không thể load calibration: {exc}") from exc

        self._validate_calibration(cam_width, cam_height)

        half = self.marker_size_m / 2.0

        self.obj_points = np.array(
            [
                [-half, half, 0.0],
                [half, half, 0.0],
                [half, -half, 0.0],
                [-half, -half, 0.0],
            ],
            dtype=np.float32,
        )

    def _validate_calibration(self, width: int, height: int) -> None:
        if self.camera_matrix.shape != (3, 3):
            raise RuntimeError(
                f"cameraMatrix phải có shape (3, 3), nhận được {self.camera_matrix.shape}"
            )
        if self.dist_coeffs.size not in {4, 5, 8, 12, 14}:
            raise RuntimeError(
                f"distCoeffs phải có 4/5/8/12/14 hệ số, nhận được {self.dist_coeffs.size}"
            )
        if not np.isfinite(self.camera_matrix).all() or not np.isfinite(self.dist_coeffs).all():
            raise RuntimeError("Calibration chứa NaN hoặc Inf")

        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        if fx <= 0 or fy <= 0:
            raise RuntimeError("fx và fy phải lớn hơn 0")
        if not (0 <= cx <= width and 0 <= cy <= height):
            raise RuntimeError("Tâm quang học nằm ngoài kích thước ảnh cấu hình")

    @staticmethod
    def _make_parameters():
        if hasattr(aruco, "DetectorParameters_create"):
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
        if hasattr(aruco, "CORNER_REFINE_SUBPIX"):
            params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
            params.cornerRefinementWinSize = 5
            params.cornerRefinementMaxIterations = 30
            params.cornerRefinementMinAccuracy = 0.1
        return params

    def detect(self, frame):
        color_code = cv2.COLOR_RGB2GRAY if self.input_color == "RGB" else cv2.COLOR_BGR2GRAY
        gray = cv2.cvtColor(frame, color_code)
        corners, ids, _ = aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.parameters
        )

        detections = []
        if ids is None:
            return detections

        for marker_corners, marker_id in zip(corners, ids):
            image_points = marker_corners.reshape(-1, 2).astype(np.float32)
            ok, rvec, tvec = cv2.solvePnP(
                self.obj_points,
                image_points,
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
            if not ok or not np.isfinite(rvec).all() or not np.isfinite(tvec).all():
                continue

            x_m, y_m, z_m = (float(value) for value in tvec.reshape(3))
            if z_m <= 0:
                continue

            projected, _ = cv2.projectPoints(
                self.obj_points, rvec, tvec, self.camera_matrix, self.dist_coeffs
            )
            projected = projected.reshape(-1, 2)
            reprojection_error_px = float(
                np.mean(np.linalg.norm(projected - image_points, axis=1))
            )
            if reprojection_error_px > self.max_reprojection_error_px:
                continue

            detections.append(
                {
                    "id": int(marker_id[0]),
                    "x_m": x_m,
                    "y_m": y_m,
                    "z_m": z_m,
                    "reprojection_error_px": reprojection_error_px,
                    "area_px": float(abs(cv2.contourArea(image_points))),
                    "corners": marker_corners,
                    "rvec": rvec,
                    "tvec": tvec,
                }
            )

        detections.sort(key=lambda detection: detection["z_m"])
        return detections

    def draw_debug(self, frame, detections):
        for detection in detections:
            aruco.drawDetectedMarkers(
                frame,
                [detection["corners"]],
                np.array([[detection["id"]]], dtype=np.int32),
            )
            cv2.drawFrameAxes(
                frame,
                self.camera_matrix,
                self.dist_coeffs,
                detection["rvec"],
                detection["tvec"],
                self.marker_size_m / 2.0,
            )
            corner = detection["corners"].reshape(-1, 2)[0]
            label = (
                f"ID:{detection['id']} X:{detection['x_m']:.2f} "
                f"Y:{detection['y_m']:.2f} Z:{detection['z_m']:.2f}m "
                f"E:{detection['reprojection_error_px']:.1f}px"
            )
            cv2.putText(
                frame,
                label,
                (int(corner[0]), int(corner[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )
        return frame
