"""
Quy ước camera:
    - Camera úp xuống, cạnh trên ảnh hướng về đuôi drone.
    - x_m dương: marker nằm bên phải ảnh, tức bên trái drone.
    - y_m dương: marker nằm phía dưới ảnh, tức phía trước drone.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import threading
import time

import cv2
from picamera2 import Picamera2

from lib_aruco_pose import ArucoPoseEstimator


LOGGER = logging.getLogger("aruco_camera")


class ArucoCamera:

    def __init__(
        self,
        marker_size_m: float,
        dictionary_name: str = "DICT_6X6_250",
        cam_width: int = 640,
        cam_height: int = 480,
        camera_matrix_file: str = "cameraMatrix.txt",
        dist_coeffs_file: str = "cameraDistortion.txt",
        max_reprojection_error_px: float = 5.0,
        confirm_frames_required: int = 3,
        target_rate_hz: float = 20.0,
        show_debug: bool = False,
        logger: logging.Logger | None = None,
    ):
        if confirm_frames_required < 1:
            raise ValueError("confirm_frames_required phải >= 1")
        if target_rate_hz <= 0:
            raise ValueError("target_rate_hz phải > 0")

        self.logger = logger or LOGGER
        self.cam_width = int(cam_width)
        self.cam_height = int(cam_height)
        self.show_debug = bool(show_debug)
        self.confirm_frames_required = int(confirm_frames_required)
        self.frame_period_sec = 1.0 / float(target_rate_hz)

        self.estimator = ArucoPoseEstimator(
            marker_size_m=marker_size_m,
            dictionary_name=dictionary_name,
            cam_width=self.cam_width,
            cam_height=self.cam_height,
            camera_matrix_file=camera_matrix_file,
            dist_coeffs_file=dist_coeffs_file,
            input_color="BGR",
            max_reprojection_error_px=max_reprojection_error_px,
        )

        self.picam2: Picamera2 | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._latest_detections: list[dict] = []
        self._latest_frame_time = 0.0
        self._last_error: str | None = None
        self._consecutive_counts: dict[int, int] = {}
        self._logged_ids: set[int] = set()
        self._last_error_log_time = 0.0

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return

        self.logger.info("Đang khởi tạo Pi Camera...")
        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(
            main={
                "size": (self.cam_width, self.cam_height),
                "format": "RGB888",
            }
        )
        self.picam2.configure(config)
        self.picam2.start()
        try:
            self._configure_camera_controls()
        except Exception:
            self.picam2.stop()
            self.picam2 = None
            raise

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._worker_loop,
            name="aruco-camera-worker",
            daemon=True,
        )
        self._thread.start()
        self.logger.info(
            "Camera ArUco sẵn sàng (%.1f Hz, xác nhận %d frame liên tiếp)",
            1.0 / self.frame_period_sec,
            self.confirm_frames_required,
        )

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                self.logger.warning("Luồng camera chưa dừng sau 2 giây")
            self._thread = None

        if self.picam2 is not None:
            try:
                self.picam2.stop()
            finally:
                self.picam2 = None

        if self.show_debug:
            cv2.destroyAllWindows()

    def get_snapshot(self) -> tuple[list[dict], float, str | None]:
        """Trả về (detections, thời điểm frame, lỗi gần nhất) theo cách thread-safe."""
        with self._lock:
            detections = [dict(detection) for detection in self._latest_detections]
            return detections, self._latest_frame_time, self._last_error

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _configure_camera_controls(self) -> None:
        if self.picam2 is None:
            raise RuntimeError("Camera chưa được khởi tạo")

        requested = {
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
        }
        available = self.picam2.camera_controls
        supported = {key: value for key, value in requested.items() if key in available}
        unsupported = sorted(set(requested) - set(supported))
        if unsupported:
            self.logger.warning(
                "Camera/libcamera không hỗ trợ, bỏ qua: %s",
                ", ".join(unsupported),
            )
        self.picam2.set_controls(supported)

    @staticmethod
    def describe_position(detection: dict) -> tuple[str, str]:
        forward_m = float(detection["y_m"])
        right_m = -float(detection["x_m"])
        epsilon = 0.02

        if forward_m > epsilon:
            longitudinal = f"phía trước {forward_m:.2f} m"
        elif forward_m < -epsilon:
            longitudinal = f"phía sau {abs(forward_m):.2f} m"
        else:
            longitudinal = "gần ngang tâm theo trục trước/sau"

        if right_m > epsilon:
            lateral = f"lệch phải {right_m:.2f} m"
        elif right_m < -epsilon:
            lateral = f"lệch trái {abs(right_m):.2f} m"
        else:
            lateral = "gần ngang tâm theo trục trái/phải"

        return longitudinal, lateral

    def _confirmed_detections(self, detections: list[dict]) -> list[dict]:
        current_ids = {int(detection["id"]) for detection in detections}
        for marker_id in list(self._consecutive_counts):
            if marker_id not in current_ids:
                del self._consecutive_counts[marker_id]

        confirmed = []
        for detection in detections:
            marker_id = int(detection["id"])
            self._consecutive_counts[marker_id] = (
                self._consecutive_counts.get(marker_id, 0) + 1
            )
            if self._consecutive_counts[marker_id] >= self.confirm_frames_required:
                confirmed.append(detection)
        return confirmed

    @staticmethod
    def _public_detection(detection: dict) -> dict:
        return {
            "id": int(detection["id"]),
            "x_m": float(detection["x_m"]),
            "y_m": float(detection["y_m"]),
            "z_m": float(detection["z_m"]),
            "reprojection_error_px": float(detection["reprojection_error_px"]),
            "area_px": float(detection["area_px"]),
        }

    def _store_frame(self, detections: list[dict], now: float) -> None:
        public_detections = [self._public_detection(item) for item in detections]
        with self._lock:
            self._latest_detections = public_detections
            self._latest_frame_time = now
            self._last_error = None

    def _store_error(self, exc: Exception, now: float) -> None:
        message = f"{type(exc).__name__}: {exc}"
        # Lỗi/cúp frame làm đứt chuỗi xác nhận; không được tính frame sau như thể marker vẫn xuất hiện liên tiếp.
        self._consecutive_counts.clear()
        with self._lock:
            self._latest_detections = []
            self._latest_frame_time = now
            self._last_error = message

        if now - self._last_error_log_time >= 2.0:
            self._last_error_log_time = now
            self.logger.error("Lỗi camera/nhận dạng: %s", message)

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            loop_start = time.monotonic()
            try:
                if self.picam2 is None:
                    raise RuntimeError("Pi Camera đã bị đóng")

                frame = self.picam2.capture_array()
                detections = self.estimator.detect(frame)
                confirmed = self._confirmed_detections(detections)
                now = time.monotonic()
                self._store_frame(confirmed, now)

                for detection in confirmed:
                    marker_id = int(detection["id"])
                    if marker_id in self._logged_ids:
                        continue
                    self._logged_ids.add(marker_id)
                    longitudinal, lateral = self.describe_position(detection)
                    self.logger.info(
                        " ID %d: %s, %s, cách camera %.2f m",
                        marker_id,
                        longitudinal,
                        lateral,
                        detection["z_m"],
                    )

                if self.show_debug:
                    self.estimator.draw_debug(frame, detections)
                    cv2.imshow("aruco_camera", frame)
                    cv2.waitKey(1)
            except Exception as exc:
                self._store_error(exc, time.monotonic())

            remaining = self.frame_period_sec - (time.monotonic() - loop_start)
            if remaining > 0:
                self._stop_event.wait(remaining)


def _default_calibration_path(filename: str) -> str:
    return str(Path(__file__).resolve().parent / filename)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Thử camera và ArUco độc lập, không cần ROS/PX4"
    )
    parser.add_argument("--marker-size", type=float, default=0.27)
    parser.add_argument("--dictionary", default="DICT_6X6_250")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument(
        "--camera-matrix",
        default=_default_calibration_path("cameraMatrix.txt"),
    )
    parser.add_argument(
        "--dist-coeffs",
        default=_default_calibration_path("cameraDistortion.txt"),
    )
    parser.add_argument("--max-reprojection-error", type=float, default=5.0)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    camera = ArucoCamera(
        marker_size_m=args.marker_size,
        dictionary_name=args.dictionary,
        cam_width=args.width,
        cam_height=args.height,
        camera_matrix_file=args.camera_matrix,
        dist_coeffs_file=args.dist_coeffs,
        max_reprojection_error_px=args.max_reprojection_error,
        show_debug=args.debug,
    )
    try:
        camera.start()
        while camera.is_running():
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        camera.stop()


if __name__ == "__main__":
    main()
