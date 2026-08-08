"""
    Mission -> khóa ArUco gần nhất -> OFFBOARD -> căn tâm/hover 5 s
    -> Mission.
    Nếu phi công chọn mode khác, dừng gửi
    setpoint, nhường quyền điều khiển cho pilot.
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass
import logging
import math
from pathlib import Path
import time
from typing import Protocol


LOGGER = logging.getLogger("aruco_mission")

AUTO_MISSION_MODE = "AUTO.MISSION"
OFFBOARD_MODE = "OFFBOARD"
AUTOMATION_MODES = {AUTO_MISSION_MODE, OFFBOARD_MODE}

IGNORE_POSITION_MASK = (1 << 0) | (1 << 1) | (1 << 2)
IGNORE_ACCELERATION_MASK = (1 << 6) | (1 << 7) | (1 << 8)
IGNORE_YAW_MASK = 1 << 10
BODY_VELOCITY_TYPE_MASK = (
    IGNORE_POSITION_MASK | IGNORE_ACCELERATION_MASK | IGNORE_YAW_MASK
)


class VisionSource(Protocol):
    def get_snapshot(self) -> tuple[list[dict], float, str | None]: ...


class VehicleInterface(Protocol):
    mode: str
    armed: bool
    horizontal_speed: float
    last_velocity_time: float

    @property
    def connected(self) -> bool: ...

    def poll(self) -> None: ...

    def send_companion_heartbeat(self, now: float) -> None: ...

    def send_body_velocity(
        self,
        forward_mps: float,
        right_mps: float,
        down_mps: float = 0.0,
        yaw_rate_rps: float = 0.0,
    ) -> None: ...

    def request_mode(self, mode: str) -> None: ...


class PymavlinkVehicle:

    def __init__(
        self,
        device: str = "/dev/serial0",
        baud: int = 921600,
        heartbeat_wait_sec: float = 10.0,
        heartbeat_timeout_sec: float = 3.0,
        source_system: int = 245,
        source_component: int = 191,
        logger: logging.Logger | None = None,
    ):
        try:
            from pymavlink import mavutil
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Chưa cài pymavlink. Hãy tạo venv --system-site-packages và "
                "chạy: python -m pip install pymavlink"
            ) from exc

        self.mavutil = mavutil
        self.logger = logger or LOGGER
        self.heartbeat_timeout_sec = float(heartbeat_timeout_sec)
        self.mode = "UNKNOWN"
        self.armed = False
        self.horizontal_speed = 0.0
        self.last_velocity_time = 0.0
        self.last_heartbeat_time = 0.0
        self.last_command_ack: dict[int, int] = {}
        self._last_companion_heartbeat = 0.0
        self._boot_reference = time.monotonic()

        self.logger.info("Mở MAVLink %s ở %d baud...", device, baud)
        self.master = mavutil.mavlink_connection(
            device,
            baud=int(baud),
            autoreconnect=True,
            robust_parsing=True,
            source_system=int(source_system),
            source_component=int(source_component),
        )
        heartbeat = self.master.wait_heartbeat(timeout=float(heartbeat_wait_sec))
        if heartbeat is None:
            self.master.close()
            raise TimeoutError(
                f"Không nhận được heartbeat PX4 trong {heartbeat_wait_sec:.1f} giây"
            )

        self.target_system = int(self.master.target_system)
        # wait_heartbeat() thường khóa system ID nhưng một số bản pymavlink vẫn
        # để target_component=0. Lấy trực tiếp component của heartbeat PX4.
        self.target_component = int(heartbeat.get_srcComponent())
        self.master.target_component = self.target_component
        self._process_message(heartbeat, time.monotonic())
        if heartbeat.autopilot != mavutil.mavlink.MAV_AUTOPILOT_PX4:
            self.master.close()
            raise RuntimeError(
                f"Thiết bị MAVLink không phải PX4 (autopilot={heartbeat.autopilot})"
            )

        self.logger.info(
            "Đã nhận PX4 heartbeat: system=%d component=%d mode=%s armed=%s",
            self.target_system,
            self.target_component,
            self.mode,
            self.armed,
        )
        self._request_message_interval(
            getattr(mavutil.mavlink, "MAVLINK_MSG_ID_LOCAL_POSITION_NED", 32),
            10.0,
        )

    @property
    def connected(self) -> bool:
        return (
            self.last_heartbeat_time > 0.0
            and time.monotonic() - self.last_heartbeat_time
            <= self.heartbeat_timeout_sec
        )

    def _request_message_interval(self, message_id: int, rate_hz: float) -> None:
        interval_us = int(1_000_000 / rate_hz)
        self.master.mav.command_long_send(
            self.target_system,
            self.target_component,
            self.mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
            0,
            float(message_id),
            float(interval_us),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )

    def _process_message(self, message, now: float) -> None:
        message_type = message.get_type()
        if message_type == "BAD_DATA":
            return

        if message_type == "HEARTBEAT":
            if int(message.get_srcSystem()) != self.target_system:
                return
            self.last_heartbeat_time = now
            raw_mode = str(self.mavutil.mode_string_v10(message)).upper()
            # pymavlink gọi PX4 AUTO.MISSION là "MISSION"; chuẩn hóa để phần
            # state machine giữ cùng tên rõ ràng đã dùng trong thiết kế cũ.
            self.mode = AUTO_MISSION_MODE if raw_mode == "MISSION" else raw_mode
            self.armed = bool(
                int(message.base_mode)
                & self.mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
            )
        elif message_type == "LOCAL_POSITION_NED":
            if int(message.get_srcSystem()) != self.target_system:
                return
            self.horizontal_speed = math.hypot(float(message.vx), float(message.vy))
            self.last_velocity_time = now
        elif message_type == "COMMAND_ACK":
            self.last_command_ack[int(message.command)] = int(message.result)
        elif message_type == "STATUSTEXT":
            raw_text = message.text
            if isinstance(raw_text, bytes):
                raw_text = raw_text.decode("utf-8", errors="replace")
            text = str(raw_text).rstrip("\x00")
            severity = int(message.severity)
            if severity <= self.mavutil.mavlink.MAV_SEVERITY_WARNING:
                self.logger.warning("PX4: %s", text)

    def poll(self) -> None:
        now = time.monotonic()
        # Giới hạn số message mỗi vòng để không làm trễ setpoint nếu serial bị dồn dữ liệu.
        for _ in range(200):
            message = self.master.recv_match(blocking=False)
            if message is None:
                break
            self._process_message(message, now)

    def send_companion_heartbeat(self, now: float) -> None:
        if now - self._last_companion_heartbeat < 1.0:
            return
        self._last_companion_heartbeat = now
        self.master.mav.heartbeat_send(
            self.mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
            self.mavutil.mavlink.MAV_AUTOPILOT_INVALID,
            0,
            0,
            self.mavutil.mavlink.MAV_STATE_ACTIVE,
            3,
        )

    def send_body_velocity(
        self,
        forward_mps: float,
        right_mps: float,
        down_mps: float = 0.0,
        yaw_rate_rps: float = 0.0,
    ) -> None:
        values = (forward_mps, right_mps, down_mps, yaw_rate_rps)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Setpoint chứa NaN hoặc Inf")

        time_boot_ms = int((time.monotonic() - self._boot_reference) * 1000)
        time_boot_ms &= 0xFFFFFFFF
        self.master.mav.set_position_target_local_ned_send(
            time_boot_ms,
            self.target_system,
            self.target_component,
            self.mavutil.mavlink.MAV_FRAME_BODY_NED,
            BODY_VELOCITY_TYPE_MASK,
            0.0,
            0.0,
            0.0,
            float(forward_mps),
            float(right_mps),
            float(down_mps),
            0.0,
            0.0,
            0.0,
            0.0,
            float(yaw_rate_rps),
        )

    def request_mode(self, mode: str) -> None:
        mode = mode.upper()
        pymavlink_mode = "MISSION" if mode == AUTO_MISSION_MODE else mode
        mode_map = self.master.mode_mapping()
        if mode_map is None or pymavlink_mode not in mode_map:
            available = ", ".join(sorted(mode_map or {}))
            raise RuntimeError(f"PX4 không có mode {mode}; các mode thấy được: {available}")
        self.master.set_mode(pymavlink_mode)
        self.logger.info("Đã gửi yêu cầu chuyển mode %s", mode)

    def close(self) -> None:
        self.master.close()


@dataclass(frozen=True)
class MissionConfig:
    hover_duration_sec: float = 5.0
    center_diameter_m: float = 0.20
    target_lost_timeout_sec: float = 1.0
    vision_stream_timeout_sec: float = 0.25
    offboard_prestream_sec: float = 1.1
    offboard_timeout_sec: float = 2.0
    mode_retry_interval_sec: float = 1.0
    max_mode_retries: int = 3
    kp_xy: float = 0.50
    kd_xy: float = 0.05
    pose_filter_alpha: float = 0.35
    derivative_filter_alpha: float = 0.25
    deadband_m: float = 0.02
    max_velocity_mps: float = 0.20
    max_acceleration_mps2: float = 0.30
    hover_speed_threshold_mps: float = 0.10

    def __post_init__(self) -> None:
        positive = {
            "hover_duration_sec": self.hover_duration_sec,
            "center_diameter_m": self.center_diameter_m,
            "target_lost_timeout_sec": self.target_lost_timeout_sec,
            "vision_stream_timeout_sec": self.vision_stream_timeout_sec,
            "offboard_prestream_sec": self.offboard_prestream_sec,
            "offboard_timeout_sec": self.offboard_timeout_sec,
            "mode_retry_interval_sec": self.mode_retry_interval_sec,
            "max_velocity_mps": self.max_velocity_mps,
            "max_acceleration_mps2": self.max_acceleration_mps2,
            "hover_speed_threshold_mps": self.hover_speed_threshold_mps,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} phải > 0")
        for name, value in {
            "pose_filter_alpha": self.pose_filter_alpha,
            "derivative_filter_alpha": self.derivative_filter_alpha,
        }.items():
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} phải nằm trong (0, 1]")
        if self.kp_xy < 0 or self.kd_xy < 0 or self.deadband_m < 0:
            raise ValueError("Kp, Kd và deadband không được âm")
        if self.max_mode_retries < 1:
            raise ValueError("max_mode_retries phải >= 1")


class MissionController:
    def __init__(
        self,
        vehicle: VehicleInterface,
        vision: VisionSource,
        config: MissionConfig | None = None,
        logger: logging.Logger | None = None,
    ):
        self.vehicle = vehicle
        self.vision = vision
        self.config = config or MissionConfig()
        self.logger = logger or LOGGER
        self.center_radius_m = self.config.center_diameter_m / 2.0

        self.was_connected = bool(vehicle.connected)
        self.manual_override = False
        self.offboard_owned_by_node = False
        self.current_state = "MISSION"
        self.current_target_id: int | None = None
        self.handled_ids: set[int] = set()
        self.latest_detections: dict[int, dict] = {}
        self.last_vision_message_time = 0.0
        self.last_target_seen_time = 0.0

        self.hover_start_time: float | None = None
        self.wait_state_start = 0.0
        self.last_mode_request_time = 0.0
        self.mode_request_attempts = 0
        self.wait_auto_reason = ""
        self.continuous_stream_start: float | None = None

        self.filtered_forward_error: float | None = None
        self.filtered_right_error: float | None = None
        self.previous_forward_error = 0.0
        self.previous_right_error = 0.0
        self.filtered_forward_derivative = 0.0
        self.filtered_right_derivative = 0.0
        self.last_controller_time = time.monotonic()
        self.last_command_forward = 0.0
        self.last_command_right = 0.0

        self.logger.info(
            "MISSION CONTROLLER sẵn sàng: pymavlink, BODY_NED, PD lọc, "
            "vùng tâm đường kính %.2f m",
            self.config.center_diameter_m,
        )

    def _update_vision(self) -> None:
        detections, frame_time, _camera_error = self.vision.get_snapshot()
        if frame_time <= 0.0:
            return

        valid: dict[int, dict] = {}
        for detection in detections:
            values = (
                detection.get("x_m"),
                detection.get("y_m"),
                detection.get("z_m"),
                detection.get("reprojection_error_px"),
            )
            if not all(isinstance(value, (int, float)) for value in values):
                continue
            if not all(math.isfinite(float(value)) for value in values):
                continue
            marker_id = int(detection.get("id", -1))
            if marker_id < 0 or float(detection["z_m"]) <= 0:
                continue
            valid[marker_id] = dict(detection)

        self.latest_detections = valid
        self.last_vision_message_time = frame_time
        if self.current_target_id in valid:
            self.last_target_seen_time = frame_time

    def _reset_controller(self) -> None:
        self.filtered_forward_error = None
        self.filtered_right_error = None
        self.previous_forward_error = 0.0
        self.previous_right_error = 0.0
        self.filtered_forward_derivative = 0.0
        self.filtered_right_derivative = 0.0
        self.last_controller_time = time.monotonic()
        self.last_command_forward = 0.0
        self.last_command_right = 0.0
        self.hover_start_time = None

    def _clear_active_target(self) -> None:
        self.current_target_id = None
        self.last_target_seen_time = 0.0
        self._reset_controller()

    def _enter_manual_override(self, mode: str) -> None:
        if not self.manual_override:
            self.logger.warning(
                "RC/PILOT OVERRIDE: PX4 đang ở mode %s. Dừng setpoint, hủy "
                "tracking, nhường quyền điều khiển cho pilot.",
                mode,
            )
        self.manual_override = True
        self.offboard_owned_by_node = False
        self.current_state = "MANUAL_OVERRIDE"
        self.continuous_stream_start = None
        self._clear_active_target()

    def _operator_has_control(self) -> bool:
        if not self.vehicle.connected:
            if self.was_connected:
                self.logger.error("Mất heartbeat PX4; dừng automation")
                self.manual_override = True
                self.offboard_owned_by_node = False
                self.current_state = "MANUAL_OVERRIDE"
                self.continuous_stream_start = None
                self._clear_active_target()
            return True

        self.was_connected = True
        mode = self.vehicle.mode
        if self.manual_override:
            if mode == AUTO_MISSION_MODE:
                self.manual_override = False
                self.current_state = "MISSION"
                self.continuous_stream_start = None
                self.logger.info(
                    "Automation được hoạt động lại"
                )
                return False
            return True

        if mode not in AUTOMATION_MODES:
            self._enter_manual_override(mode)
            return True

        if mode == OFFBOARD_MODE and not self.offboard_owned_by_node:
            self._enter_manual_override("OFFBOARD không do chương trình yêu cầu")
            return True

        return False

    def _vision_is_fresh(self, now: float) -> bool:
        return (
            self.last_vision_message_time > 0.0
            and now - self.last_vision_message_time
            <= self.config.vision_stream_timeout_sec
        )

    def _target_detection(self, now: float) -> dict | None:
        if not self._vision_is_fresh(now) or self.current_target_id is None:
            return None
        return self.latest_detections.get(self.current_target_id)

    def _select_nearest_unhandled(self, now: float) -> dict | None:
        if not self._vision_is_fresh(now):
            return None
        candidates = [
            detection
            for marker_id, detection in self.latest_detections.items()
            if marker_id not in self.handled_ids
        ]
        return min(candidates, key=lambda detection: detection["z_m"], default=None)

    def _send_zero(self) -> None:
        self.vehicle.send_body_velocity(0.0, 0.0, 0.0, 0.0)

    def _request_mode(self, mode: str) -> None:
        if self.manual_override:
            return
        self.vehicle.request_mode(mode)
        self.last_mode_request_time = time.monotonic()

    def _limit_velocity(self, forward: float, right: float, dt: float) -> tuple[float, float]:
        speed = math.hypot(forward, right)
        if speed > self.config.max_velocity_mps:
            scale = self.config.max_velocity_mps / speed
            forward *= scale
            right *= scale

        delta_forward = forward - self.last_command_forward
        delta_right = right - self.last_command_right
        delta = math.hypot(delta_forward, delta_right)
        max_delta = self.config.max_acceleration_mps2 * dt
        if delta > max_delta and delta > 0:
            scale = max_delta / delta
            forward = self.last_command_forward + delta_forward * scale
            right = self.last_command_right + delta_right * scale

        self.last_command_forward = forward
        self.last_command_right = right
        return forward, right

    def _calculate_pd_command(self, detection: dict, now: float) -> tuple[float, float]:
        # Camera nhìn xuống và cạnh trên ảnh hướng về đuôi drone:
        raw_forward_error = float(detection["y_m"])
        raw_right_error = -float(detection["x_m"])

        if self.filtered_forward_error is None:
            self.filtered_forward_error = raw_forward_error
            self.filtered_right_error = raw_right_error
            self.previous_forward_error = raw_forward_error
            self.previous_right_error = raw_right_error

        alpha = self.config.pose_filter_alpha
        self.filtered_forward_error = (
            alpha * raw_forward_error
            + (1.0 - alpha) * self.filtered_forward_error
        )
        self.filtered_right_error = (
            alpha * raw_right_error
            + (1.0 - alpha) * self.filtered_right_error
        )

        dt = max(0.02, min(0.20, now - self.last_controller_time))
        raw_d_forward = (
            self.filtered_forward_error - self.previous_forward_error
        ) / dt
        raw_d_right = (
            self.filtered_right_error - self.previous_right_error
        ) / dt

        derivative_alpha = self.config.derivative_filter_alpha
        self.filtered_forward_derivative = (
            derivative_alpha * raw_d_forward
            + (1.0 - derivative_alpha) * self.filtered_forward_derivative
        )
        self.filtered_right_derivative = (
            derivative_alpha * raw_d_right
            + (1.0 - derivative_alpha) * self.filtered_right_derivative
        )

        forward_error = self.filtered_forward_error
        right_error = self.filtered_right_error
        forward_derivative = self.filtered_forward_derivative
        right_derivative = self.filtered_right_derivative
        if abs(forward_error) < self.config.deadband_m:
            forward_error = 0.0
            forward_derivative = 0.0
        if abs(right_error) < self.config.deadband_m:
            right_error = 0.0
            right_derivative = 0.0

        forward_command = (
            self.config.kp_xy * forward_error
            + self.config.kd_xy * forward_derivative
        )
        right_command = (
            self.config.kp_xy * right_error
            + self.config.kd_xy * right_derivative
        )
        forward_command, right_command = self._limit_velocity(
            forward_command,
            right_command,
            dt,
        )

        self.previous_forward_error = self.filtered_forward_error
        self.previous_right_error = self.filtered_right_error
        self.last_controller_time = now
        return forward_command, right_command

    def _begin_return_to_mission(self, reason: str) -> None:
        self.wait_auto_reason = reason
        self.current_state = "WAIT_AUTO_MISSION"
        self.wait_state_start = time.monotonic()
        self.mode_request_attempts = 0
        self._retry_auto_mission()

    def _retry_auto_mission(self) -> None:
        if self.mode_request_attempts >= self.config.max_mode_retries:
            return
        self.mode_request_attempts += 1
        self._request_mode(AUTO_MISSION_MODE)

    def step(self, now: float | None = None) -> None:
        now = time.monotonic() if now is None else float(now)
        self.vehicle.poll()
        self.vehicle.send_companion_heartbeat(now)
        self._update_vision()

        if self._operator_has_control():
            return

        if self.current_state == "ALIGN_AND_HOVER" and self.vehicle.mode != OFFBOARD_MODE:
            self.logger.warning(
                "PX4 đã rời OFFBOARD sang %s; hủy tracking hiện tại",
                self.vehicle.mode,
            )
            self.offboard_owned_by_node = False
            self.current_state = "MISSION"
            self.continuous_stream_start = None
            self._clear_active_target()
            return

        if self.current_state == "MISSION":
            self._send_zero()
            if self.continuous_stream_start is None:
                self.continuous_stream_start = now

            if self.vehicle.mode != AUTO_MISSION_MODE or not self.vehicle.armed:
                return

            target = self._select_nearest_unhandled(now)
            if target is None:
                return

            self.current_target_id = int(target["id"])
            self.last_target_seen_time = now
            self.current_state = "WAIT_PRESTREAM"
            self.wait_state_start = now
            self.logger.info(
                "Khóa target ID %d (gần nhất, cách camera %.2f m)",
                self.current_target_id,
                target["z_m"],
            )
            return

        if self.current_state == "WAIT_PRESTREAM":
            self._send_zero()
            if self._target_detection(now) is None:
                if now - self.last_target_seen_time > self.config.target_lost_timeout_sec:
                    self.logger.warning("Mất target trước khi vào OFFBOARD; hủy tracking")
                    self.current_state = "MISSION"
                    self._clear_active_target()
                return

            if self.continuous_stream_start is None:
                self.continuous_stream_start = now
            if now - self.continuous_stream_start < self.config.offboard_prestream_sec:
                return

            self.offboard_owned_by_node = True
            self.current_state = "WAIT_OFFBOARD"
            self.wait_state_start = now
            self._request_mode(OFFBOARD_MODE)
            return

        if self.current_state == "WAIT_OFFBOARD":
            self._send_zero()
            if self.vehicle.mode == OFFBOARD_MODE:
                self._reset_controller()
                self.current_state = "ALIGN_AND_HOVER"
                self.logger.info(
                    "PX4 đã vào OFFBOARD; bắt đầu căn ID %d",
                    self.current_target_id,
                )
                return

            if now - self.wait_state_start > self.config.offboard_timeout_sec:
                self.logger.error("PX4 không vào OFFBOARD đúng thời hạn; hủy tracking")
                self.offboard_owned_by_node = False
                self.current_state = "MISSION"
                self._clear_active_target()
            return

        if self.current_state == "ALIGN_AND_HOVER":
            detection = self._target_detection(now)
            if detection is None:
                self.last_command_forward = 0.0
                self.last_command_right = 0.0
                self._send_zero()
                self.hover_start_time = None
                if now - self.last_target_seen_time > self.config.target_lost_timeout_sec:
                    self.logger.warning(
                        "Mất target ID %d quá %.1f s; trở lại mission",
                        self.current_target_id,
                        self.config.target_lost_timeout_sec,
                    )
                    self._begin_return_to_mission("target_lost")
                return

            forward_command, right_command = self._calculate_pd_command(detection, now)
            self.vehicle.send_body_velocity(
                forward_command,
                right_command,
                0.0,
                0.0,
            )

            center_error = math.hypot(
                self.filtered_forward_error or 0.0,
                self.filtered_right_error or 0.0,
            )
            if now - self.vehicle.last_velocity_time <= 0.5:
                horizontal_speed = self.vehicle.horizontal_speed
            else:
                horizontal_speed = math.hypot(forward_command, right_command)

            centered_and_stable = (
                center_error <= self.center_radius_m
                and horizontal_speed <= self.config.hover_speed_threshold_mps
            )
            if centered_and_stable:
                if self.hover_start_time is None:
                    self.hover_start_time = now
                    self.logger.info(
                        "ID %d đã vào vùng tâm bán kính %.2f m; bắt đầu đếm %.1f s",
                        self.current_target_id,
                        self.center_radius_m,
                        self.config.hover_duration_sec,
                    )
                elif now - self.hover_start_time >= self.config.hover_duration_sec:
                    completed_id = int(self.current_target_id)
                    self.handled_ids.add(completed_id)
                    self.logger.info(
                        "Hoàn thành ID %d: hover liên tục %.1f s",
                        completed_id,
                        self.config.hover_duration_sec,
                    )
                    self._begin_return_to_mission("hover_complete")
            elif self.hover_start_time is not None:
                self.hover_start_time = None
                self.logger.info("Ra khỏi vùng tâm/đang còn trôi; reset bộ đếm hover")
            return

        if self.current_state == "WAIT_AUTO_MISSION":
            if self.vehicle.mode == OFFBOARD_MODE:
                self._send_zero()

            if self.vehicle.mode == AUTO_MISSION_MODE:
                self.logger.info(
                    "PX4 đã trở lại AUTO.MISSION (%s)",
                    self.wait_auto_reason,
                )
                self.offboard_owned_by_node = False
                self.current_state = "MISSION"
                self.continuous_stream_start = now
                self._clear_active_target()
                return

            if (
                self.mode_request_attempts < self.config.max_mode_retries
                and now - self.last_mode_request_time
                >= self.config.mode_retry_interval_sec
            ):
                self._retry_auto_mission()
            elif (
                self.mode_request_attempts >= self.config.max_mode_retries
                and now - self.last_mode_request_time
                >= self.config.mode_retry_interval_sec
            ):
                self.logger.error(
                    "Không thể trở lại AUTO.MISSION sau %d lần; tiếp tục gửi zero "
                    "setpoint. Hãy dùng công tắc RC để tiếp quản.",
                    self.config.max_mode_retries,
                )
                self.last_mode_request_time = now + 3600.0

    def safe_shutdown(self, timeout_sec: float = 2.0) -> None:
        """Nếu chương trình đang sở hữu OFFBOARD, cố đưa PX4 về mission trước khi thoát."""
        try:
            self.vehicle.poll()
            if not (
                self.offboard_owned_by_node
                and self.vehicle.connected
                and self.vehicle.mode == OFFBOARD_MODE
            ):
                return

            self.logger.warning("Chương trình sắp dừng; yêu cầu PX4 trở lại MISSION")
            deadline = time.monotonic() + timeout_sec
            next_request = 0.0
            while time.monotonic() < deadline:
                now = time.monotonic()
                self._send_zero()
                if now >= next_request:
                    self.vehicle.request_mode(AUTO_MISSION_MODE)
                    next_request = now + 0.5
                time.sleep(0.05)
                self.vehicle.poll()
                if self.vehicle.mode == AUTO_MISSION_MODE:
                    self.logger.info("PX4 đã trở lại AUTO.MISSION trước khi thoát")
                    self.offboard_owned_by_node = False
                    return

            self.logger.critical(
                "Không xác nhận được AUTO.MISSION trước khi thoát; PX4 sẽ dùng "
                "failsafe mất Offboard đã cấu hình"
            )
        except Exception as exc:
            self.logger.exception("Lỗi trong safe shutdown: %s", exc)


def _default_path(filename: str) -> str:
    return str(Path(__file__).resolve().parent / filename)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PX4 waypoint + ArUco tracking bằng pymavlink, không cần ROS"
    )
    parser.add_argument("--device", default="/dev/serial0")
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--heartbeat-wait", type=float, default=10.0)
    parser.add_argument("--connect-only", action="store_true")
    parser.add_argument("--connect-test-duration", type=float, default=5.0)

    parser.add_argument("--marker-size", type=float, default=0.27)
    parser.add_argument("--dictionary", default="DICT_6X6_250")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera-matrix", default=_default_path("cameraMatrix.txt"))
    parser.add_argument(
        "--dist-coeffs",
        default=_default_path("cameraDistortion.txt"),
    )
    parser.add_argument("--max-reprojection-error", type=float, default=5.0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--loop-rate", type=float, default=20.0)

    parser.add_argument("--hover-duration", type=float, default=5.0)
    parser.add_argument("--center-diameter", type=float, default=0.20)
    parser.add_argument("--target-lost-timeout", type=float, default=1.0)
    parser.add_argument("--vision-timeout", type=float, default=0.25)
    parser.add_argument("--offboard-prestream", type=float, default=1.1)
    parser.add_argument("--offboard-timeout", type=float, default=2.0)
    parser.add_argument("--mode-retry-interval", type=float, default=1.0)
    parser.add_argument("--max-mode-retries", type=int, default=3)
    parser.add_argument("--kp", type=float, default=0.50)
    parser.add_argument("--kd", type=float, default=0.05)
    parser.add_argument("--pose-filter-alpha", type=float, default=0.35)
    parser.add_argument("--derivative-filter-alpha", type=float, default=0.25)
    parser.add_argument("--deadband", type=float, default=0.02)
    parser.add_argument("--max-velocity", type=float, default=0.20)
    parser.add_argument("--max-acceleration", type=float, default=0.30)
    parser.add_argument("--hover-speed-threshold", type=float, default=0.10)
    return parser


def run_connection_test(vehicle: PymavlinkVehicle, duration_sec: float) -> None:
    LOGGER.info(
        "Chỉ kiểm tra kết nối trong %.1f s; không gửi setpoint và không đổi mode",
        duration_sec,
    )
    deadline = time.monotonic() + duration_sec
    last_state = None
    while time.monotonic() < deadline:
        now = time.monotonic()
        vehicle.poll()
        vehicle.send_companion_heartbeat(now)
        state = (vehicle.connected, vehicle.mode, vehicle.armed)
        if state != last_state:
            LOGGER.info(
                "PX4 connected=%s mode=%s armed=%s",
                vehicle.connected,
                vehicle.mode,
                vehicle.armed,
            )
            last_state = state
        time.sleep(0.05)


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    if args.loop_rate <= 2.0:
        raise ValueError("--loop-rate phải lớn hơn 2 Hz để đáp ứng offboard PX4")

    vehicle: PymavlinkVehicle | None = None
    camera = None
    controller: MissionController | None = None
    try:
        vehicle = PymavlinkVehicle(
            device=args.device,
            baud=args.baud,
            heartbeat_wait_sec=args.heartbeat_wait,
        )
        if args.connect_only:
            run_connection_test(vehicle, args.connect_test_duration)
            return

        from uco_publish import ArucoCamera

        camera = ArucoCamera(
            marker_size_m=args.marker_size,
            dictionary_name=args.dictionary,
            cam_width=args.width,
            cam_height=args.height,
            camera_matrix_file=args.camera_matrix,
            dist_coeffs_file=args.dist_coeffs,
            max_reprojection_error_px=args.max_reprojection_error,
            target_rate_hz=args.loop_rate,
            show_debug=args.debug,
        )
        camera.start()

        config = MissionConfig(
            hover_duration_sec=args.hover_duration,
            center_diameter_m=args.center_diameter,
            target_lost_timeout_sec=args.target_lost_timeout,
            vision_stream_timeout_sec=args.vision_timeout,
            offboard_prestream_sec=args.offboard_prestream,
            offboard_timeout_sec=args.offboard_timeout,
            mode_retry_interval_sec=args.mode_retry_interval,
            max_mode_retries=args.max_mode_retries,
            kp_xy=args.kp,
            kd_xy=args.kd,
            pose_filter_alpha=args.pose_filter_alpha,
            derivative_filter_alpha=args.derivative_filter_alpha,
            deadband_m=args.deadband,
            max_velocity_mps=args.max_velocity,
            max_acceleration_mps2=args.max_acceleration,
            hover_speed_threshold_mps=args.hover_speed_threshold,
        )
        controller = MissionController(vehicle, camera, config)

        period = 1.0 / args.loop_rate
        next_tick = time.monotonic()
        while True:
            now = time.monotonic()
            controller.step(now)
            next_tick += period
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                # Không cố chạy bù nhiều vòng vì sẽ tạo burst setpoint.
                next_tick = time.monotonic()
    except KeyboardInterrupt:
        LOGGER.info("Nhận Ctrl+C")
    except Exception:
        LOGGER.exception("Chương trình dừng do lỗi")
        raise
    finally:
        if controller is not None:
            controller.safe_shutdown()
        if camera is not None:
            camera.stop()
        if vehicle is not None:
            vehicle.close()


if __name__ == "__main__":
    main()
