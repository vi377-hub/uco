#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import time

from geometry_msgs.msg import TwistStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import SetMode
from std_msgs.msg import Float32MultiArray

class MissionController(Node):
    def __init__(self):
        super().__init__('mission_controller_node')

        # --- CẤU HÌNH BAY THỰC TẾ (HỆ MÉT) ---
        self.hover_duration_sec = 2.0
        self.center_threshold_m = 0.20
        self.aruco_timeout_sec = 0.5
        self.align_lost_timeout_sec = 0.5
        self.offboard_timeout_sec = 1.0
        self.deadband_m = 0.02  # Bỏ qua sai số < 2cm để tránh rung quanh tâm do nhiễu detect

        # --- BỘ ĐIỀU KHIỂN PD (Đã nhân 100 so với bản hệ cm) ---
        self.kp_xy = 1.5  # Hệ số tỉ lệ (Lực kéo về tâm)
        self.kd_xy = 0.5  # Hệ số vi phân (Lực phanh hãm quán tính)

        # Biến lưu trữ trạng thái trước đó để tính Đạo hàm (D)
        self.prev_x_m = 0.0
        self.prev_y_m = 0.0
        self.last_pid_time = time.time()
        self.first_align_loop = True

        self.handled_ids = set()
        self.current_state = 'MISSION'
        self.wait_offboard_start = None
        self.uav_state = State()
        self.aruco_data = None
        self.last_aruco_time = 0.0
        self.hover_start_time = 0.0
        self.current_target_id = None

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.state_sub = self.create_subscription(State, '/mavros/state', self.state_cb, qos_profile)
        self.vision_sub = self.create_subscription(Float32MultiArray, '/vision/aruco_pose', self.vision_cb, qos_profile)
        self.vel_pub = self.create_publisher(TwistStamped, '/mavros/setpoint_velocity/cmd_vel', 10)
        self.set_mode_client = self.create_client(SetMode, '/mavros/set_mode')

        self.timer = self.create_timer(0.05, self.control_loop)

        self.get_logger().info(">>> MISSION CONTROLLER (PD CONTROL - HỆ MÉT ĐỒNG BỘ) ĐÃ KHỞI ĐỘNG <<<")

    def state_cb(self, msg):
        self.uav_state = msg

    def vision_cb(self, msg):
        self.aruco_data = msg.data
        self.last_aruco_time = time.time()

    def set_mode(self, custom_mode):
        if self.set_mode_client.wait_for_service(timeout_sec=1.0):
            req = SetMode.Request()
            req.custom_mode = custom_mode
            self.set_mode_client.call_async(req)
            self.get_logger().info(f"==> Yêu cầu đổi mode PX4 sang: {custom_mode}")

    def publish_velocity(self, vx, vy, vz, yaw_rate=0.0):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.twist.linear.x = float(vx)
        msg.twist.linear.y = float(vy)
        msg.twist.linear.z = float(vz)
        msg.twist.angular.x = 0.0
        msg.twist.angular.y = 0.0
        msg.twist.angular.z = float(yaw_rate)
        self.vel_pub.publish(msg)

    def reset_controller(self):
        """Reset toàn bộ trạng thái nội bộ của PD controller trước khi bắt đầu 1 lượt align mới."""
        self.hover_start_time = 0.0
        self.first_align_loop = True
        self.prev_x_m = 0.0
        self.prev_y_m = 0.0
        self.last_pid_time = time.time()

    def control_loop(self):
        # 1. TRẠNG THÁI BAY LỘ TRÌNH
        if self.current_state == 'MISSION':
            self.publish_velocity(0.0, 0.0, 0.0)

            if self.uav_state.mode not in ["AUTO.MISSION", "OFFBOARD"]:
                return

            if self.aruco_data is not None and (time.time() - self.last_aruco_time < self.aruco_timeout_sec):
                detected_id = int(self.aruco_data[0])

                if detected_id not in self.handled_ids:
                    self.get_logger().info(f"*** PHÁT HIỆN MÃ ID {detected_id}! Yêu cầu chuyển OFFBOARD ***")
                    self.current_target_id = detected_id
                    self.set_mode("OFFBOARD")
                    self.wait_offboard_start = time.time()
                    self.current_state = 'WAIT_OFFBOARD'

        # 2. TRẠNG THÁI CHỜ XÁC NHẬN OFFBOARD (chưa chạy PD ở đây)
        elif self.current_state == 'WAIT_OFFBOARD':
            self.publish_velocity(0.0, 0.0, 0.0)  # vẫn giữ luồng setpoint cho watchdog PX4

            if self.uav_state.mode == "OFFBOARD":
                self.get_logger().info("OFFBOARD đã được chấp nhận. Bắt đầu ALIGN_AND_HOVER.")
                self.reset_controller()
                self.current_state = 'ALIGN_AND_HOVER'

            elif time.time() - self.wait_offboard_start > self.offboard_timeout_sec:
                self.get_logger().warn("! TIMEOUT CHỜ OFFBOARD ! Huỷ, quay lại MISSION.")
                self.current_target_id = None
                self.current_state = 'MISSION'

        # 3. TRẠNG THÁI CĂN TÂM & HOVER
        elif self.current_state == 'ALIGN_AND_HOVER':
            if time.time() - self.last_aruco_time > self.align_lost_timeout_sec:
                self.get_logger().warn("! MẤT DẤU QUÁ LÂU ! Trả quyền về GPS, bay tiếp lộ trình.")
                self.set_mode("AUTO.MISSION")
                self.current_state = 'MISSION'
                self.current_target_id = None
                return

            marker_id, x_m, y_m, z_m = self.aruco_data
            marker_id = int(marker_id)

            if marker_id != self.current_target_id:
                return

            # Deadband: sai số quá nhỏ (nhiễu detect) thì coi như bằng 0, tránh rung quanh tâm
            if abs(x_m) < self.deadband_m:
                x_m = 0.0
            if abs(y_m) < self.deadband_m:
                y_m = 0.0

            now = time.time()
            dt = now - self.last_pid_time
            if dt <= 0:
                dt = 0.05

            if self.first_align_loop:
                self.prev_x_m = x_m
                self.prev_y_m = y_m
                self.first_align_loop = False

            dx = (x_m - self.prev_x_m) / dt
            dy = (y_m - self.prev_y_m) / dt

            # LUẬT ĐIỀU KHIỂN PD
            vx = (y_m * self.kp_xy) + (dy * self.kd_xy)
            vy = (x_m * self.kp_xy) + (dx * self.kd_xy)

            self.prev_x_m = x_m
            self.prev_y_m = y_m
            self.last_pid_time = now

            max_vel = 0.5
            vx = max(-max_vel, min(max_vel, vx))
            vy = max(-max_vel, min(max_vel, vy))

            self.publish_velocity(vx, vy, 0.0, yaw_rate=0.0)

            # Tính toán độ lệch theo mét
            error_distance = (x_m**2 + y_m**2)**0.5

            if error_distance < self.center_threshold_m:
                if self.hover_start_time == 0.0:
                    self.hover_start_time = time.time()
                    self.get_logger().info(f"Đã khóa tâm ID {self.current_target_id}. Bắt đầu đếm ngược HOVER {self.hover_duration_sec}s...")

                elif time.time() - self.hover_start_time > self.hover_duration_sec:
                    self.get_logger().info(f"==> DONE NHIỆM VỤ TẠI ID {self.current_target_id}. Tiếp tục lộ trình.")
                    self.handled_ids.add(self.current_target_id)
                    self.current_target_id = None
                    self.hover_start_time = 0.0
                    self.set_mode("AUTO.MISSION")
                    self.current_state = 'MISSION'
            else:
                self.hover_start_time = 0.0

def main(args=None):
    rclpy.init(args=args)
    node = MissionController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
