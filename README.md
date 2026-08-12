# PX4 ArUco Mission

Dự án dùng Raspberry Pi, Pi Camera Module 3 và ArUco để hỗ trợ drone PX4 tự động căn tâm marker trong khi thực hiện waypoint mission.

Camera nhận diện marker `DICT_6X6_250`, dùng thông số calibration và `solvePnP` để tính vị trí tương đối. Khi drone đang ở `AUTO.MISSION`, chương trình khóa marker gần nhất, chuyển sang `OFFBOARD`, điều khiển vận tốc theo bộ điều khiển PD để căn tâm và hover 5 giây, sau đó trở lại mission. Nếu phi công đổi mode, chương trình dừng quyền điều khiển tự động.

## File chính
- `mission.py`: kết nối PX4 qua MAVLink và điều khiển mission/Offboard.
- `uco_publish.py`: đọc Pi Camera trong luồng riêng và cung cấp pose ArUco.
- `lib_aruco_pose.py`: nhận diện marker, ước lượng pose và lọc theo reprojection error.
- `camera_calib.py`: calibration Pi Camera bằng bảng ChArUco.


