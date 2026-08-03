source /opt/ros/humble/setup.bash

cd ~/PX4-Autopilot
make px4_sitl gz_x500

source /opt/ros/jazzy/setup.bash
ros2 run mavros mavros_node --ros-args -p fcu_url:=udp://:14540@127.0.0.1:14557


source /opt/ros/jazzy/setup.bash
cd ~/Aruco_auto
python3 mission.py
 
source /opt/ros/jazzy/setup.bash
cd ~/Aruco_auto
python3 laptop.py --marker-size 10.0 --cam-id 0

