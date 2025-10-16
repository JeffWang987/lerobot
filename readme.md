~/cobot_magic/Piper_ros_private-ros-noetic/can_config.sh
cd ~/lerobot_hil-serl/
conda activate lerobot_hil-serl


conda activate aloha
roslaunch piper start_ms_piper.launch mode:=1 auto_enable:=true