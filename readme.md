bash /home/agilex-home/agilex/jeff/kill_ros.sh

~/cobot_magic/Piper_ros_private-ros-noetic/can_config.sh
cd ~/lerobot_hil-serl/
conda activate lerobot_hil-serl


conda activate aloha
roslaunch piper start_ms_piper.launch mode:=1 auto_enable:=true


export PYTHONPATH=/home/agilex-home/agilex/dengqiuping/code/lerobot/src:$PYTHONPATH
python src/lerobot/find_cameras.py dabai
python src/lerobot/find_cameras.py realsense