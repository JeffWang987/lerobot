bash /home/agilex/jeff/kill_ros.sh

cd ~/cobot_magic/Piper_ros_private-ros-noetic/
./can_config.sh

conda activate aloha
roslaunch piper start_ms_piper.launch mode:=1 auto_enable:=true


conda activate lerobot_hil-serl
cd ~/lerobot_hil-serl


python -m lerobot.rl.gym_manipulator --config_path /home/agilex-home/agilex/lerobot_hil-serl/config/single_piper_gym.json






python -m lerobot.rl.learner --config_path config/train_sac_policy_learner_yag.json

python -m lerobot.rl.actor --config_path config/train_sac_policy_actor_yag.json

python -m lerobot.rl.actor --config_path config/train_sac_policy_actor.json
python -m lerobot.rl.learner --config_path config/train_sac_policy_learner.json


python -m lerobot.rl.learner_offline --config_path config/train_sac_policy_offline.json



