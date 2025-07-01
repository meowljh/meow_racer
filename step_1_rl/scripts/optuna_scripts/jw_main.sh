#!/bin/bash
TITLE_NAME="RAND_optuna_0210_sac_SDE_JW"

python optuna_tutorial_train.py \
    --use_jw 1 \
    --use_sde 1 \
    --env_type random \
    --num_vecs 15 \
    --theta_diff 5 \
    --lidar_deg 3 \
    --max_episodes 10000 \
    --max_reward_tile 5000 \
    --terminate_penalty 200 \
    --min_movement 0.1 \
    --time_penalty 0.5 \
    --n_envs 8 \
    --reward_type "baseline" \
    --use_gas --use_steer --use_brake --use_delta --use_force \
    --screen_title $TITLE_NAME \
    --save $TITLE_NAME \
    --tensorboard_folder $TITLE_NAME