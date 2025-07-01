#!/bin/bash

TITLE_NAME="BOTH_SDE_REV_JW_0214_sac_3_5_15"

python tutorial_train.py \
    --env_type random \
    --use_sde 1 \
    --do_reverse 0.5 \
    --use_theta_diff 0 \
    --weight_save_converge 1 \
    --weight_save_interval 5 \
    --use_jw 1 \
    --num_vecs 15 \
    --theta_diff 5 \
    --lidar_deg 3 \
    --both_track_ratio 0.1 \
    --skip_rate 1 \
    --max_episodes 10000 \
    --max_reward_tile 5000 \
    --min_movement 0.1 \
    --time_penalty 0.5 \
    --n_envs 0 \
    --reward_type "baseline" \
    --screen_title $TITLE_NAME \
    --save $TITLE_NAME \
    --terminate_penalty 500 \
    --use_gas \
    --use_steer \
    --use_brake \
    --use_delta \
    --use_force