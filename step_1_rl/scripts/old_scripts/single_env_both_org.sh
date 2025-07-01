#!/bin/bash
FNAME="BOTH_ORG_0214_3_5_15"
python tutorial_train.py \
    --env_type random \
    --use_jw 0 \
    --do_reverse 0 \
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
    --screen_title $FNAME \
    --save $FNAME \
    --terminate_penalty 200 \
    --use_gas \
    --use_steer \
    --use_brake \
    --use_delta \
    --use_force
