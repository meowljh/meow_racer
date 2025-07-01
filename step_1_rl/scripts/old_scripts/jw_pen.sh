#!/bin/bash
FNAME="RAND4_1e3PEN_ONLYENV_REV_JWbaseline_0228_3_5_15"

python tutorial_train.py \
    --env_type random \
    --random_start 1 \
    --oscillation_penalty 1. \
    --straight_kappa_limit 1e-3 \
    --random_checkpoints \
    --num_random_checkpoints 4 \
    --use_jw 1 \
    --use_sde 0 \
    --do_reverse 0.5 \
    --use_theta_diff 0 \
    --weight_save_converge 1 \
    --weight_save_interval 5 \
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
    --learning_rate 1e-4 \
    --reward_type "baseline" \
    --screen_title $FNAME \
    --save $FNAME \
    --terminate_penalty 200