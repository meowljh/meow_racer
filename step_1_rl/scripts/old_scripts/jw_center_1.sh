#!/bin/bash


FNAME="CENTER_BOTH_RAND4_REV_RANDSTART_JW_0227_3515"
 

python tutorial_train.py \
    --env_type random \
    --random_checkpoints \
    --center_line_max_penalty 10. \
    --num_random_checkpoints 4 \
    --use_jw 1 \
    --use_sde 0 \
    --do_reverse  0.5 \
    --random_start 1 \
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
    --learning_rate 3e-4 \
    --reward_type "center" \
    --screen_title $FNAME \
    --save $FNAME \
    --terminate_penalty 200 