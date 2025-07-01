#!/bin/bash

FNAME="SMOOTH_BOTH_RAND4_REV_RANDSTART_JW_0327_3515"

SAC_ACTION_NOISE="none"
SAC_ENT_COEF="auto_0.1"
N_ENVS=0 # 4

python tutorial_train.py \
    --rl_algorithm sac \
    --random_seed 22 \
    --env_type random \
    --action_noise_type $SAC_ACTION_NOISE \
    --random_checkpoints \
    --ent_coef $SAC_ENT_COEF \
    --oscillation_penalty 0. \
    --num_random_checkpoints 4 \
    --use_jw 1 \
    --use_sde 0 \
    --do_reverse 0.5 \
    --random_start 1 \
    --n_steps_td 0 \
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
    --n_envs $N_ENVS \
    --learning_rate 3e-4 \
    --reward_type "baseline" \
    --screen_title $FNAME \
    --save $FNAME \
    --terminate_penalty 200 

# FNAME="BOTH_SCHED_PEN_RAND4_REV_RANDSTART_JW_0227_3515"
 

# python tutorial_train.py \
#     --env_type random \
#     --random_checkpoints \
#     --oscillation_penalty 1. \
#     --num_random_checkpoints 4 \
#     --use_jw 1 \
#     --use_sde 0 \
#     --do_reverse  0.5 \
#     --random_start 1 \
#     --use_theta_diff 0 \
#     --weight_save_converge 1 \
#     --weight_save_interval 5 \
#     --num_vecs 15 \
#     --theta_diff 5 \
#     --lidar_deg 3 \
#     --both_track_ratio 0.1 \
#     --skip_rate 1 \
#     --max_episodes 10000 \
#     --max_reward_tile 5000 \
#     --min_movement 0.1 \
#     --time_penalty 0.5 \
#     --n_envs 0 \
#     --learning_rate 3e-4 \
#     --reward_type "baseline" \
#     --screen_title $FNAME \
#     --save $FNAME \
#     --terminate_penalty 200 

# FNAME="JW_PEN_ENVONLY_RANDTRACK_RANDSTART_REV_0225_3515"

# python tutorial_train.py \
#     --env_type random \
#     --use_jw 1 \
#     --straight_kappa_limit 90. \
#     --random_checkpoints \
#     --num_random_checkpoints 4 \
#     --use_theta_diff 0 \
#     --oscillation_penalty 1. \
#     --both_track_ratio 0. \
#     --random_start 1 \
#     --do_reverse 0.5 \
#     --use_sde 0 \
#     --weight_save_converge 1 \
#     --weight_save_interval 5 \
#     --num_vecs 15 \
#     --theta_diff 5 \
#     --lidar_deg 3 \
#     --skip_rate 1 \
#     --max_episodes 10000 \
#     --max_reward_tile 5000 \
#     --min_movement 0.1 \
#     --time_penalty 0.5 \
#     --n_envs 0 \
#     --learning_rate 3e-4 \
#     --reward_type "baseline" \
#     --screen_title $FNAME \
#     --save $FNAME \
#     --terminate_penalty 200

# FNAME="JW_PEN_ENVONLY_BOTH_RANDSTART_REV_0224_3515"

# python tutorial_train.py \
#     --env_type random \
#     --use_jw 1 \
#     --random_checkpoints \
#     --oscillation_penalty 1. \
#     --both_track_ratio 0.1 \
#     --random_start 1 \
#     --do_reverse 0.5 \
#     --use_sde 0 \
#     --weight_save_converge 1 \
#     --weight_save_interval 5 \
#     --num_vecs 15 \
#     --theta_diff 5 \
#     --lidar_deg 3 \
#     --both_track_ratio 0.1 \
#     --skip_rate 1 \
#     --max_episodes 10000 \
#     --max_reward_tile 5000 \
#     --min_movement 0.1 \
#     --time_penalty 0.5 \
#     --n_envs 0 \
#     --learning_rate 3e-4 \
#     --reward_type "baseline" \
#     --screen_title $FNAME \
#     --save $FNAME \
#     --terminate_penalty 200