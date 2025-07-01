#!/bin/bash
FNAME="BOTH_RAND4_JWbaseline_0226_3_5_15"
 

python tutorial_train.py \
    --env_type random \
    --random_checkpoints \
    --num_random_checkpoints 4 \
    --use_jw 1 \
    --use_sde 0 \
    --do_reverse  0.0 \
    --random_start 0 \
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

# FNAME="BOTH_RAND4_JWbaseline_0218_3_5_15"
 

# python tutorial_train.py \
#     --env_type random \
#     --random_checkpoints \
#     --num_random_checkpoints 4 \
#     --use_jw 1 \
#     --use_sde 0 \
#     --do_reverse  0.0 \
#     --random_start 0 \
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
#     --learning_rate 1e-4 \
#     --reward_type "baseline" \
#     --screen_title $FNAME \
#     --save $FNAME \
#     --terminate_penalty 200 \
#     --use_gas \
#     --use_steer \
#     --use_brake \
#     --use_delta \
#     --use_force

# FNAME="BOTH_REV_NORANDSTART_JWbaseline_0217_3_5_15"
 

# python tutorial_train.py \
#     --env_type random \
#     --random_checkpoints \
#     --use_jw 1 \
#     --use_sde 0 \
#     --do_reverse  0.5 \
#     --random_start 0 \
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
#     --terminate_penalty 200 \
#     --use_gas \
#     --use_steer \
#     --use_brake \
#     --use_delta \
#     --use_force

# FNAME="BOTH_NOREV_RANDSTART_JWbaseline_0217_3_5_15"
 

# python tutorial_train.py \
#     --env_type random \
#     --random_checkpoints \
#     --use_jw 1 \
#     --use_sde 0 \
#     --do_reverse  0.0 \
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
#     --terminate_penalty 200 \
#     --use_gas \
#     --use_steer \
#     --use_brake \
#     --use_delta \
#     --use_force

# TITLE_NAME="RAND_JW_REV_SAC_0217_3_5_15"

# python tutorial_train.py \
#     --env_type random \
#     --use_jw 1 \
#     --use_sde 0 \
#     --do_reverse 0.3 \
#     --weight_save_converge 1 \
#     --weight_save_interval 5 \
#     --random_checkpoints \
#     --use_jw 1 \
#     --num_vecs 15 \
#     --theta_diff 5 \
#     --lidar_deg 3 \
#     --both_track_ratio 0.2 \
#     --skip_rate 1 \
#     --max_episodes 10000 \
#     --max_reward_tile 4000 \
#     --min_movement 0.1 \
#     --time_penalty 0.5 \
#     --n_envs 0 \
#     --learning_rate 3e-4 \
#     --reward_type "baseline" \
#     --screen_title $TITLE_NAME \
#     --save $TITLE_NAME \
#     --terminate_penalty 200 \
#     --use_gas \
#     --use_steer \
#     --use_brake \
#     --use_delta \
#     --use_force

# TITLE_NAME="dTHETA_RAND_JW_SAC_0213_3_5_15"

# python tutorial_train.py \
#     --env_type random \
#     --weight_save_converge 1 \
#     --weight_save_interval 5 \
#     --use_jw 1 \
#     --use_sde 0 \
#     --use_theta_diff 1 \
#     --num_vecs 15 \
#     --theta_diff 5 \
#     --lidar_deg 3 \
#     --skip_rate 1 \
#     --max_episodes 10000 \
#     --max_reward_tile 5000 \
#     --min_movement 0.1 \
#     --min_theta_movement 0.1 \
#     --time_penalty 0.5 \
#     --n_envs 0 \
#     --reward_type "baseline" \
#     --screen_title $TITLE_NAME \
#     --save $TITLE_NAME \
#     --terminate_penalty 200 \
#     --use_gas \
#     --use_steer \
#     --use_brake \
#     --use_delta \
#     --use_force