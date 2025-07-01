#!/bin/bash
FNAME="RAND4_PEN_ONLYENV_REV_JWbaseline_0301_3_5_15"

python tutorial_train.py \
    --env_type random \
    --random_start 1 \
    --random_checkpoints \
    --oscillation_penalty 1. \
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
    --learning_rate 3e-4 \
    --reward_type "baseline" \
    --screen_title $FNAME \
    --save $FNAME \
    --terminate_penalty 200

# FNAME="BOTH_RAND4_REV_RANDSTART_JW_0226_3515"
 

# python tutorial_train.py \
#     --env_type random \
#     --random_checkpoints \
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


# FNAME="BOTH_REV_NOSTARTJWbaseline_0217_3_5_15"

# python tutorial_train.py \
#     --env_type random \
#     --random_start 1 \
#     --random_checkpoints \
#     --use_jw 1 \
#     --use_sde 0 \
#     --do_reverse 0.5 \
#     --use_theta_diff 0 \
#     --weight_save_converge 1 \
#     --weight_save_interval 5 \
#     --num_vecs 15 \
#     --theta_diff 5 \
#     --lidar_deg 3 \
#     --both_track_ratio 0. \
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
#     --terminate_penalty 500 \
#     --use_gas \
#     --use_steer \
#     --use_brake \
#     --use_delta \
#     --use_force

# FNAME="BOTH_REV_JWbaseline_0217_3_5_15"

# python tutorial_train.py \
#     --env_type random \
#     --random_start 1 \
#     --random_checkpoints \
#     --use_jw 1 \
#     --use_sde 0 \
#     --do_reverse 0.5 \
#     --use_theta_diff 0 \
#     --weight_save_converge 1 \
#     --weight_save_interval 5 \
#     --num_vecs 15 \
#     --theta_diff 5 \
#     --lidar_deg 3 \
#     --both_track_ratio 0. \
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




# FNAME="BOTH_SDE_JWbaseline_0214_5_3_20"

# python tutorial_train.py \
#     --use_sde 0 \
#     --use_jw 1 \
#     --env_type random \
#     --learning_rate 3e-4 \
#     --do_reverse 0.5 \
#     --skip_rate 1 \
#     --lidar_deg 5 \
#     --num_vecs 20 \
#     --theta_diff 3 \
#     --weight_save_converge 1 \
#     --weight_save_interval 5 \
#     --min_movement 0.1 \
#     --time_penalty 0.5 \
#     --max_episodes 10000 \
#     --total_timesteps 10000 \
#     --max_reward_tile 5000 \
#     --n_envs 0 \
#     --reward_type "baseline" \
#     --screen_title $FNAME \
#     --save $FNAME \
#     --random_checkpoints \
#     --terminate_penalty 200 \
#     --use_gas \
#     --use_steer \
#     --use_brake \
#     --use_delta \
#     --use_force