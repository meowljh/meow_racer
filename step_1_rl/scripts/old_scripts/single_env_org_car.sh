#!/bin/bash


python tutorial_train.py \
    --use_jw 0 \
    --use_sde 1 \
    --lidar_deg 6 \
    --num_vecs 25 \
    --theta_diff 4 \
    --env_type random \
    --learning_rate 3e-4 \
    --skip_rate 1 \
    --min_movement 0.1 \
    --time_penalty 0.5 \
    --max_episodes 10000 \
    --total_timesteps 10000 \
    --max_reward_tile 4000 \
    --n_envs 0 \
    --reward_type "baseline" \
    --screen_title "RAND_SDE_ORGbaseline_0209_sac_6_25_4" \
    --save "RAND_SDE_ORGbaseline_0209_sac_6_25_4" \
    --random_checkpoints \
    --terminate_penalty 200 \
    --use_gas \
    --use_steer \
    --use_brake \
    --use_delta \
    --use_force