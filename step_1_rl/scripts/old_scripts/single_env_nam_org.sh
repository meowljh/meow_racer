#!/bin/bash


python tutorial_train.py \
    --use_jw 0 \
    --env_type nam \
    --learning_rate 3e-4 \
    --skip_rate 1 \
    --max_episodes 10000 \
    --learning_start 100 \
    --max_reward_tile 2500 \
    --min_movement 0.1 \
    --time_penalty 0.5 \
    --n_envs 0 \
    --reward_type "baseline" \
    --screen_title "NAM_ORGbaseline_0207_sac" \
    --save "NAM_ORGbaseline_0206_sac" \
    --terminate_penalty 200 \
    --use_gas \
    --use_steer \
    --use_brake \
    --use_delta \
    --use_force