#!/bin/bash 

python tutorial_train.py \
    --min_movement 0.1 \
    --time_penalty 0.5 \
    --n_envs 8 \
    --reward_type "baseline" \
    --screen_title "baseline_0114_sac_mm01_tp05 8ENV" \
    --save "baseline_0114_sac_mm01_tp05 8ENV" \
    --random_checkpoints \
    --use_force \
    --use_delta \
    --use_gas \
    --use_steer \
    --use_brake \
    --terminate_penalty 500