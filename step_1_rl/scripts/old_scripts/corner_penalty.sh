#!/bin/bash

python tutorial_train.py \
    --min_movement 0.1 \
    --time_penalty 0.5 \
    --n_envs 0 \
    --reward_type "baseline" \
    --screen_title "baseline_0114_sac_mm01_tp05" \
    --save "baseline_0114_sac_mm01_tp05" \
    --random_checkpoints \
    --terminate_penalty 1000