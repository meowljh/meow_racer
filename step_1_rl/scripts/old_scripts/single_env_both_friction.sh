#!/bin/bash


python tutorial_train.py \
    --env_type random \
    --both_track_ratio 0.2 \
    --replay_buffer_class None \
    --new_friction_limit 0.5 \
    --gamma 0.99 \
    --ent_coef 0.1 \
    --skip_rate 1 \
    --max_episodes 10000 \
    --max_reward_tile 4000 \
    --min_movement 0.1 \
    --time_penalty 0.5 \
    --n_envs 0 \
    --reward_type "baseline" \
    --screen_title "BOTH_FRIC1e5baseline_0204_sac" \
    --save "BOTH_FRIC1e5baseline_0204_sac" \
    --terminate_penalty 200 \
    --use_gas \
    --use_steer \
    --use_brake \
    --use_delta \
    --use_force