#!/bin/bash
NSTEP=0
BATCH_SIZE=256
LEARNING_RATE=3e-4
FNAME="OSCI_NEWPEN2_RAND4_ONLYENV_REV_JWbaseline_0325_3_5_15"
ACTION_NOISE_TYPE="none"

python tutorial_train.py \
    --random_seed 43 \
    --rl_algorithm sac \
    --action_noise_type $ACTION_NOISE_TYPE \
    --env_type random \
    --body_left_penalty 10. \
    --body_left_mode percent \
    --do_penalty_max_reward 0 \
    --oscillation_penalty 0. \
    --oscillation_max_penalty 5. \
    --straight_kappa_limit 0.001 \
    --consider_forward_vec 1 \
    --random_start 1 \
    --random_checkpoints \
    --n_steps_td $NSTEP \
    --batch_size $BATCH_SIZE \
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
    --learning_rate $LEARNING_RATE \
    --reward_type "baseline" \
    --screen_title $FNAME \
    --save $FNAME \
    --terminate_penalty 200