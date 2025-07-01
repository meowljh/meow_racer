#!/bin/bash
NSTEP=0
BATCH_SIZE=256
LEARNING_RATE=3e-4
RAND_CKPT=4
MOVEMENT_MIN_BOUND=0.1
TIME_PASS_PENALTY=0.5

KAPPA_LIMIT_FOR_STRAIGHT=0.001
CONSIDER_OSCILLATION_FORWARD_VEC=1

MAX_BODY_LEFT_PENALTY=10.

#### [OSCILLATION]: penalty for straight, reward for corner ####
MAX_OSCILLATION_PENALTY=5.
MAX_OSCILLATION_REWARD_CORNER=1.
#### [CENTER LINE DISTANCE]: NO reward for straight, LARGE reward for corner ####
## but, considers the forward vectors before entering the corner of the track because, when we are about to change the steering angle to enter the corner, oscillation is necessary for out-in-out
# in other words, the distance from the center line does not matter much. but being far from the center line is more necessary
# corner 구간에서는 center line으로부터 떨어지면서 동시에 많이 side로 이동하는게 reward가 크게 됨.
MAX_CENTER_LINE_FAR_REWARD=0.
MAX_CENTER_LINE_FAR_REWARD_CORNER=1.
#### [DELTA THETA]: LARGE reward for corner ####
DELTA_THETA_WEIGHT=

FNAME="STATEDEP_OSCI_NEWPEN3_JWbaseline_0310_3_5_15"

python tutorial_train.py \
    --env_type random \
    --body_left_penalty $MAX_BODY_LEFT_PENALTY \
    --center_line_far_max_reward $MAX_CENTER_LINE_FAR_REWARD \
    --center_line_far_max_reward_corner $MAX_CENTER_LINE_FAR_REWARD_CORNER \
    --body_left_mode percent \
    --do_penalty_max_reward 0 \
    --oscillation_penalty 0 \
    --oscillation_max_penalty $MAX_OSCILLATION_PENALTY \
    --oscillation_max_reward_corner $MAX_OSCILLATION_REWARD_CORNER \
    --straight_kappa_limit $KAPPA_LIMIT_FOR_STRAIGHT \
    --consider_forward_vec $CONSIDER_OSCILLATION_FORWARD_VEC \
    --random_start 1 \
    --random_checkpoints \
    --n_steps_td $NSTEP \
    --batch_size $BATCH_SIZE \
    --num_random_checkpoints $RAND_CKPT \
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
    --min_movement $MOVEMENT_MIN_BOUND \
    --time_penalty $TIME_PASS_PENALTY \
    --n_envs 0 \
    --learning_rate $LEARNING_RATE \
    --reward_type "baseline" \
    --screen_title $FNAME \
    --save $FNAME \
    --terminate_penalty 200