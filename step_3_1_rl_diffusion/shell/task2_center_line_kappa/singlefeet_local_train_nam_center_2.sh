#!/bin/bash
USE_NAM_ONLY=true

USE_TIME_PENALTY=true

USE_CONTINUOUS_BPS=true

LOOKAHEAD_NUM=20
LOOKAHEAD_THETA=20 #단위:m

LIDAR_NUM=30 #약 6도 간격으로 lidar 거리 계산

WORLD_DT=0.05 #0.1
ACTION_DT=0.01

ACTION_DIM=3
ALLOW_BOTH_FEET=false
ALLOW_NEG_TORQUE=true
ALWAYS_POS_VEL=0

E_C_PENALTY_WEIGHT=5
OFF_COURSE_PENALTY_WEIGHT=2
NEG_VEL_BPS_PENALTY_WEIGHT=1

USE_E_PHI_REWARD=true #false
E_PHI_REWARD_VALUE=1 

# USE_PROGRESS_REWARD=true #false
# PROGRESS_REWARD_VALUE=1

USE_PROGRESS_REWARD_CURVE=true
PROGRESS_REWARD_CURVE_VALUE=1 #0.1


USE_KAPPA_VEL_REWARD=true
KAPPA_VEL_REWARD_VALUE=5 #1
KAPPA_VEL_TARGET_THETA=20
KAPPA_VEL_FUTURE_MODE=mean
KAPPA_VEL_CONTINUOUS=true

USE_MIN_VEL_PENALTY=true #false
MIN_VEL_PEN_VALUE=3 #1
MIN_VEL=1e-3

USE_APS_BPS_DIFF=true
CAR_INIT_VX=1

TERMINATE_PENALTY=500

PENALTY_CONDITION=off_course_count

TERMINATE_VEL_COND=count_time_neg_vel
TERMINATE_VEL_PATIENCE=0.1

EXP_NAME=0520_SingleFeet_Nam_Act3_Center_timePen_minVPen_offCnt_ProgressRewardCurve

python train_main.py \
    agent.exp_name=$EXP_NAME \
    agent.policy.layer_norm=false \
    agent.algorithm.warmup_actor_step=-1.\
    agent.algorithm.batch_size=256 \
    agent.algorithm.num_epochs=100000 \
    agent.algorithm.min_num_steps_before_training=1000 \
    agent.algorithm.num_expl_steps_per_train_loop=5000 \
    agent.algorithm.num_trains_per_train_loop=5000 \
    agent.algorithm.num_train_loops_per_epoch=1 \
    agent.algorithm.max_path_length=1e+12 \
    agent.algorithm.num_step_for_expl_data_collect=1 \
    agent.replay_buffer_size=1000000 \
    agent.test.test_max_path_length=1e+12 \
    agent.test.test_log_path="D:/meow_racer_experiments" \
    environment.random_seed=77 \
    environment.observation.vx.usage=true \
    environment.observation.vy.usage=true \
    environment.observation.lookahead.usage=true \
    environment.observation.lookahead.lookahead_time=1 \
    environment.observation.lookahead.num_states=$LOOKAHEAD_NUM \
    environment.observation.lookahead.coords=false \
    environment.observation.lookahead.curvature=true \
    environment.observation.lookahead.scale_method=standard \
    environment.observation.lookahead.lookahead_theta=$LOOKAHEAD_THETA \
    environment.observation.lookahead.fixed=theta \
    environment.observation.forward_vector.usage=true \
    environment.observation.forward_vector.rotate_vec=false \
    environment.observation.lidar_sensor.usage=true \
    environment.observation.lidar_sensor.scale_method=minmax \
    environment.observation.lidar_sensor.num_lidar=$LIDAR_NUM \
    environment.track.track_density=1 \
    environment.vehicle.dt=$ACTION_DT \
    environment.vehicle.world_dt=$WORLD_DT \
    environment.vehicle.brake_on_pos_vel=$BRAKE_ON_POS_VEL \
    environment.vehicle.allow_both_feet=$ALLOW_BOTH_FEET \
    environment.vehicle.allow_neg_torque=$ALLOW_NEG_TORQUE \
    environment.vehicle.initial_vx=$CAR_INIT_VX \
    environment.vehicle.use_continuous_bps=$USE_CONTINUOUS_BPS \
    environment.vehicle.always_pos_vel=$ALWAYS_POS_VEL \
    environment.vehicle.use_aps_bps_diff=$USE_APS_BPS_DIFF \
    environment.action.action_dim=$ACTION_DIM \
    environment.do_debug_logs=0 \
    environment.track.use_nam_only=$USE_NAM_ONLY \
    environment.track.nam_ratio=0.1 \
    environment.vehicle.aps_bps_weight=1. \
    penalty.terminate.penalty_value=$TERMINATE_PENALTY \
    penalty.time_penalty.usage=$USE_TIME_PENALTY \
    penalty.terminate.off_course_condition=off_course_tire \
    penalty.terminate.neg_vel_patience=$TERMINATE_VEL_PATIENCE \
    penalty.off_course_penalty.usage=true \
    penalty.off_course_penalty.ratio_usage=true \
    penalty.off_course_penalty.penalty_value=$OFF_COURSE_PENALTY_WEIGHT \
    penalty.off_course_penalty.condition=$PENALTY_CONDITION \
    penalty.E_c_penalty.usage=true \
    penalty.E_c_penalty.penalty_value=$E_C_PENALTY_WEIGHT \
    penalty.E_c_penalty.normalize_E_c=true \
    penalty.neg_velocity_bps_penalty.usage=true \
    penalty.neg_velocity_bps_penalty.penalty_value=$NEG_VEL_BPS_PENALTY_WEIGHT \
    penalty.min_velocity_penalty.usage=$USE_MIN_VEL_PENALTY \
    penalty.min_velocity_penalty.penalty_value=$MIN_VEL_PEN_VALUE \
    penalty.min_velocity_penalty.min_velocity=$MIN_VEL \
    reward.progress_reward_curve.usage=false \
    reward.track_align_velocity_reward.usage=false \
    reward.target_velocity_reward.usage=false \
    reward.E_phi_reward.usage=$USE_E_PHI_REWARD \
    reward.E_phi_reward.reward_value=$E_PHI_REWARD_VALUE \
    reward.curvature_vel_reward.usage=$USE_KAPPA_VEL_REWARD \
    reward.curvature_vel_reward.reward_value=$KAPPA_VEL_REWARD_VALUE \
    reward.curvature_vel_reward.target_theta=$KAPPA_VEL_TARGET_THETA \
    reward.curvature_vel_reward.future_mode=$KAPPA_VEL_FUTURE_MODE \
    reward.curvature_vel_reward.continuous=$KAPPA_VEL_CONTINUOUS \
    reward.progress_reward_curve.usage=$USE_PROGRESS_REWARD_CURVE \
    reward.progress_reward_curve.reward_weight=$PROGRESS_REWARD_CURVE_VALUE

