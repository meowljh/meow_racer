#!/bin/bash
USE_NAM_ONLY=true

USE_TIME_PENALTY=false

USE_CONTINUOUS_BPS=true

LOOKAHEAD_NUM=20
LOOKAHEAD_THETA=20 #단위:m

LIDAR_NUM=30 #약 6도 간격으로 lidar 거리 계산

WORLD_DT=0.05 #0.1
ACTION_DT=0.01

E_C_PENALTY_WEIGHT=5
E_PHI_PENALTY_WEIGHT=1
OFF_COURSE_PENALTY_WEIGHT=2

TARGET_VEL_REWARD_WEIGHT=3
TARGET_VEL=11
TARGET_VEL_SIGMA=1

TERMINATE_PENALTY=500

PENALTY_CONDITION=off_course_instance

python train_main.py \
    agent.exp_name=0513_Nam_Act3_Center_targetVel \
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
    environment.vehicle.brake_on_pos_vel=true \
    environment.vehicle.allow_both_feet=true \
    environment.vehicle.allow_neg_torque=true \
    environment.vehicle.use_continuous_bps=$USE_CONTINUOUS_BPS \
    environment.action.action_dim=3 \
    environment.do_debug_logs=0 \
    environment.track.use_nam_only=$USE_NAM_ONLY \
    environment.track.nam_ratio=0.1 \
    environment.vehicle.aps_bps_weight=1. \
    penalty.terminate.penalty_value=$TERMINATE_PENALTY \
    penalty.time_penalty.usage=$USE_TIME_PENALTY \
    penalty.terminate.off_course_condition=off_course_tire \
    penalty.off_course_penalty.usage=true \
    penalty.off_course_penalty.ratio_usage=true \
    penalty.off_course_penalty.penalty_value=$OFF_COURSE_PENALTY_WEIGHT \
    penalty.off_course_penalty.condition=$PENALTY_CONDITION \
    penalty.E_phi_penalty.usage=true \
    penalty.E_phi_penalty.penalty_value=$E_PHI_PENALTY_WEIGHT \
    penalty.curvature_vel_penalty.usage=false \
    penalty.E_c_penalty.usage=true \
    penalty.E_c_penalty.penalty_value=$E_C_PENALTY_WEIGHT \
    penalty.E_c_penalty.normalize_E_c=true \
    reward.progress_reward_curve.usage=false \
    reward.track_align_velocity_reward.usage=false \
    reward.target_velocity_reward.usage=true \
    reward.target_velocity_reward.reward_weight=$TARGET_VEL_REWARD_WEIGHT \
    reward.target_velocity_reward.target_vel=$TARGET_VEL \
    reward.target_velocity_reward.gauss_sigma=$TARGET_VEL_SIGMA
