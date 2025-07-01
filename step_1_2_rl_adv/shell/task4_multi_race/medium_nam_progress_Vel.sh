#!/bin/bash

USE_NAM_ONLY=true
NAM_TRACK_RATIO=0.

#style
STYLE_TYPE=medium
STYLE_COMMON_WEIGHT=1 #0.5 #1. #0.5 #1.
STYLE_DEFENSIVE_WEIGHT=0.5 #1. #0.5 #0.25 #0.5 #1. #0.8 #0.5
STYLE_AGGRESSIVE_WEIGHT=1.5 #0.5 #1. #0.5 #0.25 #0.5  #0.5 #0.2 #0.5

#agent/policy 
WORLD_DT=0.05
ACTION_DT=0.001
USE_CONTINUOUS_BPS=true
ACTION_DIM=3
ALLOW_BOTH_FEET=false
ALLOW_NEG_TORQUE=true
ALWAYS_POS_VEL=0
USE_APS_BPS_DIFF=true
CAR_INIT_VX=1
LOOKAHEAD_NUM=20
LOOKAHEAD_THETA=20
LIDAR_NUM=30

#common reward
USE_PROGRESS_REWARD=true
PROGRESS_REWARD_VALUE=1000

USE_KAPPA_VEL_REWARD=true
KAPPA_VEL_CONTINUOUS=true
KAPPA_VEL_TARGET_THETA=20
KAPPA_VEL_FUTURE_MODE=mean
KAPPA_VEL_REWARD_VALUE=1 #10 #5 #1

USE_E_PHI_REWARD=true
E_PHI_REWARD_VALUE=0.5 #1. #0.1

USE_VELOCITY_REWARD=true
VELOCITY_REWARD_VALUE=1
NORM_VELOCITY_REWARD=true

USE_TIME_PENALTY=true
TIME_PENALTY_VALUE=0.1

USE_MIN_VEL_PENALTY=true
MIN_VEL_PEN_VALUE=3
MIN_VEL=1e-3

USE_NEG_VEL_BPS_PENALTY=true
NEG_VEL_BPS_PENALTY_WEIGHT=1

USE_OFF_COURSE_PENALTY=true
OFF_COURSE_PENALTY_WEIGHT=2
OFF_COURSE_PENALTY_RATIO=true
OFF_COURSE_PENALTY_CONDITION=off_course_count

#defensive reward
USE_E_C_PENALTY=true
E_C_PENALTY_WEIGHT=1. #5. #3. #10. #5. #10. #3. #10. #3 #10
E_C_PENALTY_NORM=true
E_C_PENALTY_AS_REWARD=false #true

USE_SMOOTH_CONTROL_PENALTY=true
APS_SMOOTH_PEN_VALUE=0.1
BPS_SMOOTH_PEN_VALUE=0.1
STEER_SMOOTH_PEN_VALUE=0.1

USE_KAPPA_VEL_PENALTY=false #true
KAPPA_VEL_PEN_VALUE=1. #5.
KAPPA_VEL_PEN_NORM_VEL=true
KAPPA_VEL_PEN_NORM_KAPPA=true

#aggressive reward
USE_NEG_VEL_APS_REWARD=true
NEG_VEL_APS_REWARD_WEIGHT=1

USE_CURV_WEIGHTED_VEL_REWARD=true
CURV_WEIGHTED_KAPPA_WEIGHT=5. #10. #5. #1. #5.
CURV_WEIGHTED_REWARD_WEIGHT=3. #1. #3. #1.5 #3.
CURV_WEIGHTED_NORM=true

USE_STRAIGHT_LINE_VEL_REWARD=false #true
STRAIGHT_LINE_VEL_REWARD_VALUE=1 #0.5
NORM_STRAIGHT_LINE_VEL=true
STRAIGHT_LINE_KAPPA_THRESH=0.01

USE_HARD_CORNER_VEL_REWARD=false #true
HARD_CORNER_VEL_REWARD_WEIGHT=2. #0.5 #1
HARD_CORNER_KAPPA_NORM=true
HARD_CORNER_KAPPA_THRESH=0.01 #0.005
HARD_CORNER_VEL_NORM=true




##medium reward shaping으로는 0.5 * aggressive reward + 0.5 * defensive reward + common reward이기 때문에
##아래의 aggressive / defensive reward의 weight들은 모두 기준보다 절반으로 우선 설정하였음.
##나중에 style conditioning 구현을 해서 학습을 시킬 때에 style shaping을 위한 객체가 필요



#terminate
PENALTY_CONDITION=off_course_count
TERMINATE_VEL_PATIENCE=0.1 #0.02
TERMINATE_PENALTY=500
TERMINATE_VEL_COND=count_time_neg_vel

#experiment
EXP_NAME=0612_SingleFeet_DTx50_NAM_MEDIUM_Act3_Center_timePen_minVPen_offCnt_VelReward_ProgressReward1000_Weighted_1_SameWeight_LargeAggressive

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
    environment.track.nam_ratio=$NAM_TRACK_RATIO \
    environment.vehicle.aps_bps_weight=1. \
    style.style_mode.type=$STYLE_TYPE \
    style.style_mode.common_weight=$STYLE_COMMON_WEIGHT \
    style.style_mode.aggressive_weight=$STYLE_AGGRESSIVE_WEIGHT \
    style.style_mode.defensive_weight=$STYLE_DEFENSIVE_WEIGHT \
    penalty.terminate.penalty_value=$TERMINATE_PENALTY \
    penalty.terminate.off_course_condition=off_course_tire \
    penalty.terminate.neg_vel_patience=$TERMINATE_VEL_PATIENCE \
    style.common.time_penalty.usage=$USE_TIME_PENALTY \
    style.common.time_penalty.penalty_value=$TIME_PENALTY_VALUE \
    style.common.progress_reward.usage=$USE_PROGRESS_REWARD \
    style.common.progress_reward.reward_value=$PROGRESS_REWARD_VALUE \
    style.common.curvature_vel_reward.usage=$USE_KAPPA_VEL_REWARD \
    style.common.curvature_vel_reward.reward_value=$KAPPA_VEL_REWARD_VALUE \
    style.common.curvature_vel_reward.target_theta=$KAPPA_VEL_TARGET_THETA \
    style.common.curvature_vel_reward.future_mode=$KAPPA_VEL_FUTURE_MODE \
    style.common.curvature_vel_reward.continuous=$KAPPA_VEL_CONTINUOUS \
    style.common.curvature_vel_reward.vehicle_speed_vx=$KAPPA_VEL_USE_VX \
    style.common.E_phi_reward.usage=$USE_E_PHI_REWARD \
    style.common.E_phi_reward.reward_value=$E_PHI_REWARD_VALUE \
    style.common.track_align_velocity_reward.usage=$USE_VELOCITY_REWARD \
    style.common.track_align_velocity_reward.reward_weight=$VELOCITY_REWARD_VALUE \
    style.common.track_align_velocity_reward.normalize_vel=$NORM_VELOCITY_REWARD \
    style.common.time_penalty.usage=$USE_TIME_PENALTY \
    style.common.time_penalty.penalty_value=$TIME_PENALTY_VALUE \
    style.common.off_course_penalty.usage=$USE_OFF_COURSE_PENALTY \
    style.common.off_course_penalty.ratio_usage=$OFF_COURSE_PENALTY_RATIO \
    style.common.off_course_penalty.penalty_value=$OFF_COURSE_PENALTY_WEIGHT \
    style.common.off_course_penalty.condition=$OFF_COURSE_PENALTY_CONDITION \
    style.common.neg_velocity_bps_penalty.usage=$USE_NEG_VEL_BPS_PENALTY \
    style.common.neg_velocity_bps_penalty.penalty_value=$NEG_VEL_BPS_PENALTY_WEIGHT \
    style.common.min_velocity_penalty.usage=$USE_MIN_VEL_PENALTY \
    style.common.min_velocity_penalty.penalty_value=$MIN_VEL_PEN_VALUE \
    style.common.min_velocity_penalty.min_velocity=$MIN_VEL \
    style.defensive.E_c_penalty.usage=$USE_E_C_PENALTY \
    style.defensive.E_c_penalty.penalty_value=$E_C_PENALTY_WEIGHT \
    style.defensive.E_c_penalty.normalize_E_c=$E_C_PENALTY_NORM \
    style.defensive.E_c_penalty.as_reward=$E_C_PENALTY_AS_REWARD \
    style.defensive.smooth_control_penalty.usage=$USE_SMOOTH_CONTROL_PENALTY \
    style.defensive.smooth_control_penalty.aps_penalty_value=$APS_SMOOTH_PEN_VALUE \
    style.defensive.smooth_control_penalty.bps_penalty_value=$BPS_SMOOTH_PEN_VALUE \
    style.defensive.smooth_control_penalty.steer_penalty_value=$STEER_SMOOTH_PEN_VALUE \
    style.defensive.curvature_velocity_penalty.usage=$USE_KAPPA_VEL_PENALTY \
    style.defensive.curvature_velocity_penalty.penalty_weight=$KAPPA_VEL_PEN_VALUE \
    style.defensive.curvature_velocity_penalty.norm_vel=$KAPPA_VEL_PEN_NORM_VEL \
    style.defensive.curvature_velocity_penalty.norm_kappa=$KAPPA_VEL_PEN_NORM_KAPPA \
    style.aggressive.neg_velocity_aps_reward.usage=$USE_NEG_VEL_APS_REWARD \
    style.aggressive.neg_velocity_aps_reward.reward_weight=$NEG_VEL_APS_REWARD_WEIGHT \
    style.aggressive.curvature_weighted_vel_reward.usage=$USE_CURV_WEIGHTED_VEL_REWARD \
    style.aggressive.curvature_weighted_vel_reward.kappa_weight_value=$CURV_WEIGHTED_KAPPA_WEIGHT \
    style.aggressive.curvature_weighted_vel_reward.reward_weight=$CURV_WEIGHTED_REWARD_WEIGHT \
    style.aggressive.curvature_weighted_vel_reward.normalize_vel=$CURV_WEIGHTED_NORM \
    style.aggressive.straight_line_vel_reward.usage=$USE_STRAIGHT_LINE_VEL_REWARD \
    style.aggressive.straight_line_vel_reward.reward_weight=$STRAIGHT_LINE_VEL_REWARD_VALUE \
    style.aggressive.straight_line_vel_reward.normalize_vel=$NORM_STRAIGHT_LINE_VEL \
    style.aggressive.straight_line_vel_reward.corner_kappa_thresh=$STRAIGHT_LINE_KAPPA_THRESH \
    style.aggressive.hard_corner_curvature_weighted_vel_reward.usage=$USE_HARD_CORNER_VEL_REWARD \
    style.aggressive.hard_corner_curvature_weighted_vel_reward.reward_weight=$HARD_CORNER_VEL_REWARD_WEIGHT \
    style.aggressive.hard_corner_curvature_weighted_vel_reward.normalize_kappa=$HARD_CORNER_KAPPA_NORM \
    style.aggressive.hard_corner_curvature_weighted_vel_reward.corner_kappa_thresh=$HARD_CORNER_KAPPA_THRESH 