#!/bin/bash

# python train_main.py --config-name config_mhead \
    # agent.exp_name=SAC_GaussPolicy_0430_Both_Act3_MHead_SF \
# python train_main.py \
#     agent.exp_name=SAC_GaussPolicy_0501_Both_Act3_Race \
python train_main.py \
    agent.exp_name=SAC_GaussPolicy_0508_Both_Act3_Race \
    agent.policy.layer_norm=false \
    agent.algorithm.warmup_actor_step=-1.\
    agent.algorithm.batch_size=256 \
    agent.algorithm.num_epochs=100000 \
    agent.algorithm.min_num_steps_before_training=1000 \
    agent.algorithm.num_expl_steps_per_train_loop=5000 \
    agent.algorithm.num_trains_per_train_loop=5000 \
    agent.algorithm.max_path_length=1000000 \
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
    environment.observation.lookahead.lookahead_time=1.6 \
    environment.observation.lookahead.num_states=20 \
    environment.observation.lookahead.coords=false \
    environment.observation.lookahead.curvature=true \
    environment.observation.lookahead.scale_method=standard \
    environment.observation.forward_vector.usage=true \
    environment.observation.forward_vector.rotate_vec=false \
    environment.observation.lidar_sensor.usage=true \
    environment.observation.lidar_sensor.scale_method=minmax \
    environment.observation.lidar_sensor.num_lidar=60 \
    environment.track.track_density=1 \
    environment.vehicle.brake_on_pos_vel=true \
    environment.vehicle.allow_both_feet=true \
    environment.action.action_dim=3 \
    environment.do_debug_logs=0 \
    environment.track.use_nam_only=false \
    environment.track.nam_ratio=0.1 \
    environment.vehicle.aps_bps_weight=1. \
    penalty.time_penalty.usage=true \
    penalty.terminate.off_course_condition=off_course_tire \
    penalty.terminate.penalty_value=200 \
    penalty.off_course_penalty.usage=true \
    penalty.off_course_penalty.ratio_usage=true \
    penalty.off_course_penalty.condition=off_course_com \
    penalty.off_course_penalty.penalty_value=1 \
    penalty.E_phi_penalty.usage=true \
    penalty.E_phi_penalty.penalty_value=1 \
    penalty.E_c_penalty.usage=false \
    reward.progress_reward_curve.usage=true \
    reward.progress_reward_curve.reward_weight=10