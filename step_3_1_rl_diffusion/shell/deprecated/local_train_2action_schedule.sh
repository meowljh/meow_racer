#!/bin/bash

python train_main.py \
    agent.exp_name=local_debug_v2_Act2_Sched \
    environment.action.action_dim=2 \
    agent.trainer.alpha_val=1. \
    environment.track.use_nam_only=false \
    environment.vehicle.zero_force_neg_vel=true \
    environment.vehicle.always_pos_vel=0.001 \
    environment.track.nam_ratio=0.1 \
    environment.vehicle.aps_bps_weight=1. \
    environment.vehicle.schedule_brake_ratio=true \
    environment.vehicle.schedule_brake_ratio_scale=0.01 \
    environment.vehicle.schedule_brake_episode=200 \
    agent.algorithm.warmup_actor_step=10000 \
    penalty.reverse_penalty.usage=true \
    penalty.reverse_penalty.penalty_value=10. \
    penalty.reverse_penalty.add_vel=false \
    penalty.reverse_penalty.use_cummulative=false \
    reward.progress_reward_curve.reward_weight=10. \
    reward.velocity_reward.usage=true \
    reward.velocity_reward.reward_weight=0.1 \
    penalty.tire_slip_penalty.usage=false \
    agent.algorithm.batch_size=256 \
    agent.algorithm.num_epochs=100000 \
    agent.algorithm.min_num_steps_before_training=1000 \
    agent.algorithm.num_expl_steps_per_train_loop=5000 \
    agent.algorithm.num_trains_per_train_loop=5000 \
    agent.replay_buffer_size=1000000 \
    agent.algorithm.max_path_length=1000000 \
    agent.algorithm.num_train_loops_per_epoch=1 \
    agent.test.test_log_path="D:/meow_racer_experiments"