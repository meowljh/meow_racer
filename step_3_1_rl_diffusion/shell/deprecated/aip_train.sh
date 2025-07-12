#!/bin/bash

python train_main.py \
    agent.exp_name=aip_debug_v1 \
    agent.algorithm.batch_size=512 \
    agent.algorithm.num_epochs=10000 \
    agent.algorithm.num_expl_steps_per_train_loop=5000 \
    agent.replay_buffer_size=1000000 \
    agent.test.test_log_path="/home/logs" \
    environment.observation.forward_vector.num_vecs 20 \
    environment.observation.forward_vector.max_val 100