#!/bin/bash

python ppo_train.py \
    --seed 42 \
    --env_name hopper \
    --actor_hidden_dims 128 128 \
    --critic_hidden_dims 128 128 \
    --lr_decay \
    --save "hopper_1"