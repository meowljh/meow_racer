#!/bin/bash
FNAME="FINETUNE_JW_0225_NAM"
FINETUNE_PATH="FINETUNE_JW_0218_NAM" 
# "BOTH_SDE_JWbaseline_0209_sac_3_5_15"
#!/bin/bash
python tutorial_train.py \
    --env_type random \
    --finetune_path $FINETUNE_PATH \
    --use_jw 1 \
    --use_sde 0 \
    --do_reverse 0 \
    --use_theta_diff 0 \
    --weight_save_converge 1 \
    --weight_save_interval 5 \
    --use_jw 1 \
    --num_vecs 15 \
    --theta_diff 5 \
    --lidar_deg 3 \
    --num_random_checkpoints 4 \
    --both_track_ratio 0.7 \
    --max_both_track_ratio 1. \
    --skip_rate 1 \
    --max_episodes 1000 \
    --max_reward_tile 5000 \
    --min_movement 0.1 \
    --time_penalty 0.5 \
    --n_envs 0 \
    --learning_rate 1e-5 \
    --reward_type "baseline" \
    --screen_title $FNAME \
    --save $FNAME \
    --terminate_penalty 200 \
    --use_gas \
    --use_steer \
    --use_brake \
    --use_delta \
    --use_force
