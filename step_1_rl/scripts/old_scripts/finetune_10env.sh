#!/bin/bash
FNAME="FINETUNE_JWbaseline_0214_3_5_15_10ENV"
FINETUNE_PATH="BOTH_SDE_JWbaseline_0209_sac_3_5_15"
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
    --both_track_ratio 0.1 \
    --skip_rate 1 \
    --max_episodes 10000 \
    --max_reward_tile 5000 \
    --min_movement 0.1 \
    --time_penalty 0.5 \
    --n_envs 10 \
    --learning_rate 1e-4 \
    --reward_type "baseline" \
    --screen_title $FNAME \
    --save $FNAME \
    --terminate_penalty 200 \
    --use_gas \
    --use_steer \
    --use_brake \
    --use_delta \
    --use_force
