FNAME="AM_NS_3A_JW_0405"

RL_ALGORITHM="sac"
REWARD_TYPE="am"

ENV_TYPE="random" #"nam"

SAC_ACTION_NOISE="linear_gaussian"
ACTION_MEAN=0
ACTION_SIGMA=0.1
ACTION_FINAL_SIGMA=0.
ACTION_MAX_STEPS=5e+7
SAC_ENT_COEF="auto_0.1"
N_ENVS=0

#### AM reward와 baseline reward 모두 reverse track, random start는 하지 않음 (최대한 예외 상황이 없도록 하기 위해서)
DO_REVERSE=0.
DO_RANDOM_START=0

RANDOM_SEED=42

MAX_EPISODES=10000
TOTAL_TIMESTEPS=10000 # maximum number of timesteps for each episode
NOISE_CHANGE_INTERVAL="episode_3000"

ACTION_NUM=3 # if 2, steering / gas&brake together
USE_ROTATED_FORWARD=1 # rotate the forward vector to match the car state world
USE_CURVATURE=0 #1
USE_THETA_REWARD=1
USE_TILE_REWARD=0

BACKWARD_PENALTY=100

python tutorial_train.py \
    --rl_algorithm $RL_ALGORITHM \
    --use_steer \
    --with_tile_reward $USE_TILE_REWARD \
    --am_with_theta_reward $USE_THETA_REWARD \
    --vel_weight 0. \
    --backward_penalty $BACKWARD_PENALTY \
    --action_num $ACTION_NUM \
    --action_noise_type $SAC_ACTION_NOISE \
    --action_noise_sigma $ACTION_SIGMA \
    --final_sigma $ACTION_FINAL_SIGMA \
    --use_rotated_forward $USE_ROTATED_FORWARD \
    --max_gaussian_step $ACTION_MAX_STEPS \
    --use_curvature $USE_CURVATURE \
    --is_aip 0 \
    --random_seed $RANDOM_SEED \
    --env_type $ENV_TYPE \
    --random_checkpoints \
    --ent_coef $SAC_ENT_COEF \
    --oscillation_penalty 0. \
    --num_random_checkpoints 4 \
    --use_jw 1 \
    --use_sde 0 \
    --do_reverse $DO_REVERSE \
    --random_start $DO_RANDOM_START \
    --n_steps_td 0 \
    --use_theta_diff 0 \
    --weight_save_converge 1 \
    --weight_save_interval 5 \
    --num_vecs 15 \
    --theta_diff 5 \
    --lidar_deg 3 \
    --both_track_ratio 0.1 \
    --skip_rate 1 \
    --max_episodes $MAX_EPISODES \
    --total_timesteps $TOTAL_TIMESTEPS \
    --max_reward_tile 5000 \
    --min_movement 0.1 \
    --time_penalty 0.5 \
    --n_envs $N_ENVS \
    --learning_rate 3e-4 \
    --reward_type $REWARD_TYPE \
    --screen_title $FNAME \
    --save $FNAME \
    --terminate_penalty 200 