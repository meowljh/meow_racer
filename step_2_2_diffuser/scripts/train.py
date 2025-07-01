import os, sys
import subprocess

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
# os.environ["PATH"].append(r"C:\Users\7459985\.mujoco\mujoco210\bin\bin".replace("\\", "/"))
MUJOCO_PATH = r"C:\Users\7459985\.mujoco\mujoco210\bin\bin".replace("\\", "/")
sys.path.append(MUJOCO_PATH)
MUJOCO_PATH_CMD = r'set PATH=C:\Users\7459985\.mujoco\mujoco210\bin\bin;%PATH%'
os.system(MUJOCO_PATH_CMD)
# subprocess.run([
#     'set', r'PATH=C:\Users\7459985\.mujoco\mujoco210\bin\bin;%PATH' #  .replace('\\', '/')
# ])
sys.path.append(root);sys.path.append(os.path.dirname(root))
from diffuser.utils import (
    Parser, Config, Trainer, report_parameters, batchify
)
#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#
'''[Trouble Shoot] 
- gym Environement unregistered error
-> fixed by uninstalling gymnasium and using the deprecated gym
[ref] https://hyunsooworld.tistory.com/entry/RL-Colab-vscode%EC%9C%BC%EB%A1%9C-mujoko-py-%EB%B0%8F-gym-%ED%99%98%EA%B2%BD-%EA%B5%AC%EC%B6%95%ED%95%98%EA%B8%B0'''
class Parser(Parser):
    dataset: str = 'hopper-medium-expert-v2'
    config: str = 'config.locomotion'

args = Parser().parse_args('diffusion')


#-----------------------------------------------------------------------------#
#---------------------------------- dataset ----------------------------------#
#-----------------------------------------------------------------------------#

dataset_config = Config(
    args.loader,
    savepath=(args.savepath, 'dataset_config.pkl'),
    env=args.dataset,
    horizon=args.horizon,
    normalizer=args.normalizer,
    preprocess_fns=args.preprocess_fns,
    use_padding=args.use_padding,
    max_path_length=args.max_path_length,
)

render_config = Config(
    args.renderer,
    savepath=(args.savepath, 'render_config.pkl'),
    env=args.dataset,
)

dataset = dataset_config()
renderer = render_config()

observation_dim = dataset.observation_dim
action_dim = dataset.action_dim


#-----------------------------------------------------------------------------#
#------------------------------ model & trainer ------------------------------#
#-----------------------------------------------------------------------------#

model_config = Config(
    args.model,
    savepath=(args.savepath, 'model_config.pkl'),
    horizon=args.horizon,
    transition_dim=observation_dim + action_dim,
    cond_dim=observation_dim,
    dim_mults=args.dim_mults,
    attention=args.attention,
    device=args.device,
)


diffusion_config = Config(
    args.diffusion,
    savepath=(args.savepath, 'diffusion_config.pkl'),
    horizon=args.horizon,
    observation_dim=observation_dim,
    action_dim=action_dim,
    n_timesteps=args.n_diffusion_steps,
    loss_type=args.loss_type,
    clip_denoised=args.clip_denoised,
    predict_epsilon=args.predict_epsilon,
    ## loss weighting
    action_weight=args.action_weight,
    loss_weights=args.loss_weights,
    loss_discount=args.loss_discount,
    device=args.device,
)

trainer_config = Config(
    Trainer, 
    savepath=(args.savepath, 'trainer_config.pkl'),
    train_batch_size=args.batch_size,
    train_lr=args.learning_rate,
    gradient_accumulate_every=args.gradient_accumulate_every,
    ema_decay=args.ema_decay,
    sample_freq=args.sample_freq,
    save_freq=args.save_freq,
    label_freq=int(args.n_train_steps // args.n_saves),
    save_parallel=args.save_parallel,
    results_folder=args.savepath,
    bucket=args.bucket,
    n_reference=args.n_reference,
)

#-----------------------------------------------------------------------------#
#-------------------------------- instantiate --------------------------------#
#-----------------------------------------------------------------------------#

model = model_config()

diffusion = diffusion_config(model)

trainer = trainer_config(diffusion, dataset, renderer)


#-----------------------------------------------------------------------------#
#------------------------ test forward & backward pass -----------------------#
#-----------------------------------------------------------------------------#

report_parameters(model)

print('Testing forward...', end=' ', flush=True)
batch = batchify(dataset[0])
loss, _ = diffusion.loss(*batch)
loss.backward()
print('✓')


#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#
# 명심해야하는 것은, 지금 학습 시키는게 RL이 아니라 이미 시뮬레이션 데이터가 있고, 이 데이터를 바탕으로 diffuser 학습시키는 것임 #
n_epochs = int(args.n_train_steps // args.n_steps_per_epoch)

for i in range(n_epochs):
    print(f'Epoch {i} / {n_epochs} | {args.savepath}')
    trainer.train(n_train_steps=args.n_steps_per_epoch) ## diffuser training ##

