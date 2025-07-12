import os, sys
from natsort import natsort
from glob import glob
import torch

ROOT=os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT)

from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.sac.policies.custom_gaussian_policy import TanhGaussianPolicy_MultiHead_Mlp
from rlkit.torch.sac.policies.style_cond_policy import TanhGaussianPolicy_StyleCond
from rlkit.torch.networks.mlp import ParallelMlp, ConcatMlp

def _load_state_dict_trained(trainer, exp_root:str, epoch_mode:str):
    # breakpoint()
    if epoch_mode == 'best':
        saved_folders = natsort.natsorted(glob(exp_root + "/test/success_*"))
        last_epoch = int(saved_folders[-1].split('_')[-1])
    elif epoch_mode == 'recent':
        saved_folders = natsort.natsorted(glob(exp_root + "/test"))
        last_epoch = int(saved_folders[-1].split('_')[-1])
    else:
        raise UserWarning

    recent_folder = f"{exp_root}/ckpt/epoch_{last_epoch}"
    
    ##[step1] network load##
    networks_dict = trainer.networks_dict
    for key, net in networks_dict.items():
        net.load_state_dict(torch.load(f"{recent_folder}/{key}.pth"))
        net.to('cuda')
        setattr(trainer, key, net)
    print(f"Successfully loaded NETWORKS from {exp_root}")
    
    ##[step2] optimizer load##
    optimizers_dict = trainer.optimizers_dict
    for key, optim in optimizers_dict.items():
        optim.load_state_dict(torch.load(f"{recent_folder}/{key}.pth"))
        # breakpoint()
        attr_name = key.replace('optim', 'optimizer')
        setattr(trainer, attr_name, optim)
    print(f"Successfully loaded OPTIMIZERS from {exp_root}")
    
    ##[step3] set epoch##
    setattr(trainer, '_start_epoch', last_epoch)
    print(f"Successfully set STARTING EPOCH to {exp_root}")
    
    return trainer, last_epoch

    
        
     
        
def load_policy(dict_cfg, obs_dim, action_dim, style_dim):
    name = dict_cfg['policy']['name']
    if name == 'TanhGaussianPolicy':
        policy = TanhGaussianPolicy(
            obs_dim = obs_dim,
            action_dim = action_dim,
            hidden_sizes = dict_cfg['agent']['policy']['hidden_sizes'],
            layer_norm=dict_cfg['agent']['policy']['layer_norm'],
            std = dict_cfg['agent']['policy']['std']
        )
    elif name == 'TanhGaussianPolicy_MultiHead_Mlp':
        
        policy = TanhGaussianPolicy_MultiHead_Mlp(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=dict_cfg['agent']['policy']['hidden_sizes'],
            head_conf_dict=dict_cfg['policy'][f"head_{action_dim}"],
            layer_norm=dict_cfg['agent']['policy']['layer_norm'],
            std = dict_cfg['agent']['policy']['std']
        )

    elif name == 'TanhGaussianPolicy_StyleCond':
        
        policy=TanhGaussianPolicy_StyleCond(
            hidden_sizes=dict_cfg['agent']['policy']['hidden_sizes'],
            obs_dim=obs_dim,
            action_dim=action_dim,
            style_dim=style_dim,
            layer_norm=dict_cfg['agent']['policy']['layer_norm'],
            std=dict_cfg['agent']['policy']['std']
        )
        
    return policy

def load_critic(dict_cfg, obs_dim, action_dim, style_dim):
    """어차피 ConcatMlp 객체에서 input들을 받으면 그대로 지정한 dimension에 따라 concat하기 때문에 concat한 결과 dimension size가 input_size와 같기만 하면 됨."""
    if style_dim > 0:
        qf1 = ParallelMlp(
            num_heads = style_dim,
            input_size=obs_dim+action_dim+style_dim,
            output_size_per_mlp = 1,
            hidden_sizes= dict_cfg['agent']['qf1']['hidden_sizes']
        )
        qf2 = ParallelMlp(
            num_heads = style_dim,
            input_size=obs_dim+action_dim+style_dim,
            output_size_per_mlp = 1,
            hidden_sizes= dict_cfg['agent']['qf2']['hidden_sizes']
        )
        target_qf1 = ParallelMlp(
            num_heads = style_dim,
            input_size=obs_dim+action_dim+style_dim,
            output_size_per_mlp = 1,
            hidden_sizes= dict_cfg['agent']['target_qf1']['hidden_sizes']
        )
        target_qf2 = ParallelMlp(
            num_heads = style_dim,
            input_size=obs_dim+action_dim+style_dim,
            output_size_per_mlp = 1,
            hidden_sizes= dict_cfg['agent']['target_qf2']['hidden_sizes']
        )
    else:
        qf1 = ConcatMlp(
            input_size=obs_dim + action_dim + style_dim,
            output_size=1,
            hidden_sizes = dict_cfg['agent']['qf1']['hidden_sizes']
        )
        qf2 = ConcatMlp(
            input_size=obs_dim + action_dim + style_dim,
            output_size = 1,
            hidden_sizes = dict_cfg['agent']['qf2']['hidden_sizes']
        )
        target_qf1 = ConcatMlp(
            input_size=obs_dim + action_dim + style_dim,
            output_size = 1,
            hidden_sizes = dict_cfg['agent']['target_qf1']['hidden_sizes']
        )
        target_qf2 = ConcatMlp(
            input_size=obs_dim + action_dim + style_dim,
            output_size = 1,
            hidden_sizes = dict_cfg['agent']['target_qf2']['hidden_sizes']
        )
    
    return qf1, qf2, target_qf1, target_qf2