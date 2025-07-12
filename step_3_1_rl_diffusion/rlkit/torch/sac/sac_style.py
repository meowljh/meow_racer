from collections import OrderedDict, namedtuple
import torch
import numpy as np
from .sac import SACTrainer

import rlkit.torch.pytorch_util as ptu
from rlkit.core.logging import add_prefix
from rlkit.core.eval_util import create_stats_ordered_dict
import gtimer as gt

SACLosses = namedtuple(
    'SACLosses', 
    'policy_loss qf1_loss qf2_loss alpha_loss'
)

class Style_SACTrainer(SACTrainer):
    def compute_loss(self, batch, skip_statistics=False):
        style_size = self.env.unwrapped.style_size #2일 것임.
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        
        dist = self.policy(obs)
        mean = dist.mean
        std = dist.stddev
        
        ### [STEP1] Compute Loss For Policy ###
        new_obs_actions, log_pi = dist.rsample_and_logprob()
        log_pi = log_pi.unsqueeze(-1)
        
        if self.use_automatic_entropy_tuning:
            #근데 이렇게 entropy threshold를 지정해서 그 값보다 작은 entropy가 나오지 않도록 tuning 하게 되었을때의 문제점 해결도 함
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            alpha = self.log_alpha.exp() #loss term에 Entropy Maximization을 위한 log값 weighting의 가중치 자동 조절
        else:
            alpha_loss = 0
            alpha = 1
        
        q_new_actions = torch.min(
            self.qf1(obs, new_obs_actions),
            self.qf2(obs, new_obs_actions)
        )
        w = obs[:,:style_size]
        Q_pi =(q_new_actions * w).sum(-1)

        # Entropy-regularized policy loss
        policy_loss = (alpha * log_pi - Q_pi).mean()

        ### [STEP2] Compute Loss for Critic ###
        w = obs[:,:style_size]
        
        q1_pred = self.qf1(obs, actions) 
        q2_pred = self.qf2(obs, actions) 
        new_dist = self.policy(next_obs)
        new_next_actions, new_log_pi = new_dist.rsample_and_logprob()
        new_log_pi = new_log_pi.unsqueeze(-1)
        
        
        target_q_values = torch.min(
            self.target_qf1(next_obs, new_next_actions),
            self.target_qf2(next_obs, new_next_actions)
        )
        q_target = rewards + self.discount * (1 - terminals) * (target_q_values - alpha * new_log_pi)

        qf1_loss = self.qf1_criterion(q1_pred * w, (q_target * w).detach())
        qf2_loss = self.qf2_criterion(q2_pred * w, (q_target * w).detach())

        """
        Save some statistics for eval
        """
        eval_statistics = OrderedDict()
        if not skip_statistics:
            eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
            eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
            eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(q1_pred),
            ))
            eval_statistics.update(create_stats_ordered_dict(
                'Q2 Predictions',
                ptu.get_numpy(q2_pred),
            ))
            eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(q_target),
            ))
            eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            policy_statistics = add_prefix(dist.get_diagnostics(), "policy/")
            eval_statistics.update(policy_statistics)
            if self.use_automatic_entropy_tuning:
                eval_statistics['Alpha'] = alpha.item()
                eval_statistics['Alpha Loss'] = alpha_loss.item()

        loss = SACLosses(
            policy_loss=policy_loss,
            qf1_loss=qf1_loss,
            qf2_loss=qf2_loss,
            alpha_loss=alpha_loss,
        )
        
        self._save_vals(
            mean=mean,
            std=std,
            loss=loss,
            eval_statistics=eval_statistics
        )

        return loss, eval_statistics