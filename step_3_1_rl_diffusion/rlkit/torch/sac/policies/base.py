import abc
import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.policies.base import ExplorationPolicy
from rlkit.torch.core import torch_ify, elem_or_tuple_to_numpy
from rlkit.torch.distributions import (
    Delta, TanhNormal, MultivariateDiagonalNormal, GaussianMixture, GaussianMixtureFull,
)
from rlkit.torch.networks import Mlp, CNN
from rlkit.torch.networks.basic import MultiInputSequential
from rlkit.torch.networks.stochastic.distribution_generator import (
    DistributionGenerator
)


class TorchStochasticPolicy(
    DistributionGenerator,
    ExplorationPolicy, metaclass=abc.ABCMeta
):
    def get_action_style(self, obs_np, Z_style):
        if isinstance(obs_np, tuple):
            obs_np = obs_np[0] 
        obs_np = obs_np[None]
        Z_style_np = np.array([Z_style])
        # torch_args = tuple(torch.cat([torch_ify(x), torch_ify(z)], dim=1) for x,z in zip(obs_np, Z_style))
        torch_args = tuple(torch_ify(x) for x in obs_np)

        dist = self(*torch_args, None)
        actions = dist.sample()
        actions = elem_or_tuple_to_numpy(actions)
        return actions[0, :], {}
    
    def get_action(self, obs_np, ):
        if isinstance(obs_np, tuple):
            obs_np = obs_np[0]
        actions = self.get_actions(obs_np[None])
        return actions[0, :], {}

    def get_actions(self, obs_np, ):
        dist = self._get_dist_from_np(obs_np)
        actions = dist.sample()
        return elem_or_tuple_to_numpy(actions)

    def _get_dist_from_np(self, *args, **kwargs):
        torch_args = tuple(torch_ify(x) for x in args)
        torch_kwargs = {k: torch_ify(v) for k, v in kwargs.items()}
        dist = self(*torch_args, **torch_kwargs)
        return dist



class PolicyFromDistributionGenerator(
    MultiInputSequential,
    TorchStochasticPolicy,
):
    """
    Usage:
    ```
    distribution_generator = FancyGenerativeModel()
    policy = PolicyFromBatchDistributionModule(distribution_generator)
    ```
    """
    pass


class MakeDeterministic(TorchStochasticPolicy):
    def __init__(
            self,
            action_distribution_generator: DistributionGenerator,
    ):
        super().__init__()
        self._action_distribution_generator = action_distribution_generator

    def forward(self, *args, **kwargs):
        dist = self._action_distribution_generator.forward(*args, **kwargs)
        return Delta(dist.mle_estimate())
