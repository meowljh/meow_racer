from typing import Callable, NamedTuple, Sequence, Tuple, Union
from dataclasses import dataclass

import math

import torch
import torch.nn as nn

from meow.torch_network.blocks import DACERPolicyNet, QNet
from meow.torch_utils.diffusion import GaussianDiffusion
