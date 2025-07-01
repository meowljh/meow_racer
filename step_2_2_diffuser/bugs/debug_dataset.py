import os, sys
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(root);sys.path.append(os.path.dirname(root))
from diffuser.utils import (
    Parser, Config
)

DATASET_NAME = "hopper-medium-expert-v2"

args = Parser().parse_args(experiment="diffusion")
