import os
import pprint
import sys
import time

import ray
from ray.tune.registry import register_env

from custom_env import CustomEnv
from tf_models import MaskedActionsCNN
from train import train

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)

ray.init(log_to_driver=False)


def env_creator(env_config):
    return CustomEnv(
        env_dir="examples/normal_simulation2/simulation_env/",
        use_gui=False,
        num_seconds=18000,
        start_at=0,
        action_every_steps=900,
    )


# temp = env_creator(None)
# temp.reset()
register_env("custom_env", env_creator)

train("custom_env")
