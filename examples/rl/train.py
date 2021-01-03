import numpy as np
from gym import spaces
from ray import tune
from ray.rllib.agents.a3c.a2c import A2CTrainer
from ray.rllib.agents.a3c.a3c import A3CTrainer
from ray.rllib.agents.a3c.a3c_tf_policy import A3CTFPolicy
from ray.rllib.agents.dqn import DQNTrainer, ApexTrainer
from ray.rllib.agents.dqn.dqn_tf_policy import DQNTFPolicy
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.ppo.appo import APPOTrainer
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
from ray.rllib.agents.ppo.appo_tf_policy import AsyncPPOTFPolicy

from ray.rllib.models import ModelCatalog
from ray.tune.logger import pretty_print

from tf_models import MaskedActionsCNN

def policy_mapping(agent_id):
    return "default"


used_policy = DQNTFPolicy

policies = {
    "default": (
        used_policy,
        spaces.Dict(
            {
                "action_mask": spaces.Box(low=0, high=1, shape=(2,), dtype=np.uint8),
            "obs": spaces.Box(low=0.0, high=10.0, shape=(12, 12, 3)),
                "extra": spaces.Box(low=0.0, high=1.0, shape=(96,)),
            }
        ),
        spaces.Discrete(2),
        {},
    ),
}


def train(env_name):
    ModelCatalog.register_custom_model("masked_actions_model", MaskedActionsCNN)
    model_config = {
        "custom_model": "masked_actions_model",
        "conv_filters": [[16, [2, 2], 1], [32, [2, 2], 1]],
        "conv_activation": "elu",
        "fcnet_hiddens": [128],
        "fcnet_activation": "elu",
    }
    tune_config = {
        "num_workers": 24,
        "num_gpus": 1,
        "batch_mode": "complete_episodes",
        "model": model_config,
        "env": env_name,
        "lr": 0.001,
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": policy_mapping,
        },
        "framework": "tf"
    }
    trainer = DQNTrainer(env=env_name, config=tune_config)
    for i in range(1000):
        print("== Iteration {}==".format(i))
        results = trainer.train()
        pretty_print(results)
        checkpoint = trainer.save()
        print("\nCheckpoint saved at {}\n".format(checkpoint))
