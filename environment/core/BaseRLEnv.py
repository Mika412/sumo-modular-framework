from abc import abstractmethod

from ray.rllib import ExternalEnv

from .BaseEnv import SumoBaseEnvironment
import matplotlib.pyplot as plt
from statistics import mean
import math
from itertools import islice

from ray.rllib.env.multi_agent_env import MultiAgentEnv

import gym


class SumoRLBaseEnvironment(MultiAgentEnv, SumoBaseEnvironment):
    def __init__(
        self,
        env_dir,
        use_gui=False,
        num_seconds=20000,
        start_at=0,
        action_every_steps=1,
    ):
        super().__init__(env_dir, False, use_gui, num_seconds, start_at)

        self.action_every_steps = action_every_steps

    def reset(self):
        self._reset()
        SumoBaseEnvironment.reset(self)
        return self.compute_observations()

    @property
    @abstractmethod
    def reward_range(self):
        pass

    @property
    @abstractmethod
    def action_space(self):
        pass

    @property
    @abstractmethod
    def observation_space(self):
        pass

    @abstractmethod
    def episode_rewards(self):
        pass

    @property
    def done(self):
        return {"__all__": self.is_done}

    def step(self, actions):
        self.step_actions(actions)

        perform_num_steps = int(
            max(self.action_every_steps, 1) / self.timestep_length_seconds)
        for i in range(perform_num_steps):
            SumoBaseEnvironment.step(self)

        self.update_reward()
        observation = self.compute_observations()

        reward = self.compute_rewards()
        return observation, reward, self.done, {}

    @abstractmethod
    def step_actions(self, action):
        pass

    @abstractmethod
    def update_reward(self):
        pass

    @abstractmethod
    def compute_observations(self):
        pass

    @abstractmethod
    def compute_rewards(self):
        pass

    @abstractmethod
    def _reset(self):
        pass
