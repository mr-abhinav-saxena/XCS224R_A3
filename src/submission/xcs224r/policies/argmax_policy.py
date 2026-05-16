import numpy as np
import pdb
from typing import Any

class ArgMaxPolicy(object):

    def __init__(self, critic: Any, use_boltzmann: bool=False) -> None:
        self.critic = critic
        self.use_boltzmann = use_boltzmann

    def set_critic(self, critic: Any) -> None:
        self.critic = critic

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        if len(obs.shape) > 3:
            observation = obs
        else:
            observation = obs[None]

        ## <DONE> return the action that maximizes the Q-value 
        # at the current observation as the output
        q_values = self.critic.qa_values(observation)

        if self.use_boltzmann:
            distribution = np.exp(q_values) / np.sum(np.exp(q_values))
            action = self.sample_discrete(distribution)
        else:
            action = q_values.argmax(-1)

        return action[0]

    def sample_discrete(self, p: np.ndarray) -> np.ndarray:
        c = p.cumsum(axis=1)
        u = np.random.rand(len(c), 1)
        choices = (u < c).argmax(axis=1)
        return choices

    ####################################
    ####################################