from collections import OrderedDict
from typing import Any, Dict

import numpy as np

from xcs224r.critics.bootstrapped_continuous_critic import \
    BootstrappedContinuousCritic
from xcs224r.infrastructure.replay_buffer import ReplayBuffer
from xcs224r.infrastructure.utils import *
from xcs224r.policies.MLP_policy import MLPPolicyAC
from .base_agent import BaseAgent


class ACAgent(BaseAgent):
    def __init__(self, env: Any, agent_params: Dict[str, Any]) -> None:
        super(ACAgent, self).__init__()

        self.env = env
        self.agent_params = agent_params

        self.gamma: float = self.agent_params['gamma']
        self.standardize_advantages: bool = self.agent_params['standardize_advantages']

        self.actor = MLPPolicyAC(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            self.agent_params['discrete'],
            self.agent_params['learning_rate'],
        )
        self.critic = BootstrappedContinuousCritic(self.agent_params)

        self.replay_buffer = ReplayBuffer()

    def train(self, ob_no: np.ndarray, ac_na: np.ndarray, re_n: np.ndarray, next_ob_no: np.ndarray, terminal_n: np.ndarray) -> Dict[str, Any]:
        raise NotImplementedError
        # Not needed for this homework

    ####################################
    ####################################

    def estimate_advantage(self, ob_no: np.ndarray, next_ob_no: np.ndarray, re_n: np.ndarray, terminal_n: np.ndarray) -> np.ndarray:
        raise NotImplementedError
        # Not needed for this homework

    ####################################
    ####################################
