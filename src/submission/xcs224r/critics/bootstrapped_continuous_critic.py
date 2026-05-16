from .base_critic import BaseCritic
from torch import nn
from torch import optim
import pdb
from typing import Any, Dict
import numpy as np

from xcs224r.infrastructure import pytorch_util as ptu


class BootstrappedContinuousCritic(nn.Module, BaseCritic):
    """
        Notes on notation:

        Prefixes and suffixes:
        ob - observation
        ac - action
        _no - this tensor should have shape (batch size /n/, observation dim)
        _na - this tensor should have shape (batch size /n/, action dim)
        _n  - this tensor should have shape (batch size /n/)

        Note: batch size /n/ is defined at runtime.
        is None
    """
    def __init__(self, hparams: Dict[str, Any]) -> None:
        super().__init__()
        self.ob_dim = hparams['ob_dim']
        self.ac_dim = hparams['ac_dim']
        self.discrete: bool = hparams['discrete']
        self.size: int = hparams['size']
        self.n_layers: int = hparams['n_layers']
        self.learning_rate: float = hparams['learning_rate']

        # critic parameters
        self.num_target_updates: int = hparams['num_target_updates']
        self.num_grad_steps_per_target_update: int = hparams['num_grad_steps_per_target_update']
        self.gamma: float = hparams['gamma']
        self.critic_network = ptu.build_mlp(
            self.ob_dim,
            1,
            n_layers=self.n_layers,
            size=self.size,
        )
        self.critic_network.to(ptu.device)
        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.critic_network.parameters(),
            self.learning_rate,
        )

    def forward(self, obs: Any) -> Any:
        return self.critic_network(obs).squeeze(1)

    def forward_np(self, obs: np.ndarray) -> np.ndarray:
        obs = ptu.from_numpy(obs)
        predictions = self(obs)
        return ptu.to_numpy(predictions)

    def update(self, ob_no: np.ndarray, ac_na: np.ndarray, next_ob_no: np.ndarray, reward_n: np.ndarray, terminal_n: np.ndarray) -> Dict[str, Any]:
        """
            Update the parameters of the critic.

            let sum_of_path_lengths be the sum of the lengths of the paths sampled from
                Agent.sample_trajectories
            let num_paths be the number of paths sampled from Agent.sample_trajectories

            arguments:
                ob_no: shape: (sum_of_path_lengths, ob_dim)
                next_ob_no: shape: (sum_of_path_lengths, ob_dim). The observation after taking one step forward
                reward_n: length: sum_of_path_lengths. Each element in reward_n is a scalar containing
                    the reward for each timestep
                terminal_n: length: sum_of_path_lengths. Each element in terminal_n is either 1 if the episode ended
                    at that timestep of 0 if the episode did not end

            returns:
                nothing
        """
        raise NotImplementedError
        # Not needed for this homework

    ####################################
    ####################################
