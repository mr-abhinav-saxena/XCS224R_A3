from .base_critic import BaseCritic
import torch
import torch.optim as optim
from torch.nn import utils
from torch import nn
import pdb
import numpy as np
from typing import Any, Dict, Tuple

from xcs224r.infrastructure import pytorch_util as ptu


class CQLCritic(BaseCritic):

    def __init__(self, hparams: Dict[str, Any], optimizer_spec: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.env_name = hparams['env_name']
        self.ob_dim = hparams['ob_dim']

        if isinstance(self.ob_dim, int):
            self.input_shape = (self.ob_dim,)
        else:
            self.input_shape = hparams['input_shape']

        self.ac_dim: int = hparams['ac_dim']
        self.double_q: bool = hparams['double_q']
        self.grad_norm_clipping: float = hparams['grad_norm_clipping']
        self.gamma: float = hparams['gamma']

        self.optimizer_spec = optimizer_spec
        network_initializer = hparams['q_func']
        self.q_net = network_initializer(self.ob_dim, self.ac_dim)
        self.q_net_target = network_initializer(self.ob_dim, self.ac_dim)
        self.optimizer = self.optimizer_spec.constructor(
            self.q_net.parameters(),
            **self.optimizer_spec.optim_kwargs
        )
        self.learning_rate_scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            self.optimizer_spec.learning_rate_schedule,
        )
        self.loss = nn.MSELoss()
        self.q_net.to(ptu.device)
        self.q_net_target.to(ptu.device)
        self.cql_alpha: float = hparams['cql_alpha']

    def dqn_loss(self, ob_no: torch.Tensor, ac_na: torch.Tensor, next_ob_no: torch.Tensor, reward_n: torch.Tensor, terminal_n: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qa_t_values = self.q_net(ob_no) # shape: (batch_size, ac_dim)
        q_t_values = torch.gather(qa_t_values, 1, ac_na.unsqueeze(1)).squeeze(1) # shape: (batch_size,)
        qa_tp1_values = self.q_net_target(next_ob_no) # shape: (batch_size, ac_dim)

        next_actions = self.q_net(next_ob_no).argmax(dim=1)
        q_tp1 = torch.gather(qa_tp1_values, 1, next_actions.unsqueeze(1)).squeeze(1)

        target = reward_n + self.gamma * q_tp1 * (1 - terminal_n)
        target = target.detach() # shape: (batch_size,)
        loss = self.loss(q_t_values, target) # shape: (batch_size,)

        return loss, qa_t_values, q_t_values


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
        ob_no = ptu.from_numpy(ob_no) # shape: (sum_of_path_lengths, ob_dim)
        ac_na = ptu.from_numpy(ac_na).to(torch.long) # shape: (sum_of_path_lengths,)
        next_ob_no = ptu.from_numpy(next_ob_no) # shape: (sum_of_path_lengths, ob_dim)
        reward_n = ptu.from_numpy(reward_n) # shape: (sum_of_path_lengths,)
        terminal_n = ptu.from_numpy(terminal_n) # shape: (sum_of_path_lengths,)

        # TODO: CQL Implementation
        # HINT: Obtain DQN loss, qa_t_values, q_t_values using self.dqn_loss
        # HINT: After calculating cql_loss, augment the loss appropriately
        # HINT: torch.logsumexp and torch.mean may be useful for calculating the cql_loss
        
        # *** START CODE HERE ***

        # Conservative Q-Learning (CQL) adds a regularizer to prevent overestimation of Q-values.
        # The CQL objective is: TD_loss + α * [mean(logsumexp(Q(s,a)) - Q(s,a_i))]
        # where logsumexp(Q(s,a)) = log(Σ_a exp(Q(s,a))) and a_i is the in-distribution action.
        
        # 1. Get DQN loss and Q-values
        dqn_loss, qa_t_values, q_t_values = self.dqn_loss(ob_no, ac_na, next_ob_no, reward_n, terminal_n)

        # 2. Compute logsumexp of Q-values over all actions for each state
        # qa_t_values shape: (batch_size, ac_dim)
        # logsumexp computes log(Σ_a exp(Q(s,a))) for each state
        q_t_logsumexp = torch.logsumexp(qa_t_values, dim=1) # shape: (batch_size,)

        # 3. Compute CQL regularizer: mean(logsumexp(Q(s,a)) - Q(s,a_i))
        # This penalizes high Q-values for OOD actions relative to in-distribution actions
        # q_t_values shape: (batch_size,) - Q-values for in-distribution actions
        cql_loss = torch.mean(q_t_logsumexp - q_t_values) # shape: (1,)

        # 4. Total loss = DQN loss + α * CQL regularizer
        # The CQL regularizer encourages conservative Q-values by penalizing
        # the difference between logsumexp (all actions) and in-distribution Q-values
        loss = dqn_loss + self.cql_alpha * cql_loss

        # *** END CODE HERE ***
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        
        info = {'Training Loss': ptu.to_numpy(loss)}
        # Note: the following will be failing until you implement above parts
        info['CQL Loss'] = ptu.to_numpy(cql_loss)
        info['Data q-values'] = ptu.to_numpy(q_t_values).mean()
        info['OOD q-values'] = ptu.to_numpy(q_t_logsumexp).mean()
        
        self.learning_rate_scheduler.step()

        return info

    def update_target_network(self) -> None:
        for target_param, param in zip(
                self.q_net_target.parameters(), self.q_net.parameters()
        ):
            target_param.data.copy_(param.data)

    def qa_values(self, obs: np.ndarray) -> np.ndarray:
        obs = ptu.from_numpy(obs)
        qa_values = self.q_net(obs)
        return ptu.to_numpy(qa_values)
