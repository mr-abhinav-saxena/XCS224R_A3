from .base_critic import BaseCritic
import torch
import torch.optim as optim
from torch.nn import utils
from torch import nn
import pdb
import numpy as np
from typing import Any, Dict

from xcs224r.infrastructure import pytorch_util as ptu

class IQLCritic(BaseCritic):

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
        self.mse_loss = nn.MSELoss()
        self.q_net.to(ptu.device)
        self.q_net_target.to(ptu.device)

        # TODO define value function
        # HINT: see Q_net definition above, using network_initializer and use .to(ptu.device)
        # HINT: Define using same hparams as Q_net, but adjust output dimensions
        # *** START CODE HERE ***

        self.v_net = network_initializer(self.ob_dim, 1)
        self.v_net.to(ptu.device)

        # *** END CODE HERE ***

        self.v_optimizer = self.optimizer_spec.constructor(
            self.v_net.parameters(),
            **self.optimizer_spec.optim_kwargs
        )
        
        self.v_learning_rate_scheduler = optim.lr_scheduler.LambdaLR(
            self.v_optimizer,
            self.optimizer_spec.learning_rate_schedule,
        )
        self.iql_expectile: float = hparams['iql_expectile']

    def expectile_loss(self, diff):
        pass
        # TODO: Implement the expectile loss given the difference between q and v
        # HINT: self.iql_expectile provides the \zeta value as described in the problem statement.
        # HINT: torch.where can be useful for implementing the piecewise nature of the expectile loss.
        # HINT: ptu.from_numpy may be useful
        # *** START CODE HERE ***

        # diff means q - v, so we need to compute the expectile loss based on this difference. The expectile loss is defined as:
        # L(v, q) = |q - v|^2 * (zeta * 1_{q - v >= 0} + (1 - zeta) * 1_{q - v < 0})
        # where zeta is the expectile parameter
        zeta = self.iql_expectile
        loss = torch.where(diff >= 0, zeta * diff ** 2, (1 - zeta) * diff ** 2)
        return loss

        # *** END CODE HERE ***


    def update_v(self, ob_no: np.ndarray, ac_na: np.ndarray) -> Dict[str, Any]:
        """
        Update value function using expectile loss
        """
        ob_no = ptu.from_numpy(ob_no)
        ac_na = ptu.from_numpy(ac_na).to(torch.long)

        # TODO: Compute loss for v_net (value_loss) using expectile_loss defined above
        # HINT: use target q network to train V
        # HINT: Use self.expectile_loss as defined above, passing in the difference between the computed targets and predictions.
        # HINT: For the computed targets, you may need to detach the target q values from the computation graph, since they should not backpropagate gradients into the q_net.
        # Name of the loss variable should be value_loss, since it will be used in the optimizer step below.
        # *** START CODE HERE ***

        # Uses the target Q-network to compute Q_target(s,a), 
        # detaches it to prevent gradients from flowing into Q, 
        # and implements L_V(φ) = E[L_2^ζ(Q_θ(s,a) - V_φ(s))] which is expectile loss to the difference. This learns V to approximate an upper expectile of Q-values.
        # Compute Q_target(s,a) for the given actions

        with torch.no_grad(): # Detach Q-values to prevent gradients from flowing into Q-network
            q_targets = self.q_net_target(ob_no) # shape (batch_size, ac_dim)
            q_targets = torch.gather(q_targets, 1, ac_na.unsqueeze(1)).squeeze(1)  # Q(s,a) for the taken actions -> shape (batch_size,)

        v_predictions = self.v_net(ob_no)  # V(s) -> shape (batch_size, 1)
        v_predictions = v_predictions.squeeze(1)  # shape (batch_size,)

        value_loss = torch.mean(self.expectile_loss(q_targets - v_predictions))  # Compute expectile loss

        # *** END CODE HERE ***
        

        self.v_optimizer.zero_grad()
        value_loss.backward()
        utils.clip_grad_value_(self.v_net.parameters(), self.grad_norm_clipping)
        self.v_optimizer.step()
        
        self.v_learning_rate_scheduler.step()

        return {'Training V Loss': ptu.to_numpy(value_loss)}



    def update_q(self, ob_no: np.ndarray, ac_na: np.ndarray, next_ob_no: np.ndarray, reward_n: np.ndarray, terminal_n: np.ndarray) -> Dict[str, Any]:
        """
        Use target v network to train Q
        """
        ob_no = ptu.from_numpy(ob_no) # shape (batch_size, ob_dim)
        ac_na = ptu.from_numpy(ac_na).to(torch.long) # shape (batch_size,)
        next_ob_no = ptu.from_numpy(next_ob_no) # shape (batch_size, ob_dim)
        reward_n = ptu.from_numpy(reward_n) # shape (batch_size,)
        terminal_n = ptu.from_numpy(terminal_n) # shape (batch_size,)
        
        
        # TODO: Compute loss for updating Q_net parameters
        # HINT: Note that if the next state is terminal, its target reward value needs to be adjusted.
        # HINT: target v values should be detached from the computation graph, since they should not backpropagate gradients into the v_net.
        # loss variable name should be loss, since it will be used in the optimizer step below.
        # *** START CODE HERE ***

        # Uses V(s') as the target (detached) instead of Q(s',a'), 
        # which avoids the expectation over actions and keeps gradients only on in-distribution actions.
        # Handles terminal states by zeroing the next state value.
        # Compute Q(s,a) predictions
        # Implements L_Q(θ) = E[(r(s,a) + γV_φ(s') - Q_θ(s,a))²]. 

        with torch.no_grad(): # detach V-values to prevent gradients from flowing into V-network
            v_next = self.v_net(next_ob_no)  # V(s') -> shape (batch_size, 1)
            v_next = v_next.squeeze(1)  # shape (batch_size,)
            q_target = reward_n + self.gamma * v_next * (1 - terminal_n)  # shape (batch_size,)

        q_predictions = self.q_net(ob_no)  # Q(s,a) -> shape (batch_size, ac_dim)
        q_predictions = torch.gather(q_predictions, 1, ac_na.unsqueeze(1)).squeeze(1) # Q(s,a) for the taken actions -> shape (batch_size,)

        loss = self.mse_loss(q_predictions, q_target)  # Compute MSE loss between Q predictions and targets

        # *** END CODE HERE ***
        self.optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_value_(self.q_net.parameters(), self.grad_norm_clipping)
        self.optimizer.step()
        
        self.learning_rate_scheduler.step()

        return {'Training Q Loss': ptu.to_numpy(loss)}

    def update_target_network(self) -> None:
        for target_param, param in zip(
                self.q_net_target.parameters(), self.q_net.parameters()
        ):
            target_param.data.copy_(param.data)

    def qa_values(self, obs: np.ndarray) -> np.ndarray:
        obs = ptu.from_numpy(obs)
        qa_values = self.q_net(obs)
        return ptu.to_numpy(qa_values)
