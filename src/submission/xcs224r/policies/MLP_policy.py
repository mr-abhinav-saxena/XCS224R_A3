import abc
import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from torch import distributions

from xcs224r.infrastructure import pytorch_util as ptu
from xcs224r.policies.base_policy import BasePolicy


class MLPPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):

    def __init__(self,
                 ac_dim: int,
                 ob_dim: int,
                 n_layers: int,
                 size: int,
                 discrete: bool=False,
                 learning_rate: float=1e-4,
                 training: bool=True,
                 nn_baseline: bool=False,
                 **kwargs: Any
                 ) -> None:
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.discrete = discrete
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline

        if self.discrete:
            self.logits_na = ptu.build_mlp(input_size=self.ob_dim,
                                           output_size=self.ac_dim,
                                           n_layers=self.n_layers,
                                           size=self.size)
            self.logits_na.to(ptu.device)
            self.mean_net: Any = None
            self.logstd: Any = None
            self.optimizer = optim.Adam(self.logits_na.parameters(),
                                        self.learning_rate)
        else:
            self.logits_na = None
            self.mean_net = ptu.build_mlp(input_size=self.ob_dim,
                                      output_size=self.ac_dim,
                                      n_layers=self.n_layers, size=self.size)
            self.logstd = nn.Parameter(
                torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
            )
            self.mean_net.to(ptu.device)
            self.logstd.to(ptu.device)
            self.optimizer = optim.Adam(
                itertools.chain([self.logstd], self.mean_net.parameters()),
                self.learning_rate
            )

        if nn_baseline:
            self.baseline: Any = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=1,
                n_layers=self.n_layers,
                size=self.size,
            )
            self.baseline.to(ptu.device)
            self.baseline_optimizer: Any = optim.Adam(
                self.baseline.parameters(),
                self.learning_rate,
            )
        else:
            self.baseline = None

    ##################################

    def save(self, filepath: str) -> None:
        torch.save(self.state_dict(), filepath)

    ##################################

    # query the policy with observation(s) to get selected action(s)
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]
        observation_t = ptu.from_numpy(observation)
        action_distribution = self(observation_t)
        action = action_distribution.sample()  # don't bother with rsample
        return ptu.to_numpy(action)

    ####################################
    ####################################

    # update/train this policy
    def update(self, observations: Union[np.ndarray, torch.Tensor], actions: Union[np.ndarray, torch.Tensor], **kwargs: Any) -> Any:
        raise NotImplementedError

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor) -> distributions.Distribution:
        if self.discrete:
            logits = self.logits_na(observation)
            action_distribution = distributions.Categorical(logits=logits)
            return action_distribution
        else:
            batch_mean = self.mean_net(observation)
            scale_tril = torch.diag(torch.exp(self.logstd))
            batch_dim = batch_mean.shape[0]
            batch_scale_tril = scale_tril.repeat(batch_dim, 1, 1)
            action_distribution = distributions.MultivariateNormal(
                batch_mean,
                scale_tril=batch_scale_tril,
            )
            return action_distribution

    ####################################
    ####################################


#####################################################
#####################################################


class MLPPolicyAC(MLPPolicy):
    # MJ: cut acs_labels_na and qvals from the signature if they are not used
    def update(
            self, observations: Union[np.ndarray, torch.Tensor], actions: Union[np.ndarray, torch.Tensor],
            adv_n: Any=None, acs_labels_na: Any=None, qvals: Any=None
    ) -> Any:
        raise NotImplementedError
        # Not needed for this homework

    ####################################
    ####################################

class MLPPolicyAWAC(MLPPolicy):
    def __init__(self,
                 ac_dim: int,
                 ob_dim: int,
                 n_layers: int,
                 size: int,
                 discrete: bool=False,
                 learning_rate: float=1e-4,
                 training: bool=True,
                 nn_baseline: bool=False,
                 lambda_awac: float=10,
                 **kwargs: Any,
                 ) -> None:
        self.lambda_awac = lambda_awac
        super().__init__(ac_dim, ob_dim, n_layers, size, discrete, learning_rate, training, nn_baseline, **kwargs)
    
    def update(self, observations: Union[np.ndarray, torch.Tensor], actions: Union[np.ndarray, torch.Tensor], adv_n: Any=None) -> float:
        if adv_n is None:
            assert False
        if isinstance(observations, np.ndarray):
            observations = ptu.from_numpy(observations)
        if isinstance(actions, np.ndarray):
            actions = ptu.from_numpy(actions)
        if isinstance(adv_n, np.ndarray):
            adv_n = ptu.from_numpy(adv_n)

        dist = self(observations)
        log_prob_n = dist.log_prob(actions)
        actor_loss = -log_prob_n * torch.exp(adv_n/self.lambda_awac)
        actor_loss = actor_loss.mean()
        
        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()
        
        return actor_loss.item()
