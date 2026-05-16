from xcs224r.infrastructure import pytorch_util as ptu
from .base_exploration_model import BaseExplorationModel
import torch.optim as optim
from torch import nn
import torch
import numpy as np
from typing import Any, Dict

def init_method_1(model: nn.Module) -> None:
    model.weight.data.uniform_()
    model.bias.data.uniform_()

def init_method_2(model: nn.Module) -> None:
    model.weight.data.normal_()
    model.bias.data.normal_()


class RNDModel(nn.Module, BaseExplorationModel):
    def __init__(self, hparams: Dict[str, Any], optimizer_spec: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.ob_dim = hparams['ob_dim']
        self.output_size: int = hparams['rnd_output_size']
        self.n_layers: int = hparams['rnd_n_layers']
        self.size: int = hparams['rnd_size']
        self.optimizer_spec = optimizer_spec

        self.f = ptu.build_mlp(self.ob_dim, self.output_size, self.n_layers, self.size, init_method=init_method_1)
        self.f_hat = ptu.build_mlp(self.ob_dim, self.output_size, self.n_layers, self.size, init_method=init_method_2)
        
        self.optimizer = self.optimizer_spec.constructor(
            self.f_hat.parameters(),
            **self.optimizer_spec.optim_kwargs
        )
        self.learning_rate_scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            self.optimizer_spec.learning_rate_schedule,
        )

        self.f.to(ptu.device)
        self.f_hat.to(ptu.device)

    def forward(self, ob_no: torch.Tensor) -> torch.Tensor:
        targets = self.f(ob_no).detach()
        predictions = self.f_hat(ob_no)
        return torch.norm(predictions - targets, dim=1)

    def forward_np(self, ob_no: np.ndarray) -> np.ndarray:
        ob_no_t = ptu.from_numpy(ob_no)
        error = self(ob_no_t)
        return ptu.to_numpy(error)

    def update(self, ob_no: np.ndarray) -> float:
        prediction_errors = self(ptu.from_numpy(ob_no))
        loss = torch.mean(prediction_errors)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
