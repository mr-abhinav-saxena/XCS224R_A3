from typing import Union, Any, List

import torch
from torch import nn
import numpy as np

Activation = Union[str, nn.Module]


_str_to_activation = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(),
    'identity': nn.Identity(),
}


def build_mlp(
        input_size: int,
        output_size: int,
        n_layers: int,
        size: int,
        activation: Activation = 'tanh',
        output_activation: Activation = 'identity',
        init_method: Any=None,
) -> nn.Module:
    """
        Builds a feedforward neural network
        arguments:
            input_placeholder: placeholder variable for the state (batch_size, input_size)
            scope: variable scope of the network
            n_layers: number of hidden layers
            size: dimension of each hidden layer
            activation: activation of each hidden layer
            input_size: size of the input layer
            output_size: size of the output layer
            output_activation: activation of the output layer
        returns:
            output_placeholder: the result of a forward pass through the hidden layers + the output layer
    """
    if isinstance(activation, str):
        activation = _str_to_activation[activation]
    if isinstance(output_activation, str):
        output_activation = _str_to_activation[output_activation]
    layers: List[nn.Module] = []
    in_size = input_size
    for _ in range(n_layers):
        curr_layer = nn.Linear(in_size, size)
        if init_method is not None:
            curr_layer.apply(init_method)
        layers.append(curr_layer)
        layers.append(activation)
        in_size = size

    last_layer = nn.Linear(in_size, output_size)
    if init_method is not None:
        last_layer.apply(init_method)

    layers.append(last_layer)
    layers.append(output_activation)
        
    return nn.Sequential(*layers)


device: Any = None


def init_gpu(use_gpu: bool=True, gpu_id: int=0) -> None:
    global device
    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda:" + str(gpu_id))
        print("Using GPU id {}".format(gpu_id))
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built() and use_gpu:
        device = torch.device("mps")
        print("PyTorch detects an Apple GPU: running on MPS")
    else:
        device = torch.device("cpu")
        print("GPU not detected. Defaulting to CPU.")


def set_device(gpu_id: int) -> None:
    torch.cuda.set_device(gpu_id)


def from_numpy(*args: Any, **kwargs: Any) -> torch.Tensor:
    return torch.from_numpy(*args, **kwargs).float().to(device)

def ones(*args: Any, **kwargs: Any) -> torch.Tensor:
    return torch.ones(*args, **kwargs).to(device)


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.to('cpu').detach().numpy()
