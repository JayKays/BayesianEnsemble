
from typing import List, Sequence

import torch
from torch import nn as nn

from blitz.modules.linear_bayesian_layer import BayesianLinear
from blitz.modules.base_bayesian_module import BayesianModule
from torch._C import is_grad_enabled


class EnsembleLinearBayesian(BayesianModule):

    """Efficient linear layer for ensemble models."""

    def __init__(
        self, num_members: int, in_size: int, out_size: int, bias: bool = True, freeze: bool = False
    ):
        super().__init__()
        self.num_members = num_members
        self.in_size = in_size
        self.out_size = out_size
        self.use_bias = bias
        self.freeze = freeze
        
        self.member_layers = [BayesianLinear(self.in_size, self.out_size, self.use_bias) for _ in range(self.num_members)]
        
        for i in range(self.num_members):
            self.add_module("bl{i}", self.member_layers[i])
        # self.params = nn.Parameter(layer.parameters() for layer in self.member_layers)
        self.log_prior = torch.Tensor([layer.log_prior for layer in self.member_layers])
        self.log_variational_posterior = torch.Tensor([layer.log_variational_posterior for layer in self.member_layers])

        self.elite_models: List[int] = None
        self.use_only_elite = False

    def forward(self, x):
        if x.ndim == 3: x = x.squeeze()
        if x.ndim == 2: x = x.unsqueeze(0).repeat(self.num_members,1,1)
        # print(x.dim())
        if self.freeze:
            xw = torch.stack([self.member_layers[i].forward_frozen(x[i]) for i in range(self.num_members)])
        else:    
            xw = torch.stack([self.member_layers[i].forward(x[i]) for i in range(self.num_members)])

            self.log_prior = torch.Tensor([layer.log_prior for layer in self.member_layers])
            self.log_variational_posterior = torch.Tensor([layer.log_variational_posterior for layer in self.member_layers])
            
        if self.use_only_elite:
            return xw[self.elite_models]

        return xw


    def extra_repr(self) -> str:
        return (
            f"num_members={self.num_members}, in_size={self.in_size}, "
            f"out_size={self.out_size}, bias={self.use_bias}"
        )

    def set_elite(self, elite_models: Sequence[int]):
        self.elite_models = list(elite_models)

    def toggle_use_only_elite(self):
        self.use_only_elite = not self.use_only_elite


if __name__ == "__main__":
    from mbrl.models.util import EnsembleLinearLayer
    from blitz.losses.kl_divergence import kl_divergence_from_nn
    input_size = 5
    output_size = 3
    ensemble_size = 3
    batch_size = 10
    test_layer = EnsembleLinearBayesian(ensemble_size, input_size, output_size)
    comp_layer = EnsembleLinearLayer(ensemble_size, input_size, output_size)
    x = torch.Tensor([1,1,1,1,1])
    batch = torch.vstack([(i+1)*x for i in range(batch_size)])
    print(test_layer.log_variational_posterior)
    print(test_layer.log_prior)
    out = test_layer.forward(batch)
    print(test_layer.log_variational_posterior)
    print(test_layer.log_prior)
    print(out)



