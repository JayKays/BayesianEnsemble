
import pathlib
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import hydra
import omegaconf
import torch
from torch import nn as nn
from torch.nn import functional as F

import mbrl.util.math

from mbrl.models.model import Ensemble
from mbrl.models.util import EnsembleLinearLayer, truncated_normal_init

from blitz.modules.base_bayesian_module import BayesianModule
from blitz.losses.kl_divergence import kl_divergence_from_nn

from .utils import EnsembleLinearBayesian

class BNN(Ensemble):
    """Implements a linear Bayesian Ensemble with the help of blitz,
    in a similar fassion as the Gaussian MLP from mbrl-lib.
    """

    def __init__(
        self,
        in_size: int,
        out_size: int,
        device: Union[str, torch.device],
        num_layers: int = 4,
        ensemble_size: int = 1,
        hid_size: int = 200,
        deterministic: bool = True,
        freeze: bool = False,
        propagation_method: Optional[str] = None,
        learn_logvar_bounds: bool = False,
        activation_fn_cfg: Optional[Union[Dict, omegaconf.DictConfig]] = None,
    ):
        super().__init__(
            ensemble_size, device, propagation_method, deterministic=deterministic
        )

        self.in_size = in_size
        self.out_size = out_size

        def create_activation():
            if activation_fn_cfg is None:
                activation_func = nn.ReLU()
            else:
                # Handle the case where activation_fn_cfg is a dict
                cfg = omegaconf.OmegaConf.create(activation_fn_cfg)
                activation_func = hydra.utils.instantiate(cfg)
            return activation_func

        def create_linear_layer(l_in, l_out):
            return EnsembleLinearBayesian(ensemble_size, l_in, l_out)

        hidden_layers = [
            nn.Sequential(create_linear_layer(in_size, hid_size), create_activation())
        ]
        for _ in range(num_layers - 1):
            hidden_layers.append(
                nn.Sequential(
                    create_linear_layer(hid_size, hid_size),
                    create_activation(),
                )
            )
        self.hidden_layers = nn.Sequential(*hidden_layers)

        self.output_layer = create_linear_layer(hid_size, out_size)
        
        # self.apply(truncated_normal_init)

        self.freeze = freeze

        if self.freeze: self.freeze_model()

        self.to(self.device)
        self.elite_models: List[int] = None


    def _maybe_toggle_layers_use_only_elite(self, only_elite: bool):
        if self.elite_models is None:
            return
        if self.num_members > 1 and only_elite:
            for layer in self.hidden_layers:
                # each layer is (linear layer, activation_func)
                layer[0].set_elite(self.elite_models)
                layer[0].toggle_use_only_elite()
            self.output_layer.set_elite(self.elite_models)
            self.output_layer.toggle_use_only_elite()

    def _default_forward(
        self, x: torch.Tensor, only_elite: bool = False, **_kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        self._maybe_toggle_layers_use_only_elite(only_elite)

        x = self.hidden_layers(x)
        output = self.output_layer(x)

        self._maybe_toggle_layers_use_only_elite(only_elite)

        return output

    def _forward_from_indices(
        self, x: torch.Tensor, model_shuffle_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        _, batch_size, _ = x.shape

        num_models = (
            len(self.elite_models) if self.elite_models is not None else len(self)
        )
        shuffled_x = x[:, model_shuffle_indices, ...].view(
            num_models, batch_size // num_models, -1
        )

        pred = self._default_forward(shuffled_x, only_elite=True)
        # note that pred is shuffled
        pred = pred.view(batch_size, -1)
        pred[model_shuffle_indices] = pred.clone()  # invert the shuffle

        return pred

    def _forward_ensemble(
        self,
        x: torch.Tensor,
        rng: Optional[torch.Generator] = None,
        propagation_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.propagation_method is None:
            mean = self._default_forward(x, only_elite=False)
            if self.num_members == 1:
                mean = mean[0]
            return mean
        assert x.ndim == 2
        model_len = (
            len(self.elite_models) if self.elite_models is not None else len(self)
        )
        if x.shape[0] % model_len != 0:
            raise ValueError(
                f"GaussianMLP ensemble requires batch size to be a multiple of the "
                f"number of models. Current batch size is {x.shape[0]} for "
                f"{model_len} models."
            )
        x = x.unsqueeze(0)
        if self.propagation_method == "random_model":
            # passing generator causes segmentation fault
            # see https://github.com/pytorch/pytorch/issues/44714
            model_indices = torch.randperm(x.shape[1], device=self.device)
            return self._forward_from_indices(x, model_indices)
        if self.propagation_method == "fixed_model":
            if propagation_indices is None:
                raise ValueError(
                    "When using propagation='fixed_model', `propagation_indices` must be provided."
                )
            return self._forward_from_indices(x, propagation_indices)
        if self.propagation_method == "expectation":
            pred = self._default_forward(x, only_elite=True)
            return pred.mean(dim=0)

        raise ValueError(f"Invalid propagation method {self.propagation_method}.")

    def forward(  # type: ignore
        self,
        x: torch.Tensor,
        rng: Optional[torch.Generator] = None,
        propagation_indices: Optional[torch.Tensor] = None,
        use_propagation: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predictions for the given input.
        """
        
        if use_propagation:
            return self._forward_ensemble(
                x, rng=rng, propagation_indices=propagation_indices
            )
        return self._default_forward(x)

    def _mse_loss(self, model_in: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert model_in.ndim == target.ndim
        
        pred_mean = self.forward(model_in, use_propagation=False)
        return F.mse_loss(pred_mean, target, reduction="none").sum((1, 2)).sum()

    def loss(
        self,
        model_in: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Computes the samples the ELBO loss via variational MonteCarlo, if froze
            the mse loss is instead calculated.
        It also includes terms for ``max_logvar`` and ``min_logvar`` with small weights,
        with positive and negative signs, respectively.
        This function returns no metadata, so the second output is set to an empty dict.
        Args:
            model_in (tensor): input tensor. The shape must be ``E x B x Id``, or ``B x Id``
                where ``E``, ``B`` and ``Id`` represent ensemble size, batch size, and input
                dimension, respectively.
            target (tensor): target tensor. The shape must be ``E x B x Id``, or ``B x Od``
                where ``E``, ``B`` and ``Od`` represent ensemble size, batch size, and output
                dimension, respectively.
        Returns:
            (tensor): a loss tensor representing the Gaussian negative log-likelihood of
            the model over the given input/target. If the model is an ensemble, returns
            the average over all models.
        """
        if self.freeze:
            self.freeze_model()
            return self._mse_loss(model_in, target), {}

        return self.sample_elbo(model_in, target), {}

    def nn_kl_divergence(self):
        """Returns the sum of the KL divergence of each of the BayesianModules of the model, which are from
            their posterior current distribution of weights relative to a scale-mixtured prior (and simpler) distribution of weights
            Parameters:
                N/a
            Returns torch.tensor with 0 dim.      
        
        """
        return kl_divergence_from_nn(self)
    
    def sample_elbo(self,
                    inputs,
                    labels,
                    sample_nbr = 100,
                    complexity_cost_weight=1):

        """ Samples the ELBO Loss for a batch of data, consisting of inputs and corresponding-by-index labels
                The ELBO Loss consists of the sum of the KL Divergence of the model
                 (explained above, interpreted as a "complexity part" of the loss)
                 with the actual criterion - (loss function) of optimization of our model
                 (the performance part of the loss).
                As we are using variational inference, it takes several (quantified by the parameter sample_nbr) Monte-Carlo
                 samples of the weights in order to gather a better approximation for the loss.
            Parameters:
                inputs: torch.tensor -> the input data to the model
                labels: torch.tensor -> label data for the performance-part of the loss calculation
                        The shape of the labels must match the label-parameter shape of the criterion (one hot encoded or as index, if needed)
                sample_nbr: int -> The number of times of the weight-sampling and predictions done in our Monte-Carlo approach to
                            gather the loss to be .backwarded in the optimization of the model.
        """

        loss = 0
        for _ in range(sample_nbr):
            loss += self._mse_loss(inputs, labels)
            loss += self.nn_kl_divergence().mean() * complexity_cost_weight
        return loss / sample_nbr
    
    def eval_score(  # type: ignore
        self, model_in: torch.Tensor, target: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Computes the squared error for the model over the given input/target.
        When model is not an ensemble, this is equivalent to
        `F.mse_loss(model(model_in, target), reduction="none")`. If the model is ensemble,
        then return is batched over the model dimension.
        This function returns no metadata, so the second output is set to an empty dict.
        Args:
            model_in (tensor): input tensor. The shape must be ``B x Id``, where `B`` and ``Id``
                batch size, and input dimension, respectively.
            target (tensor): target tensor. The shape must be ``B x Od``, where ``B`` and ``Od``
                represent batch size, and output dimension, respectively.
        Returns:
            (tensor): a tensor with the squared error per output dimension, batched over model.
        """
        assert model_in.ndim == 2 and target.ndim == 2
        with torch.no_grad():
            pred = self.forward(model_in, use_propagation=False)
            target = target.repeat((self.num_members, 1, 1))
            return F.mse_loss(pred, target, reduction="none"), {}

    def sample_propagation_indices(
        self, batch_size: int, _rng: torch.Generator
    ) -> torch.Tensor:
        model_len = (
            len(self.elite_models) if self.elite_models is not None else len(self)
        )
        if batch_size % model_len != 0:
            raise ValueError(
                f"To use GaussianMLP's ensemble propagation, the batch size [{batch_size}] must "
                f"be a multiple of the number of models [{model_len}] in the ensemble."
            )
        # rng causes segmentation fault, see https://github.com/pytorch/pytorch/issues/44714
        return torch.randperm(batch_size, device=self.device)

    def set_elite(self, elite_indices: Sequence[int]):
        if len(elite_indices) != self.num_members:
            self.elite_models = list(elite_indices)

    def freeze_model(self):
        """
        Freezes the model by making it predict using only the expected value to their BayesianModules' weights distributions
        """
        self.freeze = True

        for module in self.modules():
            if isinstance(module, (BayesianModule)):
                module.freeze = True
    
    def unfreeze_model(self):
        """
        Unfreezes the model by letting it draw its weights with uncertanity from their correspondent distributions
        """
        self.freeze = False

        for module in self.modules():
            if isinstance(module, (BayesianModule)):
                module.freeze = False
    
    def save(self, save_dir: Union[str, pathlib.Path]):
        """Saves the model to the given directory."""
        model_dict = {
            "state_dict": self.state_dict(),
            "elite_models": self.elite_models,
        }
        torch.save(model_dict, pathlib.Path(save_dir) / self._MODEL_FNAME)

    def load(self, load_dir: Union[str, pathlib.Path]):
        """Loads the model from the given path."""
        model_dict = torch.load(pathlib.Path(load_dir) / self._MODEL_FNAME)
        self.load_state_dict(model_dict["state_dict"])
        self.elite_models = model_dict["elite_models"]



if __name__ == "__main__":
    from blitz.losses.kl_divergence import kl_divergence_from_nn

    input_size = 5
    output_size = 4
    ensemble_size = 3
    hid_size = 10
    batch_size = 3
    bnn = BNN(input_size, output_size, "cpu", num_layers= 2, ensemble_size=ensemble_size, hid_size=hid_size, propagation_method="expectation")
    batch = torch.Tensor([[ 0.9863,  1.6332, -0.9724, -1.5080, -0.6192]])
    target = torch.tensor([ 0.0009, -0.0945, -0.0004,  0.1456])
    batch = torch.randn(batch_size, input_size, requires_grad=True)
    print(batch)
    test_labels = torch.empty(batch_size, dtype = int).random_(output_size)
    test_ouput = bnn.forward(batch)
    print(test_ouput)
    print(test_labels)
    print(F.one_hot(test_labels).float())
    # print(nn.CrossEntropyLoss(test_ouput, test_labels))
    optimizer = torch.optim.Adam(bnn.parameters(), lr = 1e-2)
    # print(list(bnn.parameters()))

    bnn.freeze_model()
    loss = bnn.sample_elbo(batch, F.one_hot(test_labels, num_classes = output_size).float()-1, 10)
    print(bnn.forward(batch))
    loss.backward()
    optimizer.step()
    print(bnn.forward(batch))
