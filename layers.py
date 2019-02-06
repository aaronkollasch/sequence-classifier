import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from functions import clamp, rsample, kl_normal, log_one_plus_exp


class HyperparameterError(ValueError):
    pass


# def _anneal(step, sampler_hyperparams):
#     warm_up = sampler_hyperparams["warm_up"]
#     annealing_type = sampler_hyperparams["annealing_type"]
#     if annealing_type == "linear":
#         return min(step / warm_up, 1.)
#     elif annealing_type == "piecewise_linear":
#         return clamp(
#             torch.tensor(step - warm_up).float().sigmoid().item()
#             * ((step - warm_up) / warm_up),
#             0., 1.
#         )
#     elif annealing_type == "sigmoid":
#         slope = sampler_hyperparams["sigmoid_slope"]
#         return torch.tensor(slope * (step - warm_up)).sigmoid().item()


class BayesianRegularizer(object):
    """Implementation of weight uncertainty
    From https://arxiv.org/abs/1505.05424"""
    def __init__(self, name, rho_type='rho', prior_type='gaussian', prior_params=(0., 1.)):
        self.name = name
        self.rho_type = rho_type  # rho, log_sigma
        self.prior_type = prior_type  # gaussian, mix2gaussians  # TODO implement MoG
        self.prior_params = prior_params

    def rho_to_sigma(self, rho):
        if self.rho_type == 'rho':
            return log_one_plus_exp(rho)
        elif self.rho_type == 'log_sigma':
            return rho.exp()

    def rho_to_log_sigma(self, rho):
        if self.rho_type == 'rho':
            return log_one_plus_exp(rho).log()
        elif self.rho_type == 'log_sigma':
            return rho

    def compute_weight(self, module, stddev=1.):
        mu = getattr(module, self.name + '_mu')
        if stddev == 0.:
            return mu
        sigma = self.rho_to_sigma(getattr(module, self.name + '_rho'))
        return rsample(mu, sigma, stddev=stddev)

    def weight_costs(self, module):
        mu = getattr(module, self.name + '_mu')
        sigma = self.rho_to_sigma(getattr(module, self.name + '_rho'))
        return kl_normal(mu, sigma, *self.prior_params).sum()

    def apply(self, module):
        weight = getattr(module, self.name)
        del module._parameters[self.name]  # remove w from parameter list

        module.register_parameter(self.name + '_mu', Parameter(weight.data))
        module.register_parameter(self.name + '_rho', Parameter(-7 * torch.ones_like(weight).data))
        setattr(module, self.name, self.compute_weight(module))

    def remove(self, module):
        weight = getattr(module, self.name + '_mu')
        delattr(module, self.name)
        del module._parameters[self.name + '_mu']
        del module._parameters[self.name + '_rho']
        module.register_parameter(self.name, Parameter(weight.data))

    def __call__(self, module, stddev=1.):
        setattr(module, self.name, self.compute_weight(module, stddev=stddev))


class GRU(nn.GRU):
    def __init__(self, *args, **kwargs):
        kwargs.pop('hyperparams', None)
        nn.GRU.__init__(self, *args, **kwargs)

    def weight_costs(self):
        return [param.pow(2).sum() for param in self.parameters()]

    def forward(self, x, hx=None, step=0):
        return nn.GRU.forward(self, x, hx)


class BayesianGRU(nn.GRU):
    def __init__(self, *args, **kwargs):
        self.hyperparams = kwargs.pop('hyperparams')
        nn.GRU.__init__(self, *args, **kwargs)

        self.regularizers = {}
        names = [name for name, _ in self.named_parameters()]
        for name in names:
            regularizer = BayesianRegularizer(
                name,
                rho_type=self.hyperparams['regularization']['rho_type'],
                prior_type=self.hyperparams['regularization']['prior_type'],
                prior_params=self.hyperparams['regularization']['prior_params'],
            )
            regularizer.apply(self)
            self.regularizers[name] = regularizer

    def weight_costs(self):
        return [regularizer.weight_costs(self) for regularizer in self.regularizers.values()]

    def _anneal(self, step):
        warm_up = self.hyperparams["sampler_hyperparams"]["warm_up"]
        annealing_type = self.hyperparams["sampler_hyperparams"]["annealing_type"]
        if annealing_type == "linear":
            return min(step / warm_up, 1.)
        elif annealing_type == "piecewise_linear":
            return clamp(torch.tensor(step - warm_up).float().sigmoid().item() * ((step - warm_up) / warm_up))
        elif annealing_type == "sigmoid":
            slope = self.hyperparams["sampler_hyperparams"]["sigmoid_slope"]
            return torch.sigmoid(torch.tensor(slope * (step - warm_up))).item()

    def forward(self, x, hx=None, step=0):
        stddev = self._anneal(step)  # if self.training else 0.
        for regularizer in self.regularizers.values():
            regularizer(self, stddev=stddev)  # resample weights

        output = nn.GRU.forward(self, x, hx)

        for layer_param_names in self._all_weights:
            for name in layer_param_names:
                delattr(self, name)

        return output

    @property
    def _flat_weights(self):  # redefine necessary to prevent nn.GRU from using the wrong weights
        return [p for layerparams in self.all_weights for p in layerparams]

    @property
    def all_weights(self):  # included with _flat_weights for interpretability; may not be necessary
        return [[getattr(self, weight) for weight in weights] for weights in self._all_weights]

