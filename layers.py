import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.distributions as dist

from functions import anneal, rsample, kl_normal, mle_mixture_gaussians


class HyperparameterError(ValueError):
    pass


class GaussianWeightUncertainty(object):
    """Implementation of weight uncertainty
    From https://arxiv.org/abs/1505.05424
    See also https://www.nitarshan.com/bayes-by-backprop/"""
    def __init__(self, name, hyperparams):
        self.name = name
        self.rho_type = hyperparams['rho_type']  # lin_sigma, log_sigma
        self.prior_type = hyperparams['prior_type']  # gaussian, scale_mix_gaussian
        self.prior_params = hyperparams['prior_params']

    def rho_to_sigma(self, rho):
        if self.rho_type == 'lin_sigma':
            return F.softplus(rho)
        elif self.rho_type == 'log_sigma':
            return rho.exp()

    def sample_weight(self, module, stddev=1.):
        mu = getattr(module, self.name + '_mu')
        if not module.training or stddev == 0.:
            return mu
        sigma = self.rho_to_sigma(getattr(module, self.name + '_rho'))
        return rsample(mu, sigma, stddev=stddev)

    def weight_costs(self, module):
        mu = getattr(module, self.name + '_mu')
        sigma = self.rho_to_sigma(getattr(module, self.name + '_rho'))
        if self.prior_type == 'gaussian':
            return kl_normal(mu, sigma, *self.prior_params)
        elif self.prior_type == 'scale_mix_gaussian':
            # No analytical solution for KL divergence,
            # so use current sample from variational posterior instead.
            if module.training:
                weight = getattr(module, self.name)
            else:
                weight = mu
            log_prior = mle_mixture_gaussians(weight, *self.prior_params)
            log_variational_posterior = dist.Normal(mu, sigma).log_prob(weight)
            return log_variational_posterior - log_prior

    def weight_cost(self, module):
        return self.weight_costs(module).sum()

    def apply(self, module):
        weight = getattr(module, self.name)
        del module._parameters[self.name]  # remove w from parameter list

        module.register_parameter(self.name + '_mu', Parameter(weight.data))
        module.register_parameter(self.name + '_rho', Parameter(-7 * torch.ones_like(weight).data))
        object.__setattr__(module, self.name, self.sample_weight(module))

    def remove(self, module):
        weight = getattr(module, self.name + '_mu')
        delattr(module, self.name)
        del module._parameters[self.name + '_mu']
        del module._parameters[self.name + '_rho']
        module.register_parameter(self.name, Parameter(weight.data))

    def __call__(self, module, stddev=1.):
        object.__setattr__(module, self.name, self.sample_weight(module, stddev=stddev))


class Linear(nn.Linear):
    def __init__(self, *args, **kwargs):
        kwargs.pop('hyperparams', None)
        nn.Linear.__init__(self, *args, **kwargs)

    def weight_costs(self):
        return [self.weight.pow(2).sum(), self.bias.pow(2).sum()]

    def forward(self, x, step=0):
        return nn.Linear.forward(self, x)


class BayesianLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        self.hyperparams = kwargs.pop('hyperparams')
        nn.Linear.__init__(self, *args, **kwargs)

        self.regularizers = {}
        names = [name for name, _ in self.named_parameters()]
        for name in names:
            regularizer = GaussianWeightUncertainty(name, self.hyperparams['regularization'])
            regularizer.apply(self)
            self.regularizers[name] = regularizer

    def weight_costs(self):
        return [regularizer.weight_cost(self) for regularizer in self.regularizers.values()]

    def forward(self, x, step=0):
        stddev = anneal(step, self.hyperparams['sampler_hyperparams'])
        for regularizer in self.regularizers.values():
            regularizer(self, stddev=stddev)

        output = nn.Linear.forward(self, x)
        return output


class BatchNorm1d(nn.BatchNorm1d):
    def __init__(self, *args, **kwargs):
        kwargs.pop('hyperparams', None)
        nn.BatchNorm1d.__init__(self, *args, **kwargs)

    def weight_costs(self):
        return [self.weight.pow(2).sum(), self.bias.pow(2).sum()]

    def forward(self, x, step=0):
        return nn.BatchNorm1d.forward(self, x)


class BayesianBatchNorm1d(nn.BatchNorm1d):
    def __init__(self, *args, **kwargs):
        self.hyperparams = kwargs.pop('hyperparams')
        nn.BatchNorm1d.__init__(self, *args, **kwargs)

        self.regularizers = {}
        names = [name for name, _ in self.named_parameters()]
        for name in names:
            regularizer = GaussianWeightUncertainty(name, self.hyperparams['regularization'])
            regularizer.apply(self)
            self.regularizers[name] = regularizer

    def weight_costs(self):
        return [regularizer.weight_cost(self) for regularizer in self.regularizers.values()]

    def forward(self, x, step=0):
        stddev = anneal(step, self.hyperparams['sampler_hyperparams'])
        for regularizer in self.regularizers.values():
            regularizer(self, stddev=stddev)

        output = nn.BatchNorm1d.forward(self, x)
        return output


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
            regularizer = GaussianWeightUncertainty(name, self.hyperparams['regularization'])
            regularizer.apply(self)
            self.regularizers[name] = regularizer

    def weight_costs(self):
        return [regularizer.weight_cost(self) for regularizer in self.regularizers.values()]

    def forward(self, x, hx=None, step=0):
        stddev = anneal(step, self.hyperparams['sampler_hyperparams'])
        for regularizer in self.regularizers.values():
            regularizer(self, stddev=stddev)

        output = nn.GRU.forward(self, x, hx)
        return output

    @property
    def _flat_weights(self):  # redefine necessary to prevent nn.GRU from using the wrong weights
        return [p for layerparams in self.all_weights for p in layerparams]

    @property
    def all_weights(self):  # included with _flat_weights for interpretability, but may not be necessary
        return [[getattr(self, weight) for weight in weights] for weights in self._all_weights]

