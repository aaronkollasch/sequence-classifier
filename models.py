from collections import OrderedDict
from typing import Union, Dict, Sequence
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from functions import Nonlinearity
from utils import recursive_update
import layers


class DiscriminativeRNN(nn.Module):
    model_type = 'discriminative_rnn'

    def __init__(self, dims=None, hyperparams=None):
        super(DiscriminativeRNN, self).__init__()
        self.dims = {
            "batch": 10,
            "alphabet": 21,
            "length": 256
        }
        if dims is not None:
            self.dims.update(dims)
        self.dims.setdefault('input', self.dims['alphabet'])

        self.hyperparams: Dict[str, Dict[str, Union[int, bool, float, str, Sequence]]] = {
            'rnn': {
                'hidden_size': 100,
                'num_layers': 2,
                'bidirectional': True,
                'dropout_p': 0.2,
                'use_output': 'global_avg',  # global_avg, final
            },
            'dense': {
                'num_layers': 2,
                'hidden_size': 50,
                'output_features': 1,
                'nonlinearity': 'relu',
                'normalization': 'batch',
                'dropout_p': 0.5,
                'ordering': ['nonlin', 'norm', 'dropout', 'linear'],
            },
            'sampler_hyperparams': {
                'warm_up': 10000,
                'annealing_type': 'linear',
                'anneal_kl': True,
                'anneal_noise': True
            },
            'regularization': {
                'l2': True,
                'l2_lambda': 1.,
                'bayesian': False,  # if True, disables l2 regularization
                'bayesian_lambda': 0.1,
                'rho_type': 'log_sigma',  # lin_sigma, log_sigma
                'prior_type': 'gaussian',  # gaussian, scale_mix_gaussian
                'prior_params': None,
            },
            'optimization': {
                'optimizer': 'Adam',
                'lr': 0.001,
                'weight_decay': 0,  # trainer will divide by n_eff before using
                'clip': 10.0,
            }
        }
        if hyperparams is not None:
            recursive_update(self.hyperparams, hyperparams)

        if self.hyperparams['regularization']['bayesian']:
            self.hyperparams['regularization']['l2'] = False
        elif self.hyperparams['regularization']['l2']:
            # torch built-in weight decay is more efficient than manual calculation
            self.hyperparams['optimization']['weight_decay'] = self.hyperparams['regularization']['l2_lambda']
        if self.hyperparams['regularization']['prior_params'] is None:
            if self.hyperparams['regularization']['prior_type'] == 'gaussian':
                self.hyperparams['regularization']['prior_params'] = (0., 1.)
            elif self.hyperparams['regularization']['prior_type'] == 'scale_mix_gaussian':
                self.hyperparams['regularization']['prior_params'] = (0.1, 0., 1., 0., 0.001)
        if self.hyperparams['regularization']['bayesian'] and (
            self.hyperparams['rnn']['dropout_p'] > 0 or
            self.hyperparams['dense']['dropout_p'] > 0
        ):
            warnings.warn("Using both weight uncertainty and dropout")

        # Initialize RNN modules
        rnn_params = self.hyperparams['rnn']
        if self.hyperparams['regularization']['bayesian']:
            rnn = layers.BayesianGRU
            bayesian_params = {
                'sampler_hyperparams': self.hyperparams['sampler_hyperparams'],
                'regularization': self.hyperparams['regularization']
            }
        else:
            rnn = layers.GRU
            bayesian_params = None

        self.rnn = rnn(
            input_size=self.dims['input'], hidden_size=rnn_params['hidden_size'],
            num_layers=rnn_params['num_layers'], batch_first=True,
            bidirectional=rnn_params['bidirectional'], dropout=rnn_params['dropout_p'],
            hyperparams=bayesian_params,
        )
        self.hidden = None  # don't save as parameter or buffer
        rnn_output_size = rnn_params['hidden_size'] * (1 + rnn_params['bidirectional'])

        # Initialize dense layers
        dense_params = self.hyperparams['dense']
        dense_net = OrderedDict()
        norm = None
        if self.hyperparams['regularization']['bayesian']:
            linear = layers.BayesianLinear
            if dense_params['normalization'] == 'batch':
                norm = layers.BayesianBatchNorm1d
        else:
            linear = layers.Linear
            if dense_params['normalization'] == 'batch':
                norm = layers.BatchNorm1d

        for i in range(1, dense_params['num_layers']+1):
            input_size = output_size = dense_params['hidden_size']
            if i == 1:
                input_size = rnn_output_size
            if i == dense_params['num_layers']:
                output_size = dense_params['output_features']

            for layer_type in dense_params['ordering']:
                if layer_type == 'norm' and norm is not None:
                    dense_net[f'norm_{i}'] = norm(input_size, hyperparams=bayesian_params)
                elif layer_type == 'nonlin':
                    dense_net[f'nonlin_{i}'] = Nonlinearity(dense_params['nonlinearity'])
                elif layer_type == 'dropout':
                    dense_net[f'dropout_{i}'] = nn.Dropout(dense_params['dropout_p'])
                elif layer_type == 'linear':
                    dense_net[f'linear_{i}'] = linear(input_size, output_size, hyperparams=bayesian_params)
        self.dense_net_modules = dense_net
        self.dense_net = nn.Sequential(dense_net)

        self.step = 0
        self.image_summaries = {}

    def init_hidden(self, batch_size=1, device='cpu'):
        rnn_params = self.hyperparams['rnn']
        num_layers = rnn_params['num_layers'] * (1 + rnn_params['bidirectional'])
        h0 = torch.zeros(num_layers, batch_size, rnn_params['hidden_size']).to(device)
        # c0 = torch.zeros(rnn_params['num_layers'], batch_size, hidden_size).to(device)  # for use with LSTM model
        return h0

    def weight_costs(self):
        return (
            self.rnn.weight_costs() +
            [cost
             for name, module in self.dense_net_modules.items()
             if name.startswith('linear') or name.startswith('norm')
             for cost in module.weight_costs()]
        )

    def weight_cost(self):
        return torch.stack(self.weight_costs()).sum()

    def parameter_count(self):
        return sum(param.numel() for param in self.parameters())

    def forward(self, inputs, input_masks):
        """
        :param inputs: tensor(batch, length, channels)
        :param input_masks: tensor(batch, length, 1)
        :return:
        """
        n_batch = inputs.size(0)
        # self.hidden: (num_layers * num_directions, batch, hidden_size)
        self.hidden = self.init_hidden(n_batch, inputs.device)

        x = inputs * input_masks

        # out_states: (batch, seq_len, num_directions * hidden_size)
        out_states, self.hidden = self.rnn(x, self.hidden, step=self.step)

        # hiddens: (batch, num_directions * hidden_size)
        rnn_params = self.hyperparams['rnn']
        if rnn_params['use_output'] == 'global_avg':
            # average over masked output states
            hiddens = (out_states * input_masks).sum(1) / input_masks.sum(1)
        else:
            # get top layer's hidden state
            num_layers = rnn_params['num_layers']
            hidden_size = rnn_params['hidden_size']
            num_directions = 1 + rnn_params['bidirectional']
            hiddens = self.hidden.view(num_layers, num_directions, n_batch, hidden_size)[-1]
            hiddens = hiddens.permute(1, 0, 2).contiguous().view(n_batch, num_directions * hidden_size)

        # (batch, output_features)
        features_logits = self.dense_net(hiddens)
        return features_logits

    def calculate_loss(self, logits, targets, n_eff=1., pos_weight=None, reduction='mean'):
        reg_params = self.hyperparams['regularization']
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight, reduction=reduction)
        loss = ce_loss

        # regularization
        if reg_params['bayesian']:
            loss += self.weight_cost() * reg_params['bayesian_lambda'] / n_eff
        elif reg_params['l2']:
            # # Skip; use built-in optimizer weight_decay instead
            # loss += self.weight_cost() * reg_params['l2_lambda'] / n_eff
            pass

        return {
            'loss': loss,
            'ce_loss': ce_loss
        }

    @staticmethod
    def predict(logits, threshold=0.5):
        return (logits.sigmoid() > threshold).float()

    @staticmethod
    def calculate_accuracy(logits, targets, threshold=0.5, reduction='mean'):
        error = 1 - (targets - (logits.sigmoid() > threshold).float()).abs()
        if reduction == 'mean':
            return error.mean()
        else:
            return error
