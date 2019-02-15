from collections import OrderedDict
from typing import Union, Dict, Sequence
import warnings
from copy import deepcopy
import itertools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from functions import Nonlinearity, l2_normalize, clamp
from utils import recursive_update
import layers


class RNN(nn.Module):
    """Abstract RNN class."""
    MODEL_TYPE = 'abstract_rnn'
    DEFAULT_DIMS = {
            "batch": 10,
            "alphabet": 21,
            "length": 256
        }
    DEFAULT_PARAMS = {}

    def __init__(self, dims=None, hyperparams=None):
        super(RNN, self).__init__()

        self.dims = self.DEFAULT_DIMS.copy()
        if dims is not None:
            self.dims.update(dims)
        self.dims.setdefault('input', self.dims['alphabet'])

        self.hyperparams: Dict[str, Dict[str, Union[int, bool, float, str, Sequence]]] = deepcopy(self.DEFAULT_PARAMS)
        if hyperparams is not None:
            recursive_update(self.hyperparams, hyperparams)

        self.step = 0
        self.image_summaries = {}

    def forward(self, *args):
        raise NotImplementedError

    def weight_costs(self):
        raise NotImplementedError

    def weight_cost(self):
        return torch.stack(self.weight_costs()).sum()

    def parameter_count(self):
        return sum(param.numel() for param in self.parameters())


class DiscriminativeRNN(RNN):
    MODEL_TYPE = 'discriminative_rnn'
    DEFAULT_PARAMS = {
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

    def __init__(self, dims=None, hyperparams=None):
        super(DiscriminativeRNN, self).__init__(dims=dims, hyperparams=hyperparams)

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
            # # Skip; using built-in optimizer weight_decay instead
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


class GenerativeRNN(RNN):
    """Implementation of generative RNN
    from https://arxiv.org/abs/1703.01898"""
    MODEL_TYPE = 'generative_rnn'
    DEFAULT_PARAMS = {
        'rnn': {
            'hidden_size': 100,
            'num_layers': 2,
            'bidirectional': False,
            'dropout_p': 0.2
        },
        'label_embedding': {
            'features': 1,
            'hidden_size': 20,
        },
        'dense': {
            'num_layers': 1,
            'ordering': ['nonlin', 'norm', 'dropout', 'linear'],
            'hidden_size': 50,
            'nonlinearity': 'relu',
            'normalization': 'none',
            'dropout_p': 0.5,
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

    def __init__(self, dims=None, hyperparams=None):
        super(GenerativeRNN, self).__init__(dims=dims, hyperparams=hyperparams)

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
        if self.hyperparams['rnn']['bidirectional']:
            warnings.warn("Using bidirectional RNN with generative model; violates autoregressive conditions")

        rnn_params = self.hyperparams['rnn']
        emb_params = self.hyperparams['label_embedding']
        dense_params = self.hyperparams['dense']

        bayesian_params = None
        norm = None
        if self.hyperparams['regularization']['bayesian']:
            rnn = layers.BayesianGRU
            linear = layers.BayesianLinear
            if dense_params['normalization'] == 'batch':
                norm = layers.BayesianBatchNorm1d
            bayesian_params = {
                'sampler_hyperparams': self.hyperparams['sampler_hyperparams'],
                'regularization': self.hyperparams['regularization']
            }
        else:
            rnn = layers.GRU
            linear = layers.Linear
            if dense_params['normalization'] == 'batch':
                norm = layers.BatchNorm1d

        # Initialize RNN modules
        self.rnn = rnn(
            input_size=self.dims['input'], hidden_size=rnn_params['hidden_size'],
            num_layers=rnn_params['num_layers'], batch_first=True,
            bidirectional=rnn_params['bidirectional'], dropout=rnn_params['dropout_p'],
            hyperparams=bayesian_params,
        )
        self.hidden = None  # don't save as parameter or buffer
        rnn_output_size = rnn_params['hidden_size'] * (1 + rnn_params['bidirectional'])

        # Initialize label embedding
        self.label_embedding = nn.Linear(emb_params['features'], emb_params['hidden_size'])
        concat_rep_size = rnn_output_size + emb_params['hidden_size']

        # Initialize dense layers
        dense_net = OrderedDict()
        for i in range(1, dense_params['num_layers']+1):
            input_size = output_size = dense_params['hidden_size']
            if i == 1:
                input_size = concat_rep_size
            if i == dense_params['num_layers']:
                output_size = self.dims['alphabet']

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

        self._enable_gradient = 'redb'
        self._count_labels = torch.zeros(emb_params['features'])
        self._label_count = 0

    @property
    def enable_gradient(self):
        return self._enable_gradient

    @enable_gradient.setter
    def enable_gradient(self, value):
        if self._enable_gradient == value:
            return
        self._enable_gradient = value
        for p in self.rnn.parameters():
            p.requires_grad = 'r' in value
            if 'r' not in value:
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()
        for p in self.label_embedding.parameters():
            p.requires_grad = 'e' in value
            if 'e' not in value:
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()
        for name, p in self.dense_net.named_parameters():
            if 'bias' in name:
                continue
            p.requires_grad = 'd' in value
            if 'd' not in value:
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()
        for name, p in self.dense_net.named_parameters():
            if 'bias' not in name:
                continue
            p.requires_grad = 'b' in value
            if 'b' not in value:
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()

    def init_hidden(self, batch_size=1, device='cpu'):
        rnn_params = self.hyperparams['rnn']
        num_layers = rnn_params['num_layers'] * (1 + rnn_params['bidirectional'])
        h0 = torch.zeros(num_layers, batch_size, rnn_params['hidden_size']).to(device)
        # c0 = torch.zeros(rnn_params['num_layers'], batch_size, hidden_size).to(device)  # for use with LSTM model
        return h0

    def p_labels(self):
        """

        :return: tensor(n_labels)
        """
        if self._label_count == 0:
            return None
        else:
            return self._count_labels / self._label_count

    def weight_costs(self):
        return (
            self.rnn.weight_costs() +
            [cost
             for name, module in self.dense_net_modules.items()
             if name.startswith('linear') or name.startswith('norm')
             for cost in module.weight_costs()]
        )

    def forward(self, inputs, input_masks, labels=None):
        """Calculate p(x_t|x_<t, y)
        :param inputs: tensor(batch, length, channels)
        :param input_masks: tensor(batch, length, 1)
        :param labels: tensor(batch, num_labels) or None
        :return: output_logits: tensor(batch, length, channels
        """
        n_batch = inputs.size(0)
        # self.hidden: (num_layers * num_directions, batch, hidden_size)
        self.hidden = self.init_hidden(n_batch, inputs.device)

        x = inputs * input_masks

        # out_states: (batch, seq_len, num_directions * hidden_size)
        out_states, self.hidden = self.rnn(x, self.hidden, step=self.step)

        # embedding: (batch, emb_size)
        if labels is None:
            label_embedding = torch.randn(
                n_batch, self.hyperparams['label_embedding']['hidden_size'],
                dtype=out_states.dtype, device=out_states.device)
            label_embedding = l2_normalize(label_embedding, 1, eps=1.)  # clamp norm ≤ 1
        else:
            label_embedding = self.label_embedding(labels)

        # hiddens: (batch, seq_len, num_directions * hidden_size + emb_size)
        hiddens = torch.cat([
            out_states,
            label_embedding.unsqueeze(1).expand(-1, out_states.size(1), -1)
        ], -1)

        # (batch, seq_len, alphabet_size)
        output_logits = self.dense_net(hiddens)
        return output_logits

    @staticmethod
    def reconstruction_loss(seq_logits, target_seqs, mask):
        seq_reconstruct = F.log_softmax(seq_logits.transpose(1, 2), 1)
        # cross_entropy = F.cross_entropy(seq_logits.transpose(1, 2), target_seqs.argmax(2), reduction='none')
        cross_entropy = F.nll_loss(seq_reconstruct, target_seqs.argmax(2), reduction='none')
        cross_entropy = cross_entropy * mask.squeeze(2)
        ce_loss_per_seq = cross_entropy.sum(1)
        bitperchar_per_seq = ce_loss_per_seq / mask.sum([1, 2])
        ce_loss = ce_loss_per_seq.mean()
        bitperchar = bitperchar_per_seq.mean()
        return {
            'seq_reconstruct': seq_reconstruct,
            'ce_loss': ce_loss,
            'ce_loss_per_seq': ce_loss_per_seq,
            'bitperchar': bitperchar,
            'bitperchar_per_seq': bitperchar_per_seq
        }

    def calculate_loss(
            self, seq_logits, target_seqs, mask, n_eff, labels=None, pos_weight=None,
    ):
        """

        :param seq_logits: (N, L, C)
        :param target_seqs: (N, L, C) as one-hot
        :param mask: (N, L, 1)
        :param n_eff:
        :param labels: (N, n_labels)
        :param pos_weight: (n_labels)
        :return:
        """
        reg_params = self.hyperparams['regularization']

        # cross-entropy
        reconstruction_loss = self.reconstruction_loss(
            seq_logits, target_seqs, mask
        )
        if labels is not None:
            self._count_labels += labels.detach().sum(0)
            self._label_count += labels.size(0)
            if pos_weight is not None:
                weights_per_seq = (labels * pos_weight.unsqueeze(0) + (1-labels)).prod(1)
                loss_per_seq = reconstruction_loss['ce_loss_per_seq'] * weights_per_seq
                loss = loss_per_seq.mean()
        else:
            loss_per_seq = reconstruction_loss['ce_loss_per_seq']
            loss = reconstruction_loss['ce_loss']

        # regularization
        if reg_params['bayesian']:
            loss += self.weight_cost() * reg_params['bayesian_lambda'] / n_eff
        elif reg_params['l2']:
            # # Skip; use built-in optimizer weight_decay instead
            # loss += self.weight_cost() * reg_params['l2_lambda'] / n_eff
            pass

        seq_reconstruct = reconstruction_loss.pop('seq_reconstruct')
        self.image_summaries['SeqReconstruct'] = dict(
            img=seq_reconstruct.transpose(1, 2).unsqueeze(3).detach(), max_outputs=3)
        self.image_summaries['SeqTarget'] = dict(
            img=target_seqs.transpose(1, 2).unsqueeze(3).detach(), max_outputs=3)
        self.image_summaries['SeqDelta'] = dict(
            img=(seq_reconstruct - target_seqs.transpose(1, 2)).transpose(1, 2).unsqueeze(3).detach(), max_outputs=3)

        output = {
            'loss': loss,
            'ce_loss': None,
            'bitperchar': None,
            'loss_per_seq': loss_per_seq,
            'bitperchar_per_seq': None,
            'ce_loss_per_seq': None
        }
        output.update(reconstruction_loss)
        return output

    def predict_all_y(self, inputs, input_masks, outputs):
        """Get p(x|y)p(y) for all possible y

        :return: (n_batch, n_choices), (n_choices, n_features)
        """
        n_batch = inputs.size(0)
        seq_len = inputs.size(1)
        n_features = self.hyperparams['label_embedding']['features']

        self.hidden = self.init_hidden(n_batch, inputs.device)
        x = inputs * input_masks

        # out_states: (batch, seq_len, num_directions * hidden_size)
        out_states, self.hidden = self.rnn(x, self.hidden, step=self.step)

        # y_choices: (n_choices, n_features)
        y_choices = torch.Tensor(
            [i for i in itertools.product([0., 1.], repeat=n_features)],
            device=inputs.device
        )

        # p_labels: (n_features, [p_0, p_1])
        p_labels = self.p_labels()
        p_labels = torch.stack([1-p_labels, p_labels]).transpose(0, 1)

        # p_y_choices: (n_choices)
        p_y_choices = (torch.eye(2, device=inputs.device)[y_choices.long()] * p_labels.unsqueeze(0))
        p_y_choices = p_y_choices.sum(2).clamp_min(1e-6).log().sum(1)

        log_probs = []
        for y, p_y in zip(y_choices, p_y_choices):
            label_embedding = self.label_embedding(y)
            hiddens = torch.cat([
                out_states,
                label_embedding.unsqueeze(0).unsqueeze(1).expand(n_batch, seq_len, -1)
            ], -1)
            output_logits = self.dense_net(hiddens)

            # calculate p(x|y)p(y)
            log_probs.append((output_logits.log_softmax(2) * outputs).sum([1, 2]) + p_y)

        log_probs = torch.stack(log_probs, 1)
        return log_probs, y_choices

    @staticmethod
    def predict_logits(log_probs, y_choices):
        """Estimate logits for p(y|x)
        by subtracting logits for negative choices
        from logits for positive choices for each feature.
        Estimate is inaccurate when n_features > 1.

        :return: (n_batch, n_features)
        """
        return (log_probs.unsqueeze(2) * (2 * y_choices - 1).unsqueeze(0)).sum(1)

    @staticmethod
    def predict(log_probs, y_choices):
        """Get max_y(p(x|y)p(y)

        :return: (n_batch, n_features)
        """
        best_y = log_probs.argmax(1)
        predictions = y_choices[best_y]
        return predictions

    @staticmethod
    def calculate_accuracy(logits, targets, threshold=0.5, reduction='mean'):
        error = 1 - (targets - (logits.sigmoid() > threshold).float()).abs()
        if reduction == 'mean':
            return error.mean()
        else:
            return error
