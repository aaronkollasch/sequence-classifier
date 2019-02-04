from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from functions import Nonlinearity
from utils import recursive_update


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

        self.hyperparams = {
            'rnn': {
                'hidden_size': 100,
                'num_layers': 2,
                'bidirectional': True,
                'dropout_p': 0.2,
                'use_output': 'global_avg',  # global_avg, final
                'final_dropout_p': 0.2,
            },
            'dense': {
                'num_layers': 2,
                'hidden_size': 50,
                'output_features': 1,
                'nonlinearity': 'elu',
                'normalization': 'batch',
                'dropout_p': 0.5,
            }
        }
        if hyperparams is not None:
            recursive_update(self.hyperparams, hyperparams)

        rnn_params = self.hyperparams['rnn']
        # self.W_embedding = nn.Linear(self.dims['alphabet'], rnn_params['hidden_size'])
        self.rnn = nn.GRU(
            input_size=self.dims['alphabet'], hidden_size=rnn_params['hidden_size'],
            num_layers=rnn_params['num_layers'], batch_first=True,
            bidirectional=rnn_params['bidirectional'], dropout=rnn_params['dropout_p'])
        self.hidden = None  # don't save as parameter or buffer
        self.rnn_final_dropout = nn.Dropout(rnn_params['final_dropout_p'])
        rnn_output_size = rnn_params['hidden_size'] * (1 + rnn_params['bidirectional'])

        dense_params = self.hyperparams['dense']
        dense_net = OrderedDict()
        for i in range(1, dense_params['num_layers']+1):
            input_size = output_size = dense_params['hidden_size']
            if i == 1:
                input_size = rnn_output_size
            if i == dense_params['num_layers']:
                output_size = dense_params['output_features']

            if dense_params['normalization'] == 'batch':
                dense_net[f'norm_{i}'] = nn.BatchNorm1d(input_size)
            dense_net[f'nonlin_{i}'] = Nonlinearity(dense_params['nonlinearity'])
            dense_net[f'dropout_{i}'] = nn.Dropout(dense_params['dropout_p'])
            dense_net[f'linear_{i}'] = nn.Linear(input_size, output_size)
        self.dense_net = nn.Sequential(dense_net)

        self.step = 0
        self.image_summaries = {}

    def init_hidden(self, batch_size=1, device='cpu'):
        rnn_params = self.hyperparams['rnn']
        num_layers = rnn_params['num_layers'] * (1 + rnn_params['bidirectional'])
        h0 = torch.zeros(num_layers, batch_size, rnn_params['hidden_size']).to(device)
        # c0 = torch.zeros(rnn_params['num_layers'], batch_size, hidden_size).to(device)  # for use with LSTM model
        return h0

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
        out_states, self.hidden = self.rnn(x, self.hidden)

        # hiddens: (batch, num_directions * hidden_size)
        rnn_params = self.hyperparams['rnn']
        if rnn_params['use_output'] == 'global_avg':
            # average over output states
            hiddens = (out_states * input_masks).sum(1) / input_masks.sum(1)
        else:
            # get top layer's hidden state
            num_layers = rnn_params['num_layers']
            hidden_size = rnn_params['hidden_size']
            num_directions = 1 + rnn_params['bidirectional']
            hiddens = self.hidden.view(num_layers, num_directions, n_batch, hidden_size)[-1]
            hiddens = hiddens.permute(1, 0, 2).contiguous().view(n_batch, num_directions * hidden_size)
        hiddens = self.rnn_final_dropout(hiddens)

        # (batch, output_features)
        features_logits = self.dense_net(hiddens)
        return features_logits

    @staticmethod
    def calculate_loss(logits, targets, pos_weight=None, reduction='mean'):
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight, reduction=reduction)

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
