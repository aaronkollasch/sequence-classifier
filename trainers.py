import os
import time

import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data

from data_loaders import GeneratorDataLoader
from model_logging import Logger
from utils import temp_seed


class ClassifierTrainer:
    default_params = {
            'optimizer': 'Adam',
            'lr': 0.001,
            'weight_decay': 0,
            'clip': 10.0,
            'snapshot_path': None,
            'snapshot_name': 'snapshot',
            'snapshot_interval': 1000,
        }

    def __init__(
            self,
            model,
            data_loader,
            optimizer=None,
            params=None,
            lr=None,
            weight_decay=None,
            gradient_clipping=None,
            logger=Logger(),
            snapshot_path=None,
            snapshot_name=None,
            snapshot_interval=None,
            snapshot_exec_template=None,
            device=torch.device('cpu')
    ):
        self.params = self.default_params.copy()
        self.params.update(model.hyperparams['optimization'])
        if params is not None:
            self.params.update(params)
        if optimizer is not None:
            self.params['optimizer'] = optimizer
        if lr is not None:
            self.params['lr'] = lr
        if weight_decay is not None:
            self.params['weight_decay'] = weight_decay
        if gradient_clipping is not None:
            self.params['clip'] = gradient_clipping
        if snapshot_path is not None:
            self.params['snapshot_path'] = snapshot_path
        if snapshot_name is not None:
            self.params['snapshot_name'] = snapshot_name
        if snapshot_interval is not None:
            self.params['snapshot_interval'] = snapshot_interval
        if snapshot_exec_template is not None:
            self.params['snapshot_exec_template'] = snapshot_exec_template
        if self.params['weight_decay'] > 0:
            self.params['weight_decay'] = self.params['weight_decay'] / data_loader.dataset.n_eff

        self.model = model
        self.loader = data_loader

        self.optimizer_type = getattr(optim, self.params['optimizer'])
        self.logger = logger
        self.logger.trainer = self
        self.device = device

        self.optimizer = self.optimizer_type(
            params=self.model.parameters(),
            lr=self.params['lr'],
            weight_decay=self.params['weight_decay'])

    def train(self, steps=1e8):
        self.model.train()

        losses, accuracies, true_outputs, logits, rocs = self.validate()
        print(f"validation losses: {', '.join(['{:6.4f}'.format(loss) for loss in losses])}", flush=True)
        print(f"validation accuracies: {', '.join(['{:6.2f}%'.format(acc * 100) for acc in accuracies])}", flush=True)
        print(f"validation true values: {', '.join(['{:6.4f}'.format(val) for val in true_outputs])}", flush=True)
        print(f"validation average logits: {', '.join(['{:6.4f}'.format(logit) for logit in logits])}", flush=True)
        print(f"validation AUCs: {', '.join(['{:6.4f}'.format(roc) for roc in rocs])}", flush=True)

        pos_weights = self.loader.dataset.comparison_pos_weights.to(self.device)
        data_iter = iter(self.loader)
        n_eff = self.loader.dataset.n_eff

        # print('    step  step-t load-t    loss      accuracy', flush=True)
        for step in range(int(self.model.step) + 1, int(steps) + 1):
            self.model.step = step
            # start = time.time()

            batch = next(data_iter)
            for key in batch.keys():
                batch[key] = batch[key].to(self.device, non_blocking=True)
            # data_load_time = time.time()-start

            output_logits = self.model(batch['input'], batch['mask'])
            losses = self.model.calculate_loss(output_logits, batch['output'], n_eff=n_eff, pos_weight=pos_weights)
            accuracy = self.model.calculate_accuracy(output_logits, batch['output'])
            del self.model.hidden

            self.optimizer.zero_grad()
            losses['loss'].backward()

            if self.params['clip'] is not None:
                total_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.params['clip'])
            else:
                total_norm = 0.0

            for key in losses:
                losses[key] = losses[key].detach()
            losses.update({'accuracy': accuracy.detach(), 'grad_norm': total_norm})

            self.optimizer.step()

            if step % self.params['snapshot_interval'] == 0:
                if self.params['snapshot_path'] is not None:
                    self.save_state()

            self.logger.log(step, losses, total_norm)
            # print("{: 8d} {:6.3f} {:5.4f} {:11.6f} {:11.6f}".format(
            #     step, time.time()-start, data_load_time, loss.detach(), accuracy.detach()), flush=True)

    def validate(self):
        self.model.eval()
        self.loader.dataset.test()
        self.loader.dataset.unlimited_epoch = False

        with torch.no_grad():
            loader = GeneratorDataLoader(self.loader.dataset, num_workers=self.loader.num_workers)
            pos_weights = self.loader.dataset.comparison_pos_weights.to(self.device)
            n_eff = len(self.loader.dataset.cdr_seqs_train)

            true_outputs = []
            logits = []
            losses = []
            accuracies = []
            for batch in loader:
                for key in batch.keys():
                    batch[key] = batch[key].to(self.device, non_blocking=True)
                output_logits = self.model(batch['input'], batch['mask'])

                loss_dict = self.model.calculate_loss(
                    output_logits, batch['output'], n_eff=n_eff, pos_weight=pos_weights, reduction='none')
                error = self.model.calculate_accuracy(
                    output_logits, batch['output'], reduction='none')

                true_outputs.append(batch['output'])
                logits.append(output_logits)
                losses.append(loss_dict['loss'])
                accuracies.append(error)

            true_outputs = torch.cat(true_outputs, 0).cpu().numpy()
            logits = torch.cat(logits, 0).cpu().numpy()
            roc_scores = roc_auc_score(true_outputs, logits, average=None)
            if isinstance(roc_scores, np.ndarray):
                roc_scores = roc_scores.tolist()
            else:
                roc_scores = [roc_scores]

            true_outputs = true_outputs.mean(0).tolist()
            logits = logits.mean(0).tolist()
            losses = torch.cat(losses, 0).mean(0).tolist()
            accuracies = torch.cat(accuracies, 0).mean(0).tolist()

        self.model.train()
        self.loader.dataset.train()
        self.loader.dataset.unlimited_epoch = True
        return losses, accuracies, true_outputs, logits, roc_scores

    def test(self, data_loader, model_eval=True, num_samples=1):  # TODO implement
        if model_eval:
            self.model.eval()

        print('    step  step-t  CE-loss     bit-per-char', flush=True)
        for i_iter in range(num_samples):  # TODO implement sampling
            output = {
                'name': [],
                'mean': [],
                'bitperchar': [],
                'sequence': []
            }

            for i_batch, batch in enumerate(data_loader):
                start = time.time()
                for key in batch.keys():
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device, non_blocking=True)

                with torch.no_grad():
                    output_logits = self.model(batch['input'], batch['mask'])
                    pred = self.model.predict(output_logits)

                    ce_loss = losses['ce_loss_per_seq']
                    if self.run_fr:
                        ce_loss_mean = ce_loss.mean(0)
                    else:
                        ce_loss_mean = ce_loss
                    ce_loss_per_char = ce_loss_mean / batch['prot_mask_decoder'].sum([1, 2, 3])

                output['name'].extend(batch['names'])
                output['sequence'].extend(batch['sequences'])
                output['mean'].extend(ce_loss_mean.numpy())
                output['bitperchar'].extend(ce_loss_per_char.numpy())

                print("{: 8d} {:6.3f} {:11.6f} {:11.6f}".format(
                    i_batch, time.time()-start, ce_loss_mean.mean(), ce_loss_per_char.mean()),
                    flush=True)

        self.model.train()
        return output

    def save_state(self, last_batch=None):
        snapshot = f"{self.params['snapshot_path']}/{self.params['snapshot_name']}_{self.model.step}.pth"
        revive_exec = f"{self.params['snapshot_path']}/revive_executable/{self.params['snapshot_name']}.sh"
        torch.save(
            {
                'step': self.model.step,
                'model_type': self.model.model_type,
                'model_state_dict': self.model.state_dict(),
                'model_dims': self.model.dims,
                'model_hyperparams': self.model.hyperparams,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'train_params': self.params,
                'last_batch': last_batch
            },
            snapshot
        )
        with open(revive_exec, "w") as f:
            snapshot_exec = self.params['snapshot_exec_template'].format(
                restore=os.path.abspath(snapshot)
            )
            f.write(snapshot_exec)

    def load_state(self, checkpoint, map_location=None):
        if not isinstance(checkpoint, dict):
            checkpoint = torch.load(checkpoint, map_location=map_location)
        if self.model.model_type != checkpoint['model_type']:
            print("Warning: model type mismatch: loaded type {} for model type {}".format(
                checkpoint['model_type'], self.model.model_type
            ))
        if self.model.hyperparams != checkpoint['model_hyperparams']:
            print("Warning: model hyperparameter mismatch")
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.model.step = checkpoint['step']
        self.params.update(checkpoint['train_params'])
