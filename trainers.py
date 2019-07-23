import os
import time

import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

from data_loaders import GeneratorDataLoader, IPISingleDataset, IPIMultiDataset, TrainValTestDataset
from model_logging import Logger
from utils import enter_local_rng_state, exit_local_rng_state


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

        validation = self.validate()
        if validation is not None:
            losses, accuracies, true_outputs, logits, rocs = validation
            print(f"val  losses: {', '.join(['{:6.4f}'.format(loss) for loss in losses])}, "
                  f"accuracies: {', '.join(['{:6.2f}%'.format(acc * 100) for acc in accuracies])}, "
                  f"logits: {', '.join(['{:6.4f}'.format(logit) for logit in logits])}, "
                  f"AUCs: {', '.join(['{:6.4f}'.format(roc) for roc in rocs])}",
                  flush=True)

        testing = self.test(num_samples=1)
        if testing is not None:
            losses, accuracies, true_outputs, logits, rocs = testing
            print(f"test losses: {', '.join(['{:6.4f}'.format(loss) for loss in losses])}, "
                  f"accuracies: {', '.join(['{:6.2f}%'.format(acc * 100) for acc in accuracies])}, "
                  f"logits: {', '.join(['{:6.4f}'.format(logit) for logit in logits])}, "
                  f"AUCs: {', '.join(['{:6.4f}'.format(roc) for roc in rocs])}",
                  flush=True)

        pos_weights = self.loader.dataset.comparison_pos_weights.to(self.device)
        data_iter = iter(self.loader)
        n_eff = self.loader.dataset.n_eff

        # print('    step  step-t load-t    loss      accuracy', flush=True)
        for step in range(int(self.model.step) + 1, int(steps) + 1):
            self.model.step = step
            # start = time.time()

            batch = next(data_iter)
            for key in batch.keys():
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device, non_blocking=True)
            # data_load_time = time.time()-start

            output_logits = self.model(batch['input'], batch['mask'])
            losses = self.model.calculate_loss(output_logits, batch['label'], n_eff=n_eff, pos_weight=pos_weights)
            accuracy = self.model.calculate_accuracy(output_logits, batch['label'])
            if hasattr(self.model, 'hidden'):
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

            if step in [1000, 2500, 5000, 10000, 25000, 50000, 100000, 250000, 500000, 1000000, 2500000] or \
                    step % self.params['snapshot_interval'] == 0:
                if self.params['snapshot_path'] is not None:
                    self.save_state()

            self.logger.log(step, losses, total_norm)
            # print("{: 8d} {:6.3f} {:5.4f} {:11.6f} {:11.6f}".format(
            #     step, time.time()-start, data_load_time, loss.detach(), accuracy.detach()), flush=True)

    def validate(self, r_seed=42):
        self.loader.dataset.val()
        try:
            n_eff = self.loader.dataset.n_eff
        except ValueError:
            n_eff = 0
        if n_eff == 0:
            self.loader.dataset.train()
            return None

        self.model.eval()
        self.loader.dataset.unlimited_epoch = False
        prev_state = enter_local_rng_state(r_seed)

        with torch.no_grad():
            loader = GeneratorDataLoader(self.loader.dataset, num_workers=self.loader.num_workers)
            pos_weights = self.loader.dataset.comparison_pos_weights.to(self.device)

            true_outputs = []
            logits = []
            losses = []
            accuracies = []
            for batch in loader:
                for key in batch.keys():
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device, non_blocking=True)
                output_logits = self.model(batch['input'], batch['mask'])

                loss_dict = self.model.calculate_loss(
                    output_logits, batch['label'], n_eff=n_eff, pos_weight=pos_weights, reduction='none')
                error = self.model.calculate_accuracy(
                    output_logits, batch['label'], reduction='none')

                true_outputs.append(batch['label'])
                logits.append(output_logits)
                losses.append(loss_dict['ce_loss'])
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
        exit_local_rng_state(prev_state)
        return losses, accuracies, true_outputs, logits, roc_scores

    def test(self, model_eval=True, num_samples=1, r_seed=42):
        if isinstance(self.loader.dataset, TrainValTestDataset):
            self.loader.dataset.test()

        try:
            n_eff = self.loader.dataset.n_eff
        except ValueError:
            n_eff = 0
        if n_eff == 0:
            self.loader.dataset.train()
            return None

        if model_eval:
            self.model.eval()
        prev_state = enter_local_rng_state(r_seed)
        self.loader.dataset.unlimited_epoch = False
        loader = GeneratorDataLoader(self.loader.dataset, num_workers=self.loader.num_workers)
        pos_weights = self.loader.dataset.comparison_pos_weights.to(self.device)

        true_outputs = []
        sequences = []
        logits = []
        for i_iter in range(num_samples):
            true_outputs = []
            logits_i = []
            sequences = []

            for i_batch, batch in enumerate(loader):
                for key in batch.keys():
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device, non_blocking=True)

                with torch.no_grad():
                    output_logits = self.model(batch['input'], batch['mask'])

                true_outputs.append(batch['label'])
                sequences.extend(batch['sequences'])
                logits_i.append(output_logits)

            true_outputs = torch.cat(true_outputs, 0)
            logits.append(torch.cat(logits_i, 0))

        logits = torch.stack(logits, 0).mean(0)

        loss_dict = self.model.calculate_loss(
            logits, true_outputs, n_eff=n_eff, pos_weight=pos_weights, reduction='none')
        error = self.model.calculate_accuracy(
            logits, true_outputs, reduction='none')

        true_outputs = true_outputs.cpu().numpy()
        logits = logits.cpu().numpy()
        roc_scores = roc_auc_score(true_outputs, logits, average=None)
        if isinstance(roc_scores, np.ndarray):
            roc_scores = roc_scores.tolist()
        else:
            roc_scores = [roc_scores]

        # TODO return output per test sequence as well
        true_outputs = true_outputs.mean(0).tolist()
        logits = logits.mean(0).tolist()
        losses = loss_dict['ce_loss'].mean(0).cpu().tolist()
        accuracies = error.mean(0).cpu().tolist()

        self.model.train()
        if isinstance(self.loader.dataset, TrainValTestDataset):
            self.loader.dataset.train()
        self.loader.dataset.unlimited_epoch = True
        exit_local_rng_state(prev_state)
        return losses, accuracies, true_outputs, logits, roc_scores

    def save_state(self, last_batch=None):
        snapshot = f"{self.params['snapshot_path']}/{self.params['snapshot_name']}/{self.model.step}.pth"
        revive_exec = f"{self.params['snapshot_path']}/revive_executable/{self.params['snapshot_name']}.sh"
        if not os.path.exists(os.path.dirname(snapshot)):
            os.makedirs(os.path.dirname(snapshot), exist_ok=True)
        torch.save(
            {
                'step': self.model.step,
                'model_type': self.model.MODEL_TYPE,
                'model_state_dict': self.model.state_dict(),
                'model_dims': self.model.dims,
                'model_hyperparams': self.model.hyperparams,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'train_params': self.params,
                'dataset_params': self.loader.dataset.params,
                'last_batch': last_batch
            },
            snapshot
        )
        if 'snapshot_exec_template' in self.params:
            if not os.path.exists(os.path.dirname(revive_exec)):
                os.makedirs(os.path.dirname(revive_exec), exist_ok=True)
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


class GenerativeClassifierTrainer:
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
        start_step = self.model.step

        validation = self.validate()
        if validation is not None:
            losses, accuracies, true_outputs, logits, rocs = validation
            print(f"val  losses: {', '.join(['{:6.4f}'.format(loss) for loss in losses])}, "
                  f"accuracies: {', '.join(['{:6.2f}%'.format(acc * 100) for acc in accuracies])}, "
                  f"logits: {', '.join(['{:6.4f}'.format(logit) for logit in logits])}, "
                  f"AUCs: {', '.join(['{:6.4f}'.format(roc) for roc in rocs])}",
                  flush=True)

        testing = self.test(num_samples=1)
        if testing is not None:
            losses, accuracies, true_outputs, logits, rocs = testing
            print(f"test losses: {', '.join(['{:6.4f}'.format(loss) for loss in losses])}, "
                  f"accuracies: {', '.join(['{:6.2f}%'.format(acc * 100) for acc in accuracies])}, "
                  f"logits: {', '.join(['{:6.4f}'.format(logit) for logit in logits])}, "
                  f"AUCs: {', '.join(['{:6.4f}'.format(roc) for roc in rocs])}",
                  flush=True)

        try:
            pos_weights = self.loader.dataset.comparison_pos_weights.to(self.device)
        except AttributeError:
            pos_weights = None
        data_iter = iter(self.loader)
        n_eff = self.loader.dataset.n_eff

        # print('    step  step-t load-t   loss       CE-loss    bitperchar', flush=True)
        for step in range(int(self.model.step) + 1, int(steps) + 1):
            self.model.step = step
            # start = time.time()

            batch = next(data_iter)
            for key in batch.keys():
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device, non_blocking=True)
            # data_load_time = time.time()-start

            output_logits = self.model(batch['input'], batch['mask'], batch.get('label', None))
            losses = self.model.calculate_loss(output_logits, batch['decoder_output'], batch['mask'], n_eff=n_eff,
                                               labels=batch.get('label', None), pos_weight=pos_weights)
            del self.model.hidden

            self.optimizer.zero_grad()
            losses['loss'].backward()

            if self.params['clip'] is not None:
                total_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.params['clip'])
            else:
                total_norm = 0.0

            for key in losses:
                losses[key] = losses[key].detach()
            losses.update({'grad_norm': total_norm})

            self.optimizer.step()

            if (step-start_step) % self.params['snapshot_interval'] == 0:
                if self.params['snapshot_path'] is not None:
                    self.save_state()

            self.logger.log(step, losses, total_norm)
            # print("{: 8d} {:6.3f} {:5.4f} {:11.6f} {:11.6f} {:11.8f}".format(
            #     step, time.time() - start, data_load_time,
            #     loss.detach(), ce_loss.detach(), bitperchar.detach()), flush=True)

    def validate(self):
        if isinstance(self.loader.dataset, IPISingleDataset) or isinstance(self.loader.dataset, IPIMultiDataset):
            pass
        else:
            return None
        self.model.eval()
        self.loader.dataset.val()
        self.loader.dataset.unlimited_epoch = False

        with torch.no_grad():
            loader = GeneratorDataLoader(self.loader.dataset, num_workers=self.loader.num_workers)
            pos_weights = self.loader.dataset.comparison_pos_weights.to(self.device)

            true_outputs = []
            logits = []
            losses = []
            accuracies = []
            for batch in loader:
                for key in batch.keys():
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device, non_blocking=True)

                log_probs, y_choices = self.model.predict_all_y(batch['input'], batch['mask'], batch['decoder_output'])
                output_logits = self.model.predict_logits(log_probs, y_choices)

                error = self.model.calculate_accuracy(
                    output_logits, batch['label'], reduction='none')
                ce_loss = F.binary_cross_entropy_with_logits(
                    output_logits, batch['label'], pos_weight=pos_weights, reduction='none')

                true_outputs.append(batch['label'])
                logits.append(output_logits)
                losses.append(ce_loss)
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

    def test(self, model_eval=True, num_samples=1, r_seed=42):
        if isinstance(self.loader.dataset, IPISingleDataset) or isinstance(self.loader.dataset, IPIMultiDataset):
            pass
        else:
            return None
        self.model.eval()
        if model_eval:
            self.model.eval()
        if isinstance(self.loader.dataset, TrainValTestDataset):
            self.loader.dataset.test()
        self.loader.dataset.unlimited_epoch = False
        prev_state = enter_local_rng_state(r_seed)

        loader = GeneratorDataLoader(self.loader.dataset, num_workers=self.loader.num_workers)
        pos_weights = self.loader.dataset.comparison_pos_weights.to(self.device)
        n_eff = len(self.loader.dataset.cdr_seqs_train)

        true_outputs = []
        sequences = []
        logits = []
        for i_iter in range(num_samples):
            true_outputs = []
            logits_i = []
            sequences = []

            for i_batch, batch in enumerate(loader):
                for key in batch.keys():
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device, non_blocking=True)

                with torch.no_grad():
                    log_probs, y_choices = self.model.predict_all_y(batch['input'], batch['mask'],
                                                                    batch['decoder_output'])
                    output_logits = self.model.predict_logits(log_probs, y_choices)

                true_outputs.append(batch['label'])
                sequences.extend(batch['sequences'])
                logits_i.append(output_logits)

            true_outputs = torch.cat(true_outputs, 0)
            logits.append(torch.cat(logits_i, 0))

        logits = torch.stack(logits, 0).mean(0)

        error = self.model.calculate_accuracy(
            logits, true_outputs, reduction='none')
        ce_loss = F.binary_cross_entropy_with_logits(
            logits, true_outputs, pos_weight=pos_weights, reduction='none')

        true_outputs = true_outputs.cpu().numpy()
        logits = logits.cpu().numpy()
        roc_scores = roc_auc_score(true_outputs, logits, average=None)
        if isinstance(roc_scores, np.ndarray):
            roc_scores = roc_scores.tolist()
        else:
            roc_scores = [roc_scores]

        # TODO return output per test sequence as well
        true_outputs = true_outputs.mean(0).tolist()
        logits = logits.mean(0).tolist()
        losses = ce_loss.mean(0).cpu().tolist()
        accuracies = error.mean(0).cpu().tolist()

        self.model.train()
        if isinstance(self.loader.dataset, TrainValTestDataset):
            self.loader.dataset.train()
        self.loader.dataset.unlimited_epoch = True
        exit_local_rng_state(prev_state)
        return losses, accuracies, true_outputs, logits, roc_scores

    def save_state(self, last_batch=None):
        snapshot = f"{self.params['snapshot_path']}/{self.params['snapshot_name']}/{self.model.step}.pth"
        revive_exec = f"{self.params['snapshot_path']}/revive_executable/{self.params['snapshot_name']}.sh"
        if not os.path.exists(os.path.dirname(snapshot)):
            os.makedirs(os.path.dirname(snapshot), exist_ok=True)
        torch.save(
            {
                'step': self.model.step,
                'model_type': self.model.MODEL_TYPE,
                'model_state_dict': self.model.state_dict(),
                'model_dims': self.model.dims,
                'model_hyperparams': self.model.hyperparams,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'train_params': self.params,
                'dataset_params': self.loader.dataset.params,
                'last_batch': last_batch
            },
            snapshot
        )
        if 'snapshot_exec_template' in self.params:
            if not os.path.exists(os.path.dirname(revive_exec)):
                os.makedirs(os.path.dirname(revive_exec), exist_ok=True)
            with open(revive_exec, "w") as f:
                snapshot_exec = self.params['snapshot_exec_template'].format(
                    restore=os.path.abspath(snapshot)
                )
                f.write(snapshot_exec)

    def load_state(self, checkpoint, map_location=None):
        if not isinstance(checkpoint, dict):
            checkpoint = torch.load(checkpoint, map_location=map_location)
        if self.model.MODEL_TYPE != checkpoint['model_type']:
            print("Warning: model type mismatch: loaded type {} for model type {}".format(
                checkpoint['model_type'], self.model.MODEL_TYPE
            ))
        if self.model.hyperparams != checkpoint['model_hyperparams']:
            print("Warning: model hyperparameter mismatch")
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.model.step = checkpoint['step']
        self.params.update(checkpoint['train_params'])
