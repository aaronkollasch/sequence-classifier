#!/usr/bin/env python
import sys
import argparse
import time
import json
import warnings
import hashlib

import numpy as np
import torch

import data_loaders
import models
import trainers
import model_logging
from utils import get_cuda_version, get_cudnn_version

# working_dir = '/n/groups/marks/projects/antibodies/sequence-classifier/code'
# data_dir = '/n/groups/marks/projects/antibodies/sequence-classifier/code'
working_dir = '.'
data_dir = '.'

parser = argparse.ArgumentParser(description="Train an generative RNN classifier model.")
parser.add_argument("--hidden-size", type=int, default=100,
                    help="Number of channels in the RNN hidden state.")
parser.add_argument("--num-layers", type=int, default=2,
                    help="Number of RNN layers.")
parser.add_argument("--hidden-size-dense", type=int, default=50,
                    help="Number of channels in the dense hidden state.")
parser.add_argument("--num-layers-dense", type=int, default=2,
                    help="Number of dense layers.")
parser.add_argument("--num-pretrain-iterations", metavar='N', type=int, default=250005,
                    help="Number of iterations to run the unlabeled model.")
parser.add_argument("--num-finetune-iterations", metavar='N', type=int, default=50000,
                    help="Number of iterations to run the labeled model.")
parser.add_argument("--batch-size", metavar='B', type=int, default=32,
                    help="Batch size.")
parser.add_argument("--dataset", metavar='D', type=str, default=None,
                    help="Dataset name for fitting model. Alignment weights must be computed beforehand.")
parser.add_argument("--labeled-dataset", metavar='D', type=str, default=None,
                    help="Dataset name for fitting model. Alignment weights must be computed beforehand.")
parser.add_argument("--test-datasets", type=str, default=[], nargs='*',
                    help="Datasets names in \"dataset\" column to use for testing.")
parser.add_argument("--train-val-split", type=float, default=0.9,
                    help="Proportion of training data to use for training.")
parser.add_argument("--include-inputs", nargs='*', default=('seq', 'vh'),
                    help="Data to include in the input. (seq, vh, vl)")
parser.add_argument("--num-data-workers", metavar='N', type=int, default=4,
                    help="Number of workers to load data")
parser.add_argument("--restore", type=str, default=None,
                    help="Snapshot path for restoring a model to continue training.")
parser.add_argument("--run-name", metavar='NAME', type=str, default=None,
                    help="Name of run")
parser.add_argument("--r-seed", type=int, default=42,
                    help="Random seed")
parser.add_argument("--dropout-p-rnn", type=float, default=0.2,
                    help="Dropout probability (drop rate, not keep rate)")
parser.add_argument("--dropout-p-dense", type=float, default=0.5,
                    help="Dropout probability (drop rate, not keep rate)")
parser.add_argument("--bayes-reg", action='store_true',
                    help="Use bayesian weight uncertainty")
parser.add_argument("--no-cuda", action='store_true',
                    help="Disable GPU training")
args = parser.parse_args()

if len(args.test_datasets) == 0:
    warnings.warn('No test datasets specified.')

if args.run_name is None:
    args.run_name = f"{args.dataset.split('/')[-1].split('.')[0]}_generative" \
        f"_n-r-{args.num_layers}-d-{args.num_layers_dense}" \
        f"_h-r-{args.hidden_size}-d-{args.hidden_size_dense}" \
        f"_drop-r-{args.dropout_p_rnn}-d-{args.dropout_p_dense}" \
        f"_reg-{'bayes' if args.bayes_reg else'l2'}" \
        f"_test-{','.join(args.test_datasets)}" \
        f"_rseed-{args.r_seed}_start-{time.strftime('%y%b%d_%H%M', time.localtime())}"

restore_args = " \\\n  ".join(sys.argv[1:])
if "--run-name" not in restore_args:
    restore_args += f" \\\n  --run-name {args.run_name}"

sbatch_executable = f"""#!/bin/bash
#SBATCH -c 4                               # Request one core
#SBATCH -N 1                               # Request one node (if you request more than one core with -c, also using
                                           # -N 1 means all cores will be on the same node)
#SBATCH -t 2-11:59                         # Runtime in D-HH:MM format
#SBATCH -p gpu                             # Partition to run in
#SBATCH --gres=gpu:1
#SBATCH --mem=40G                          # Memory total in MB (for all cores)
#SBATCH -o slurm_files/slurm-%j.out        # File to which STDOUT + STDERR will be written, including job ID in filename
hostname
pwd
module load gcc/6.2.0 cuda/9.0
srun stdbuf -oL -eL {sys.executable} \\
  {sys.argv[0]} \\
  {restore_args} \\
  --restore {{restore}}
"""

if args.restore is not None:
    # prevent from repeating batches/seed when restoring at intermediate point
    # script is repeatable as long as restored at same point with same restore string
    args.r_seed += int(hashlib.sha1(args.restore.encode()).hexdigest(), 16)
    args.r_seed = args.r_seed % (2 ** 32 - 1)  # limit of np.random.seed

np.random.seed(args.r_seed)
torch.manual_seed(args.r_seed)
torch.cuda.manual_seed_all(args.r_seed)


def _init_fn(worker_id):
    np.random.seed(args.r_seed + worker_id)


print("OS: ", sys.platform)
print("Python: ", sys.version)
print("PyTorch: ", torch.__version__)
print("Numpy: ", np.__version__)

USE_CUDA = not args.no_cuda
device = torch.device("cuda:0" if USE_CUDA and torch.cuda.is_available() else "cpu")
print('Using device:', device)
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3, 1), 'GB')
    print(get_cuda_version())
    print("CuDNN Version ", get_cudnn_version())
print()

print("Run:", args.run_name)


def get_dataset(step=0):
    if step < args.num_pretrain_iterations:
        _dataset = data_loaders.VHClusteredAntibodyDataset(
            batch_size=args.batch_size,
            working_dir=data_dir,
            dataset=args.dataset,
            matching=True,
            unlimited_epoch=True,
            include_inputs=args.include_inputs,
            output_shape='NLC',
            output_types='decoder',
        )
        _loader = data_loaders.GeneratorDataLoader(
            _dataset,
            num_workers=args.num_data_workers,
            pin_memory=True,
            worker_init_fn=_init_fn
        )
        return _dataset, _loader
    else:
        _dataset = data_loaders.IPIMultiDataset(
            batch_size=args.batch_size,
            working_dir=working_dir,
            dataset=args.labeled_dataset,
            test_datasets=args.test_datasets,
            train_val_split=args.train_val_split,
            comparisons=(('Aff1', 'PSR1', 2., 2.),),
            matching=True,
            unlimited_epoch=True,
            include_inputs=args.include_inputs,
            output_shape='NLC',
            output_types='decoder',
        )
        _loader = data_loaders.GeneratorDataLoader(
            _dataset,
            num_workers=args.num_data_workers,
            pin_memory=True,
            worker_init_fn=_init_fn
        )
        return _dataset, _loader


if args.restore is not None:
    print("Restoring model from:", args.restore)
    checkpoint = torch.load(args.restore, map_location='cpu' if device.type == 'cpu' else None)
    dataset, loader = get_dataset(checkpoint['step'])
    dims = checkpoint['model_dims']
    hyperparams = checkpoint['model_hyperparams']
    trainer_params = checkpoint['train_params']
    if args.dropout_p_rnn is not None:
        hyperparams['rnn']['dropout_p'] = args.dropout_p_rnn
    if args.dropout_p_dense is not None:
        hyperparams['dense']['dropout_p'] = args.dropout_p_dense
    model = models.GenerativeRNN(dims=dims, hyperparams=hyperparams)
else:
    dataset, loader = get_dataset()
    checkpoint = args.restore
    trainer_params = None
    dims = {'input': dataset.input_dim}
    hyperparams = {'rnn': {}, 'dense': {}, 'regularization': {}}
    for param_name_1, param_name_2, param in (
            ('rnn', 'hidden_size', args.hidden_size),
            ('rnn', 'num_layers', args.num_layers),
            ('rnn', 'dropout_p', args.dropout_p_rnn),
            ('dense', 'hidden_size', args.hidden_size_dense),
            ('dense', 'num_layers', args.num_layers_dense),
            ('dense', 'dropout_p', args.dropout_p_dense),
            ('regularization', 'bayesian', args.bayes_reg)
    ):
        if param is not None:
            hyperparams[param_name_1][param_name_2] = param
    model = models.GenerativeRNN(dims=dims, hyperparams=hyperparams)
model.to(device)

trainer = trainers.GenerativeClassifierTrainer(
    model=model,
    data_loader=loader,
    params=trainer_params,
    snapshot_path=working_dir + '/snapshots',
    snapshot_name=args.run_name+'_pretrain',
    snapshot_interval=args.num_pretrain_iterations // 10,
    snapshot_exec_template=sbatch_executable,
    device=device,
    # logger=model_logging.Logger(),
    logger=model_logging.TensorboardLogger(
        log_interval=500,
        validation_interval=1000,
        generate_interval=5000,
        test_interval=1000,
        log_dir=working_dir + '/logs/' + args.run_name,
        print_output=True,
    )
)
if args.restore is not None:
    trainer.load_state(checkpoint)

# print("Hyperparameters:", json.dumps(model.hyperparams, indent=4))
# print("Training parameters:", json.dumps(
#     {key: value for key, value in trainer.params.items() if key != 'snapshot_exec_template'}, indent=4))
# print("Dataset parameters:", json.dumps(dataset.params, indent=4))
# print("Num trainable parameters:", model.parameter_count())

if model.step < args.num_pretrain_iterations:
    for name, p in model.dense_net.named_parameters():
        if 'bias' in name:
            p.data.zero_()
    model.enable_gradient = 'r'

    print()
    print("Model:", model.__class__.__name__)
    print("Gradient enabled:", model.enable_gradient)
    print("Hyperparameters:", json.dumps(model.hyperparams, indent=4))
    print("Trainer:", trainer.__class__.__name__)
    print("Training parameters:", json.dumps(
        {key: value for key, value in trainer.params.items() if key != 'snapshot_exec_template'}, indent=4))
    print("Dataset:", dataset.__class__.__name__)
    print("Dataset parameters:", json.dumps(dataset.params, indent=4))
    print("Num trainable parameters:", model.parameter_count())
    print(f"Pre-training for {args.num_pretrain_iterations - model.step} iterations.")

    trainer.train(steps=args.num_pretrain_iterations)
    del dataset
    del loader
    del trainer.loader
    dataset, loader = get_dataset(model.step)
    trainer.loader = loader
    # print("Dataset parameters:", json.dumps(dataset.params, indent=4))

trainer.params['snapshot_interval'] = args.num_finetune_iterations // 10
trainer.params['snapshot_name'] = trainer.params['snapshot_name'].replace('_pretrain', '_finetune')
model.enable_gradient = 'redb'

print()
print("Model:", model.__class__.__name__)
print("Gradient enabled:", model.enable_gradient)
print("Hyperparameters:", json.dumps(model.hyperparams, indent=4))
print("Trainer:", trainer.__class__.__name__)
print("Training parameters:", json.dumps(
    {key: value for key, value in trainer.params.items() if key != 'snapshot_exec_template'}, indent=4))
print("Dataset:", dataset.__class__.__name__)
print("Dataset parameters:", json.dumps(dataset.params, indent=4))
print("Num trainable parameters:", model.parameter_count())
print(f"Training for {args.num_pretrain_iterations + args.num_finetune_iterations - model.step} iterations.")

trainer.train(steps=args.num_pretrain_iterations + args.num_finetune_iterations)
