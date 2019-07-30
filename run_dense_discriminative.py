#!/usr/bin/env python
import sys
import os
import argparse
import time
import json
import hashlib
import warnings

import numpy as np
import torch

import data_loaders
import models
import trainers
import model_logging
from utils import get_cuda_version, get_cudnn_version, Tee

working_dir = '/n/groups/marks/projects/antibodies/sequence-classifier/code'
data_dir = '/n/groups/marks/projects/antibodies/sequence-classifier/code'


###################
# PARSE ARGUMENTS #
###################

parser = argparse.ArgumentParser(description="Train a dense discriminative classifier model.")
parser.add_argument("--hidden-size", type=int, default=50,
                    help="Number of channels in the dense hidden state.")
parser.add_argument("--num-layers", type=int, default=2,
                    help="Number of dense layers.")
parser.add_argument("--num-iterations", type=int, default=100000,
                    help="Number of iterations to run the model.")
parser.add_argument("--batch-size", type=int, default=32,
                    help="Batch size.")
parser.add_argument("--dataset", type=str, default=None,
                    help="Dataset name for fitting model.")
parser.add_argument("--test-datasets", type=str, default=[], nargs='*',
                    help="Datasets names in \"dataset\" column to use for testing.")
parser.add_argument("--comparison", type=str, default="Aff1-PSR1",
                    help="Comparison to test against.")
parser.add_argument("--comparison-thresh", type=float, default=2.,
                    help="Minimum counts to include in comparison.")
parser.add_argument("--train-val-split", type=float, default=0.9,
                    help="Proportion of training data to use for training.")
parser.add_argument("--include-inputs", nargs='*', default=('seq', 'vh', 'vl'),
                    help="Data to include in the input. (seq, vh, vl)")
parser.add_argument("--max-k", type=int, default=3,
                    help="Maximum k-mer size.")
parser.add_argument("--include-length", action='store_true',
                    help="Include the CDR3 length as a feature in the k-mer vector.")
parser.add_argument("--num-data-workers", type=int, default=4,
                    help="Number of workers to load data")
parser.add_argument("--restore", type=str, default=None,
                    help="Snapshot path for restoring a model to continue training.")
parser.add_argument("--run-name", type=str, default=None,
                    help="Name of run")
parser.add_argument("--r-seed", type=int, default=42,
                    help="Random seed")
parser.add_argument("--dropout-p", type=float, default=0.5,
                    help="Dropout probability (drop rate, not keep rate)")
parser.add_argument("--bayes-reg", action='store_true',
                    help="Use bayesian weight uncertainty")
parser.add_argument("--no-cuda", action='store_true',
                    help="Disable GPU training")
args = parser.parse_args()

if len(args.test_datasets) == 0:
    warnings.warn('No test datasets specified.')


########################
# MAKE RUN DESCRIPTORS #
########################

if args.run_name is None:
    args.run_name = f"{args.dataset.split('/')[-1].split('.')[0]}" \
        f"_dense_n-{args.num_layers}_h-{args.hidden_size}_drop-{args.dropout_p}" \
        f"_reg-{'bayes' if args.bayes_reg else'l2'}" \
        f"_test-{','.join(args.test_datasets)}" \
        f"_rseed-{args.r_seed}_start-{time.strftime('%y%b%d_%H%M', time.localtime())}"

restore_args = " \\\n  ".join(sys.argv[1:])
if "--run-name" not in restore_args:
    restore_args += f" \\\n --run-name {args.run_name}"

sbatch_executable = f"""#!/bin/bash
#SBATCH -c 4                               # Request one core
#SBATCH -N 1                               # Request one node (if you request more than one core with -c, also using
                                           # -N 1 means all cores will be on the same node)
#SBATCH -t 0-11:59                         # Runtime in D-HH:MM format
#SBATCH -p gpu                             # Partition to run in
#SBATCH --gres=gpu:1
#SBATCH --mem=30G                          # Memory total in MB (for all cores)
#SBATCH -o slurm_files/slurm-%j.out        # File to which STDOUT + STDERR will be written, including job ID in filename
hostname
pwd
module load gcc/6.2.0 cuda/9.0
srun stdbuf -oL -eL {sys.executable} \\
  {sys.argv[0]} \\
  {restore_args} \\
  --restore {{restore}}
"""


####################
# SET RANDOM SEEDS #
####################

if args.restore is not None:
    # prevent from repeating batches/seed when restoring at intermediate point
    # script is repeatable as long as restored at same point with same restore string and same num_workers
    args.r_seed += int(hashlib.sha1(args.restore.encode()).hexdigest(), 16)
    args.r_seed = args.r_seed % (2 ** 32 - 1)  # limit of np.random.seed

np.random.seed(args.r_seed)
torch.manual_seed(args.r_seed)
torch.cuda.manual_seed_all(args.r_seed)


def _init_fn(worker_id):
    np.random.seed(args.r_seed + worker_id)


#####################
# PRINT SYSTEM INFO #
#####################

os.makedirs(f'logs/{args.run_name}', exist_ok=True)
log_f = Tee(f'logs/{args.run_name}/log.txt', 'a')

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


#############
# LOAD DATA #
#############

print("Run:", args.run_name)

print("Loading data.")
dataset = data_loaders.IPIMultiDataset(
    batch_size=args.batch_size,
    working_dir=data_dir,
    dataset=args.dataset,
    test_datasets=args.test_datasets,
    train_val_split=args.train_val_split,
    matching=True,
    unlimited_epoch=True,
    include_inputs=args.include_inputs,
    comparisons=(
        (
            args.comparison.split('-')[0],
            args.comparison.split('-')[1],
            args.comparison_thresh,
            args.comparison_thresh
        ),
    ),
    output_shape='NLC',
    output_types='kmer_vector',
    kmer_params=dict(max_k=args.max_k, include_length=args.include_length),
)
loader = data_loaders.GeneratorDataLoader(
    dataset,
    num_workers=args.num_data_workers,
    pin_memory=True,
    worker_init_fn=_init_fn
)


##############
# LOAD MODEL #
##############

if args.restore is not None:
    print("Restoring model from:", args.restore)
    checkpoint = torch.load(args.restore, map_location='cpu' if device.type == 'cpu' else None)
    dims = checkpoint['model_dims']
    hyperparams = checkpoint['model_hyperparams']
    trainer_params = checkpoint['train_params']
    if args.dropout_p is not None:
        hyperparams['dense']['dropout_p'] = args.dropout_p
    model = models.DiscriminativeDense(dims=dims, hyperparams=hyperparams)
else:
    checkpoint = args.restore
    trainer_params = None
    dims = {'length': 1, 'input': dataset.input_dim}
    hyperparams = {'dense': {}, 'regularization': {}}
    for param_name_1, param_name_2, param in (
        ('dense', 'hidden_size', args.hidden_size),
        ('dense', 'num_layers', args.num_layers),
        ('dense', 'dropout_p', args.dropout_p),
        ('regularization', 'bayesian', args.bayes_reg)
    ):
        if param is not None:
            hyperparams[param_name_1][param_name_2] = param
    model = models.DiscriminativeDense(dims=dims, hyperparams=hyperparams)
model.to(device)


################
# RUN TRAINING #
################

trainer = trainers.ClassifierTrainer(
    model=model,
    data_loader=loader,
    params=trainer_params,
    snapshot_path=working_dir + '/snapshots',
    snapshot_name=args.run_name,
    snapshot_interval=args.num_iterations // 10,
    snapshot_exec_template=sbatch_executable,
    device=device,
    # logger=model_logging.Logger(),
    logger=model_logging.TensorboardLogger(
        log_interval=100,
        validation_interval=500,
        generate_interval=500,
        test_interval=500,
        log_dir=working_dir + '/logs/' + args.run_name,
        print_output=True
    )
)
if args.restore is not None:
    trainer.load_state(checkpoint)

print()
print("Model:", model.__class__.__name__)
print("Hyperparameters:", json.dumps(model.hyperparams, indent=4))
print("Trainer:", trainer.__class__.__name__)
print("Training parameters:", json.dumps(
    {key: value for key, value in trainer.params.items() if key != 'snapshot_exec_template'}, indent=4))
print("Dataset:", dataset.__class__.__name__)
print("Dataset parameters:", json.dumps(dataset.params, indent=4))
print("Num trainable parameters:", model.parameter_count())
print(f"Training for {args.num_iterations - model.step} iterations.")

trainer.train(steps=args.num_iterations)

print(model.dense_net_modules['linear_1'].weight)
print(model.dense_net_modules['linear_1'].bias)
