#!/usr/bin/env python
import sys
import os
import argparse
import json
import hashlib

import numpy as np
import pandas as pd
import torch

sys.path.append('..')
import data_loaders
import models
import trainers
from utils import get_cuda_version, get_cudnn_version, get_github_head_hash

working_dir = '/n/groups/marks/projects/antibodies/sequence-classifier/code'
data_dir = '/n/groups/marks/projects/antibodies/sequence-classifier/code'


###################
# PARSE ARGUMENTS #
###################

parser = argparse.ArgumentParser(description="Train an discriminative RNN classifier model.")
parser.add_argument("--restore", type=str, default='', required=True,
                    help="Snapshot name for restoring the model")
parser.add_argument("--dataset", type=str, default=None,
                    help="Dataset name for fitting model.")
parser.add_argument("--output-train", type=str, default='output', required=True,
                    help="Directory and filename of the training output data.")
parser.add_argument("--output-val", type=str, default='output', required=True,
                    help="Directory and filename of the validation output data.")
parser.add_argument("--batch-size", type=int, default=100,
                    help="Batch size.")
parser.add_argument("--num-samples", type=int, default=1,
                    help="Number of iterations to run the model.")
parser.add_argument("--dropout-p-rnn", type=float, default=0.,
                    help="Dropout probability (drop rate, not keep rate)")
parser.add_argument("--dropout-p-dense", type=float, default=0.,
                    help="Dropout probability (drop rate, not keep rate)")
parser.add_argument("--num-data-workers", type=int, default=0,
                    help="Number of workers to load data")
parser.add_argument("--r-seed", type=int, default=42,
                    help="Random seed")
parser.add_argument("--no-cuda", action='store_true',
                    help="Disable GPU training")
args = parser.parse_args()


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

print('Call:', ' '.join(sys.argv))
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

print("git hash:", str(get_github_head_hash()))
print()


###################
# LOAD CHECKPOINT #
###################

print("Restoring model from:", args.restore)
checkpoint = torch.load(os.path.join('../snapshots', args.restore), map_location='cpu' if device.type == 'cpu' else None)


#############
# LOAD DATA #
#############

if 'include_inputs' not in checkpoint['dataset_params']:
    inputs = ['seq']
    if checkpoint['dataset_params']['include_vl']:
        inputs.append('vl')
    if checkpoint['dataset_params']['include_vh']:
        inputs.append('vh')
    checkpoint['dataset_params']['include_inputs'] = inputs

print("Loading data.")
dataset = data_loaders.IPITwoClassSingleClusteredSequenceDataset(
    batch_size=args.batch_size,
    working_dir=data_dir,
    dataset=args.dataset,
    classes=checkpoint['dataset_params']['classes'],
    train_val_split=checkpoint['dataset_params']['train_val_split'],
    matching=True,
    include_inputs=checkpoint['dataset_params']['include_inputs'],
    output_shape=checkpoint['dataset_params']['output_shape'],
    output_types=checkpoint['dataset_params']['output_types'],
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

print("Restoring model from:", args.restore)
dims = checkpoint['model_dims']
hyperparams = checkpoint['model_hyperparams']
trainer_params = checkpoint['train_params']
if args.dropout_p_rnn is not None:
    hyperparams['rnn']['dropout_p'] = args.dropout_p_rnn
if args.dropout_p_dense is not None:
    hyperparams['dense']['dropout_p'] = args.dropout_p_dense
model = models.DiscriminativeRNN(dims=dims, hyperparams=hyperparams)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
print("Num parameters:", model.parameter_count())


################
# RUN TRAINING #
################

trainer = trainers.ClassifierTrainer(
    model=model,
    data_loader=loader,
    params=trainer_params,
    device=device,
)

print()
print("Model:", model.__class__.__name__)
print("Hyperparameters:", json.dumps(model.hyperparams, indent=4))
print("Trainer:", trainer.__class__.__name__)
print("Training parameters:", json.dumps(
    {key: value for key, value in trainer.params.items() if key != 'snapshot_exec_template'}, indent=4))
print("Dataset:", dataset.__class__.__name__)
print("Dataset parameters:", json.dumps(dataset.params, indent=4))
print("Num trainable parameters:", model.parameter_count())

output, logits = trainer.test(model_eval=False, dataset_state='train', num_samples=args.num_samples, raw_output=True)
output = pd.DataFrame(output, columns=output.keys())
output = pd.concat([output, pd.DataFrame(logits, columns=[f'logits_{i}' for i in range(logits.shape[1])])], axis=1)
output.to_csv(args.output_train, index=False)

output, logits = trainer.test(model_eval=False, dataset_state='val', num_samples=args.num_samples, raw_output=True)
output = pd.DataFrame(output, columns=output.keys())
output = pd.concat([output, pd.DataFrame(logits, columns=[f'logits_{i}' for i in range(logits.shape[1])])], axis=1)
output.to_csv(args.output_val, index=False)

print("Done!")
