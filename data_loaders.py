import os
import glob
import math

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from Bio.SeqIO.FastaIO import SimpleFastaParser

from utils import temp_seed

PROTEIN_ALPHABET = 'ACDEFGHIKLMNPQRSTVWY*'
PROTEIN_REORDERED_ALPHABET = 'DEKRHNQSTPGAVILMCFYW*'
RNA_ALPHABET = 'ACGU*'
DNA_ALPHABET = 'ACGT*'
START_END = "*"


def get_alphabet(alphabet_type='protein'):
    if alphabet_type == 'protein':
        return PROTEIN_ALPHABET, PROTEIN_REORDERED_ALPHABET
    elif alphabet_type == 'RNA':
        return RNA_ALPHABET, RNA_ALPHABET
    elif alphabet_type == 'DNA':
        return DNA_ALPHABET, DNA_ALPHABET
    else:
        raise ValueError('unknown alphabet type')


class GeneratorDataset(data.Dataset):
    """A Dataset that can be used as a generator"""
    def __init__(
            self,
            batch_size=32,
            unlimited_epoch=True,
    ):
        self.batch_size = batch_size
        self.unlimited_epoch = unlimited_epoch

    @property
    def n_eff(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        if self.unlimited_epoch:
            return 2 ** 62
        else:
            return math.ceil(self.n_eff / self.batch_size)

    @staticmethod
    def collate_fn(batch):
        return batch[0]


class GeneratorDataLoader(data.DataLoader):
    """A DataLoader used with a GeneratorDataset"""
    def __init__(self, dataset: GeneratorDataset, **kwargs):
        kwargs.update(dict(
            batch_size=1, shuffle=False, sampler=None, batch_sampler=None, collate_fn=dataset.collate_fn,
        ))
        super(GeneratorDataLoader, self).__init__(
            dataset,
            **kwargs)


class TrainTestDataset(data.Dataset):
    """A Dataset that has training and testing modes"""
    def __init__(self):
        self._training = True

    def train(self, training=True):
        self._training = training

    def test(self):
        self.train(False)

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class SequenceDataset(GeneratorDataset):
    """Abstract sequence dataset"""
    supported_output_shapes = ['NCHW', 'NHWC', 'NLC']

    def __init__(
            self,
            batch_size=32,
            unlimited_epoch=True,
            alphabet_type='protein',
            reverse=False,
            matching=False,
            output_shape='NCHW',
    ):
        super(SequenceDataset, self).__init__(batch_size=batch_size, unlimited_epoch=unlimited_epoch)

        self.alphabet_type = alphabet_type
        self.reverse = reverse
        self.matching = matching
        self.output_shape = output_shape

        if output_shape not in self.supported_output_shapes:
            raise KeyError(f'Unsupported output shape: {output_shape}')

        # Load up the alphabet type to use, whether that be DNA, RNA, or protein
        if self.alphabet_type == 'protein':
            self.alphabet = PROTEIN_ALPHABET
            self.reorder_alphabet = PROTEIN_REORDERED_ALPHABET
        elif self.alphabet_type == 'RNA':
            self.alphabet = RNA_ALPHABET
            self.reorder_alphabet = RNA_ALPHABET
        elif self.alphabet_type == 'DNA':
            self.alphabet = DNA_ALPHABET
            self.reorder_alphabet = DNA_ALPHABET

        # Make a dictionary that goes from aa to a number for one-hot
        self.aa_dict = {}
        self.idx_to_aa = {}
        for i, aa in enumerate(self.alphabet):
            self.aa_dict[aa] = i
            self.idx_to_aa[i] = aa

    @property
    def n_eff(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def sequences_to_onehot(self, sequences, reverse=None, matching=None):
        """

        :param sequences: list/iterable of strings
        :param reverse: reverse the sequences
        :param matching: output forward and reverse sequences
        :return: dictionary of strings
        """
        reverse = self.reverse if reverse is None else reverse
        matching = self.matching if matching is None else matching
        num_seqs = len(sequences)
        max_seq_len = max([len(seq) for seq in sequences]) + 1
        prot_decoder_output = torch.zeros((num_seqs, len(self.alphabet), 1, max_seq_len))
        prot_decoder_input = torch.zeros((num_seqs, len(self.alphabet), 1, max_seq_len))

        if matching:
            prot_decoder_output_r = torch.zeros((num_seqs, len(self.alphabet), 1, max_seq_len))
            prot_decoder_input_r = torch.zeros((num_seqs, len(self.alphabet), 1, max_seq_len))

        prot_mask_decoder = torch.zeros((num_seqs, 1, 1, max_seq_len))

        for i, sequence in enumerate(sequences):
            if reverse:
                sequence = sequence[::-1]

            decoder_input_seq = '*' + sequence
            decoder_output_seq = sequence + '*'

            if matching:
                sequence_r = sequence[::-1]
                decoder_input_seq_r = '*' + sequence_r
                decoder_output_seq_r = sequence_r + '*'

            for j in range(len(decoder_input_seq)):
                prot_decoder_input[i, self.aa_dict[decoder_input_seq[j]], 0, j] = 1
                prot_decoder_output[i, self.aa_dict[decoder_output_seq[j]], 0, j] = 1
                prot_mask_decoder[i, 0, 0, j] = 1

                if matching:
                    prot_decoder_input_r[i, self.aa_dict[decoder_input_seq_r[j]], 0, j] = 1
                    prot_decoder_output_r[i, self.aa_dict[decoder_output_seq_r[j]], 0, j] = 1

        if matching:
            output = {
                'prot_decoder_input': prot_decoder_input,
                'prot_decoder_output': prot_decoder_output,
                'prot_mask_decoder': prot_mask_decoder,
                'prot_decoder_input_r': prot_decoder_input_r,
                'prot_decoder_output_r': prot_decoder_output_r
            }
        else:
            output = {
                'prot_decoder_input': prot_decoder_input,
                'prot_decoder_output': prot_decoder_output,
                'prot_mask_decoder': prot_mask_decoder
            }

        for key in output.keys():
            if self.output_shape == 'NHWC':
                output[key] = output[key].permute(0, 2, 3, 1).contiguous()
            elif self.output_shape == 'NLC':
                output[key] = output[key].squeeze(2).permute(0, 2, 1).contiguous()

        return output


class FastaDataset(SequenceDataset):
    """Load batches of sequences from a fasta file, either sequentially or sampled isotropically"""
    def __init__(
            self,
            dataset='',
            working_dir='.',
            batch_size=32,
            unlimited_epoch=False,
            alphabet_type='protein',
            reverse=False,
            matching=False,
            output_shape='NCHW',
    ):
        super(FastaDataset, self).__init__(
            batch_size=batch_size,
            unlimited_epoch=unlimited_epoch,
            alphabet_type=alphabet_type,
            reverse=reverse,
            matching=matching,
            output_shape=output_shape,
        )
        self.dataset = dataset
        self.working_dir = working_dir

        self.names = None
        self.sequences = None

        self.load_data()

    def load_data(self):
        filename = os.path.join(self.working_dir, self.dataset)
        names_list = []
        sequence_list = []

        with open(filename, 'r') as fa:
            for i, (title, seq) in enumerate(SimpleFastaParser(fa)):
                valid = True
                for letter in seq:
                    if letter not in self.aa_dict:
                        valid = False
                if not valid:
                    continue

                names_list.append(title)
                sequence_list.append(seq)

        self.names = np.array(names_list)
        self.sequences = np.array(sequence_list)

        print("Number of sequences:", self.n_eff)

    @property
    def n_eff(self):
        return len(self.sequences)  # not a true n_eff

    def __getitem__(self, index):
        """
        :param index: batch index; ignored if unlimited_epoch
        :return: batch of size self.batch_size
        """

        if self.unlimited_epoch:
            indices = np.random.randint(0, self.n_eff, self.batch_size)
        else:
            first_index = index * self.batch_size
            last_index = min((index+1) * self.batch_size, self.n_eff)
            indices = np.arange(first_index, last_index)

        seqs = self.sequences[indices]
        batch = self.sequences_to_onehot(seqs)
        batch['names'] = self.names[indices]
        batch['sequences'] = seqs
        return batch


class SingleFamilyDataset(SequenceDataset):
    def __init__(
            self,
            dataset='',
            working_dir='.',
            batch_size=32,
            unlimited_epoch=True,
            alphabet_type='protein',
            reverse=False,
            matching=False,
            output_shape='NCHW',
    ):
        super(SingleFamilyDataset, self).__init__(
            batch_size=batch_size,
            unlimited_epoch=unlimited_epoch,
            alphabet_type=alphabet_type,
            reverse=reverse,
            matching=matching,
            output_shape=output_shape,
        )
        self.dataset = dataset
        self.working_dir = working_dir

        self.family_name_to_sequence_list = {}
        self.family_name_to_sequence_weight_list = {}
        self.family_name_to_n_eff = {}
        self.family_name_list = []
        self.family_idx_list = []
        self.family_name = ''
        self.family_name_to_idx = {}
        self.idx_to_family_name = {}

        self.seq_len = 0
        self.num_families = 0
        self.max_family_size = 0

        self.load_data()

    def load_data(self):
        max_seq_len = 0
        max_family_size = 0
        family_name = ''
        weight_list = []

        f_names = glob.glob(self.working_dir + '/datasets/sequences/' + self.dataset + '*.fa')
        if len(f_names) != 1:
            raise AssertionError('Wrong number of families: {}'.format(len(f_names)))

        for filename in f_names:
            sequence_list = []
            weight_list = []

            family_name_list = filename.split('/')[-1].split('_')
            family_name = family_name_list[0] + '_' + family_name_list[1]
            print(family_name)

            family_size = 0
            ind_family_idx_list = []
            with open(filename, 'r') as fa:
                for i, (title, seq) in enumerate(SimpleFastaParser(fa)):
                    weight = float(title.split(':')[-1])
                    valid = True
                    for letter in seq:
                        if letter not in self.aa_dict:
                            valid = False
                    if not valid:
                        continue

                    sequence_list.append(seq)
                    ind_family_idx_list.append(family_size)
                    weight_list.append(weight)
                    family_size += 1
                    if len(seq) > max_seq_len:
                        max_seq_len = len(seq)

            if family_size > max_family_size:
                max_family_size = family_size

            self.family_name_to_sequence_list[family_name] = sequence_list
            self.family_name_to_sequence_weight_list[family_name] = (
                np.asarray(weight_list) / np.sum(weight_list)
            ).tolist()
            self.family_name_to_n_eff[family_name] = np.sum(weight_list)
            self.family_name = family_name
            self.family_name_list.append(family_name)
            self.family_idx_list.append(ind_family_idx_list)

        self.family_name = family_name
        self.seq_len = max_seq_len
        self.num_families = len(self.family_name_list)
        self.max_family_size = max_family_size

        print("Number of families:", self.num_families)
        print("Neff:", np.sum(weight_list))
        print("Max family size:", max_family_size)

        for i, family_name in enumerate(self.family_name_list):
            self.family_name_to_idx[family_name] = i
            self.idx_to_family_name[i] = family_name

    @property
    def n_eff(self):
        return self.family_name_to_n_eff[self.family_name]

    def __getitem__(self, index):
        """
        :param index: ignored
        :return: batch of size self.batch_size
        """
        family_name = self.family_name
        family_seqs = self.family_name_to_sequence_list[family_name]
        family_weights = self.family_name_to_sequence_weight_list[family_name]

        seq_idx = np.random.choice(len(family_seqs), self.batch_size, p=family_weights)
        seqs = [family_seqs[idx] for idx in seq_idx]

        batch = self.sequences_to_onehot(seqs)
        return batch


class DoubleWeightedNanobodyDataset(SequenceDataset):
    def __init__(
            self,
            dataset='',
            working_dir='.',
            batch_size=32,
            unlimited_epoch=True,
            alphabet_type='protein',
            reverse=False,
            matching=False,
            output_shape='NCHW',
    ):
        super(DoubleWeightedNanobodyDataset, self).__init__(
            batch_size=batch_size,
            unlimited_epoch=unlimited_epoch,
            alphabet_type=alphabet_type,
            reverse=reverse,
            matching=matching,
            output_shape=output_shape,
        )
        self.dataset = dataset
        self.working_dir = working_dir

        self.name_to_sequence = {}
        self.clu1_to_clu2_to_seq_names = {}
        self.clu1_to_clu2_to_clu_size = {}
        self.clu1_list = []

        self.load_data()

    def load_data(self):
        filename = self.working_dir + '/datasets/' + self.dataset
        with open(filename, 'r') as fa:
            for i, (title, seq) in enumerate(SimpleFastaParser(fa)):
                name, clu1, clu2 = title.split(':')
                valid = True
                for letter in seq:
                    if letter not in self.aa_dict:
                        valid = False
                if not valid:
                    continue

                self.name_to_sequence[name] = seq
                if clu1 in self.clu1_to_clu2_to_seq_names:
                    if clu2 in self.clu1_to_clu2_to_seq_names[clu1]:
                        self.clu1_to_clu2_to_seq_names[clu1][clu2].append(name)
                        self.clu1_to_clu2_to_clu_size[clu1][clu2] += 1
                    else:
                        self.clu1_to_clu2_to_seq_names[clu1][clu2] = [name]
                        self.clu1_to_clu2_to_clu_size[clu1][clu2] = 1
                else:
                    self.clu1_to_clu2_to_seq_names[clu1] = {clu2: [name]}
                    self.clu1_to_clu2_to_clu_size[clu1] = {clu2: 1}

        self.clu1_list = list(self.clu1_to_clu2_to_seq_names.keys())
        print("Num clusters:", len(self.clu1_list))

    @property
    def n_eff(self):
        return len(self.clu1_list)

    def __getitem__(self, index):
        """
        :param index: ignored
        :return: batch of size self.batch_size
        """
        seqs = []
        for i in range(self.batch_size):
            # Pick a cluster id80
            clu1_idx = np.random.randint(0, len(self.clu1_list))
            clu1 = self.clu1_list[clu1_idx]

            # Then pick a cluster id90 from the cluster id80s
            clu2 = np.random.choice(list(self.clu1_to_clu2_to_seq_names[clu1].keys()))

            # Then pick a random sequence all in those clusters
            seq_name = np.random.choice(self.clu1_to_clu2_to_seq_names[clu1][clu2])

            # then grab the associated sequence
            seqs.append(self.name_to_sequence[seq_name])

        batch = self.sequences_to_onehot(seqs)
        return batch


class IPISequenceDataset(SequenceDataset, TrainTestDataset):
    def __init__(
            self,
            dataset='',
            working_dir='.',
            batch_size=32,
            unlimited_epoch=True,
            alphabet_type='protein',
            reverse=False,
            matching=False,
            output_shape='NCHW',
            comparisons=(('Aff1', 'PSR1', 0., 0.),),  # before, after, thresh_before, thresh_after
            train_test_split=1.0,
            split_seed=42,
    ):
        SequenceDataset.__init__(
            self,
            batch_size=batch_size,
            unlimited_epoch=unlimited_epoch,
            alphabet_type=alphabet_type,
            reverse=reverse,
            matching=matching,
            output_shape=output_shape,
        )
        TrainTestDataset.__init__(self)
        self.dataset = dataset
        self.working_dir = working_dir
        self.comparisons = comparisons
        self.train_test_split = train_test_split
        self.split_seed = split_seed

        self.light_to_idx = {'VK1-39': 0, 'VL1-51': 1, 'VK3-15': 2}
        self.heavy_to_idx = {'VH1-46': 0, 'VH1-69': 1, 'VH3-7': 2, 'VH3-15': 3, 'VH4-39': 4, 'VH5-51': 5}
        self.input_dim = len(self.alphabet) + len(self.light_to_idx) + len(self.heavy_to_idx)

        self.cdr_to_output = {}
        self.cdr_to_heavy = {}
        self.cdr_to_light = {}
        self.all_cdr_seqs = []
        self.cdr_seqs_train = []
        self.cdr_seqs_test = []
        self.comparison_pos_weights = torch.ones(len(comparisons))

        self.load_data()

    def load_data(self):
        seq_col = 'CDR3'
        heavy_col = 'heavy'
        light_col = 'light'
        count_cols = list({col: None for comparison in self.comparisons for col in comparison[:2]}.keys())

        # load data file
        filename = os.path.join(self.working_dir, self.dataset)
        use_cols = [seq_col, heavy_col, light_col] + count_cols
        df = pd.read_csv(filename, usecols=use_cols)
        df[count_cols] = df[count_cols].fillna(0.)

        # load output data
        comparison_cdr_to_output = []
        for i_comparison, comparison in enumerate(self.comparisons):
            before, after, before_threshold, after_threshold = comparison
            comp_df = df.loc[(df[before] > before_threshold) | (df[after] > after_threshold), :]

            comp_out = pd.Series((comp_df[after] > after_threshold).astype(int).values, index=comp_df[seq_col])
            pos_weight = (len(comp_out) - comp_out.sum()) / comp_out.sum()
            comparison_cdr_to_output.append(comp_out.to_dict())
            self.comparison_pos_weights[i_comparison] = pos_weight
            print(f'comparison: {comparison}, {len(comp_out)} seqs, '
                  f'{comp_out.mean() * 100:0.1f}% positive, {pos_weight:0.4f} pos_weight')

        # keep only sequences with all output information
        all_cdrs = set.intersection(*(set(d.keys()) for d in comparison_cdr_to_output))
        df = df[df[seq_col].isin(all_cdrs)]
        self.all_cdr_seqs = df[seq_col].values
        print(f'total seqs after intersection: {len(self.all_cdr_seqs)}')

        # split data into train-test
        with temp_seed(self.split_seed):
            indices = np.random.permutation(len(self.all_cdr_seqs))
            partition = math.ceil(len(indices) * self.train_test_split)
            training_idx, test_idx = indices[:partition], indices[partition:]
            self.cdr_seqs_train, self.cdr_seqs_test = self.all_cdr_seqs[training_idx], self.all_cdr_seqs[test_idx]
            print(f'train-test split: {self.train_test_split}')
            print(f'num train, test seqs: {len(self.cdr_seqs_train)}, {len(self.cdr_seqs_test)}')

        # make table of output values
        self.cdr_to_output = {}
        for cdr in df[seq_col]:
            output = []
            for d in comparison_cdr_to_output:
                output.append(d.get(cdr, 0))
            self.cdr_to_output[cdr] = output

        df = df.set_index(seq_col)
        self.cdr_to_heavy = df[heavy_col].apply(self.heavy_to_idx.get).to_dict()
        self.cdr_to_light = df[light_col].apply(self.light_to_idx.get).to_dict()

    @property
    def n_eff(self):
        return len(self.cdr_seqs)

    @property
    def cdr_seqs(self):
        if self._training:
            return self.cdr_seqs_train
        else:
            return self.cdr_seqs_test

    def sequences_to_onehot(self, sequences, reverse=None, matching=None):
        num_seqs = len(sequences)
        max_seq_len = max(len(seq) for seq in sequences)

        seq_arr = torch.zeros(num_seqs, max_seq_len, len(self.alphabet))
        seq_mask = torch.zeros(num_seqs, max_seq_len, 1)
        light_arr = torch.zeros(num_seqs, max_seq_len, len(self.light_to_idx))
        heavy_arr = torch.zeros(num_seqs, max_seq_len, len(self.heavy_to_idx))

        for i, cdr in enumerate(sequences):
            for j, aa in enumerate(cdr):
                seq_arr[i, j, self.aa_dict[aa]] = 1.
                seq_mask[i, j, 0] = 1.
                light_arr[i, j, self.cdr_to_light[cdr]] = 1.
                heavy_arr[i, j, self.cdr_to_heavy[cdr]] = 1.

        input_arr = torch.cat([seq_arr, light_arr, heavy_arr], dim=-1)
        return {'input': input_arr, 'mask': seq_mask}

    def __getitem__(self, index):
        cdr_seqs = self.cdr_seqs
        if len(cdr_seqs) == 0:
            return None

        output_arr = torch.zeros(self.batch_size, len(self.comparisons))
        cdr_indices = np.random.randint(0, len(cdr_seqs), self.batch_size)
        seqs = list(cdr_seqs[cdr_indices])
        for i, seq in enumerate(seqs):
            for j, output in enumerate(self.cdr_to_output[seq]):
                output_arr[i, j] = output

        batch = self.sequences_to_onehot(seqs)
        batch['output'] = output_arr

        return batch
