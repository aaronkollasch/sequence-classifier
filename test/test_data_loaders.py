import unittest

import numpy as np
import torch
import torch.nn.functional as F

from data_loaders import sequences_to_aligned_onehot


class TestFunctions(unittest.TestCase):
    def test_sequences_to_aligned_onehot(self):
        expect_arr = np.array([
            [
                [1., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
                [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
                [0., 1., 1., 0., 0., 0., 0., 1., 0., 0.],
                [0., 0., 0., 1., 0., 0., 0., 0., 0., 1.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
            ],
            [
                [1., 0., 0., 0., 0., 0., 0., 1., 1., 0.],
                [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                [0., 1., 1., 0., 0., 0., 1., 0., 0., 0.],
                [0., 0., 0., 1., 0., 0., 0., 0., 0., 1.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
            ],
            [
                [1., 0., 0., 0., 0., 0., 1., 1., 0., 0.],
                [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                [0., 1., 1., 0., 0., 1., 0., 0., 0., 1.],
                [0., 0., 0., 1., 0., 0., 0., 0., 1., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
            ],
            [
                [1., 0., 0., 0., 0., 1., 1., 0., 1., 0.],
                [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                [0., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 1., 0., 0., 0., 1., 0., 1.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
            ]
        ])
        expect_mask = np.array([
            [[[1., 1., 1., 1., 0., 0., 1., 1., 1., 1.]]],
            [[[1., 1., 1., 1., 0., 1., 1., 1., 1., 1.]]],
            [[[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]]],
            [[[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]]]
        ])
        seqs = ['ACCDBCAD', 'ACCDBCAAD', 'ACCDBCAADC', 'ACCDBCAADAD']
        char_map = {c: i for i, c in enumerate('ABCD-')}
        arr, mask = sequences_to_aligned_onehot(seqs, char_map, max_seq_len=10, gap_char='-', mask_gaps=True)
        print(arr)
        print(mask)
        self.assertTrue(np.allclose(expect_arr, arr.squeeze(2)) and np.allclose(expect_mask, mask))
