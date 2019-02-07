import unittest

import torch
import torch.nn.functional as F

from functions import log_one_plus_exp, Normalize


class TestFunctions(unittest.TestCase):
    def test_log_one_plus_exp_range(self):
        x = torch.FloatTensor([-1000, -100, -10., -1., -0.5, -0.1, 0.0, 0.1, 0.5, 1., 10, 100, 1000])
        self.assertTrue(torch.allclose(
            -F.logsigmoid(-x),
            log_one_plus_exp(x)
        ))

    def test_log_one_plus_exp_randn(self):
        x = torch.randn(1000)
        self.assertTrue(torch.allclose(
            torch.log1p(torch.exp(x)),
            log_one_plus_exp(x)
        ))

    def test_normalize(self):
        self.assertTrue(Normalize.test())
