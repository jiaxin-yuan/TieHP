"""A3B (design b): each target predicted from the state before its tie block; test-time π₀ mixture."""
import unittest

import torch

from model.losses import block_start, pre_block_index, pre_block_set_nll, set_nll
from train.train import pre_block_mixture


class PreBlockTests(unittest.TestCase):
    def setUp(self):
        """One case, 6 events: blocks {0}, {1, 2, 3}, {4}, {5}; 4 activities."""
        torch.manual_seed(0)
        self.dt = torch.tensor([[0., 1., 0., 0., 2., 1.]])
        self.logits = torch.randn(1, 6, 4)
        self.R = torch.ones(1, 6, 4, dtype=torch.bool)
        self.mask = torch.tensor([[False, False, True, True, True, True]])

    def test_index(self):
        """Block members 1–3 map to state 0; untied events to the previous position."""
        self.assertEqual(block_start(self.dt).tolist(), [[0, 1, 1, 1, 4, 5]])
        self.assertEqual(pre_block_index(self.dt).tolist(), [[0, 0, 0, 0, 3, 4]])

    def test_equals_a3_without_ties(self):
        """With every Δt > 0, A3B's loss is A3's set likelihood."""
        dt = torch.tensor([[0., 1., 1., 1., 2., 1.]])
        R = torch.rand(1, 6, 4) > 0.5
        R[..., 0] = True
        self.assertTrue(torch.allclose(pre_block_set_nll(self.logits, R, self.mask, dt), set_nll(self.logits, R, self.mask)))

    def test_in_block_targets_use_pre_block_state(self):
        """Targets 2 and 3 (inside block 1–3) are scored on the logits of position 0, not 1 and 2."""
        got = pre_block_set_nll(self.logits, self.R, self.mask, self.dt)
        moved = self.logits.clone()
        moved[0, 1], moved[0, 2] = self.logits[0, 0], self.logits[0, 0]   # targets 2, 3 read positions 1, 2 in set_nll
        self.assertTrue(torch.allclose(got, set_nll(moved, self.R, self.mask)))

    def test_mixture(self):
        """Test p_act sums to 1; with π₀ → 1 the emitted members of the current block get zero probability."""
        types = torch.tensor([1, 2, 3, 4, 1, 2])
        p_new = torch.softmax(self.logits[0, 2], -1)
        p = pre_block_mixture(p_new, self.logits[0], torch.tensor(50.), types, 1, 3)   # target 3, block started at 1
        self.assertAlmostEqual(float(p.sum()), 1.0, places=5)
        self.assertLess(float(p[[1, 2]].sum()), 1e-6)                     # activities 2, 3 (positions 1, 2) emitted


if __name__ == "__main__":
    unittest.main()
