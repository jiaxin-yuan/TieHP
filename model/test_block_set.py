"""A3M (design b′): activity set likelihood only on Δt > 0 targets; time loss unchanged."""
import unittest

import torch

from model.losses import block_set_nll, set_nll
from model.model import Transformer
from train.train import VARIANTS, batch_loss, model_config


class BlockSetTests(unittest.TestCase):
    def setUp(self):
        """Two cases, 4 events, 3 activities; case 0 has a tie block at positions 1–2."""
        torch.manual_seed(0)
        self.logits = torch.randn(2, 4, 3, requires_grad=True)
        self.R = torch.zeros(2, 4, 3, dtype=torch.bool)
        self.R[:, :, 0] = True
        self.R[0, 1, 1] = True                                    # ambiguous block start
        self.dt = torch.tensor([[0., 1., 0., 2.], [0., 1., 3., 1.]])
        self.mask = torch.tensor([[False, True, True, True], [False, True, True, True]])

    def test_equals_set_nll_without_zero_gap_targets(self):
        """block_set_nll = set_nll with Δt = 0 targets removed from the mask."""
        got = block_set_nll(self.logits, self.R, self.mask, self.dt)
        want = set_nll(self.logits, self.R, self.mask & (self.dt > 0))
        self.assertTrue(torch.equal(got, want))
        self.assertLess(float(got), float(set_nll(self.logits, self.R, self.mask)))

    def test_zero_gap_target_gets_no_gradient(self):
        """The output predicting the Δt = 0 target (case 0, position 2) has zero gradient."""
        block_set_nll(self.logits, self.R, self.mask, self.dt).backward()
        self.assertTrue(torch.all(self.logits.grad[0, 1] == 0))   # position 1 predicts target 2 (Δt = 0)
        self.assertTrue(torch.any(self.logits.grad[0, 0] != 0))   # position 0 predicts target 1 (Δt > 0)

    def test_trainer_time_loss_matches_a3(self):
        """A3M and A3 share model and time loss; only the activity loss differs."""
        self.assertEqual(VARIANTS["A3M"][0], VARIANTS["A3"][0])
        model = Transformer(**model_config("A3", "2k", 3)).eval()
        types = torch.tensor([[1, 2, 3, 1], [1, 3, 2, 1]])
        b = {"types": types, "event_time": self.dt.cumsum(1), "dt": self.dt, "R": self.R,
             "target_mask": self.mask}
        a3 = batch_loss(model, b, VARIANTS["A3"][1], 5)
        a3m = batch_loss(model, b, VARIANTS["A3M"][1], 5)
        self.assertTrue(torch.equal(a3[1], a3m[1]))               # time loss identical
        self.assertEqual(a3[2], a3m[2])                            # same target count
        self.assertFalse(torch.equal(a3[0], a3m[0]))              # activity loss differs


if __name__ == "__main__":
    unittest.main()
