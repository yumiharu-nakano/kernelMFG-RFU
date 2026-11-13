import unittest

import torch

from kernel_sot import sde_integrate_with_cost


class ConstantDrift(torch.nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward(self, _t, x):
        return torch.full_like(x, self.value)


class SDECostTests(unittest.TestCase):
    def test_cost_uses_interval_widths_on_nonuniform_grid(self):
        # For u=2 in one dimension over [0, 2],
        # (1/2) int_0^2 |u|^2 dt = 4 exactly.
        t_grid = torch.tensor([0.0, 0.1, 0.7, 2.0])
        _, cost = sde_integrate_with_cost(
            ConstantDrift(2.0),
            torch.zeros(5, 1),
            t_grid,
            sigma=0.0,
            t_subsample=3,
        )
        self.assertAlmostEqual(float(cost), 4.0, places=6)

    def test_nonpositive_time_subsample_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "t_subsample must be at least one"):
            sde_integrate_with_cost(
                ConstantDrift(1.0),
                torch.zeros(2, 1),
                torch.tensor([0.0, 1.0]),
                sigma=0.0,
                t_subsample=0,
            )

    def test_grid_requires_two_points(self):
        with self.assertRaisesRegex(ValueError, "at least two time points"):
            sde_integrate_with_cost(
                ConstantDrift(1.0),
                torch.zeros(2, 1),
                torch.tensor([0.0]),
                sigma=0.0,
            )


if __name__ == "__main__":
    unittest.main()
