"""
Neural network architectures for the controlled SDE drift.
"""

import torch
from torch import nn


class DriftNet(nn.Module):
    """
    Feedforward drift network u_theta(t, x) used in the controlled SDE
    dX_t = u_theta(t, X_t) dt + sigma dW_t.

    Architecture: linear -> ReLU -> LayerNorm stacked, with the final linear
    layer initialized with small weights so that the initial drift is near zero.

    This matches the architecture in Section 5 of the paper.
    """

    def __init__(self, d: int, hidden_dims=(64, 32)):
        super().__init__()
        layers = []
        prev = 1 + d  # time scalar + state vector
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.LayerNorm(h)]
            prev = h
        layers.append(nn.Linear(prev, d))
        self.net = nn.Sequential(*layers)
        with torch.no_grad():
            self.net[-1].weight.mul_(0.01)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, t_scalar: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        t_scalar : scalar tensor (a single time) or (B, 1) batch of times.
        x        : (B, d) state batch.

        Returns
        -------
        u(t, x)  : (B, d).
        """
        if t_scalar.dim() == 0:
            t_expanded = t_scalar.expand(x.size(0), 1)
        else:
            t_expanded = t_scalar
        return self.net(torch.cat([t_expanded, x], dim=1))
