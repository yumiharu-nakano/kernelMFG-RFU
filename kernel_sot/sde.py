"""
Euler-Maruyama integration of the controlled SDE
    dX_t = b(t, X_t) dt + u_theta(t, X_t) dt + sigma dW_t,
where b is a (possibly zero) reference drift and u_theta is the trained control.
"""

from typing import Callable, Optional

import torch


def sde_integrate(
    drift_net,
    x0: torch.Tensor,
    t_grid: torch.Tensor,
    sigma: float,
    return_path: bool = False,
    ref_drift: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
):
    """
    Integrate dX_t = (ref_drift + drift_net) dt + sigma dW_t over t in [0, 1].

    Parameters
    ----------
    drift_net  : callable (t_scalar, x) -> drift (B, d). Typically a DriftNet.
    x0         : (B, d) initial samples.
    t_grid     : 1D tensor of time grid points (length T).
    sigma      : diffusion coefficient.
    return_path: if True, return full path (T, B, d); else only X_1.
    ref_drift  : optional reference drift function (e.g., -nabla V for Langevin).

    Returns
    -------
    X_1 (B, d) or full path (T, B, d).
    """
    dt_all = t_grid[1:] - t_grid[:-1]
    sqrt_dt_all = dt_all.sqrt()
    x = x0
    if return_path:
        path = [x]
    for i in range(len(dt_all)):
        u = drift_net(t_grid[i], x)
        if ref_drift is not None:
            u = u + ref_drift(t_grid[i], x)
        dW = torch.randn_like(x) * sqrt_dt_all[i]
        x = x + u * dt_all[i] + sigma * dW
        if return_path:
            path.append(x)
    if return_path:
        return torch.stack(path, dim=0)
    return x


def sde_integrate_with_cost(
    drift_net,
    x0: torch.Tensor,
    t_grid: torch.Tensor,
    sigma: float,
    t_subsample: int = 10,
    ref_drift: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
):
    """
    Same as sde_integrate, but also returns an unbiased estimate of the empirical
    control cost
        (1/B) sum_i (1/T) sum_t |u_theta(t, X_t^{(i)})|^2
    where the time-average uses t_subsample randomly chosen time indices for
    memory efficiency. Returns (X_1, control_cost).

    Note: when ref_drift is given, the cost is still computed for u_theta only
    (the controlled part), not the reference drift.
    """
    T = len(t_grid)
    dt_all = t_grid[1:] - t_grid[:-1]
    sqrt_dt_all = dt_all.sqrt()
    cost_indices = set(torch.randperm(T - 1)[:t_subsample].tolist())

    x = x0
    control_sum = torch.tensor(0.0, device=x0.device)
    count = 0
    for i in range(T - 1):
        u = drift_net(t_grid[i], x)
        if i in cost_indices:
            control_sum = control_sum + (u ** 2).sum(-1).mean()
            count += 1
        full_drift = u
        if ref_drift is not None:
            full_drift = full_drift + ref_drift(t_grid[i], x)
        dW = torch.randn_like(x) * sqrt_dt_all[i]
        x = x + full_drift * dt_all[i] + sigma * dW
    control_cost = control_sum / max(count, 1)
    return x, control_cost
