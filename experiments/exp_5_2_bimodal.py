"""
Section 5.2 -- Schrödinger bridge: bimodal target (d = 2).

Reproduces Section 5.2 of the paper:
  mu_0 = delta_0,
  mu_1 = (1/2) N((2,2)^T, 0.25 I_2) + (1/2) N((-2,-2)^T, 0.25 I_2),
  sigma = 0.5, alpha = 1.0, M = 200, N = 64, lambda^{-1} = 0.01, 2000 epochs.

Outputs MMD^2 (eval) over multiple seeds and a sample-paths/terminal figure.
"""

import argparse
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kernel_sot import (
    DriftNet, mmd2_rf_ustat, sde_integrate, sde_integrate_with_cost,
    set_seed, ensure_dir,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--d", type=int, default=2)
    p.add_argument("--sigma", type=float, default=0.5)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--M", type=int, default=200)
    p.add_argument("--N", type=int, default=64)
    p.add_argument("--lam_inv", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--time_steps", type=int, default=20)
    p.add_argument("--hidden", type=int, nargs="+", default=[64, 32])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--output_dir", type=str, default="results")
    return p.parse_args()


def run_one(args, seed: int):
    set_seed(seed)
    centers = torch.tensor([[2.0, 2.0], [-2.0, -2.0]])

    def sample_nu(n):
        idx = torch.randint(0, 2, (n,))
        return centers[idx] + 0.5 * torch.randn(n, args.d)

    drift = DriftNet(args.d, hidden_dims=tuple(args.hidden))
    optimizer = optim.Adam(drift.parameters(), lr=args.lr)
    t_grid = torch.linspace(0.0, 1.0, args.time_steps)

    for ep in range(args.epochs):
        optimizer.zero_grad()
        x0 = torch.zeros(2 * args.N, args.d)
        x1, ctrl = sde_integrate_with_cost(
            drift, x0, t_grid, args.sigma, t_subsample=8)
        y = sample_nu(2 * args.N)
        mmd2 = mmd2_rf_ustat(x1, y, M=args.M, alpha=args.alpha)
        loss = ctrl * args.lam_inv + mmd2
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        x0 = torch.zeros(1000, args.d)
        x_path = sde_integrate(drift, x0, t_grid, args.sigma, return_path=True)
        x1 = x_path[-1]
        Y = sample_nu(1000)
        mmd2_eval = mmd2_rf_ustat(x1, Y, M=500, alpha=args.alpha).item()

    return drift, x_path, x1, Y, mmd2_eval


def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    print(f"Bimodal SBP (d={args.d}): {len(args.seeds)} seeds, "
          f"{args.epochs} epochs each\n")

    mmd2_evals = []
    last_run = None
    for s in args.seeds:
        t0 = time.time()
        drift, x_path, x1, Y, mmd2_eval = run_one(args, s)
        elapsed = time.time() - t0
        print(f"  seed={s}: MMD²={mmd2_eval:.4e}  ({elapsed:.0f}s)")
        mmd2_evals.append(mmd2_eval)
        last_run = (drift, x_path, x1, Y)

    mmd2_evals = np.array(mmd2_evals)
    print(f"\n=> MMD² = {mmd2_evals.mean():.4e} ± {mmd2_evals.std():.4e}")

    # Save JSON
    out_path = os.path.join(args.output_dir, "exp_5_2_bimodal.json")
    with open(out_path, "w") as f:
        json.dump({
            "args": vars(args),
            "mmd2_evals": mmd2_evals.tolist(),
            "mmd2_mean": float(mmd2_evals.mean()),
            "mmd2_std":  float(mmd2_evals.std()),
        }, f, indent=2)
    print(f"Saved: {out_path}")

    # Figure (last seed only, for visualization)
    if last_run is not None:
        drift, x_path, x1, Y = last_run
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        xp = x_path.numpy()
        for i in range(min(80, xp.shape[1])):
            axes[0].plot(xp[:, i, 0], xp[:, i, 1], alpha=0.08, color="blue", lw=0.5)
        axes[0].set_xlim(-5, 5); axes[0].set_ylim(-5, 5); axes[0].set_aspect("equal")
        axes[0].set_title(f"Sample paths (seed={args.seeds[-1]})")
        axes[1].scatter(Y[:, 0], Y[:, 1], s=2, alpha=0.3, label=r"target $\nu$")
        axes[1].scatter(x1[:, 0], x1[:, 1], s=2, alpha=0.3, label=r"$X_1$ (SBP)")
        axes[1].legend(markerscale=4); axes[1].set_aspect("equal")
        axes[1].set_title(rf"Terminal distribution, MMD$^2={mmd2_evals[-1]:.4e}$")
        plt.tight_layout()
        fig_path = os.path.join(args.output_dir, "exp_5_2_bimodal.png")
        plt.savefig(fig_path, dpi=150); plt.close()
        print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()
