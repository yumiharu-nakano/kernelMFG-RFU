"""
Section 5.3 -- Schrödinger bridge: high-dimensional Gaussian shift.

Reproduces Table 3 of the paper (5-seed mean ± std):
  mu_0 = delta_0,
  mu_1 = N(m, I_d) with m = (3, 0, ..., 0),
  sigma = 0.5, alpha = 1/d.

Hyperparameters per dimension (Table 2):
  d=10:  M=400,  N=80, hidden=(128, 64),  alpha=0.1,  lam_inv=1e-3,  epochs=4000
  d=50:  M=800,  N=80, hidden=(256,128),  alpha=0.02, lam_inv=5e-4,  epochs=5000
  d=100: M=1500, N=80, hidden=(512,256),  alpha=0.01, lam_inv=3e-4,  epochs=6000
"""

import argparse
import json
import os
import sys
import time
from typing import List

import numpy as np
import torch
from torch import optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kernel_sot import (
    DriftNet, mmd2_rf_ustat, sde_integrate, sde_integrate_with_cost,
    set_seed, ensure_dir,
)

# Per-dimension hyperparameters from Table 2 of the paper
CONFIGS = {
    10:  {"M": 400,  "hidden": (128, 64),  "alpha": 0.1,  "lam_inv": 1e-3, "epochs": 4000},
    50:  {"M": 800,  "hidden": (256, 128), "alpha": 0.02, "lam_inv": 5e-4, "epochs": 5000},
    100: {"M": 1500, "hidden": (512, 256), "alpha": 0.01, "lam_inv": 3e-4, "epochs": 6000},
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dims", type=int, nargs="+", default=[10, 50],
                   help="Subset of {10, 50, 100} to run")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--N", type=int, default=80)
    p.add_argument("--sigma", type=float, default=0.5)
    p.add_argument("--time_steps", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--output_dir", type=str, default="results")
    return p.parse_args()


def run_one(d, seed, args):
    cfg = CONFIGS[d]
    set_seed(seed)
    shift = torch.zeros(d); shift[0] = 3.0

    def sample_nu(n):
        return shift + torch.randn(n, d)

    drift = DriftNet(d, hidden_dims=cfg["hidden"])
    optimizer = optim.Adam(drift.parameters(), lr=args.lr)
    t_grid = torch.linspace(0.0, 1.0, args.time_steps)

    for ep in range(cfg["epochs"]):
        optimizer.zero_grad()
        x0 = torch.zeros(2 * args.N, d)
        x1, ctrl = sde_integrate_with_cost(
            drift, x0, t_grid, args.sigma, t_subsample=8)
        y = sample_nu(2 * args.N)
        mmd2 = mmd2_rf_ustat(x1, y, M=cfg["M"], alpha=cfg["alpha"])
        loss = ctrl * cfg["lam_inv"] + mmd2
        loss.backward()
        optimizer.step()

    n_eval = 1000 if d <= 50 else 500
    with torch.no_grad():
        x0 = torch.zeros(n_eval, d)
        x1 = sde_integrate(drift, x0, t_grid, args.sigma)
        Y = sample_nu(n_eval)
        m_est = x1.mean(0)
        std_mean = x1.std(0).mean().item()
        mmd2_eval = mmd2_rf_ustat(x1, Y, M=500, alpha=cfg["alpha"]).item()

    return {
        "E_X1_1": m_est[0].item(),
        "E_X1_rest": m_est[1:].mean().item(),
        "std_mean": std_mean,
        "mmd2_eval": mmd2_eval,
        "mean_err_sq": ((m_est - shift) ** 2).sum().item(),
    }


def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    all_results = {}
    for d in args.dims:
        if d not in CONFIGS:
            print(f"  d={d}: no config; skipping")
            continue
        print(f"\n--- d = {d} (config: {CONFIGS[d]}) ---")
        per_dim = []
        for s in args.seeds:
            t0 = time.time()
            info = run_one(d, s, args)
            elapsed = time.time() - t0
            print(f"  seed={s}: E[X1]_1={info['E_X1_1']:.3f} "
                  f"rest={info['E_X1_rest']:+.4f} std={info['std_mean']:.3f} "
                  f"MMD²={info['mmd2_eval']:.4e} ({elapsed:.0f}s)")
            per_dim.append(info)

        agg = {
            k + "_mean": float(np.mean([r[k] for r in per_dim]))
            for k in ["E_X1_1", "E_X1_rest", "std_mean", "mmd2_eval", "mean_err_sq"]
        }
        agg.update({
            k + "_std": float(np.std([r[k] for r in per_dim]))
            for k in ["E_X1_1", "E_X1_rest", "std_mean", "mmd2_eval", "mean_err_sq"]
        })
        agg["runs"] = per_dim
        all_results[str(d)] = agg

        print(f"  => E[X1]_1 = {agg['E_X1_1_mean']:.3f} ± {agg['E_X1_1_std']:.3f}")
        print(f"     rest    = {agg['E_X1_rest_mean']:+.4f} ± {agg['E_X1_rest_std']:.4f}"
              "   (= (1/(d-1)) sum_{k=2}^d E[X_1]_k, averaged over seeds)")
        print(f"     std     = {agg['std_mean_mean']:.3f} ± {agg['std_mean_std']:.3f}")
        print(f"     MMD²    = {agg['mmd2_eval_mean']:.4e} ± {agg['mmd2_eval_std']:.4e}")

    out_path = os.path.join(args.output_dir, "exp_5_3_gauss_shift.json")
    with open(out_path, "w") as f:
        json.dump({"args": vars(args), "configs": CONFIGS,
                   "results": all_results}, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
