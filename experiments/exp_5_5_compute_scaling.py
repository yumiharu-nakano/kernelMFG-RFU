"""
Section 5.5 -- Computational scaling: Kernel U-stat vs RF U-stat.

Reproduces Tables 6 and 7 of the paper:
- Per-iteration time of the SBP training loop (SDE integration + MMD + backprop)
  for varying batch size N.
- Stand-alone MMD computation time (forward + backward), in milliseconds.

The kernel U-stat has O(N^2) cost; the RF U-stat has O(NM) cost.
The crossover where RF becomes faster occurs around N ~ M.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch import optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kernel_sot import (
    DriftNet, mmd2_kernel_ustat, mmd2_rf_ustat,
    sde_integrate_with_cost, set_seed, ensure_dir,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["sbp", "mmd", "both"], default="both")
    p.add_argument("--d", type=int, default=10)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--N_values_sbp", type=int, nargs="+", default=[64, 128, 256, 500])
    p.add_argument("--N_values_mmd", type=int, nargs="+", default=[100, 200, 500, 1000, 2000])
    p.add_argument("--M", type=int, default=400, help="RF count for SBP timing")
    p.add_argument("--M_mmd", type=int, default=500, help="RF count for MMD-only timing")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--time_steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir", type=str, default="results")
    return p.parse_args()


def time_sbp_iter(args, N, method, n_iters=50):
    """Measure mean per-iteration time for one SBP training step."""
    set_seed(args.seed)
    drift = DriftNet(args.d, hidden_dims=(128, 64))
    optimizer = optim.Adam(drift.parameters(), lr=1e-3)
    t_grid = torch.linspace(0.0, 1.0, args.time_steps)
    shift = torch.zeros(args.d); shift[0] = 3.0
    sample_nu = lambda n: shift + torch.randn(n, args.d)
    sigma = 0.5

    # Warm-up
    for _ in range(3):
        optimizer.zero_grad()
        x0 = torch.zeros(2 * N, args.d)
        x1, ctrl = sde_integrate_with_cost(drift, x0, t_grid, sigma)
        y = sample_nu(2 * N)
        if method == "kernel":
            mmd2 = mmd2_kernel_ustat(x1, y, args.alpha)
        else:
            mmd2 = mmd2_rf_ustat(x1, y, M=args.M, alpha=args.alpha)
        (ctrl * 1e-3 + mmd2).backward()
        optimizer.step()

    # Measure
    t0 = time.time()
    for _ in range(n_iters):
        optimizer.zero_grad()
        x0 = torch.zeros(2 * N, args.d)
        x1, ctrl = sde_integrate_with_cost(drift, x0, t_grid, sigma)
        y = sample_nu(2 * N)
        if method == "kernel":
            mmd2 = mmd2_kernel_ustat(x1, y, args.alpha)
        else:
            mmd2 = mmd2_rf_ustat(x1, y, M=args.M, alpha=args.alpha)
        (ctrl * 1e-3 + mmd2).backward()
        optimizer.step()
    return (time.time() - t0) / n_iters * 1000.0  # ms


def time_mmd_only(args, N, method, n_iters=200):
    """Forward + backward time of the MMD estimator alone."""
    set_seed(args.seed)
    shift = torch.zeros(args.d); shift[0] = 3.0
    X = torch.randn(N, args.d, requires_grad=True)
    Y = shift + torch.randn(N, args.d)

    # Warm-up
    for _ in range(3):
        if method == "kernel":
            v = mmd2_kernel_ustat(X, Y, args.alpha)
        else:
            v = mmd2_rf_ustat(X, Y, M=args.M_mmd, alpha=args.alpha)
        if X.grad is not None:
            X.grad.zero_()
        v.backward()

    t0 = time.time()
    for _ in range(n_iters):
        if method == "kernel":
            v = mmd2_kernel_ustat(X, Y, args.alpha)
        else:
            v = mmd2_rf_ustat(X, Y, M=args.M_mmd, alpha=args.alpha)
        X.grad = None
        v.backward()
    return (time.time() - t0) / n_iters * 1000.0


def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    output = {"args": vars(args)}

    if args.mode in ("sbp", "both"):
        print("=" * 60)
        print(f"SBP per-iteration timing  (d={args.d}, RF M={args.M})")
        print("=" * 60)
        sbp_table = []
        header = f"{'N':>6} {'Kernel (ms)':>14} {'RF U-stat (ms)':>16} {'Speedup':>10}"
        print(header)
        for N in args.N_values_sbp:
            ker_times, rf_times = [], []
            for _ in range(args.repeats):
                ker_times.append(time_sbp_iter(args, N, "kernel"))
                rf_times.append(time_sbp_iter(args, N, "rf"))
            ker = float(np.mean(ker_times))
            rf  = float(np.mean(rf_times))
            speedup = ker / rf
            print(f"{N:>6d} {ker:>14.2f} {rf:>16.2f} {speedup:>9.2f}x")
            sbp_table.append({"N": N, "kernel_ms": ker, "rf_ms": rf, "speedup": speedup})
        output["sbp_per_iter"] = sbp_table

    if args.mode in ("mmd", "both"):
        print("\n" + "=" * 60)
        print(f"MMD-only forward+backward (d={args.d}, RF M={args.M_mmd})")
        print("=" * 60)
        mmd_table = []
        header = f"{'N':>6} {'Kernel (ms)':>14} {'RF U-stat (ms)':>16} {'Speedup':>10}"
        print(header)
        for N in args.N_values_mmd:
            ker_times, rf_times = [], []
            for _ in range(args.repeats):
                ker_times.append(time_mmd_only(args, N, "kernel"))
                rf_times.append(time_mmd_only(args, N, "rf"))
            ker = float(np.mean(ker_times))
            rf  = float(np.mean(rf_times))
            speedup = ker / rf
            print(f"{N:>6d} {ker:>14.3f} {rf:>16.3f} {speedup:>9.2f}x")
            mmd_table.append({"N": N, "kernel_ms": ker, "rf_ms": rf, "speedup": speedup})
        output["mmd_only"] = mmd_table

    out_path = os.path.join(args.output_dir, "exp_5_6_compute_scaling.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
