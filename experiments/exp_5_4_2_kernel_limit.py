"""
Section 5.4.2 -- Kernel U-statistic limit (M -> infinity).

Verifies Corollary 4.2: as M -> infinity, the RF U-statistic penalty converges to
the kernel U-statistic penalty bar_gamma_K^2(D), and both yield the same SBP solution.

Setup: d=10 Gaussian shift SBP. Compare:
  (A) Kernel U-stat (M = infinity, O(N^2) cost)
  (B) RF U-stat with M in {100, 400, 1600}.
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
    DriftNet, mmd2_kernel_ustat, mmd2_rf_ustat,
    sde_integrate, sde_integrate_with_cost, set_seed, ensure_dir,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--d", type=int, default=10)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--N", type=int, default=80)
    p.add_argument("--sigma", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=5000)
    p.add_argument("--time_steps", type=int, default=20)
    p.add_argument("--hidden", type=int, nargs="+", default=[128, 64])
    p.add_argument("--lam_inv", type=float, default=1e-3)
    p.add_argument("--M_values", type=int, nargs="+", default=[100, 400, 1600])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--output_dir", type=str, default="results")
    return p.parse_args()


def run_one(args, seed, method, M=None):
    set_seed(seed)
    shift = torch.zeros(args.d); shift[0] = 3.0

    def sample_nu(n):
        return shift + torch.randn(n, args.d)

    drift = DriftNet(args.d, hidden_dims=tuple(args.hidden))
    optimizer = optim.Adam(drift.parameters(), lr=1e-3)
    t_grid = torch.linspace(0.0, 1.0, args.time_steps)

    t0 = time.time()
    for ep in range(args.epochs):
        optimizer.zero_grad()
        x0 = torch.zeros(2 * args.N, args.d)
        x1, ctrl = sde_integrate_with_cost(drift, x0, t_grid, args.sigma, t_subsample=8)
        y = sample_nu(2 * args.N)
        if method == "kernel":
            mmd2 = mmd2_kernel_ustat(x1, y, args.alpha)
        else:
            mmd2 = mmd2_rf_ustat(x1, y, M=M, alpha=args.alpha)
        loss = ctrl * args.lam_inv + mmd2
        loss.backward()
        optimizer.step()
    train_time = time.time() - t0

    with torch.no_grad():
        x0 = torch.zeros(1000, args.d)
        x1 = sde_integrate(drift, x0, t_grid, args.sigma)
        Y = sample_nu(1000)
        m_est = x1.mean(0)
        std_mean = x1.std(0).mean().item()
        mmd2_eval_kernel = mmd2_kernel_ustat(x1, Y, args.alpha).item()

    return {
        "method": method, "M": M, "seed": seed,
        "E_X1_1": m_est[0].item(),
        "std_mean": std_mean,
        "mean_err_sq": ((m_est - shift) ** 2).sum().item(),
        "mmd2_eval_kernel": mmd2_eval_kernel,
        "iter_time_ms": train_time / args.epochs * 1000.0,
    }


def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    print(f"Kernel-limit comparison, d={args.d}, N={args.N}, "
          f"lambda^-1={args.lam_inv}, epochs={args.epochs}\n")

    settings = [("kernel", None, "Kernel $U$-stat ($M=\\infty$)")] + \
               [("rf", M, f"RF $U$-stat ($M={M}$)") for M in args.M_values]

    runs = {}
    for method, M, label in settings:
        print(f"--- {label} ---")
        runs[label] = []
        for s in args.seeds:
            t0 = time.time()
            info = run_one(args, s, method, M)
            elapsed = time.time() - t0
            print(f"  seed={s}: E[X1]_1={info['E_X1_1']:.3f}  "
                  f"|err|²={info['mean_err_sq']:.4f}  "
                  f"std={info['std_mean']:.3f}  "
                  f"MMD²(ker)={info['mmd2_eval_kernel']:.5f}  "
                  f"iter={info['iter_time_ms']:.1f}ms  ({elapsed:.0f}s)")
            runs[label].append(info)

    summary = {}
    for label, lst in runs.items():
        s = {}
        for k in ["mmd2_eval_kernel", "mean_err_sq", "E_X1_1",
                  "std_mean", "iter_time_ms"]:
            s[k + "_mean"] = float(np.mean([r[k] for r in lst]))
            s[k + "_std"]  = float(np.std([r[k] for r in lst]))
        summary[label] = s

    print("\n" + "=" * 60)
    for label, s in summary.items():
        print(f"\n{label}:")
        print(f"  E[X_1]_1   = {s['E_X1_1_mean']:.3f} ± {s['E_X1_1_std']:.3f}")
        print(f"  std (mean) = {s['std_mean_mean']:.3f} ± {s['std_mean_std']:.3f}")
        print(f"  |err|²     = {s['mean_err_sq_mean']:.3f} ± {s['mean_err_sq_std']:.3f}")
        print(f"  MMD²(ker)  = {s['mmd2_eval_kernel_mean']:.4e} ± {s['mmd2_eval_kernel_std']:.4e}")
        print(f"  iter time  = {s['iter_time_ms_mean']:.2f} ± {s['iter_time_ms_std']:.2f} ms")

    out_path = os.path.join(args.output_dir, "exp_5_4_2_kernel_limit.json")
    with open(out_path, "w") as f:
        json.dump({"args": vars(args), "runs": runs, "summary": summary}, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Bar chart
    labels = list(summary.keys())
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    means = [summary[l]["mmd2_eval_kernel_mean"] for l in labels]
    stds  = [summary[l]["mmd2_eval_kernel_std"] for l in labels]
    axes[0].bar(range(len(labels)), means, yerr=stds, capsize=5, color="C0", alpha=0.7)
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    axes[0].set_ylabel(r"Terminal MMD$^2$ (kernel U-stat eval)")
    axes[0].set_title("Solution quality")
    axes[0].grid(True, axis="y", alpha=0.3)

    times = [summary[l]["iter_time_ms_mean"] for l in labels]
    times_sd = [summary[l]["iter_time_ms_std"] for l in labels]
    axes[1].bar(range(len(labels)), times, yerr=times_sd, capsize=5, color="C1", alpha=0.7)
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    axes[1].set_ylabel("Per-iteration time (ms)")
    axes[1].set_title("Computational cost")
    axes[1].grid(True, axis="y", alpha=0.3)

    errs = [summary[l]["mean_err_sq_mean"] for l in labels]
    errs_sd = [summary[l]["mean_err_sq_std"] for l in labels]
    axes[2].bar(range(len(labels)), errs, yerr=errs_sd, capsize=5, color="C2", alpha=0.7)
    axes[2].set_xticks(range(len(labels)))
    axes[2].set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    axes[2].set_ylabel(r"$|\mathbb{E}[X_1]-m|^2$")
    axes[2].set_title("Mean tracking error")
    axes[2].grid(True, axis="y", alpha=0.3)

    plt.suptitle(rf"Kernel U-stat ($M=\infty$) vs RF U-stat in SBP "
                 rf"($d={args.d}$, $N={args.N}$, {len(args.seeds)} seeds)", fontsize=12)
    plt.tight_layout()
    fig_path = os.path.join(args.output_dir, "exp_5_4_2_kernel_limit.png")
    plt.savefig(fig_path, dpi=150); plt.close()
    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()
