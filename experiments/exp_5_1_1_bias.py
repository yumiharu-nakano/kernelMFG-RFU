"""
Section 5.1.1 -- Bias comparison of estimators.

Reproduces Table 1 of the paper: under mu = nu = N(0, I_2), so that gamma_K^2 = 0,
the kernel U-stat and proposed RF U-stat have sample mean ~ 0,
while the standard RFF V-stat has positive bias of order Phi(0) / N.

Output:
  results/exp_5_1_1_bias.json
  prints a LaTeX table.

Default settings: d=2, N=200, M=200, alpha=1.0, T=2000 trials.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

# Make `kernel_sot` importable when running this script directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kernel_sot import (
    mmd2_kernel_ustat,
    mmd2_rf_ustat,
    mmd2_rff_vstat,
    set_seed,
    ensure_dir,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--d", type=int, default=2)
    p.add_argument("--N", type=int, default=200)
    p.add_argument("--M", type=int, default=200)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--trials", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir", type=str, default="results")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.output_dir)

    print(f"Bias comparison: d={args.d}, N={args.N}, M={args.M}, "
          f"alpha={args.alpha}, T={args.trials} trials\n")

    estimators = {
        "Kernel U-stat": lambda X, Y: mmd2_kernel_ustat(X, Y, args.alpha).item(),
        "RFF V-stat":    lambda X, Y: mmd2_rff_vstat(X, Y, args.M, args.alpha).item(),
        "RF U-stat":     lambda X, Y: mmd2_rf_ustat(X, Y, args.M, args.alpha).item(),
    }

    samples = {name: [] for name in estimators}
    for _ in range(args.trials):
        X = torch.randn(args.N, args.d)
        Y = torch.randn(args.N, args.d)  # mu = nu so true gamma_K^2 = 0
        for name, fn in estimators.items():
            samples[name].append(fn(X, Y))

    summary = {}
    for name, vals in samples.items():
        a = np.array(vals)
        summary[name] = {
            "mean": float(a.mean()),
            "std":  float(a.std(ddof=1)),
            "se_mean": float(a.std(ddof=1) / np.sqrt(args.trials)),
        }
        print(f"  {name:14s} mean={summary[name]['mean']:+.4e}  "
              f"std={summary[name]['std']:.4e}")

    out_path = os.path.join(args.output_dir, "exp_5_1_1_bias.json")
    with open(out_path, "w") as f:
        json.dump({"args": vars(args), "summary": summary}, f, indent=2)
    print(f"\nSaved: {out_path}")

    # LaTeX table for paper
    print("\n--- LaTeX table ---")
    print(r"\begin{tabular}{lcc}")
    print(r"\toprule")
    print(r"Estimator & Mean & Std \\")
    print(r"\midrule")
    for name in ["Kernel U-stat", "RFF V-stat", "RF U-stat"]:
        s = summary[name]
        print(f"{name} & ${s['mean']:+.1e}$ & ${s['std']:.1e}$ \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


if __name__ == "__main__":
    main()
