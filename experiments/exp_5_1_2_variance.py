"""
Section 5.1.2 -- Variance scaling of the RF U-statistic estimator.

Reproduces Table 2 / Figure (variance vs 1/M and 1/N) of the paper.
Verifies the variance bound of Theorem 3.1(iii):
    Var(hat_gamma_{M,N}^2) = O(1/M) + O(1/N).

Setup: mu = N(0, I_d), nu = N(m, I_d) with m_1 = 1 (alternative case to avoid
degenerate U-stat regime), d=10, alpha=1/d.
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kernel_sot import mmd2_rf_ustat, set_seed, ensure_dir


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--d", type=int, default=10)
    p.add_argument("--shift1", type=float, default=1.0,
                   help="First-coordinate shift of nu = N(m, I_d)")
    p.add_argument("--alpha", type=float, default=None, help="default 1/d")
    p.add_argument("--trials", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir", type=str, default="results")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.output_dir)
    alpha = args.alpha if args.alpha is not None else 1.0 / args.d

    M_VALUES = [50, 100, 200, 500, 1000, 2000, 5000]
    N_VALUES = [50, 100, 200, 500, 1000]
    shift = torch.zeros(args.d); shift[0] = args.shift1

    print(f"Variance scaling: d={args.d}, alpha={alpha}, "
          f"mu=N(0,I), nu=N(m,I) with m_1={args.shift1}, T={args.trials} trials/cell\n")

    @torch.no_grad()
    def one_estimate(M, N):
        X = torch.randn(N, args.d)
        Y = shift + torch.randn(N, args.d)
        return mmd2_rf_ustat(X, Y, M=M, alpha=alpha).item()

    results = {}
    t0 = time.time()
    for M in M_VALUES:
        for N in N_VALUES:
            samples = np.array([one_estimate(M, N) for _ in range(args.trials)])
            mean = float(samples.mean())
            var  = float(samples.var(ddof=1))
            results[(M, N)] = {"mean": mean, "var": var}
            print(f"  M={M:>5d} N={N:>5d}  mean={mean:+.4e}  var={var:.3e}")
    print(f"\nTotal: {time.time() - t0:.0f}s")

    # NNLS fit Var = c1/M + c2/N
    rows = [(1.0 / M, 1.0 / N, results[(M, N)]["var"])
            for M in M_VALUES for N in N_VALUES]
    A = np.array([(r[0], r[1]) for r in rows])
    y = np.array([r[2] for r in rows])
    try:
        from scipy.optimize import nnls
        coef, _ = nnls(A, y)
    except ImportError:
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        coef = np.maximum(coef, 0)
    c1, c2 = float(coef[0]), float(coef[1])
    print(f"\nFit: Var ≈ {c1:.4f}/M + {c2:.4f}/N")

    # ---- save JSON ----
    json_path = os.path.join(args.output_dir, "exp_5_1_2_variance.json")
    with open(json_path, "w") as f:
        json.dump({
            "args": vars(args),
            "alpha": alpha,
            "M_values": M_VALUES, "N_values": N_VALUES,
            "results": {f"M={M}_N={N}": results[(M, N)]
                        for M in M_VALUES for N in N_VALUES},
            "fit_c1_per_M": c1, "fit_c2_per_N": c2,
        }, f, indent=2)
    print(f"Saved: {json_path}")

    # ---- plots ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    cmap = plt.cm.viridis(np.linspace(0, 0.85, len(N_VALUES)))
    for i, N in enumerate(N_VALUES):
        Ms = np.array(M_VALUES)
        vars_ = np.array([results[(M, N)]["var"] for M in M_VALUES])
        ax.plot(1.0 / Ms, vars_, marker="o", color=cmap[i], label=f"N={N}", lw=1.3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$1/M$"); ax.set_ylabel(r"$\mathrm{Var}(\hat{\gamma}^2_{M,N})$")
    ax.set_title(r"Variance vs $1/M$ at fixed $N$")
    ax.grid(True, which="both", alpha=0.3); ax.legend()

    ax = axes[1]
    cmap2 = plt.cm.plasma(np.linspace(0, 0.85, len(M_VALUES)))
    for i, M in enumerate(M_VALUES):
        Ns = np.array(N_VALUES)
        vars_ = np.array([results[(M, N)]["var"] for N in N_VALUES])
        ax.plot(1.0 / Ns, vars_, marker="s", color=cmap2[i], label=f"M={M}", lw=1.3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$1/N$"); ax.set_ylabel(r"$\mathrm{Var}(\hat{\gamma}^2_{M,N})$")
    ax.set_title(r"Variance vs $1/N$ at fixed $M$")
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8, ncol=2)

    plt.suptitle(rf"Variance scaling, $d={args.d}$, $\alpha=1/d$, "
                 rf"fit Var $\approx {c1:.4f}/M + {c2:.4f}/N$", fontsize=12)
    plt.tight_layout()
    fig_path = os.path.join(args.output_dir, "exp_5_1_2_variance.png")
    plt.savefig(fig_path, dpi=150); plt.close()
    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()
