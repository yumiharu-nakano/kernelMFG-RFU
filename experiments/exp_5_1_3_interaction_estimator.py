"""
Section 5.1.3 (sbp_h02) -- Microbenchmark for the interaction estimator
                          $\\hat{\\mathcal{R}}_{M,N}$ (Theorem 3.3).

Setup (mirrors Sections 5.1.1-5.1.2 for the terminal estimator):
  - target distribution: nu = N(0, I_d) in dimension d
  - kernel: W(x, y) = exp(-alpha |x - y|^2) (Gaussian, Psi(0) = 1)
  - analytic value: R[nu] = (1/2) * (1 + 4 alpha)^(-d/2)

Verifications:
  (a) Bias check (Theorem 3.3(i)): grand mean of hat_R over many trials
      matches the analytic value, while the V-statistic counterpart
      shows the predicted bias 1/(2N).
  (b) Variance decomposition (Theorem 3.3(iii)): empirical Var(hat_R)
      across (M, N) grid is well-fit by c1/M + c2/N.

The setup is analogous to Section 5.1.1 (bias) and Section 5.1.2 (variance)
for the terminal estimator: same target, same kernel, same trial structure,
but applied to the self-interaction estimator hat_R_{M,N}[hat_nu^N] instead.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kernel_sot import interaction_rf_ustat, interaction_v_stat, set_seed, ensure_dir


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--d", type=int, default=2)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--n_trials_bias", type=int, default=2000,
                   help="Number of trials for the bias check (Table 5.1 analogue).")
    p.add_argument("--N_bias", type=int, default=200,
                   help="Sample size for the bias check.")
    p.add_argument("--M_bias", type=int, default=500,
                   help="Random-feature count for the bias check.")
    p.add_argument("--Ns", type=int, nargs="+",
                   default=[50, 100, 200, 500, 1000],
                   help="Sample sizes for the variance grid.")
    p.add_argument("--Ms", type=int, nargs="+",
                   default=[20, 50, 100, 500, 1000, 5000],
                   help="Random-feature counts for the variance grid.")
    p.add_argument("--n_trials_var", type=int, default=500,
                   help="Number of trials per cell for the variance grid.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir", type=str, default="results")
    return p.parse_args()


def analytic_R(alpha, d):
    """Closed form: R[N(0, I_d)] = (1/2)(1 + 4*alpha)^(-d/2)."""
    return 0.5 * (1.0 + 4.0 * alpha) ** (-d / 2.0)


def sample_nu(N, d):
    return torch.randn(N, d)


def run_bias_check(args):
    R_star = analytic_R(args.alpha, args.d)
    print(f"\n=== Bias check (target N(0, I_{args.d}), alpha={args.alpha}) ===")
    print(f"Analytic R[nu] = (1/2)(1 + 4*alpha)^(-d/2) = {R_star:.6f}")
    set_seed(args.seed)
    vals_u, vals_v = [], []
    for _ in range(args.n_trials_bias):
        X = sample_nu(args.N_bias, args.d)
        vals_u.append(interaction_rf_ustat(X, args.M_bias, args.alpha).item())
        vals_v.append(interaction_v_stat(X, args.M_bias, args.alpha).item())
    u = np.array(vals_u); v = np.array(vals_v)
    mean_u = u.mean(); se_u = u.std(ddof=1) / np.sqrt(len(u))
    mean_v = v.mean(); se_v = v.std(ddof=1) / np.sqrt(len(v))
    bias_u = mean_u - R_star
    bias_v = mean_v - R_star
    bias_v_theoretical = 1.0 / (2.0 * args.N_bias)   # = Psi(0) / (2N) for Gaussian
    print(f"  hat_R_U-stat:  mean = {mean_u:.6f} ± {se_u:.6f}  bias = {bias_u:+.6f}")
    print(f"  hat_R_V-stat:  mean = {mean_v:.6f} ± {se_v:.6f}  bias = {bias_v:+.6f}")
    print(f"  Theoretical V-stat bias = 1/(2N) = {bias_v_theoretical:.6f}")
    return {
        "R_analytic": R_star,
        "u_stat_mean": float(mean_u), "u_stat_se": float(se_u),
        "u_stat_bias": float(bias_u),
        "v_stat_mean": float(mean_v), "v_stat_se": float(se_v),
        "v_stat_bias": float(bias_v),
        "v_stat_bias_theoretical": bias_v_theoretical,
        "N": args.N_bias, "M": args.M_bias, "n_trials": args.n_trials_bias,
    }


def run_variance_grid(args):
    print(f"\n=== Variance scaling (Var(hat_R) vs (M, N), trials={args.n_trials_var}) ===")
    var_grid = {}
    for N in args.Ns:
        var_grid[N] = {}
        for M in args.Ms:
            set_seed(args.seed + 10 * N + M)
            vals = []
            for _ in range(args.n_trials_var):
                X = sample_nu(N, args.d)
                vals.append(interaction_rf_ustat(X, M, args.alpha).item())
            v = np.var(vals, ddof=1)
            var_grid[N][M] = float(v)
        print(f"  N={N}: " + "  ".join(
            f"M={M}: Var={var_grid[N][M]:.3e}" for M in args.Ms))
    # Fit Var = c1/M + c2/N via non-negative least squares
    from scipy.optimize import nnls
    rows, b = [], []
    for N in args.Ns:
        for M in args.Ms:
            rows.append([1.0 / M, 1.0 / N])
            b.append(var_grid[N][M])
    A = np.array(rows); b_vec = np.array(b)
    coef, _ = nnls(A, b_vec)
    c1, c2 = coef
    print(f"\n  Best fit Var(hat_R_{{M,N}}) ≈ {c1:.4f}/M + {c2:.4f}/N")
    # Goodness of fit
    pred = A @ coef
    ss_res = ((b_vec - pred) ** 2).sum()
    ss_tot = ((b_vec - b_vec.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    print(f"  R^2 = {r2:.4f}")
    return {
        "var_grid": {str(N): var_grid[N] for N in args.Ns},
        "fit_c1_per_M": float(c1),
        "fit_c2_per_N": float(c2),
        "r2": float(r2),
        "Ns": args.Ns, "Ms": args.Ms, "n_trials": args.n_trials_var,
    }


def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    print(f"Microbench for interaction estimator (Theorem 3.3)")
    print(f"  d = {args.d}, alpha = {args.alpha}")

    bias = run_bias_check(args)
    var = run_variance_grid(args)

    out_path = os.path.join(args.output_dir,
                             f"exp_5_1_4_interaction_d{args.d}.json")
    with open(out_path, "w") as f:
        json.dump({"args": vars(args),
                   "bias_check": bias,
                   "variance_grid": var}, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
