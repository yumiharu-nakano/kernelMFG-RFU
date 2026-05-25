"""
Section 5.4.1 -- Penalty parameter sweep (lambda-sweep).

Numerically verifies Theorem 4.1(i):
    sqrt(lambda_n) * gamma_K(Law(X_1^{(n)}), mu_1) -> 0  as lambda_n -> infinity.

Setup: d=10 Gaussian shift SBP (Section 5.3 settings).
Sweep lambda^{-1} in {1e-2, 3e-3, 1e-3, 3e-4, 1e-4} over multiple seeds.
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
    p.add_argument("--d", type=int, default=10)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--M", type=int, default=400)
    p.add_argument("--N", type=int, default=80)
    p.add_argument("--sigma", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=5000)
    p.add_argument("--time_steps", type=int, default=20)
    p.add_argument("--hidden", type=int, nargs="+", default=[128, 64])
    p.add_argument("--lam_invs", type=float, nargs="+",
                   default=[1e-2, 3e-3, 1e-3, 3e-4, 1e-4])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--output_dir", type=str, default="results")
    return p.parse_args()


def run_one(args, seed, lam_inv):
    set_seed(seed)
    shift = torch.zeros(args.d); shift[0] = 3.0

    def sample_nu(n):
        return shift + torch.randn(n, args.d)

    drift = DriftNet(args.d, hidden_dims=tuple(args.hidden))
    optimizer = optim.Adam(drift.parameters(), lr=1e-3)
    t_grid = torch.linspace(0.0, 1.0, args.time_steps)

    last_ctrl = 0.0
    for ep in range(args.epochs):
        optimizer.zero_grad()
        x0 = torch.zeros(2 * args.N, args.d)
        x1, ctrl = sde_integrate_with_cost(drift, x0, t_grid, args.sigma, t_subsample=8)
        y = sample_nu(2 * args.N)
        mmd2 = mmd2_rf_ustat(x1, y, M=args.M, alpha=args.alpha)
        loss = ctrl * lam_inv + mmd2
        loss.backward()
        optimizer.step()
        last_ctrl = ctrl.item()

    with torch.no_grad():
        x0 = torch.zeros(1000, args.d)
        x1 = sde_integrate(drift, x0, t_grid, args.sigma)
        Y = sample_nu(1000)
        m_est = x1.mean(0)
        mmd2_eval = mmd2_rf_ustat(x1, Y, M=500, alpha=args.alpha).item()

    return {
        "mmd2_eval": mmd2_eval,
        "mean_err_sq": ((m_est - shift) ** 2).sum().item(),
        "ctrl_final": last_ctrl,
        "E_X1_1": m_est[0].item(),
    }


def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    print(f"Lambda sweep, d={args.d}, M={args.M}, N={args.N}, "
          f"epochs={args.epochs}, seeds={args.seeds}\n")

    results = {}
    t_total = time.time()
    for lam_inv in args.lam_invs:
        print(f"--- lambda^-1 = {lam_inv:.0e} ---")
        runs = []
        for s in args.seeds:
            t0 = time.time()
            info = run_one(args, s, lam_inv)
            elapsed = time.time() - t0
            print(f"  seed={s}: E[X1]_1={info['E_X1_1']:.3f}  "
                  f"|err|²={info['mean_err_sq']:.4f}  "
                  f"MMD²={info['mmd2_eval']:.5f}  ctrl={info['ctrl_final']:.2f}  "
                  f"({elapsed:.0f}s)")
            runs.append(info)
        agg = {
            "lam_inv": lam_inv,
            "runs": runs,
            **{k + "_mean": float(np.mean([r[k] for r in runs]))
               for k in ["mmd2_eval", "mean_err_sq", "ctrl_final", "E_X1_1"]},
            **{k + "_std": float(np.std([r[k] for r in runs]))
               for k in ["mmd2_eval", "mean_err_sq", "ctrl_final", "E_X1_1"]},
        }
        results[f"{lam_inv:.0e}"] = agg
        print(f"  => E[X1]_1     = {agg['E_X1_1_mean']:.3f} ± {agg['E_X1_1_std']:.3f}")
        print(f"     MMD²        = {agg['mmd2_eval_mean']:.4e} ± {agg['mmd2_eval_std']:.4e}")
        print(f"     |E[X1]-m|²  = {agg['mean_err_sq_mean']:.4f} ± {agg['mean_err_sq_std']:.4f}")
        print(f"     ctrl cost   = {agg['ctrl_final_mean']:.2f} ± {agg['ctrl_final_std']:.2f}\n")
    print(f"Total: {time.time() - t_total:.0f}s")

    # Save JSON (per-dimension filename to avoid overwrite across sweeps)
    out_path = os.path.join(args.output_dir, f"exp_5_4_1_lambda_sweep_d{args.d}.json")
    with open(out_path, "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    print(f"Saved: {out_path}")

    # ----- LaTeX table (matches Table 5.5 of the paper) -----
    print("\n" + "=" * 60)
    print("LaTeX table for paper (Table 5.5):")
    print("=" * 60)
    print(r"""
\begin{tabular}{ccccc}
\toprule
$\lambda^{-1}$ & $\EE[X_1]_1$ & MMD${}^2$ & $|\EE[X_1]-m|^2$ & Control cost \\
\midrule""")
    for li in args.lam_invs:
        r = results[f"{li:.0e}"]
        exponent = int(np.floor(np.log10(max(r['mmd2_eval_mean'], 1e-300))))
        mantissa_mean = r['mmd2_eval_mean'] / (10.0 ** exponent)
        mantissa_std  = r['mmd2_eval_std']  / (10.0 ** exponent)
        # Lambda label
        if abs(np.log10(li) - round(np.log10(li))) < 1e-9:
            lam_str = f"10^{{{int(round(np.log10(li)))}}}"
        else:
            mantissa = li * 10 ** (-int(np.floor(np.log10(li))))
            lam_str = f"{mantissa:.0f}\\times 10^{{{int(np.floor(np.log10(li)))}}}"
        print(f"${lam_str}$ & "
              f"${r['E_X1_1_mean']:.2f}\\pm {r['E_X1_1_std']:.2f}$ & "
              f"$({mantissa_mean:.1f}\\pm {mantissa_std:.1f})\\times 10^{{{exponent}}}$ & "
              f"${r['mean_err_sq_mean']:.3f}\\pm {r['mean_err_sq_std']:.3f}$ & "
              f"${r['ctrl_final_mean']:.1f}\\pm {r['ctrl_final_std']:.1f}$ \\\\")
    print(r"""\bottomrule
\end{tabular}""")

    # Plots
    lam_invs = np.array(args.lam_invs)
    mmd2_means = np.array([results[f"{li:.0e}"]["mmd2_eval_mean"] for li in lam_invs])
    mmd2_stds  = np.array([results[f"{li:.0e}"]["mmd2_eval_std"] for li in lam_invs])
    err_means  = np.array([results[f"{li:.0e}"]["mean_err_sq_mean"] for li in lam_invs])
    err_stds   = np.array([results[f"{li:.0e}"]["mean_err_sq_std"] for li in lam_invs])
    ctrl_means = np.array([results[f"{li:.0e}"]["ctrl_final_mean"] for li in lam_invs])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].errorbar(lam_invs, mmd2_means, yerr=mmd2_stds, marker="o", capsize=4)
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].invert_xaxis()
    axes[0].set_xlabel(r"$\lambda^{-1}$"); axes[0].set_ylabel(r"MMD$^2$")
    axes[0].set_title(r"Terminal MMD$^2$ vs $\lambda^{-1}$"); axes[0].grid(True, alpha=0.3)

    axes[1].errorbar(lam_invs, err_means, yerr=err_stds, marker="s", capsize=4, color="C1")
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].invert_xaxis()
    axes[1].set_xlabel(r"$\lambda^{-1}$"); axes[1].set_ylabel(r"$|\mathbb{E}[X_1]-m|^2$")
    axes[1].set_title(r"Mean error vs $\lambda^{-1}$"); axes[1].grid(True, alpha=0.3)

    axes[2].plot(mmd2_means, ctrl_means, marker="D", color="C2")
    for i, li in enumerate(lam_invs):
        axes[2].annotate(f"{li:.0e}", (mmd2_means[i], ctrl_means[i]),
                         textcoords="offset points", xytext=(5, 5), fontsize=8)
    axes[2].set_xscale("log")
    axes[2].set_xlabel(r"MMD$^2$"); axes[2].set_ylabel("Control cost")
    axes[2].set_title(r"Cost-vs-fidelity trade-off"); axes[2].grid(True, alpha=0.3)

    plt.suptitle(rf"Penalty sweep, $d={args.d}$ Gaussian shift "
                 rf"({len(args.seeds)} seeds, {args.epochs} epochs)", fontsize=12)
    plt.tight_layout()
    fig_path = os.path.join(args.output_dir, f"exp_5_4_1_lambda_sweep_d{args.d}.png")
    plt.savefig(fig_path, dpi=150); plt.close()
    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()
