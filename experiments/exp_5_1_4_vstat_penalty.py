"""
Section 5.1.4 -- RFF V-statistic penalty vs RF U-statistic penalty for SBP.

Addresses the reviewer's concern (Major Issue 2): the paper claims that
"unbiasedness is essential" for the penalty method, but provides no direct
SBP-training comparison demonstrating the failure of the biased V-statistic.

Setup: identical to Section 5.3 at d=10 (Gaussian shift target).
We train two SBPs:
  (i)  U-stat:  penalty = mmd2_rf_ustat   (unbiased)
  (ii) V-stat:  penalty = mmd2_rff_vstat  (biased, standard RFF)
All other hyperparameters are identical:
  M=400, N=80, hidden=(128, 64), alpha=0.1, epochs=4000, lam_inv=1e-3.

Evaluation: the trained drift's terminal distribution is summarized by
  - E[X_1]_1 (first coordinate, target = 3.0)
  - mean_err_sq = |E[X_1] - m|^2
  - MMD^2 computed with kernel U-stat (fair, unbiased measurement)
  - "spurious_penalty" -- the V-stat estimate of MMD^2(mu_1, mu_1) on a
    fresh independent batch (this is the floor the V-stat penalty cannot
    descend below).

Output: results/exp_5_1_3_vstat_penalty.json and a LaTeX table.
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
    DriftNet, mmd2_kernel_ustat, mmd2_rf_ustat, mmd2_rff_vstat,
    sde_integrate, sde_integrate_with_cost, set_seed, ensure_dir,
)

# Match Section 5.3 d=10 setup exactly (baseline).
DEFAULT_CFG = {"d": 10, "M": 400, "N": 80, "hidden": (128, 64),
               "alpha": 0.1, "lam_inv": 1e-3, "epochs": 4000,
               "sigma": 0.5, "time_steps": 20, "lr": 1e-3}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--N", type=int, default=DEFAULT_CFG["N"],
                   help="Batch size; overrides default 80 to test small-N regime")
    p.add_argument("--lam_inv", type=float, default=DEFAULT_CFG["lam_inv"],
                   help="1/lambda; overrides default 1e-3 to test large-lambda regime")
    p.add_argument("--epochs", type=int, default=DEFAULT_CFG["epochs"])
    p.add_argument("--tag", type=str, default=None,
                   help="Suffix added to output filename, e.g. 'N20_lam1e-5'")
    p.add_argument("--output_dir", type=str, default="results")
    return p.parse_args()


def build_cfg(args):
    cfg = dict(DEFAULT_CFG)
    cfg["N"] = args.N
    cfg["lam_inv"] = args.lam_inv
    cfg["epochs"] = args.epochs
    return cfg


def run_one(seed, method, cfg):
    """method: 'ustat' or 'vstat'."""
    d, N, sigma, T = cfg["d"], cfg["N"], cfg["sigma"], cfg["time_steps"]
    set_seed(seed)
    shift = torch.zeros(d); shift[0] = 3.0

    def sample_nu(n):
        return shift + torch.randn(n, d)

    drift = DriftNet(d, hidden_dims=cfg["hidden"])
    optimizer = optim.Adam(drift.parameters(), lr=cfg["lr"])
    t_grid = torch.linspace(0.0, 1.0, T)

    penalty_fn = mmd2_rf_ustat if method == "ustat" else mmd2_rff_vstat

    for ep in range(cfg["epochs"]):
        optimizer.zero_grad()
        x0 = torch.zeros(N, d)
        x1, ctrl = sde_integrate_with_cost(drift, x0, t_grid, sigma, t_subsample=8)
        y = sample_nu(N)
        mmd2 = penalty_fn(x1, y, M=cfg["M"], alpha=cfg["alpha"])
        loss = ctrl * cfg["lam_inv"] + mmd2
        loss.backward()
        optimizer.step()

    # --- evaluation on a large fresh batch -----------------------------
    n_eval = 1000
    with torch.no_grad():
        x0_eval = torch.zeros(n_eval, d)
        x1_eval = sde_integrate(drift, x0_eval, t_grid, sigma)
        Y_eval = sample_nu(n_eval)
        m_est = x1_eval.mean(0)
        mmd2_eval = mmd2_kernel_ustat(x1_eval, Y_eval, cfg["alpha"]).item()
        # Spurious V-stat floor measured at SAME N as in training.
        Y_a, Y_b = sample_nu(N), sample_nu(N)
        spurious_v = mmd2_rff_vstat(Y_a, Y_b, M=cfg["M"], alpha=cfg["alpha"]).item()
        spurious_u = mmd2_rf_ustat(Y_a, Y_b, M=cfg["M"], alpha=cfg["alpha"]).item()

    return {
        "method": method, "seed": seed,
        "E_X1_1": m_est[0].item(),
        "mean_err_sq": ((m_est - shift) ** 2).sum().item(),
        "mmd2_eval_kernel_ustat": mmd2_eval,
        "spurious_vstat_floor": spurious_v,
        "spurious_ustat_floor": spurious_u,
    }


def aggregate(runs, keys):
    out = {}
    for k in keys:
        vals = [r[k] for r in runs]
        out[f"{k}_mean"] = float(np.mean(vals))
        out[f"{k}_std"] = float(np.std(vals))
    return out


def main():
    args = parse_args()
    cfg = build_cfg(args)
    ensure_dir(args.output_dir)
    tag = args.tag or f"N{cfg['N']}_lam{cfg['lam_inv']:.0e}"
    print(f"V-stat vs U-stat penalty SBP comparison [{tag}], "
          f"d={cfg['d']}, N={cfg['N']}, lam_inv={cfg['lam_inv']}, "
          f"epochs={cfg['epochs']}, seeds={args.seeds}\n")

    per_method = {"ustat": [], "vstat": []}
    keys = ["E_X1_1", "mean_err_sq", "mmd2_eval_kernel_ustat",
            "spurious_vstat_floor", "spurious_ustat_floor"]

    for method in ["ustat", "vstat"]:
        print(f"--- method = {method} ---")
        for s in args.seeds:
            t0 = time.time()
            info = run_one(s, method, cfg)
            elapsed = time.time() - t0
            print(f"  seed={s}: E[X1]_1={info['E_X1_1']:.3f}  "
                  f"|err|^2={info['mean_err_sq']:.4f}  "
                  f"MMD^2={info['mmd2_eval_kernel_ustat']:.4e}  "
                  f"spurious(V)={info['spurious_vstat_floor']:.4e}  "
                  f"({elapsed:.0f}s)")
            per_method[method].append(info)
        agg = aggregate(per_method[method], keys)
        print(f"  => {method}: E[X1]_1 = {agg['E_X1_1_mean']:.3f} "
              f"+/- {agg['E_X1_1_std']:.3f}, "
              f"MMD^2 (eval) = {agg['mmd2_eval_kernel_ustat_mean']:.4e} "
              f"+/- {agg['mmd2_eval_kernel_ustat_std']:.4e}\n")

    summary = {m: aggregate(runs, keys) for m, runs in per_method.items()}
    summary["per_run"] = per_method

    out_path = os.path.join(args.output_dir,
                            f"exp_5_1_3_vstat_penalty_{tag}.json")
    with open(out_path, "w") as f:
        json.dump({"config": cfg, "args": vars(args), "summary": summary},
                  f, indent=2)
    print(f"Saved: {out_path}")

    # ---------------- LaTeX table -----------------
    print("\n" + "=" * 60)
    print("LaTeX table for paper (V-stat vs U-stat penalty SBP):")
    print("=" * 60)
    print(r"""
\begin{tabular}{lcccc}
\toprule
Penalty & $\EE[X_1]_1$ & $|\EE[X_1]-m|^2$ & MMD${}^2$ (eval) & V-stat floor $\hat V(\mu_1,\mu_1)$ \\
\midrule""")
    for method, label in [("ustat", "RF $U$-stat (proposed)"),
                          ("vstat", "RFF $V$-stat \\cite{rah-rec:2007}")]:
        a = summary[method]

        def fmt(mean_key, std_key, fmt=".3f"):
            return f"${a[mean_key]:{fmt}}\\pm{a[std_key]:{fmt}}$"

        def fmt_sci(mean_key, std_key):
            m, s = a[mean_key], a[std_key]
            exp = int(np.floor(np.log10(max(m, 1e-300))))
            mm, ss = m / 10**exp, s / 10**exp
            return f"$({mm:.1f}\\pm{ss:.1f})\\times 10^{{{exp}}}$"

        print(f"{label} & "
              f"{fmt('E_X1_1_mean', 'E_X1_1_std', '.2f')} & "
              f"{fmt('mean_err_sq_mean', 'mean_err_sq_std', '.2f')} & "
              f"{fmt_sci('mmd2_eval_kernel_ustat_mean', 'mmd2_eval_kernel_ustat_std')} & "
              f"{fmt_sci('spurious_vstat_floor_mean', 'spurious_vstat_floor_std')} \\\\")
    print(r"""\bottomrule
\end{tabular}""")


if __name__ == "__main__":
    main()
