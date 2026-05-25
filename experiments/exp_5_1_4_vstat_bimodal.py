"""
Section 5.1.4 (extension) -- V-stat vs U-stat penalty on a *bimodal* target.

The Gaussian-shift comparison shows V-stat and U-stat penalties give
indistinguishable results across (N, lambda). The bimodal target has a
landscape with two equal-mass modes that may amplify the bias-gradient
effect, since the V-stat bias depends on E[K(X, X')] which differs
sharply between bimodal and unimodal distributions of the same first two
moments.

Setup matches Section 5.2 (bimodal d=2):
  mu_0 = delta_0, mu_1 = 0.5 N((2,2), 0.25 I) + 0.5 N((-2,-2), 0.25 I)
  M=200, hidden=(64,32), alpha=1.0, epochs=2000, sigma=0.5.
We sweep N in {16, 64} and lambda^-1 in {1e-2, 1e-4}.

Output: results/exp_5_1_3_vstat_bimodal_*.json + LaTeX table.
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

DEFAULT_CFG = {"d": 2, "M": 200, "N": 64, "hidden": (64, 32),
               "alpha": 1.0, "lam_inv": 1e-2, "epochs": 2000,
               "sigma": 0.5, "time_steps": 20, "lr": 1e-3}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--N", type=int, default=DEFAULT_CFG["N"])
    p.add_argument("--lam_inv", type=float, default=DEFAULT_CFG["lam_inv"])
    p.add_argument("--epochs", type=int, default=DEFAULT_CFG["epochs"])
    p.add_argument("--tag", type=str, default=None)
    p.add_argument("--output_dir", type=str, default="results")
    return p.parse_args()


def build_cfg(args):
    cfg = dict(DEFAULT_CFG)
    cfg["N"] = args.N
    cfg["lam_inv"] = args.lam_inv
    cfg["epochs"] = args.epochs
    return cfg


def sample_bimodal(n, d):
    centers = torch.tensor([[2.0, 2.0], [-2.0, -2.0]])
    idx = torch.randint(0, 2, (n,))
    return centers[idx] + 0.5 * torch.randn(n, d)


def mode_assignment(x):
    """Return fractions in each mode (positive/negative cluster) and
    a 'mode-symmetry error' which is |fraction_pos - 0.5|."""
    # x is (n, 2). Mode of point: 1 if x[:,0] + x[:,1] > 0 else 0
    s = (x[:, 0] + x[:, 1]).numpy()
    f_pos = float((s > 0).mean())
    f_neg = 1.0 - f_pos
    sym_err = abs(f_pos - 0.5)
    return f_pos, f_neg, sym_err


def run_one(seed, method, cfg):
    d, N, sigma, T = cfg["d"], cfg["N"], cfg["sigma"], cfg["time_steps"]
    set_seed(seed)

    drift = DriftNet(d, hidden_dims=cfg["hidden"])
    optimizer = optim.Adam(drift.parameters(), lr=cfg["lr"])
    t_grid = torch.linspace(0.0, 1.0, T)

    penalty_fn = mmd2_rf_ustat if method == "ustat" else mmd2_rff_vstat

    for ep in range(cfg["epochs"]):
        optimizer.zero_grad()
        x0 = torch.zeros(N, d)
        x1, ctrl = sde_integrate_with_cost(drift, x0, t_grid, sigma, t_subsample=8)
        y = sample_bimodal(N, d)
        mmd2 = penalty_fn(x1, y, M=cfg["M"], alpha=cfg["alpha"])
        loss = ctrl * cfg["lam_inv"] + mmd2
        loss.backward()
        optimizer.step()

    n_eval = 2000
    with torch.no_grad():
        x0 = torch.zeros(n_eval, d)
        x1 = sde_integrate(drift, x0, t_grid, sigma)
        Y = sample_bimodal(n_eval, d)
        mmd2_eval = mmd2_kernel_ustat(x1, Y, cfg["alpha"]).item()
        f_pos, f_neg, sym_err = mode_assignment(x1)
        # E[X1]_sym: expectation of (X1_0 + X1_1) / 2, should be 0 for bimodal
        e_x1_sym = float(((x1[:, 0] + x1[:, 1]) / 2.0).mean())
        # Variance of X1 in the diagonal direction (should be 8 + 0.25 = 8.25)
        var_x1_diag = float(((x1[:, 0] + x1[:, 1]) / np.sqrt(2.0)).var())

    return {
        "method": method, "seed": seed,
        "mmd2_eval_kernel_ustat": mmd2_eval,
        "frac_pos_mode": f_pos,
        "mode_symmetry_err": sym_err,
        "E_X1_diag": e_x1_sym,
        "var_X1_diag": var_x1_diag,
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
    print(f"V-stat vs U-stat penalty SBP, BIMODAL target [{tag}], "
          f"d={cfg['d']}, N={cfg['N']}, lam_inv={cfg['lam_inv']}, "
          f"seeds={args.seeds}\n")

    per_method = {"ustat": [], "vstat": []}
    keys = ["mmd2_eval_kernel_ustat", "frac_pos_mode", "mode_symmetry_err",
            "E_X1_diag", "var_X1_diag"]

    for method in ["ustat", "vstat"]:
        print(f"--- method = {method} ---")
        for s in args.seeds:
            t0 = time.time()
            info = run_one(s, method, cfg)
            elapsed = time.time() - t0
            print(f"  seed={s}: MMD^2={info['mmd2_eval_kernel_ustat']:.4e}  "
                  f"f_pos={info['frac_pos_mode']:.3f}  "
                  f"sym_err={info['mode_symmetry_err']:.3f}  "
                  f"var_diag={info['var_X1_diag']:.3f}  ({elapsed:.0f}s)")
            per_method[method].append(info)
        agg = aggregate(per_method[method], keys)
        print(f"  => {method}: MMD^2 = {agg['mmd2_eval_kernel_ustat_mean']:.4e} "
              f"+/- {agg['mmd2_eval_kernel_ustat_std']:.4e}, "
              f"mode_sym_err = {agg['mode_symmetry_err_mean']:.3f} "
              f"+/- {agg['mode_symmetry_err_std']:.3f}\n")

    summary = {m: aggregate(runs, keys) for m, runs in per_method.items()}
    summary["per_run"] = per_method
    out_path = os.path.join(args.output_dir,
                            f"exp_5_1_3_vstat_bimodal_{tag}.json")
    with open(out_path, "w") as f:
        json.dump({"config": cfg, "args": vars(args), "summary": summary},
                  f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
