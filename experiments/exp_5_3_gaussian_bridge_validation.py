"""Analytic validation for the d=10 Gaussian Schrödinger bridge.

The reference process is dX_t = sigma dB_t, X_0=0, and the hard terminal
constraint is X_1 ~ N(m, I_d).  Its exact marginal law is

    X_t ~ N(t m, [t^2 + sigma^2 t(1-t)] I_d),

and its optimal feedback is

    u*(t,x) = ((1-sigma^2)x + sigma^2 m)
              / (sigma^2 + (1-sigma^2)t).

We train the finite-MMD-penalty problem at the strongest penalty used in the
paper and compare its marginal flow, feedback, and control energy with this
hard-constraint bridge.  The two optimization problems are intentionally
distinguished in the output.
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kernel_sot import (  # noqa: E402
    DriftNet,
    ensure_dir,
    mmd2_rf_ustat,
    sde_integrate,
    sde_integrate_with_cost,
    set_seed,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--d", type=int, default=10)
    p.add_argument("--sigma", type=float, default=0.5)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--M", type=int, default=400)
    p.add_argument("--N", type=int, default=160)
    p.add_argument("--lam_inv", type=float, default=2e-4)
    p.add_argument("--epochs", type=int, default=5000)
    p.add_argument("--time_steps", type=int, default=20)
    p.add_argument("--hidden", type=int, nargs="+", default=[128, 64])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--n_eval", type=int, default=4000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--output_dir", type=str,
                   default="results/gaussian_bridge_validation")
    return p.parse_args()


def exact_drift(t, x, shift, sigma):
    sigma2 = sigma ** 2
    denominator = sigma2 + (1.0 - sigma2) * t
    return ((1.0 - sigma2) * x + sigma2 * shift) / denominator


def exact_energy(d, shift_norm_sq, sigma):
    sigma2 = sigma ** 2
    return 0.5 * (
        d + shift_norm_sq - d * sigma2 + d * sigma2 * np.log(sigma2)
    )


def train_and_evaluate(args, seed):
    set_seed(seed)
    shift = torch.zeros(args.d)
    shift[0] = 3.0
    # ``time_steps`` denotes the number of Euler intervals, so the grid has
    # ``time_steps + 1`` points including both endpoints.
    t_grid = torch.linspace(0.0, 1.0, args.time_steps + 1)
    drift = DriftNet(args.d, hidden_dims=tuple(args.hidden))
    optimizer = optim.Adam(drift.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        optimizer.zero_grad()
        x0 = torch.zeros(args.N, args.d)
        x1, control_energy = sde_integrate_with_cost(
            drift, x0, t_grid, args.sigma, t_subsample=8
        )
        target = shift + torch.randn(args.N, args.d)
        mmd2 = mmd2_rf_ustat(x1, target, M=args.M, alpha=args.alpha)
        loss = args.lam_inv * control_energy + mmd2
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 1000 == 0:
            print(
                f"seed={seed} epoch={epoch + 1}/{args.epochs} "
                f"energy={control_energy.item():.4f} mmd2={mmd2.item():.5f}",
                flush=True,
            )

    with torch.no_grad():
        set_seed(10000 + seed)
        path = sde_integrate(
            drift,
            torch.zeros(args.n_eval, args.d),
            t_grid,
            args.sigma,
            return_path=True,
        )
        mean_first = path[:, :, 0].mean(dim=1)
        std_first = path[:, :, 0].std(dim=1, unbiased=True)

        dt = t_grid[1:] - t_grid[:-1]
        learned_energy = torch.tensor(0.0)
        drift_error_sq = torch.tensor(0.0)
        exact_drift_sq = torch.tensor(0.0)
        for i, delta_t in enumerate(dt):
            learned_u = drift(t_grid[i], path[i])
            exact_u = exact_drift(t_grid[i], path[i], shift, args.sigma)
            learned_energy += 0.5 * (learned_u ** 2).sum(dim=1).mean() * delta_t
            drift_error_sq += ((learned_u - exact_u) ** 2).sum(dim=1).mean() * delta_t
            exact_drift_sq += (exact_u ** 2).sum(dim=1).mean() * delta_t

        target = shift + torch.randn(args.n_eval, args.d)
        terminal_mmd2 = mmd2_rf_ustat(
            path[-1], target, M=2000, alpha=args.alpha
        )
        relative_drift_l2 = torch.sqrt(drift_error_sq / exact_drift_sq)

    return {
        "seed": seed,
        "mean_first": mean_first.cpu().numpy(),
        "std_first": std_first.cpu().numpy(),
        "sample_paths": path[:, :40, 0].cpu().numpy(),
        "learned_energy": float(learned_energy),
        "relative_drift_l2": float(relative_drift_l2),
        "terminal_mean_first": float(mean_first[-1]),
        "terminal_std_first": float(std_first[-1]),
        "terminal_mmd2": float(terminal_mmd2),
    }


def aggregate_scalar(runs, key):
    values = np.array([run[key] for run in runs], dtype=float)
    ddof = 1 if len(values) > 1 else 0
    return {"mean": float(values.mean()), "std": float(values.std(ddof=ddof))}


def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    runs = [train_and_evaluate(args, seed) for seed in args.seeds]

    times = np.linspace(0.0, 1.0, args.time_steps + 1)
    sigma2 = args.sigma ** 2
    exact_mean = 3.0 * times
    exact_std = np.sqrt(times ** 2 + sigma2 * times * (1.0 - times))
    exact_cost = exact_energy(args.d, 9.0, args.sigma)

    mean_curves = np.stack([run["mean_first"] for run in runs])
    std_curves = np.stack([run["std_first"] for run in runs])
    seed_ddof = 1 if len(runs) > 1 else 0
    summary = {
        "args": vars(args),
        "exact_hard_constraint": {
            "control_energy": exact_cost,
            "terminal_mean_first": 3.0,
            "terminal_std_first": 1.0,
        },
        "finite_penalty_learned": {
            key: aggregate_scalar(runs, key)
            for key in [
                "learned_energy",
                "relative_drift_l2",
                "terminal_mean_first",
                "terminal_std_first",
                "terminal_mmd2",
            ]
        },
        "runs": [
            {key: value for key, value in run.items()
             if key not in {"mean_first", "std_first", "sample_paths"}}
            for run in runs
        ],
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    np.savez(
        os.path.join(args.output_dir, "marginal_flow.npz"),
        times=times,
        learned_mean=mean_curves,
        learned_std=std_curves,
        exact_mean=exact_mean,
        exact_std=exact_std,
        sample_paths=np.stack([run["sample_paths"] for run in runs]),
    )

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.2))
    learned_mean = mean_curves.mean(axis=0)
    learned_mean_sd = mean_curves.std(axis=0, ddof=seed_ddof)
    axes[0].plot(times, exact_mean, "k--", label="exact bridge")
    axes[0].plot(times, learned_mean, color="C0", label="learned")
    axes[0].fill_between(
        times, learned_mean - learned_mean_sd, learned_mean + learned_mean_sd,
        color="C0", alpha=0.2, label="seed SD",
    )
    axes[0].set(xlabel="$t$", ylabel=r"$\mathbb{E}[X_{t,1}]$", title="Mean flow")
    axes[0].legend(frameon=False, fontsize=8)

    learned_std = std_curves.mean(axis=0)
    learned_std_sd = std_curves.std(axis=0, ddof=seed_ddof)
    axes[1].plot(times, exact_std, "k--", label="exact bridge")
    axes[1].plot(times, learned_std, color="C1", label="learned")
    axes[1].fill_between(
        times, learned_std - learned_std_sd, learned_std + learned_std_sd,
        color="C1", alpha=0.2,
    )
    axes[1].set(xlabel="$t$", ylabel=r"$\mathrm{Std}(X_{t,1})$", title="Marginal spread")

    representative = runs[0]["sample_paths"]
    axes[2].plot(times, representative, color="C0", alpha=0.15, linewidth=0.6)
    central_80_z = 1.2815515655446004
    axes[2].fill_between(
        times,
        exact_mean - central_80_z * exact_std,
        exact_mean + central_80_z * exact_std,
        color="k", alpha=0.12, label="exact 10--90% band",
    )
    axes[2].plot(times, exact_mean, "k--", linewidth=1.2)
    axes[2].set(xlabel="$t$", ylabel=r"$X_{t,1}$", title="Learned sample paths")
    axes[2].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for suffix in ["pdf", "png"]:
        fig.savefig(
            os.path.join(args.output_dir, f"gaussian_bridge_validation.{suffix}"),
            dpi=220,
            bbox_inches="tight",
        )
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
