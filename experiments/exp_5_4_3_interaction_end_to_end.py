"""End-to-end test of the translation-invariant interaction estimator.

The controlled diffusion is trained for the same type of Gaussian terminal
target used elsewhere in the paper, now with c R[Law(X_t)] in the running
cost, where R uses a Gaussian translation-invariant kernel.  The experiment
compares c=0 with c>0 and a small interaction-bandwidth sensitivity grid.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch import optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kernel_sot import (  # noqa: E402
    DriftNet,
    ensure_dir,
    interaction_rf_ustat,
    mmd2_rf_ustat,
    sde_integrate,
    set_seed,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--d", type=int, default=2)
    p.add_argument("--sigma", type=float, default=0.5)
    p.add_argument("--alpha_terminal", type=float, default=0.5)
    p.add_argument("--M_terminal", type=int, default=200)
    p.add_argument("--M_interaction", type=int, default=64)
    p.add_argument("--N", type=int, default=128)
    p.add_argument("--lam_inv", type=float, default=0.02)
    p.add_argument("--c", type=float, default=5.0)
    p.add_argument("--alpha_interactions", type=float, nargs="+",
                   default=[0.5, 1.0, 2.0])
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--time_steps", type=int, default=20)
    p.add_argument("--interaction_time_samples", type=int, default=4)
    p.add_argument("--hidden", type=int, nargs="+", default=[64, 32])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--n_eval", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--output_dir", type=str,
                   default="results/interaction_end_to_end")
    return p.parse_args()


def integrate_training(drift, x0, t_grid, sigma, M, alpha, n_time_samples):
    dt = t_grid[1:] - t_grid[:-1]
    selected = set(
        torch.randperm(len(dt))[:min(n_time_samples, len(dt))].tolist()
    )
    x = x0
    energy = torch.tensor(0.0)
    interaction_sum = torch.tensor(0.0)
    for i, delta_t in enumerate(dt):
        u = drift(t_grid[i], x)
        energy = energy + 0.5 * (u ** 2).sum(dim=1).mean() * delta_t
        if i in selected:
            interaction_sum = interaction_sum + interaction_rf_ustat(x, M, alpha)
        x = x + u * delta_t + sigma * torch.randn_like(x) * torch.sqrt(delta_t)
    interaction = interaction_sum / max(len(selected), 1)
    return x, energy, interaction


def evaluate(drift, args, shift, alpha_interaction, seed):
    with torch.no_grad():
        set_seed(20000 + seed)
        # ``time_steps`` is the number of Euler intervals.  With the default
        # even value, the resulting grid contains t=1/2 exactly.
        t_grid = torch.linspace(0.0, 1.0, args.time_steps + 1)
        path = sde_integrate(
            drift, torch.zeros(args.n_eval, args.d), t_grid, args.sigma,
            return_path=True,
        )
        dt = t_grid[1:] - t_grid[:-1]
        energy = torch.tensor(0.0)
        interactions = []
        for i, delta_t in enumerate(dt):
            u = drift(t_grid[i], path[i])
            energy += 0.5 * (u ** 2).sum(dim=1).mean() * delta_t
            interactions.append(
                interaction_rf_ustat(path[i], 2000, alpha_interaction)
            )
        interaction_integral = torch.stack(interactions).mul(dt).sum()
        target = shift + torch.randn(args.n_eval, args.d)
        terminal_mmd2 = mmd2_rf_ustat(
            path[-1], target, M=2000, alpha=args.alpha_terminal
        )
        variances = path.var(dim=1, unbiased=True).sum(dim=1)
        midpoint_index = args.time_steps // 2
        midpoint_time = float(t_grid[midpoint_index])
        if not np.isclose(midpoint_time, 0.5):
            raise ValueError(
                "time_steps must be even so that the evaluation grid contains t=1/2"
            )
        terminal_mean_error = ((path[-1].mean(dim=0) - shift) ** 2).sum()
    return {
        "energy": float(energy),
        "interaction_integral": float(interaction_integral),
        "terminal_mmd2": float(terminal_mmd2),
        "terminal_mean_error_sq": float(terminal_mean_error),
        "midpoint_time": midpoint_time,
        "midpoint_total_variance": float(variances[midpoint_index]),
        "terminal_total_variance": float(variances[-1]),
    }


def train_one(args, c, alpha_interaction, seed):
    set_seed(seed)
    shift = torch.zeros(args.d)
    shift[0] = 2.0
    t_grid = torch.linspace(0.0, 1.0, args.time_steps + 1)
    drift = DriftNet(args.d, hidden_dims=tuple(args.hidden))
    optimizer = optim.Adam(drift.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        x1, energy, interaction = integrate_training(
            drift,
            torch.zeros(args.N, args.d),
            t_grid,
            args.sigma,
            args.M_interaction,
            alpha_interaction,
            args.interaction_time_samples,
        )
        target = shift + torch.randn(args.N, args.d)
        terminal_mmd2 = mmd2_rf_ustat(
            x1, target, M=args.M_terminal, alpha=args.alpha_terminal
        )
        loss = args.lam_inv * (energy + c * interaction) + terminal_mmd2
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 500 == 0:
            print(
                f"c={c:g} alpha_W={alpha_interaction:g} seed={seed} "
                f"epoch={epoch + 1}/{args.epochs}",
                flush=True,
            )
    metrics = evaluate(drift, args, shift, alpha_interaction, seed)
    metrics.update({"seed": seed, "c": c, "alpha_interaction": alpha_interaction})
    return metrics


def summarize(runs):
    scalar_keys = [
        "energy",
        "interaction_integral",
        "terminal_mmd2",
        "terminal_mean_error_sq",
        "midpoint_total_variance",
        "terminal_total_variance",
    ]
    ddof = 1 if len(runs) > 1 else 0
    return {
        key: {
            "mean": float(np.mean([run[key] for run in runs])),
            "std": float(np.std([run[key] for run in runs], ddof=ddof)),
        }
        for key in scalar_keys
    }


def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    configurations = [(0.0, 1.0)] + [
        (args.c, alpha) for alpha in args.alpha_interactions
    ]
    output = {"args": vars(args), "configurations": []}
    for c, alpha in configurations:
        runs = [train_one(args, c, alpha, seed) for seed in args.seeds]
        output["configurations"].append({
            "c": c,
            "alpha_interaction": alpha,
            "summary": summarize(runs),
            "runs": runs,
        })
    output_path = os.path.join(args.output_dir, "summary.json")
    with open(output_path, "w") as handle:
        json.dump(output, handle, indent=2)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
