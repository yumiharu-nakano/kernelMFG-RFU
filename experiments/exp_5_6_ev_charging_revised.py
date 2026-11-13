"""Physically consistent EV charging experiment.

This experiment intentionally lives next to, rather than overwriting,
``exp_5_6_ev_charging.py``.  It repairs the two main modelling problems in the
original experiment:

1. the same applied charging power drives the SOC and enters aggregate demand;
2. the neural policy is a bounded V1G charging fraction in [0, 1].

For an EV with SOC ``s`` and bounded log-heterogeneity ``h``, let

    a_theta(t,s,h) in [0,1],
    P_theta(t,s,h) = eta(h) p_max(s) a_theta(t,s,h).

The normalized grid power ``P_theta`` is used in both

    ds_t = charge_rate * P_theta dt + sigma dB_t

and

    D_t = E[P_theta(t,s_t,h)].

The training objective is

    0.5 E[int P_theta^2 dt]
      + c int D_t^2 dt
      + lambda MMD_K^2(Law(s_T), mu_T).

The running term depends on the distribution of state-control pairs, so this is
an algorithmic extension of the state-law interaction theorem in the paper; the
script does not claim that the theorem covers the trained neural policy.

The SOC Euler step is projected to [0,1].  Boundary-hit rates are reported so
that the projection is observable rather than hidden.  Initial and target SOC,
and the heterogeneity variable, are sampled from truncated Gaussian laws.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import NormalDist

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kernel_sot import mmd2_kernel_ustat, mmd2_rf_ustat, set_seed


@dataclass(frozen=True)
class PhysicalConfig:
    initial_mean: float = 0.20
    initial_std: float = 0.05
    target_mean: float = 0.85
    target_std: float = 0.05
    service_threshold: float = 0.80
    sigma_h: float = 0.30
    h_max: float = 0.90
    charge_rate: float = 1.0
    soc_low: float = 0.0
    soc_high: float = 1.0


class ChargePolicy(nn.Module):
    """Feed-forward V1G policy with scalar action in (0,1)."""

    def __init__(self, state_dim: int = 2, hidden_dims: tuple[int, ...] = (64, 32)):
        super().__init__()
        layers: list[nn.Module] = []
        width = state_dim + 1
        for hidden in hidden_dims:
            layers.extend((nn.Linear(width, hidden), nn.ReLU(), nn.LayerNorm(hidden)))
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)
        with torch.no_grad():
            self.net[-1].weight.mul_(0.01)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        if t.ndim == 0:
            t = t.expand(state.shape[0], 1)
        elif t.ndim == 1:
            t = t[:, None]
        logits = self.net(torch.cat((t, state), dim=1))
        return torch.sigmoid(logits)


def p_max_profile(s: torch.Tensor, beta: float = 20.0,
                  s_low: float = 0.10, s_high: float = 0.85) -> torch.Tensor:
    """Smooth normalized CC--CV charging-capability profile."""
    return torch.sigmoid(beta * (s - s_low)) * torch.sigmoid(beta * (s_high - s))


def truncated_exp_mean(sigma_h: float, h_max: float) -> float:
    """E[exp(H) | |H| <= h_max] for H ~ N(0, sigma_h^2)."""
    standard = NormalDist()
    denominator = standard.cdf(h_max / sigma_h) - standard.cdf(-h_max / sigma_h)
    numerator = math.exp(0.5 * sigma_h**2) * (
        standard.cdf((h_max - sigma_h**2) / sigma_h)
        - standard.cdf((-h_max - sigma_h**2) / sigma_h)
    )
    return numerator / denominator


def eta_h(h: torch.Tensor, physical: PhysicalConfig) -> torch.Tensor:
    """Positive bounded speed multiplier, normalized to have mean one."""
    normalizer = truncated_exp_mean(physical.sigma_h, physical.h_max)
    return torch.exp(h) / normalizer


def sample_truncated_normal(n: int, mean: float, std: float, low: float,
                            high: float, device: torch.device) -> torch.Tensor:
    """Rejection sample a scalar truncated Gaussian as an (n,1) tensor."""
    accepted: list[torch.Tensor] = []
    count = 0
    while count < n:
        draw = mean + std * torch.randn(max(2 * (n - count), 64), 1, device=device)
        keep = draw[(draw[:, 0] >= low) & (draw[:, 0] <= high)]
        if keep.numel():
            accepted.append(keep)
            count += keep.shape[0]
    return torch.cat(accepted, dim=0)[:n]


def sample_initial(n: int, physical: PhysicalConfig, device: torch.device) -> torch.Tensor:
    soc = sample_truncated_normal(
        n, physical.initial_mean, physical.initial_std,
        physical.soc_low, physical.soc_high, device,
    )
    h = sample_truncated_normal(
        n, 0.0, physical.sigma_h, -physical.h_max, physical.h_max, device,
    )
    return torch.cat((soc, h), dim=1)


def sample_target(n: int, physical: PhysicalConfig, device: torch.device) -> torch.Tensor:
    return sample_truncated_normal(
        n, physical.target_mean, physical.target_std,
        physical.soc_low, physical.soc_high, device,
    )


def aggregate_square_ustat(power: torch.Tensor) -> torch.Tensor:
    """Unbiased O(N) U-statistic for (E[power])^2."""
    values = power.reshape(-1)
    n = values.shape[0]
    if n < 2:
        raise ValueError("aggregate_square_ustat requires at least two particles")
    return (values.sum().square() - values.square().sum()) / (n * (n - 1))


def integrate_paths(policy: ChargePolicy, x0: torch.Tensor, t_grid: torch.Tensor,
                    sigma: float, physical: PhysicalConfig,
                    collect_diagnostics: bool = False) -> dict[str, torch.Tensor]:
    """Projected Euler--Maruyama using the same power in dynamics and demand."""
    state = x0.clone()
    states = [state]
    actions: list[torch.Tensor] = []
    powers: list[torch.Tensor] = []
    power_sq_integral = torch.zeros((), device=x0.device)
    boundary_hits = torch.zeros((), device=x0.device)
    total_updates = 0

    for k in range(len(t_grid) - 1):
        dt = float(t_grid[k + 1] - t_grid[k])
        t = t_grid[k].expand(state.shape[0], 1)
        action = policy(t, state)
        soc, h = state[:, :1], state[:, 1:2]
        power = eta_h(h, physical) * p_max_profile(soc) * action
        pre_projection = (
            soc
            + physical.charge_rate * power * dt
            + sigma * math.sqrt(dt) * torch.randn_like(soc)
        )
        if collect_diagnostics:
            boundary_hits = boundary_hits + (
                (pre_projection < physical.soc_low)
                | (pre_projection > physical.soc_high)
            ).sum()
            total_updates += pre_projection.numel()
        next_soc = pre_projection.clamp(physical.soc_low, physical.soc_high)
        state = torch.cat((next_soc, h), dim=1)
        power_sq_integral = power_sq_integral + power.square().mean() * dt
        states.append(state)
        actions.append(action)
        powers.append(power)

    hit_rate = boundary_hits / max(total_updates, 1)
    return {
        "states": torch.stack(states),
        "actions": torch.stack(actions),
        "powers": torch.stack(powers),
        "power_sq_integral": power_sq_integral,
        "boundary_hit_rate": hit_rate,
    }


def choose_device(name: str) -> torch.device:
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def train_one(args: argparse.Namespace, physical: PhysicalConfig,
              congestion_weight: float, seed: int,
              device: torch.device) -> tuple[dict, dict[str, np.ndarray]]:
    set_seed(seed)
    policy = ChargePolicy(hidden_dims=tuple(args.hidden)).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)
    t_grid = torch.linspace(0.0, args.T, args.time_steps + 1, device=device)
    checkpoints: list[dict] = []

    start = time.time()
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()
        x0 = sample_initial(args.N, physical, device)
        path = integrate_paths(policy, x0, t_grid, args.sigma, physical)
        terminal_soc = path["states"][-1, :, :1]
        target = sample_target(args.N, physical, device)
        terminal_mmd = mmd2_rf_ustat(
            terminal_soc, target, M=args.M, alpha=args.alpha_K,
        )
        congestion = torch.zeros((), device=device)
        for k in range(args.time_steps):
            dt = float(t_grid[k + 1] - t_grid[k])
            congestion = congestion + aggregate_square_ustat(path["powers"][k]) * dt
        loss = (
            0.5 * path["power_sq_integral"]
            + congestion_weight * congestion
            + args.lam * terminal_mmd
        )
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            checkpoints.append({
                "epoch": epoch,
                "loss": float(loss.detach().cpu()),
                "power_sq_integral": float(path["power_sq_integral"].detach().cpu()),
                "congestion": float(congestion.detach().cpu()),
                "terminal_mmd_train": float(terminal_mmd.detach().cpu()),
            })

    training_seconds = time.time() - start

    # Use common held-out randomness for the same seed across all c values.
    set_seed(args.eval_seed_offset + seed)
    with torch.no_grad():
        x0_eval = sample_initial(args.n_eval, physical, device)
        eval_path = integrate_paths(
            policy, x0_eval, t_grid, args.sigma, physical,
            collect_diagnostics=True,
        )
        terminal_soc = eval_path["states"][-1, :, :1]
        target_eval = sample_target(args.n_eval, physical, device)
        terminal_mmd = mmd2_kernel_ustat(
            terminal_soc, target_eval, args.alpha_K,
        )
        mean_power_t = eval_path["powers"].mean(dim=1).squeeze(-1)
        mean_action_t = eval_path["actions"].mean(dim=1).squeeze(-1)
        initial_soc = x0_eval[:, :1]
        terminal_shortfall = (physical.service_threshold - terminal_soc).clamp_min(0.0)
        terminal_error = terminal_soc - physical.target_mean

        summary = {
            "c": congestion_weight,
            "seed": seed,
            "training_seconds": training_seconds,
            "terminal_soc_mean": float(terminal_soc.mean().cpu()),
            "terminal_soc_std": float(terminal_soc.std().cpu()),
            "terminal_mmd2": float(terminal_mmd.cpu()),
            "mean_shortfall_below_service": float(terminal_shortfall.mean().cpu()),
            "service_failure_rate": float((terminal_soc < physical.service_threshold).float().mean().cpu()),
            "target_rmse": float(terminal_error.square().mean().sqrt().cpu()),
            "peak_mean_power": float(mean_power_t.max().cpu()),
            "time_average_mean_power": float(mean_power_t.mean().cpu()),
            "delivered_soc_mean": float((terminal_soc - initial_soc).mean().cpu()),
            "mean_action": float(eval_path["actions"].mean().cpu()),
            "max_action": float(eval_path["actions"].max().cpu()),
            "boundary_hit_rate": float(eval_path["boundary_hit_rate"].cpu()),
            "mean_power_t": mean_power_t.cpu().tolist(),
            "mean_action_t": mean_action_t.cpu().tolist(),
            "checkpoints": checkpoints,
        }
        plot_data = {
            "t_grid": t_grid.cpu().numpy(),
            "states": eval_path["states"].cpu().numpy(),
            "powers": eval_path["powers"].cpu().numpy(),
            "actions": eval_path["actions"].cpu().numpy(),
            "target": target_eval.cpu().numpy(),
        }
    return summary, plot_data


def aggregate_runs(runs: list[dict], c_value: float) -> dict:
    keys = [
        "terminal_soc_mean", "terminal_soc_std", "terminal_mmd2",
        "mean_shortfall_below_service", "service_failure_rate", "target_rmse",
        "peak_mean_power", "time_average_mean_power", "delivered_soc_mean",
        "mean_action", "max_action", "boundary_hit_rate", "training_seconds",
    ]
    result = {"c": c_value, "n_seeds": len(runs)}
    for key in keys:
        values = np.asarray([run[key] for run in runs], dtype=float)
        result[f"{key}_mean"] = float(values.mean())
        result[f"{key}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    curves = np.asarray([run["mean_power_t"] for run in runs], dtype=float)
    result["mean_power_t_mean"] = curves.mean(axis=0).tolist()
    result["mean_power_t_std"] = (
        curves.std(axis=0, ddof=1).tolist() if len(curves) > 1
        else np.zeros_like(curves[0]).tolist()
    )
    return result


def make_figure(output_path: Path, aggregates: list[dict],
                representative: dict[float, dict[str, np.ndarray]],
                physical: PhysicalConfig) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2))
    colors = plt.cm.viridis(np.linspace(0.08, 0.9, len(aggregates)))

    ax = axes[0, 0]
    for color, aggregate in zip(colors, aggregates):
        data = representative[float(aggregate["c"])]
        t = data["t_grid"][:-1]
        mean = np.asarray(aggregate["mean_power_t_mean"])
        std = np.asarray(aggregate["mean_power_t_std"])
        ax.plot(t, mean, color=color, lw=2, label=f"c={aggregate['c']:g}")
        ax.fill_between(t, mean - std, mean + std, color=color, alpha=0.14)
    ax.set(title="Applied aggregate grid power", xlabel="t", ylabel="mean normalized power")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[0, 1]
    for color, aggregate in zip(colors, aggregates):
        c_value = float(aggregate["c"])
        terminal = representative[c_value]["states"][-1, :, 0]
        ax.hist(terminal, bins=45, density=True, histtype="step", lw=1.8,
                color=color, label=f"c={c_value:g}")
    target = next(iter(representative.values()))["target"][:, 0]
    ax.hist(target, bins=45, density=True, histtype="step", lw=2,
            color="black", linestyle="--", label="target")
    ax.axvline(physical.service_threshold, color="firebrick", linestyle=":",
               label="service threshold")
    ax.set(title="Held-out terminal SOC", xlabel="SOC", ylabel="density")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    peak = [a["peak_mean_power_mean"] for a in aggregates]
    peak_err = [a["peak_mean_power_std"] for a in aggregates]
    failure = [100 * a["service_failure_rate_mean"] for a in aggregates]
    failure_err = [100 * a["service_failure_rate_std"] for a in aggregates]
    ax.errorbar(peak, failure, xerr=peak_err, yerr=failure_err, fmt="o-",
                capsize=4, color="tab:blue", lw=2)
    for aggregate, x_value, y_value in zip(aggregates, peak, failure):
        ax.annotate(f"c={aggregate['c']:g}", (x_value, y_value),
                    xytext=(5, 5), textcoords="offset points", fontsize=8)
    target_failure = 100 * NormalDist().cdf(
        (physical.service_threshold - physical.target_mean) / physical.target_std
    )
    ax.axhline(target_failure, color="black", linestyle="--", lw=1.3,
               label=f"target law: {target_failure:.1f}%")
    ax.set(title="Operational trade-off", xlabel="peak mean power",
           ylabel="service failures below 0.80 (%)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    c_values = [a["c"] for a in aggregates]
    soc_mean = [a["terminal_soc_mean_mean"] for a in aggregates]
    soc_std = [a["terminal_soc_std_mean"] for a in aggregates]
    ax.errorbar(c_values, soc_mean, yerr=soc_std, fmt="o-", capsize=4,
                color="tab:green", label="terminal SOC mean +/- SOC std")
    ax.axhline(physical.target_mean, color="black", linestyle="--", label="target mean")
    ax.axhline(physical.service_threshold, color="firebrick", linestyle=":",
               label="service threshold")
    ax.set_xscale("symlog", linthresh=1.0)
    ax.set(title="Terminal service quality", xlabel="congestion weight c", ylabel="SOC")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    fig.suptitle("Revised EV charging experiment: physical power consistency and bounded V1G control")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def make_compact_figure(output_path: Path, aggregates: list[dict],
                        representative: dict[float, dict[str, np.ndarray]],
                        physical: PhysicalConfig) -> None:
    """Compact two-panel summary of demand and terminal-service trade-offs."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    colors = plt.cm.viridis(np.linspace(0.08, 0.9, len(aggregates)))

    ax = axes[0]
    for color, aggregate in zip(colors, aggregates):
        c_value = float(aggregate["c"])
        t = representative[c_value]["t_grid"][:-1]
        mean = np.asarray(aggregate["mean_power_t_mean"])
        std = np.asarray(aggregate["mean_power_t_std"])
        ax.plot(t, mean, color=color, lw=2.2, label=f"c={c_value:g}")
        ax.fill_between(t, mean - std, mean + std, color=color, alpha=0.14)
    ax.set(title="Fleet-mean normalized charging power", xlabel="t",
           ylabel=r"$D_t$: fleet-mean normalized power")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)

    ax = axes[1]
    peak = [a["peak_mean_power_mean"] for a in aggregates]
    peak_err = [a["peak_mean_power_std"] for a in aggregates]
    failure = [100 * a["service_failure_rate_mean"] for a in aggregates]
    failure_err = [100 * a["service_failure_rate_std"] for a in aggregates]
    ax.errorbar(peak, failure, xerr=peak_err, yerr=failure_err, fmt="o-",
                capsize=4, color="tab:blue", lw=2.2)
    for aggregate, x_value, y_value in zip(aggregates, peak, failure):
        ax.annotate(f"c={aggregate['c']:g}", (x_value, y_value),
                    xytext=(5, 5), textcoords="offset points", fontsize=9)
    target_failure = 100 * NormalDist().cdf(
        (physical.service_threshold - physical.target_mean) / physical.target_std
    )
    ax.axhline(target_failure, color="black", linestyle="--", lw=1.4,
               label=f"target law: {target_failure:.1f}%")
    ax.set(title="Maximum-demand reduction vs. terminal service",
           xlabel=r"$\max_t D_t$",
           ylabel="P(terminal SOC < 0.80) (%)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--c_values", type=float, nargs="+", default=[0.0, 100.0, 300.0, 1000.0]
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--N", type=int, default=160)
    parser.add_argument("--M", type=int, default=400)
    parser.add_argument("--n_eval", type=int, default=2000)
    parser.add_argument("--time_steps", type=int, default=24)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--sigma", type=float, default=0.02)
    parser.add_argument("--lam", type=float, default=1000.0)
    parser.add_argument("--alpha_K", type=float, default=50.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, nargs="+", default=[64, 32])
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--eval_seed_offset", type=int, default=100_000)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="cpu")
    parser.add_argument("--output_dir", default="results/ev_revised")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    physical = PhysicalConfig()
    device = choose_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eta_mean = truncated_exp_mean(physical.sigma_h, physical.h_max)
    print(f"device={device}; raw truncated E[exp(H)]={eta_mean:.6f}; normalized E[eta]=1")
    print(f"c_values={args.c_values}; seeds={args.seeds}; epochs={args.epochs}")

    all_runs: list[dict] = []
    aggregates: list[dict] = []
    representative: dict[float, dict[str, np.ndarray]] = {}

    for c_value in args.c_values:
        c_runs: list[dict] = []
        for seed in args.seeds:
            print(f"training c={c_value:g}, seed={seed}", flush=True)
            run, plot_data = train_one(args, physical, c_value, seed, device)
            c_runs.append(run)
            all_runs.append(run)
            if seed == args.seeds[0]:
                representative[float(c_value)] = plot_data
                np.savez_compressed(
                    output_dir / f"representative_c{c_value:g}_seed{seed}.npz",
                    **plot_data,
                )
            print(
                f"  peak={run['peak_mean_power']:.4f}; "
                f"SOC={run['terminal_soc_mean']:.4f}+/-{run['terminal_soc_std']:.4f}; "
                f"failure={100*run['service_failure_rate']:.2f}%; "
                f"boundary={100*run['boundary_hit_rate']:.3f}%; "
                f"{run['training_seconds']:.1f}s",
                flush=True,
            )
        aggregate = aggregate_runs(c_runs, c_value)
        aggregates.append(aggregate)

    payload = {
        "model_scope": (
            "Physically consistent V1G numerical extension with mean-control interaction; "
            "not claimed to be covered by the paper's state-law interaction theorem."
        ),
        "args": vars(args),
        "physical": asdict(physical),
        "eta_raw_truncated_mean": eta_mean,
        "aggregates": aggregates,
        "runs": all_runs,
    }
    json_path = output_dir / "ev_revised_summary.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    csv_path = output_dir / "ev_revised_summary.csv"
    flat_keys = [key for key in aggregates[0] if not key.endswith("_t_mean") and not key.endswith("_t_std")]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=flat_keys)
        writer.writeheader()
        for aggregate in aggregates:
            writer.writerow({key: aggregate[key] for key in flat_keys})

    figure_path = output_dir / "ev_revised_summary.png"
    make_figure(figure_path, aggregates, representative, physical)
    compact_figure_path = output_dir / "ev_revised_compact.png"
    make_compact_figure(compact_figure_path, aggregates, representative, physical)
    print(f"saved {json_path}")
    print(f"saved {csv_path}")
    print(f"saved {figure_path}")
    print(f"saved {compact_figure_path}")


if __name__ == "__main__":
    main()
