"""
Section 5.6 (sbp_h02) -- EV charging fleet potential MFG (soft-target MFOT)
with physical per-vehicle heterogeneity and aggregate-demand congestion.

Setup (d=2):
  Each EV is described by X_t = (s_t, h_t) in R x R where
    s_t  ∈ R  -- State of charge (SOC), starting around 0.2, target 0.85.
    h_t  ∈ R  -- Physical heterogeneity coordinate: log charging-speed
                multiplier. eta(h) := exp(h) scales the SOC change rate
                produced by a unit demand, modelling the combined effect
                of battery capacity, charger maximum power, and conversion
                efficiency. h is constant in time (dh_t = 0).

  Controlled diffusion (T = 1):
    d s_t = eta(h) * u_theta(t, X_t) dt + sigma dW_t
    d h_t = 0
  with X_0 = (s_0, h_0), s_0 ~ N(0.20, 0.05^2), h_0 ~ N(0, SIGMA_H^2),
  SIGMA_H = 0.3 (eta(h) is log-normal with mean 1 and 95% range ~[0.55, 1.8]).

  Costs (potential MFG / soft-target MFOT):
    (i)  Control effort:                   (1/2) E[ int_0^T |u_theta|^2 dt ]
    (ii) Aggregate-demand congestion:      c * int_0^T D[Law(X_t)]^2 dt
         D[nu] = ∫ eta(h) u*(s) nu(ds, dh)
         u*(s) = sigmoid(beta*(s-s_low)) * sigmoid(beta*(s_high-s))
                  smooth Li-ion charging-power profile (CC plateau, CV taper)
         An unbiased O(N) U-statistic estimates D[nu]^2 with
         g_i := eta(h_i) * u*(s_i):
           Rhat_N = ((sum_i g_i)^2 - sum_i g_i^2) / (N(N-1))
    (iii) Terminal MMD penalty:             lambda * gamma_K(Law(s_T), mu_T)^2
         K(x,y) = exp(-alpha_K (x-y)^2)        (1-D Gaussian kernel on SOC)
         mu_T   = N(0.85, 0.05^2) target SOC distribution at deadline.

  Without congestion (c=0), the framework drives the fleet toward the target
  SOC distribution while the physical heterogeneity eta(h) gives each EV its
  intrinsic charging speed. With strong congestion (c>0), the drift uses
  (s, h) to actively stagger schedules across vehicles, broadening the
  terminal SOC distribution and reducing peak and time-averaged aggregate
  demand.

Output:
  results/exp_ev_charging.json    aggregate statistics over seeds
  results/exp_ev_charging.png     visualization
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
    DriftNet,
    mmd2_rf_ustat, mmd2_kernel_ustat,
    sde_integrate, sde_integrate_with_cost,
    set_seed, ensure_dir,
)


# ----- physical heterogeneity, aggregate-demand profile, U-statistic --------
def u_star(s, beta=20.0, s_low=0.1, s_high=0.85):
    """SOC-dependent Li-ion charging-power profile.

    u*(s) = sigmoid(beta*(s-s_low)) * sigmoid(beta*(s_high-s))
    smoothly rises to 1 on [0.2, ~0.8] (constant-current phase)
    and tapers to 0 above 0.85 (constant-voltage phase) and below 0.05.
    Returns a tensor of the same shape as s.
    """
    return torch.sigmoid(beta * (s - s_low)) * torch.sigmoid(beta * (s_high - s))


def eta_h(h):
    """Per-vehicle physical charging-speed multiplier: eta(h) = exp(h).

    With h_0 ~ N(0, sigma_h^2) (sigma_h=0.3 here), eta(h) is log-normal with
    mean 1 and 95% range approximately [0.55, 1.8]: faster-charging EVs have
    eta > 1 (smaller battery / higher-power charger / higher efficiency),
    slower EVs have eta < 1. The factor eta(h) enters both the SDE and the
    aggregate demand.
    """
    return torch.exp(h)


def aggregate_demand_sq_ustat(s, h):
    """Unbiased U-statistic estimator of D[nu]^2 where D[nu] = E[eta(h) u*(s)].

    Uses the algebraic identity
        (1/(N(N-1))) sum_{i ne j} g(X_i) g(X_j)
        = ((sum_i g(X_i))^2 - sum_i g(X_i)^2) / (N(N-1))
    with g(X) = eta(h) u*(s).

    Args:
        s, h: (N, 1) tensors of SOC and physical-heterogeneity samples.
    Returns:
        scalar tensor: unbiased estimate of D^2.
    """
    g = (eta_h(h) * u_star(s)).squeeze(-1)   # (N,)
    N = g.shape[0]
    sum_g = g.sum()
    sum_g_sq = (g ** 2).sum()
    return (sum_g ** 2 - sum_g_sq) / (N * (N - 1))


def aggregate_demand_mean(s, h):
    """Estimator of D[nu] = E[eta(h) u*(s)]: simple sample mean."""
    return (eta_h(h) * u_star(s)).mean()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--d", type=int, default=2,
                   help="State dimension. d=2 means (SOC, h) where h is the "
                        "scalar log charging-speed multiplier.")
    p.add_argument("--sigma", type=float, default=0.05,
                   help="SDE diffusion coefficient on SOC.")
    p.add_argument("--T", type=float, default=1.0)
    p.add_argument("--time_steps", type=int, default=20)
    p.add_argument("--M", type=int, default=300)
    p.add_argument("--N", type=int, default=128)
    p.add_argument("--epochs", type=int, default=3000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lam", type=float, default=1000.0,
                   help="Terminal MMD penalty weight (lambda).")
    p.add_argument("--c", type=float, default=10.0,
                   help="Congestion weight (c). Set to 0 for no-congestion baseline.")
    p.add_argument("--alpha_K", type=float, default=50.0,
                   help="Terminal kernel bandwidth K(x,y)=exp(-alpha_K (x-y)^2) on SOC.")
    p.add_argument("--hidden", type=int, nargs="+", default=[64, 32])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--output_dir", type=str, default="results")
    p.add_argument("--tag", type=str, default=None,
                   help="Suffix for output filenames.")
    return p.parse_args()


# ----- distributions ------------------------------------------------------------
SIGMA_H = 0.3   # std of physical heterogeneity h (log charging-speed factor)


def sample_initial(n, d, device=None):
    """X_0 = (s_0, h_0) with s_0 ~ N(0.20, 0.05^2) and h_0 ~ N(0, SIGMA_H^2 I_{d-1}).

    h is the per-vehicle physical heterogeneity (log charging-speed factor in
    the d=2 case). It enters both the SDE (through eta(h) * u_theta) and the
    aggregate-demand cost (through D = E[eta(h) u*(s)]).
    """
    s = 0.20 + 0.05 * torch.randn(n, 1, device=device)
    if d == 1:
        return s
    h = SIGMA_H * torch.randn(n, d - 1, device=device)
    return torch.cat([s, h], dim=1)


def sample_target_SOC(n, device=None):
    """mu_T = N(0.85, 0.05^2) on SOC."""
    return 0.85 + 0.05 * torch.randn(n, 1, device=device)


# ----- SDE integration with per-vehicle physical eta(h) factor -----------------
def sde_integrate_full_paths(drift, x0, t_grid, sigma, controlled_dim=1):
    """
    Euler-Maruyama returning the full trajectory.

    The SOC dynamics include the per-vehicle physical charging-speed factor
    eta(h) = exp(h): ds_t = eta(h) * u_theta(t, X_t) * dt + sigma * dB_t.
    Only the SOC coordinate (first column) has nontrivial dynamics; the
    heterogeneity coordinates h are kept constant (dh_t = 0).

    The control cost penalizes the demand fraction u_theta[..., 0] only,
    consistent with the running cost (1/2) E[|u|^2]. Physical interpretation:
    u_theta is the controlled "demand fraction", and the actual SOC change
    rate is eta(h) * u_theta.

    Returns:
        x_path  : (T_steps + 1, N, d) tensor
        ctrl_sq : scalar = (1/N) sum_i int_0^T |u_0(t, X_t^i)|^2 dt
    """
    T_steps = len(t_grid) - 1
    N, d = x0.shape
    x = x0.clone()
    # Physical charging-speed factor eta(h) for each vehicle, constant over time
    if d >= 2:
        h_init = x0[:, 1:2]                      # (N, 1) using h_1 only
        eta_per_vehicle = eta_h(h_init)          # (N, 1)
    else:
        eta_per_vehicle = torch.ones(N, 1, device=x0.device)
    x_path = [x]
    ctrl_sq = torch.tensor(0.0, device=x0.device)
    for k in range(T_steps):
        t = t_grid[k].expand(N, 1)
        dt = float(t_grid[k + 1] - t_grid[k])
        u_t = drift(t, x)                        # (N, d)
        # Control cost on the (raw) demand fraction:
        ctrl_sq = ctrl_sq + (u_t[:, :controlled_dim] ** 2).sum(dim=1).mean() * dt
        # SDE step: SOC evolves as ds = eta(h) * u_theta * dt + sigma * dB
        x_new = x.clone()
        eps = torch.randn(N, controlled_dim, device=x.device)
        x_new[:, :controlled_dim] = (
            x[:, :controlled_dim]
            + eta_per_vehicle * u_t[:, :controlled_dim] * dt
            + sigma * np.sqrt(dt) * eps
        )
        x = x_new
        x_path.append(x)
    return torch.stack(x_path, dim=0), ctrl_sq


# ----- training loop -----------------------------------------------------------
def train_one(args, seed):
    set_seed(seed)
    d = args.d
    sigma = args.sigma
    T = args.T
    t_grid = torch.linspace(0.0, T, args.time_steps + 1)

    drift = DriftNet(d, hidden_dims=tuple(args.hidden))
    optimizer = optim.Adam(drift.parameters(), lr=args.lr)

    losses = []
    for ep in range(args.epochs):
        optimizer.zero_grad()
        x0 = sample_initial(args.N, d)
        x_path, ctrl_sq = sde_integrate_full_paths(
            drift, x0, t_grid, sigma, controlled_dim=1)
        # Terminal SOC samples for the MMD penalty:
        s_T = x_path[-1][:, 0:1]                # (N, 1), SOC only
        y_T = sample_target_SOC(args.N)         # (N, 1)
        terminal_mmd = mmd2_rf_ustat(s_T, y_T, M=args.M, alpha=args.alpha_K)
        # Aggregate-demand congestion R^(D)[Law(X_t)] = D[nu_t]^2
        # where D[nu] = E[eta(h) u*(s)]. The per-vehicle physical heterogeneity
        # eta(h) enters both the SDE (above) and the aggregate demand here.
        if args.c > 0:
            congestion = torch.tensor(0.0)
            T_steps = len(t_grid) - 1
            h_path = x_path[:, :, 1:2] if d >= 2 else None     # (T+1, N, 1)
            for k in range(1, T_steps + 1):    # skip t=0 (initial law is fixed)
                soc_t = x_path[k][:, 0:1]      # SOC only (N, 1)
                h_t = h_path[k] if h_path is not None else torch.zeros_like(soc_t)
                dt = float(t_grid[k] - t_grid[k - 1])
                congestion = congestion + aggregate_demand_sq_ustat(soc_t, h_t) * dt
        else:
            congestion = torch.tensor(0.0)
        loss = 0.5 * ctrl_sq + args.c * congestion + args.lam * terminal_mmd
        loss.backward()
        optimizer.step()
        if (ep + 1) % max(1, args.epochs // 10) == 0:
            losses.append({
                "epoch": ep + 1,
                "loss": float(loss),
                "ctrl_sq": float(ctrl_sq),
                "congestion": float(congestion),
                "terminal_mmd": float(terminal_mmd),
            })

    # ----- evaluation on a large fresh batch -----
    with torch.no_grad():
        n_eval = 2000
        x0_eval = sample_initial(n_eval, d)
        x_path_eval, _ = sde_integrate_full_paths(
            drift, x0_eval, t_grid, sigma, controlled_dim=1)
        s_T_eval = x_path_eval[-1][:, 0:1]
        y_T_eval = sample_target_SOC(n_eval)
        # Evaluate terminal discrepancy with KERNEL U-statistic for fair comparison
        terminal_mmd_eval = mmd2_kernel_ustat(s_T_eval, y_T_eval,
                                              args.alpha_K).item()
        s_T_mean = float(s_T_eval.mean())
        s_T_std = float(s_T_eval.std())
        # Aggregate demand D[nu_t] = E[eta(h) u*(s)] and squared D^2 across time
        demand_per_t = []
        demand_sq_per_t = []
        drift_abs_per_t = []   # |u_theta(t, X_t)| (active SOC component) over time
        h_path_eval = x_path_eval[:, :, 1:2] if d >= 2 else None
        for k in range(1, len(t_grid)):
            soc_t = x_path_eval[k][:, 0:1]
            h_t = h_path_eval[k] if h_path_eval is not None else torch.zeros_like(soc_t)
            demand_per_t.append(aggregate_demand_mean(soc_t, h_t).item())
            demand_sq_per_t.append(aggregate_demand_sq_ustat(soc_t, h_t).item())
            # Drift magnitude at this time step on the evaluation batch
            t_k = t_grid[k].expand(n_eval, 1)
            u_t = drift(t_k, x_path_eval[k])     # (N_eval, d)
            drift_abs_per_t.append(float(u_t[:, 0].abs().max()))
        # u_n_infty: sup over (t, X_t) seen on the evaluation batch
        u_n_infty = max(drift_abs_per_t)
        return {
            "seed": seed,
            "terminal_mmd_eval": terminal_mmd_eval,
            "s_T_mean": s_T_mean,
            "s_T_std": s_T_std,
            "peak_demand": max(demand_per_t),
            "mean_demand_t": float(np.mean(demand_per_t)),
            "peak_demand_sq": max(demand_sq_per_t),
            "mean_demand_sq_t": float(np.mean(demand_sq_per_t)),
            "u_n_infty": u_n_infty,
            "losses": losses,
            "x_path_eval": x_path_eval.cpu().numpy(),   # for visualization (first run only)
            "y_T_eval": y_T_eval.cpu().numpy(),
            "t_grid": t_grid.cpu().numpy(),
            "demand_per_t": demand_per_t,
            "demand_sq_per_t": demand_sq_per_t,
            "drift_abs_per_t": drift_abs_per_t,
        }


# ----- main --------------------------------------------------------------------
def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    tag = args.tag or f"d{args.d}_c{args.c:.1f}_lam{args.lam:.0f}"
    print(f"EV charging MFG [{tag}], d={args.d}, c={args.c}, lambda={args.lam}, "
          f"epochs={args.epochs}, seeds={args.seeds}\n")

    runs = []
    for s in args.seeds:
        t0 = time.time()
        info = train_one(args, s)
        elapsed = time.time() - t0
        print(f"  seed={s}: s_T mean={info['s_T_mean']:.4f}  "
              f"s_T std={info['s_T_std']:.4f}  "
              f"MMD²={info['terminal_mmd_eval']:.4e}  "
              f"peak_D={info['peak_demand']:.4f}  "
              f"||u||={info['u_n_infty']:.3f}  "
              f"({elapsed:.0f}s)")
        runs.append(info)

    summary = {
        "s_T_mean_mean": float(np.mean([r["s_T_mean"] for r in runs])),
        "s_T_mean_std":  float(np.std ([r["s_T_mean"] for r in runs])),
        "s_T_std_mean":  float(np.mean([r["s_T_std"] for r in runs])),
        "terminal_mmd_mean": float(np.mean([r["terminal_mmd_eval"] for r in runs])),
        "terminal_mmd_std":  float(np.std ([r["terminal_mmd_eval"] for r in runs])),
        "peak_demand_mean":    float(np.mean([r["peak_demand"]    for r in runs])),
        "peak_demand_std":     float(np.std ([r["peak_demand"]    for r in runs])),
        "peak_demand_sq_mean": float(np.mean([r["peak_demand_sq"] for r in runs])),
        "peak_demand_sq_std":  float(np.std ([r["peak_demand_sq"] for r in runs])),
        "mean_demand_t_mean":    float(np.mean([r["mean_demand_t"]    for r in runs])),
        "mean_demand_sq_t_mean": float(np.mean([r["mean_demand_sq_t"] for r in runs])),
        "u_n_infty_mean": float(np.mean([r["u_n_infty"] for r in runs])),
        "u_n_infty_std":  float(np.std ([r["u_n_infty"] for r in runs])),
    }
    print("\nSummary:")
    for k, v in summary.items():
        print(f"  {k} = {v:.4f}")

    # ----- save JSON (omit heavy arrays for storage) -----
    out_path = os.path.join(args.output_dir, f"exp_ev_charging_{tag}.json")
    with open(out_path, "w") as f:
        json.dump({"args": vars(args), "summary": summary,
                   "per_run": [{k: r[k] for k in
                                ["seed", "terminal_mmd_eval", "s_T_mean",
                                 "s_T_std", "peak_demand", "mean_demand_t",
                                 "peak_demand_sq", "mean_demand_sq_t",
                                 "u_n_infty", "losses", "demand_per_t",
                                 "demand_sq_per_t", "drift_abs_per_t"]}
                               for r in runs]}, f, indent=2)
    print(f"Saved: {out_path}")

    # ----- visualization (first run) -----
    info = runs[0]
    x_path = info["x_path_eval"]    # (T+1, N, d)
    y_T = info["y_T_eval"][:, 0]
    t_grid = info["t_grid"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # (a) SOC trajectories (first 200 EVs)
    ax = axes[0]
    n_show = min(200, x_path.shape[1])
    for i in range(n_show):
        ax.plot(t_grid, x_path[:, i, 0], alpha=0.1, color="C0")
    ax.axhline(0.85, color="k", linestyle="--", lw=1, label="target SOC mean")
    ax.set_xlabel("$t$"); ax.set_ylabel("SOC")
    ax.set_title(f"SOC trajectories ($c={args.c}$, $\\lambda={args.lam}$)")
    ax.legend(loc="lower right")

    # (b) Terminal SOC histogram vs target
    ax = axes[1]
    ax.hist(x_path[-1, :, 0], bins=40, alpha=0.7, density=True, label="trained $\\nu_T$")
    ax.hist(y_T, bins=40, alpha=0.4, density=True, label="target $\\mu_T$")
    ax.set_xlabel("SOC at $t = T$"); ax.set_ylabel("density")
    ax.set_title("Terminal SOC distribution"); ax.legend()

    # (c) Aggregate demand D[nu_t] over time
    ax = axes[2]
    ax.plot(t_grid[1:], info["demand_per_t"], marker="o", lw=1.5, label="$D[\\nu_t]$")
    ax.plot(t_grid[1:], info["demand_sq_per_t"], marker="s", lw=1.5,
            alpha=0.7, label="$D[\\nu_t]^2$")
    ax.set_xlabel("$t$"); ax.set_ylabel("aggregate demand")
    ax.set_title("Aggregate demand across time"); ax.grid(True, alpha=0.3); ax.legend()

    plt.suptitle(f"EV charging MFG, $d={args.d}$, $c={args.c}$, $\\lambda={args.lam}$")
    plt.tight_layout()
    fig_path = os.path.join(args.output_dir, f"exp_ev_charging_{tag}.png")
    plt.savefig(fig_path, dpi=130)
    plt.close()
    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()
