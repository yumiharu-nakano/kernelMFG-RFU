# Kernel-based potential mean-field games with unbiased random Fourier *U*-statistics

Reference implementation accompanying the paper

> **Kernel-based potential mean-field games with unbiased random Fourier *U*-statistics.**
> Yumiharu Nakano. 2026.
> [arXiv:2605.29371](https://arxiv.org/abs/2605.29371)

## Overview

This paper develops a computational framework for *potential mean-field games* (MFGs)
in which the running interaction cost and the terminal target cost are both expressed
through reproducing-kernel maximum mean discrepancy (MMD) penalties.
Both costs are estimated from finite-sample empirical distributions using a
**random Fourier *U*-statistic representation** that is unbiased and has linear cost
in the batch size *N*.
The drift of the controlled diffusion is parametrized by a neural network and
trained via stochastic gradient descent.

The paper proves a sample-level almost-sure convergence theorem and an explicit
non-asymptotic rate of convergence, and recovers the kernel-MMD-penalty
Schrödinger bridge problem of [Nakano 2024] as the special case of a vanishing
interaction cost.
Numerical experiments illustrate the method on the Schrödinger bridge problem
in dimensions up to *d* = 100 and on an electric vehicle charging coordination
MFG with a physical per-vehicle heterogeneity (log charging-speed multiplier).

## Repository structure

```
.
├── kernel_sot/                              # Importable Python package
│   ├── estimators.py                        #   MMD^2 estimators: kernel U-stat,
│   │                                        #     RF U-stat, RFF V-stat,
│   │                                        #     interaction RF U-stat
│   ├── networks.py                          #   DriftNet (controlled SDE drift)
│   ├── sde.py                               #   Euler-Maruyama integration
│   └── utils.py                             #   Reproducibility helpers
├── experiments/                             # One script per subsection of §5
│   ├── exp_5_1_1_bias.py                    # §5.1.1  Bias comparison
│   ├── exp_5_1_2_variance.py                # §5.1.2  Variance scaling (terminal)
│   ├── exp_5_1_3_interaction_estimator.py   # §5.1.3  Interaction estimator
│   │                                        #         (bias + variance, Thm 3.3)
│   ├── exp_5_1_4_vstat_penalty.py           # §5.1.4  Penalty bias in SBP
│   │                                        #         training (Gaussian shift)
│   ├── exp_5_1_4_vstat_bimodal.py           # §5.1.4  Penalty bias, bimodal
│   │                                        #         supplementary
│   ├── exp_5_2_bimodal.py                   # §5.2    Bimodal target (d=2)
│   ├── exp_5_3_gauss_shift.py               # §5.3    High-dim Gaussian shift
│   ├── exp_5_4_1_lambda_sweep.py            # §5.4.1  Penalty parameter sweep
│   ├── exp_5_4_2_kernel_limit.py            # §5.4.2  Kernel U-stat limit (M to inf)
│   ├── exp_5_5_compute_scaling.py           # §5.5    Computational scaling
│   └── exp_5_6_ev_charging.py               # §5.6    EV charging fleet MFG
├── results/                                 # Outputs of experiments (re-generated)
├── requirements.txt
├── pyproject.toml
└── LICENSE
```

## Installation

```bash
git clone https://github.com/yumiharu-nakano/kernelMFG-RFU.git
cd kernelMFG-RFU
pip install -e .
```

This installs the `kernel_sot` package in development mode and pulls in the
dependencies (PyTorch, NumPy, SciPy, Matplotlib).

Tested with Python 3.9–3.12 and PyTorch ≥ 2.0 on CPU. CUDA and MPS work for the
larger experiments but are not required.

## Quick start

```python
import torch
from kernel_sot import mmd2_rf_ustat, interaction_rf_ustat

# Terminal MMD^2 estimator (unbiased, O(NM)):
X = torch.randn(200, 10)
Y = 0.5 + torch.randn(200, 10)
gamma2 = mmd2_rf_ustat(X, Y, M=500, alpha=0.1)
print(f"hat_gamma^2(X, Y) = {gamma2.item():.4f}")

# Interaction (self-kernel) estimator (unbiased, O(NM)):
R = interaction_rf_ustat(X, M=500, alpha=0.1)
print(f"hat_R[hat_nu^N]   = {R.item():.4f}")
```

Reproduce the interaction-estimator microbenchmark of §5.1.3
(verifies Theorem 3.3: unbiasedness and Var = O(1/M) + O(1/N)):

```bash
python experiments/exp_5_1_3_interaction_estimator.py
```

Reproduce the EV charging fleet potential MFG of §5.6 (d=2, default settings):

```bash
python experiments/exp_5_6_ev_charging.py --d 2 --c 100 --seeds 0 1 2
```

## Running individual experiments

Every experiment script supports `--help`. Common flags:

| Flag                                 | Meaning                                                |
| ------------------------------------ | ------------------------------------------------------ |
| `--seeds 0 1 2 3 4`                  | Random seeds                                           |
| `--epochs N`                         | Number of training epochs                              |
| `--output_dir DIR`                   | Where to write JSON and plots (default `results/`)     |
| `--N`, `--M`, `--alpha`, `--lam_inv` | SBP / estimator hyperparameters                        |

Examples:

```bash
# Single-seed quick check of the high-dim Gaussian shift SBP at d=10:
python experiments/exp_5_3_gauss_shift.py --dims 10 --seeds 0

# Penalty sweep at d=10:
python experiments/exp_5_4_1_lambda_sweep.py --d 10 --lam_invs 1e-2 1e-3 1e-4 --seeds 0 1 2

# Computational scaling (kernel vs RF U-stat, MMD only):
python experiments/exp_5_5_compute_scaling.py
```

## Mapping of paper sections to scripts

| Paper section                                       | Script                                       |
| --------------------------------------------------- | -------------------------------------------- |
| §5.1.1  Bias comparison (terminal estimator)        | `exp_5_1_1_bias.py`                          |
| §5.1.2  Variance scaling (terminal estimator)       | `exp_5_1_2_variance.py`                      |
| §5.1.3  Interaction estimator (bias and variance)   | `exp_5_1_3_interaction_estimator.py`         |
| §5.1.4  Effect of the penalty bias on SBP training  | `exp_5_1_4_vstat_penalty.py` and `exp_5_1_4_vstat_bimodal.py` |
| §5.2    Bimodal target (d = 2)                      | `exp_5_2_bimodal.py`                         |
| §5.3    High-dimensional Gaussian shift             | `exp_5_3_gauss_shift.py`                     |
| §5.4.1  Penalty parameter sweep                     | `exp_5_4_1_lambda_sweep.py`                  |
| §5.4.2  Kernel *U*-statistic limit                  | `exp_5_4_2_kernel_limit.py`                  |
| §5.5    Computational scaling                       | `exp_5_5_compute_scaling.py`                 |
| §5.6    Potential MFG: EV charging fleet            | `exp_5_6_ev_charging.py`                     |

## Implementation notes

* All experiments fit on a single CPU. The Gaussian-shift *d* = 100 sweep is
  the longest single run (a few hours).
* The estimators in `kernel_sot.estimators` are stateless functions; reseed via
  `kernel_sot.utils.set_seed` if you need deterministic outputs.
* The kernel-MMD-penalty Schrödinger bridge of Nakano (2024)
  ([JJIAM, to appear](https://arxiv.org/abs/2310.14522))
  is recovered by setting the congestion weight `c = 0` in the algorithm.

## Citation

```bibtex
@article{nakano2026mfg,
  title  = {Kernel-based potential mean-field games with unbiased random {Fourier} {$U$}-statistics},
  author = {Nakano, Yumiharu},
  journal = {arXiv preprint arXiv:2605.29371 [math.OC]},
  year   = {2026},
}
```

The Schrödinger-bridge predecessor:

```bibtex
@article{nakano2024sb,
  title   = {A kernel-based method for {Schr{\"o}dinger} bridges},
  author  = {Nakano, Yumiharu},
  journal = {Japan Journal of Industrial and Applied Mathematics},
  year    = {2026},
  note    = {to appear; preprint at arXiv:2310.14522 [math.OC]}
}
```

## License

MIT. See [LICENSE](LICENSE).
