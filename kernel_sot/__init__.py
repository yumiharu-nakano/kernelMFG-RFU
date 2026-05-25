"""
kernel_sot: Kernel-based Schrödinger bridge via random Fourier U-statistics.

Reference implementation for the paper
  "High-dimensional Schrödinger bridges via unbiased random Fourier U-statistics".

Modules:
  estimators  -- MMD^2 estimators: kernel U-stat, RF U-stat, RFF V-stat
  networks    -- DriftNet (controlled SDE drift)
  sde         -- Euler-Maruyama integration (with control cost tracking)
  utils       -- seeding, logging
"""

from .estimators import (
    mmd2_kernel_ustat,
    mmd2_rf_ustat,
    mmd2_rff_vstat,
    interaction_rf_ustat,
    interaction_v_stat,
)
from .networks import DriftNet
from .sde import sde_integrate, sde_integrate_with_cost
from .utils import set_seed, ensure_dir

__all__ = [
    "mmd2_kernel_ustat",
    "mmd2_rf_ustat",
    "mmd2_rff_vstat",
    "interaction_rf_ustat",
    "interaction_v_stat",
    "DriftNet",
    "sde_integrate",
    "sde_integrate_with_cost",
    "set_seed",
    "ensure_dir",
]

__version__ = "0.1.0"
