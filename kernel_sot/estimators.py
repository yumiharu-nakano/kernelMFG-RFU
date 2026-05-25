"""
MMD^2 estimators for the Gaussian kernel K(x, y) = exp(-alpha |x - y|^2).

Three estimators, all unbiased only for the kernel and RF U-statistic:

  mmd2_kernel_ustat(X, Y, alpha)
      Standard kernel U-statistic, O(N^2) complexity.

  mmd2_rf_ustat(X, Y, M, alpha)
      Proposed: random Fourier U-statistic, O(NM) complexity, unbiased.

  mmd2_rff_vstat(X, Y, M, alpha)
      Standard RFF V-statistic (Rahimi-Recht), O(NM) complexity, biased.
"""

import math
import torch


def mmd2_kernel_ustat(X: torch.Tensor, Y: torch.Tensor, alpha: float) -> torch.Tensor:
    r"""
    Kernel U-statistic estimator of $\gamma_K^2(\mu, \nu)$ for the Gaussian kernel.

        bar_gamma_K^2(D) = 1/(N(N-1)) sum_{i!=j} K(X_i, X_j)
                          - 2/N^2 sum_{i,j} K(X_i, Y_j)
                          + 1/(N(N-1)) sum_{i!=j} K(Y_i, Y_j).

    Complexity: O(N^2) for batch size N.

    Parameters
    ----------
    X, Y : (N, d) tensors of i.i.d. samples from mu and nu, respectively.
    alpha : kernel bandwidth.

    Returns
    -------
    A scalar tensor (differentiable w.r.t. X if needed).
    """
    N_X, N_Y = X.shape[0], Y.shape[0]
    XX = (X * X).sum(-1)
    YY = (Y * Y).sum(-1)
    K_XX = torch.exp(-alpha * (XX[:, None] + XX[None, :] - 2 * X @ X.T))
    K_YY = torch.exp(-alpha * (YY[:, None] + YY[None, :] - 2 * Y @ Y.T))
    K_XY = torch.exp(-alpha * (XX[:, None] + YY[None, :] - 2 * X @ Y.T))
    diag_X = 1 - torch.eye(N_X, device=X.device)
    diag_Y = 1 - torch.eye(N_Y, device=Y.device)
    U_XX = (K_XX * diag_X).sum() / (N_X * (N_X - 1))
    U_YY = (K_YY * diag_Y).sum() / (N_Y * (N_Y - 1))
    V_XY = K_XY.sum() / (N_X * N_Y)
    return U_XX - 2 * V_XY + U_YY


def mmd2_rf_ustat(
    X: torch.Tensor, Y: torch.Tensor, M: int, alpha: float
) -> torch.Tensor:
    r"""
    Random Fourier $U$-statistic estimator (proposed in this paper):

        hat_gamma_{M, N}^2 = (Phi(0)/M) sum_r [U_XX(Z_r) - 2 V_XY(Z_r) + U_YY(Z_r)]

    where Z_r ~ N(0, 2 alpha I_d) and U_XX, V_XY, U_YY are computed via
    sums of cos/sin in O(N) per frequency. Complexity: O(NM).

    Unbiased for $\gamma_K^2(\mu, \nu)$ for any M >= 1, N >= 2.

    Parameters
    ----------
    X, Y : (N, d) tensors.
    M : number of random frequencies.
    alpha : kernel bandwidth.
    """
    d = X.shape[1]
    N_X, N_Y = X.shape[0], Y.shape[0]
    Z = torch.randn(M, d, device=X.device) * math.sqrt(2.0 * alpha)
    pX = X @ Z.T
    pY = Y @ Z.T
    cX, sX = pX.cos().sum(0), pX.sin().sum(0)
    cY, sY = pY.cos().sum(0), pY.sin().sum(0)
    # U-statistics: (sum cos)^2 + (sum sin)^2 - N over N(N-1)
    U_XX = (cX ** 2 + sX ** 2 - N_X) / (N_X * (N_X - 1))
    U_YY = (cY ** 2 + sY ** 2 - N_Y) / (N_Y * (N_Y - 1))
    # V-statistic for the cross term (independent samples, naturally unbiased)
    V_XY = (cX * cY + sX * sY) / (N_X * N_Y)
    return (U_XX - 2 * V_XY + U_YY).mean()


def mmd2_rff_vstat(
    X: torch.Tensor, Y: torch.Tensor, M: int, alpha: float
) -> torch.Tensor:
    r"""
    Standard RFF V-statistic estimator (Rahimi-Recht 2007):

        V = || (1/N) sum_i phi(X_i) - (1/N) sum_j phi(Y_j) ||^2

    where phi(x) = (cos(z^T x), sin(z^T x)) for z ~ N(0, 2 alpha I_d).
    Has positive bias of order O(Phi(0) / N) when mu = nu (cf. Remark 3.2 in paper).
    """
    d = X.shape[1]
    N_X, N_Y = X.shape[0], Y.shape[0]
    Z = torch.randn(M, d, device=X.device) * math.sqrt(2.0 * alpha)
    pX = X @ Z.T
    pY = Y @ Z.T
    cX, sX = pX.cos().sum(0), pX.sin().sum(0)
    cY, sY = pY.cos().sum(0), pY.sin().sum(0)
    # V-statistic uses 1/N^2 with diagonal included
    V_XX = (cX ** 2 + sX ** 2) / (N_X * N_X)
    V_YY = (cY ** 2 + sY ** 2) / (N_Y * N_Y)
    V_XY = (cX * cY + sX * sY) / (N_X * N_Y)
    return (V_XX - 2 * V_XY + V_YY).mean()


def interaction_rf_ustat(
    X: torch.Tensor, M: int, alpha: float
) -> torch.Tensor:
    r"""
    Random Fourier U-statistic estimator for the kernel self-interaction cost

        R[nu] := (1/2) int int W(x, y) nu(dx) nu(dy),

    where W(x, y) = exp(-alpha |x - y|^2) is the Gaussian kernel
    (translation-invariant, positive Fourier transform: Psi(0) = 1, Z ~ N(0, 2 alpha I_d)).

    Construction (cf. Section 3.1.interaction and Theorem 3.2 of the MFG paper):

        hat_R_{M, N}[hat_nu^N]
            = (Psi(0) / (2 M)) sum_r U_XX(tilde Z_r)
            = (1 / (2 M)) sum_r (S_c^2 + S_s^2 - N) / (N (N - 1))

    where S_c(z) = sum_i cos(z^T X_i), S_s(z) = sum_i sin(z^T X_i),
    and the random frequencies tilde Z_r ~ N(0, 2 alpha I_d) are independent of the samples.

    Properties (Theorem 3.2):
      - Unbiased for R[nu] for any M >= 1, N >= 2.
      - Variance = O(1/M) + O(1/N).
      - O(N M) cost.

    The estimator removes the diagonal self-term of the V-statistic via the (S_c^2 + S_s^2 - N)
    expression, exactly as in the terminal MMD estimator mmd2_rf_ustat.

    Parameters
    ----------
    X     : (N, d) tensor of i.i.d. samples from nu.
    M     : number of random frequencies.
    alpha : kernel bandwidth (W(x,y) = exp(-alpha |x-y|^2)).

    Returns
    -------
    scalar torch.Tensor estimating R[nu].
    """
    d = X.shape[1]
    N = X.shape[0]
    if N < 2:
        raise ValueError(f"interaction_rf_ustat requires N >= 2, got N={N}")
    Z = torch.randn(M, d, device=X.device) * math.sqrt(2.0 * alpha)
    pX = X @ Z.T                              # (N, M)
    cX = pX.cos().sum(0)                      # (M,)
    sX = pX.sin().sum(0)                      # (M,)
    # U-statistic: (S_c^2 + S_s^2 - N) / (N (N - 1))
    U_XX = (cX ** 2 + sX ** 2 - N) / (N * (N - 1))
    # R[nu] has the factor 1/2 baked in: Psi(0) = 1 for Gaussian, so coefficient = 1/(2 M).
    return U_XX.mean() / 2.0


def interaction_v_stat(
    X: torch.Tensor, M: int, alpha: float
) -> torch.Tensor:
    r"""
    Standard V-statistic counterpart to interaction_rf_ustat, retained for
    sanity checks and bias comparisons.

        hat_R_V[hat_nu^N] = (1 / (2 M)) sum_r (S_c^2 + S_s^2) / N^2

    Has positive bias of order Psi(0) / (2 N) since the diagonal self-term
    contributes (1 / (2 N)) Psi(0) = 1 / (2 N) for the Gaussian kernel.
    Not used for training in the main paper (we use the unbiased U-statistic).
    """
    d = X.shape[1]
    N = X.shape[0]
    Z = torch.randn(M, d, device=X.device) * math.sqrt(2.0 * alpha)
    pX = X @ Z.T
    cX = pX.cos().sum(0)
    sX = pX.sin().sum(0)
    V_XX = (cX ** 2 + sX ** 2) / (N * N)
    return V_XX.mean() / 2.0
