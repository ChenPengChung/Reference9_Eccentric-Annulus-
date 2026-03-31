"""
Bipolar Coordinate System Constants
====================================
Compute alpha, beta, c, gamma_shift from geometry (r1, r2, eccentricity).

Two formulations are implemented:
  1. PDF formulation (Eqs. 4a-4e): uses gamma=r1/r2, phi=e/(r2-r1)
  2. Snyder-Goldstein formulation: uses F, M intermediate variables

Both are mathematically equivalent (diff < 1e-15 for c).
The Snyder convention is used internally (alpha=eta_min, beta=eta_max).
"""

import numpy as np
import warnings


def compute_bipolar_constants_pdf(r1, r2, e):
    """
    Compute bipolar constants using the PDF document formulation (Eqs. 4a-4e).

    Parameters
    ----------
    r1 : float
        Inner cylinder radius (r1 > 0).
    r2 : float
        Outer cylinder radius (r2 > r1).
    e : float
        Physical eccentricity distance (0 < e < r2 - r1).

    Returns
    -------
    dict with keys:
        alpha_pdf : float  -- eta boundary for INNER circle (larger value)
        beta_pdf  : float  -- eta boundary for OUTER circle (smaller value)
        c         : float  -- pole distance
        gamma_r   : float  -- radius ratio r1/r2
        phi       : float  -- relative eccentricity e/(r2-r1)

    Notes
    -----
    In the PDF convention:
      alpha -> inner circle (larger eta value)
      beta  -> outer circle (smaller eta value)
    This is SWAPPED relative to the Snyder convention used internally.
    """
    if r1 <= 0:
        raise ValueError(f"r1 must be > 0, got {r1}")
    if r2 <= r1:
        raise ValueError(f"r2 must be > r1, got r2={r2}, r1={r1}")
    if e <= 0:
        raise ValueError(
            "Eccentricity must be > 0. Bipolar coordinates are singular "
            "for concentric annuli (e=0). Use polar coordinates instead."
        )
    if e >= r2 - r1:
        raise ValueError(
            f"Eccentricity e={e} must be < gap={r2 - r1}. "
            "The inner circle must not touch or intersect the outer circle."
        )

    gamma_r = r1 / r2
    phi = e / (r2 - r1)

    # Eq. (4b): cosh(alpha) = (1/gamma) * [gamma*(1+phi^2) + (1-phi^2)] / (2*phi)
    cosh_alpha = (1.0 / gamma_r) * (gamma_r * (1 + phi**2) + (1 - phi**2)) / (2 * phi)

    # Eq. (4c): cosh(beta) = [gamma*(1-phi^2) + (1+phi^2)] / (2*phi)
    cosh_beta = (gamma_r * (1 - phi**2) + (1 + phi**2)) / (2 * phi)

    # Validate arccosh inputs (must be >= 1)
    if cosh_alpha < 1.0:
        raise ValueError(
            f"cosh(alpha) = {cosh_alpha:.6f} < 1: invalid geometry "
            f"(r1={r1}, r2={r2}, e={e})")
    if cosh_beta < 1.0:
        raise ValueError(
            f"cosh(beta) = {cosh_beta:.6f} < 1: invalid geometry "
            f"(r1={r1}, r2={r2}, e={e})")

    alpha_pdf = np.arccosh(cosh_alpha)
    beta_pdf = np.arccosh(cosh_beta)

    # Eq. (4a): c = r1 * sinh(alpha)
    c = r1 * np.sinh(alpha_pdf)

    return {
        "alpha_pdf": alpha_pdf,
        "beta_pdf": beta_pdf,
        "c": c,
        "gamma_r": gamma_r,
        "phi": phi,
    }


def compute_bipolar_constants_snyder(R1, R2, epsilon):
    """
    Compute bipolar constants using the Snyder-Goldstein formulation.

    This matches the existing eccentric_annulus_grid_tool.py implementation.

    Parameters
    ----------
    R1 : float
        Inner radius.
    R2 : float
        Outer radius (R2 > R1).
    epsilon : float
        Relative eccentricity (0 < epsilon < 1).
        Physical offset = epsilon * (R2 - R1).

    Returns
    -------
    dict with keys:
        alpha       : float  -- eta_min (maps to OUTER circle R2)
        beta        : float  -- eta_max (maps to INNER circle R1)
        c           : float  -- pole distance (= M)
        gamma_shift : float  -- vertical shift to center outer circle at origin
        F           : float  -- intermediate variable
        shift       : float  -- physical eccentricity distance

    Notes
    -----
    In the Snyder convention:
      alpha = smaller eta value -> outer boundary (R2)
      beta  = larger eta value  -> inner boundary (R1)
    """
    if R1 <= 0:
        raise ValueError(f"R1 must be > 0, got {R1}")
    if R2 <= R1:
        raise ValueError(f"R2 must be > R1, got R2={R2}, R1={R1}")
    if epsilon <= 0 or epsilon >= 1:
        raise ValueError(
            f"epsilon must be in (0, 1), got {epsilon}. "
            "Use 0 < epsilon < 1 for eccentric annulus."
        )

    shift = epsilon * (R2 - R1)
    b = abs(shift)

    F = (R2**2 - R1**2 + b**2) / (2 * b)
    disc = F**2 - R2**2
    if disc < 0:
        raise ValueError(
            f"F^2 - R2^2 = {disc:.6e} < 0: invalid geometry "
            f"(R1={R1}, R2={R2}, epsilon={epsilon})")
    M = np.sqrt(disc)

    alpha = 0.5 * np.log((F + M) / (F - M))
    beta = 0.5 * np.log((F - b + M) / (F - b - M))

    c = M
    gamma_shift = M / np.tanh(alpha)

    return {
        "alpha": alpha,
        "beta": beta,
        "c": c,
        "gamma_shift": gamma_shift,
        "F": F,
        "shift": shift,
    }


def cross_validate_constants(r1, r2, e, tol=1e-10):
    """
    Compute constants from BOTH methods and verify they match.

    Parameters
    ----------
    r1, r2 : float
        Inner and outer radii.
    e : float
        Physical eccentricity distance.
    tol : float
        Tolerance for discrepancy warning.

    Returns
    -------
    dict with unified constants (Snyder convention):
        alpha       : float  -- eta_min (outer circle)
        beta        : float  -- eta_max (inner circle)
        c           : float  -- pole distance
        gamma_shift : float  -- vertical shift
        gamma_r     : float  -- radius ratio r1/r2
        phi         : float  -- relative eccentricity
        F           : float  -- Snyder intermediate
        shift       : float  -- physical eccentricity
        kappa       : float  -- inverse radius ratio r2/r1

    Raises
    ------
    Warning if discrepancy > tol.
    """
    epsilon = e / (r2 - r1)

    pdf = compute_bipolar_constants_pdf(r1, r2, e)
    sny = compute_bipolar_constants_snyder(r1, r2, epsilon)

    # Cross-check c values
    diff_c = abs(pdf["c"] - sny["c"])
    if diff_c > tol:
        warnings.warn(
            f"Bipolar constant 'c' mismatch: PDF={pdf['c']:.15e}, "
            f"Snyder={sny['c']:.15e}, diff={diff_c:.2e}"
        )

    # Cross-check: PDF alpha should match Snyder beta (both = inner circle)
    diff_inner = abs(pdf["alpha_pdf"] - sny["beta"])
    if diff_inner > tol:
        warnings.warn(
            f"Inner circle eta mismatch: PDF_alpha={pdf['alpha_pdf']:.15e}, "
            f"Snyder_beta={sny['beta']:.15e}, diff={diff_inner:.2e}"
        )

    # Cross-check: PDF beta should match Snyder alpha (both = outer circle)
    diff_outer = abs(pdf["beta_pdf"] - sny["alpha"])
    if diff_outer > tol:
        warnings.warn(
            f"Outer circle eta mismatch: PDF_beta={pdf['beta_pdf']:.15e}, "
            f"Snyder_alpha={sny['alpha']:.15e}, diff={diff_outer:.2e}"
        )

    # Cross-check gamma_shift
    diff_gamma = abs(pdf.get("gamma_shift", 0) - sny.get("gamma_shift", 0))
    if diff_gamma > tol:
        warnings.warn(
            f"gamma_shift mismatch: PDF={pdf.get('gamma_shift', 0):.15e}, "
            f"Snyder={sny.get('gamma_shift', 0):.15e}, diff={diff_gamma:.2e}"
        )

    # Validation summary
    all_ok = (diff_c <= tol) and (diff_inner <= tol) and (diff_outer <= tol)
    validation = {
        "passed": all_ok,
        "diff_c": diff_c,
        "diff_inner": diff_inner,
        "diff_outer": diff_outer,
    }

    # Return unified result using Snyder convention:
    #   alpha = outer circle (smaller eta)
    #   beta  = inner circle (larger eta)
    return {
        "c": sny["c"],
        "alpha": sny["alpha"],       # outer circle eta (smaller)
        "beta": sny["beta"],         # inner circle eta (larger)
        "gamma_shift": sny.get("gamma_shift", 0.0),
        "gamma_r": pdf.get("gamma_r", r1 / r2),
        "phi": pdf.get("phi", epsilon),
        "kappa": 0.0,
        "alpha_pdf": pdf.get("alpha_pdf", sny["beta"]),
        "beta_pdf": pdf.get("beta_pdf", sny["alpha"]),
        "validation": validation,
    }