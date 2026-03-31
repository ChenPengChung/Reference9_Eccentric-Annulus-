"""
Grid Stretching & GILBM Stability Estimation
==============================================
Vinokur two-sided tanh stretching for wall clustering, plus
pre-computed GILBM stability reference tables specifically calibrated
for eccentric annulus bipolar coordinate grids.

KEY DIFFERENCE from Periodic Hill (grid_zeta_tool.py):
  The bipolar coordinate mapping has INHERENT metric non-uniformity
  (dr_ratio ~8.8 even at GAMMA=0). The stability window for GAMMA
  is therefore MUCH TIGHTER than the periodic hill case.
"""

import numpy as np


# ============================================================
#  1. Vinokur Tanh Stretching
# ============================================================

def vinokur_tanh(eta, gamma, alpha=0.5):
    """
    Vinokur two-sided tanh clustering.  eta in [0,1].
    gamma=0 => identity (uniform).  Monotonic for all gamma >= 0.

    Parameters
    ----------
    eta : ndarray
        Uniform parameter in [0, 1].
    gamma : float
        Stretching intensity (GAMMA).
        0.0 = uniform, higher = stronger wall clustering.
    alpha : float
        Symmetry parameter (ALPHA).
        0.5 = symmetric (both walls equal).
        <0.5 = cluster toward eta_min (outer wall).
        >0.5 = cluster toward eta_max (inner wall).

    Returns
    -------
    zeta : ndarray
        Stretched parameter in [0, 1].
    """
    if gamma < 1e-14:
        return eta.copy()
    denom = np.tanh(gamma * alpha)
    if abs(denom) < 1e-30:
        return eta.copy()
    zeta = 0.5 * (1.0 + np.tanh(gamma * (eta - alpha)) / denom)
    zeta[0] = 0.0
    zeta[-1] = 1.0
    return zeta


def apply_radial_stretching(N_eta, alpha_bip, beta_bip, gamma, alpha_s=0.5):
    """
    Generate non-uniform eta distribution with Vinokur wall clustering.

    Maps uniform t in [0,1] -> stretched t -> eta in [alpha_bip, beta_bip].

    For LBM on eccentric annulus:
      - eta_min = alpha_bip corresponds to OUTER wall (R2)
      - eta_max = beta_bip  corresponds to INNER wall (R1)
      - alpha_s < 0.5: cluster toward outer wall
      - alpha_s > 0.5: cluster toward inner wall (narrow gap region)
      - alpha_s = 0.5: symmetric clustering (RECOMMENDED)

    Parameters
    ----------
    N_eta : int
        Number of radial grid points.
    alpha_bip, beta_bip : float
        Bipolar coordinate boundaries.
    gamma : float
        Vinokur stretching intensity.
    alpha_s : float
        Symmetry parameter.

    Returns
    -------
    eta : ndarray (N_eta,)
        Stretched eta distribution.
    """
    t = np.linspace(0, 1, N_eta)
    if gamma > 1e-14:
        t = vinokur_tanh(t, gamma, alpha_s)
    eta = alpha_bip + t * (beta_bip - alpha_bip)
    return eta


def apply_circumferential_stretching(N_xi, gamma_xi, alpha_xi=0.5):
    """
    Generate non-uniform xi distribution (optional, usually uniform).

    For high eccentricity, can cluster points near xi=+/-pi (narrow gap)
    or xi=0 (wide gap) to balance resolution.

    NOTE: circumferential stretching is rarely needed for bipolar grids
    because the conformal mapping already concentrates physical-space
    points in the narrow gap region.

    Parameters
    ----------
    N_xi : int
        Number of circumferential grid points.
    gamma_xi : float
        Vinokur stretching intensity (0 = uniform).
    alpha_xi : float
        Symmetry parameter.

    Returns
    -------
    xi : ndarray (N_xi,)
        xi distribution in [-pi, pi].
    """
    t = np.linspace(0, 1, N_xi)
    if gamma_xi > 1e-14:
        t = vinokur_tanh(t, gamma_xi, alpha_xi)
    xi = -np.pi + t * 2 * np.pi
    return xi


# ============================================================
#  2. GILBM Stability Estimation
# ============================================================

def estimate_gilbm_stability_eccentric(x_grid, y_grid,
                                        Uref=0.05, Re=100, L_char=None,
                                        CFL_lambda=0.5):
    """
    Estimate GILBM stability parameters for the eccentric annulus grid.

    The MRT collision operator requires omega in [0.5, 2.0].
    omega = 0.5 + 3 * niu / dt_global
    dt_global = CFL_lambda / max|c_tilde|

    For D2Q9:
      c_tilde_xi  = xi_x * e_x + xi_y * e_y
      c_tilde_eta = eta_x * e_x + eta_y * e_y

    where (xi_x, xi_y, eta_x, eta_y) = inverse metric tensor components.

    Parameters
    ----------
    x_grid, y_grid : ndarray (N_eta, N_xi)
        Physical grid coordinates.
    Uref : float
        Reference velocity (lattice units).
    Re : float
        Reynolds number.
    L_char : float or None
        Characteristic length (default: estimated from grid extent).
    CFL_lambda : float
        CFL number (default 0.5).

    Returns
    -------
    dict: omega, c_max, dt_global, dr_min, dr_max, dr_ratio, status
    """
    nj, ni = x_grid.shape

    if L_char is None:
        # Estimate from grid extent (roughly the gap)
        r_outer = np.sqrt(x_grid[0, :]**2 + y_grid[0, :]**2).mean()
        r_inner = np.sqrt(x_grid[-1, :]**2 + y_grid[-1, :]**2).mean()
        L_char = abs(r_outer - r_inner)
        if L_char < 1e-10:
            L_char = 1.0

    niu = Uref * L_char / Re

    # D2Q9 velocity set: e_x, e_y components
    e_x = [0, 1, 0, -1,  0, 1, -1, -1,  1]
    e_y = [0, 0, 1,  0, -1, 1,  1, -1, -1]

    # Numerical Jacobian (central FD, one-sided at boundaries)
    x_xi = np.zeros_like(x_grid)
    y_xi = np.zeros_like(y_grid)
    x_eta = np.zeros_like(x_grid)
    y_eta = np.zeros_like(y_grid)

    # xi derivatives (along i-direction)
    x_xi[:, 1:-1] = (x_grid[:, 2:] - x_grid[:, :-2]) / 2.0
    y_xi[:, 1:-1] = (y_grid[:, 2:] - y_grid[:, :-2]) / 2.0
    x_xi[:, 0] = x_grid[:, 1] - x_grid[:, 0]
    x_xi[:, -1] = x_grid[:, -1] - x_grid[:, -2]
    y_xi[:, 0] = y_grid[:, 1] - y_grid[:, 0]
    y_xi[:, -1] = y_grid[:, -1] - y_grid[:, -2]

    # eta derivatives (along j-direction)
    x_eta[1:-1, :] = (x_grid[2:, :] - x_grid[:-2, :]) / 2.0
    y_eta[1:-1, :] = (y_grid[2:, :] - y_grid[:-2, :]) / 2.0
    x_eta[0, :] = x_grid[1, :] - x_grid[0, :]
    x_eta[-1, :] = x_grid[-1, :] - x_grid[-2, :]
    y_eta[0, :] = y_grid[1, :] - y_grid[0, :]
    y_eta[-1, :] = y_grid[-1, :] - y_grid[-2, :]

    # Jacobian determinant
    J = x_xi * y_eta - x_eta * y_xi

    # Inverse metric (contravariant base vectors)
    eps = 1e-30
    xi_x = np.where(np.abs(J) > eps, y_eta / J, 0)
    xi_y = np.where(np.abs(J) > eps, -x_eta / J, 0)
    eta_x = np.where(np.abs(J) > eps, -y_xi / J, 0)
    eta_y = np.where(np.abs(J) > eps, x_xi / J, 0)

    # Interior slice (avoid boundary FD artifacts)
    sl = (slice(1, -1), slice(1, -1))

    # Max contravariant velocity over all D2Q9 directions
    max_c = 0.0
    for a in range(1, 9):
        c_xi = np.abs(xi_x[sl] * e_x[a] + xi_y[sl] * e_y[a])
        c_eta = np.abs(eta_x[sl] * e_x[a] + eta_y[sl] * e_y[a])
        max_c = max(max_c, c_xi.max(), c_eta.max())

    # Radial spacing analysis
    dr_min = 1e30
    dr_max = 0.0
    for i in range(ni):
        dr = np.sqrt(np.diff(x_grid[:, i])**2 + np.diff(y_grid[:, i])**2)
        dr_pos = dr[dr > 0]
        if len(dr_pos) > 0:
            dr_min = min(dr_min, dr_pos.min())
            dr_max = max(dr_max, dr_pos.max())
    dr_ratio = dr_max / dr_min if dr_min > 0 else float("inf")

    # LBM parameters
    dt_global = CFL_lambda / max_c if max_c > 0 else 1.0
    omega = 0.5 + 3.0 * niu / dt_global

    # Status classification
    if omega > 2.0:
        status = "UNSTABLE"
    elif omega > 1.5:
        status = "MARGINAL"
    elif omega > 1.2:
        status = "OK"
    elif omega >= 0.55:
        status = "OPTIMAL"
    else:
        status = "GOOD"

    return {
        "omega": omega,
        "dt_global": dt_global,
        "c_max": max_c,
        "dr_min": dr_min,
        "dr_max": dr_max,
        "dr_ratio": dr_ratio,
        "status": status,
        "niu": niu,
        "L_char": L_char,
    }


# ============================================================
#  3. Pre-Computed Stability Tables
# ============================================================

def print_gilbm_stability_table_eccentric():
    """
    Print calibrated GILBM stability reference for eccentric annulus.

    Calibrated for: R1=1, R2=3, eps=0.5, Grid=200x80
    Flow: Uref=0.05, Re=100, L_char=2.0 (gap), CFL=0.5, ALPHA=0.5
    """
    print()
    print("  " + "=" * 78)
    print("   GILBM Stability Reference -- Eccentric Annulus (Bipolar Coordinates)")
    print("   R1=1, R2=3, eps=0.5, Grid=200x80, Re=100, Uref=0.05, CFL=0.5")
    print("  " + "=" * 78)
    print(f"  {'GAMMA':>6} | {'omega':>8} | {'max|c~|':>10} | {'dr_ratio':>10}"
          f" | {'Status':<12} | Note")
    print("  " + "-" * 78)

    table = [
        (0.0, 1.3500, 142,  8.8, "OK",       "UNIFORM (inherent bipolar non-uniformity)"),
        (0.2, 1.3553, 143,  8.8, "OK",       "Very mild clustering"),
        (0.4, 1.3714, 145,  8.8, "OK",       "Mild clustering"),
        (0.6, 1.3985, 150,  8.8, "OK",       "Mild clustering"),
        (0.8, 1.4374, 156,  8.9, "OK",       "Moderate clustering"),
        (1.0, 1.4889, 165,  8.9, "OK",       "Recommended (good balance)"),
        (1.2, 1.5541, 176,  8.9, "MARGINAL", "Strong clustering, approaching limit"),
        (1.5, 1.6814, 197,  9.0, "MARGINAL", "Strong clustering"),
        (2.0, 1.9897, 248, 11.1, "MARGINAL", "Very strong (near omega=2.0 limit!)"),
        (2.5, 2.4605, 327, 15.7, "UNSTABLE", "!! WILL DIVERGE !!"),
        (3.0, 3.1670, 445, 23.5, "UNSTABLE", "!! WILL DIVERGE !!"),
    ]
    for gamma, omega, c_max, ratio, status, note in table:
        marker = ""
        if gamma == 1.0:
            marker = " <-- RECOMMENDED"
        elif status == "UNSTABLE":
            marker = " ***"
        print(f"  {gamma:6.1f} | {omega:8.4f} | {c_max:10d} | {ratio:10.1f}"
              f" | {status:<12} | {note}{marker}")

    print("  " + "-" * 78)
    print()
    print("  !! KEY DIFFERENCE from Periodic Hill (grid_zeta_tool.py) !!")
    print("  The bipolar coordinate mapping has INHERENT metric non-uniformity:")
    print("    dr_ratio ~8.8 even with GAMMA=0 (uniform in computational space)")
    print("    max|c_tilde| ~142 at GAMMA=0 vs ~209 for periodic hill at GAMMA=0")
    print()
    print("  RECOMMENDATION for eccentric annulus bipolar grid:")
    print("    GAMMA = 0.0~1.0  -> OPTIMAL/OK range (omega < 1.5)")
    print("    GAMMA = 1.0      -> Best balance: wall clustering + stability")
    print("    GAMMA >= 2.0     -> DANGEROUS: omega approaches 2.0 limit")
    print("    GAMMA >= 2.5     -> UNSTABLE: omega > 2.0, WILL DIVERGE")
    print()
    print("  Compare with periodic hill (grid_zeta_tool.py):")
    print("    Periodic hill GAMMA=2.0 -> omega=0.63 (OPTIMAL)")
    print("    Eccentric ann GAMMA=2.0 -> omega=1.99 (MARGINAL!)")
    print("    The safe GAMMA here is ~HALF of the periodic hill value.")
    print()


def print_gilbm_stability_warning_eccentric(gamma, omega, c_max,
                                             dt_global, dr_ratio, status):
    """
    Print concise GILBM stability warning after grid generation.
    """
    print()
    print("  " + "=" * 62)
    print("   GILBM Stability Check (Eccentric Annulus)")
    print("  " + "=" * 62)
    print(f"    GAMMA         = {gamma:.4f}")
    print(f"    omega_global  = {omega:.4f}", end="")
    if omega > 2.0:
        print("  *** UNSTABLE (omega > 2.0) ***")
    elif omega > 1.5:
        print("  ** MARGINAL (omega > 1.5) **")
    elif omega > 1.2:
        print("  * OK (omega > 1.2)")
    else:
        print("  [OPTIMAL]")
    print(f"    max|c_tilde|  = {c_max:.1f}")
    print(f"    dt_global     = {dt_global:.4e}")
    print(f"    dr_ratio      = {dr_ratio:.1f} (includes bipolar metric effect)")
    print(f"    Status        = {status}")

    if omega > 2.0:
        print()
        print("  !! WARNING: This grid WILL DIVERGE in GILBM !!")
        print("  !! The bipolar mapping amplifies metric non-uniformity !!")
        print("  !! Reduce GAMMA to <= 1.0 for safe operation !!")
        print("  !! (Unlike periodic hill where GAMMA=2.0 is safe) !!")
    elif omega > 1.5:
        print()
        print("  ** CAUTION: Marginal stability. Bipolar coordinates add")
        print("     inherent dr_ratio ~8-9x. Consider reducing GAMMA to ~1.0.")
        print("     Alternatively, reduce grid resolution or increase Re.")
    elif omega > 1.2:
        print()
        print("  Note: omega > 1.2 due to bipolar coordinate metric effect.")
        print("  This is normal for eccentric annulus grids even at low GAMMA.")

    print("  " + "=" * 62)
    print()


def check_eccentricity_stability(epsilon, N_xi, N_eta, gamma_stretch):
    """
    Warn user if the chosen eccentricity + resolution + GAMMA combination
    is likely to cause stability issues.

    Parameters
    ----------
    epsilon : float
        Relative eccentricity (0 < eps < 1).
    N_xi, N_eta : int
        Grid resolution.
    gamma_stretch : float
        Vinokur stretching parameter.

    Returns
    -------
    dict with keys: risk_level, message, max_safe_gamma
    """
    if epsilon <= 0.3:
        max_safe = 1.5
        risk = "LOW"
        msg = "Low eccentricity: GAMMA up to 1.5 is safe."
    elif epsilon <= 0.5:
        max_safe = 1.0
        risk = "MODERATE"
        msg = "Moderate eccentricity: GAMMA up to 1.0 is recommended."
    elif epsilon <= 0.7:
        max_safe = 0.5
        risk = "HIGH"
        msg = ("High eccentricity: GAMMA should be <= 0.5. "
               "The narrow gap amplifies metric non-uniformity.")
    elif epsilon <= 0.9:
        max_safe = 0.0
        risk = "VERY HIGH"
        msg = ("Very high eccentricity: Recommend GAMMA=0 (uniform). "
               "Consider finer grid or alternative methods.")
    else:
        max_safe = 0.0
        risk = "EXTREME"
        msg = ("Extreme eccentricity (eps > 0.9): Very challenging. "
               "GAMMA=0 only. May need adaptive refinement.")

    # Resolution effect: higher N_eta increases omega
    if N_eta > 120 and gamma_stretch > 0.5:
        msg += (f" (N_eta={N_eta} is high; omega scales ~linearly with N_eta. "
                f"Consider reducing GAMMA.)")

    warning = None
    if gamma_stretch > max_safe:
        warning = (
            f"GAMMA={gamma_stretch:.1f} exceeds recommended maximum "
            f"of {max_safe:.1f} for eps={epsilon:.2f}. "
            f"Risk of GILBM instability."
        )

    return {
        "risk_level": risk,
        "message": msg,
        "max_safe_gamma": max_safe,
        "warning": warning,
    }


def estimate_max_resolution(R1, R2, epsilon, gamma_stretch=1.0,
                            Uref=0.05, Re=100, CFL=0.5, omega_limit=1.8):
    """
    Estimate the maximum safe grid resolution for a given geometry and GAMMA.

    max|c_tilde| scales approximately linearly with N_eta for fixed geometry.
    omega = 0.5 + 3*niu / (CFL/max_c) = 0.5 + 3*niu*max_c/CFL.

    Parameters
    ----------
    R1, R2, epsilon : float
        Geometry parameters.
    gamma_stretch : float
        Vinokur GAMMA.
    Uref, Re, CFL : float
        Flow parameters.
    omega_limit : float
        Target omega limit.

    Returns
    -------
    dict: estimated_max_Neta, notes
    """
    L_char = R2 - R1
    niu = Uref * L_char / Re

    # Empirical: max_c ~ k * N_eta, where k depends on geometry and GAMMA
    # From calibration (R1=1, R2=3, eps=0.5, GAMMA=1.0):
    #   N_eta=40 -> c_max~82, N_eta=80 -> c_max~165, N_eta=120 -> c_max~248
    # So k ~ 165/80 = 2.06 for this geometry

    # For omega_limit: omega = 0.5 + 3*niu*max_c/CFL
    # max_c_limit = (omega_limit - 0.5) * CFL / (3*niu)
    max_c_limit = (omega_limit - 0.5) * CFL / (3 * niu)

    # k factor scales with eccentricity and GAMMA
    # Base k at eps=0.5, GAMMA=0: ~142/80 = 1.775
    # With GAMMA=1.0: ~165/80 = 2.0625
    k_base = 1.775 * (1 + 0.5 * epsilon)  # higher eps -> higher k
    k_gamma = k_base * (1 + 0.15 * gamma_stretch)  # GAMMA effect

    if k_gamma > 0:
        est_max_Neta = int(max_c_limit / k_gamma)
    else:
        est_max_Neta = 999

    return {
        "estimated_max_Neta": est_max_Neta,
        "max_c_limit": max_c_limit,
        "k_factor": k_gamma,
        "notes": (
            f"Approximate limit for omega < {omega_limit}: N_eta ~ {est_max_Neta}. "
            f"Actual limit depends on grid details."
        ),
    }
