"""
Eccentric Annulus Grid Tool
===========================
Generate structured grids for eccentric annular domains using
conformal mapping (bipolar coordinates).

Based on the conformal mapping theory from:
  Lauer-Bare Z. & Gaertig E., "Conformal Mappings with SymPy",
  SciPy 2021 Proceedings.

Output format is compatible with grid_zeta_tool.py (Tecplot .dat).

Mathematical basis:
  Computational domain: rectangle [−π, π] × [α, β]
  Physical domain:      eccentric annulus with radii R1, R2, eccentricity ε

  Bipolar coordinate mapping:
    ζ = ξ + i·η    (computational coordinates)
    z = c · tan(ζ/2) − i·γ   (physical coordinates)

  where c (pole), α (bottom), β (top), γ (shift) are determined
  by R1, R2, and the eccentricity ε.

Usage:
  python eccentric_annulus_grid_tool.py              # interactive mode
  python eccentric_annulus_grid_tool.py --auto       # auto mode (from config)
  python eccentric_annulus_grid_tool.py --quick      # quick demo
"""

import sys
import re
import numpy as np
from pathlib import Path

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

try:
    import sympy as sym
    _HAS_SYMPY = True
except ImportError:
    _HAS_SYMPY = False


# ============================================================
#  1.  Bipolar Constants (from conformalMaps/mappings.py)
# ============================================================

def compute_bipolar_constants(R1, R2, epsilon):
    """
    Compute bipolar coordinate system constants for an eccentric annulus.

    Parameters
    ----------
    R1 : float
        Inner radius.
    R2 : float
        Outer radius.
    epsilon : float
        Relative eccentricity (0 ≤ ε < 1).
        Physical shift = ε × (R2 − R1).

    Returns
    -------
    alpha : float
        Bottom boundary of computational rectangle (η direction).
    beta : float
        Top boundary of computational rectangle (η direction).
    gamma : float
        Vertical shift to center outer circle at origin.
    c : float
        Pole of bipolar coordinate system.
    """
    shift = epsilon * (R2 - R1)

    if abs(shift) < 1e-15:
        raise ValueError(
            "Eccentricity is zero (concentric annulus). "
            "Bipolar mapping is singular for ε=0. Use polar coordinates instead."
        )

    # Use SymPy for exact symbolic computation, then convert to float
    if _HAS_SYMPY:
        R1s, R2s, bs = sym.symbols('R1 R2 b', real=True)
        Fs = (R2s**2 - R1s**2 + bs**2) / (2 * bs)
        Ms = sym.sqrt(Fs**2 - R2s**2)

        alphas = sym.Rational(1, 2) * sym.ln((Fs + Ms) / (Fs - Ms))
        betas = sym.Rational(1, 2) * sym.ln((Fs - bs + Ms) / (Fs - bs - Ms))

        subs = {R2s: R2, R1s: R1, bs: abs(shift)}

        F_val = float(Fs.subs(subs))
        M_val = float(Ms.subs(subs))
        alpha_val = float(alphas.subs(subs))
        beta_val = float(betas.subs(subs))
        c_val = M_val
        gamma_val = float((Ms * sym.coth(alphas)).subs(subs))
    else:
        # Pure numpy fallback
        b = abs(shift)
        F_val = (R2**2 - R1**2 + b**2) / (2 * b)
        M_val = np.sqrt(F_val**2 - R2**2)
        alpha_val = 0.5 * np.log((F_val + M_val) / (F_val - M_val))
        beta_val = 0.5 * np.log((F_val - b + M_val) / (F_val - b - M_val))
        c_val = M_val
        gamma_val = M_val / np.tanh(alpha_val)

    return alpha_val, beta_val, gamma_val, c_val


# ============================================================
#  2.  Grid Generation
# ============================================================

def generate_eccentric_annulus_grid(R1, R2, epsilon, Ni, Nj):
    """
    Generate a structured grid for an eccentric annulus.

    The grid is created by uniformly discretising the computational
    rectangle [−π, π] × [α, β] and then applying the bipolar
    conformal mapping.

    Parameters
    ----------
    R1 : float
        Inner radius.
    R2 : float
        Outer radius.
    epsilon : float
        Relative eccentricity (0 < ε < 1).
    Ni : int
        Number of grid points in the circumferential direction (ξ).
    Nj : int
        Number of grid points in the radial direction (η).

    Returns
    -------
    x : ndarray (Nj, Ni)
        Physical x-coordinates.
    y : ndarray (Nj, Ni)
        Physical y-coordinates.
    xi_grid : ndarray (Nj, Ni)
        Computational ξ-coordinates.
    eta_grid : ndarray (Nj, Ni)
        Computational η-coordinates.
    params : dict
        Dictionary of computed parameters (alpha, beta, gamma, c, R1, R2, epsilon).
    """
    alpha, beta, gamma, c = compute_bipolar_constants(R1, R2, epsilon)

    # Computational grid (uniform)
    xi = np.linspace(-np.pi, np.pi, Ni)
    eta = np.linspace(alpha, beta, Nj)
    Xi, Eta = np.meshgrid(xi, eta)

    # Conformal mapping: z = c * tan(ζ/2) − i·γ
    zeta = Xi + 1j * Eta
    z_phys = c * np.tan(zeta / 2)

    x = np.real(z_phys)
    y = np.imag(z_phys) - gamma

    params = {
        "R1": R1, "R2": R2, "epsilon": epsilon,
        "alpha": alpha, "beta": beta, "gamma": gamma, "c": c,
        "Ni": Ni, "Nj": Nj,
    }

    return x, y, Xi, Eta, params


def generate_eccentric_annulus_grid_stretched(R1, R2, epsilon, Ni, Nj,
                                              stretch_xi=0.0, stretch_eta=0.0,
                                              alpha_s=0.5):
    """
    Generate an eccentric annulus grid with optional Vinokur tanh stretching.

    Parameters
    ----------
    R1, R2, epsilon : float
        Geometry (same as generate_eccentric_annulus_grid).
    Ni, Nj : int
        Grid point counts.
    stretch_xi : float
        Vinokur stretching parameter for circumferential direction (0=uniform).
    stretch_eta : float
        Vinokur stretching parameter for radial direction (0=uniform).
    alpha_s : float
        Symmetry parameter for Vinokur stretching (0.5=symmetric).

    Returns
    -------
    x, y, xi_grid, eta_grid, params : same as generate_eccentric_annulus_grid
    """
    alpha, beta, gamma, c = compute_bipolar_constants(R1, R2, epsilon)

    # Computational grid with stretching
    t_xi = np.linspace(0, 1, Ni)
    t_eta = np.linspace(0, 1, Nj)

    if stretch_xi > 1e-14:
        t_xi = _vinokur_tanh(t_xi, stretch_xi, alpha_s)
    if stretch_eta > 1e-14:
        t_eta = _vinokur_tanh(t_eta, stretch_eta, alpha_s)

    xi = -np.pi + t_xi * 2 * np.pi
    eta = alpha + t_eta * (beta - alpha)

    Xi, Eta = np.meshgrid(xi, eta)

    zeta = Xi + 1j * Eta
    z_phys = c * np.tan(zeta / 2)

    x = np.real(z_phys)
    y = np.imag(z_phys) - gamma

    params = {
        "R1": R1, "R2": R2, "epsilon": epsilon,
        "alpha": alpha, "beta": beta, "gamma": gamma, "c": c,
        "Ni": Ni, "Nj": Nj,
        "stretch_xi": stretch_xi, "stretch_eta": stretch_eta,
        "alpha_s": alpha_s,
    }

    return x, y, Xi, Eta, params


def _vinokur_tanh(eta, gamma, alpha=0.5):
    """Vinokur two-sided tanh clustering. eta in [0,1]."""
    if gamma < 1e-14:
        return eta.copy()
    denom = np.tanh(gamma * alpha)
    if abs(denom) < 1e-30:
        return eta.copy()
    zeta = 0.5 * (1.0 + np.tanh(gamma * (eta - alpha)) / denom)
    zeta[0] = 0.0
    zeta[-1] = 1.0
    return zeta


# ============================================================
#  3.  Export: Tecplot .dat format (compatible with grid_zeta_tool)
# ============================================================

def write_tecplot_dat(filepath, x, y, title="Eccentric annulus grid",
                      zone_title="EccentricAnnulus"):
    """
    Write grid to Tecplot .dat file.

    Format is identical to grid_zeta_tool.py output:
      TITLE = "..."
      VARIABLES = "x corner"
      "y corner"
      ZONE T="..."
       I=Ni, J=Nj, K=1, F=POINT
      DT=(SINGLE SINGLE)
       x1 y1
       x2 y2
       ...

    Parameters
    ----------
    filepath : str or Path
        Output file path.
    x, y : ndarray (Nj, Ni)
        Grid coordinates.
    title : str
        Tecplot title.
    zone_title : str
        Zone title.
    """
    nj, ni = x.shape
    filepath = Path(filepath)
    with open(filepath, "w") as f:
        f.write(f'TITLE     = "{title}"\n')
        f.write('VARIABLES = "x corner"\n')
        f.write('"y corner"\n')
        f.write(f'ZONE T="{zone_title}"\n')
        f.write(f' I={ni}, J={nj}, K=1,F=POINT\n')
        f.write('DT=(SINGLE SINGLE )\n')
        for j in range(nj):
            for i in range(ni):
                f.write(f" {x[j, i]: .9E} {y[j, i]: .9E}\n")
    print(f"  [written] {filepath}")


def write_csv(filepath, x, y):
    """Write grid as CSV with columns: i, j, x, y."""
    nj, ni = x.shape
    filepath = Path(filepath)
    with open(filepath, "w") as f:
        f.write("i,j,x,y\n")
        for j in range(nj):
            for i in range(ni):
                f.write(f"{i},{j},{x[j, i]:.9E},{y[j, i]:.9E}\n")
    print(f"  [written] {filepath}")


# ============================================================
#  4.  Parser: read Tecplot .dat (for re-reading generated grids)
# ============================================================

def parse_tecplot_dat(filepath):
    """
    Parse a Tecplot .dat grid file.

    Returns
    -------
    x, y : ndarray (nj, ni)
    ni, nj : int
    """
    filepath = Path(filepath)
    with open(filepath, "r", encoding="latin-1") as f:
        lines = f.readlines()

    ni = nj = None
    header_lines = 0
    for idx, line in enumerate(lines):
        if "I=" in line.upper():
            parts = line.replace(",", " ").replace("=", " ").upper().split()
            for k, tok in enumerate(parts):
                if tok == "I":
                    ni = int(parts[k + 1])
                if tok == "J":
                    nj = int(parts[k + 1])
            header_lines = idx + 2
            break

    if ni is None or nj is None:
        raise ValueError("Cannot find I/J dimensions in header")

    data_lines = lines[header_lines:]
    x_flat, y_flat = [], []
    for dl in data_lines:
        dl = dl.strip()
        if not dl:
            continue
        vals = dl.split()
        if len(vals) >= 2:
            x_flat.append(float(vals[0]))
            y_flat.append(float(vals[1]))

    expected = ni * nj
    if len(x_flat) != expected:
        raise ValueError(
            f"Expected {expected} points (I={ni} x J={nj}), got {len(x_flat)}"
        )

    x = np.array(x_flat).reshape(nj, ni)
    y = np.array(y_flat).reshape(nj, ni)
    return x, y, ni, nj


# ============================================================
#  5.  Visualization
# ============================================================

def plot_grid(x, y, title="Eccentric Annulus Grid", savepath=None,
              figsize=(9, 9), show_boundary=True):
    """Plot the grid with inner/outer boundary highlighted."""
    if not _HAS_MPL:
        if savepath:
            print(f"  [skip plot] matplotlib not available: {savepath}")
        return
    nj, ni = x.shape
    fig, ax = plt.subplots(figsize=figsize)

    # Horizontal lines (constant η)
    for j in range(nj):
        if j == 0:
            color, lw = 'blue', 1.2
        elif j == nj - 1:
            color, lw = 'red', 1.2
        else:
            color, lw = 'grey', 0.3
        ax.plot(x[j, :], y[j, :], color=color, lw=lw)

    # Vertical lines (constant ξ)
    for i in range(ni):
        ax.plot(x[:, i], y[:, i], 'k-', lw=0.3)

    if show_boundary:
        # Close the inner boundary (j=0) and outer boundary (j=-1)
        ax.plot(np.append(x[0, :], x[0, 0]),
                np.append(y[0, :], y[0, 0]), 'b-', lw=1.5, label='Inner (R1)')
        ax.plot(np.append(x[-1, :], x[-1, 0]),
                np.append(y[-1, :], y[-1, 0]), 'r-', lw=1.5, label='Outer (R2)')
        ax.legend(fontsize=9)

    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.grid(True, ls='--', alpha=0.3)
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=200)
        print(f"  [saved] {savepath}")
    plt.close(fig)


def plot_computational_grid(xi, eta, title="Computational Domain",
                            savepath=None, figsize=(12, 6)):
    """Plot the computational (ξ, η) grid."""
    if not _HAS_MPL:
        if savepath:
            print(f"  [skip plot] matplotlib not available: {savepath}")
        return
    nj, ni = xi.shape
    fig, ax = plt.subplots(figsize=figsize)
    for j in range(nj):
        ax.plot(xi[j, :], eta[j, :], 'k-', lw=0.3)
    for i in range(ni):
        ax.plot(xi[:, i], eta[:, i], 'k-', lw=0.3)
    ax.set_xlabel('ξ')
    ax.set_ylabel('η')
    ax.set_title(title)
    ax.grid(True, ls='--', alpha=0.3)
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=200)
        print(f"  [saved] {savepath}")
    plt.close(fig)


def plot_compare(x1, y1, x2, y2, labels=("Grid 1", "Grid 2"),
                 title="Grid Comparison", savepath=None, figsize=(9, 18)):
    """Side-by-side comparison of two grids."""
    if not _HAS_MPL:
        if savepath:
            print(f"  [skip plot] matplotlib not available: {savepath}")
        return
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    for ax, xg, yg, lbl in zip(axes, [x1, x2], [y1, y2], labels):
        nj, ni = xg.shape
        for j in range(nj):
            ax.plot(xg[j, :], yg[j, :], 'k-', lw=0.25)
        for i in range(ni):
            ax.plot(xg[:, i], yg[:, i], 'k-', lw=0.25)
        ax.set_aspect('equal')
        ax.set_ylabel('y')
        ax.set_title(lbl)
    axes[-1].set_xlabel('x')
    fig.suptitle(title, fontsize=14, y=1.01)
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches='tight')
        print(f"  [saved] {savepath}")
    plt.close(fig)


def plot_radial_spacing(x, y, params, icol=0, savepath=None):
    """Plot radial (wall-normal) spacing at a given ξ column."""
    if not _HAS_MPL:
        if savepath:
            print(f"  [skip plot] matplotlib not available: {savepath}")
        return
    nj, ni = x.shape
    dr = np.sqrt(np.diff(x[:, icol])**2 + np.diff(y[:, icol])**2)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(nj - 1), dr, 'o-', ms=3)
    ax.set_xlabel('j index (radial)')
    ax.set_ylabel('Δr (spacing)')
    ax.set_title(f'Radial spacing at ξ column i={icol}')
    ax.grid(True, ls='--', alpha=0.4)
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=200)
        print(f"  [saved] {savepath}")
    plt.close(fig)


# ============================================================
#  6.  Grid Quality Metrics
# ============================================================

def compute_grid_quality(x, y):
    """
    Compute grid quality metrics.

    Returns
    -------
    dict with keys:
        min_area, max_area, area_ratio,
        min_aspect, max_aspect, mean_aspect,
        min_skew, max_skew, mean_skew,
        orthogonality_max_deg
    """
    nj, ni = x.shape

    # Cell areas (using cross product of diagonals)
    areas = []
    aspects = []
    skews = []
    ortho_angles = []

    for j in range(nj - 1):
        for i in range(ni - 1):
            # Cell corners
            x00, y00 = x[j, i], y[j, i]
            x10, y10 = x[j, i + 1], y[j, i + 1]
            x01, y01 = x[j + 1, i], y[j + 1, i]
            x11, y11 = x[j + 1, i + 1], y[j + 1, i + 1]

            # Diagonals
            d1x = x11 - x00
            d1y = y11 - y00
            d2x = x10 - x01
            d2y = y10 - y01
            area = abs(d1x * d2y - d1y * d2x) / 2.0
            areas.append(area)

            # Edge lengths
            e1 = np.sqrt((x10 - x00)**2 + (y10 - y00)**2)
            e2 = np.sqrt((x01 - x00)**2 + (y01 - y00)**2)
            e3 = np.sqrt((x11 - x10)**2 + (y11 - y10)**2)
            e4 = np.sqrt((x11 - x01)**2 + (y11 - y01)**2)

            max_e = max(e1, e2, e3, e4)
            min_e = min(e1, e2, e3, e4)
            if min_e > 1e-30:
                aspects.append(max_e / min_e)

            # Orthogonality at (j, i) corner
            if e1 > 1e-30 and e2 > 1e-30:
                dot = (x10 - x00) * (x01 - x00) + (y10 - y00) * (y01 - y00)
                cos_a = np.clip(dot / (e1 * e2), -1, 1)
                angle = np.degrees(np.arccos(cos_a))
                ortho_angles.append(abs(angle - 90.0))

    areas = np.array(areas)
    aspects = np.array(aspects)

    return {
        "min_area": areas.min(),
        "max_area": areas.max(),
        "area_ratio": areas.max() / areas.min() if areas.min() > 0 else float('inf'),
        "min_aspect": aspects.min() if len(aspects) > 0 else 0,
        "max_aspect": aspects.max() if len(aspects) > 0 else 0,
        "mean_aspect": aspects.mean() if len(aspects) > 0 else 0,
        "orthogonality_max_deg": max(ortho_angles) if ortho_angles else 0,
        "orthogonality_mean_deg": np.mean(ortho_angles) if ortho_angles else 0,
    }


def print_grid_quality(quality, params):
    """Print grid quality report."""
    print()
    print("  " + "=" * 62)
    print("   Grid Quality Report")
    print("  " + "=" * 62)
    print(f"    Geometry:  R1={params['R1']}, R2={params['R2']}, ε={params['epsilon']}")
    print(f"    Grid:      I={params['Ni']} (circumferential) x J={params['Nj']} (radial)")
    print(f"    Bipolar:   α={params['alpha']:.6f}, β={params['beta']:.6f}")
    print(f"               c={params['c']:.6f}, γ={params['gamma']:.6f}")
    print("  " + "-" * 62)
    print(f"    Cell area range:  [{quality['min_area']:.4e}, {quality['max_area']:.4e}]")
    print(f"    Area ratio:       {quality['area_ratio']:.2f}")
    print(f"    Aspect ratio:     [{quality['min_aspect']:.2f}, {quality['max_aspect']:.2f}]"
          f"  mean={quality['mean_aspect']:.2f}")
    print(f"    Orthogonality:    max deviation = {quality['orthogonality_max_deg']:.2f}°"
          f"  mean = {quality['orthogonality_mean_deg']:.2f}°")
    print("  " + "=" * 62)
    print()


# ============================================================
#  7.  Interactive Helpers
# ============================================================

def ask_float(prompt, default, lo=None, hi=None):
    while True:
        raw = input(f"  {prompt} [default={default}]: ").strip()
        if raw == "":
            return default
        try:
            val = float(raw)
        except ValueError:
            print("    ** Invalid number, try again.")
            continue
        if lo is not None and val < lo:
            print(f"    ** Must be >= {lo}, try again.")
            continue
        if hi is not None and val > hi:
            print(f"    ** Must be <= {hi}, try again.")
            continue
        return val


def ask_int(prompt, default, lo=None, hi=None):
    while True:
        raw = input(f"  {prompt} [default={default}]: ").strip()
        if raw == "":
            return default
        try:
            val = int(raw)
        except ValueError:
            print("    ** Invalid integer, try again.")
            continue
        if lo is not None and val < lo:
            print(f"    ** Must be >= {lo}, try again.")
            continue
        if hi is not None and val > hi:
            print(f"    ** Must be <= {hi}, try again.")
            continue
        return val


def ask_yes_no(prompt, default_yes=True):
    hint = "Y/n" if default_yes else "y/N"
    raw = input(f"  {prompt} [{hint}]: ").strip().lower()
    if raw == "":
        return default_yes
    return raw in ("y", "yes")


# ============================================================
#  8.  Config File Support
# ============================================================

def parse_config(filepath):
    """
    Parse a simple key=value config file.
    Lines starting with # are comments. Blank lines ignored.

    Expected keys: R1, R2, EPSILON, NI, NJ, STRETCH_XI, STRETCH_ETA, ALPHA_S
    """
    config = {}
    filepath = Path(filepath)
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                key, val = line.split('=', 1)
                key = key.strip().upper()
                val = val.strip()
                try:
                    if '.' in val or 'e' in val.lower():
                        config[key] = float(val)
                    else:
                        config[key] = int(val)
                except ValueError:
                    config[key] = val
    return config


def write_default_config(filepath):
    """Write a default config file."""
    filepath = Path(filepath)
    content = """# Eccentric Annulus Grid Configuration
# =====================================

# Geometry
R1 = 5.0
R2 = 7.6
EPSILON = 0.5

# Grid resolution
# NI = circumferential (ξ direction) grid points
# NJ = radial (η direction) grid points
NI = 128
NJ = 64

# Stretching (Vinokur tanh, 0 = uniform)
STRETCH_XI = 0.0
STRETCH_ETA = 0.0
ALPHA_S = 0.5
"""
    with open(filepath, "w") as f:
        f.write(content)
    print(f"  [written] Default config: {filepath}")
    return filepath


# ============================================================
#  9.  Main Workflow
# ============================================================

def generate_and_export(R1, R2, epsilon, Ni, Nj,
                        stretch_xi=0.0, stretch_eta=0.0, alpha_s=0.5,
                        output_dir=None, prefix="eccentric_annulus"):
    """
    Full pipeline: generate grid, export files, plot, print quality.

    Parameters
    ----------
    R1, R2 : float
        Inner and outer radii.
    epsilon : float
        Relative eccentricity.
    Ni, Nj : int
        Grid point counts (circumferential, radial).
    stretch_xi, stretch_eta : float
        Vinokur stretching parameters (0=uniform).
    alpha_s : float
        Stretching symmetry parameter.
    output_dir : str or Path or None
        Output directory (default: current directory).
    prefix : str
        Output filename prefix.

    Returns
    -------
    x, y : ndarray
        Grid coordinates.
    params : dict
        Grid parameters.
    """
    if output_dir is None:
        output_dir = Path.cwd()
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    tag = f"I{Ni}_J{Nj}_e{epsilon}"

    print()
    print("  " + "=" * 62)
    print("   Eccentric Annulus Grid Generation")
    print("  " + "=" * 62)
    print(f"    R1 = {R1},  R2 = {R2},  ε = {epsilon}")
    print(f"    Grid: I={Ni} (circumferential) x J={Nj} (radial)")
    if stretch_xi > 0 or stretch_eta > 0:
        print(f"    Stretching: ξ={stretch_xi}, η={stretch_eta}, α_s={alpha_s}")
    print()

    # --- Step 1: Compute constants ---
    print("  [1/5] Computing bipolar constants ...")
    alpha, beta, gamma, c = compute_bipolar_constants(R1, R2, epsilon)
    print(f"         α = {alpha:.8f}")
    print(f"         β = {beta:.8f}")
    print(f"         c = {c:.8f}")
    print(f"         γ = {gamma:.8f}")

    # --- Step 2: Generate grid ---
    print("  [2/5] Generating grid ...")
    if stretch_xi > 1e-14 or stretch_eta > 1e-14:
        x, y, xi, eta, params = generate_eccentric_annulus_grid_stretched(
            R1, R2, epsilon, Ni, Nj, stretch_xi, stretch_eta, alpha_s)
    else:
        x, y, xi, eta, params = generate_eccentric_annulus_grid(
            R1, R2, epsilon, Ni, Nj)
    print(f"         Grid shape: ({x.shape[0]}, {x.shape[1]})")

    # --- Step 3: Quality check ---
    print("  [3/5] Computing grid quality ...")
    quality = compute_grid_quality(x, y)
    print_grid_quality(quality, params)

    # --- Step 4: Export ---
    print("  [4/5] Exporting files ...")

    # Tecplot .dat (primary output, compatible with grid_zeta_tool)
    dat_path = output_dir / f"{prefix}_{tag}.dat"
    write_tecplot_dat(dat_path, x, y,
                      title=f"Eccentric annulus R1={R1} R2={R2} eps={epsilon}",
                      zone_title=f"I{Ni}_J{Nj}")

    # CSV (supplementary)
    csv_path = output_dir / f"{prefix}_{tag}.csv"
    write_csv(csv_path, x, y)

    # --- Step 5: Plot ---
    print("  [5/5] Generating plots ...")

    plot_grid(x, y,
              title=f"Eccentric Annulus: R1={R1}, R2={R2}, ε={epsilon}\n"
                    f"Grid: I={Ni} x J={Nj}",
              savepath=output_dir / f"{prefix}_{tag}_grid.png")

    plot_computational_grid(xi, eta,
                            title=f"Computational Domain (ξ, η)\n"
                                  f"ξ ∈ [−π, π], η ∈ [{alpha:.4f}, {beta:.4f}]",
                            savepath=output_dir / f"{prefix}_{tag}_comp.png")

    plot_radial_spacing(x, y, params, icol=0,
                        savepath=output_dir / f"{prefix}_{tag}_spacing.png")

    print()
    print("  " + "=" * 62)
    print(f"   Output: {dat_path}")
    print("  " + "=" * 62)
    print()

    return x, y, params


# ============================================================
#  MAIN
# ============================================================

if __name__ == "__main__":

    script_dir = Path(__file__).resolve().parent

    # --quick: demo with default parameters
    if "--quick" in sys.argv:
        print()
        print("=" * 62)
        print("  Eccentric Annulus Grid Tool -- Quick Demo")
        print("=" * 62)
        generate_and_export(
            R1=5.0, R2=7.6, epsilon=0.5,
            Ni=128, Nj=64,
            output_dir=script_dir)
        sys.exit(0)

    # --auto: read from config file
    if "--auto" in sys.argv:
        config_candidates = [
            script_dir / "eccentric_annulus.cfg",
            Path.cwd() / "eccentric_annulus.cfg",
        ]
        config_path = None
        for c in config_candidates:
            if c.exists():
                config_path = c
                break
        if config_path is None:
            print("  Config file not found. Creating default ...")
            config_path = write_default_config(script_dir / "eccentric_annulus.cfg")

        print()
        print("=" * 62)
        print("  Eccentric Annulus Grid Tool -- AUTO MODE")
        print(f"  Config: {config_path}")
        print("=" * 62)

        cfg = parse_config(config_path)
        generate_and_export(
            R1=cfg.get("R1", 5.0),
            R2=cfg.get("R2", 7.6),
            epsilon=cfg.get("EPSILON", 0.5),
            Ni=cfg.get("NI", 128),
            Nj=cfg.get("NJ", 64),
            stretch_xi=cfg.get("STRETCH_XI", 0.0),
            stretch_eta=cfg.get("STRETCH_ETA", 0.0),
            alpha_s=cfg.get("ALPHA_S", 0.5),
            output_dir=script_dir)
        sys.exit(0)

    # Interactive mode
    print()
    print("=" * 62)
    print("  Eccentric Annulus Grid Tool")
    print("  (Conformal mapping with bipolar coordinates)")
    print("=" * 62)

    # --- Step 1: Geometry ---
    print()
    print("-" * 62)
    print("  [Step 1] Define eccentric annulus geometry")
    print("-" * 62)
    print()
    print("  R1 -- Inner radius")
    R1 = ask_float("R1", default=5.0, lo=0.01)
    print()
    print("  R2 -- Outer radius (must be > R1)")
    R2 = ask_float("R2", default=7.6, lo=R1 + 0.01)
    print()
    print("  ε -- Relative eccentricity (0 < ε < 1)")
    print(f"       Physical offset = ε × (R2−R1) = ε × {R2 - R1:.4f}")
    epsilon = ask_float("ε", default=0.5, lo=0.01, hi=0.99)

    # --- Step 2: Grid resolution ---
    print()
    print("-" * 62)
    print("  [Step 2] Set grid resolution")
    print("-" * 62)
    print()
    print("  Ni -- Circumferential grid points (ξ direction)")
    Ni = ask_int("Ni", default=128, lo=10, hi=10000)
    print()
    print("  Nj -- Radial grid points (η direction)")
    Nj = ask_int("Nj", default=64, lo=5, hi=10000)

    # --- Step 3: Stretching ---
    print()
    print("-" * 62)
    print("  [Step 3] Grid stretching (optional)")
    print("-" * 62)
    do_stretch = ask_yes_no("Apply Vinokur stretching?", default_yes=False)

    stretch_xi = 0.0
    stretch_eta = 0.0
    alpha_s = 0.5

    if do_stretch:
        print()
        print("  Stretch ξ -- circumferential stretching")
        print("               0 = uniform, >0 = wall clustering")
        stretch_xi = ask_float("Stretch ξ", default=0.0, lo=0.0, hi=10.0)
        print()
        print("  Stretch η -- radial stretching")
        print("               0 = uniform, >0 = wall clustering")
        stretch_eta = ask_float("Stretch η", default=0.0, lo=0.0, hi=10.0)
        print()
        print("  α_s -- Stretching symmetry (0.5 = symmetric)")
        alpha_s = ask_float("α_s", default=0.5, lo=0.01, hi=0.99)

    # --- Step 4: Summary and generate ---
    print()
    print("-" * 62)
    print("  [Step 4] Configuration summary")
    print("-" * 62)
    print(f"    R1 = {R1},  R2 = {R2},  ε = {epsilon}")
    print(f"    Gap = {R2 - R1:.4f},  Offset = {epsilon * (R2 - R1):.4f}")
    print(f"    Grid: Ni={Ni} x Nj={Nj} = {Ni * Nj} points")
    if do_stretch:
        print(f"    Stretching: ξ={stretch_xi}, η={stretch_eta}, α_s={alpha_s}")
    print()

    proceed = ask_yes_no("Generate grid?", default_yes=True)
    if not proceed:
        print("  Aborted.")
        sys.exit(0)

    generate_and_export(
        R1=R1, R2=R2, epsilon=epsilon,
        Ni=Ni, Nj=Nj,
        stretch_xi=stretch_xi, stretch_eta=stretch_eta, alpha_s=alpha_s,
        output_dir=script_dir)

    # --- Optional: parametric sweep ---
    print()
    do_sweep = ask_yes_no("Generate eccentricity sweep?", default_yes=False)
    if do_sweep and _HAS_MPL:
        epsilons = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
        fig, axes = plt.subplots(len(epsilons), 1,
                                 figsize=(9, 4 * len(epsilons)))
        for ax, eps in zip(axes, epsilons):
            try:
                xn, yn, _, _, _ = generate_eccentric_annulus_grid(
                    R1, R2, eps, Ni, Nj)
                nj, ni = xn.shape
                for j in range(nj):
                    ax.plot(xn[j, :], yn[j, :], 'k-', lw=0.2)
                for i in range(0, ni, max(1, ni // 40)):
                    ax.plot(xn[:, i], yn[:, i], 'k-', lw=0.2)
                ax.set_aspect('equal')
                ax.set_ylabel('y')
                ax.set_title(f'ε = {eps}', fontsize=10, loc='left')
            except Exception as e:
                ax.set_title(f'ε = {eps}: {e}', fontsize=10, loc='left')
        axes[-1].set_xlabel('x')
        fig.suptitle(f'Eccentricity Sweep: R1={R1}, R2={R2}', fontsize=14)
        plt.tight_layout()
        sweep_path = script_dir / f"sweep_R1{R1}_R2{R2}_I{Ni}_J{Nj}.png"
        fig.savefig(sweep_path, dpi=200, bbox_inches='tight')
        print(f"  [saved] {sweep_path}")
        plt.close(fig)

    print()
    print("=" * 62)
    print("  ALL DONE")
    print("=" * 62)
