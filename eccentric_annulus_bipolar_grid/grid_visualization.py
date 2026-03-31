"""
Grid Visualization Utilities
==============================
Matplotlib-based plotting for bipolar grid, metrics, and quality maps.
"""

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


def _check_mpl(savepath=None):
    if not _HAS_MPL:
        if savepath:
            print(f"  [skip plot] matplotlib not available: {savepath}")
        return False
    return True


def plot_physical_grid(x, y, params=None, savepath=None, figsize=(9, 9),
                       show_boundary=True, every_xi=1, every_eta=1):
    """
    Plot the grid in physical (x, y) space with colored boundaries.

    Parameters
    ----------
    x, y : ndarray (nj, ni)
        Grid coordinates.
    params : dict or None
        Grid parameters for title.
    savepath : str or None
        Save path for PNG.
    figsize : tuple
        Figure size.
    show_boundary : bool
        Highlight inner/outer boundaries.
    every_xi, every_eta : int
        Plot every N-th line (for dense grids).
    """
    if not _check_mpl(savepath):
        return

    nj, ni = x.shape
    fig, ax = plt.subplots(figsize=figsize)

    # Constant-eta lines (circumferential)
    for j in range(0, nj, every_eta):
        if j == 0:
            color, lw = "blue", 1.2
        elif j == nj - 1:
            color, lw = "red", 1.2
        else:
            color, lw = "grey", 0.3
        ax.plot(x[j, :], y[j, :], color=color, lw=lw)

    # Constant-xi lines (radial)
    for i in range(0, ni, every_xi):
        ax.plot(x[:, i], y[:, i], "k-", lw=0.3)

    if show_boundary:
        ax.plot(np.append(x[0, :], x[0, 0]),
                np.append(y[0, :], y[0, 0]),
                "b-", lw=1.5, label="Outer (R2)")
        ax.plot(np.append(x[-1, :], x[-1, 0]),
                np.append(y[-1, :], y[-1, 0]),
                "r-", lw=1.5, label="Inner (R1)")
        ax.legend(fontsize=9)

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    if params:
        title = (f"Eccentric Annulus: R1={params.get('R1', '?')}, "
                 f"R2={params.get('R2', '?')}, "
                 f"eps={params.get('epsilon', '?')}\n"
                 f"Grid: I={params.get('N_xi', '?')} x "
                 f"J={params.get('N_eta', '?')}")
        ax.set_title(title)
    else:
        ax.set_title("Eccentric Annulus Grid")

    ax.grid(True, ls="--", alpha=0.3)
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=200)
        print(f"  [saved] {savepath}")
    plt.close(fig)


def plot_computational_grid(xi, eta, savepath=None, figsize=(12, 6)):
    """Plot the computational (xi, eta) grid."""
    if not _check_mpl(savepath):
        return

    nj, ni = xi.shape
    fig, ax = plt.subplots(figsize=figsize)
    for j in range(nj):
        ax.plot(xi[j, :], eta[j, :], "k-", lw=0.3)
    for i in range(ni):
        ax.plot(xi[:, i], eta[:, i], "k-", lw=0.3)
    ax.set_xlabel("xi")
    ax.set_ylabel("eta")
    ax.set_title("Computational Domain (xi, eta)")
    ax.grid(True, ls="--", alpha=0.3)
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=200)
        print(f"  [saved] {savepath}")
    plt.close(fig)


def plot_scale_factor_field(x, y, h, savepath=None, figsize=(10, 9)):
    """Contour plot of the scale factor h(xi, eta) on physical domain."""
    if not _check_mpl(savepath):
        return

    fig, ax = plt.subplots(figsize=figsize)
    cf = ax.contourf(x, y, h, levels=50, cmap="viridis")
    plt.colorbar(cf, ax=ax, label="h = c / (cos(xi) + cosh(eta))")
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Scale Factor h(xi, eta)")
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=200)
        print(f"  [saved] {savepath}")
    plt.close(fig)


def plot_jacobian_field(x, y, detJ, savepath=None, figsize=(10, 9)):
    """Contour plot of Jacobian determinant."""
    if not _check_mpl(savepath):
        return

    fig, ax = plt.subplots(figsize=figsize)
    cf = ax.contourf(x, y, detJ, levels=50, cmap="plasma")
    plt.colorbar(cf, ax=ax, label="|J| = det(Jacobian)")
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Jacobian Determinant")
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=200)
        print(f"  [saved] {savepath}")
    plt.close(fig)


def plot_grid_quality_map(x, y, quality_field, title="Quality",
                          savepath=None, figsize=(10, 9), cmap="coolwarm"):
    """Plot aspect ratio, skewness, or orthogonality deviation map."""
    if not _check_mpl(savepath):
        return

    fig, ax = plt.subplots(figsize=figsize)
    nj, ni = quality_field.shape
    # Quality fields are cell-centered (nj-1, ni-1); node grid is (nj+1, ni+1).
    # Use the full node grid as corner coordinates for pcolormesh.
    cf = ax.pcolormesh(x[:nj + 1, :ni + 1], y[:nj + 1, :ni + 1],
                       quality_field, cmap=cmap, shading="flat")
    plt.colorbar(cf, ax=ax, label=title)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=200)
        print(f"  [saved] {savepath}")
    plt.close(fig)


def plot_radial_spacing_profile(x, y, icols=None, savepath=None, figsize=(10, 6)):
    """
    Plot dr vs j-index at several xi locations.

    Parameters
    ----------
    x, y : ndarray (nj, ni)
        Grid coordinates.
    icols : list of int or None
        Column indices to plot. Default: [0, ni//4, ni//2, 3*ni//4].
    savepath : str or None
        Save path.
    """
    if not _check_mpl(savepath):
        return

    nj, ni = x.shape
    if icols is None:
        icols = [0, ni // 4, ni // 2, 3 * ni // 4]

    fig, ax = plt.subplots(figsize=figsize)
    for ic in icols:
        if ic >= ni:
            continue
        dr = np.sqrt(np.diff(x[:, ic])**2 + np.diff(y[:, ic])**2)
        ax.plot(range(nj - 1), dr, "o-", ms=2, label=f"i={ic}")
    ax.set_xlabel("j index (radial)")
    ax.set_ylabel("dr (physical spacing)")
    ax.set_title("Radial Spacing Profile")
    ax.legend(fontsize=8)
    ax.grid(True, ls="--", alpha=0.4)
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=200)
        print(f"  [saved] {savepath}")
    plt.close(fig)


def plot_eccentricity_sweep(r1, r2, eccentricities, N_xi, N_eta,
                            savepath=None, figsize=(10, None)):
    """
    Generate multi-panel plot showing grids at different eccentricities.

    Parameters
    ----------
    r1, r2 : float
        Inner and outer radii.
    eccentricities : list of float
        Eccentricity values (relative, 0 < eps < 1).
    N_xi, N_eta : int
        Grid resolution.
    savepath : str or None
        Save path.
    """
    if not _check_mpl(savepath):
        return

    from .bipolar_constants import compute_bipolar_constants_snyder
    from .bipolar_transform import bipolar_to_cartesian

    n_eps = len(eccentricities)
    if figsize[1] is None:
        figsize = (figsize[0], 4 * n_eps)

    fig, axes = plt.subplots(n_eps, 1, figsize=figsize)
    if n_eps == 1:
        axes = [axes]

    for ax, eps in zip(axes, eccentricities):
        try:
            consts = compute_bipolar_constants_snyder(r1, r2, eps)
            alpha = consts["alpha"]
            beta = consts["beta"]
            c = consts["c"]
            gamma_shift = consts["gamma_shift"]

            xi = np.linspace(-np.pi, np.pi, N_xi)
            eta = np.linspace(alpha, beta, N_eta)
            Xi, Eta = np.meshgrid(xi, eta)
            xg, yg = bipolar_to_cartesian(Xi, Eta, c, gamma_shift)

            nj, ni = xg.shape
            for j in range(nj):
                ax.plot(xg[j, :], yg[j, :], "k-", lw=0.2)
            for i in range(0, ni, max(1, ni // 40)):
                ax.plot(xg[:, i], yg[:, i], "k-", lw=0.2)
            ax.set_aspect("equal")
            ax.set_ylabel("y")
            ax.set_title(f"eps = {eps}", fontsize=10, loc="left")
        except Exception as e:
            ax.set_title(f"eps = {eps}: {e}", fontsize=10, loc="left")

    axes[-1].set_xlabel("x")
    fig.suptitle(f"Eccentricity Sweep: R1={r1}, R2={r2}", fontsize=14)
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches="tight")
        print(f"  [saved] {savepath}")
    plt.close(fig)


def plot_gamma_sweep(x_grids, y_grids, gammas, params,
                     savepath=None, figsize=(10, None)):
    """
    Multi-panel plot showing grids at different GAMMA stretching values.

    Parameters
    ----------
    x_grids, y_grids : list of ndarray
        Grid coordinates for each GAMMA.
    gammas : list of float
        GAMMA values.
    params : dict
        Grid parameters for title.
    savepath : str or None
        Save path.
    """
    if not _check_mpl(savepath):
        return

    n = len(gammas)
    if figsize[1] is None:
        figsize = (figsize[0], 4 * n)

    fig, axes = plt.subplots(n, 1, figsize=figsize)
    if n == 1:
        axes = [axes]

    for ax, xg, yg, g in zip(axes, x_grids, y_grids, gammas):
        nj, ni = xg.shape
        for j in range(nj):
            ax.plot(xg[j, :], yg[j, :], "k-", lw=0.2)
        for i in range(0, ni, max(1, ni // 40)):
            ax.plot(xg[:, i], yg[:, i], "k-", lw=0.2)
        ax.set_aspect("equal")
        ax.set_ylabel("y")
        ax.set_title(f"GAMMA = {g:.1f}", fontsize=10, loc="left")

    axes[-1].set_xlabel("x")
    fig.suptitle(
        f"GAMMA Sweep: R1={params.get('R1', '?')}, "
        f"R2={params.get('R2', '?')}, "
        f"eps={params.get('epsilon', '?')}",
        fontsize=14)
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches="tight")
        print(f"  [saved] {savepath}")
    plt.close(fig)
