"""
LBM Grid Adapter
==================
Convert curvilinear bipolar grid to LBM lattice data structures.

For general interpolation LBM on curvilinear grids:
  1. Each grid node becomes a lattice site
  2. The streaming step uses interpolation because neighbours in
     computational space don't align with D2Q9 lattice velocities
  3. Scale factor h determines local physical spacing
  4. Contravariant velocities c_tilde = J^{-T} * e_alpha transform
     lattice velocities to computational space
"""

import numpy as np


# D2Q9 lattice velocity set
D2Q9_EX = np.array([0, 1, 0, -1,  0, 1, -1, -1,  1])
D2Q9_EY = np.array([0, 0, 1,  0, -1, 1,  1, -1, -1])
D2Q9_W = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
D2Q9_Q = 9


def create_lbm_lattice_info(x, y, h, detJ, R1, R2, eccentricity,
                             xi_grid=None, eta_grid=None):
    """
    Convert curvilinear grid to LBM lattice data structure.

    Parameters
    ----------
    x, y : ndarray (N_eta, N_xi)
        Physical coordinates.
    h : ndarray (N_eta, N_xi)
        Scale factor.
    detJ : ndarray (N_eta, N_xi)
        Jacobian determinant.
    R1 : float
        Inner radius.
    R2 : float
        Outer radius.
    eccentricity : float
        Physical eccentricity distance.
    xi_grid, eta_grid : ndarray or None
        Computational coordinates.

    Returns
    -------
    dict with lattice info arrays:
        node_x, node_y, h, detJ, boundary_type,
        node_xi, node_eta (if provided)
    """
    boundary_type = identify_boundary_nodes(x, y, R1, R2, eccentricity)

    result = {
        "node_x": x,
        "node_y": y,
        "h": h,
        "detJ": detJ,
        "boundary_type": boundary_type,
    }

    if xi_grid is not None:
        result["node_xi"] = xi_grid
        result["node_eta"] = eta_grid

    return result


def identify_boundary_nodes(x, y, R1, R2, eccentricity, tol=1e-6):
    """
    Classify grid nodes by boundary type.

    In the bipolar grid convention:
      j = 0      -> eta = alpha (outer wall, R2)
      j = nj - 1 -> eta = beta  (inner wall, R1)
      i = 0 and i = ni - 1 -> xi = -pi and xi = +pi (periodic seam)

    Parameters
    ----------
    x, y : ndarray (nj, ni)
        Grid coordinates.
    R1 : float
        Inner radius.
    R2 : float
        Outer radius.
    eccentricity : float
        Physical eccentricity distance.
    tol : float
        Tolerance for boundary identification.

    Returns
    -------
    boundary_type : ndarray (nj, ni), int
        0 = interior
        1 = inner wall  (j = nj-1, eta = beta)
        2 = outer wall  (j = 0,    eta = alpha)
        3 = periodic    (i = 0 or i = ni-1)
    """
    nj, ni = x.shape
    boundary_type = np.zeros((nj, ni), dtype=np.int32)

    # Outer wall: j = 0
    boundary_type[0, :] = 2
    # Inner wall: j = nj-1
    boundary_type[-1, :] = 1
    # Periodic boundaries (interior j-rows only; corners keep wall type)
    boundary_type[1:-1, 0] = 3
    boundary_type[1:-1, -1] = 3

    return boundary_type


def compute_contravariant_velocities(xi_grid, eta_grid, c):
    """
    Compute contravariant lattice velocities c_tilde for each D2Q9 direction.

    For GILBM, the streaming step advects along transformed lattice velocities:
        c_tilde^1_a = xi_x  * e_x_a + xi_y  * e_y_a
        c_tilde^2_a = eta_x * e_x_a + eta_y * e_y_a

    where (xi_x, xi_y, eta_x, eta_y) = J^{-T} are the inverse metric components.

    For the conformal tan-based bipolar mapping, the inverse metric simplifies:
        xi_x  =  (1/h^2) * dx/dxi,    xi_y  = (1/h^2) * dy/dxi
        eta_x =  (1/h^2) * dx/deta,   eta_y = (1/h^2) * dy/deta

    Parameters
    ----------
    xi_grid, eta_grid : ndarray (nj, ni)
        Computational coordinates.
    c : float
        Bipolar pole distance.

    Returns
    -------
    dict with:
        c_tilde_xi  : ndarray (nj, ni, 9) -- contravariant xi  for each D2Q9 dir
        c_tilde_eta : ndarray (nj, ni, 9) -- contravariant eta for each D2Q9 dir
        max_c       : float -- max |c_tilde| over all nodes and directions
    """
    from .grid_metrics import compute_jacobian_analytic, compute_inverse_metric

    jac = compute_jacobian_analytic(xi_grid, eta_grid, c)
    inv = compute_inverse_metric(
        jac["J11"], jac["J12"], jac["J21"], jac["J22"], jac["detJ"])

    nj, ni = xi_grid.shape
    c_tilde_xi = np.zeros((nj, ni, D2Q9_Q))
    c_tilde_eta = np.zeros((nj, ni, D2Q9_Q))

    for a in range(D2Q9_Q):
        c_tilde_xi[:, :, a] = inv["xi_x"] * D2Q9_EX[a] + inv["xi_y"] * D2Q9_EY[a]
        c_tilde_eta[:, :, a] = inv["eta_x"] * D2Q9_EX[a] + inv["eta_y"] * D2Q9_EY[a]

    # Max contravariant velocity (interior only, skip boundary FD artifacts)
    sl = (slice(1, -1), slice(1, -1), slice(None))
    max_c = max(np.abs(c_tilde_xi[sl]).max(), np.abs(c_tilde_eta[sl]).max())

    return {
        "c_tilde_xi": c_tilde_xi,
        "c_tilde_eta": c_tilde_eta,
        "max_c": max_c,
    }


def compute_interpolation_stencil(xi_grid, eta_grid, c):
    """
    For each lattice node and each D2Q9 direction, compute the departure
    point and bilinear interpolation weights for the GILBM streaming step.

    The departure point for direction a at node (j, i) is:
        xi_dep  = xi[j, i]  - c_tilde_xi[j, i, a]
        eta_dep = eta[j, i] - c_tilde_eta[j, i, a]

    Bilinear interpolation then reconstructs f from the four surrounding
    nodes in computational space.

    Parameters
    ----------
    xi_grid, eta_grid : ndarray (nj, ni)
        Computational coordinates.
    c : float
        Bipolar pole distance.

    Returns
    -------
    dict with:
        departure_xi     : ndarray (nj, ni, 9)
        departure_eta    : ndarray (nj, ni, 9)
        interp_i0        : ndarray (nj, ni, 9), int -- lower i-index of stencil
        interp_j0        : ndarray (nj, ni, 9), int -- lower j-index of stencil
        interp_wx        : ndarray (nj, ni, 9)      -- weight in xi direction
        interp_wy        : ndarray (nj, ni, 9)      -- weight in eta direction
    """
    nj, ni = xi_grid.shape

    cvel = compute_contravariant_velocities(xi_grid, eta_grid, c)
    c_tilde_xi = cvel["c_tilde_xi"]
    c_tilde_eta = cvel["c_tilde_eta"]

    # Computational spacing
    dxi = (xi_grid[0, -1] - xi_grid[0, 0]) / (ni - 1) if ni > 1 else 1.0
    deta = (eta_grid[-1, 0] - eta_grid[0, 0]) / (nj - 1) if nj > 1 else 1.0

    departure_xi = np.zeros((nj, ni, D2Q9_Q))
    departure_eta = np.zeros((nj, ni, D2Q9_Q))

    for a in range(D2Q9_Q):
        departure_xi[:, :, a] = xi_grid - c_tilde_xi[:, :, a] * dxi
        departure_eta[:, :, a] = eta_grid - c_tilde_eta[:, :, a] * deta

    # Wrap xi to [-pi, pi] for periodicity (handle floating-point edge)
    xi_min = xi_grid[0, 0]
    xi_range = xi_grid[0, -1] - xi_grid[0, 0]
    if abs(xi_range - 2 * np.pi) < 1e-10:
        departure_xi = xi_min + (departure_xi - xi_min) % xi_range

    # Clamp eta to valid domain
    eta_min = eta_grid[0, 0]
    eta_max = eta_grid[-1, 0]
    departure_eta = np.clip(departure_eta, eta_min, eta_max)

    # Compute bilinear interpolation indices and weights
    interp_i0 = np.zeros((nj, ni, D2Q9_Q), dtype=np.int32)
    interp_j0 = np.zeros((nj, ni, D2Q9_Q), dtype=np.int32)
    interp_wx = np.zeros((nj, ni, D2Q9_Q))
    interp_wy = np.zeros((nj, ni, D2Q9_Q))

    dxi_safe = dxi if abs(dxi) > 1e-30 else 1.0
    deta_safe = deta if abs(deta) > 1e-30 else 1.0

    for a in range(D2Q9_Q):
        # Fractional index in xi direction
        fi = (departure_xi[:, :, a] - xi_min) / dxi_safe
        fi = np.clip(fi, 0, ni - 1 - 1e-10)
        i0 = np.floor(fi).astype(np.int32)
        i0 = np.clip(i0, 0, ni - 2)
        interp_i0[:, :, a] = i0
        interp_wx[:, :, a] = fi - i0

        # Fractional index in eta direction
        fj = (departure_eta[:, :, a] - eta_min) / deta_safe
        fj = np.clip(fj, 0, nj - 1 - 1e-10)
        j0 = np.floor(fj).astype(np.int32)
        j0 = np.clip(j0, 0, nj - 2)
        interp_j0[:, :, a] = j0
        interp_wy[:, :, a] = fj - j0

    return {
        "departure_xi": departure_xi,
        "departure_eta": departure_eta,
        "interp_i0": interp_i0,
        "interp_j0": interp_j0,
        "interp_wx": interp_wx,
        "interp_wy": interp_wy,
    }
