"""
LBM Grid Adapter
==================
Convert curvilinear bipolar grid to LBM lattice data structures.

For general interpolation LBM on curvilinear grids:
  1. Each grid node becomes a lattice site
  2. The streaming step uses interpolation because neighbours in
     computational space don't align with D2Q9 lattice velocities
  3. Scale factor h determines local physical spacing
"""

import numpy as np


# D2Q9 lattice velocity set
D2Q9_EX = np.array([0, 1, 0, -1,  0, 1, -1, -1,  1])
D2Q9_EY = np.array([0, 0, 1,  0, -1, 1,  1, -1, -1])
D2Q9_W = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])


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
    dict with lattice info arrays.
    """
    nj, ni = x.shape

    # Boundary classification
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
    Classify grid nodes.

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
        1 = inner wall (eta = beta, j = nj-1)
        2 = outer wall (eta = alpha, j = 0)
        3 = periodic boundary (xi = -pi or xi = pi, i = 0 or i = ni-1)
    """
    nj, ni = x.shape

    # In bipolar grid convention:
    #   j=0    -> eta = alpha (outer wall, R2)
    #   j=nj-1 -> eta = beta  (inner wall, R1)
    #   i=0 and i=ni-1 -> xi = -pi and xi = +pi (periodic seam)

    boundary_type = np.zeros((nj, ni), dtype=np.int32)

    # Outer wall: j = 0
    boundary_type[0, :] = 2

    # Inner wall: j = nj-1
    boundary_type[-1, :] = 1

    # Periodic boundaries: i = 0 and i = ni-1
    # (only mark interior points, corners get wall type)
    boundary_type[1:-1, 0] = 3
    boundary_type[1:-1, -1] = 3

    return boundary_type


def compute_interpolation_weights(xi_grid, eta_grid, c, gamma_shift):
    """
    For each lattice node and each D2Q9 velocity direction,
    compute the departure point in computational space for the
    streaming step on the curvilinear grid.

    In GILBM, the streaming step advects along lattice velocities
    in computational space. The contravariant velocity at each node
    determines where the departure point falls.

    Parameters
    ----------
    xi_grid, eta_grid : ndarray (nj, ni)
        Computational coordinates.
    c : float
        Bipolar pole distance.
    gamma_shift : float
        Vertical shift.

    Returns
    -------
    dict with:
        departure_xi  : ndarray (nj, ni, 9) -- departure xi for each direction
        departure_eta : ndarray (nj, ni, 9) -- departure eta for each direction
    """
    nj, ni = xi_grid.shape

    # Scale factor
    h = c / (np.cosh(eta_grid) - np.cos(xi_grid))

    # In computational space, the lattice velocity for direction a is:
    #   c_tilde_xi  = (1/h) * (xi_x * ex + xi_y * ey)  * delta_xi
    #   c_tilde_eta = (1/h) * (eta_x * ex + eta_y * ey) * delta_eta
    # For conformal mapping, the contravariant components simplify.

    # Computational spacing
    dxi = 2 * np.pi / (ni - 1) if ni > 1 else 1.0
    deta = (eta_grid[-1, 0] - eta_grid[0, 0]) / (nj - 1) if nj > 1 else 1.0

    departure_xi = np.zeros((nj, ni, 9))
    departure_eta = np.zeros((nj, ni, 9))

    for a in range(9):
        # Departure point (first-order, single timestep)
        # In uniform computational space, the departure offset is just (ex, ey)
        # scaled by the local metric
        departure_xi[:, :, a] = xi_grid - D2Q9_EX[a] * dxi
        departure_eta[:, :, a] = eta_grid - D2Q9_EY[a] * deta

    # Wrap xi to [-pi, pi] for periodicity
    departure_xi = ((departure_xi + np.pi) % (2 * np.pi)) - np.pi

    # Clamp eta to [eta_min, eta_max]
    eta_min = eta_grid[0, 0]
    eta_max = eta_grid[-1, 0]
    departure_eta = np.clip(departure_eta, eta_min, eta_max)

    return {
        "departure_xi": departure_xi,
        "departure_eta": departure_eta,
    }
