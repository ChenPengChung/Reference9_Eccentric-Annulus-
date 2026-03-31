"""
Grid Metrics: Jacobian, Metric Tensor, Scale Factors, Christoffel Symbols
==========================================================================
Critical for GILBM: the curvilinear grid metrics determine how the LBM
collision-streaming operates on the non-uniform mesh.

For the tan-based bipolar mapping z = c * tan(zeta/2):
  - The scale factor h = c / (cos(xi) + cosh(eta))
  - The metric tensor is diagonal: g11 = g22 = h^2, g12 = 0
  - The Jacobian determinant |J| = h^2

NOTE: The PDF document uses a different convention z = ic*cot(zeta/2) with
scale factor h_pdf = c / (cosh(eta) - cos(xi)). The code uses the tan-based
mapping, so the denominator has a PLUS sign: cos(xi) + cosh(eta).
"""

import numpy as np


def compute_scale_factor(xi, eta, c):
    """
    Bipolar coordinate scale factor for z = c * tan(zeta/2).

        h(xi, eta) = c / (cos(xi) + cosh(eta))

    Derivation:
        |dz/dzeta| = (c/2) / |cos(zeta/2)|^2
        |cos((xi+i*eta)/2)|^2 = (cos(xi) + cosh(eta)) / 2
        h = |dz/dzeta| = c / (cos(xi) + cosh(eta))

    Parameters
    ----------
    xi, eta : ndarray
        Bipolar coordinates (can be 1D or 2D meshgrid arrays).
    c : float
        Bipolar pole distance.

    Returns
    -------
    h : ndarray
        Scale factor at each grid point.
    """
    denom = np.cos(xi) + np.cosh(eta)
    # Protect against zero at poles (xi=+-pi, eta=0)
    denom = np.where(np.abs(denom) > 1e-30, denom, 1e-30)
    return c / denom


def compute_jacobian_analytic(xi, eta, c):
    """
    Compute the Jacobian matrix components analytically for bipolar coordinates.

    The conformal mapping z = c * tan(zeta/2) has derivative:
        dz/dzeta = (c/2) / cos^2(zeta/2)

    For the Jacobian matrix J = [[dx/dxi, dx/deta], [dy/dxi, dy/deta]]:
        dz/dxi  = dz/dzeta * 1
        dz/deta = dz/dzeta * i

    For conformal mappings, |J| = h^2 where h = c / (cos(xi) + cosh(eta)).

    Parameters
    ----------
    xi, eta : ndarray (2D meshgrid)
        Bipolar coordinates.
    c : float
        Pole distance.

    Returns
    -------
    dict with keys:
        J11, J12, J21, J22 : ndarray -- Jacobian matrix components
        detJ               : ndarray -- Jacobian determinant (= h^2)
        h                  : ndarray -- scale factor
    """
    h = compute_scale_factor(xi, eta, c)

    zeta = xi + 1j * eta
    half_zeta = zeta / 2.0
    cos2 = np.cos(half_zeta)**2
    # Protect against singularity
    cos2 = np.where(np.abs(cos2) > 1e-60, cos2, 1e-60)
    dz_dzeta = c / (2.0 * cos2)

    dz_dxi = dz_dzeta        # dzeta/dxi = 1
    dz_deta = 1j * dz_dzeta  # dzeta/deta = i

    J11 = np.real(dz_dxi)    # dx/dxi
    J12 = np.real(dz_deta)   # dx/deta
    J21 = np.imag(dz_dxi)    # dy/dxi
    J22 = np.imag(dz_deta)   # dy/deta

    detJ = J11 * J22 - J12 * J21

    return {
        "J11": J11, "J12": J12,
        "J21": J21, "J22": J22,
        "detJ": detJ,
        "h": h,
    }


def compute_jacobian_numerical(x, y):
    """
    Compute the Jacobian matrix numerically from grid coordinates.

    Uses central finite differences (one-sided at boundaries).
    Derivative is with respect to index (not physical coordinate),
    so dx/dxi ~ (x[i+1] - x[i-1]) / 2  for unit computational spacing.

    Parameters
    ----------
    x, y : ndarray (nj, ni)
        Physical grid coordinates.

    Returns
    -------
    dict with keys:
        J11, J12, J21, J22 : ndarray -- Jacobian matrix components
        detJ               : ndarray -- Jacobian determinant
    """
    nj, ni = x.shape

    # dx/dxi, dy/dxi (along i-direction, j fixed)
    J11 = np.zeros_like(x)  # dx/dxi
    J21 = np.zeros_like(y)  # dy/dxi
    # Interior: central difference (x[i+1] - x[i-1]) / 2
    J11[:, 1:-1] = (x[:, 2:] - x[:, :-2]) / 2.0
    J21[:, 1:-1] = (y[:, 2:] - y[:, :-2]) / 2.0
    # Boundaries: one-sided
    J11[:, 0] = x[:, 1] - x[:, 0]
    J11[:, -1] = x[:, -1] - x[:, -2]
    J21[:, 0] = y[:, 1] - y[:, 0]
    J21[:, -1] = y[:, -1] - y[:, -2]

    # dx/deta, dy/deta (along j-direction, i fixed)
    J12 = np.zeros_like(x)  # dx/deta
    J22 = np.zeros_like(y)  # dy/deta
    J12[1:-1, :] = (x[2:, :] - x[:-2, :]) / 2.0
    J22[1:-1, :] = (y[2:, :] - y[:-2, :]) / 2.0
    J12[0, :] = x[1, :] - x[0, :]
    J12[-1, :] = x[-1, :] - x[-2, :]
    J22[0, :] = y[1, :] - y[0, :]
    J22[-1, :] = y[-1, :] - y[-2, :]

    detJ = J11 * J22 - J12 * J21

    return {
        "J11": J11, "J12": J12,
        "J21": J21, "J22": J22,
        "detJ": detJ,
    }


def compute_metric_tensor(J11, J12, J21, J22):
    """
    Compute the covariant metric tensor g_ij = J^T * J.

    For a conformal mapping:
      g11 = g22 = h^2 (scale factor squared)
      g12 = 0          (orthogonal grid)

    Parameters
    ----------
    J11, J12, J21, J22 : ndarray
        Jacobian matrix components.

    Returns
    -------
    dict with keys:
        g11, g12, g22 : ndarray -- metric tensor components
        orthogonality_error : float -- max |g12| / sqrt(g11*g22)
    """
    g11 = J11**2 + J21**2
    g12 = J11 * J12 + J21 * J22
    g22 = J12**2 + J22**2

    denom = np.sqrt(g11 * g22)
    denom = np.where(denom > 1e-30, denom, 1e-30)
    ortho_err = np.max(np.abs(g12) / denom)

    return {
        "g11": g11,
        "g12": g12,
        "g22": g22,
        "orthogonality_error": ortho_err,
    }


def compute_inverse_metric(J11, J12, J21, J22, detJ):
    """
    Compute the inverse metric (contravariant) components.

    xi_x  =  J22 / detJ,   xi_y  = -J12 / detJ
    eta_x = -J21 / detJ,   eta_y =  J11 / detJ

    These are needed for the GILBM contravariant velocity computation.

    Parameters
    ----------
    J11, J12, J21, J22 : ndarray
        Jacobian matrix components.
    detJ : ndarray
        Jacobian determinant.

    Returns
    -------
    dict with keys: xi_x, xi_y, eta_x, eta_y
    """
    eps = 1e-30
    safe_detJ = np.where(np.abs(detJ) > eps, detJ, eps)

    xi_x = J22 / safe_detJ
    xi_y = -J12 / safe_detJ
    eta_x = -J21 / safe_detJ
    eta_y = J11 / safe_detJ

    return {
        "xi_x": xi_x, "xi_y": xi_y,
        "eta_x": eta_x, "eta_y": eta_y,
    }


def compute_christoffel_symbols(xi, eta, c):
    """
    Compute Christoffel symbols for the bipolar coordinate system.

    For the tan-based mapping z = c * tan(zeta/2), the scale factor is:
        h = c / (cos(xi) + cosh(eta))

    Partial derivatives:
        dh/dxi  =  c * sin(xi)   / (cos(xi) + cosh(eta))^2
                 =  h * sin(xi)   / (cos(xi) + cosh(eta))

        dh/deta = -c * sinh(eta)  / (cos(xi) + cosh(eta))^2
                 = -h * sinh(eta)  / (cos(xi) + cosh(eta))

    For conformal coordinates (h1 = h2 = h), the non-zero Christoffel
    symbols of the second kind are:
        Gamma^1_{11} =  (1/h) * dh/dxi    = sin(xi) / D
        Gamma^1_{22} = -(1/h) * dh/dxi    = -sin(xi) / D
        Gamma^1_{12} =  (1/h) * dh/deta   = -sinh(eta) / D
        Gamma^2_{11} = -(1/h) * dh/deta   = sinh(eta) / D
        Gamma^2_{22} =  (1/h) * dh/deta   = -sinh(eta) / D
        Gamma^2_{12} =  (1/h) * dh/dxi    = sin(xi) / D

    where D = cos(xi) + cosh(eta).

    Parameters
    ----------
    xi, eta : ndarray (2D)
        Bipolar coordinates.
    c : float
        Pole distance.

    Returns
    -------
    dict with Christoffel symbol arrays and partial derivatives of h.
    """
    D = np.cos(xi) + np.cosh(eta)
    D_safe = np.where(np.abs(D) > 1e-30, D, 1e-30)
    h = c / D_safe

    dh_dxi = h * np.sin(xi) / D_safe
    dh_deta = -h * np.sinh(eta) / D_safe

    inv_h = 1.0 / np.where(np.abs(h) > 1e-30, h, 1e-30)

    # Conformal Christoffel symbols (h1 = h2 = h)
    # Using standard formulas for orthogonal conformal coordinates:
    #   Gamma^i_{jk} from Aris (1962) or any tensor analysis reference.
    #
    # For conformal coords where h1 = h2 = h:
    #   Gamma^1_{11} =  (1/h) * dh/dxi
    #   Gamma^1_{22} = -(1/h) * dh/dxi
    #   Gamma^1_{12} =  (1/h) * dh/deta
    #   Gamma^2_{11} = -(1/h) * dh/deta
    #   Gamma^2_{22} =  (1/h) * dh/deta
    #   Gamma^2_{12} =  (1/h) * dh/dxi

    Gamma_1_11 = inv_h * dh_dxi
    Gamma_1_22 = -inv_h * dh_dxi
    Gamma_1_12 = inv_h * dh_deta
    Gamma_2_11 = -inv_h * dh_deta
    Gamma_2_22 = inv_h * dh_deta
    Gamma_2_12 = inv_h * dh_dxi

    return {
        "h": h,
        "dh_dxi": dh_dxi,
        "dh_deta": dh_deta,
        "Gamma_1_11": Gamma_1_11,
        "Gamma_1_22": Gamma_1_22,
        "Gamma_1_12": Gamma_1_12,
        "Gamma_2_11": Gamma_2_11,
        "Gamma_2_22": Gamma_2_22,
        "Gamma_2_12": Gamma_2_12,
    }