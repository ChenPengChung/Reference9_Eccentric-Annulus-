"""
Grid Metrics: Jacobian, Metric Tensor, Scale Factors, Christoffel Symbols
==========================================================================
Critical for GILBM: the curvilinear grid metrics determine how the LBM
collision-streaming operates on the non-uniform mesh.

For bipolar coordinates, the mapping is conformal, so:
  - The scale factor h = c / (cosh(eta) - cos(xi)) characterises everything
  - The metric tensor is diagonal: g11 = g22 = h^2, g12 = 0
  - The Jacobian determinant |J| = h^2
"""

import numpy as np


def compute_scale_factor(xi, eta, c):
    """
    Bipolar coordinate scale factor.

        h(xi, eta) = c / (cosh(eta) - cos(xi))

    This is the key metric for LBM: it relates lattice spacing
    to physical spacing at each grid point.

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
    return c / (np.cosh(eta) - np.cos(xi))


def compute_jacobian_analytic(xi, eta, c):
    """
    Compute the Jacobian matrix components analytically for bipolar coordinates.

    The conformal mapping z = c * tan(zeta/2) has derivative:
        dz/dzeta = (c/2) / cos^2(zeta/2)

    For the Jacobian matrix J = [[dx/dxi, dx/deta], [dy/dxi, dy/deta]]:

    Since the mapping is conformal, we can use the Cauchy-Riemann equations.
    Let w = u + iv = f(zeta) where zeta = xi + i*eta:
        dx/dxi = dy/deta = Re(dw/dzeta)    (= h_x term)
        dx/deta = -dy/dxi = Im(dw/dzeta)   (Cauchy-Riemann)

    But it's simpler to compute via the scale factor h:
        |dz/dzeta| = h = c / (cosh(eta) - cos(xi))

    For conformal mappings, the Jacobian determinant |J| = h^2.

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

    # Analytic partial derivatives of the conformal mapping
    # z = c * tan((xi + i*eta)/2)
    # dz/dxi = (c/2) * sec^2((xi+i*eta)/2) (has real and imag parts)
    zeta = xi + 1j * eta
    half_zeta = zeta / 2.0
    cos2 = np.cos(half_zeta)**2
    # dz/dzeta = c / (2 * cos^2(zeta/2))
    dz_dzeta = c / (2.0 * cos2)

    # dz/dxi = dz/dzeta * dzeta/dxi = dz/dzeta * 1
    # dz/deta = dz/dzeta * dzeta/deta = dz/dzeta * i
    dz_dxi = dz_dzeta
    dz_deta = 1j * dz_dzeta

    # J = [[dx/dxi, dx/deta],
    #      [dy/dxi, dy/deta]]
    # Note: y includes the -gamma_shift, but shift doesn't affect derivatives
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
    J11[:, 1:-1] = (x[:, 2:] - x[:, :-2]) / 2.0
    J21[:, 1:-1] = (y[:, 2:] - y[:, :-2]) / 2.0
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

    # Orthogonality check: g12 should be ~0 for conformal mapping
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

    For bipolar (orthogonal, conformal) coordinates with scale factor
    h = c / (cosh(eta) - cos(xi)):

        dh/dxi  = c * sin(xi) / (cosh(eta) - cos(xi))^2
                 = h * sin(xi) / (cosh(eta) - cos(xi))

        dh/deta = -c * sinh(eta) / (cosh(eta) - cos(xi))^2
                 = -h * sinh(eta) / (cosh(eta) - cos(xi))

    Christoffel symbols (for orthogonal coordinates):
        Gamma^1_{11} = (1/h) * dh/dxi
        Gamma^1_{22} = -(1/h) * dh/dxi
        Gamma^2_{12} = (1/h) * dh/deta   (= Gamma^2_{21})
        Gamma^2_{22} = (1/h) * dh/deta
        Gamma^1_{12} = (1/h) * dh/deta   (note: for conformal h1=h2=h)

    Parameters
    ----------
    xi, eta : ndarray (2D)
        Bipolar coordinates.
    c : float
        Pole distance.

    Returns
    -------
    dict with Christoffel symbol arrays.
    """
    denom = np.cosh(eta) - np.cos(xi)
    h = c / denom

    # Partial derivatives of h
    dh_dxi = h * np.sin(xi) / denom
    dh_deta = -h * np.sinh(eta) / denom

    # For conformal coordinates (h1 = h2 = h):
    inv_h = 1.0 / np.where(np.abs(h) > 1e-30, h, 1e-30)

    return {
        "dh_dxi": dh_dxi,
        "dh_deta": dh_deta,
        "Gamma_xi_xi_xi": inv_h * dh_dxi,
        "Gamma_xi_eta_eta": -inv_h * dh_dxi,
        "Gamma_eta_xi_xi": inv_h * dh_deta,
        "Gamma_eta_eta_eta": inv_h * dh_deta,
        "Gamma_eta_xi_eta": inv_h * dh_dxi,   # mixed
        "Gamma_xi_xi_eta": inv_h * dh_deta,    # mixed
    }
