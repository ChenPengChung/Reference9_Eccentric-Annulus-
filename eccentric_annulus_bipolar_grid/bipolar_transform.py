"""
Bipolar Coordinate Transformations
====================================
Forward: (xi, eta) -> (x, y)
Inverse: (x, y) -> (xi, eta)

The conformal mapping:
    z = c * tan(zeta/2) - i * gamma_shift
where zeta = xi + i*eta.

Two methods are provided for the forward transform:
  1. Complex arithmetic (used for grid generation)
  2. Explicit real formulas from PDF Eqs. (3a, 3b) (for verification)
"""

import numpy as np


def bipolar_to_cartesian(xi, eta, c, gamma_shift):
    """
    Forward transformation: (xi, eta) -> (x, y).

    Uses the complex conformal mapping:
        z = c * tan((xi + i*eta) / 2)
        x = Re(z),  y = Im(z) - gamma_shift

    Parameters
    ----------
    xi : ndarray
        Circumferential coordinate, xi in [-pi, pi].
    eta : ndarray
        Radial coordinate, eta in [alpha, beta].
    c : float
        Bipolar pole distance.
    gamma_shift : float
        Vertical shift to center outer circle at origin.

    Returns
    -------
    x, y : ndarray
        Physical Cartesian coordinates.
    """
    zeta = xi + 1j * eta
    z_phys = c * np.tan(zeta / 2)
    x = np.real(z_phys)
    y = np.imag(z_phys) - gamma_shift
    return x, y


def bipolar_to_cartesian_explicit(xi, eta, c, gamma_shift):
    """
    Forward transformation using explicit real formulas derived from
    the code's conformal mapping z = c * tan(zeta/2).

    Using the identity tan(a + ib) = [sin(2a) + i*sinh(2b)] / [cos(2a) + cosh(2b)]
    with a = xi/2, b = eta/2:

        x = c * sin(xi)   / (cos(xi) + cosh(eta))
        y = c * sinh(eta)  / (cos(xi) + cosh(eta)) - gamma_shift

    This is mathematically equivalent to the complex formula but
    uses only real arithmetic. Useful for verification.

    NOTE: The PDF document uses a different convention (z = ic*cot(zeta/2))
    where poles are on the x-axis. The code's convention (z = c*tan(zeta/2))
    has poles on the y-axis. The formulas here match the code's convention.

    Parameters
    ----------
    xi, eta : ndarray
        Bipolar coordinates.
    c : float
        Pole distance.
    gamma_shift : float
        Vertical shift.

    Returns
    -------
    x, y : ndarray
        Physical Cartesian coordinates.
    """
    denom = np.cos(xi) + np.cosh(eta)
    x = c * np.sin(xi) / denom
    y = c * np.sinh(eta) / denom - gamma_shift
    return x, y


def cartesian_to_bipolar(x, y, c, gamma_shift):
    """
    Inverse transformation: (x, y) -> (xi, eta).

    Derived from the code's convention z = c * tan(zeta/2):
        w = x + i*y' where y' = y + gamma_shift
        tan(zeta/2) = w / c
        zeta = 2 * arctan(w / c)

    In real coordinates:
        eta = 0.5 * ln( [x^2 + (y'+c)^2] / [x^2 + (y'-c)^2] )
        xi  = atan2(2*x*c, c^2 - x^2 - y'^2)

    NOTE: These differ from the PDF formulas because the code uses
    z = c*tan(zeta/2) (poles on y-axis) while the PDF uses
    z = ic*cot(zeta/2) (poles on x-axis).

    Parameters
    ----------
    x, y : ndarray
        Physical Cartesian coordinates.
    c : float
        Bipolar pole distance.
    gamma_shift : float
        Vertical shift.

    Returns
    -------
    xi, eta : ndarray
        Bipolar coordinates. xi in [-pi, pi], eta > 0.
    """
    yp = y + gamma_shift  # undo shift

    # e^(2*eta) = [x^2 + (y'+c)^2] / [x^2 + (y'-c)^2]
    num = x**2 + (yp + c)**2
    den = x**2 + (yp - c)**2
    # Protect against log(0) at the poles
    ratio = np.where(den > 1e-30, num / den, 1e30)
    eta = 0.5 * np.log(ratio)

    # tan(xi) = 2*x*c / (c^2 - x^2 - y'^2)
    xi = np.arctan2(2 * x * c, c**2 - x**2 - yp**2)

    return xi, eta


def verify_roundtrip(xi_orig, eta_orig, c, gamma_shift, tol=1e-10):
    """
    Verify forward->inverse roundtrip accuracy.

    Parameters
    ----------
    xi_orig, eta_orig : ndarray
        Original bipolar coordinates.
    c, gamma_shift : float
        Bipolar constants.
    tol : float
        Tolerance for maximum absolute error.

    Returns
    -------
    dict with keys:
        max_err_xi, max_err_eta, passed
    """
    x, y = bipolar_to_cartesian(xi_orig, eta_orig, c, gamma_shift)
    xi_rec, eta_rec = cartesian_to_bipolar(x, y, c, gamma_shift)

    # Handle xi periodicity: both should be in [-pi, pi]
    dxi = xi_rec - xi_orig
    # Wrap to [-pi, pi]
    dxi = (dxi + np.pi) % (2 * np.pi) - np.pi

    max_err_xi = np.max(np.abs(dxi))
    max_err_eta = np.max(np.abs(eta_rec - eta_orig))

    return {
        "max_err_xi": max_err_xi,
        "max_err_eta": max_err_eta,
        "passed": max(max_err_xi, max_err_eta) <= tol,
    }
