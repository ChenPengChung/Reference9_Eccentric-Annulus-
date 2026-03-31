"""
Tests for conformal consistency: detJ == h^2, metric tensor, and Christoffel symbols.

This test suite catches CRITICAL-1 type bugs where the scale factor formula
doesn't match the actual mapping used.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from eccentric_annulus_bipolar_grid.bipolar_constants import cross_validate_constants
from eccentric_annulus_bipolar_grid.grid_metrics import (
    compute_scale_factor,
    compute_jacobian_analytic,
    compute_metric_tensor,
    compute_christoffel_symbols,
)


class TestConformalDetJEqualsH2:
    """
    For a conformal mapping, |J| = h^2 EXACTLY.
    This is the single most important consistency check.
    """

    @pytest.mark.parametrize("R1,R2,eps", [
        (1.0, 3.0, 0.5),
        (5.0, 7.6, 0.5),
        (1.0, 2.0, 0.1),
        (2.0, 5.0, 0.7),
        (1.0, 3.0, 0.9),
    ])
    def test_detJ_equals_h_squared(self, R1, R2, eps):
        """detJ must equal h^2 everywhere for conformal mapping."""
        e = eps * (R2 - R1)
        consts = cross_validate_constants(R1, R2, e)
        c = consts["c"]
        alpha = consts["alpha"]
        beta = consts["beta"]

        N_xi, N_eta = 100, 40
        xi = np.linspace(-np.pi * 0.99, np.pi * 0.99, N_xi)
        eta = np.linspace(alpha + 0.01, beta - 0.01, N_eta)
        Xi, Eta = np.meshgrid(xi, eta)

        h = compute_scale_factor(Xi, Eta, c)
        jac = compute_jacobian_analytic(Xi, Eta, c)

        # The key test: detJ should equal h^2
        rel_err = np.abs(jac["detJ"] - h**2) / (h**2 + 1e-30)
        max_rel_err = rel_err.max()

        assert max_rel_err < 1e-10, (
            f"detJ != h^2: max relative error = {max_rel_err:.2e} "
            f"(should be < 1e-10 for conformal mapping)")


class TestMetricTensorConformal:
    """For conformal mapping, g11 = g22 = h^2 and g12 = 0."""

    def test_diagonal_metric(self):
        R1, R2, eps = 1.0, 3.0, 0.5
        e = eps * (R2 - R1)
        consts = cross_validate_constants(R1, R2, e)
        c = consts["c"]
        alpha, beta = consts["alpha"], consts["beta"]

        N_xi, N_eta = 80, 30
        xi = np.linspace(-np.pi * 0.99, np.pi * 0.99, N_xi)
        eta = np.linspace(alpha + 0.01, beta - 0.01, N_eta)
        Xi, Eta = np.meshgrid(xi, eta)

        h = compute_scale_factor(Xi, Eta, c)
        jac = compute_jacobian_analytic(Xi, Eta, c)
        met = compute_metric_tensor(jac["J11"], jac["J12"],
                                     jac["J21"], jac["J22"])

        # g11 = g22 = h^2
        np.testing.assert_allclose(met["g11"], h**2, rtol=1e-10)
        np.testing.assert_allclose(met["g22"], h**2, rtol=1e-10)

        # g12 = 0
        assert met["orthogonality_error"] < 1e-10


class TestChristoffelSymbols:
    """Basic sanity checks on Christoffel symbols."""

    def test_christoffel_at_symmetric_point(self):
        """At xi=0, sin(xi)=0, so dh/dxi=0 and related Gammas vanish."""
        R1, R2, eps = 1.0, 3.0, 0.5
        e = eps * (R2 - R1)
        consts = cross_validate_constants(R1, R2, e)
        c = consts["c"]
        alpha, beta = consts["alpha"], consts["beta"]

        eta_mid = (alpha + beta) / 2
        xi = np.array([[0.0]])
        eta = np.array([[eta_mid]])

        chris = compute_christoffel_symbols(xi, eta, c)

        # At xi=0, sin(0) = 0, so dh/dxi = 0
        assert abs(chris["dh_dxi"][0, 0]) < 1e-14
        # Gamma_1_11 = (1/h)*dh/dxi = 0
        assert abs(chris["Gamma_1_11"][0, 0]) < 1e-14
        # dh/deta should be nonzero (sinh(eta_mid) != 0)
        assert abs(chris["dh_deta"][0, 0]) > 1e-6

    def test_christoffel_symmetry(self):
        """Gamma^1_{12} should use dh/deta, Gamma^2_{12} should use dh/dxi."""
        R1, R2, eps = 1.0, 3.0, 0.5
        e = eps * (R2 - R1)
        consts = cross_validate_constants(R1, R2, e)
        c = consts["c"]

        xi = np.array([[1.0]])
        eta = np.array([[(consts["alpha"] + consts["beta"]) / 2]])

        chris = compute_christoffel_symbols(xi, eta, c)
        h = c / (np.cos(xi) + np.cosh(eta))

        # Verify: Gamma_1_12 = (1/h) * dh/deta
        expected = chris["dh_deta"] / h
        np.testing.assert_allclose(chris["Gamma_1_12"], expected, rtol=1e-12)

        # Verify: Gamma_2_12 = (1/h) * dh/dxi
        expected2 = chris["dh_dxi"] / h
        np.testing.assert_allclose(chris["Gamma_2_12"], expected2, rtol=1e-12)


class TestScaleFactorRange:
    """Scale factor h should be positive everywhere in the domain."""

    @pytest.mark.parametrize("eps", [0.1, 0.5, 0.9])
    def test_h_positive(self, eps):
        R1, R2 = 1.0, 3.0
        e = eps * (R2 - R1)
        consts = cross_validate_constants(R1, R2, e)
        c = consts["c"]
        alpha, beta = consts["alpha"], consts["beta"]

        xi = np.linspace(-np.pi, np.pi, 100)
        eta = np.linspace(alpha, beta, 40)
        Xi, Eta = np.meshgrid(xi, eta)

        h = compute_scale_factor(Xi, Eta, c)
        assert np.all(h > 0), f"Scale factor has non-positive values: min={h.min()}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
