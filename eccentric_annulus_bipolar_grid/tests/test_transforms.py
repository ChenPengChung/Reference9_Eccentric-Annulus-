"""
Tests for bipolar coordinate transformations.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from eccentric_annulus_bipolar_grid.bipolar_constants import cross_validate_constants
from eccentric_annulus_bipolar_grid.bipolar_transform import (
    bipolar_to_cartesian,
    bipolar_to_cartesian_explicit,
    cartesian_to_bipolar,
    verify_roundtrip,
)


class TestForwardTransform:
    """Test (xi, eta) -> (x, y) transformation."""

    def setup_method(self):
        self.R1, self.R2, self.eps = 1.0, 3.0, 0.5
        self.e = self.eps * (self.R2 - self.R1)
        self.consts = cross_validate_constants(self.R1, self.R2, self.e)
        self.c = self.consts["c"]
        self.gamma_shift = self.consts["gamma_shift"]
        self.alpha = self.consts["alpha"]
        self.beta = self.consts["beta"]

    def test_complex_vs_explicit_agree(self):
        """Both forward transform methods should give identical results."""
        N_xi, N_eta = 50, 20
        xi = np.linspace(-np.pi, np.pi, N_xi)
        eta = np.linspace(self.alpha, self.beta, N_eta)
        Xi, Eta = np.meshgrid(xi, eta)

        x1, y1 = bipolar_to_cartesian(Xi, Eta, self.c, self.gamma_shift)
        x2, y2 = bipolar_to_cartesian_explicit(Xi, Eta, self.c, self.gamma_shift)

        np.testing.assert_allclose(x1, x2, atol=1e-10)
        np.testing.assert_allclose(y1, y2, atol=1e-10)

    def test_inner_boundary_is_circle(self):
        """Points at eta=beta should form a circle of radius R1."""
        N_xi = 200
        xi = np.linspace(-np.pi, np.pi, N_xi)
        eta = np.full_like(xi, self.beta)

        x, y = bipolar_to_cartesian(xi, eta, self.c, self.gamma_shift)

        # For z = c*tan(zeta/2), constant-eta lines are circles.
        # In the pre-shift coordinates (y' = y + gamma_shift):
        #   Circle center at (0, c*cosh(eta)/sinh(eta)), radius c/sinh(eta)
        # After shift: center_y = c*cosh(eta)/sinh(eta) - gamma_shift
        cy = self.c * np.cosh(self.beta) / np.sinh(self.beta) - self.gamma_shift
        cx = 0.0

        r = np.sqrt((x - cx)**2 + (y - cy)**2)
        np.testing.assert_allclose(r, self.R1, atol=1e-8)

    def test_outer_boundary_is_circle(self):
        """Points at eta=alpha should form a circle of radius R2."""
        N_xi = 200
        xi = np.linspace(-np.pi, np.pi, N_xi)
        eta = np.full_like(xi, self.alpha)

        x, y = bipolar_to_cartesian(xi, eta, self.c, self.gamma_shift)

        # Outer circle: center at (0, c*coth(alpha) - gamma_shift)
        # Since gamma_shift = c*coth(alpha), center_y = 0
        cy = self.c * np.cosh(self.alpha) / np.sinh(self.alpha) - self.gamma_shift
        cx = 0.0

        r = np.sqrt((x - cx)**2 + (y - cy)**2)
        np.testing.assert_allclose(r, self.R2, atol=1e-8)


class TestRoundtrip:
    """Test (xi, eta) -> (x, y) -> (xi, eta) roundtrip."""

    @pytest.mark.parametrize("R1,R2,eps", [
        (1.0, 3.0, 0.5),
        (5.0, 7.6, 0.5),
        (1.0, 2.0, 0.1),
        (2.0, 5.0, 0.7),
    ])
    def test_roundtrip_accuracy(self, R1, R2, eps):
        e = eps * (R2 - R1)
        consts = cross_validate_constants(R1, R2, e)
        c = consts["c"]
        gs = consts["gamma_shift"]
        alpha = consts["alpha"]
        beta = consts["beta"]

        N_xi, N_eta = 50, 20
        xi = np.linspace(-np.pi * 0.99, np.pi * 0.99, N_xi)
        eta = np.linspace(alpha * 1.01, beta * 0.99, N_eta)
        Xi, Eta = np.meshgrid(xi, eta)

        result = verify_roundtrip(Xi, Eta, c, gs)
        assert result["passed"], (
            f"Roundtrip failed: max_err_xi={result['max_err_xi']:.2e}, "
            f"max_err_eta={result['max_err_eta']:.2e}")


class TestConformalOrthogonality:
    """g12 metric component should be ~0 everywhere (conformal = orthogonal)."""

    def test_orthogonality(self):
        R1, R2, eps = 1.0, 3.0, 0.5
        e = eps * (R2 - R1)
        consts = cross_validate_constants(R1, R2, e)

        from eccentric_annulus_bipolar_grid.grid_metrics import (
            compute_jacobian_analytic, compute_metric_tensor)

        N_xi, N_eta = 100, 40
        xi = np.linspace(-np.pi, np.pi, N_xi)
        eta = np.linspace(consts["alpha"], consts["beta"], N_eta)
        Xi, Eta = np.meshgrid(xi, eta)

        jac = compute_jacobian_analytic(Xi, Eta, consts["c"])
        met = compute_metric_tensor(jac["J11"], jac["J12"],
                                     jac["J21"], jac["J22"])

        # For conformal mapping, g12 should be ~0
        assert met["orthogonality_error"] < 1e-10, (
            f"Orthogonality error too large: {met['orthogonality_error']:.2e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
