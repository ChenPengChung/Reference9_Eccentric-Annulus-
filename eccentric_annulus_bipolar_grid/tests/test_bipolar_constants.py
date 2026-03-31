"""
Tests for bipolar coordinate constants computation.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from eccentric_annulus_bipolar_grid.bipolar_constants import (
    compute_bipolar_constants_pdf,
    compute_bipolar_constants_snyder,
    cross_validate_constants,
)


class TestKnownCase:
    """Test against existing code values: R1=5, R2=7.6, eps=0.5."""

    def setup_method(self):
        self.R1 = 5.0
        self.R2 = 7.6
        self.epsilon = 0.5
        self.e = self.epsilon * (self.R2 - self.R1)

    def test_snyder_constants(self):
        result = compute_bipolar_constants_snyder(self.R1, self.R2, self.epsilon)
        # Known values from eccentric_annulus_grid_tool.py
        assert result["alpha"] == pytest.approx(1.1542, abs=0.001)
        assert result["beta"] == pytest.approx(1.5175, abs=0.001)
        assert result["c"] == pytest.approx(10.8537, abs=0.001)

    def test_pdf_constants(self):
        result = compute_bipolar_constants_pdf(self.R1, self.R2, self.e)
        # PDF alpha should match Snyder beta (inner circle)
        assert result["alpha_pdf"] == pytest.approx(1.5175, abs=0.001)
        # PDF beta should match Snyder alpha (outer circle)
        assert result["beta_pdf"] == pytest.approx(1.1542, abs=0.001)
        assert result["c"] == pytest.approx(10.8537, abs=0.001)


class TestPDFvsSnyderEquivalence:
    """Both formulations give same alpha, beta, c."""

    @pytest.mark.parametrize("R1,R2,eps", [
        (1.0, 3.0, 0.5),
        (5.0, 7.6, 0.5),
        (1.0, 2.0, 0.1),
        (1.0, 2.0, 0.9),
        (2.0, 5.0, 0.3),
    ])
    def test_cross_validation(self, R1, R2, eps):
        e = eps * (R2 - R1)
        result = cross_validate_constants(R1, R2, e)
        assert result["validation"]["passed"], (
            f"Cross-validation failed: {result['validation']}"
        )
        assert result["validation"]["diff_c"] < 1e-10


class TestBoundaryCircles:
    """Check that eta=alpha maps to R2 circle and eta=beta to R1 circle."""

    def test_outer_circle_radius(self):
        R1, R2, eps = 1.0, 3.0, 0.5
        e = eps * (R2 - R1)
        consts = cross_validate_constants(R1, R2, e)
        c = consts["c"]
        gamma_shift = consts["gamma_shift"]
        alpha = consts["alpha"]

        # At eta=alpha, the constant-eta circle has:
        # center at (c*coth(alpha), 0) and radius c/sinh(alpha)
        r_outer = c / np.sinh(alpha)
        assert r_outer == pytest.approx(R2, abs=1e-6)

    def test_inner_circle_radius(self):
        R1, R2, eps = 1.0, 3.0, 0.5
        e = eps * (R2 - R1)
        consts = cross_validate_constants(R1, R2, e)
        c = consts["c"]
        beta = consts["beta"]

        r_inner = c / np.sinh(beta)
        assert r_inner == pytest.approx(R1, abs=1e-6)


class TestEdgeCases:
    """Test error handling for edge cases."""

    def test_zero_eccentricity_raises(self):
        with pytest.raises(ValueError, match="must be > 0"):
            compute_bipolar_constants_pdf(1.0, 3.0, 0.0)

    def test_zero_eccentricity_snyder_raises(self):
        with pytest.raises(ValueError, match="must be in"):
            compute_bipolar_constants_snyder(1.0, 3.0, 0.0)

    def test_r1_greater_than_r2_raises(self):
        with pytest.raises(ValueError, match="must be > R1"):
            compute_bipolar_constants_snyder(5.0, 3.0, 0.5)

    def test_eccentricity_too_large_raises(self):
        with pytest.raises(ValueError, match="must be < gap"):
            compute_bipolar_constants_pdf(1.0, 3.0, 2.5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
