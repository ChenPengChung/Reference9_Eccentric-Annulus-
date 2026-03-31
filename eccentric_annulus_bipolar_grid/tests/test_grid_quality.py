"""
Tests for grid quality metrics and the full grid generation pipeline.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from eccentric_annulus_bipolar_grid.bipolar_grid_generator import BipolarGridGenerator
from eccentric_annulus_bipolar_grid.grid_quality import compute_grid_quality


class TestGridGeneration:
    """Test the full grid generation pipeline."""

    def test_basic_generation(self):
        gen = BipolarGridGenerator(r1=1.0, r2=3.0, eccentricity=1.0)
        gen.set_resolution(N_xi=50, N_eta=20)
        result = gen.generate()

        assert result["x"].shape == (20, 50)
        assert result["y"].shape == (20, 50)
        assert result["Xi"].shape == (20, 50)
        assert result["Eta"].shape == (20, 50)

    def test_stretched_generation(self):
        gen = BipolarGridGenerator(r1=1.0, r2=3.0, eccentricity=1.0)
        gen.set_resolution(N_xi=50, N_eta=20)
        gen.set_stretching(stretch_eta=1.0, alpha_s=0.5)
        result = gen.generate()

        assert result["x"].shape == (20, 50)
        assert "omega" in result["stability"]

    def test_quality_metrics_exist(self):
        gen = BipolarGridGenerator(r1=1.0, r2=3.0, eccentricity=1.0)
        gen.set_resolution(N_xi=50, N_eta=20)
        result = gen.generate()

        q = result["quality"]
        assert "min_area" in q
        assert "max_area" in q
        assert "area_ratio" in q
        assert "min_aspect" in q
        assert "orthogonality_max_deg" in q


class TestGridOrthogonality:
    """Conformal bipolar grids should be orthogonal."""

    @pytest.mark.parametrize("eps", [0.1, 0.3, 0.5, 0.7])
    def test_orthogonality_vs_eccentricity(self, eps):
        e = eps * 2.0  # gap = R2 - R1 = 2.0
        gen = BipolarGridGenerator(r1=1.0, r2=3.0, eccentricity=e)
        gen.set_resolution(N_xi=100, N_eta=40)
        result = gen.generate()

        # Orthogonality from metric tensor (analytic)
        assert result["metrics"]["orthogonality_error"] < 1e-10

        # Orthogonality from cell-level check (numerical, coarser tolerance)
        # At 100x40 resolution with high eccentricity, cell-level deviations
        # can reach ~6-7 degrees due to large aspect ratios in narrow gap.
        # The analytic check (metric tensor) confirms true orthogonality.
        assert result["quality"]["orthogonality_max_deg"] < 10.0


class TestGILBMStability:
    """Test GILBM stability estimation."""

    def test_stability_returns_valid(self):
        gen = BipolarGridGenerator(r1=1.0, r2=3.0, eccentricity=1.0)
        gen.set_resolution(N_xi=50, N_eta=20)
        result = gen.generate()

        s = result["stability"]
        assert s["omega"] > 0.5
        assert s["c_max"] > 0
        assert s["dr_ratio"] > 1
        assert s["status"] in ("GOOD", "OPTIMAL", "OK", "MARGINAL", "UNSTABLE")

    def test_higher_gamma_increases_omega(self):
        """More stretching -> higher omega (less stable)."""
        omegas = []
        for gamma in [0.0, 1.0, 2.0]:
            gen = BipolarGridGenerator(r1=1.0, r2=3.0, eccentricity=1.0)
            gen.set_resolution(N_xi=50, N_eta=20)
            gen.set_stretching(stretch_eta=gamma)
            result = gen.generate()
            omegas.append(result["stability"]["omega"])

        # omega should increase with GAMMA
        assert omegas[1] >= omegas[0]
        assert omegas[2] >= omegas[1]


class TestPeriodicity:
    """Test that xi = -pi and xi = +pi produce the same physical points."""

    def test_periodic_seam(self):
        gen = BipolarGridGenerator(r1=1.0, r2=3.0, eccentricity=1.0)
        gen.set_resolution(N_xi=100, N_eta=30)
        result = gen.generate()

        x, y = result["x"], result["y"]

        # First and last columns should be the same (periodic)
        np.testing.assert_allclose(x[:, 0], x[:, -1], atol=1e-10)
        np.testing.assert_allclose(y[:, 0], y[:, -1], atol=1e-10)


class TestConcentric:
    """Test near-concentric limit (small eccentricity)."""

    def test_small_eccentricity_approaches_concentric(self):
        """Grid with tiny eccentricity should approximate concentric annulus."""
        gen = BipolarGridGenerator(r1=1.0, r2=3.0, eccentricity=0.01)
        gen.set_resolution(N_xi=100, N_eta=30)
        result = gen.generate()

        x, y = result["x"], result["y"]

        # Outer boundary should be approximately circular centered at origin
        r_outer = np.sqrt(x[0, :]**2 + y[0, :]**2)
        # Allow some deviation because the circle is not exactly centered at origin
        # (it's centered at (gamma_shift, -gamma_shift) shifted)
        # But the radius should be close to R2
        assert np.std(r_outer) / np.mean(r_outer) < 0.05


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
