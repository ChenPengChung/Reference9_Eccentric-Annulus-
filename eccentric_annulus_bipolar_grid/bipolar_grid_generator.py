"""
Bipolar Grid Generator -- Main Grid Engine
============================================
High-level class that orchestrates grid generation, metrics computation,
quality checks, export, and visualization.

Usage:
    gen = BipolarGridGenerator(r1=1.0, r2=3.0, eccentricity=0.5)
    gen.set_resolution(N_xi=200, N_eta=80)
    gen.set_stretching(stretch_eta=1.0)
    result = gen.generate()
    gen.export("output/grid.dat", format="tecplot")
    gen.plot("output/grid.png")
"""

import numpy as np
from pathlib import Path

from .bipolar_constants import (
    compute_bipolar_constants_snyder,
    cross_validate_constants,
)
from .bipolar_transform import (
    bipolar_to_cartesian,
    bipolar_to_cartesian_explicit,
    verify_roundtrip,
)
from .grid_metrics import (
    compute_scale_factor,
    compute_jacobian_analytic,
    compute_metric_tensor,
    compute_inverse_metric,
    compute_christoffel_symbols,
)
from .grid_stretching import (
    apply_radial_stretching,
    apply_circumferential_stretching,
    estimate_gilbm_stability_eccentric,
    print_gilbm_stability_warning_eccentric,
    check_eccentricity_stability,
)
from .grid_quality import (
    compute_grid_quality,
    compute_spacing_profile,
    print_grid_quality,
)
from .grid_io import write_tecplot_dat, write_csv, write_hdf5
from .grid_visualization import (
    plot_physical_grid,
    plot_computational_grid,
    plot_scale_factor_field,
    plot_jacobian_field,
    plot_radial_spacing_profile,
)


class BipolarGridGenerator:
    """
    Main grid generation class for eccentric annular domains.

    The generator uses bipolar (conformal) coordinate mapping to produce
    body-fitted structured meshes suitable for GILBM solvers.
    """

    def __init__(self, r1, r2, eccentricity):
        """
        Initialise with geometry and compute bipolar constants.

        Parameters
        ----------
        r1 : float
            Inner cylinder radius (> 0).
        r2 : float
            Outer cylinder radius (> r1).
        eccentricity : float
            Physical offset distance between circle centres (0 < e < r2-r1).
        """
        self.r1 = r1
        self.r2 = r2
        self.eccentricity = eccentricity
        self.epsilon = eccentricity / (r2 - r1)

        # Compute and cross-validate constants
        self.constants = cross_validate_constants(r1, r2, eccentricity)

        self.alpha = self.constants["alpha"]
        self.beta = self.constants["beta"]
        self.c = self.constants["c"]
        self.gamma_shift = self.constants["gamma_shift"]

        # Resolution (set later)
        self.N_xi = None
        self.N_eta = None

        # Stretching (defaults: uniform)
        self.stretch_xi = 0.0
        self.stretch_eta = 0.0
        self.alpha_s = 0.5

        # Generated data (populated by generate())
        self.x = None
        self.y = None
        self.Xi = None
        self.Eta = None
        self.metrics = None
        self.quality = None
        self.stability = None

    def set_resolution(self, N_xi, N_eta):
        """Set grid resolution."""
        if N_xi < 10:
            raise ValueError(f"N_xi must be >= 10, got {N_xi}")
        if N_eta < 5:
            raise ValueError(f"N_eta must be >= 5, got {N_eta}")
        self.N_xi = N_xi
        self.N_eta = N_eta

    def set_stretching(self, stretch_xi=0.0, stretch_eta=0.0, alpha_s=0.5):
        """
        Configure grid stretching (Vinokur tanh).

        Parameters
        ----------
        stretch_xi : float
            Circumferential stretching GAMMA (0 = uniform).
        stretch_eta : float
            Radial stretching GAMMA (0 = uniform).
        alpha_s : float
            Symmetry parameter (0.5 = symmetric).
        """
        self.stretch_xi = stretch_xi
        self.stretch_eta = stretch_eta
        self.alpha_s = alpha_s

    def generate(self):
        """
        Generate the complete grid.

        Steps:
          1. Create computational grid (xi, eta) with optional stretching
          2. Apply bipolar transformation to get (x, y)
          3. Compute Jacobian, metric tensor, scale factors
          4. Compute grid quality metrics
          5. Estimate GILBM stability

        Returns
        -------
        dict with keys:
            x, y           : physical coordinates (nj, ni)
            Xi, Eta        : computational coordinates (nj, ni)
            metrics        : dict of metric fields
            quality        : dict of quality metrics
            stability      : dict of GILBM stability info
            params         : dict of all parameters
        """
        if self.N_xi is None or self.N_eta is None:
            raise RuntimeError("Call set_resolution() before generate()")

        # Step 1: Computational grid with stretching
        xi = apply_circumferential_stretching(
            self.N_xi, self.stretch_xi, alpha_xi=0.5)
        eta = apply_radial_stretching(
            self.N_eta, self.alpha, self.beta,
            self.stretch_eta, self.alpha_s)

        Xi, Eta = np.meshgrid(xi, eta)

        # Step 2: Bipolar transformation
        x, y = bipolar_to_cartesian(Xi, Eta, self.c, self.gamma_shift)

        # Step 3: Metrics
        h = compute_scale_factor(Xi, Eta, self.c)
        jac = compute_jacobian_analytic(Xi, Eta, self.c)
        met = compute_metric_tensor(jac["J11"], jac["J12"],
                                     jac["J21"], jac["J22"])
        inv_met = compute_inverse_metric(jac["J11"], jac["J12"],
                                          jac["J21"], jac["J22"],
                                          jac["detJ"])
        christoffel = compute_christoffel_symbols(Xi, Eta, self.c)

        metrics = {
            "h": h,
            "detJ": jac["detJ"],
            "J11": jac["J11"], "J12": jac["J12"],
            "J21": jac["J21"], "J22": jac["J22"],
            "g11": met["g11"], "g12": met["g12"], "g22": met["g22"],
            "orthogonality_error": met["orthogonality_error"],
            **inv_met,
            **christoffel,
        }

        # Step 4: Quality
        quality = compute_grid_quality(x, y)

        # Step 5: GILBM stability
        stability = estimate_gilbm_stability_eccentric(
            x, y, Uref=0.05, Re=100, L_char=self.r2 - self.r1)

        # Store
        self.x = x
        self.y = y
        self.Xi = Xi
        self.Eta = Eta
        self.metrics = metrics
        self.quality = quality
        self.stability = stability

        self._params = self._build_params()

        return {
            "x": x, "y": y,
            "Xi": Xi, "Eta": Eta,
            "metrics": metrics,
            "quality": quality,
            "stability": stability,
            "params": self._params,
        }

    def generate_half_domain(self):
        """
        Generate only the upper half (y >= 0) using symmetry.

        Returns the same structure as generate() but with half the domain.
        """
        result = self.generate()

        # Find rows where mean y >= 0
        y_mean = result["y"].mean(axis=1)
        upper = y_mean >= -1e-10

        for key in ("x", "y", "Xi", "Eta"):
            result[key] = result[key][upper, :]

        for key, val in result["metrics"].items():
            if isinstance(val, np.ndarray) and val.ndim == 2:
                result["metrics"][key] = val[upper, :]

        # Recompute quality for half domain
        result["quality"] = compute_grid_quality(result["x"], result["y"])

        self.x = result["x"]
        self.y = result["y"]
        self.Xi = result["Xi"]
        self.Eta = result["Eta"]
        self.metrics = result["metrics"]
        self.quality = result["quality"]

        # Update params to reflect half-domain shape
        self._params["N_eta_actual"] = result["x"].shape[0]
        self._params["half_domain"] = True
        result["params"] = self._params

        return result

    def export(self, filepath, format="tecplot"):
        """
        Export grid in specified format.

        Parameters
        ----------
        filepath : str or Path
            Output file path.
        format : str
            "tecplot", "csv", or "hdf5".
        """
        if self.x is None:
            raise RuntimeError("Call generate() before export()")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        extra = {"h": self.metrics["h"], "detJ": self.metrics["detJ"]}

        if format == "tecplot":
            write_tecplot_dat(
                filepath, self.x, self.y,
                title=(f"Eccentric annulus R1={self.r1} R2={self.r2} "
                       f"eps={self.epsilon:.4f}"),
                zone_title=f"I{self.N_xi}_J{self.N_eta}",
                extra_fields=extra)
        elif format == "csv":
            write_csv(filepath, self.x, self.y,
                      xi=self.Xi, eta=self.Eta, extra_fields=extra)
        elif format == "hdf5":
            write_hdf5(filepath, self.x, self.y,
                       metrics=self.metrics, params=self._params)
        else:
            raise ValueError(f"Unknown format: {format}")

    def export_lbm_lattice(self, filepath, dx=None):
        """
        Export grid data formatted for LBM solver input.

        Includes: node positions, scale factors, Jacobian,
        boundary flags, and computational coordinates.

        Parameters
        ----------
        filepath : str or Path
            Output file path.
        dx : float or None
            Lattice spacing (auto if None).
        """
        if self.x is None:
            raise RuntimeError("Call generate() before export_lbm_lattice()")

        from .lbm_grid_adapter import create_lbm_lattice_info
        lattice = create_lbm_lattice_info(
            self.x, self.y, self.metrics["h"], self.metrics["detJ"],
            self.r1, self.r2, self.eccentricity,
            self.Xi, self.Eta)

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            filepath,
            **lattice,
            R1=self.r1, R2=self.r2, eccentricity=self.eccentricity,
            alpha=self.alpha, beta=self.beta,
            c=self.c, gamma_shift=self.gamma_shift,
        )
        print(f"  [written] {filepath}")

    def plot(self, savepath=None, output_dir=None, prefix="eccentric_annulus"):
        """
        Generate comprehensive visualisation.

        Parameters
        ----------
        savepath : str or None
            If given, save a single grid plot.
        output_dir : str or Path or None
            If given, save all plots to this directory.
        prefix : str
            Filename prefix.
        """
        if self.x is None:
            raise RuntimeError("Call generate() before plot()")

        if savepath:
            plot_physical_grid(self.x, self.y, params=self._params,
                               savepath=savepath)
            return

        if output_dir is None:
            return

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        tag = f"I{self.N_xi}_J{self.N_eta}_e{self.epsilon:.2f}"

        plot_physical_grid(
            self.x, self.y, params=self._params,
            savepath=output_dir / f"{prefix}_{tag}_grid.png")

        plot_computational_grid(
            self.Xi, self.Eta,
            savepath=output_dir / f"{prefix}_{tag}_comp.png")

        plot_scale_factor_field(
            self.x, self.y, self.metrics["h"],
            savepath=output_dir / f"{prefix}_{tag}_scale_factor.png")

        plot_jacobian_field(
            self.x, self.y, self.metrics["detJ"],
            savepath=output_dir / f"{prefix}_{tag}_jacobian.png")

        plot_radial_spacing_profile(
            self.x, self.y,
            savepath=output_dir / f"{prefix}_{tag}_spacing.png")

    def quality_report(self):
        """Print and return grid quality metrics."""
        if self.quality is None:
            raise RuntimeError("Call generate() before quality_report()")
        print_grid_quality(self.quality, self._params)
        return self.quality

    def stability_report(self):
        """Print and return GILBM stability report."""
        if self.stability is None:
            raise RuntimeError("Call generate() before stability_report()")
        print_gilbm_stability_warning_eccentric(
            self.stretch_eta,
            self.stability["omega"],
            self.stability["c_max"],
            self.stability["dt_global"],
            self.stability["dr_ratio"],
            self.stability["status"])
        return self.stability

    def _build_params(self):
        """Build unified parameter dictionary."""
        return {
            "R1": self.r1,
            "R2": self.r2,
            "epsilon": self.epsilon,
            "eccentricity": self.eccentricity,
            "alpha": self.alpha,
            "beta": self.beta,
            "c": self.c,
            "gamma_shift": self.gamma_shift,
            "gamma_r": self.constants["gamma_r"],
            "phi": self.constants["phi"],
            "kappa": self.constants["kappa"],
            "N_xi": self.N_xi,
            "N_eta": self.N_eta,
            "stretch_xi": self.stretch_xi,
            "stretch_eta": self.stretch_eta,
            "alpha_s": getattr(self, 'alpha_s', 0.5),
            "gap": self.r2 - self.r1,
        }