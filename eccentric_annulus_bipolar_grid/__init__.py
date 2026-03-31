"""
Eccentric Annulus Bipolar Grid Generator
========================================
Structured grid generator for eccentric annular domains using
bipolar coordinate conformal mapping, designed for general
interpolation Lattice Boltzmann Method (GILBM) solvers.

Mathematical foundation:
  Snyder & Goldstein (1965), "Fully developed laminar flow in an eccentric annulus"
  Lauer-Bare Z. & Gaertig E., "Conformal Mappings with SymPy", SciPy 2021
"""

from .bipolar_constants import (
    compute_bipolar_constants_pdf,
    compute_bipolar_constants_snyder,
    cross_validate_constants,
)
from .bipolar_transform import bipolar_to_cartesian, cartesian_to_bipolar
from .bipolar_grid_generator import BipolarGridGenerator
