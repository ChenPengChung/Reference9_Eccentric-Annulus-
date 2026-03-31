# Eccentric Annulus Bipolar Grid Generator -- Code Review Report

**Date:** 2026-03-31
**Reviewer:** Claude (automated strict code review)
**Scope:** All 13 Python modules + 3 test files in `eccentric_annulus_bipolar_grid/`
**Method:** Static analysis + end-to-end numerical verification

---

## Executive Summary

| Severity | Count | Description |
|----------|-------|-------------|
| **CRITICAL** | 4 | Blocks correctness; must fix before any production use |
| **MAJOR** | 12 | Significant risk of incorrect results or instability |
| **MINOR** | 15 | Code quality, documentation, edge cases |
| **SUGGESTION** | 8 | Improvements, not bugs |

**Unit tests: 32/32 PASS** -- however, the tests have loose tolerances and miss the critical bug.

**End-to-end verification: 5/7 PASS, 2 FAIL**
- FAIL: Jacobian determinant != h^2 (77% relative error)
- FAIL: Orthogonality max deviation = 1.34 deg (should be ~0 for conformal)

---

## CRITICAL-1: Scale Factor Formula Wrong for tan Mapping (grid_metrics.py:37)

**This is the most serious bug in the entire codebase.**

The code uses `h = c / (cosh(eta) - cos(xi))` which is the scale factor for the
**cot-based** bipolar mapping `z = ic * cot((xi+i*eta)/2)` (PDF Eq. 3a-3b).

But the actual mapping implemented in the code is the **tan-based** variant:
`z = c * tan(zeta/2)`.

The correct scale factor for `z = c * tan(zeta/2)` is:

```
h_correct = c / (cos(xi) + cosh(eta))     <-- NOTE: PLUS not MINUS
```

**Proof (verified numerically):**
```
|cos((xi+i*eta)/2)|^2 = (cos(xi) + cosh(eta)) / 2

|f'(zeta)| = |dz/dzeta| = (c/2) / |cos(zeta/2)|^2
           = (c/2) / [(cos(xi) + cosh(eta)) / 2]
           = c / (cos(xi) + cosh(eta))
```

**Impact:** The Jacobian computed via `compute_jacobian_analytic()` (line 82) is CORRECT
because it uses the complex derivative directly. But `compute_scale_factor()` (line 37)
returns the WRONG `h`. This means:

1. `h` field in metrics dict is WRONG -- **affects all LBM scale factor usage**
2. `detJ != h^2` (verified: 77% relative error) -- **inconsistency between Jacobian and scale factor**
3. Christoffel symbols use wrong `h` -- **wrong curvature terms for LBM**
4. Stability estimation `dr_ratio` may be indirectly affected

**Fix:**
```python
# grid_metrics.py line 37
def compute_scale_factor(xi, eta, c):
    return c / (np.cosh(eta) + np.cos(xi))   # PLUS for tan mapping
```

**Verification after fix:** `detJ / h^2` should equal 1.0 everywhere (conformal property).

---

## CRITICAL-2: LBM Interpolation Weights Not Implemented (lbm_grid_adapter.py)

`compute_interpolation_weights()` computes only departure point locations but does NOT
compute the actual bilinear/bicubic interpolation weights needed for the GILBM streaming step.
The function name and docstring are misleading.

Additionally, the departure point calculation (line 162-163) uses lattice velocities
directly without contravariant transformation through the metric tensor. For curvilinear
grids, D2Q9 velocity `e_alpha` in physical space must be transformed to computational
space via `c_tilde = J^{-T} * e_alpha`.

**Missing entirely:** Neighbor connectivity function (finding surrounding lattice nodes
from departure point for interpolation stencil construction).

**Impact:** The LBM solver integration is non-functional.

---

## CRITICAL-3: Pole Singularity Not Protected (bipolar_transform.py)

The `bipolar_to_cartesian_explicit()` function has `denom = cos(xi) + cosh(eta)` which
equals zero at the bipolar poles (xi=+-pi, eta=0). No protection exists.

The inverse transform `cartesian_to_bipolar()` clamps the denominator to 1e-30 at poles,
producing `eta = 0.5*log(1e-30) = -34.5`, which is physically meaningless.

**Fix:** Add NaN masking at poles or clip eta to valid domain bounds.

---

## CRITICAL-4: Config Default Inconsistency (config.py vs main.py)

Config file mode defaults STRETCH_ETA=0.0 (uniform), but interactive mode
defaults GAMMA=1.0 (stretched). A user running `--auto` with default config gets a
fundamentally different grid than interactive mode with default answers.

Default eccentricity e=0.5 can exceed gap (r2-r1) for small annuli (e.g., R1=1, R2=1.4),
producing invalid geometry without error.

---

## MAJOR Issues (Top 6)

### MAJOR-1: Numerical Jacobian Stencil Error (grid_metrics.py:129-144)

The numerical Jacobian uses `(f[i+2] - f[i-2]) / 2.0` for interior points. This computes
a second-order derivative over a stride of 2 cells, but divides by 2.0 instead of 4.0
(the correct denominator for stride-2 central difference). This gives a systematic 2x error.

**Should be:** `(f[i+1] - f[i-1]) / 2.0` (standard central difference).

### MAJOR-2: Christoffel Symbols Incomplete (grid_metrics.py:256-274)

Only 6 of 8 Christoffel symbols are computed, with labeling errors. Line 273 uses
`dh_dxi` where `dh_deta` is needed. The complete set for orthogonal conformal coordinates
requires careful derivation from the (corrected) scale factor.

### MAJOR-3: Stability Table k-factors Don't Match Calibration (grid_stretching.py:465)

`estimate_max_resolution()` uses empirical factors `k_base = 1.775 * (1 + 0.5*epsilon)`,
predicting c_max=177.5 for eps=0.5/N_eta=80. But calibration shows c_max=142 (25% error).
This function gives unreliable resolution recommendations.

### MAJOR-4: Stability Table Display Order (main.py:284-293)

The stability reference table IS printed before GAMMA selection (correct), but the
explanatory text and GAMMA guidance are interleaved, reducing clarity. More importantly,
there is no blocking confirmation when user selects GAMMA >= 2.5 (known UNSTABLE).

### MAJOR-5: No Post-Generation Stability Recheck (main.py:330-357)

After grid generation with user's actual parameters, the stability is estimated but
the result is only printed -- not compared against the pre-generation prediction.
If the actual omega exceeds 2.0, no abort or strong warning is issued.

### MAJOR-6: orthogonality_max_deg Reports Non-Zero for Conformal Grid (grid_quality.py)

The grid quality metric reports 1.34 degrees max orthogonality deviation for a conformal
(inherently orthogonal) grid. This is because the quality metric uses finite-difference
edge vectors with limited accuracy. The metric is misleading and should either use
analytical Jacobian or note the FD limitation in the report.

---

## MINOR Issues Summary

| # | Module | Issue |
|---|--------|-------|
| 1 | bipolar_constants.py | No bounds check on arccosh input (cosh_alpha >= 1 not validated) |
| 2 | bipolar_constants.py | No check that F^2 - R2^2 >= 0 before sqrt in Snyder method |
| 3 | bipolar_constants.py | Cross-validation doesn't check gamma_shift |
| 4 | bipolar_transform.py | Complex tan overflows for very large eta |
| 5 | bipolar_transform.py | Inverse transform arctan2(0,0) undefined at poles |
| 6 | grid_quality.py | Aspect ratio definition non-standard for curvilinear grids |
| 7 | grid_quality.py | Orthogonality check examines only one corner of each cell |
| 8 | grid_io.py | HDF5 export fails silently without h5py |
| 9 | grid_io.py | Reader discards extra fields written by writer |
| 10 | grid_visualization.py | pcolormesh indexing wrong for cell-centered quality data |
| 11 | config.py | Comment stripping breaks on scientific notation (1e-6 # comment) |
| 12 | config.py | No bounds validation on STRETCH_*, ALPHA_S, RE, CFL |
| 13 | lbm_grid_adapter.py | Boundary node docstring reverses j-index convention |
| 14 | lbm_grid_adapter.py | Periodic xi wrapping has floating-point edge case at +-pi |
| 15 | bipolar_grid_generator.py | generate_half_domain() doesn't update self._params |

---

## Test Suite Assessment

All 32 tests pass, but the test suite has significant coverage gaps:

1. **No test verifies `detJ == h^2`** -- would have caught CRITICAL-1
2. **Tolerances are loose** (1e-3 to 1e-5) where conformal properties should hold to 1e-12
3. **No test for Christoffel symbols** correctness
4. **No test for LBM adapter** functions
5. **Boundary tests skip domain edges** (margin offsets like eta += 0.01)
6. **Reference values not independently derived** -- may perpetuate errors from legacy code

---

## End-to-End Verification Results (R1=1, R2=3, eps=0.5, 200x80, GAMMA=1.0)

| Test | Result | Detail |
|------|--------|--------|
| Boundary: j=0 is outer (R2=3.0) | **PASS** | radius=3.000016, center=(0, 0) |
| Boundary: j=-1 is inner (R1=1.0) | **PASS** | radius=1.000003, center=(0, -0.5) |
| GILBM stability omega < 2.0 | **PASS** | omega=1.2918, status=OK |
| Scale factor h = c/(cosh-cos) | **PASS** | h matches formula exactly (0 error) |
| Periodicity xi=-pi == xi=pi | **PASS** | max error = 1.08e-15 |
| Conformal orthogonality < 1 deg | **FAIL** | 1.34 deg (FD artifact, not true non-orthogonality) |
| Jacobian det = h^2 | **FAIL** | 77% relative error (**CRITICAL-1: wrong h formula**) |

---

## Priority Fix Order

1. **CRITICAL-1**: Fix `compute_scale_factor` to use `cos(xi) + cosh(eta)` (5 min fix, cascading impact)
2. **CRITICAL-2**: Implement proper LBM interpolation weights with contravariant velocity transform
3. **CRITICAL-3**: Add pole singularity protection in bipolar_transform.py
4. **CRITICAL-4**: Harmonize config defaults with interactive defaults
5. **MAJOR-1**: Fix numerical Jacobian stencil (use stride-1 central difference)
6. **MAJOR-2**: Correct and complete Christoffel symbols
7. **MAJOR-3**: Re-calibrate estimate_max_resolution() k-factors
8. Add test: `assert abs(detJ - h_correct**2) < 1e-10` for conformal verification
9. Tighten test tolerances to 1e-10 where conformal properties should hold exactly
