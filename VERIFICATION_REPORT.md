# Eccentric Annulus Bipolar Grid — Verification Report

**Date:** 2026-03-31
**Verifier:** Claude (independent verification of Claude Code's fix report)
**Verdict:** Claude Code's claim of "43/43 tests pass" was **FALSE** at the time of verification. After targeted manual fixes, all 43 tests now pass.

---

## Critical Finding: Stale `.pyc` Bytecode Cache

The root cause of ALL discrepancies was a **stale Python bytecode cache** (`.pyc` files in `__pycache__/`).

Claude Code's edit operations (applying CRITICAL/MAJOR/MINOR fixes) **truncated 6 source files** mid-edit. However, because the old `.pyc` bytecode cache had timestamps NEWER than the `.py` source files (`.py` at 22:37, `.pyc` at 22:59), Python loaded the cached bytecode instead of recompiling from source. This created a deceptive situation:

- Source `.py` files: contained partial fixes but were **truncated** (syntax errors)
- Cached `.pyc` files: contained the **old pre-fix code** (no syntax errors, but with the original bugs)
- Python loaded `.pyc` → tests appeared to work (35/43 pass, with 8 failures from the unfixed bugs)
- Claude Code likely tested in a fresh environment or different Python version where `.pyc` cache didn't exist, hence "43/43 pass"

### Truncated Files Found

| File | Truncation Point | What Was Missing |
|------|-----------------|------------------|
| `grid_metrics.py` | Line 271 (mid-comment) | Complete Christoffel symbols computation + return dict |
| `bipolar_grid_generator.py` | Line 411 (open dict) | Closing of `_build_params()` return dict |
| `bipolar_constants.py` | Line 224 (mid-comment) | `cross_validate_constants()` return statement |
| `bipolar_transform.py` | Line 164 (dangling space) | `verify_roundtrip()` completion + return dict |
| `config.py` | Line 135 (mid-variable) | Parameter validation + return dict |
| `grid_quality.py` | Line 167 (mid-string) | `print_grid_quality()` closing string |

## Fixes Applied

### 1. Scale Factor Formula (CRITICAL-1) — Already Fixed in Source
The source code at `grid_metrics.py:43` correctly shows:
```python
denom = np.cos(xi) + np.cosh(eta)  # PLUS for tan mapping
```
But the `.pyc` cache still had the old formula `np.cosh(eta) - np.cos(xi)` (MINUS, for cot mapping).

**Resolution:** Touching the `.py` file timestamp forced Python to recompile from the correct source.

### 2. Christoffel Symbols (grid_metrics.py)
Completed the truncated function with all 6 Christoffel symbols for conformal coordinates:
- `Gamma_1_11 = (1/h) * dh/dxi = sin(xi) / D`
- `Gamma_1_22 = -(1/h) * dh/dxi`
- `Gamma_1_12 = (1/h) * dh/deta = -sinh(eta) / D`
- `Gamma_2_11 = -(1/h) * dh/deta`
- `Gamma_2_22 = (1/h) * dh/deta`
- `Gamma_2_12 = (1/h) * dh/dxi`

where `D = cos(xi) + cosh(eta)`.

### 3. cross_validate_constants Return (bipolar_constants.py)
Added proper return dict with unified Snyder-convention keys + validation sub-dict.

### 4. verify_roundtrip Completion (bipolar_transform.py)
Completed the function to return `{max_err_xi, max_err_eta, passed}`.

### 5. Config Validation (config.py)
Completed parameter bounds validation and return dict.

### 6. Grid Quality Report (grid_quality.py)
Completed truncated print string.

## End-to-End Verification Results (R1=1, R2=3, eps=0.5, 200×80)

| Test | Result | Value |
|------|--------|-------|
| detJ == h² (conformal) | **PASS** | max rel err = 1.60e-15 |
| g11 == h² | **PASS** | max rel err = 1.60e-15 |
| g22 == h² | **PASS** | max rel err = 1.60e-15 |
| g12 == 0 (orthogonality) | **PASS** | max = 0.00e+00 |
| h > 0 everywhere | **PASS** | min(h) = 7.51e-01 |
| Christoffel Gamma_1_11 | **PASS** | max rel err = 3.88e-16 |
| Unit tests | **PASS** | 43/43 pass, 1.32s |

## Remaining Warnings (Non-Critical)

The `gamma_shift` cross-validation warning fires on every call because the PDF formulation doesn't compute `gamma_shift` (returns 0) while the Snyder formulation does. This is expected and harmless — the Snyder value is used.

## Recommendation

**Delete all `.pyc` cache files** on the user's machine to ensure Python always compiles from the corrected source:
```bash
find eccentric_annulus_bipolar_grid -name "__pycache__" -exec rm -rf {} +
```
