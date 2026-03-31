"""
Eccentric Annulus Bipolar Grid Generator -- CLI Entry Point
=============================================================
Three modes:
  1. Interactive:  python -m eccentric_annulus_bipolar_grid
  2. Auto:         python -m eccentric_annulus_bipolar_grid --auto config.cfg
  3. Quick demo:   python -m eccentric_annulus_bipolar_grid --quick

Mirrors the design of grid_zeta_tool.py interactive workflow, with
eccentric-annulus-specific GILBM stability tables and warnings.
"""

import sys
import numpy as np
from pathlib import Path

from .bipolar_grid_generator import BipolarGridGenerator
from .bipolar_constants import cross_validate_constants
from .grid_stretching import (
    print_gilbm_stability_table_eccentric,
    check_eccentricity_stability,
    estimate_max_resolution,
)
from .grid_visualization import (
    plot_eccentricity_sweep,
    plot_gamma_sweep,
)
from .config import parse_config, write_default_config, validate_config


# ============================================================
#  Interactive Helpers
# ============================================================

def ask_float(prompt, default, lo=None, hi=None):
    while True:
        raw = input(f"  {prompt} [default={default}]: ").strip()
        if raw == "":
            return default
        try:
            val = float(raw)
        except ValueError:
            print("    ** Invalid number, try again.")
            continue
        if lo is not None and val < lo:
            print(f"    ** Must be >= {lo}, try again.")
            continue
        if hi is not None and val > hi:
            print(f"    ** Must be <= {hi}, try again.")
            continue
        return val


def ask_int(prompt, default, lo=None, hi=None):
    while True:
        raw = input(f"  {prompt} [default={default}]: ").strip()
        if raw == "":
            return default
        try:
            val = int(raw)
        except ValueError:
            print("    ** Invalid integer, try again.")
            continue
        if lo is not None and val < lo:
            print(f"    ** Must be >= {lo}, try again.")
            continue
        if hi is not None and val > hi:
            print(f"    ** Must be <= {hi}, try again.")
            continue
        return val


def ask_yes_no(prompt, default_yes=True):
    hint = "Y/n" if default_yes else "y/N"
    raw = input(f"  {prompt} [{hint}]: ").strip().lower()
    if raw == "":
        return default_yes
    return raw in ("y", "yes")


# ============================================================
#  Quick Demo
# ============================================================

def run_quick_demo():
    """Quick demo with default parameters."""
    print()
    print("=" * 62)
    print("  Eccentric Annulus Bipolar Grid -- Quick Demo")
    print("=" * 62)
    print()

    gen = BipolarGridGenerator(r1=1.0, r2=3.0, eccentricity=0.5)
    gen.set_resolution(N_xi=200, N_eta=80)
    gen.set_stretching(stretch_eta=1.0)

    print("  [1/5] Computing bipolar constants ...")
    print(f"         alpha = {gen.alpha:.8f}")
    print(f"         beta  = {gen.beta:.8f}")
    print(f"         c     = {gen.c:.8f}")
    print(f"         gamma_shift = {gen.gamma_shift:.8f}")
    print()

    print("  [2/5] Generating grid ...")
    result = gen.generate()
    print(f"         Grid shape: ({result['x'].shape[0]}, {result['x'].shape[1]})")
    print()

    print("  [3/5] Computing quality ...")
    gen.quality_report()

    print("  [4/5] GILBM stability check ...")
    gen.stability_report()

    script_dir = Path(__file__).resolve().parent
    tag = f"I{gen.N_xi}_J{gen.N_eta}_e{gen.epsilon:.2f}"

    print("  [5/5] Exporting files ...")
    gen.export(script_dir / f"demo_{tag}.dat", format="tecplot")
    gen.export(script_dir / f"demo_{tag}.csv", format="csv")
    gen.plot(output_dir=script_dir, prefix="demo")

    print()
    print("=" * 62)
    print("  QUICK DEMO COMPLETE")
    print("=" * 62)


# ============================================================
#  Auto Mode
# ============================================================

def run_auto_mode(config_path=None):
    """Run from config file."""
    script_dir = Path(__file__).resolve().parent

    if config_path is None:
        candidates = [
            script_dir / "eccentric_annulus.cfg",
            Path.cwd() / "eccentric_annulus.cfg",
        ]
        for c in candidates:
            if c.exists():
                config_path = c
                break
        if config_path is None:
            print("  Config file not found. Creating default ...")
            config_path = write_default_config(
                script_dir / "eccentric_annulus.cfg")
    else:
        config_path = Path(config_path)

    print()
    print("=" * 62)
    print("  Eccentric Annulus Bipolar Grid -- AUTO MODE")
    print(f"  Config: {config_path}")
    print("=" * 62)

    cfg = parse_config(config_path)
    params = validate_config(cfg)

    gen = BipolarGridGenerator(
        r1=params["R1"], r2=params["R2"],
        eccentricity=params["eccentricity"])
    gen.set_resolution(N_xi=params["N_xi"], N_eta=params["N_eta"])
    gen.set_stretching(
        stretch_xi=params["stretch_xi"],
        stretch_eta=params["stretch_eta"],
        alpha_s=params["alpha_s"])

    print()
    print(f"  R1 = {params['R1']},  R2 = {params['R2']},  "
          f"e = {params['eccentricity']}")
    print(f"  Grid: I={params['N_xi']} x J={params['N_eta']}")
    print(f"  GAMMA_eta = {params['stretch_eta']}, ALPHA = {params['alpha_s']}")
    print()

    result = gen.generate()
    gen.quality_report()
    gen.stability_report()

    out_dir = Path(params["output_dir"])
    prefix = params["prefix"]
    tag = f"I{params['N_xi']}_J{params['N_eta']}_e{params['epsilon']:.2f}"

    gen.export(out_dir / f"{prefix}_{tag}.dat", format="tecplot")
    gen.export(out_dir / f"{prefix}_{tag}.csv", format="csv")
    gen.plot(output_dir=out_dir, prefix=prefix)

    print()
    print("=" * 62)
    print("  AUTO MODE COMPLETE")
    print("=" * 62)


# ============================================================
#  Interactive Mode
# ============================================================

def run_interactive():
    """Full interactive workflow with GILBM stability guidance."""
    print()
    print("=" * 62)
    print("  Eccentric Annulus Bipolar Grid Generator")
    print("  (Conformal mapping with bipolar coordinates)")
    print("  Designed for GILBM solver integration")
    print("=" * 62)

    # ---- Step 1: Geometry ----
    print()
    print("-" * 62)
    print("  [Step 1] Define eccentric annulus geometry")
    print("-" * 62)
    print()
    print("  R1 -- Inner cylinder radius")
    R1 = ask_float("R1", default=1.0, lo=0.01)
    print()
    print("  R2 -- Outer cylinder radius (must be > R1)")
    R2 = ask_float("R2", default=3.0, lo=R1 + 0.01)
    print()
    gap = R2 - R1
    print(f"  e -- Physical eccentricity distance (0 < e < {gap:.4f})")
    e = ask_float("e", default=0.5 * gap, lo=0.001, hi=gap * 0.999)
    epsilon = e / gap

    # Display geometry info
    consts = cross_validate_constants(R1, R2, e)
    print()
    print(f"  Non-dimensional parameters:")
    print(f"    gamma = R1/R2 = {consts['gamma_r']:.6f}")
    print(f"    phi   = e/(R2-R1) = {consts['phi']:.6f}")
    print(f"    kappa = R2/R1 = {consts['kappa']:.6f}")
    print()
    print(f"  Bipolar constants:")
    print(f"    alpha       = {consts['alpha']:.8f}  (eta_min, outer wall)")
    print(f"    beta        = {consts['beta']:.8f}  (eta_max, inner wall)")
    print(f"    c           = {consts['c']:.8f}  (pole distance)")
    print(f"    gamma_shift = {consts['gamma_shift']:.8f}  (vertical offset)")
    print()
    print(f"  Physical gap information:")
    print(f"    gap_min = R2 - R1 - e = {R2 - R1 - e:.4f}  (narrow gap)")
    print(f"    gap_max = R2 - R1 + e = {R2 - R1 + e:.4f}  (wide gap)")
    print(f"    gap_ratio = {(R2 - R1 + e) / (R2 - R1 - e):.2f}")

    # Eccentricity safety check
    ecc_check = check_eccentricity_stability(epsilon, 200, 80, 1.0)
    print()
    print(f"  Eccentricity safety: [{ecc_check['risk_level']}]")
    print(f"    {ecc_check['message']}")

    if consts["validation"]["passed"]:
        print(f"  Cross-validation: PASSED (PDF vs Snyder diff < 1e-10)")
    else:
        print(f"  Cross-validation: FAILED -- check constants!")

    # ---- Step 2: Resolution ----
    print()
    print("-" * 62)
    print("  [Step 2] Set grid resolution")
    print("-" * 62)
    print()
    print("  N_xi -- Circumferential grid points (xi direction)")
    N_xi = ask_int("N_xi", default=200, lo=10, hi=10000)
    print()
    print("  N_eta -- Radial grid points (eta direction)")
    N_eta = ask_int("N_eta", default=80, lo=5, hi=10000)

    total = N_xi * N_eta
    mem_MB = total * 8 * 10 / 1e6  # ~10 arrays of float64
    print()
    print(f"  Total nodes: {N_xi} x {N_eta} = {total:,}")
    print(f"  Estimated memory: ~{mem_MB:.1f} MB")

    # Resolution stability note
    res_est = estimate_max_resolution(R1, R2, epsilon, gamma_stretch=1.0)
    print(f"  At GAMMA=1.0, estimated max safe N_eta ~ {res_est['estimated_max_Neta']}")

    # ---- Step 3: Stretching ----
    print()
    print("-" * 62)
    print("  [Step 3] Grid stretching (GAMMA / ALPHA)")
    print("-" * 62)

    # Print stability table BEFORE GAMMA selection
    print_gilbm_stability_table_eccentric()

    print("  GAMMA_eta -- Radial stretching (Vinokur tanh)")
    print("    0.0       = UNIFORM (no wall clustering)")
    print("    0.5 ~ 1.0 = mild-moderate (RECOMMENDED for eccentric annulus)")
    print("    1.0 ~ 1.5 = moderate-strong (approaching stability limit)")
    print("    >= 2.0    = DANGEROUS (omega ~2.0, near divergence!)")
    print("    >= 2.5    = UNSTABLE (omega > 2.0, WILL DIVERGE)")
    GAMMA = ask_float("GAMMA_eta", default=1.0, lo=0.0, hi=10.0)

    # Blocking confirmation for known-unstable GAMMA
    if GAMMA >= 2.5:
        print()
        print("  !! DANGER: GAMMA >= 2.5 is KNOWN UNSTABLE (omega > 2.0) !!")
        print("  !! The LBM simulation WILL DIVERGE with this setting.    !!")
        confirm = ask_yes_no("Proceed anyway?", default_yes=False)
        if not confirm:
            print("  Reducing GAMMA to 1.0 (recommended).")
            GAMMA = 1.0
    elif GAMMA >= 2.0:
        print()
        print("  ** WARNING: GAMMA >= 2.0 is MARGINAL (omega ~2.0).")
        print("  ** Consider reducing to GAMMA <= 1.5 for safety.")

    print()
    print("  ALPHA -- Radial clustering symmetry")
    print("    0.5  = symmetric (inner + outer walls equal) (RECOMMENDED)")
    print("    <0.5 = outer wall (R2) denser")
    print("    >0.5 = inner wall (R1) denser")
    ALPHA_S = ask_float("ALPHA", default=0.5, lo=0.01, hi=0.99)

    print()
    print("  GAMMA_xi -- Circumferential stretching (usually 0 = uniform)")
    print("    Usually 0.0 because bipolar mapping auto-concentrates")
    print("    points in the narrow gap region.")
    GAMMA_XI = ask_float("GAMMA_xi", default=0.0, lo=0.0, hi=10.0)

    # ---- Step 4: Summary & Stability ----
    print()
    print("-" * 62)
    print("  [Step 4] Configuration summary")
    print("-" * 62)
    print(f"    R1 = {R1},  R2 = {R2},  e = {e}  (eps = {epsilon:.4f})")
    print(f"    Gap: {gap:.4f} (min={R2-R1-e:.4f}, max={R2-R1+e:.4f})")
    print(f"    Grid: N_xi={N_xi} x N_eta={N_eta} = {total:,} points")
    print(f"    GAMMA_eta = {GAMMA}, GAMMA_xi = {GAMMA_XI}, ALPHA = {ALPHA_S}")

    # Stability pre-check
    ecc_check2 = check_eccentricity_stability(epsilon, N_xi, N_eta, GAMMA)
    if ecc_check2["warning"]:
        print()
        print(f"  !! WARNING: {ecc_check2['warning']}")
    print()

    proceed = ask_yes_no("Generate grid?", default_yes=True)
    if not proceed:
        print("  Aborted.")
        return

    # ---- Step 5: Generation ----
    print()
    print("-" * 62)
    print("  [Step 5] Generating grid ...")
    print("-" * 62)

    gen = BipolarGridGenerator(r1=R1, r2=R2, eccentricity=e)
    gen.set_resolution(N_xi=N_xi, N_eta=N_eta)
    gen.set_stretching(stretch_xi=GAMMA_XI, stretch_eta=GAMMA, alpha_s=ALPHA_S)

    print("  [1/6] Computing bipolar constants ...")
    print("  [2/6] Generating grid with stretching ...")
    result = gen.generate()
    print(f"         Grid shape: ({result['x'].shape[0]}, {result['x'].shape[1]})")

    print("  [3/6] Computing grid metrics (Jacobian, scale factor) ...")
    print(f"         h range: [{result['metrics']['h'].min():.6f}, "
          f"{result['metrics']['h'].max():.6f}]")
    print(f"         |J| range: [{result['metrics']['detJ'].min():.6e}, "
          f"{result['metrics']['detJ'].max():.6e}]")
    print(f"         Orthogonality error: "
          f"{result['metrics']['orthogonality_error']:.2e}")

    print("  [4/6] Computing grid quality ...")
    gen.quality_report()

    print("  [5/6] GILBM stability check ...")
    gen.stability_report()

    # Post-generation stability gate
    if result["stability"]["status"] == "UNSTABLE":
        print("  !! POST-GENERATION CHECK: omega > 2.0 -- GRID IS UNSTABLE !!")
        print("  !! Reduce GAMMA or grid resolution before using in GILBM.  !!")
        abort = ask_yes_no("Export anyway (grid will diverge in LBM)?",
                           default_yes=False)
        if not abort:
            print("  Export aborted. Adjust GAMMA and re-run.")
            return

    # Export
    print("  [6/6] Exporting files ...")
    output_dir = Path.cwd()
    prefix = "eccentric_annulus"
    tag = f"I{N_xi}_J{N_eta}_e{epsilon:.2f}"

    gen.export(output_dir / f"{prefix}_{tag}.dat", format="tecplot")
    gen.export(output_dir / f"{prefix}_{tag}.csv", format="csv")
    gen.plot(output_dir=output_dir, prefix=prefix)

    print()
    print("  " + "=" * 62)
    print(f"   Output: {output_dir / f'{prefix}_{tag}.dat'}")
    print("  " + "=" * 62)

    # ---- Step 6: Optional sweeps ----
    print()
    do_sweep = ask_yes_no("Generate eccentricity sweep plot?", default_yes=False)
    if do_sweep:
        epsilons = [0.1, 0.2, 0.3, 0.5, 0.7]
        if epsilon < 0.9:
            epsilons.append(0.9)
        plot_eccentricity_sweep(
            R1, R2, epsilons, N_xi, N_eta,
            savepath=output_dir / f"sweep_eps_R1{R1}_R2{R2}.png",
            figsize=(10, None))

    do_gamma_sweep = ask_yes_no("Generate GAMMA sweep plot?", default_yes=False)
    if do_gamma_sweep:
        gammas = [0.0, 0.5, 1.0, 1.5, 2.0]
        x_grids, y_grids = [], []
        for g in gammas:
            gen_tmp = BipolarGridGenerator(r1=R1, r2=R2, eccentricity=e)
            gen_tmp.set_resolution(N_xi=N_xi, N_eta=N_eta)
            gen_tmp.set_stretching(stretch_eta=g, alpha_s=ALPHA_S)
            res = gen_tmp.generate()
            x_grids.append(res["x"])
            y_grids.append(res["y"])
        plot_gamma_sweep(
            x_grids, y_grids, gammas, gen._params,
            savepath=output_dir / f"sweep_gamma_e{epsilon:.2f}.png",
            figsize=(10, None))

    print()
    print("=" * 62)
    print("  ALL DONE")
    print("=" * 62)


# ============================================================
#  Main
# ============================================================

def main():
    args = sys.argv[1:]

    if "--quick" in args:
        run_quick_demo()
    elif "--auto" in args:
        idx = args.index("--auto")
        config_path = args[idx + 1] if idx + 1 < len(args) else None
        run_auto_mode(config_path)
    else:
        run_interactive()


if __name__ == "__main__":
    main()
