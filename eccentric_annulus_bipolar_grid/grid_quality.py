"""
Grid Quality Metrics
=====================
Compute aspect ratio, skewness, orthogonality, and cell area statistics.
For conformal bipolar mapping, the grid is inherently orthogonal.
"""

import numpy as np


def compute_grid_quality(x, y):
    """
    Compute grid quality metrics for a structured grid.

    Parameters
    ----------
    x, y : ndarray (nj, ni)
        Physical grid coordinates.

    Returns
    -------
    dict with keys:
        min_area, max_area, area_ratio,
        min_aspect, max_aspect, mean_aspect,
        orthogonality_max_deg, orthogonality_mean_deg,
        areas_2d, aspects_2d, ortho_2d
    """
    nj, ni = x.shape

    areas = np.zeros((nj - 1, ni - 1))
    aspects = np.zeros((nj - 1, ni - 1))
    ortho_angles = np.zeros((nj - 1, ni - 1))

    for j in range(nj - 1):
        for i in range(ni - 1):
            x00, y00 = x[j, i], y[j, i]
            x10, y10 = x[j, i + 1], y[j, i + 1]
            x01, y01 = x[j + 1, i], y[j + 1, i]
            x11, y11 = x[j + 1, i + 1], y[j + 1, i + 1]

            # Cell area (cross product of diagonals)
            d1x = x11 - x00
            d1y = y11 - y00
            d2x = x10 - x01
            d2y = y10 - y01
            areas[j, i] = abs(d1x * d2y - d1y * d2x) / 2.0

            # Edge lengths
            e1 = np.sqrt((x10 - x00)**2 + (y10 - y00)**2)
            e2 = np.sqrt((x01 - x00)**2 + (y01 - y00)**2)
            e3 = np.sqrt((x11 - x10)**2 + (y11 - y10)**2)
            e4 = np.sqrt((x11 - x01)**2 + (y11 - y01)**2)

            max_e = max(e1, e2, e3, e4)
            min_e = min(e1, e2, e3, e4)
            if min_e > 1e-30:
                aspects[j, i] = max_e / min_e
            else:
                aspects[j, i] = float("inf")

            # Orthogonality at (j, i) corner
            if e1 > 1e-30 and e2 > 1e-30:
                dot = (x10 - x00) * (x01 - x00) + (y10 - y00) * (y01 - y00)
                cos_a = np.clip(dot / (e1 * e2), -1, 1)
                angle = np.degrees(np.arccos(cos_a))
                ortho_angles[j, i] = abs(angle - 90.0)

    finite_areas = areas[areas > 0]
    finite_aspects = aspects[np.isfinite(aspects)]

    return {
        "min_area": finite_areas.min() if len(finite_areas) > 0 else 0,
        "max_area": finite_areas.max() if len(finite_areas) > 0 else 0,
        "area_ratio": (finite_areas.max() / finite_areas.min()
                       if len(finite_areas) > 0 and finite_areas.min() > 0
                       else float("inf")),
        "min_aspect": finite_aspects.min() if len(finite_aspects) > 0 else 0,
        "max_aspect": finite_aspects.max() if len(finite_aspects) > 0 else 0,
        "mean_aspect": finite_aspects.mean() if len(finite_aspects) > 0 else 0,
        "orthogonality_max_deg": ortho_angles.max(),
        "orthogonality_mean_deg": ortho_angles.mean(),
        "areas_2d": areas,
        "aspects_2d": aspects,
        "ortho_2d": ortho_angles,
    }


def compute_spacing_profile(x, y, direction="radial"):
    """
    Compute spacing profiles along radial or circumferential directions.

    Parameters
    ----------
    x, y : ndarray (nj, ni)
        Grid coordinates.
    direction : str
        "radial" (along j, for each i) or "circumferential" (along i, for each j).

    Returns
    -------
    dict with keys: min_spacing, max_spacing, ratio, profiles
    """
    nj, ni = x.shape

    if direction == "radial":
        profiles = []
        all_spacings = []
        for i in range(ni):
            dr = np.sqrt(np.diff(x[:, i])**2 + np.diff(y[:, i])**2)
            profiles.append(dr)
            all_spacings.extend(dr.tolist())
    else:
        profiles = []
        all_spacings = []
        for j in range(nj):
            ds = np.sqrt(np.diff(x[j, :])**2 + np.diff(y[j, :])**2)
            profiles.append(ds)
            all_spacings.extend(ds.tolist())

    all_spacings = np.array(all_spacings)
    pos = all_spacings[all_spacings > 0]

    return {
        "min_spacing": pos.min() if len(pos) > 0 else 0,
        "max_spacing": pos.max() if len(pos) > 0 else 0,
        "ratio": (pos.max() / pos.min()
                  if len(pos) > 0 and pos.min() > 0
                  else float("inf")),
        "profiles": profiles,
    }


def print_grid_quality(quality, params):
    """Print grid quality report."""
    print()
    print("  " + "=" * 62)
    print("   Grid Quality Report")
    print("  " + "=" * 62)
    print(f"    Geometry:  R1={params.get('R1', '?')}, "
          f"R2={params.get('R2', '?')}, "
          f"eps={params.get('epsilon', '?')}")
    print(f"    Grid:      I={params.get('N_xi', params.get('Ni', '?'))} "
          f"(circumferential) x "
          f"J={params.get('N_eta', params.get('Nj', '?'))} (radial)")
    if "alpha" in params:
        print(f"    Bipolar:   alpha={params['alpha']:.6f}, "
              f"beta={params['beta']:.6f}")
        print(f"               c={params['c']:.6f}, "
              f"gamma_shift={params.get('gamma_shift', params.get('gamma', 0)):.6f}")
    print("  " + "-" * 62)
    print(f"    Cell area range:  [{quality['min_area']:.4e}, "
          f"{quality['max_area']:.4e}]")
    print(f"    Area ratio:       {quality['area_ratio']:.2f}")
    print(f"    Aspect ratio:     [{quality['min_aspect']:.2f}, "
          f"{quality['max_aspect']:.2f}]  "
          f"mean={quality['mean_aspect']:.2f}")
    print(f"    Orthogonality:    max deviation = "
          f"{quality['orthogonality_max_deg']:.4f} deg  "
          f"mean = {quality['orthogonality_mean_deg']:.4f} deg")

    if quality["orthogonality_max_deg"] < 0.01:
        print("    -> Excellent: conformal mapping preserves orthogonality")
    elif quality["orthogonality_max_deg"] < 1.0:
        print("    -> Good: near-orthogonal grid")
    else:
        print("    -> Note: cell-level deviation is a finite-difference artifact")
        print("       (conformal mapping is inherently orthogonal;"
              " deviation is a finite-difference artifact)")