"""
Grid I/O: Tecplot .dat, CSV, HDF5 Export/Import
=================================================
Output formats compatible with grid_zeta_tool.py and LBM solver.
"""

import numpy as np
from pathlib import Path


def write_tecplot_dat(filepath, x, y, title="Eccentric annulus grid",
                      zone_title="EccentricAnnulus", extra_fields=None):
    """
    Write grid to Tecplot .dat file.

    Format is identical to grid_zeta_tool.py output:
      TITLE = "..."
      VARIABLES = "x corner"
      "y corner"
      [optional extra variables]
      ZONE T="..."
       I=Ni, J=Nj, K=1, F=POINT
      DT=(SINGLE SINGLE ...)
       x1 y1 [extra1 ...]
       x2 y2 [extra2 ...]

    Parameters
    ----------
    filepath : str or Path
        Output file path.
    x, y : ndarray (Nj, Ni)
        Grid coordinates.
    title : str
        Tecplot title.
    zone_title : str
        Zone title.
    extra_fields : dict or None
        Additional fields to write, e.g. {"h": scale_factor, "detJ": jacobian}.
    """
    nj, ni = x.shape
    filepath = Path(filepath)

    var_names = ['"x corner"', '"y corner"']
    if extra_fields:
        for name in extra_fields:
            var_names.append(f'"{name}"')

    n_vars = len(var_names)

    with open(filepath, "w") as f:
        f.write(f'TITLE     = "{title}"\n')
        f.write(f'VARIABLES = {var_names[0]}\n')
        for vn in var_names[1:]:
            f.write(f'{vn}\n')
        f.write(f'ZONE T="{zone_title}"\n')
        f.write(f' I={ni}, J={nj}, K=1,F=POINT\n')
        dt_str = " ".join(["SINGLE"] * n_vars)
        f.write(f'DT=({dt_str})\n')
        for j in range(nj):
            for i in range(ni):
                line = f" {x[j, i]: .9E} {y[j, i]: .9E}"
                if extra_fields:
                    for name in extra_fields:
                        line += f" {extra_fields[name][j, i]: .9E}"
                f.write(line + "\n")

    print(f"  [written] {filepath}")


def write_csv(filepath, x, y, xi=None, eta=None, extra_fields=None):
    """
    Write grid as CSV with columns: i, j, x, y [, xi, eta, extra...].

    Parameters
    ----------
    filepath : str or Path
        Output file path.
    x, y : ndarray (Nj, Ni)
        Grid coordinates.
    xi, eta : ndarray or None
        Computational coordinates.
    extra_fields : dict or None
        Additional fields.
    """
    nj, ni = x.shape
    filepath = Path(filepath)

    headers = ["i", "j", "x", "y"]
    if xi is not None:
        headers.extend(["xi", "eta"])
    if extra_fields:
        headers.extend(extra_fields.keys())

    with open(filepath, "w") as f:
        f.write(",".join(headers) + "\n")
        for j in range(nj):
            for i in range(ni):
                vals = [str(i), str(j),
                        f"{x[j, i]:.9E}", f"{y[j, i]:.9E}"]
                if xi is not None:
                    vals.extend([f"{xi[j, i]:.9E}", f"{eta[j, i]:.9E}"])
                if extra_fields:
                    for name in extra_fields:
                        vals.append(f"{extra_fields[name][j, i]:.9E}")
                f.write(",".join(vals) + "\n")

    print(f"  [written] {filepath}")


def write_hdf5(filepath, x, y, metrics=None, params=None):
    """
    Write grid to HDF5 format for large grids and LBM solver input.

    Parameters
    ----------
    filepath : str or Path
        Output file path.
    x, y : ndarray (Nj, Ni)
        Grid coordinates.
    metrics : dict or None
        Grid metrics (h, detJ, J11, etc.).
    params : dict or None
        Grid parameters.
    """
    try:
        import h5py
    except ImportError:
        print("  [skip] h5py not installed; HDF5 export unavailable.")
        return

    filepath = Path(filepath)
    with h5py.File(filepath, "w") as f:
        grp = f.create_group("grid")
        grp.create_dataset("x", data=x, compression="gzip")
        grp.create_dataset("y", data=y, compression="gzip")

        if metrics:
            mgrp = f.create_group("metrics")
            for key, val in metrics.items():
                if isinstance(val, np.ndarray):
                    mgrp.create_dataset(key, data=val, compression="gzip")

        if params:
            pgrp = f.create_group("params")
            for key, val in params.items():
                if isinstance(val, (int, float, str, bool)):
                    pgrp.attrs[key] = val
                elif isinstance(val, dict):
                    subgrp = pgrp.create_group(key)
                    for k2, v2 in val.items():
                        if isinstance(v2, (int, float, str, bool)):
                            subgrp.attrs[k2] = v2

    print(f"  [written] {filepath}")


def read_tecplot_dat(filepath):
    """
    Parse a Tecplot .dat grid file.

    Parameters
    ----------
    filepath : str or Path
        Input file path.

    Returns
    -------
    x, y : ndarray (nj, ni)
        Grid coordinates.
    ni, nj : int
        Grid dimensions.
    """
    filepath = Path(filepath)
    with open(filepath, "r", encoding="latin-1") as f:
        lines = f.readlines()

    ni = nj = None
    header_lines = 0
    for idx, line in enumerate(lines):
        if "I=" in line.upper():
            parts = line.replace(",", " ").replace("=", " ").upper().split()
            for k, tok in enumerate(parts):
                if tok == "I":
                    ni = int(parts[k + 1])
                if tok == "J":
                    nj = int(parts[k + 1])
            header_lines = idx + 2
            break

    if ni is None or nj is None:
        raise ValueError("Cannot find I/J dimensions in header")

    data_lines = lines[header_lines:]
    x_flat, y_flat = [], []
    for dl in data_lines:
        dl = dl.strip()
        if not dl:
            continue
        vals = dl.split()
        if len(vals) >= 2:
            x_flat.append(float(vals[0]))
            y_flat.append(float(vals[1]))

    expected = ni * nj
    if len(x_flat) != expected:
        raise ValueError(
            f"Expected {expected} points (I={ni} x J={nj}), got {len(x_flat)}"
        )

    x = np.array(x_flat).reshape(nj, ni)
    y = np.array(y_flat).reshape(nj, ni)
    return x, y, ni, nj
