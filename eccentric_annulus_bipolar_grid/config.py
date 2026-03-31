"""
Configuration File Parser and Validator
=========================================
Supports simple key=value config files and command-line arguments.
"""

from pathlib import Path


def parse_config(filepath):
    """
    Parse a simple key=value config file.
    Lines starting with # are comments. Blank lines ignored.

    Expected keys:
        R1, R2, ECCENTRICITY (or EPSILON),
        N_XI (or NI), N_ETA (or NJ),
        STRETCH_XI, STRETCH_ETA, ALPHA_S,
        OUTPUT_DIR, PREFIX,
        UREF, RE, CFL
    """
    config = {}
    filepath = Path(filepath)
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, val = line.split("=", 1)
                key = key.strip().upper()
                val = val.strip()
                # Strip inline comments
                if "#" in val:
                    val = val[:val.index("#")].strip()
                try:
                    if "." in val or "e" in val.lower():
                        config[key] = float(val)
                    else:
                        config[key] = int(val)
                except ValueError:
                    config[key] = val
    return config


def write_default_config(filepath):
    """Write a default config file."""
    filepath = Path(filepath)
    content = """# Eccentric Annulus Bipolar Grid Configuration
# ==============================================

# Geometry
R1 = 1.0            # Inner radius
R2 = 3.0            # Outer radius
ECCENTRICITY = 0.5  # Physical eccentricity distance (0 < e < R2-R1)

# Grid resolution
N_XI = 200           # Circumferential (xi) grid points
N_ETA = 80           # Radial (eta) grid points

# Stretching (Vinokur tanh, 0 = uniform)
STRETCH_XI = 0.0     # Circumferential stretching GAMMA
STRETCH_ETA = 1.0    # Radial stretching GAMMA (recommended: 0~1.0)
ALPHA_S = 0.5        # Stretching symmetry (0.5 = symmetric)

# Output
OUTPUT_DIR = .       # Output directory
PREFIX = eccentric_annulus  # Filename prefix

# LBM parameters (for stability estimation)
UREF = 0.05          # Reference velocity
RE = 100             # Reynolds number
CFL = 0.5            # CFL number
"""
    with open(filepath, "w") as f:
        f.write(content)
    print(f"  [written] Default config: {filepath}")
    return filepath


def validate_config(config):
    """
    Validate parsed config values and return normalised parameters.

    Returns
    -------
    dict with normalised parameters.

    Raises
    ------
    ValueError for invalid parameters.
    """
    r1 = config.get("R1", 1.0)
    r2 = config.get("R2", 3.0)

    if r1 <= 0:
        raise ValueError(f"R1 must be > 0, got {r1}")
    if r2 <= r1:
        raise ValueError(f"R2 must be > R1, got R2={r2}, R1={r1}")

    # Eccentricity: accept either physical distance or relative
    if "ECCENTRICITY" in config:
        e = config["ECCENTRICITY"]
        if e <= 0 or e >= r2 - r1:
            raise ValueError(
                f"ECCENTRICITY must be in (0, {r2 - r1}), got {e}")
    elif "EPSILON" in config:
        eps = config["EPSILON"]
        if eps <= 0 or eps >= 1:
            raise ValueError(f"EPSILON must be in (0, 1), got {eps}")
        e = eps * (r2 - r1)
    else:
        e = 0.5  # default

    n_xi = config.get("N_XI", config.get("NI", 200))
    n_eta = config.get("N_ETA", config.get("NJ", 80))

    if n_xi < 10:
        raise ValueError(f"N_XI must be >= 10, got {n_xi}")
    if n_eta < 5:
        raise ValueError(f"N_ETA must be >= 5, got {n_eta}")

    return {
        "R1": r1,
        "R2": r2,
        "eccentricity": e,
        "epsilon": e / (r2 - r1),
        "N_xi": int(n_xi),
        "N_eta": int(n_eta),
        "stretch_xi": config.get("STRETCH_XI", 0.0),
        "stretch_eta": config.get("STRETCH_ETA", 0.0),
        "alpha_s": config.get("ALPHA_S", 0.5),
        "output_dir": config.get("OUTPUT_DIR", "."),
        "prefix": config.get("PREFIX", "eccentric_annulus"),
        "Uref": config.get("UREF", 0.05),
        "Re": config.get("RE", 100),
        "CFL": config.get("CFL", 0.5),
    }
