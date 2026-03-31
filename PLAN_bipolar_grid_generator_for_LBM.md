# Eccentric Annulus Grid Generator -- Implementation Plan for Claude Code

## Purpose

Build a **bipolar-coordinate-based structured grid generator** for eccentric annular domains,
designed to produce body-fitted curvilinear meshes that feed into a **general interpolation
Lattice Boltzmann Method (LBM)** solver (PeriodicHill simulation framework).

The mathematical foundation comes from:
- Snyder & Goldstein (1965), "Fully developed laminar flow in an eccentric annulus"
- PDF document `Eccentric_tansformation.pdf` (bipolar coordinate transformation)
- Existing reference implementation `eccentric_annulus_grid_tool.py`

---

## 1. Mathematical Framework (from PDF Document)

### 1.1 Bipolar Coordinate Definition

The bipolar coordinate system maps a complex plane:

```
z = x + iy = ic * cot( (xi + i*eta) / 2 )
```

where `(xi, eta)` are the bipolar coordinates and `c` is the pole distance.

### 1.2 Forward Transformation: (xi, eta) -> (x, y)

From PDF Equations (3a) and (3b):

```
x = c * sinh(eta) / (cosh(eta) - cos(xi))          ... (3a)
y = c * sin(xi)   / (cosh(eta) - cos(xi))           ... (3b)
```

### 1.3 Inverse Transformation: (x, y) -> (xi, eta)

From PDF Equations (3c) and (3d):

```
e^(2*eta) = [ y^2 + (x + c)^2 ] / [ y^2 + (x - c)^2 ]     ... (3c)
tan(xi)   = 2*y*c / (x^2 + y^2 - c^2)                       ... (3d)
```

### 1.4 Constant-eta Circles (Eq. 3e)

Lines of constant eta are circles in the physical plane:
```
y^2 + (x - c*coth(eta))^2 = c^2 / sinh^2(eta)
```
- Center at `(c*coth(eta), 0)`, radius = `c / sinh(eta)`

### 1.5 Geometry Constants (from PDF Eqs. 4a-4e)

Given user inputs `r1` (inner radius), `r2` (outer radius), `e` (eccentricity distance):

```
gamma = r1 / r2                                              ... (4d)
phi   = e / (r2 - r1)                                        ... (4e)

c = r1 * sinh(alpha) = r2 * sinh(beta)                       ... (4a)

cosh(alpha) = (1/gamma) * [ gamma*(1 + phi^2) + (1 - phi^2) ] / (2*phi)   ... (4b)
cosh(beta)  = [ gamma*(1 - phi^2) + (1 + phi^2) ] / (2*phi)               ... (4c)
```

**Derivation chain:**
1. User provides `r1, r2, e` (physical dimensions)
2. Compute `gamma = r1/r2` and `phi = e/(r2-r1)`
3. Compute `alpha = acosh(...)` from Eq.(4b)
4. Compute `beta  = acosh(...)` from Eq.(4c)
5. Compute `c = r1 * sinh(alpha)` from Eq.(4a)
6. The computational rectangle is `xi in [-pi, pi]` x `eta in [alpha, beta]`

### 1.6 Alternative Constants (existing code approach -- Snyder & Goldstein)

The existing `eccentric_annulus_grid_tool.py` uses a different but equivalent formulation:

```
shift = epsilon * (R2 - R1)        # physical offset = relative_eccentricity * gap
b     = |shift|
F     = (R2^2 - R1^2 + b^2) / (2*b)
M     = sqrt(F^2 - R2^2)           # M = c (pole distance)
alpha = 0.5 * ln((F + M) / (F - M))
beta  = 0.5 * ln((F - b + M) / (F - b - M))
gamma_shift = M * coth(alpha)       # vertical shift for centering
```

Both approaches are mathematically equivalent but **alpha/beta naming is SWAPPED**:

| | Snyder (existing code) | PDF (Eqs. 4a-4e) |
|---|---|---|
| Smaller value (~1.154 for test case) | Called `alpha` -> maps to **OUTER** circle R2 | Called `beta` -> maps to **OUTER** circle R2 |
| Larger value (~1.517 for test case) | Called `beta` -> maps to **INNER** circle R1 | Called `alpha` -> maps to **INNER** circle R1 |

**Key insight:** `c` is identical in both (diff < 1e-15). The computational domain is always
`eta in [smaller_value, larger_value]`, regardless of naming.

**Verified with R1=5, R2=7.6, epsilon=0.5:**
- `c = 10.8537`, `gamma_shift = 13.25`
- eta_min = 1.1542 -> outer circle (radius 7.6, centered at origin)
- eta_max = 1.5175 -> inner circle (radius 5.0, centered at (0, -1.3))

The new code should implement **BOTH** formulations and cross-validate. Use the Snyder
convention internally (alpha=eta_min, beta=eta_max) for consistency with existing code.

---

## 2. User Input Parameters Design

### 2.1 Primary Geometry Parameters (REQUIRED)

| Parameter | Symbol | Description | Constraint | Default |
|-----------|--------|-------------|------------|---------|
| `r1` | r_1 | Inner cylinder radius | r1 > 0 | 1.0 |
| `r2` | r_2 | Outer cylinder radius | r2 > r1 | 3.0 |
| `eccentricity` | e | Physical offset distance between centers | 0 < e < (r2 - r1) | 0.5 |

**Derived non-dimensional parameters (computed automatically, displayed to user):**
- `gamma = r1 / r2` (radius ratio)
- `phi = e / (r2 - r1)` (relative eccentricity, 0 < phi < 1)
- `kappa = r2 / r1` (inverse radius ratio, for LBM literature compatibility)

### 2.2 Grid Resolution Parameters (REQUIRED)

| Parameter | Symbol | Description | Constraint | Default |
|-----------|--------|-------------|------------|---------|
| `N_xi` | N_xi | Grid points in circumferential direction (xi) | >= 10 | 200 |
| `N_eta` | N_eta | Grid points in radial direction (eta) | >= 5 | 80 |

**Design notes for LBM:**
- `N_xi` should be large enough to resolve the narrow gap region (bottom of annulus when e > 0)
- `N_eta` controls radial resolution; for LBM interpolation, at least 10-20 points across the gap
- Total lattice nodes ~ N_xi * N_eta; typical LBM runs: 200x80 to 400x200

### 2.3 Grid Stretching Parameters (OPTIONAL)

| Parameter | Symbol | Description | Constraint | Default |
|-----------|--------|-------------|------------|---------|
| `stretch_eta` | s_eta | Radial stretching (Vinokur tanh) | >= 0 | 0.0 |
| `stretch_xi` | s_xi | Circumferential stretching | >= 0 | 0.0 |
| `stretch_symmetry` | alpha_s | Stretching symmetry factor | 0 < alpha_s < 1 | 0.5 |

**Purpose:** Cluster grid points near inner/outer walls for boundary layer resolution in LBM.

### 2.4 LBM-Specific Parameters (OPTIONAL, for future solver integration)

| Parameter | Symbol | Description | Constraint | Default |
|-----------|--------|-------------|------------|---------|
| `lattice_dx` | dx | Lattice spacing in physical units | > 0 | auto |
| `lattice_dt` | dt | Lattice time step | > 0 | auto |
| `Re` | Re | Reynolds number | > 0 | 100 |
| `output_format` | -- | Output file format | tecplot/csv/hdf5 | tecplot |

### 2.5 Output Control Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `output_dir` | Output directory path | current directory |
| `prefix` | Filename prefix | "eccentric_annulus" |
| `plot_grid` | Generate grid visualization PNG | True |
| `plot_quality` | Generate quality metrics plot | True |
| `export_metrics` | Export Jacobian, metric tensor fields | True |
| `half_domain` | Generate only upper half (symmetry) | False |

---

## 3. Implementation Plan -- File Structure

```
eccentric_annulus_bipolar_grid/
|
|-- bipolar_grid_generator.py        # MAIN: Grid generation engine
|-- bipolar_constants.py             # Bipolar coordinate constants computation
|-- bipolar_transform.py             # Forward/inverse coordinate transformations
|-- grid_metrics.py                  # Jacobian, metric tensors, scale factors
|-- grid_quality.py                  # Quality metrics (aspect ratio, skewness, orthogonality)
|-- grid_stretching.py               # Vinokur tanh stretching + custom clustering
|-- grid_io.py                       # Export: Tecplot .dat, CSV, HDF5
|-- grid_visualization.py            # Matplotlib plotting utilities
|-- lbm_grid_adapter.py             # Convert curvilinear grid to LBM lattice format
|-- config.py                        # Config file parser and validator
|-- main.py                          # CLI entry point with interactive/auto/quick modes
|-- tests/
|   |-- test_bipolar_constants.py    # Validate alpha, beta, c against known cases
|   |-- test_transforms.py          # Round-trip (xi,eta)->(x,y)->(xi,eta) test
|   |-- test_grid_quality.py        # Orthogonality of conformal mapping
|   |-- test_concentric_limit.py    # phi->0 should recover concentric annulus
|-- examples/
|   |-- example_default.cfg          # Default config file
|   |-- example_high_eccentricity.cfg
```

---

## 4. Implementation Steps (for Claude Code)

### Step 1: `bipolar_constants.py` -- Compute Bipolar System Constants

```python
def compute_bipolar_constants_pdf(r1, r2, e):
    """
    Compute bipolar constants using the PDF document formulation (Eqs. 4a-4e).

    Input: r1 (inner radius), r2 (outer radius), e (eccentricity distance)
    Output: alpha, beta, c, gamma, phi

    Algorithm:
      gamma = r1 / r2
      phi   = e / (r2 - r1)
      cosh(alpha) = (1/gamma) * [gamma*(1+phi^2) + (1-phi^2)] / (2*phi)
      cosh(beta)  = [gamma*(1-phi^2) + (1+phi^2)] / (2*phi)
      alpha = acosh(...)
      beta  = acosh(...)
      c     = r1 * sinh(alpha)
    """
    pass

def compute_bipolar_constants_snyder(R1, R2, epsilon):
    """
    Compute bipolar constants using the Snyder-Goldstein formulation
    (existing code approach).

    Input: R1, R2, epsilon (relative eccentricity, 0<eps<1)
    Output: alpha, beta, c, gamma_shift

    Algorithm:
      shift = epsilon * (R2 - R1)
      b = |shift|
      F = (R2^2 - R1^2 + b^2) / (2*b)
      M = sqrt(F^2 - R2^2)
      alpha = 0.5 * ln((F+M)/(F-M))
      beta  = 0.5 * ln((F-b+M)/(F-b-M))
      c = M
      gamma_shift = M * coth(alpha)
    """
    pass

def cross_validate_constants(r1, r2, e):
    """
    Compute constants from BOTH methods and verify they match
    within numerical tolerance. Raise warning if discrepancy > 1e-10.
    """
    pass
```

### Step 2: `bipolar_transform.py` -- Coordinate Transformations

```python
def bipolar_to_cartesian(xi, eta, c, gamma_shift):
    """
    Forward transformation: (xi, eta) -> (x, y)

    Method 1 (complex): z = ic * cot((xi + i*eta)/2)
      x = Re(z), y = Im(z) - gamma_shift

    Method 2 (explicit, from PDF Eq. 3a, 3b):
      x = c * sinh(eta) / (cosh(eta) - cos(xi))
      y = c * sin(xi)   / (cosh(eta) - cos(xi))
      Then shift: y = y - gamma_shift  (to center outer circle at origin)

    Both methods should be implemented; Method 1 is used for grid generation,
    Method 2 for verification.
    """
    pass

def cartesian_to_bipolar(x, y, c, gamma_shift):
    """
    Inverse transformation: (x, y) -> (xi, eta)

    From PDF Eq. (3c, 3d):
      eta = 0.5 * ln( [y'^2 + (x+c)^2] / [y'^2 + (x-c)^2] )
      xi  = atan2(2*y'*c, x^2 + y'^2 - c^2)
    where y' = y + gamma_shift

    Used for: mapping arbitrary physical points back to computational space,
    needed for LBM interpolation stencil construction.
    """
    pass
```

### Step 3: `grid_metrics.py` -- Jacobian and Metric Tensor

**Critical for LBM general interpolation:** the curvilinear grid metrics determine how
the LBM collision-streaming operates on the non-uniform mesh.

```python
def compute_jacobian(xi, eta, c, gamma_shift):
    """
    Compute the Jacobian matrix J and its determinant |J| at each grid point.

    J = | dx/dxi   dx/deta |
        | dy/dxi   dy/deta |

    For bipolar coordinates, the scale factor h is:
      h = c / (cosh(eta) - cos(xi))

    The Jacobian determinant |J| = h^2

    Returns: J11, J12, J21, J22, detJ (all as 2D arrays)
    """
    pass

def compute_metric_tensor(J11, J12, J21, J22):
    """
    Compute the covariant metric tensor g_ij = J^T * J

    g11 = J11^2 + J21^2   (= h^2 for conformal mapping)
    g12 = J11*J12 + J21*J22  (= 0 for conformal mapping, orthogonal)
    g22 = J12^2 + J22^2   (= h^2 for conformal mapping)

    For a conformal mapping, g12 should be zero (orthogonal grid).
    This serves as a quality check.
    """
    pass

def compute_scale_factor(xi, eta, c):
    """
    Bipolar coordinate scale factor (PDF implicit):
      h(xi, eta) = c / (cosh(eta) - cos(xi))

    This is the key metric for LBM: it relates lattice spacing
    to physical spacing at each grid point.
    """
    pass

def compute_christoffel_symbols(xi, eta, c):
    """
    Compute Christoffel symbols for the curvilinear coordinate system.
    Needed for the general interpolation LBM to correctly handle
    advection on the curvilinear grid.

    For bipolar (orthogonal, conformal):
      Gamma^xi_{xi,xi}   = -dh/dxi / h
      Gamma^xi_{eta,eta}  =  dh/dxi / h
      Gamma^eta_{xi,xi}   =  dh/deta / h
      Gamma^eta_{eta,eta} = -dh/deta / h
    """
    pass
```

### Step 4: `grid_stretching.py` -- Non-Uniform Point Distribution

```python
def vinokur_tanh_stretch(t, gamma, alpha_s=0.5):
    """
    Vinokur two-sided tanh clustering.
    t in [0, 1] -> stretched t in [0, 1]

    gamma = 0: uniform
    gamma > 0: cluster toward both ends (walls)
    alpha_s: symmetry (0.5 = symmetric, <0.5 = cluster toward inner wall)
    """
    pass

def wall_clustering_eta(N_eta, alpha, beta, stretch_param, cluster_inner=True):
    """
    Generate non-uniform eta distribution with wall clustering.

    For LBM: cluster points near inner wall (eta=alpha) where
    the gap is narrowest in eccentric configuration.
    """
    pass

def adaptive_xi_clustering(N_xi, eccentricity_ratio):
    """
    For high eccentricity, cluster xi points near xi=pi (narrow gap region)
    and xi=0 (wide gap region) to balance resolution.
    """
    pass
```

### Step 5: `bipolar_grid_generator.py` -- Main Grid Engine

```python
class BipolarGridGenerator:
    """
    Main grid generation class.

    Usage:
      gen = BipolarGridGenerator(r1=1.0, r2=3.0, eccentricity=0.5)
      gen.set_resolution(N_xi=200, N_eta=80)
      gen.set_stretching(stretch_eta=2.0)
      x, y, metrics = gen.generate()
      gen.export("output/grid.dat", format="tecplot")
      gen.plot("output/grid.png")
    """

    def __init__(self, r1, r2, eccentricity):
        """Validate inputs and compute bipolar constants."""
        pass

    def set_resolution(self, N_xi, N_eta):
        """Set grid resolution."""
        pass

    def set_stretching(self, stretch_xi=0.0, stretch_eta=0.0, alpha_s=0.5):
        """Configure grid stretching."""
        pass

    def generate(self):
        """
        Generate the complete grid.

        Steps:
        1. Create computational grid (xi, eta) with optional stretching
        2. Apply bipolar transformation to get (x, y)
        3. Compute Jacobian, metric tensor, scale factors
        4. Compute grid quality metrics
        5. Return grid + all metrics
        """
        pass

    def generate_half_domain(self):
        """Generate only upper half (y >= 0) using symmetry."""
        pass

    def export(self, filepath, format="tecplot"):
        """Export grid in specified format."""
        pass

    def export_lbm_lattice(self, filepath, dx=None):
        """
        Export grid data formatted for LBM solver input.
        Includes: node positions, scale factors, Jacobian,
        neighbor connectivity, boundary flags.
        """
        pass

    def plot(self, savepath=None):
        """Generate comprehensive visualization."""
        pass

    def quality_report(self):
        """Print and return grid quality metrics."""
        pass
```

### Step 6: `lbm_grid_adapter.py` -- LBM-Specific Grid Processing

```python
def create_lbm_lattice_info(x, y, scale_factor, jacobian, params):
    """
    Convert curvilinear grid to LBM lattice data structure.

    For general interpolation LBM on curvilinear grids:
    1. Each grid node becomes a lattice site
    2. The streaming step uses interpolation because neighbors
       in computational space don't align with D2Q9 lattice velocities
    3. Scale factor h determines local physical spacing

    Output: dictionary with
      - node_x, node_y: physical coordinates
      - node_xi, node_eta: computational coordinates
      - h: scale factor at each node
      - detJ: Jacobian determinant
      - boundary_type: 0=interior, 1=inner_wall, 2=outer_wall, 3=periodic
      - neighbors: interpolation stencil data for each direction
    """
    pass

def compute_interpolation_weights(xi_grid, eta_grid, c, D2Q9_directions):
    """
    For each lattice node and each D2Q9 velocity direction,
    compute the interpolation weights for the streaming step
    on the curvilinear grid.

    The D2Q9 velocities in computational space must account for
    the local metric (scale factor h).
    """
    pass

def identify_boundary_nodes(x, y, r1, r2, eccentricity, tol=1e-6):
    """
    Classify grid nodes:
    - Inner wall (eta = alpha, i.e., j=0 row)
    - Outer wall (eta = beta, i.e., j=N_eta-1 row)
    - Periodic boundaries (xi = -pi and xi = pi are the same line)
    - Interior nodes
    """
    pass
```

### Step 7: `grid_io.py` -- File I/O

```python
def write_tecplot_dat(filepath, x, y, **extra_fields):
    """Tecplot ASCII .dat format (compatible with existing tools)."""
    pass

def write_csv(filepath, x, y, **extra_fields):
    """CSV with columns: i, j, x, y, xi, eta, h, detJ, ..."""
    pass

def write_hdf5(filepath, x, y, metrics, params):
    """HDF5 for large grids and LBM solver input."""
    pass

def read_tecplot_dat(filepath):
    """Parse existing Tecplot grid files."""
    pass
```

### Step 8: `grid_visualization.py` -- Plotting

```python
def plot_physical_grid(x, y, params, savepath=None):
    """Plot the grid in physical (x, y) space with colored boundaries."""
    pass

def plot_computational_grid(xi, eta, savepath=None):
    """Plot the uniform computational rectangle."""
    pass

def plot_scale_factor_field(x, y, h, savepath=None):
    """Contour plot of the scale factor h(xi,eta) on physical domain."""
    pass

def plot_jacobian_field(x, y, detJ, savepath=None):
    """Contour plot of Jacobian determinant."""
    pass

def plot_grid_quality_map(x, y, quality_field, title, savepath=None):
    """Plot aspect ratio, skewness, or orthogonality deviation map."""
    pass

def plot_eccentricity_sweep(r1, r2, eccentricities, N_xi, N_eta, savepath=None):
    """Generate multi-panel plot showing grids at different eccentricities."""
    pass

def plot_radial_spacing_profile(x, y, params, savepath=None):
    """Plot dr vs j-index at several xi locations."""
    pass
```

### Step 9: `main.py` -- CLI Entry Point

Three modes:
1. **Interactive mode** (`python main.py`): step-by-step prompts
2. **Auto mode** (`python main.py --auto config.cfg`): read config file
3. **Quick demo** (`python main.py --quick`): default parameters

Interactive mode flow:
```
[Step 1] Geometry Definition
  - Input r1 (inner radius)           [default=1.0]
  - Input r2 (outer radius)           [default=3.0]
  - Input e  (eccentricity distance)  [default=0.5]
  -> Display: gamma, phi, alpha, beta, c (computed constants)
  -> Display: gap_min = r2 - r1 - e, gap_max = r2 - r1 + e

[Step 2] Grid Resolution
  - Input N_xi  (circumferential)     [default=200]
  - Input N_eta (radial)              [default=80]
  -> Display: total nodes, estimated memory

[Step 3] Stretching (optional)
  - Enable stretching? [Y/n]
  - Input stretch_eta (radial wall clustering)   [default=2.0]
  - Input stretch_xi  (circumferential)          [default=0.0]
  -> Display: first cell height estimate

[Step 4] LBM Parameters (optional)
  - Export LBM lattice data? [Y/n]
  - Input Re (Reynolds number)        [default=100]

[Step 5] Output Options
  - Output directory                  [default=./output]
  - Output format (tecplot/csv/hdf5)  [default=tecplot]
  - Generate plots?                   [default=yes]
  - Half-domain only?                 [default=no]

[Step 6] Summary & Confirmation
  -> Display all parameters
  -> "Generate grid? [Y/n]"

[Step 7] Generation & Export
  -> Progress bar: constants -> grid -> metrics -> quality -> export -> plot
  -> Print quality report
  -> Print output file paths
```

### Step 10: Tests

```python
# test_bipolar_constants.py
def test_known_case():
    """Test against existing code values: R1=5, R2=7.6, eps=0.5"""
    pass

def test_pdf_vs_snyder():
    """Both formulations give same alpha, beta, c"""
    pass

def test_symmetric_case():
    """phi=0.5, gamma varies: check boundary circles match r1, r2"""
    pass

# test_transforms.py
def test_roundtrip():
    """(xi,eta) -> (x,y) -> (xi,eta) should recover original"""
    pass

def test_inner_boundary_is_circle():
    """Points at eta=alpha should form a circle of radius r1"""
    pass

def test_outer_boundary_is_circle():
    """Points at eta=beta should form a circle of radius r2"""
    pass

def test_conformal_orthogonality():
    """g12 metric component should be ~0 everywhere (conformal = orthogonal)"""
    pass

# test_concentric_limit.py
def test_small_eccentricity():
    """As e -> 0 (but not zero), grid should approach concentric annulus"""
    pass
```

---

## 5. Key Validation Criteria

1. **Boundary fidelity**: Inner boundary nodes lie on circle(center=(e,0), radius=r1), outer on circle(origin, radius=r2)
2. **Orthogonality**: Since bipolar mapping is conformal, grid lines should be orthogonal everywhere; max deviation < 0.01 degrees
3. **Smooth Jacobian**: No singularities in the Jacobian within the domain
4. **Scale factor range**: h_min / h_max indicates grid non-uniformity; report this ratio
5. **Periodicity**: xi = -pi and xi = +pi produce the same physical point (periodic seam)
6. **Consistency**: PDF formulation (Eq.4) and Snyder formulation give identical constants to machine precision

---

## 6. Important Notes for Implementation

### 6.1 Singularity at epsilon = 0
Bipolar coordinates are **singular** when eccentricity = 0 (concentric case). The code must:
- Reject e = 0 with a clear error message
- Suggest using polar coordinates for the concentric case
- Handle very small e (e.g., e < 1e-10) gracefully

### 6.2 Periodicity in xi
- `xi = -pi` and `xi = pi` are the **same physical line** (the line connecting the two poles)
- For LBM, this means periodic boundary conditions in the xi-direction
- The grid should NOT duplicate the xi = pi column (or handle it as a ghost layer)

### 6.3 Narrow Gap Consideration
- When eccentricity is large (phi -> 1), the narrow gap region (near xi = pi for the offset direction) has very few physical-space grid points even with many computational-space points
- Consider **adaptive refinement** or strong **stretching in xi** for high eccentricity cases
- Report the minimum physical cell size as a quality metric

### 6.4 Conformal Mapping Property
- The bipolar coordinate transformation is conformal (angle-preserving)
- This means the grid is **inherently orthogonal** -- a major advantage for LBM
- The scale factor `h = c / (cosh(eta) - cos(xi))` fully characterizes the local mesh size
- For the general interpolation LBM, orthogonality simplifies the interpolation stencil construction

### 6.5 Coordinate Convention
- The existing code places the **outer circle centered at the origin** and shifts the inner circle
- The PDF uses the convention where poles are at `(+c, 0)` and `(-c, 0)` on the x-axis
- gamma_shift aligns the outer circle center to the coordinate origin
- Keep this convention consistent throughout

---

## 7. Dependencies

```
numpy          # core computation
matplotlib     # visualization
sympy          # symbolic validation (optional, for cross-checking)
h5py           # HDF5 export (optional)
```

---

## 8. Execution Priority

| Priority | Task | Reason |
|----------|------|--------|
| P0 | bipolar_constants.py | Foundation: all other modules depend on alpha, beta, c |
| P0 | bipolar_transform.py | Core transformation functions |
| P1 | bipolar_grid_generator.py | Main grid engine |
| P1 | grid_metrics.py | Essential for LBM (Jacobian, scale factors) |
| P1 | grid_io.py | Output functionality |
| P2 | grid_stretching.py | Quality improvement |
| P2 | grid_quality.py | Validation |
| P2 | grid_visualization.py | Visualization |
| P3 | lbm_grid_adapter.py | LBM-specific processing |
| P3 | main.py | CLI interface |
| P3 | tests/ | Correctness validation |
