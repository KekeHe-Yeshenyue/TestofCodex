# NEGF Quantum Transport Simulation for Silicon Nanowire & GAA Transistors

## Overview

This code implements the **Non-Equilibrium Green's Function (NEGF)** formalism
for ballistic quantum transport in semiconductor nanowires. It provides two
complementary simulation modes unified in a single module:

| Mode | Geometry | GF Method | Self-Consistency |
|---|---|---|---|
| **Rectangular nanowire** | Ny x Nz cross-section | Block-matrix RGF | No (fixed potential) |
| **GAA transistor** | Cylindrical (1D/2D) | Full-matrix inversion | Yes (NEGF-Poisson loop) |

### Key Algorithms

1. **Surface Green's Function** (Sancho-Rubio decimation) for semi-infinite leads
2. **Recursive Green's Function (RGF)** for block-tridiagonal systems
3. **Full G^R, G^A, G^<** via direct inversion for GAA mode
4. **Self-consistent NEGF-Poisson loop** for electrostatics

### Key Features

- Multiple materials: Si, Ge, InGaAs, GaAs (configurable effective mass, bandgap, dielectric)
- Transmission T(E), current (Landauer-Buttiker), LDOS, electron density
- Bond current for spatially-resolved current flow
- Transfer characteristics (I_D vs V_G), output characteristics (I_D vs V_D)
- Hamiltonian export (npz, csv, txt, mat formats)
- Comprehensive test suite

## Physical Model

### Tight-Binding Hamiltonian (Effective-Mass Approximation)

The nanowire is discretized on a grid with lattice spacing `a` (rectangular) or
`dz` (cylindrical). Hopping parameters:

```
t_alpha = h_bar^2 / (2 m*_alpha a^2),    alpha in {x, y, z}
```

The Hamiltonian is block-tridiagonal along the transport direction:

```
        [ H_1   V    0    0   ...  0    0   ]
        [ V+   H_2   V    0   ...  0    0   ]
H_dev = [ 0    V+   H_3   V   ...  0    0   ]
        [ :                  .               ]
        [ 0    0    0    0   ... V+   H_Nx  ]
```

### Surface Green's Function (Sancho-Rubio)

For semi-infinite leads, the iterative decimation converges exponentially:

```
Initialize:  eps_s = H_slice,  eps = H_slice,  alpha = V,  beta = V+

Iterate until ||alpha|| < tol:
    g = (EI - eps)^(-1)
    eps_s <- eps_s + alpha . g . beta
    eps   <- eps + alpha . g . beta + beta . g . alpha
    alpha <- alpha . g . alpha
    beta  <- beta . g . beta

Result: g_s = (EI - eps_s)^(-1)
```

Contact self-energy: `Sigma = V+ . g_s . V`

### Recursive Green's Function (Block-Matrix RGF)

O(Nx . N_orb^3) algorithm for the rectangular nanowire:

**Forward sweep** (left-connected GFs):
```
gL[0] = (EI - H_1 - Sigma_L)^(-1)
gL[i] = (EI - H_i - V+ . gL[i-1] . V)^(-1)
```

**Backward sweep** (full diagonal blocks):
```
G[Nx-1] = gL[Nx-1]
G[i] = gL[i] + gL[i] . V . G[i+1] . V+ . gL[i]
```

G^n (electron correlation) is computed with an analogous two-pass RGF using
in-scattering functions from the contacts.

### Transport Quantities

- **Transmission**: `T(E) = Tr[Gamma_L . G^R . Gamma_R . G^A]`
- **Current**: `I = (2e/h) integral T(E) [f_S(E) - f_D(E)] dE`
- **LDOS**: `LDOS(x, E) = -(1/pi) Im[Tr(G^R_xx)]`
- **Electron density**: `n(r) = -(1/pi) integral Im[G^<(r,r,E)] dE`
- **Bond current**: `I_{l->l+1} = (2e/h_bar) Re{Tr[H_{l,l+1} . G^<_{l+1,l}]}`

### Self-Consistent Loop (GAA Mode)

```
Init density rho -> Poisson (V) -> Hamiltonian H(V) -> NEGF (G^<)
    -> new rho -> mix -> check convergence -> repeat
```

## Usage

### Requirements

```
pip install numpy scipy matplotlib
```

### Rectangular Nanowire (Block-RGF)

```bash
python negf_silicon_nanowire.py
```

```python
from negf_silicon_nanowire import DeviceParams, solve_negf, plot_results

p = DeviceParams(Ny=3, Nz=3, Nx=40, V_bias=0.3, V_gate=0.0)
E, T, I, LDOS = solve_negf(p)
plot_results(E, T, I, LDOS, p)
```

### GAA Transistor (Self-Consistent)

```bash
python run_gaa_simulation.py
```

```python
from negf_silicon_nanowire import GAADeviceParams, SelfConsistentNEGF

params = GAADeviceParams(
    channel_length=12e-9, nanowire_radius=2.5e-9,
    nz=40, nr=1, vg=0.4, vd=0.1,
    source_doping=1e20, drain_doping=1e20,
)
solver = SelfConsistentNEGF(params, mixing=0.15, max_iter=30, tol=1e-3)
results = solver.solve(E_min=-0.5, E_max=0.8, n_energy=80)
```

### Material Comparison

```python
from negf_silicon_nanowire import Material, MaterialType, GAADeviceParams

params = GAADeviceParams(channel_length=10e-9, nz=30, nr=1, vg=0.3)
params.material = Material.get_material(MaterialType.INGAAS)
```

### Run Tests

```bash
python test_negf.py
```

## File Structure

| File | Description |
|---|---|
| `negf_silicon_nanowire.py` | Unified NEGF module (rectangular + GAA) |
| `run_gaa_simulation.py` | GAA simulation examples (transfer, output, LDOS, materials) |
| `test_negf.py` | Comprehensive test suite (12 tests) |
| `README_NEGF.md` | This documentation |

## Code Structure (`negf_silicon_nanowire.py`)

### Part A: Rectangular Nanowire (Block-RGF)

| Function | Description |
|---|---|
| `build_slice_hamiltonian()` | 2D tight-binding H for one Ny x Nz cross-section |
| `build_coupling_matrix()` | Inter-slice hopping V = -t_x I |
| `potential_profile()` | Electrostatic potential along the wire |
| `surface_green_function()` | Functional Sancho-Rubio surface GF |
| `contact_self_energy()` | Lead self-energy Sigma = V+ g_s V |
| `recursive_green_function()` | Block-RGF: G^R, G^n diagonal blocks, T(E) |
| `solve_negf()` | Full energy sweep -> T(E), I, LDOS |
| `plot_results()` | Four-panel visualization |
| `compute_iv_curve()` | Bias sweep for I-V |

### Part B: GAA Transistor (Full-Matrix)

| Class | Description |
|---|---|
| `MaterialType` / `Material` | Semiconductor material database |
| `GAADeviceParams` | Cylindrical transistor parameters |
| `SurfaceGreenFunction` | Sancho-Rubio + iterative surface GF |
| `NEGFSolver` | G^R, G^A, G^<, T(E), LDOS, bond current, H export |
| `PoissonSolver` | 1D Poisson with GAA gate coupling |
| `SelfConsistentNEGF` | NEGF-Poisson self-consistent loop |

## References

1. S. Datta, *Quantum Transport: Atom to Transistor*, Cambridge (2005).
2. M.P. Lopez-Sancho et al., J. Phys. F: Met. Phys. **15**, 851 (1985).
3. A. Svizhenko et al., J. Appl. Phys. **91**, 2343 (2002).
4. R. Lake et al., J. Appl. Phys. **81**, 7845 (1997).
