# NEGF Quantum Transport Simulation for Silicon Nanowire

## Overview

This code implements the **Non-Equilibrium Green's Function (NEGF)** formalism
for ballistic quantum transport in a silicon nanowire, using an effective-mass
tight-binding Hamiltonian. It demonstrates two key numerical techniques:

1. **Surface Green's Function** (Sancho-Rubio iterative method) — to model
   semi-infinite source/drain contacts.
2. **Recursive Green's Function (RGF)** — to efficiently invert the large
   device Hamiltonian layer-by-layer.

## Physical Model

### Tight-Binding Hamiltonian (Effective-Mass Approximation)

The nanowire is discretized on a 3D grid with lattice spacing `a`. The
transport direction is `x`, while the cross-section spans `y` and `z`.

The discretized single-band effective-mass Schrödinger equation gives:

```
H |ψ⟩ = E |ψ⟩
```

with hopping parameters:

```
t_α = ℏ² / (2 m*_α a²),    α ∈ {x, y, z}
```

The Hamiltonian is block-tridiagonal along the transport direction:

```
        ┌                                    ┐
        │ H₁   V    0    0   ...  0    0     │
        │ V†   H₂   V    0   ...  0    0     │
H_dev = │ 0    V†   H₃   V   ...  0    0     │
        │ ⋮                  ⋱                │
        │ 0    0    0    0   ... V†   H_Nx    │
        └                                    ┘
```

- `Hᵢ` = on-site block for slice `i` (includes 2D cross-section Hamiltonian
  + electrostatic potential)
- `V` = inter-slice coupling matrix (diagonal, = −tₓ I)

## Algorithms

### 1. Surface Green's Function (Sancho-Rubio Method)

To model semi-infinite leads, we need the surface Green's function `gₛ(E)` of a
semi-infinite chain of identical slices. Direct inversion is impossible for an
infinite system, so we use the iterative decimation technique of
Lopez-Sancho et al. (1985):

```
Initialize:
    εₛ = H_slice,   ε = H_slice
    α  = V,          β = V†

Iterate until ‖α‖ < tolerance:
    g_ε  = (EI - ε)⁻¹
    εₛ  ← εₛ + α · g_ε · β
    ε   ← ε  + α · g_ε · β + β · g_ε · α
    α   ← α · g_ε · α
    β   ← β · g_ε · β

Result:
    gₛ = (EI - εₛ)⁻¹
```

This converges exponentially fast (each iteration doubles the effective lead
length). The contact self-energy is then:

```
Σ_L = V† · gₛ_L · V
Σ_R = V† · gₛ_R · V
```

### 2. Recursive Green's Function (RGF) Algorithm

Instead of inverting the full `(Nx·N_orb × Nx·N_orb)` matrix, the RGF
algorithm obtains the diagonal blocks of G^R in O(Nx · N_orb³) operations:

**Forward sweep** (left-connected Green's functions):

```
gL[0] = (EI - H₁ - Σ_L)⁻¹

gL[i] = (EI - Hᵢ - V† · gL[i-1] · V)⁻¹,   i = 1, ..., Nx-1
         (add Σ_R for the last slice)
```

**Backward sweep** (full diagonal blocks):

```
G[Nx-1] = gL[Nx-1]

G[i] = gL[i] + gL[i] · V · G[i+1] · V† · gL[i],   i = Nx-2, ..., 0
```

The electron correlation function G^n (needed for carrier density) is computed
with a similar two-pass RGF using in-scattering functions from the contacts.

### 3. Transmission (Fisher-Lee / Caroli Formula)

```
T(E) = Tr[Γ_L · G^R · Γ_R · G^A]
```

where `Γ = i(Σ - Σ†)` is the broadening matrix and `G^A = (G^R)†`.

The off-diagonal block `G^R_{0,Nx-1}` is built from the forward sweep:

```
G_{0,N-1} = gL[0] · V · gL[1] · V · ... · V · G[Nx-1]
```

### 4. Current Density (Landauer-Büttiker)

```
I = (2e/h) ∫ T(E) [f_S(E) - f_D(E)] dE
```

The factor of 2 accounts for spin degeneracy. `f_S` and `f_D` are Fermi-Dirac
distributions of source (μ_S = 0) and drain (μ_D = −eV_bias).

### 5. Local Density of States

```
LDOS(x, E) = −(1/π) Im[ Tr(G^R_{xx}) ]
```

## Usage

### Requirements

```
pip install numpy scipy matplotlib
```

### Quick Run

```bash
python negf_silicon_nanowire.py
```

This runs the default simulation (3×3 cross-section, 40 slices, V_bias=0.3 V)
and produces:
- Console output with current and conductance
- `negf_results.png` with four subplots:
  (a) Transmission spectrum
  (b) Current integrand
  (c) LDOS map
  (d) Band diagram

### Custom Parameters

```python
from negf_silicon_nanowire import DeviceParams, solve_negf, plot_results

p = DeviceParams(
    Ny=5, Nz=5,          # larger cross-section
    Nx=80,                # longer channel
    V_bias=0.5,           # higher bias
    V_gate=0.2,           # gate voltage
    T=77.0,               # low temperature
    E_min=-1.0, E_max=3.0,
    NE=400,
)

E, T, I, LDOS = solve_negf(p)
plot_results(E, T, I, LDOS, p)
```

### I-V Curve

```python
from negf_silicon_nanowire import DeviceParams, compute_iv_curve
import numpy as np

p = DeviceParams(Ny=3, Nz=3, Nx=40, NE=150)
V_values = np.linspace(0.01, 0.6, 12)
currents = compute_iv_curve(p, V_values)
```

## Code Structure

| Function | Description |
|---|---|
| `build_slice_hamiltonian()` | 2D tight-binding H for one cross-section |
| `build_coupling_matrix()` | Inter-slice hopping V = −tₓ I |
| `potential_profile()` | Electrostatic potential along the wire |
| `surface_green_function()` | Sancho-Rubio iterative surface GF |
| `contact_self_energy()` | Lead self-energy Σ = V† gₛ V |
| `broadening()` | Coupling matrix Γ = i(Σ − Σ†) |
| `recursive_green_function()` | Full RGF: G^R, G^n, T(E) |
| `solve_negf()` | Main solver: energy sweep → T(E), I, LDOS |
| `plot_results()` | Four-panel visualization |
| `compute_iv_curve()` | Bias sweep for I-V characteristic |

## References

1. S. Datta, *Quantum Transport: Atom to Transistor*, Cambridge University
   Press (2005).
2. M.P. Lopez-Sancho, J.M. Lopez-Sancho, J. Rubio, *Highly convergent schemes
   for the calculation of bulk and surface Green functions*, J. Phys. F: Met.
   Phys. **15**, 851 (1985).
3. A. Svizhenko, M.P. Anantram, T.R. Govindan, B. Biegel, R. Venugopal,
   *Two-dimensional quantum mechanical modeling of nanotransistors*, J. Appl.
   Phys. **91**, 2343 (2002).
4. R. Lake, G. Klimeck, R.C. Bowen, D. Jovanovic, *Single and multiband
   modeling of quantum electron transport through layered semiconductor
   devices*, J. Appl. Phys. **81**, 7845 (1997).
