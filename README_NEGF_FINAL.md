# `negf_sinw_final.py` — finalized NEGF code for silicon nanowire (GAA) transistors

One Python file that combines every earlier NEGF file in this repository into a
single code for running NEGF simulations of silicon nanowire transistors.
No existing file was modified; the new files are:

| file | purpose |
|---|---|
| `negf_sinw_final.py` | the combined code (module + command-line tool) |
| `test_negf_sinw_final.py` | tests (9 fast, 1 slow SCF regression) |
| `README_NEGF_FINAL.md` | this document |
| `negf_final_figures/` | figures from the runs quoted below |

```bash
pip install numpy scipy matplotlib
python negf_sinw_final.py selftest                   # 8 physics identities, ~15 s
python negf_sinw_final.py bias --quick --workers 4   # one self-consistent bias point
python negf_sinw_final.py --help                     # all subcommands
```

---

## 1. How the earlier files differ, and what went into the final code

There were two families of code.

**A. `negf_silicon_nanowire.py`** (+ `run_gaa_simulation.py`, `test_negf.py`).
This was my first file, later merged with the GAA code from the `Test` repository.
It has two unrelated halves:

* Part A: a rectangular Ny×Nz wire with a block recursive Green function and a
  *prescribed* linear potential (no electrostatics).
* Part B: a 1-D (or r–z) "GAA" solver that inverts the full matrix, plus a
  self-consistent loop.

**B. `negf_suite/`** (written afterwards).

* Level 1: a Datta-style 1-D teaching code.
* Level 2: a TranSIESTA-style self-consistent NEGF–Poisson GAA solver.
* Level 3: 6-band k·p hole transport.

Every core algorithm here is checked against an independent reference in the tests.

I ran numerical experiments on the old module before merging. They reproduced the following problems:

| old module | experiment | result |
|---|---|---|
| Part A electron correlation Gⁿ from the block RGF | compare with Gⁿ = G Σⁱⁿ G† from a dense inverse | wrong by ~100 %, although G^R was exact |
| Part B `PoissonSolver` | read + run | not a Poisson solver: hard-coded 0.3 eV barrier, 0.25 V threshold, charge feedback × 0.01 |
| Part B electron density | run `calculate_electron_density` | negative everywhere (sign error) |
| Part B leads vs device | lead band shifted by the doping Fermi level, device band not | 0.09 eV band step at both contacts |
| Part B drain bias | V_ds = +0.2 V | negative current: drain Fermi level raised instead of lowered |
| Part B bond current | compare with Landauer | 10¹³ "A" instead of ~10⁻⁵ A/eV (missing 1/2π and energy units) |
| Both parts | fixed real-axis energy grid, η = 10⁻⁶ eV | narrow channel resonances missed |
| Part B | 1-D chain for `nr = 1` | the nanowire radius only entered as a volume, with no transverse confinement |

Feature by feature, this is what the final code keeps and from where:

| capability | old module | negf_suite | final code |
|---|---|---|---|
| Lead self-energy (Lopez-Sancho) | yes (two copies) | Level 2 | Level 2 version, cached per energy |
| Recursive Green function | Part A (Gⁿ wrong) | Level 2 | Level 2, plus first-column blocks for the bond current |
| Equilibrium density | real-axis grid | TranSIESTA contour + Ozaki poles | both, selectable with `--density` |
| Non-equilibrium density | real-axis grid | adaptive quadrature, Brandbyge weighting | kept |
| Electrostatics | heuristic | 3-D Poisson, square gate-all-around | **3-D Poisson, square or circular gate-all-around** |
| Lead Fermi level | doping formula (non-degenerate, band step) | charge neutrality (fixed bracket) | charge neutrality with an adaptive bracket (works for light masses) |
| Cross-section | Ny×Nz rectangle or 1-D | square | **square or circle** (diameter W) |
| Valleys / materials | Si, Ge, InGaAs, GaAs (single mass) | Si single or 3 Δ valleys | all of them |
| Bond (local) current | wrong units | – | **fixed**, equal to Landauer to 3·10⁻⁷ at small η |
| LDOS / spectral function | yes | yes | yes |
| Hamiltonian export | dense, npz/csv/txt/mat | – | block-sparse, npz/mat/csv |
| Non-self-consistent quick mode | Part A | – | `--no-scf` (analytic gate barrier + drain drop) |
| I_d–V_g, I_d–V_ds, material comparison | runner script | I_d–V_g | `idvg`, `idvd`, `materials` subcommands |
| k·p holes | – | Level 3 | `kp` subcommand |
| Parallel energy loop | – | Level 2 | `--workers N`, with BLAS pinned to one thread |
| Datta 1-D model | – | Level 1 | limit `--W` equal to `--a` (one orbital per slice); Level 1 stays the teaching code |

Not carried over:

* **Büttiker-probe dephasing.** It lives only in Level 1; the final code is ballistic.
* **The r–z "cylindrical" Hamiltonian of old Part B.** The circular cross-section on a Cartesian grid replaces it.

---

## 2. Physics and conventions

* **Conduction band.** Effective-mass finite differences with spacing *a*. Silicon uses one isotropic valley (m\* = 0.26, fast) or the three Δ-valley pairs of a [100] wire (`--valleys si3`). Ge, InGaAs and GaAs use a single isotropic valley; for Ge, whose minimum is at L, this is a simplification.
* **Geometry.** A hard-wall Si core (square W×W or a circle of diameter W) sits inside an oxide shell t_ox. The metal gate wraps the oxide over the gate length L_g. The n⁺⁺ extensions L_s and L_d continue as semi-infinite leads.
* **Energies.** E = 0 is the source Fermi level. μ_S = 0 and μ_D = −V_ds. U = −φ is the electron potential energy. The gate is held at φ_S + V_g − φ_ms. A positive I_d means electrons flow from source to drain.
* **Current.** I = g(q/h)∫T(f_S − f_D)dE, with g = spin × valley degeneracy.
* **Numerical steps** (tagged `[TS-1]`…`[TS-8]` in the source, following TranSIESTA):
  1. Lead band offset from charge neutrality.
  2. Lopez-Sancho lead self-energies.
  3. Recursive Green function.
  4. Equilibrium density on a complex contour.
  5. Non-equilibrium density on the real axis, with adaptive Gauss–Kronrod quadrature and Brandbyge weighting.
  6. 3-D finite-volume Poisson with Gummel–Newton linearisation.
  7. Self-consistent loop with linear mixing.
  8. Transmission, current, LDOS and bond current.

---

## 3. Validation

`python test_negf_sinw_final.py`: all pass in 38 s. `NEGF_SLOW=1 NEGF_WORKERS=4 python test_negf_sinw_final.py` adds the SCF regression (194 s).

| check | result |
|---|---|
| RGF diagonal, last-column and first-column blocks vs dense inverse | 9·10⁻¹⁶ |
| Lopez-Sancho vs analytic −t e^{ika} | 6·10⁻¹⁰ |
| Equilibrium density: TranSIESTA contour / Ozaki poles vs 20 001-point real axis | 1.3·10⁻⁷ / 1.5·10⁻⁷ |
| Uniform wire: T = number of open subbands | exactly 1, 3, 4 |
| Bond current on every bond vs Landauer T(f_L − f_R) | 2.8·10⁻⁷ (η = 10⁻⁹) |
| Circular-GAA Poisson satisfies the discrete equation | 7·10⁻¹⁰ |
| 6-band k·p heavy-hole mass [001] | 0.2772 = 1/(γ₁ − 2γ₂) |
| Lead Fermi level inside the first subband for Si and InGaAs | yes |
| Higher V_g lowers the channel potential; export round-trip; mode-space convergence | yes |
| **SCF regression vs `negf_suite` Level 2** (quick device, V_g = 0.5 V) | **2.9919 µA in both codes** |

With the default device broadening η = 10⁻⁴ eV, the bond current in a full run is constant along the wire to about 1 %. The broadening acts as a weak particle sink; lower `eta` in `NWParams` if you need tighter conservation.

---

## 4. Results and timings

All runs used the cloud sandbox of this session: a 4-core Xeon at 2.1 GHz with 15 GB RAM, no GPU, Python 3.11, numpy 2.4 and scipy 1.17, with `--workers 4`. The quick device is a 1.8 nm Si core with L_s/L_g/L_d = 3/6/3 nm and V_ds = 0.3 V. Figures are in `negf_final_figures/`.

| command | result | wall time |
|---|---|---|
| `selftest` | 8/8 pass | 15 s |
| `bias --quick --Vg 0.5` (square) | I_d = 2.9919 µA, 16 SCF iterations | 199 s |
| `bias --quick --shape circle --W 2.1 --Vg 0.6` | I_d = 3.397 µA, 13 SCF iterations | 157 s |
| `bias --quick --valleys si3 --Vg 0.5` | I_d = 2.281 µA (three Δ valleys) | 493 s |
| `bias --quick --no-scf --Vg 0.5` | I_d = 3.490 µA (analytic potential) | 40 s |
| `idvg --quick` (V_g = 0 … 0.8 V, 5 points) | 0.0000 / 0.0023 / 0.797 / 3.247 / 3.173 µA | 679 s |
| `idvd --quick` (V_g = 0.5 and 0.7 V, 3 V_ds each) | saturation above V_ds ≈ 0.17 V | 1008 s |
| `materials --quick --no-scf --Vg 0.5` | Si 3.49, Ge 6.36, GaAs 10.88, InGaAs 17.76 µA | 98 s |
| `kp` (Si holes, 64 modes) | 23.1 → 0.10 µA for V_g = 0 → 0.45 V | 39 s |
| `export-h --quick` | 1440×1440 sparse, Hermitian | 2 s |

**Running on your own computer.** The quick device and every non-SCF or k·p run take seconds to a few minutes on any laptop. The default device (2.4 nm core, 16 nm long) takes about 8 minutes per self-consistent bias point on 4 cores. Three-valley Si costs about 2.5× the single-valley model.

---

## 5. Limitations

* **Model level.** Effective-mass and k·p Hamiltonians, not atomistic or DFT.
* **Transport.** Ballistic only, no phonon or roughness scattering.
* **Wavefunction boundary.** Hard wall at the Si/oxide interface.
* **k·p.** The hole k·p part is not self-consistent.
* **Material comparison.** It uses the same flat-band offset for every material, so it compares band structures, not optimised devices.
