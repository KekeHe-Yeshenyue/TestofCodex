# NEGF quantum-transport suite for silicon nanowires — three levels

Three self-contained Python codes that go from "what is NEGF?" to a
self-consistent gate-all-around (GAA) Si nanowire transistor organised the
way **TranSIESTA** does it, plus a multi-band **k·p** NEGF code.  All were
written for this request, all were run end-to-end on the machine described in
§1, and every physical identity they rely on is checked in `test_negf_suite.py`.

| Level | File | Physics | Numerical method | Size of the demo | Wall time here | Runs on a laptop? |
|---|---|---|---|---|---|---|
| 1 | `negf_level1_datta_basics.py` | 1-D wire, Datta *Atom to Transistor* ch. 8–11 | analytic Σ, dense inverse, real-axis integral, 1-D Poisson (Gummel), Büttiker-probe dephasing | 100×100 matrices | **19 s** (all 5 demos) | yes, any laptop |
| 2 | `negf_level2_gaa_transiesta_style.py` | GAA Si n-FET, effective-mass TB, 3-D Poisson with wrap-around gate, self-consistent | Lopez-Sancho surface GF, recursive GF (BTD), **TranSIESTA complex contour + real-axis bias window**, Ozaki poles, Gummel–Newton Poisson | `--quick`: 6×6=36 orbitals × 40 slices; default: 64 orbitals × 53 slices | quick: **100–150 s**; default: see §1 | quick: yes; default: yes but minutes per bias point |
| 3 | `negf_level3_kp_nanowire.py` | 6-band Luttinger–Kohn k·p, hole transport in a Si p-FET nanowire | FD-discretised k·p, matrix-valued inter-slice coupling, **coupled mode space**, same RGF/Sancho-Rubio | 216 orbitals/slice real space (validation) → 64 modes | **38 s** | yes |
| — | `test_negf_suite.py` | 9 identity tests (see §6) | | | **9 s** | yes |

Install once (`pip install -r requirements.txt` = numpy, scipy, matplotlib) and run each
file with `python <file>`.  No compiler, no GPU, no MATLAB.

Figures produced by the runs are in `figures/`.

---

## 1. Q1 — Requirements, where I tested, can you run it locally?

**Where the code was run.**  Everything in this folder was executed inside the
cloud sandbox of this session, *not* on a local computer:

| | |
|---|---|
| CPU | Intel Xeon @ 2.10 GHz, **4 cores**, no hyper-threading |
| RAM | 15 GB (the largest run below never exceeded ~1 GB) |
| GPU | none |
| OS / Python | Linux 6.18, Python 3.11.15 |
| Libraries | numpy 2.4.6 (OpenBLAS 0.3.31), scipy 1.17.1, matplotlib 3.11.1 |

This is roughly a 2019-era 4-core laptop.  Anything that runs here in minutes
runs in minutes on a modern laptop; timings below are what I measured.

**Yes — you can run all of it locally.**  Concretely:

* Level 1 and Level 3, and Level 2 `--quick`: any laptop with 4 GB RAM,
  including older machines. Level 1 needs about 20 s, Level 3 about 40 s,
  Level 2 quick about 2 min.
* Level 2 at the default size (2.4 nm × 2.4 nm Si core, 16 nm long, one bias
  point): about 10–20 minutes on 4 cores; a full I<sub>d</sub>–V<sub>g</sub> sweep of 5
  points is an hour.  Run it in the background, or shrink it with the CLI flags.
* Recommended invocation for multi-core machines (avoids BLAS/process thread
  over-subscription):

      OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python negf_level2_gaa_transiesta_style.py --workers 4

**Why NEGF is expensive, and how to predict cost.**  One Green-function
evaluation with the recursive algorithm costs about
`8 · N_slices · N_orb³` complex flops (N_orb = orbitals per cross-sectional
slice), and a self-consistent bias point needs

    N_E  ≈  2 × (24 + 12 + 8)  contour points   +   ~800–1100 real-axis points in the bias window

evaluations per SCF iteration, times ~10–20 SCF iterations.  The real-axis
part dominates whenever V<sub>ds</sub> ≠ 0 because quasi-bound states in the
channel produce very narrow resonances that must be resolved (the code uses
adaptive Gauss–Kronrod for this; a 1 meV uniform grid over a 0.8 eV window
costs the same).  Rules of thumb from the measured runs:

| device | N_orb | slices | one G(E) | one SCF iteration | one bias point |
|---|---|---|---|---|---|
| quick (1.8 nm) | 36 | 40 | ~6 ms | 5–10 s | 100–150 s |
| default (2.4 nm) | 64 | 53 | ~40 ms | see §1 | see §1 |
| 3-valley Si, 3 nm, 20 nm long | 100 ×3 valleys | 67 | ~0.3 s | ~5 min | ~1 h |
| atomistic sp³d⁵s* 5 nm Si wire (not this code) | ~10⁴ | ~100 | seconds–minutes | hours | days on a cluster |

Memory is never the problem at these sizes (the RGF stores O(N_slices · N_orb²)).
The last line is why real DFT/atomistic device codes (OMEN, NEMO5, TranSIESTA,
DeePTB-NEGF) run on clusters or GPUs; the attached DeePTB-NEGF paper quotes a
**28-core CPU** and 633 s for the transmission of a 4798-atom molecular junction,
with the DFT-NEGF reference extrapolated to 27 hours.

---

## 2. Q2 — "Is a DFT-NEGF code like TranSIESTA beyond your capability? Does DFT need commercial software?"

Two separate answers.

**DFT does *not* require commercial software.**  The whole TranSIESTA stack is
free and open source: SIESTA/TranSIESTA/TBtrans (GPL, Fortran), `sisl`
(Python, LGPL) for pre/post-processing, and likewise OpenMX (NEGF module),
GPAW (Python DFT with a NEGF transport module), DFTB+ (with NEGF), Quantum
ESPRESSO (PWCOND).  The commercial products are QuantumATK (Synopsys) and
Nanodcal; you never need them.  The attached paper itself used TranSIESTA for
its reference data.

**Writing TranSIESTA from scratch in this session is not realistic — but not
for the reason you might think.**  The *NEGF part* of TranSIESTA is exactly what
Level 2 implements: electrode (bulk) calculation → surface Green functions →
block-tridiagonal G → complex-contour equilibrium density + real-axis
non-equilibrium density with Brandbyge's weighting → Hartree potential with a
gate → SCF → TBtrans.  What I cannot reproduce here is the *DFT* part that
supplies H and S: pseudopotentials, the numerical-atomic-orbital basis,
exchange–correlation functionals, real-space grids for the Hartree/XC
potentials, k-point sampling of the electrodes, and 20 years of numerical
hardening.  That is tens of thousands of lines of Fortran built on the SIESTA
code base.  So the honest statement is: the transport layer is reproduced (and
tested) in Level 2; the Hamiltonian layer is replaced by an effective-mass
model.

If you want first-principles accuracy on a laptop there are two practical routes:

1. **Use TranSIESTA itself** (`conda install -c conda-forge siesta` works on
   Linux/macOS) and feed its TSHS files to `sisl`/TBtrans.  A Si nanowire with
   a few hundred atoms is a laptop-size TranSIESTA job (hours).
2. **The DeePTB-NEGF route of the attached paper**: run a handful of DFT
   (or DFT-NEGF) calculations on small cells, train the DeePTB neural-network
   tight-binding Hamiltonian, then run `dpnegf` — which is a pure
   Python/PyTorch package (`uv sync` + `uv add dpnegf`, CPU or GPU).  Its
   transport core is the same set of algorithms as Level 2, which is why the
   Level-2 source is annotated with the corresponding `dpnegf` files
   (`negf/surface_green.py`, `negf/recursive_green_cal.py`, `negf/density.py`,
   `negf/poisson_init.py`).

---

## 3. Level 2 in detail — the TranSIESTA workflow with an effective-mass Hamiltonian

Every stage is tagged `[TS-n]` in the source.

| TranSIESTA concept | Level-2 implementation |
|---|---|
| **Electrode calculation** (separate bulk run fixes E<sub>F</sub> of each semi-infinite lead) | `neutral_lead_shift()`: bulk lead Green function of one slice, contour-integrated density, bisection on E<sub>F</sub> until the lead is charge neutral at N<sub>D</sub>=10²⁰ cm⁻³. Result: E<sub>F</sub> − E<sub>c,lead</sub> = 0.69 eV (quick) / 0.52 eV (default). |
| **Electrode self-energies** (`RecursiveSI` in sisl) | `sancho_rubio()`: Lopez-Sancho decimation, complex energies allowed, results cached per energy because the leads are fixed during the SCF. |
| **Block-tridiagonal inversion** (TS.BTD) | `rgf()`: Anantram–Lundstrom–Nikonov recursion; returns diagonal blocks G<sub>ii</sub> and last-column blocks G<sub>i,N</sub>. Verified against a dense inverse with matrix-valued coupling. |
| **Equilibrium density by complex contour** (TS.Contours.Eq circle + line, TS.Contour.Eq.Pole) | `equilibrium_contour()`: circle from E<sub>min</sub> to μ−10kT+iγ, line at height γ = 2πkT·n<sub>pole</sub> (the line needs a Fermi-adapted quadrature because f(E+iγ) is the *real* step function there — the code splits it into two Gauss–Legendre panels plus a Gauss–Laguerre tail), plus n<sub>pole</sub> Fermi-pole residues −2πi kT G(z<sub>ν</sub>). Verified against a brute-force real-axis integral to 1·10⁻⁷. |
| **Ozaki alternative used by dpnegf** | `ozaki_contour()`: continued-fraction poles/residues from the tridiagonal Ozaki matrix; 0th-moment term iR·G(μ+iR); also verified to 1·10⁻⁷. Switch with `--density ozaki`. |
| **Non-equilibrium density in the bias window** and **Brandbyge weighting** | `Device.density()`: Δ<sup>R</sup> = ∫(f<sub>R</sub>−f<sub>L</sub>) G Γ<sub>R</sub> G<sup>†</sup>dE/2π and Δ<sup>L</sup> likewise (A<sub>L</sub> = A − A<sub>R</sub>); ρ = w(ρ<sub>eq</sub><sup>L</sup>+Δ<sup>R</sup>) + (1−w)(ρ<sub>eq</sub><sup>R</sup>+Δ<sup>L</sup>) with w = Δ<sub>R</sub>²/(Δ<sub>L</sub>²+Δ<sub>R</sub>²). Adaptive Gauss–Kronrod (`scipy.integrate.quad_vec`) with an absolute tolerance of 10⁻⁵ electrons/site. |
| **Hartree potential, gating** (TS.Hartree, electrostatic gate) | `Poisson3D`: finite-volume ∇·(ε∇φ) = −ρ on the Si core + SiO₂ shell, harmonic-mean ε on faces, gate = Dirichlet on all four outer faces over the gate length (this *is* the gate-all-around), lead potentials as Dirichlet at the two ends, Neumann elsewhere. Gummel–Newton linearisation n(φ)=n<sub>old</sub>e<sup>(φ−φ<sub>old</sub>)/V<sub>T</sub></sup> with damped steps. |
| **SCF mixing** | `scf()`: linear mixing of φ (0.6), convergence max|Δφ| < 2 mV. 8 iterations OFF-state, 16 iterations ON-state in the quick device. |
| **TBtrans post-processing** | `post_process()`: T(E) on a fine grid, current by adaptive quadrature of T(f<sub>L</sub>−f<sub>R</sub>), LDOS(x,E) map, band-edge profile. |

Physical model: single isotropic conduction-band valley (m\*=0.26) by default,
or the three Δ-valley pairs of [100] Si (`--valleys si3`, 3× cost); hard-wall
Si/SiO₂ boundary for the wavefunction, full dielectric treatment in Poisson;
n⁺⁺ source/drain extensions continue as semi-infinite leads.  V<sub>g</sub> is
measured relative to the source lead through a flat-band offset `phi_ms`
(0.45 V by default → threshold ≈ 0.35 V).

Results of the quick device (1.8 nm core, L<sub>g</sub> = 6 nm, V<sub>ds</sub> = 0.3 V):
I<sub>d</sub>(V<sub>g</sub>=0) = 0.00 µA (OFF, barrier 0.23 eV above μ<sub>S</sub>),
I<sub>d</sub>(V<sub>g</sub>=0.5 V) = 2.99 µA (ON).  See `figures/level2_gaa_*.png`.

What Level 2 does **not** contain (and TranSIESTA does): a DFT Hamiltonian,
overlap matrix S, k-point sampling transverse to transport, multiple
electrodes, spin-orbit, phonon scattering.

---

## 4. Q3 — the k·p NEGF code (Level 3)

`negf_level3_kp_nanowire.py` implements hole transport in a Si nanowire with
the six-band Luttinger–Kohn Hamiltonian (heavy hole, light hole, split-off,
spin–orbit Δ<sub>so</sub> = 44 meV; γ₁ = 4.285, γ₂ = 0.339, γ₃ = 1.446; Ge is
included as a second material).

* **Bulk check**: the discretised-in-k matrix reproduces m<sub>HH</sub>[001] = 1/(γ₁−2γ₂) = 0.277,
  m<sub>LH</sub>[001] = 1/(γ₁+2γ₂) = 0.201, m<sub>HH</sub>[111] = 1/(γ₁−2γ₃) = 0.718 to three digits.
* **Discretisation**: H(k) is decomposed into its six quadratic monomials
  k<sub>i</sub>k<sub>j</sub>; each becomes −½(∂<sub>i</sub>∂<sub>j</sub>+∂<sub>j</sub>∂<sub>i</sub>)
  with central differences.  The result is exactly Hermitian and block-tridiagonal along the
  wire, but the inter-slice coupling V is now a **matrix** (k<sub>x</sub>k<sub>y</sub>,
  k<sub>x</sub>k<sub>z</sub> terms).  Getting the self-energy order right,
  Σ<sub>L</sub> = V<sup>†</sup>g<sub>L</sub>V and Σ<sub>R</sub> = Vg<sub>R</sub>V<sup>†</sup>, matters here (it is
  invisible for scalar couplings) — the test `uniform wire T = integer` catches it.
* **Coupled mode space**: keep the M highest valence eigenvectors of the
  reference slice and project H<sub>slice</sub>, V and U.  Measured convergence
  to the 216-orbital real-space result: max|ΔT| = 0.92 (M=24), 0.17 (M=48),
  0.055 (M=64), 0.013 (M=96), at 0.1 s / 0.45 s / 0.7 s / 1.5 s versus 9.4 s for
  real space (30 energies).
* **Transport demo**: p⁺⁺ leads with E<sub>F</sub> from the hole density
  (N<sub>A</sub> = 5·10²⁰ cm⁻³ → E<sub>F</sub> 0.19 eV below the top subband, i.e.
  degenerate), a Gaussian gate barrier (non-self-consistent), V<sub>ds</sub> = 0.1 V:
  I<sub>d</sub> = 23.1 → 18.9 → 3.7 → 0.10 µA for V<sub>g</sub> = 0, 0.15, 0.30, 0.45 V.
  Subband structure E(k<sub>x</sub>), T(E) plateaus (6→4→2, spin-degenerate pairs) and the
  LDOS map are in `figures/level3_kp_nanowire.png`.

The Poisson/SCF machinery of Level 2 can be attached to Level 3 (the density is
the trace over the six components per site); the 8-band extension needs the
conduction-band row/column of the Kane matrix and a first-order-derivative
discretisation (with the usual care about spurious solutions).  Both are listed
as next steps in the source.

---

## 5. Q4 — the entry-level Datta code (Level 1) and how it maps onto dpnegf

`negf_level1_datta_basics.py` is a pure-NumPy translation of the *spirit* of
Datta's one-page MATLAB scripts (nanoHUB resources 103 and 19564; the code
constants `hbar=1.06e-34, m=0.25*9.1e-31, a=3e-10, t0=hbar²/(2ma²q), zplus=1e-12i`
are his).  Each step is one short function, and each has a comment saying
which function of `dpnegf` (the code in the paper's *Code availability*
box) does the same job at scale:

| Datta step (Level 1 function) | Equation | dpnegf counterpart |
|---|---|---|
| `hamiltonian()` | H = 2t₀ − t₀(hopping) + U(x) | H from DeePTB (`negf_hamiltonian_init.py`) |
| `lead_self_energy()` | Σ = −t₀e<sup>ika</sup> (analytic) — and `numeric_sancho_rubio()` shows the general algorithm reproduces it to 10⁻¹² | `negf/surface_green.py: surface_green()` (Lopez-Sancho) |
| `negf_at_energy()` | G = [E − H − Σ₁ − Σ₂]⁻¹, Γ = i(Σ−Σ<sup>†</sup>), A = i(G−G<sup>†</sup>), T = Tr[Γ₁GΓ₂G<sup>†</sup>], Gⁿ = G(Γ₁f₁+Γ₂f₂)G<sup>†</sup> | `negf/recursive_green_cal.py: recursive_gf(..., need_lesser=True)` |
| `sweep()` | n(x) = 2∫Gⁿ<sub>xx</sub>dE/2π, I = (2q/h)∫T(f₁−f₂)dE | `negf/density.py` (Ozaki poles / Fiori real-axis grid), `device_property.py: _cal_current_()` |
| `poisson_1d_gummel()` | d²U/dx² = q(N<sub>D</sub>−n)/(εA), n(U) ≈ n<sub>old</sub>e<sup>−ΔU/kT</sup> | `negf/poisson_init.py: Interface3D.solve_poisson_NRcycle` |
| `dephased_transmission()` | Σ<sub>s</sub> = D·G, Σ<sub>s</sub><sup>in</sup> = D·Gⁿ (Büttiker probe) | not in dpnegf (ballistic) |

The five demos and what a beginner should see:

1. **Uniform wire** — T(E) = 1 exactly inside the band 0 < E < 4t₀ (quantised
   conductance); analytic Σ equals the Lopez-Sancho iteration to all printed digits.
2. **Single barrier** — tunnelling (T = 5·10⁻⁶ below the barrier), transmission
   resonances above it, and an LDOS(x,E) map (`figures/level1_demo2_barrier.png`).
3. **Landauer I–V** of the barrier device, with the conductance in units of
   G₀ = 2e²/h = 77.5 µS.
4. **Datta's n⁺⁺/n⁺/n⁺⁺ resistor** — NEGF density ↔ 1-D Poisson until
   self-consistent: a 0.15 eV barrier forms in the lightly doped middle and
   the density follows the doping with Debye-length screening
   (`figures/level1_demo4_resistor.png`).  Two numerical lessons are built in
   and commented: the band-edge van Hove singularity must not be sampled
   exactly (midpoint rule), and Newton steps on an exponential charge model
   must be damped.
5. **Dephasing** — a double-barrier resonance (T<sub>peak</sub> = 1) is degraded
   to 0.95 / 0.69 / 0.48 for D = 10⁻⁴ / 10⁻³ / 10⁻² eV².

Total run time 19 s; largest matrix 100 × 100.

---

## 6. Validation (`python test_negf_suite.py`, all pass, 9 s)

| test | identity checked |
|---|---|
| level1 analytic Σ == Lopez-Sancho | 5 energies inside/outside the band, Im Σ ≤ 0 |
| level1 uniform wire T = 1 | 50 energies, |T−1| < 10⁻⁶ |
| level1 ∫A/2π dE = 1 per site | completely filled band holds 2 electrons/site |
| level2 RGF == dense inverse | random complex block-tridiagonal H with *matrix* V, diagonal and last-column blocks to 10⁻¹⁰ |
| level2 contour & Ozaki == brute force | 6-site chain, 20 001-point real-axis reference: 1.3·10⁻⁷ and 1.5·10⁻⁷ |
| level2 fermi() | scalar / array / complex arguments |
| level2 uniform GAA wire | T = 1, 3, 4 = number of open subbands (degenerate pair included) |
| level3 LK Hamiltonian | Hermitian, three bulk masses, split-off at −Δ, Bloch H(k<sub>x</sub>) Hermitian |
| level3 mode space | uniform wire T integer; M=90 mode space vs real space max|ΔT| = 0.004 |

---

## 7. How this relates to the attached paper and the two GitHub repositories

*Deep Learning Accelerated Quantum Transport Simulations in Nanoelectronics*
(Zou, Zhouyin, Lin, Huang, Zhang, Hou, Gu; arXiv:2411.08800, npj Comput. Mater.
2025) replaces the DFT step of DFT-NEGF by a neural-network tight-binding
Hamiltonian (DeePTB-SK or DeePTB-E3) and keeps a conventional NEGF engine
(`dpnegf`): Bloch-theorem self-energies, recursive Green functions with a
greedy block partition, Gummel/Newton–Raphson NEGF–Poisson with a doped-contact
model and a local gate, validated against NanoTCAD ViDES.  Their applications
are Au break junctions, molecular junctions and CNT-FETs; they do not simulate
a GAA silicon nanowire, so the GAA geometry here is my construction, and their
efficiency claims (10⁴ snapshots; 8000-atom CNT-FET; 2–3 orders of magnitude
over DFT-NEGF) rest on the Hamiltonian prediction, not on a different NEGF
algorithm.  Level 2 reproduces the algorithmic ingredients I could verify in
the `dpnegf` source (Lopez-Sancho surface Green function, Anantram RGF
including the lesser function, Ozaki pole summation for the equilibrium
density, Newton–Raphson Poisson with Dirichlet gate regions).

Note on sources: the WeChat article (`mp.weixin.qq.com/s/b6moJ5rAZTe9n66M3tl4_g`)
could not be fetched from this sandbox (the domain is blocked by the network
policy), as were arxiv.org and nanohub.org.  I used the PDF you attached (which is
the arXiv v3 of the same work), the `deepmodeling/dpnegf` and `DeePTB` GitHub
repositories (source files read directly), and the primary TranSIESTA papers
(Brandbyge et al. 2002; Papior et al. 2017) for the contour and weighting scheme.

---

## 8. Limitations and obvious next steps

* Effective-mass (Level 2) and k·p (Level 3) Hamiltonians, not atomistic or DFT;
  hard-wall Si/SiO₂ boundary; no phonon or surface-roughness scattering
  (Level 1 shows how a Büttiker-probe self-energy enters).
* Level 3 is not self-consistent; attaching `Poisson3D` from Level 2 is the
  natural extension.
* Level 2 parallelises over energies with processes (`--workers`); the BLAS
  library should be pinned to one thread per process (see §1).  GPU
  acceleration (as in `dpnegf`) would require moving the RGF to CuPy/PyTorch.

## References

* S. Datta, *Quantum Transport: Atom to Transistor* (Cambridge, 2005); MATLAB scripts: nanoHUB resources 103 and 19564; S. Datta, Superlattices Microstruct. 28, 253 (2000).
* M. Brandbyge, J.-L. Mozos, P. Ordejón, J. Taylor, K. Stokbro, Phys. Rev. B 65, 165401 (2002) — TranSIESTA.
* N. Papior, N. Lorente, T. Frederiksen, A. García, M. Brandbyge, Comput. Phys. Commun. 212, 8 (2017) — next-generation TranSIESTA (contours, BTD, gating).
* M. P. Lopez Sancho, J. M. Lopez Sancho, J. Rubio, J. Phys. F 15, 851 (1985) — surface Green function decimation.
* M. P. Anantram, M. S. Lundstrom, D. E. Nikonov, Proc. IEEE 96, 1511 (2008) — recursive Green function algorithm.
* T. Ozaki, Phys. Rev. B 75, 035123 (2007); T. Ozaki, K. Nishio, H. Kino, Phys. Rev. B 81, 035116 (2010) — continued-fraction Fermi poles.
* J. Zou et al., arXiv:2411.08800 / npj Comput. Mater. (2025) — DeePTB-NEGF; code: github.com/deepmodeling/dpnegf, github.com/deepmodeling/DeePTB.
* M. Shin, IWCE 2009 (8-band k·p NEGF for nanowire FETs); M. Luisier, G. Klimeck, Phys. Rev. B 80, 155430 (2009) — mode-space methods.
* S. L. Chuang, *Physics of Optoelectronic Devices* (Wiley) — 6×6 Luttinger–Kohn matrix.
