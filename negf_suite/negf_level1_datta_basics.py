#!/usr/bin/env python3
"""
=============================================================================
 LEVEL 1  --  NEGF for absolute beginners  (Datta, "Atom to Transistor" style)
=============================================================================

A ~300-line, pure NumPy translation of the *spirit* of Supriyo Datta's
famous "one page" MATLAB scripts (nanoHUB resources 103 and 19564, and the
numerical examples of *Quantum Transport: Atom to Transistor*, Cambridge
2005, chapters 8-11).  Everything is a 1-D chain of grid points, every
matrix is small and dense, and every quantity is computed with a single
matrix inverse -- exactly like Datta's `G = inv((E+zplus)*eye(Np)-H-sig1-sig2)`.

WHAT YOU WILL LEARN (each is one short function below)
  1. Hamiltonian of a 1-D wire on a grid            H = 2t0 - t0(hop)  + U(x)
  2. Contact self-energy of a semi-infinite wire    Sigma = -t0 * exp(i k a)
  3. Retarded Green's function                      G = [E - H - Sigma1 - Sigma2]^-1
  4. Broadening / spectral function / LDOS          Gamma = i(Sigma - Sigma^+),  A = i(G - G^+)
  5. Transmission (Fisher-Lee / Caroli)             T = Tr[Gamma1 G Gamma2 G^+]
  6. Electron correlation function and density      G^n = G (Gamma1 f1 + Gamma2 f2) G^+
  7. Landauer current                               I = (2q/h) Int T (f1 - f2) dE
  8. Self-consistent potential (Poisson)            Datta's n++ / n+ / n++ resistor
  9. Phase-breaking with a Buettiker probe          Sigma_s = D * G  (Datta ch. 10)

HOW THIS MAPS ONTO A "REAL" CODE  (dpnegf, github.com/deepmodeling/dpnegf)
  Datta step                     ->  dpnegf file / function
  -----------------------------------------------------------------------
  analytic Sigma = -t0 e^{ika}   ->  negf/surface_green.py : surface_green()
                                     (Lopez-Sancho iteration, works for any
                                      multi-orbital lead, see numeric_sancho_rubio()
                                      below which reproduces it in 15 lines)
  dense inverse  inv(E-H-Sigma)  ->  negf/recursive_green_cal.py : recursive_gf()
                                     (block-tridiagonal RGF, same maths, O(N) cost)
  Gn = G Sigma_in G^+            ->  recursive_gf(..., need_lesser=True, s_in=[Gamma f])
  energy-grid density integral   ->  negf/density.py : Ozaki.integrate()
                                     (complex-plane pole sum instead of a real grid)
  1-D Poisson below              ->  negf/poisson_init.py : Interface3D (3-D, Newton)
  Landauer current               ->  negf/device_property.py : _cal_current_()

REQUIREMENTS   python >= 3.9, numpy, scipy, matplotlib.  No compiler, no GPU.
RUN            python negf_level1_datta_basics.py            (all demos, ~20 s)
               python negf_level1_datta_basics.py --demo 2   (single demo)
HARDWARE       Any laptop.  Largest matrix here is 100 x 100.
=============================================================================
"""
import argparse
import time
import numpy as np
from numpy.linalg import inv

# ----------------------------------------------------------------------------
# 0. Constants -- literally the first lines of every Datta script
# ----------------------------------------------------------------------------
hbar = 1.06e-34          # J s
q    = 1.6e-19           # C
m    = 0.25 * 9.1e-31    # effective mass (kg)   (Datta uses 0.25 m0)
a    = 3e-10             # grid spacing (m)      (0.3 nm)
t0   = hbar**2 / (2 * m * a**2 * q)   # hopping energy in eV  (~1.69 eV)
kT   = 0.025             # eV  (room temperature)
zplus = 1e-12j           # tiny imaginary part  "E + i0+"

def fermi(E, mu):
    """Fermi-Dirac function f(E) = 1/(1+exp((E-mu)/kT))  (E, mu in eV)."""
    x = np.clip((E - mu) / kT, -200, 200)
    return 1.0 / (1.0 + np.exp(x))


# ----------------------------------------------------------------------------
# 1. Hamiltonian of a 1-D wire discretised on Np grid points
#    Finite differences of -(hbar^2/2m) d^2/dx^2 + U(x)  give a tridiagonal H:
#       H(i,i)   = 2 t0 + U(i)
#       H(i,i+1) = H(i+1,i) = -t0
# ----------------------------------------------------------------------------
def hamiltonian(U):
    Np = len(U)
    H = 2 * t0 * np.eye(Np) - t0 * np.eye(Np, k=1) - t0 * np.eye(Np, k=-1)
    return H + np.diag(U)


# ----------------------------------------------------------------------------
# 2. Contact self-energy of a semi-infinite 1-D lead
#    Dispersion of the lead:  E = U + 2 t0 (1 - cos ka)
#    Surface Green's function g_s = -(1/t0) e^{ika}; self-energy Sigma = t0^2 g_s
#       Sigma = -t0 * exp(i k a)        (Datta eq. 8.1.x, `sig1(1,1) = -t0*exp(i*ka)`)
#    Only the grid point touching the contact is affected.
# ----------------------------------------------------------------------------
def lead_self_energy(E, U_edge):
    """Return the scalar Sigma = -t0 exp(i k a) with the decaying/outgoing root."""
    # z = exp(ika) solves  z + 1/z = 2 - (E - U)/t0   (from the dispersion relation)
    c = 2.0 - (E + zplus - U_edge) / t0
    disc = np.sqrt(c * c - 4.0 + 0j)
    z1, z2 = (c + disc) / 2, (c - disc) / 2
    z = z1 if abs(z1) <= abs(z2) else z2      # |z| <= 1  ->  retarded (Im Sigma <= 0)
    return -t0 * z

def numeric_sancho_rubio(E, h00, h01, eta=1e-8, tol=1e-12, maxit=200):
    """
    The SAME quantity computed the way dpnegf/TranSIESTA do it for an arbitrary
    (multi-orbital) lead: Lopez-Sancho (Sancho-Rubio) decimation.
    h00 = principal-layer Hamiltonian, h01 = coupling to the next layer.
    Returns the surface Green's function g_s.  Compare with lead_self_energy():
    Sigma = h01^+ g_s h01.
    """
    I = np.eye(h00.shape[0])
    Ez = (E + 1j * eta) * I
    eps_s, eps = h00.copy(), h00.copy()
    alpha, beta = h01.copy(), h01.conj().T.copy()
    for _ in range(maxit):
        g = inv(Ez - eps)
        eps_s = eps_s + alpha @ g @ beta
        eps   = eps + alpha @ g @ beta + beta @ g @ alpha
        alpha, beta = alpha @ g @ alpha, beta @ g @ beta
        if np.abs(alpha).max() < tol:
            break
    return inv(Ez - eps_s)


# ----------------------------------------------------------------------------
# 3.-6.  One energy point: G, Gamma, A, T, Gn
# ----------------------------------------------------------------------------
def negf_at_energy(E, H, U, mu1, mu2):
    """All the NEGF quantities at one energy -- this is THE core of the method."""
    Np = H.shape[0]
    sig1 = np.zeros((Np, Np), complex); sig1[0, 0]   = lead_self_energy(E, U[0])
    sig2 = np.zeros((Np, Np), complex); sig2[-1, -1] = lead_self_energy(E, U[-1])
    G    = inv((E + zplus) * np.eye(Np) - H - sig1 - sig2)      # retarded GF
    gam1 = 1j * (sig1 - sig1.conj().T)                           # broadening, contact 1
    gam2 = 1j * (sig2 - sig2.conj().T)                           # broadening, contact 2
    A    = 1j * (G - G.conj().T)                                 # spectral function
    T    = np.real(np.trace(gam1 @ G @ gam2 @ G.conj().T))       # transmission
    f1, f2 = fermi(E, mu1), fermi(E, mu2)
    Gn   = G @ (gam1 * f1 + gam2 * f2) @ G.conj().T              # electron correlation fn
    return dict(G=G, A=A, T=T, Gn=Gn, f1=f1, f2=f2)


# ----------------------------------------------------------------------------
# 7. Sweep energy: transmission spectrum, LDOS map, density and current
# ----------------------------------------------------------------------------
def sweep(U, mu1, mu2, E_grid, spin=2):
    """
    Integrate over energy.  Returns T(E), LDOS(x,E), electron density n(x) [per site]
    and current I [A].   n(x) = spin * Int dE  Gn(x,x)/(2 pi)   (Datta eq. 9.5.x)
                        I    = spin * (q/h) Int dE T(E) (f1 - f2)   (Landauer)
    """
    H = hamiltonian(U)
    Np = len(U)
    dE = E_grid[1] - E_grid[0]
    # Midpoint rule: the 1-D density of states has a 1/sqrt(E-Ec) van Hove
    # singularity at the band edge; sampling *exactly* at E = Ec gives a huge
    # spurious contribution.  Shifting by dE/2 makes the rule robust.
    E_grid = E_grid + 0.5 * dE
    T = np.zeros_like(E_grid); LDOS = np.zeros((len(E_grid), Np)); n = np.zeros(Np); I = 0.0
    for k, E in enumerate(E_grid):
        r = negf_at_energy(E, H, U, mu1, mu2)
        T[k] = r['T']
        LDOS[k] = np.real(np.diag(r['A'])) / (2 * np.pi)
        n += spin * np.real(np.diag(r['Gn'])) / (2 * np.pi) * dE
        I += spin * (q * q / (2 * np.pi * hbar)) * r['T'] * (r['f1'] - r['f2']) * dE
    return T, LDOS, n, I


# ----------------------------------------------------------------------------
# 8. Self-consistent Poisson (Datta 2000, Superlattices & Microstructures 28, 253)
#    1-D:  d^2U/dx^2 = -(q^2/eps) (N_D - n)/a      with U fixed in the contacts
#    Gummel damping:  n is updated as n exp((U_old - U_new)/kT) inside the
#    Newton step so the iteration is stable (same trick as dpnegf poisson_init).
# ----------------------------------------------------------------------------
A_CS = (5e-9) ** 2          # cross-section area of the wire used to turn a 1-D
                            # line density into a 3-D charge density (5 nm x 5 nm)

def poisson_1d_gummel(N_D, n_old, U_old, U_bc_left, U_bc_right, eps_r=11.7):
    """
    One Gummel/Newton step of the 1-D nonlinear Poisson equation  (U in eV):
        d2U/dx2 = (q/eps) * (N_D - n)/(a*A_CS)          (electron energy U = -q*phi)
    The electron density is linearised as   n(U) ~ n_old * exp(-(U - U_old)/kT)
    (thermal equilibrium response) so that the Newton step is well damped.
    This is exactly the trick used in device simulators (and in dpnegf's
    poisson_init.py, solve_poisson_NRcycle) to make NEGF-Poisson converge.
    """
    eps = eps_r * 8.85e-12
    Np = len(n_old)
    c = (q / eps) * a**2 / (a * A_CS)                 # [eV] per (electron per site)
    U = U_old.copy()
    for _ in range(20):                               # Newton iterations
        n_lin = n_old * np.exp(np.clip(-(U - U_old) / kT, -20, 20))
        F = np.zeros(Np); J = np.zeros((Np, Np))
        # interior: (U[i-1] - 2U[i] + U[i+1]) - c*(N_D - n(U)) = 0
        for i in range(1, Np - 1):
            F[i] = U[i - 1] - 2 * U[i] + U[i + 1] - c * (N_D[i] - n_lin[i])
            J[i, i - 1] = 1; J[i, i + 1] = 1
            J[i, i] = -2 - c * n_lin[i] / kT          # d/dU of  +c*n(U)
        F[0] = U[0] - U_bc_left;   J[0, 0] = 1
        F[-1] = U[-1] - U_bc_right; J[-1, -1] = 1
        dU = np.linalg.solve(J, -F)
        dU = np.clip(dU, -2 * kT, 2 * kT)             # damped Newton: never jump more than 2kT
        U += dU
        if np.abs(dU).max() < 1e-7:
            break
    return U

def self_consistent_resistor(Np=60, V=0.05, n_iter=60, mix=0.7, verbose=True):
    """
    Datta's toy 'n++ / n+ / n++' resistor (Superlattices & Microstructures 28,
    253 (2000)): uniform wire with a lightly doped middle third.  Doping is a
    fixed positive background charge N_D(x) (electrons per site).
    Returns U(x), n(x), N_D(x), I, E_grid.
    """
    mu1, mu2 = 0.0, -V                       # source / drain Fermi levels (eV)
    Ec = -0.25                               # band bottom in the n++ contacts (Ef - Ec = 0.25 eV)
    E_grid = np.linspace(Ec - 0.1, 0.6, 400)
    # doping = equilibrium density of a *uniform* n++ wire whose band bottom is Ec
    _, _, n_uniform, _ = sweep(np.full(Np, Ec), mu1, mu1, E_grid)
    N_D = n_uniform.copy()
    N_D[Np // 3: 2 * Np // 3] *= 0.3         # n+ middle third: 30 % of the contact doping
    U = Ec - V * np.arange(Np) / (Np - 1)    # initial guess: linear drop
    for it in range(n_iter):
        E_grid = np.linspace(U.min() - 0.1, 0.6, 400)          # always cover the band bottom
        _, _, n, I = sweep(U, mu1, mu2, E_grid)               # NEGF: U -> n
        U_new = poisson_1d_gummel(N_D, n, U, Ec, Ec - V)      # Poisson: n -> U
        dU = np.abs(U_new - U).max()
        U = U + mix * (U_new - U)
        if verbose and (it % 5 == 0 or dU < 1e-4):
            print(f"   SCF iter {it:3d}   max|dU| = {dU:.2e} eV   I = {I*1e6:.3f} uA   barrier = {U.max()-Ec:.3f} eV")
        if dU < 1e-4:
            break
    return U, n, N_D, I, E_grid


# ----------------------------------------------------------------------------
# 9. Phase-breaking: Buettiker-probe / "D" model of Datta ch. 10
#    Sigma_s(E)    = D * G(E)          (element-wise, momentum & phase relaxing)
#    Sigma_s^in(E) = D * Gn(E)
#    Iterate at each energy until G and Gn are self-consistent.
# ----------------------------------------------------------------------------
def dephased_transmission(U, mu1, mu2, E, D, n_iter=50):
    Np = len(U); H = hamiltonian(U)
    sig1 = np.zeros((Np, Np), complex); sig1[0, 0]   = lead_self_energy(E, U[0])
    sig2 = np.zeros((Np, Np), complex); sig2[-1, -1] = lead_self_energy(E, U[-1])
    gam1 = 1j * (sig1 - sig1.conj().T); gam2 = 1j * (sig2 - sig2.conj().T)
    f1, f2 = fermi(E, mu1), fermi(E, mu2)
    sig_s = np.zeros((Np, Np), complex); sig_in_s = np.zeros((Np, Np), complex)
    for _ in range(n_iter):
        G  = inv((E + zplus) * np.eye(Np) - H - sig1 - sig2 - sig_s)
        Gn = G @ (gam1 * f1 + gam2 * f2 + sig_in_s) @ G.conj().T
        sig_s_new, sig_in_new = D * G, D * Gn                     # D scalar here
        if np.abs(sig_s_new - sig_s).max() < 1e-9:
            break
        sig_s, sig_in_s = sig_s_new, sig_in_new
    A = 1j * (G - G.conj().T)
    # effective transmission from the current at contact 1:  I1 ~ Tr[Sigma1^in A - Gamma1 Gn]
    T_eff = np.real(np.trace(gam1 * f1 @ A - gam1 @ Gn)) / (f1 - f2 + 1e-30)
    return T_eff


# ============================================================================
#  DEMOS  (each one is a figure you would find in the book)
# ============================================================================
def demo1_uniform_wire_and_sancho_rubio():
    print("\n=== Demo 1: uniform wire -> T(E) = 1 inside the band; analytic vs numeric Sigma ===")
    Np = 40; U = np.zeros(Np)
    E_grid = np.linspace(-0.5, 4 * t0 + 0.5, 300)
    t1 = time.time(); T, LDOS, n, I = sweep(U, 0.2, 0.2, E_grid); dt = time.time() - t1
    inband = (E_grid > 0) & (E_grid < 4 * t0)
    print(f"   band spans 0 .. 4 t0 = {4*t0:.3f} eV ;  <T> in band = {T[inband].mean():.6f}  (expected 1)")
    print(f"   sweep of {len(E_grid)} energies, Np={Np}: {dt:.2f} s")
    # analytic vs Lopez-Sancho
    E = 1.0
    g_s = numeric_sancho_rubio(E, np.array([[2 * t0]]), np.array([[-t0]]))
    sig_num = (-t0) * g_s[0, 0] * (-t0)
    sig_ana = lead_self_energy(E, 0.0)
    print(f"   Sigma analytic  = {sig_ana:.6f}")
    print(f"   Sigma Lopez-Sancho (dpnegf surface_green style) = {sig_num:.6f}")
    return E_grid, T

def demo2_barrier(plot=True):
    print("\n=== Demo 2: single potential barrier -> tunnelling + resonances, LDOS map ===")
    Np = 100; U = np.zeros(Np); U[40:60] = 0.4          # 6-nm-wide, 0.4-eV barrier
    E_grid = np.linspace(0.0, 1.0, 400)
    t1 = time.time(); T, LDOS, n, I = sweep(U, 0.2, 0.2, E_grid); dt = time.time() - t1
    print(f"   T(E=0.2 eV, below barrier) = {np.interp(0.2, E_grid, T):.3e}  (tunnelling)")
    print(f"   T(E=0.8 eV, above barrier) = {np.interp(0.8, E_grid, T):.3f}")
    print(f"   {len(E_grid)} energies x Np={Np}: {dt:.2f} s")
    if plot:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(14, 4))
        ax[0].plot(np.arange(Np) * a * 1e9, U); ax[0].set_xlabel("x (nm)"); ax[0].set_ylabel("U (eV)"); ax[0].set_title("Potential")
        ax[1].semilogy(E_grid, T); ax[1].set_xlabel("E (eV)"); ax[1].set_ylabel("T(E)"); ax[1].set_title("Transmission")
        im = ax[2].imshow(LDOS, origin="lower", aspect="auto", extent=[0, Np * a * 1e9, E_grid[0], E_grid[-1]], cmap="hot")
        ax[2].set_xlabel("x (nm)"); ax[2].set_ylabel("E (eV)"); ax[2].set_title("LDOS(x,E) = A(x,x;E)/2pi"); fig.colorbar(im, ax=ax[2])
        fig.tight_layout(); fig.savefig("level1_demo2_barrier.png", dpi=130); print("   saved level1_demo2_barrier.png")
    return E_grid, T

def demo3_IV():
    print("\n=== Demo 3: Landauer I-V of the barrier device ===")
    Np = 100; U0 = np.zeros(Np); U0[40:60] = 0.3
    Vs = np.linspace(0, 0.3, 7); Is = []
    E_grid = np.linspace(-0.2, 1.0, 300)
    t1 = time.time()
    for V in Vs:
        U = U0 - V * np.arange(Np) / (Np - 1)      # linear drop (non-self-consistent)
        _, _, _, I = sweep(U, 0.2, 0.2 - V, E_grid)
        Is.append(I)
    print(f"   {len(Vs)} bias points: {time.time()-t1:.2f} s")
    for V, I in zip(Vs, Is):
        print(f"   V = {V:.2f} V   I = {I*1e6:8.3f} uA")
    G0 = 2 * q * q / (2 * np.pi * hbar)
    print(f"   low-bias conductance G = {Is[1]/Vs[1]/G0:.3f} G0   (G0 = 2e^2/h = 77.5 uS)")
    return Vs, np.array(Is)

def demo4_self_consistent(plot=True):
    print("\n=== Demo 4: Datta's n++/n+/n++ resistor, self-consistent NEGF + Poisson ===")
    t1 = time.time()
    U, n, N_D, I, E_grid = self_consistent_resistor(Np=60, V=0.05)
    print(f"   converged in {time.time()-t1:.1f} s ;  I = {I*1e6:.3f} uA at 50 mV")
    if plot:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        x = np.arange(len(U)) * a * 1e9
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(x, U, label="U(x) self-consistent"); ax[0].axhline(0, ls="--", c="k", lw=.7, label="mu1"); ax[0].axhline(-0.05, ls=":", c="k", lw=.7, label="mu2")
        ax[0].set_xlabel("x (nm)"); ax[0].set_ylabel("eV"); ax[0].legend(); ax[0].set_title("Conduction band edge")
        ax[1].plot(x, n / a * 1e-9, label="n(x)"); ax[1].plot(x, N_D / a * 1e-9, "--", label="N_D(x)")
        ax[1].set_xlabel("x (nm)"); ax[1].set_ylabel("electrons / nm"); ax[1].legend(); ax[1].set_title("Density vs doping")
        fig.tight_layout(); fig.savefig("level1_demo4_resistor.png", dpi=130); print("   saved level1_demo4_resistor.png")

def demo5_dephasing():
    print("\n=== Demo 5: dephasing (Buettiker-probe 'D' model): resonance broadening ===")
    Np = 60; U = np.zeros(Np); U[20] = 1.0; U[40] = 1.0     # double barrier -> resonant level
    E_grid = np.linspace(0.02, 0.4, 150)
    T_coh, _, _, _ = sweep(U, 0.1, 0.1, E_grid)
    k = np.argmax(T_coh); E_res = E_grid[k]
    print(f"   coherent resonance at E = {E_res:.4f} eV, T_peak = {T_coh[k]:.3f}")
    for D in (1e-4, 1e-3, 1e-2):
        Td = dephased_transmission(U, 0.1, 0.1 - 1e-3, E_res, D)
        print(f"   D = {D:.0e} eV^2 :  T_eff(E_res) = {Td:.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Level-1 NEGF (Datta style)")
    p.add_argument("--demo", type=int, default=0, help="1..5, or 0 for all")
    p.add_argument("--noplot", action="store_true")
    args = p.parse_args()
    print(f"t0 = hbar^2/(2 m a^2) = {t0:.4f} eV   (a = {a*1e9:.1f} nm, m* = 0.25 m0)")
    T0 = time.time()
    demos = {1: demo1_uniform_wire_and_sancho_rubio, 2: lambda: demo2_barrier(not args.noplot),
             3: demo3_IV, 4: lambda: demo4_self_consistent(not args.noplot), 5: demo5_dephasing}
    for k in ([args.demo] if args.demo else sorted(demos)):
        demos[k]()
    print(f"\nTotal wall time: {time.time()-T0:.1f} s")
