#!/usr/bin/env python3
"""
=============================================================================
 LEVEL 3  --  Multi-band k.p NEGF for hole transport in a Si nanowire (p-FET)
=============================================================================

Six-band Luttinger-Kohn (LK) k.p Hamiltonian (heavy-hole, light-hole and
split-off bands with spin-orbit coupling), discretised by finite differences
on a 3-D grid and solved with the *same* NEGF machinery as Level 2
(Sancho-Rubio surface Green function + recursive Green function), plus the
coupled-mode-space (CMS) reduction that makes multi-band k.p NEGF affordable
(cf. Shin, "NEGF simulation of nanowire FETs using the eight-band k.p method",
IWCE 2009; Luisier & Klimeck, PRB 80, 155430 (2009)).

Contents
  1. LK 6x6 bulk Hamiltonian  H(k)  and its decomposition into the six
     quadratic monomials  kx^2, ky^2, kz^2, kx ky, kx kz, ky kz   (Chuang,
     "Physics of Optoelectronic Devices", eq. 4.5.x; Si parameters from
     Lawaetz / nextnano database:  g1 = 4.285, g2 = 0.339, g3 = 1.446,
     Delta_so = 44 meV).  Bulk effective masses are checked numerically.
  2. Finite-difference discretisation.  k_i k_j -> -(1/2)(d_i d_j + d_j d_i)
     with central differences; the result is exactly Hermitian and block-
     tridiagonal along the transport direction x with block size 6*Ny*Nz.
     The inter-slice coupling V is now a *matrix* (kx ky, kx kz terms).
  3. Wire subband structure E(kx) from the slice blocks (validation).
  4. Coupled mode space: keep the M highest valence modes of the reference
     slice, project Hd, V, U.  Compared against real space.
  5. NEGF transport: T(E), LDOS, I(V_g) for a gate-controlled channel with a
     smooth (non-self-consistent) barrier; p++ leads with the Fermi level
     fixed by the hole density (contour integration, as in Level 2).

Requirements  numpy, scipy, matplotlib + negf_level2_gaa_transiesta_style.py
              (imported for sancho_rubio, rgf, equilibrium_contour, fermi).
Run           python negf_level3_kp_nanowire.py            (~40 s)
Hardware      laptop.  Real-space validation uses 6x6x6 -> 216 orbitals/slice;
              production runs use the mode space (M ~ 64).
=============================================================================
"""
import argparse, time
import numpy as np
from numpy.linalg import inv, eigh
from negf_level2_gaa_transiesta_style import sancho_rubio, rgf, equilibrium_contour, fermi, HBAR, Q, M0, KB

E0 = HBAR**2 / (2 * M0) / Q * 1e18          # hbar^2/(2 m0) in eV nm^2  (= 0.0381 eV nm^2)

# --------------------------------------------------------------------------- 1. LK Hamiltonian
MATERIALS = {   # gamma1, gamma2, gamma3, Delta_so (eV)
    "Si": (4.285, 0.339, 1.446, 0.044),
    "Ge": (13.38, 4.24, 5.69, 0.290),
}

def lk6(kx, ky, kz, mat="Si"):
    """6x6 Luttinger-Kohn Hamiltonian (eV) in the |3/2,+-3/2>,|3/2,+-1/2>,|1/2,+-1/2>
    basis; k in 1/nm.  Valence-band maximum at E = 0 (electron energies, bands go down)."""
    g1, g2, g3, D = MATERIALS[mat]
    P = E0 * g1 * (kx**2 + ky**2 + kz**2)
    Qm = E0 * g2 * (kx**2 + ky**2 - 2 * kz**2)
    R = E0 * np.sqrt(3) * (-g2 * (kx**2 - ky**2) + 2j * g3 * kx * ky)
    S = E0 * 2 * np.sqrt(3) * g3 * (kx - 1j * ky) * kz
    Rc, Sc = np.conj(R), np.conj(S)
    r2, r32 = np.sqrt(2), np.sqrt(1.5)
    H = np.array([
        [P + Qm,    -S,        R,        0,        -S / r2,     r2 * R],
        [-Sc,       P - Qm,    0,        R,        -r2 * Qm,    r32 * S],
        [Rc,        0,         P - Qm,   S,        r32 * Sc,    r2 * Qm],
        [0,         Rc,        Sc,       P + Qm,   -r2 * Rc,    -Sc / r2],
        [-Sc / r2,  -r2 * Qm,  r32 * S,  -r2 * R,  P + D,       0],
        [r2 * Rc,   r32 * Sc,  r2 * Qm,  -S / r2,  0,           P + D]], dtype=complex)
    return -H

def lk_monomials(mat="Si"):
    """Return C0 and the Hermitian 6x6 coefficient matrices of kx^2,ky^2,kz^2,kxky,kxkz,kykz."""
    H = lambda kx, ky, kz: lk6(kx, ky, kz, mat)
    C0 = H(0, 0, 0)
    Cxx = H(1, 0, 0) - C0; Cyy = H(0, 1, 0) - C0; Czz = H(0, 0, 1) - C0
    Cxy = H(1, 1, 0) - C0 - Cxx - Cyy
    Cxz = H(1, 0, 1) - C0 - Cxx - Czz
    Cyz = H(0, 1, 1) - C0 - Cyy - Czz
    return C0, Cxx, Cyy, Czz, Cxy, Cxz, Cyz

def check_bulk_masses(mat="Si"):
    g1, g2, g3, D = MATERIALS[mat]
    k = 0.02
    E001 = np.sort(np.linalg.eigvalsh(lk6(0, 0, k, mat)))[::-1]       # top two = HH (deg 2), next LH
    E111 = np.sort(np.linalg.eigvalsh(lk6(k / np.sqrt(3), k / np.sqrt(3), k / np.sqrt(3), mat)))[::-1]
    m = lambda E: -E0 * k**2 / E
    print(f"   {mat}: m_HH[001] = {m(E001[0]):.3f} (analytic 1/(g1-2g2) = {1/(g1-2*g2):.3f}),  "
          f"m_LH[001] = {m(E001[2]):.3f} (1/(g1+2g2) = {1/(g1+2*g2):.3f}),  "
          f"m_HH[111] = {m(E111[0]):.3f} (1/(g1-2g3) = {1/(g1-2*g3):.3f})")


# --------------------------------------------------------------------------- 2. discretisation
class KPWire:
    """Block-tridiagonal 6-band k.p Hamiltonian of a rectangular nanowire, transport along x."""
    def __init__(self, Ny, Nz, a_nm, mat="Si"):
        self.Ny, self.Nz, self.a, self.mat = Ny, Nz, a_nm, mat
        Ncs = Ny * Nz; self.Ncs = Ncs; self.N = 6 * Ncs
        C0, Cxx, Cyy, Czz, Cxy, Cxz, Cyz = lk_monomials(mat)
        I = np.eye(Ncs)
        Sy = np.kron(np.eye(Ny, k=1), np.eye(Nz))      # shift +1 in y  (index = iy*Nz + iz)
        Sz = np.kron(np.eye(Ny), np.eye(Nz, k=1))      # shift +1 in z
        Dy = (Sy - Sy.T) / (2 * a_nm); Dz = (Sz - Sz.T) / (2 * a_nm)          # anti-Hermitian
        Kyy = (2 * I - Sy - Sy.T) / a_nm**2; Kzz = (2 * I - Sz - Sz.T) / a_nm**2
        Kyz = -Dy @ Dz                                                             # k_y k_z
        # on-slice block (without potential) and coupling to slice i+1
        self.H0 = (np.kron(C0, I) + np.kron(Cxx, 2 / a_nm**2 * I) + np.kron(Cyy, Kyy)
                   + np.kron(Czz, Kzz) + np.kron(Cyz, Kyz))
        self.V = (np.kron(Cxx, -I / a_nm**2) + np.kron(Cxy, -Dy / (2 * a_nm)) + np.kron(Cxz, -Dz / (2 * a_nm)))
        assert np.allclose(self.H0, self.H0.conj().T), "H0 not Hermitian"

    def slice_block(self, U_cs):
        """Add electron potential energy U(y,z) (eV, length Ncs) to all six bands."""
        return self.H0 + np.kron(np.eye(6), np.diag(U_cs))

    def bloch(self, kx):
        return self.H0 + self.V * np.exp(1j * kx * self.a) + self.V.conj().T * np.exp(-1j * kx * self.a)

    def subbands(self, kxs, nb=12):
        return np.array([np.sort(np.linalg.eigvalsh(self.bloch(k)))[::-1][:nb] for k in kxs])


# --------------------------------------------------------------------------- 4. mode space
class ModeSpace:
    """Coupled mode space: Phi = M highest valence eigenvectors of the reference slice block."""
    def __init__(self, wire: KPWire, U_ref_cs, M):
        Hs = wire.slice_block(U_ref_cs)
        w, v = eigh(Hs)
        order = np.argsort(w)[::-1][:M]
        self.Phi = v[:, order]; self.M = M
        self.wire = wire
    def project(self, A):            # A is N x N
        return self.Phi.conj().T @ A @ self.Phi
    def slice_block(self, U_cs):
        return self.project(self.wire.slice_block(U_cs))
    @property
    def V(self):
        return self.project(self.wire.V)


# --------------------------------------------------------------------------- 5. transport
class KPDevice:
    def __init__(self, wire, U_profile, U_lead_L, U_lead_R, mode_space=None, eta=1e-5):
        """
        wire       : KPWire
        U_profile  : (Nx, Ncs) electron potential energy on the core
        U_lead_L/R : flat lead potentials (scalar)
        mode_space : ModeSpace or None (real space)
        """
        self.wire, self.eta = wire, eta
        self.Nx = U_profile.shape[0]
        ms = mode_space
        blk = (ms.slice_block if ms else wire.slice_block)
        self.V = ms.V if ms else wire.V
        self.Hd = [blk(U_profile[i]) for i in range(self.Nx)]
        NcsU = np.ones(wire.Ncs)
        self.hL00 = blk(U_lead_L * NcsU); self.hR00 = blk(U_lead_R * NcsU)
        self.n = self.Hd[0].shape[0]
        self.ms = ms

    def sigmas(self, E):
        """
        Left lead occupies slices -1,-2,...; moving *into* it the inter-slice coupling
        is H_{-1,-2} = V^+, and the device couples to it through H_{0,-1} = V^+:
            Sigma_L = H_{0,-1} g_L H_{-1,0} = V^+ g_L V
        Right lead: H_{N-1,N} = V,  Sigma_R = V g_R V^+.
        (For a scalar coupling V = -t I the order is irrelevant; for k.p it is not.)
        """
        Vd = self.V.conj().T
        gL = sancho_rubio(E, self.hL00, Vd)
        gR = sancho_rubio(E, self.hR00, self.V)
        SigL = Vd @ gL @ self.V
        SigR = self.V @ gR @ Vd
        return SigL, SigR

    def transmission(self, Es):
        T = np.zeros(len(Es))
        for k, E in enumerate(Es):
            z = E + 1j * self.eta
            SigL, SigR = self.sigmas(z)
            grd, glc = rgf(z, self.Hd, self.V, SigL, SigR, True)
            GamL = 1j * (SigL - SigL.conj().T); GamR = 1j * (SigR - SigR.conj().T)
            T[k] = np.real(np.trace(GamL @ glc[0] @ GamR @ glc[0].conj().T))
        return T

    def ldos_x(self, Es):
        out = np.zeros((len(Es), self.Nx))
        for k, E in enumerate(Es):
            z = E + 1j * self.eta
            SigL, SigR = self.sigmas(z)
            grd, _ = rgf(z, self.Hd, self.V, SigL, SigR, False)
            out[k] = [-np.imag(np.trace(g)) / np.pi for g in grd]
        return out

    def current(self, Es, T, mu_L, mu_R, kT):
        """Electron-picture Landauer current through the valence band (holes), A."""
        fL = fermi(Es, mu_L, kT).real; fR = fermi(Es, mu_R, kT).real
        return Q / (2 * np.pi * HBAR) * Q * np.trapezoid(T * (fL - fR), Es)     # spin is inside the 6 bands


def lead_fermi_level_holes(wire, ms, N_A_cm3, kT, a_nm):
    """
    Hole density in a semi-infinite p++ lead with flat potential U=0 as a function
    of mu; bisection so that p = N_A a^3 per grid cell.  p = (1/pi) Int (1-f) Im[Tr g_bulk] dE
    computed with the Level-2 complex contour applied to (1-f) = f(-E) mirror trick.
    Returns mu (eV) relative to the valence band top of the lead (which is at ~0).
    """
    blk = (ms.slice_block if ms else wire.slice_block)
    h00 = blk(np.zeros(wire.Ncs)); V = ms.V if ms else wire.V
    n_cell = N_A_cm3 * 1e6 * (a_nm * 1e-9)**3 * wire.Ncs      # holes per slice
    def holes(mu):
        # holes = states above mu: p = -1/pi Im Int (1 - f(E)) G(E) dE.  Use E -> -E mirror:
        # define Hm = -H so that (1-f_mu(E)) = f_{-mu}(-E); integrate f over the mirrored system.
        h00m = -h00; Vm = -V
        z, w = equilibrium_contour(-mu, kT, -0.8, 20, 12, 8)         # bands of -H lie above -0.1
        tot = 0.0
        for zk, wk in zip(z, w):
            g = sancho_rubio(zk, h00m, Vm.conj().T); g2 = sancho_rubio(zk, h00m, Vm)
            G = inv(zk * np.eye(h00.shape[0]) - h00m - Vm @ g @ Vm.conj().T - Vm.conj().T @ g2 @ Vm)
            tot += wk * np.trace(G)
        return -np.imag(tot) / np.pi
    lo, hi = -0.6, 0.3
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if holes(mid) > n_cell: lo = mid       # more holes -> mu too low -> raise
        else: hi = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------- main demo
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--W", type=float, default=2.4, help="wire width (nm)")
    ap.add_argument("--a", type=float, default=0.4, help="grid spacing (nm)")
    ap.add_argument("--L", type=float, default=12.0, help="device length (nm)")
    ap.add_argument("--M", type=int, default=64, help="number of modes in mode space")
    ap.add_argument("--mat", default="Si", choices=list(MATERIALS))
    ap.add_argument("--nE", type=int, default=120)
    ap.add_argument("--NA", type=float, default=5e20, help="acceptor density of the p++ leads (cm^-3)")
    ap.add_argument("--noplot", action="store_true")
    args = ap.parse_args()
    T0 = time.time()
    kT = KB * 300 / Q
    print("=== 1. LK bulk check ===")
    check_bulk_masses(args.mat)

    Ny = Nz = int(round(args.W / args.a)); Nx = int(round(args.L / args.a))
    wire = KPWire(Ny, Nz, args.a, args.mat)
    print(f"\n=== 2. Wire: {args.W}x{args.W} nm, a={args.a} nm -> {Ny}x{Nz} points, 6 bands -> {wire.N} orbitals/slice, {Nx} slices ===")

    print("\n=== 3. Subband structure E(kx) (top 8 valence subbands) ===")
    kxs = np.linspace(0, np.pi / args.a, 60)
    t1 = time.time(); Eb = wire.subbands(kxs, 8); print(f"   {time.time()-t1:.1f} s ;  top subband at k=0: {Eb[0,0]:.4f} eV (confinement shift below bulk VBM)")

    print(f"\n=== 4. Coupled mode space: convergence of T(E) towards real space (short wire) ===")
    U_flat = np.zeros(wire.Ncs)
    # smooth gate barrier for holes: valence band pushed DOWN in the middle (positive V_g on a p-FET)
    x = np.arange(Nx) * args.a; L = x[-1]
    def profile(barrier):     # electron potential energy (eV); holes see -U
        return -barrier * np.exp(-((x - L / 2) / (0.22 * L))**2)      # gaussian dip of the VB in channel
    Nx_short = 12
    Uprof = np.tile(profile(0.15)[:Nx_short, None], (1, wire.Ncs))
    Es = np.linspace(Eb[0, 0] - 0.35, Eb[0, 0] + 0.02, 30)
    t1 = time.time(); dev_rs = KPDevice(wire, Uprof, 0.0, 0.0, None); T_rs = dev_rs.transmission(Es); t_rs = time.time() - t1
    print(f"   real space ({wire.N} orbitals/slice): {t_rs:.1f} s for {len(Es)} energies")
    for M in sorted({24, 48, args.M}):
        ms_ = ModeSpace(wire, U_flat, M)
        t1 = time.time(); T_ms = KPDevice(wire, Uprof, 0.0, 0.0, ms_).transmission(Es); t_ms = time.time() - t1
        print(f"   mode space M={M:3d}: {t_ms:5.2f} s   max|T_ms - T_rs| = {np.abs(T_rs-T_ms).max():.4f}   mean = {np.abs(T_rs-T_ms).mean():.4f}")
    ms = ModeSpace(wire, U_flat, args.M)
    # uniform wire: T must be an integer = number of propagating subbands
    T_uni = KPDevice(wire, np.zeros((6, wire.Ncs)), 0.0, 0.0, ms).transmission(Es[::7])
    print("   uniform wire T(E) (should be integers):", np.round(T_uni, 3))

    print("\n=== 5. p++ lead Fermi level and gate response (mode space, non-self-consistent) ===")
    N_A = args.NA
    t1 = time.time(); mu = lead_fermi_level_holes(wire, ms, N_A, kT, args.a); print(f"   N_A = {N_A:.0e} cm^-3 -> mu = {mu:+.4f} eV relative to lead VB top (top subband at {Eb[0,0]:.4f}) [{time.time()-t1:.1f}s]")
    V_ds = 0.1
    Es = np.linspace(mu - 0.35, Eb[0, 0] + 0.05, args.nE)
    results = []
    for Vg in (0.0, 0.15, 0.3, 0.45):
        # simple capacitive coupling: channel VB shifted by -eta_g*Vg (holes see a barrier), drain side lowered by V_ds
        Uprof = np.tile(profile(0.8 * Vg)[:, None], (1, wire.Ncs)) - V_ds * (x / L)[:, None]
        dev = KPDevice(wire, Uprof, 0.0, -V_ds, ms)
        t1 = time.time(); T = dev.transmission(Es); I = dev.current(Es, T, mu, mu - V_ds, kT)
        results.append((Vg, T, I))
        print(f"   V_g = {Vg:.2f} V   I_d = {I*1e6:8.4f} uA   ({time.time()-t1:.1f} s for {args.nE} energies)")
    if not args.noplot:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
        for b in range(Eb.shape[1]): ax[0].plot(kxs, Eb[:, b], "k-", lw=.8)
        ax[0].set_xlabel("k_x (1/nm)"); ax[0].set_ylabel("E (eV)"); ax[0].set_title(f"{args.mat} NW valence subbands (6-band k.p)"); ax[0].set_ylim(Eb[0, 0] - 0.5, Eb[0, 0] + 0.02)
        for Vg, T, I in results: ax[1].plot(Es, T, label=f"Vg={Vg:.2f} V, I={I*1e6:.2f} uA")
        ax[1].axvline(mu, c="k", ls="--", lw=.7); ax[1].axvline(mu - V_ds, c="k", ls=":", lw=.7)
        ax[1].set_xlabel("E (eV)"); ax[1].set_ylabel("T(E)"); ax[1].legend(fontsize=8); ax[1].set_title("Hole transmission (mode space)")
        Eld = np.linspace(mu - 0.35, Eb[0, 0] + 0.05, 70)
        ld = dev.ldos_x(Eld)
        im = ax[2].imshow(np.log10(ld + 1e-4), origin="lower", aspect="auto", extent=[x[0], x[-1], Eld[0], Eld[-1]], cmap="inferno")
        ax[2].set_xlabel("x (nm)"); ax[2].set_ylabel("E (eV)"); ax[2].set_title(f"log10 LDOS, Vg={results[-1][0]} V"); fig.colorbar(im, ax=ax[2])
        fig.tight_layout(); fig.savefig("level3_kp_nanowire.png", dpi=130); print("   saved level3_kp_nanowire.png")
    print(f"\nTotal wall time: {time.time()-T0:.1f} s")

if __name__ == "__main__":
    main()
