#!/usr/bin/env python3
"""
=============================================================================
 negf_sinw_final.py  --  finalized NEGF quantum-transport code for silicon
                         nanowire (gate-all-around) transistors
=============================================================================

One self-contained module that combines everything written earlier in this
repository, keeping only implementations that were verified, and fixing the
problems found in the older module:

  source file                                   what was taken from it
  --------------------------------------------  ------------------------------------------
  negf_suite/negf_level2_gaa_transiesta_style   core engine: Sancho-Rubio leads, recursive
                                                Green function, TranSIESTA complex contour
                                                + Ozaki poles, adaptive non-equilibrium
                                                density with Brandbyge weighting, 3-D
                                                Poisson with wrap-around gate, SCF, workers
  negf_suite/negf_level3_kp_nanowire            6-band Luttinger-Kohn k.p hole transport,
                                                coupled mode space            (subcommand kp)
  negf_suite/negf_level1_datta_basics           1-D limit (--W equal to --a gives a single
                                                orbital per slice), teaching-level docs
  negf_silicon_nanowire.py (old)                material database, anisotropic valley masses,
                                                bond (local) current, spectral function /
                                                LDOS, Hamiltonian export, non-self-consistent
                                                quick mode, I-V / output / material sweeps
  run_gaa_simulation.py (old)                   transfer, output and material-comparison
                                                workflows                     (subcommands)

Problems of the older module that are NOT carried over (each reproduced by a
numerical experiment before this file was written):
  * block-RGF electron correlation G^n was wrong by ~100 % against a dense
    inverse   -> here G^n is built from the lead-resolved spectral functions,
    checked against a dense inverse in the self-test;
  * the GAA "PoissonSolver" was a hard-coded barrier model (0.3 eV barrier,
    0.25 V threshold, screening x 0.01) -> here a real 3-D finite-volume
    Poisson equation with the gate as a boundary condition;
  * negative electron density and a 0.09 eV band step between leads and
    device -> here the lead band offset comes from charge neutrality and the
    device potential is continuous with it;
  * drain Fermi level sign reversed (negative drain current for positive V_ds)
    -> here mu_S = 0, mu_D = -q V_ds, positive I_d for an n-FET;
  * bond current off by 10^18 (missing 1/2pi, energy units)
    -> here I_{i->i+1} = -g (q/h) Int 2 Im Tr[H_{i,i+1} G^n_{i+1,i}] dE,
    equal to the Landauer current to 1e-5 in the self-test;
  * fixed real-axis energy grids with eta = 1e-6 miss narrow resonances
    -> adaptive Gauss-Kronrod quadrature on the real axis, contours elsewhere.

-----------------------------------------------------------------------------
PHYSICAL MODEL
  * Conduction band in the effective-mass approximation, finite differences
    on a grid of spacing a.  Si: one isotropic valley (m* = 0.26, fast) or the
    three Delta-valley pairs of a [100] wire (m_l = 0.916, m_t = 0.19).  Other
    materials (Ge, InGaAs, GaAs) with a single isotropic valley.  Spin 2.
  * Cross-section: square (W x W) or circular (diameter W) Si core, hard-wall
    Si/oxide boundary for the wavefunction, oxide shell t_ox, metal gate all
    around the oxide over the gate length (gate-all-around).
  * n++ source/drain extensions continue as semi-infinite leads; their band
    offset is fixed by charge neutrality at the donor density N_D.
  * Valence band (p-FET, holes): 6-band Luttinger-Kohn k.p (subcommand kp).

ENERGY / SIGN CONVENTIONS (all energies in eV, lengths in nm)
  E = 0 is the source Fermi level.  mu_S = 0, mu_D = -V_ds.
  U = -phi is the electron potential energy; the source lead band edge sits at
  U_lead < 0 (degenerate n++), the drain lead at U_lead - V_ds.
  The gate electrode is at phi_gate = phi_S + V_g - phi_ms (phi_ms = flat-band
  / work-function offset).  I_d > 0 means electrons flow source -> drain.

NUMERICAL METHOD (TranSIESTA-style, see the comments tagged [TS-n])
  [TS-1] lead Fermi level from charge neutrality   neutral_lead_shift()
  [TS-2] lead self-energies, Lopez-Sancho          sancho_rubio(), Lead
  [TS-3] block-tridiagonal Green function (RGF)    rgf()
  [TS-4] equilibrium density, complex contour      equilibrium_contour(), ozaki_contour()
  [TS-5] non-equilibrium density, real axis        Device.density()
  [TS-6] Hartree potential with gate (3-D Poisson) Poisson3D
  [TS-7] self-consistent loop                      scf()
  [TS-8] transmission, current, LDOS, bond current post_process()

REQUIREMENTS   python >= 3.9, numpy, scipy, matplotlib.  No compiler, no GPU.
USAGE
  python negf_sinw_final.py selftest                    # 8 identity checks, ~15 s
  python negf_sinw_final.py bias --quick                # one SCF bias point, ~2 min
  python negf_sinw_final.py bias --shape circle --W 3 --Vg 0.6 --workers 4
  python negf_sinw_final.py idvg --quick --workers 4    # transfer curve
  python negf_sinw_final.py idvd --quick --workers 4    # output curves
  python negf_sinw_final.py materials --quick --no-scf  # Si / Ge / InGaAs / GaAs
  python negf_sinw_final.py kp                          # 6-band k.p hole transport
  python negf_sinw_final.py export-h --quick            # device Hamiltonian to disk
  python negf_sinw_final.py <cmd> --help                # all options
Measured on a 4-core Xeon 2.1 GHz: see README_NEGF_FINAL.md.
=============================================================================
"""
import os

# Small dense matrices + process-level parallelism: threaded BLAS only hurts, and with
# several worker processes it slowed one Green-function evaluation ~300x.  Must be set
# before numpy is imported.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict, replace
from multiprocessing import Pool

import numpy as np
from numpy.linalg import inv, eigh
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.integrate import quad_vec
from scipy.linalg import eigh_tridiagonal

# =============================================================================
# 0. Constants, Fermi function, materials
# =============================================================================
HBAR = 1.054571817e-34      # J s
Q = 1.602176634e-19         # C
M0 = 9.1093837015e-31       # kg
KB = 1.380649e-23           # J/K
EPS0 = 8.8541878128e-12     # F/m
G0 = 2 * Q**2 / (2 * np.pi * HBAR)            # 2e^2/h = 77.48 uS
E0_NM = HBAR**2 / (2 * M0) / Q * 1e18         # hbar^2/(2 m0) = 0.0381 eV nm^2
Q_OVER_H_EV = Q / (2 * np.pi * HBAR) * Q      # q/h times (eV -> J): I[A] = g*this*Int T dE[eV]


def fermi(E, mu, kT):
    """Fermi function for real or complex E (complex needed on the contour), scalar or array."""
    scalar = np.ndim(E) == 0
    x = np.atleast_1d(np.asarray((np.asarray(E) - mu) / kT, dtype=complex))
    out = np.empty_like(x)
    xr = x.real
    big, small = xr > 40, xr < -40
    mid = ~(big | small)
    out[big], out[small] = 0.0, 1.0
    out[mid] = 1.0 / (1.0 + np.exp(x[mid]))
    return complex(out[0]) if scalar else out


def fermi_r(E, mu, kT):
    """Real Fermi function (real E)."""
    return np.real(fermi(E, mu, kT))


# Electron (conduction-band) materials.  valleys: list of (m_x, m_y, m_z, degeneracy);
# x is the transport direction.  Ge and III-V are single isotropic valleys (a
# simplification for Ge, whose minimum is at L).
ELECTRON_MATERIALS = {
    "Si": dict(eps=11.7, Eg=1.12, valleys={
        "single": [(0.26, 0.26, 0.26, 1)],
        "si3": [(0.916, 0.19, 0.19, 2), (0.19, 0.916, 0.19, 2), (0.19, 0.19, 0.916, 2)]}),
    "Ge": dict(eps=16.0, Eg=0.66, valleys={"single": [(0.12, 0.12, 0.12, 1)]}),
    "InGaAs": dict(eps=13.9, Eg=0.74, valleys={"single": [(0.041, 0.041, 0.041, 1)]}),
    "GaAs": dict(eps=12.9, Eg=1.42, valleys={"single": [(0.067, 0.067, 0.067, 1)]}),
}
# Valence band: Luttinger parameters gamma1, gamma2, gamma3 and spin-orbit Delta_so (eV)
HOLE_MATERIALS = {"Si": (4.285, 0.339, 1.446, 0.044), "Ge": (13.38, 4.24, 5.69, 0.290)}


# =============================================================================
# 1. Parameters
# =============================================================================
@dataclass
class NWParams:
    # geometry (nm)
    shape: str = "square"    # "square" (W x W) or "circle" (diameter W)
    W: float = 2.4           # Si core width / diameter
    t_ox: float = 1.0        # gate oxide thickness
    L_s: float = 4.0         # source extension (n++)
    L_g: float = 8.0         # gate length (undoped channel)
    L_d: float = 4.0         # drain extension (n++)
    a: float = 0.3           # grid spacing for H and Poisson
    # material
    material: str = "Si"
    valleys: str = "single"  # "single" or "si3" (Si only)
    eps_ox: float = 3.9
    N_D: float = 1e20        # donors in S/D and leads (cm^-3)
    T: float = 300.0
    # bias
    V_g: float = 0.4
    V_ds: float = 0.3
    phi_ms: float = 0.45     # flat-band offset (V): phi_gate = phi_S + V_g - phi_ms
    # numerics
    eta: float = 1e-4        # real-axis broadening (eV)
    n_circle: int = 24
    n_line: int = 12
    n_pole: int = 8
    neq_tol: float = 1e-5    # abs. tolerance of the real-axis density integral (electrons/site)
    density_method: str = "contour"   # "contour" (TranSIESTA) or "ozaki" (dpnegf)
    ozaki_M: int = 60
    scf: bool = True         # False: analytic potential (fast exploration)
    scf_tol: float = 2e-3    # V
    scf_maxiter: int = 40
    mixing: float = 0.6
    n_workers: int = 1

    @property
    def kT(self):
        return KB * self.T / Q

    @property
    def eps_si(self):
        return ELECTRON_MATERIALS[self.material]["eps"]

    @property
    def valley_list(self):
        vs = ELECTRON_MATERIALS[self.material]["valleys"]
        if self.valleys not in vs:
            raise ValueError(f"valley model '{self.valleys}' not available for {self.material}: {list(vs)}")
        return vs[self.valleys]

    @property
    def Nx(self):
        return int(round((self.L_s + self.L_g + self.L_d) / self.a))

    @property
    def gate_window(self):
        return int(round(self.L_s / self.a)), int(round((self.L_s + self.L_g) / self.a))


QUICK = dict(W=1.8, L_s=3.0, L_g=6.0, L_d=3.0, n_circle=16, n_line=9, n_pole=6, scf_maxiter=30)


# =============================================================================
# 2. Cross-section and Hamiltonian blocks
# =============================================================================
def cross_section_mask(p: NWParams):
    """Boolean (N, N) mask of Si grid points in the cross-section."""
    N = max(1, int(round(p.W / p.a)))
    if p.shape == "square" or N == 1:
        return np.ones((N, N), bool)
    if p.shape != "circle":
        raise ValueError("shape must be 'square' or 'circle'")
    c = (np.arange(N) - (N - 1) / 2) * p.a
    Y, Z = np.meshgrid(c, c, indexing="ij")
    return Y**2 + Z**2 <= (p.W / 2) ** 2 + 1e-9


def cross_section_hamiltonian(mask, a_nm, my, mz):
    """2-D finite-difference kinetic energy on the masked Si points (hard wall), eV.
    Site order: C-order over (iy, iz) restricted to the mask (matches Poisson3D)."""
    ty = E0_NM / (my * a_nm**2)
    tz = E0_NM / (mz * a_nm**2)
    Ny, Nz = mask.shape
    lab = -np.ones(mask.shape, int)
    lab[mask] = np.arange(mask.sum())
    H = np.zeros((mask.sum(), mask.sum()))
    for iy in range(Ny):
        for iz in range(Nz):
            i = lab[iy, iz]
            if i < 0:
                continue
            H[i, i] = (2 * ty if Ny > 1 else 0.0) + (2 * tz if Nz > 1 else 0.0)
            if iy + 1 < Ny and lab[iy + 1, iz] >= 0:
                j = lab[iy + 1, iz]; H[i, j] = H[j, i] = -ty
            if iz + 1 < Nz and lab[iy, iz + 1] >= 0:
                j = lab[iy, iz + 1]; H[i, j] = H[j, i] = -tz
    return H


# =============================================================================
# 3. Core NEGF numerics
# =============================================================================
def sancho_rubio(E, h00, h01, eta=0.0, tol=1e-10, maxit=300):
    """[TS-2] Lopez-Sancho decimation: surface Green function of a semi-infinite lead
    with principal-layer Hamiltonian h00 and coupling h01 pointing INTO the lead."""
    N = h00.shape[0]
    Ez = (E + 1j * eta) * np.eye(N)
    eps_s = h00.astype(complex).copy()
    eps = eps_s.copy()
    alpha = h01.astype(complex).copy()
    beta = alpha.conj().T.copy()
    for _ in range(maxit):
        g = inv(Ez - eps)
        agb = alpha @ g @ beta
        eps_s = eps_s + agb
        eps = eps + agb + beta @ g @ alpha
        alpha, beta = alpha @ g @ alpha, beta @ g @ beta
        if np.abs(alpha).max() < tol and np.abs(beta).max() < tol:
            break
    return inv(Ez - eps_s)


def rgf(E, Hd, V, SigL, SigR, need_lastcol=True):
    """[TS-3] Recursive Green function (Anantram, Lundstrom, Nikonov, Proc. IEEE 96,
    1511 (2008)) for block-tridiagonal H with H_{i,i+1} = V, H_{i+1,i} = V^+.
    Returns diagonal blocks G_ii and (optionally) last-column blocks G_{i,K-1}."""
    K = len(Hd)
    I = np.eye(Hd[0].shape[0])
    Vd = V.conj().T
    gL = [None] * K
    gL[0] = inv(E * I - Hd[0] - SigL - (SigR if K == 1 else 0))
    for i in range(1, K):
        D = E * I - Hd[i] - Vd @ gL[i - 1] @ V
        if i == K - 1:
            D = D - SigR
        gL[i] = inv(D)
    grd = [None] * K
    grd[-1] = gL[-1]
    glc = [None] * K if need_lastcol else None
    if need_lastcol:
        glc[-1] = gL[-1]
    for i in range(K - 2, -1, -1):
        gV = gL[i] @ V
        grd[i] = gL[i] + gV @ grd[i + 1] @ Vd @ gL[i]
        if need_lastcol:
            glc[i] = gV @ glc[i + 1]
    return grd, glc


def rgf_first_column(E, Hd, V, SigL, SigR):
    """First-column blocks G_{i,0}, from the RGF of the mirrored device."""
    _, glc_rev = rgf(E, Hd[::-1], V.conj().T, SigR, SigL, True)
    return glc_rev[::-1]


def equilibrium_contour(mu, kT, E_min, n_circle=24, n_line=12, n_pole=8):
    """[TS-4] TranSIESTA-style contour for rho_eq = -1/pi Im Int f(E) G(E) dE.
    Returns (z, w) with rho_eq = -1/pi Im sum_k w_k G(z_k):
      circle from E_min to P = mu - 10 kT + i gamma, line at height gamma = 2 pi kT n_pole
      (where f(E + i gamma) = f(E) is real: split into two Gauss-Legendre panels and a
      Gauss-Laguerre tail), and n_pole Fermi poles with residue weight -2 pi i kT."""
    gamma = 2 * np.pi * kT * n_pole
    P = mu - 10 * kT
    c = ((P**2 + gamma**2) - E_min**2) / (2 * (P - E_min))
    R = c - E_min
    th1 = np.arctan2(gamma, P - c)
    x, wx = np.polynomial.legendre.leggauss(n_circle)
    th = 0.5 * (np.pi - th1) * x + 0.5 * (np.pi + th1)
    wth = 0.5 * (np.pi - th1) * wx
    z_c = c + R * np.exp(1j * th)
    w_c = -fermi(z_c, mu, kT) * 1j * R * np.exp(1j * th) * wth
    n_a = max(2, n_line // 3); n_b = max(2, n_line // 3); n_c = max(2, n_line - n_a - n_b)
    x, wx = np.polynomial.legendre.leggauss(n_a)
    ra = 0.5 * (mu - 3 * kT - P) * x + 0.5 * (mu - 3 * kT + P); wa = 0.5 * (mu - 3 * kT - P) * wx
    x, wx = np.polynomial.legendre.leggauss(n_b)
    rb = 3 * kT * x + mu; wb = 3 * kT * wx
    xl, wl = np.polynomial.laguerre.laggauss(n_c)
    rc = mu + 3 * kT + kT * xl; wc = kT * wl * np.exp(xl)
    re = np.concatenate([ra, rb, rc]); wre = np.concatenate([wa, wb, wc])
    z_l = re + 1j * gamma
    w_l = fermi(z_l, mu, kT) * wre
    nu = np.arange(n_pole)
    z_p = mu + 1j * np.pi * kT * (2 * nu + 1)
    w_p = np.full(n_pole, -2j * np.pi * kT)
    return np.concatenate([z_c, z_l, z_p]), np.concatenate([w_c, w_l, w_p])


def ozaki_poles(M):
    """Ozaki continued-fraction poles/residues (PRB 75, 035123 (2007)), as in dpnegf."""
    N = 2 * M
    off = np.array([0.5 / np.sqrt((2 * n - 1) * (2 * n + 1)) for n in range(1, N)])
    ev, evec = eigh_tridiagonal(np.zeros(N), off, select="i", select_range=(N // 2, N - 1))
    return np.flip(1.0 / ev), np.flip(np.abs(evec[0, :]) ** 2 / (4 * ev**2))


def ozaki_contour(mu, kT, M=60, R_big=1e4):
    """Imaginary-axis alternative to equilibrium_contour() (same return convention)."""
    zeta, Rp = ozaki_poles(M)
    z = np.concatenate([[mu + 1j * R_big], mu + 1j * kT * zeta])
    w = np.concatenate([[-(1j * np.pi / 2) * (1j * R_big)], -2j * np.pi * kT * Rp])
    return z, w


# =============================================================================
# 4. Leads and device
# =============================================================================
class Lead:
    """Semi-infinite lead = periodic repetition of one slice with band offset U_lead.
    Self-energies are cached per energy (leads never change during the SCF)."""

    def __init__(self, H_cs, tx, U_lead=0.0):
        N = H_cs.shape[0]
        self.N = N
        self.h00 = H_cs + (2 * tx + U_lead) * np.eye(N)
        self.h01 = -tx * np.eye(N)
        self._cache = {}

    def self_energy(self, E):
        key = complex(E)
        if key not in self._cache:
            if len(self._cache) > 4000:
                self._cache.clear()
            g = sancho_rubio(E, self.h00, self.h01)
            self._cache[key] = self.h01.conj().T @ g @ self.h01
        return self._cache[key]

    def bulk_gf(self, E):
        s = self.self_energy(E)
        return inv(E * np.eye(self.N) - self.h00 - 2 * s)


class Device:
    """All NEGF quantities of one valley for a given potential U (Nx, N_site)."""

    def __init__(self, p: NWParams, mask, mx, my, mz, g_v, U_lead_S, U_lead_D):
        self.p = p
        self.g = 2 * g_v                                   # spin x valley degeneracy
        a = p.a
        self.tx = E0_NM / (mx * a**2)
        self.H_cs = cross_section_hamiltonian(mask, a, my, mz)
        self.N = self.H_cs.shape[0]
        self.Nx = p.Nx
        self.V = -self.tx * np.eye(self.N)
        self.leadL = Lead(self.H_cs, self.tx, U_lead_S)
        self.leadR = Lead(self.H_cs, self.tx, U_lead_D)
        self.mu_L, self.mu_R = 0.0, -p.V_ds
        self.E_min = min(U_lead_S, U_lead_D) - 1.5
        self.U = None
        self._n_neq_eval = 0

    def set_potential(self, U):
        self.U = np.asarray(U)
        self.Hd = [self.H_cs + np.diag(2 * self.tx + self.U[i]) for i in range(self.Nx)]

    # --- one energy ---------------------------------------------------------------
    def _G(self, z, need_lastcol):
        SigL, SigR = self.leadL.self_energy(z), self.leadR.self_energy(z)
        grd, glc = rgf(z, self.Hd, self.V, SigL, SigR, need_lastcol)
        return grd, glc, SigL, SigR

    def _diagG(self, z):
        grd, _, _, _ = self._G(z, False)
        return np.concatenate([np.diag(g) for g in grd])

    def _neq_point(self, E):
        """Real axis: diagonals of A_L = G Gam_L G^+ and A_R = G Gam_R G^+, and T(E)."""
        grd, glc, SigL, SigR = self._G(E + 1j * self.p.eta, True)
        GamL = 1j * (SigL - SigL.conj().T)
        GamR = 1j * (SigR - SigR.conj().T)
        A_R = np.concatenate([np.real(np.einsum("ij,jk,ik->i", g, GamR, g.conj())) for g in glc])
        A = np.concatenate([np.real(np.diag(1j * (g - g.conj().T))) for g in grd])
        T = np.real(np.trace(GamL @ glc[0] @ GamR @ glc[0].conj().T))
        return A - A_R, A_R, T

    def _bond_point(self, E):
        """Bond-current integrand j_i(E), i = 0..Nx-2 (dimensionless; I_i = g q/h Int j_i dE).
        G^n_{i+1,i} = G_{i+1,0} Gam_L G_{i,0}^+ f_L + G_{i+1,K-1} Gam_R G_{i,K-1}^+ f_R,
        j_i = -2 Im Tr[H_{i,i+1} G^n_{i+1,i}].  Equals T (f_L - f_R) for ballistic transport
        (to O(eta): the device broadening eta acts as a weak particle sink)."""
        z = E + 1j * self.p.eta
        grd, glc, SigL, SigR = self._G(z, True)
        gfc = rgf_first_column(z, self.Hd, self.V, SigL, SigR)
        GamL = 1j * (SigL - SigL.conj().T)
        GamR = 1j * (SigR - SigR.conj().T)
        fL, fR = fermi_r(E, self.mu_L, self.p.kT), fermi_r(E, self.mu_R, self.p.kT)
        j = np.empty(self.Nx - 1)
        for i in range(self.Nx - 1):
            Gn = fL * gfc[i + 1] @ GamL @ gfc[i].conj().T + fR * glc[i + 1] @ GamR @ glc[i].conj().T
            j[i] = -2 * np.imag(np.trace(self.V @ Gn))
        return j

    # --- [TS-4]/[TS-5] density -----------------------------------------------------
    def _eq_points(self, mu):
        p = self.p
        if self.U.min() - 0.2 < self.E_min:
            self.E_min = self.U.min() - 0.5
        if p.density_method == "ozaki":
            return ozaki_contour(mu, p.kT, p.ozaki_M)
        return equilibrium_contour(mu, p.kT, self.E_min, p.n_circle, p.n_line, p.n_pole)

    def density(self):
        """Electrons per site (Nx*N,), times degeneracy g.
        rho = w (rho_eq^L + D^R) + (1-w)(rho_eq^R + D^L),  w = D_R^2/(D_L^2 + D_R^2)."""
        p, kT = self.p, self.p.kT
        with _WorkerPool(self, p.n_workers) as pmap:
            zL, wL = self._eq_points(self.mu_L)
            rho_eq_L = -np.imag(_sum_weighted(pmap, zL, wL)) / np.pi
            if abs(p.V_ds) < 1e-9:
                self._n_neq_eval = 0
                return self.g * rho_eq_L
            zR, wR = self._eq_points(self.mu_R)
            rho_eq_R = -np.imag(_sum_weighted(pmap, zR, wR)) / np.pi
            lo = min(self.mu_L, self.mu_R) - 10 * kT
            hi = max(self.mu_L, self.mu_R) + 10 * kT
            val, _, info = quad_vec(_neq_integrand, lo, hi, epsabs=p.neq_tol, epsrel=1e-4,
                                    norm="max", limit=400, workers=pmap, full_output=True)
            self._n_neq_eval = info.neval
        n = self.Nx * self.N
        dR, dL = val[:n], val[n:]
        w = dR**2 / (dL**2 + dR**2 + 1e-30)
        return self.g * (w * (rho_eq_L + dR) + (1 - w) * (rho_eq_R + dL))

    # --- [TS-8] observables ---------------------------------------------------------
    def transmission(self, Es):
        """T(E) = Tr[Gam_L G Gam_R G^+] per spin, this valley only."""
        with _WorkerPool(self, self.p.n_workers) as pmap:
            return np.array([r[2] for r in pmap(_work_neq, list(Es))])

    def current(self, lo, hi):
        """I = g (q/h) Int T (f_L - f_R) dE  in A (adaptive quadrature)."""
        with _WorkerPool(self, self.p.n_workers) as pmap:
            val, _ = quad_vec(_current_integrand, lo, hi, epsabs=1e-6, epsrel=1e-4, limit=400, workers=pmap)
        return self.g * Q_OVER_H_EV * val

    def bond_current(self, lo, hi):
        """Energy-integrated current through every bond i -> i+1 (A)."""
        with _WorkerPool(self, self.p.n_workers) as pmap:
            val, _ = quad_vec(_bond_integrand, lo, hi, epsabs=1e-6, epsrel=1e-4, norm="max",
                              limit=400, workers=pmap)
        return self.g * Q_OVER_H_EV * val

    def ldos_x(self, Es):
        """LDOS(x, E) summed over the cross-section, times g  (states / eV / slice)."""
        with _WorkerPool(self, self.p.n_workers) as pmap:
            diags = pmap(_work_diag, [E + 1j * self.p.eta for E in Es])
        return self.g * np.array([-np.imag(d).reshape(self.Nx, self.N).sum(1) / np.pi for d in diags])


# --- multiprocessing over energies (fork; each call snapshots the current device) ---
_DEV = None


def _init_worker(dev):
    global _DEV
    _DEV = dev


def _work_diag(z):
    return _DEV._diagG(z)


def _work_neq(E):
    return _DEV._neq_point(E)


def _neq_integrand(E):
    d = _DEV
    A_L, A_R, _ = d._neq_point(E)
    fL, fR = fermi_r(E, d.mu_L, d.p.kT), fermi_r(E, d.mu_R, d.p.kT)
    return np.concatenate([(fR - fL) * A_R, (fL - fR) * A_L]) / (2 * np.pi)


def _current_integrand(E):
    d = _DEV
    return d._neq_point(E)[2] * (fermi_r(E, d.mu_L, d.p.kT) - fermi_r(E, d.mu_R, d.p.kT))


def _bond_integrand(E):
    return _DEV._bond_point(E)


class _WorkerPool:
    def __init__(self, dev, n):
        self.dev, self.n, self.pool = dev, n, None

    def __enter__(self):
        _init_worker(self.dev)
        if self.n > 1:
            self.pool = Pool(self.n, initializer=_init_worker, initargs=(self.dev,))
            return self.pool.map
        return lambda fn, xs: list(map(fn, xs))

    def __exit__(self, *exc):
        if self.pool:
            self.pool.close()
            self.pool.join()


def _sum_weighted(pmap, zs, ws):
    return sum(w * d for w, d in zip(ws, pmap(_work_diag, list(zs))))


# =============================================================================
# 5. [TS-1] lead Fermi level from charge neutrality
# =============================================================================
def neutral_lead_shift(p: NWParams, mask):
    """Band offset U_lead (eV, relative to mu_S = 0) of the n++ lead such that its
    electron density per site equals N_D a^3 (summed over valleys and spin)."""
    nD_site = p.N_D * 1e6 * (p.a * 1e-9) ** 3
    leads = []
    for (mx, my, mz, g) in p.valley_list:
        H_cs = cross_section_hamiltonian(mask, p.a, my, mz)
        leads.append((Lead(H_cs, E0_NM / (mx * p.a**2), 0.0), 2 * g))
    N = mask.sum()

    def n_of_mu(mu):
        n = 0.0
        for lead, g in leads:
            z, w = equilibrium_contour(mu, p.kT, min(-0.5, e_sub - 1.0), p.n_circle, p.n_line, p.n_pole)
            tot = sum(wk * np.trace(lead.bulk_gf(zk)) for zk, wk in zip(z, w))
            n += g * (-np.imag(tot) / np.pi) / N
        return n

    # bracket: start just below the lowest lead subband, expand upward until the lead
    # holds enough electrons (light-mass materials have subbands several eV up).
    # lead band bottom = lowest subband: min eig(h00) - 2 t_x, with h01 = -t_x
    e_sub = min(np.linalg.eigvalsh(lead.h00)[0] + 2 * lead.h01[0, 0].real for lead, _ in leads)
    lo, step = e_sub - 0.3, 0.5
    hi = lo + step
    while n_of_mu(hi) < nD_site:
        lo, hi, step = hi, hi + 2 * step, 2 * step
        if hi > e_sub + 50:
            raise RuntimeError("lead Fermi level search did not converge")
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if n_of_mu(mid) > nD_site:
            hi = mid
        else:
            lo = mid
    return -0.5 * (lo + hi)


# =============================================================================
# 6. [TS-6] 3-D Poisson with gate-all-around boundary
# =============================================================================
class Poisson3D:
    """Finite-volume div(eps grad phi) = -rho on a box holding the Si core, the oxide
    and (for the circle) the gate metal.  Dirichlet: gate over the gate length (outer
    box faces for the square; everything beyond radius W/2 + t_ox for the circle) and
    the two lead planes.  Neumann elsewhere.  Gummel-Newton linearisation
    n(phi) = n_old exp((phi - phi_old)/V_T) with steps damped to 2 V_T."""

    def __init__(self, p: NWParams, mask, phi_S, phi_D):
        self.p = p
        a = p.a * 1e-9
        self.a = a
        Nc = mask.shape[0]
        n_ox = max(1, int(round(p.t_ox / p.a))) + (1 if p.shape == "circle" else 0)
        Nx, Nb = p.Nx, Nc + 2 * n_ox
        self.shape = (Nx, Nb, Nb)
        idx = np.arange(Nx * Nb * Nb).reshape(self.shape)
        core = np.zeros(self.shape, bool)
        core[:, n_ox:n_ox + Nc, n_ox:n_ox + Nc] = mask[None]
        self.core_idx = idx[core]
        self.n_site = int(mask.sum())
        eps = np.where(core, p.eps_si, p.eps_ox) * EPS0
        ix0, ix1 = p.gate_window
        dmask = np.zeros(self.shape, bool)
        dval = np.zeros(self.shape)
        phi_gate = phi_S + p.V_g - p.phi_ms
        if p.shape == "circle":
            c = (np.arange(Nb) - (Nb - 1) / 2) * p.a
            Y, Z = np.meshgrid(c, c, indexing="ij")
            metal = np.sqrt(Y**2 + Z**2) > p.W / 2 + p.t_ox
            dmask[ix0:ix1][:, metal] = True
        else:
            for f in (np.s_[ix0:ix1, 0, :], np.s_[ix0:ix1, -1, :], np.s_[ix0:ix1, :, 0], np.s_[ix0:ix1, :, -1]):
                dmask[f] = True
        dval[dmask] = phi_gate
        dmask[0], dval[0] = True, phi_S
        dmask[-1], dval[-1] = True, phi_D
        self.dir_mask, self.dir_val = dmask.ravel(), dval.ravel()
        rows, cols, vals = [], [], []
        for ax, n_ax in enumerate(self.shape):
            lo = [slice(None)] * 3; hi = [slice(None)] * 3
            lo[ax], hi[ax] = slice(0, n_ax - 1), slice(1, n_ax)
            e1, e2 = eps[tuple(lo)], eps[tuple(hi)]
            ef = (2 * e1 * e2 / (e1 + e2)).ravel() / a**2
            i1, i2 = idx[tuple(lo)].ravel(), idx[tuple(hi)].ravel()
            rows += [i1, i2, i1, i2]; cols += [i2, i1, i1, i2]; vals += [ef, ef, -ef, -ef]
        L = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(idx.size, idx.size))
        keep = (~self.dir_mask).astype(float)
        self.L = (sp.diags(keep) @ L + sp.diags(self.dir_mask.astype(float))).tocsr()
        self.L_raw = L.tocsr()
        nd = np.zeros(self.shape)
        nd_site = p.N_D * 1e6 * a**3
        nd[:ix0][core[:ix0]] = nd_site
        nd[ix1:][core[ix1:]] = nd_site
        self.ND_site = nd.ravel()[self.core_idx]

    def _full(self, v_core):
        out = np.zeros(self.dir_mask.size)
        out[self.core_idx] = v_core
        return out

    def solve(self, n_site, phi_old, V_T):
        a = self.a
        phi = phi_old.copy()
        n_old, nd = self._full(n_site), self._full(self.ND_site)
        core = self._full(np.ones(len(self.core_idx))).astype(bool)
        for _ in range(80):
            n_lin = np.where(core, n_old * np.exp(np.clip((phi - phi_old) / V_T, -20, 20)), 0.0)
            rho = Q * (nd - n_lin) / a**3
            F = self.L @ phi + np.where(self.dir_mask, 0.0, rho)
            F[self.dir_mask] = phi[self.dir_mask] - self.dir_val[self.dir_mask]
            J = self.L + sp.diags(np.where(self.dir_mask, 0.0, -Q * n_lin / (V_T * a**3)))
            dphi = np.clip(spla.spsolve(J.tocsc(), -F), -2 * V_T, 2 * V_T)
            phi = phi + dphi
            if np.abs(dphi).max() < 1e-8:
                break
        return phi

    def residual(self, phi, n_site):
        """Max |div(eps grad phi) + rho| / max|rho| on non-Dirichlet points (for testing)."""
        rho = Q * (self._full(self.ND_site) - self._full(n_site)) / self.a**3
        r = (self.L_raw @ phi + rho)[~self.dir_mask]
        return np.abs(r).max() / max(np.abs(rho).max(), 1e-30)

    def core_potential(self, phi):
        return (-phi[self.core_idx]).reshape(self.p.Nx, self.n_site)


def analytic_potential(p: NWParams, U_lead, n_site):
    """Non-self-consistent U(x): gate barrier (phi_ms - V_g) with 1-nm tanh edges plus a
    linear drain drop across the gate region.  Fast exploration only."""
    x = (np.arange(p.Nx) + 0.5) * p.a
    x0, x1 = p.L_s, p.L_s + p.L_g
    win = 0.5 * (np.tanh((x - x0) / 1.0) - np.tanh((x - x1) / 1.0))
    ramp = np.clip((x - x0) / max(p.L_g, 1e-9), 0, 1)
    U = U_lead + (p.phi_ms - p.V_g) * win - p.V_ds * ramp
    return np.repeat(U[:, None], n_site, axis=1)


# =============================================================================
# 7. [TS-7] Self-consistent loop and [TS-8] post-processing
# =============================================================================
def build_devices(p: NWParams, U_lead=None, verbose=True):
    mask = cross_section_mask(p)
    t0 = time.time()
    if U_lead is None:
        U_lead = neutral_lead_shift(p, mask)
        if verbose:
            print(f"[TS-1] lead N_D={p.N_D:.1e} cm^-3 -> E_F - E_c,lead = {-U_lead:.3f} eV  [{time.time()-t0:.1f}s]",
                  flush=True)
    devs = [Device(p, mask, mx, my, mz, g, U_lead, U_lead - p.V_ds) for (mx, my, mz, g) in p.valley_list]
    return mask, U_lead, devs


def scf(p: NWParams, verbose=True, U_lead=None, phi_init=None):
    """Self-consistent NEGF-Poisson (or the analytic potential if p.scf is False)."""
    t0 = time.time()
    mask, U_lead, devs = build_devices(p, U_lead, verbose)
    phi_S, phi_D = -U_lead, -U_lead + p.V_ds
    pois = Poisson3D(p, mask, phi_S, phi_D)
    if not p.scf:
        U = analytic_potential(p, U_lead, pois.n_site)
        for d in devs:
            d.set_potential(U)
        n = sum(d.density() for d in devs)
        return dict(devs=devs, pois=pois, phi=None, U=U, n=n, U_lead=U_lead, hist=[], time=time.time() - t0)
    if phi_init is None:
        xg = np.arange(pois.shape[0]) / (pois.shape[0] - 1)
        phi = np.broadcast_to((phi_S + (phi_D - phi_S) * xg)[:, None, None], pois.shape).ravel().copy()
    else:
        phi = phi_init.copy()
    phi[pois.dir_mask] = pois.dir_val[pois.dir_mask]
    hist = []
    n = None
    for it in range(p.scf_maxiter):
        t1 = time.time()
        U = pois.core_potential(phi)
        n = np.zeros(p.Nx * pois.n_site)
        for d in devs:
            d.set_potential(U)
            n += d.density()
        phi_new = pois.solve(n, phi, p.kT)
        dphi = np.abs(phi_new - phi).max()
        phi = phi + p.mixing * (phi_new - phi)
        hist.append(float(dphi))
        if verbose:
            nev = sum(d._n_neq_eval for d in devs)
            print(f"   SCF {it:2d}  max|dphi| = {dphi:.2e} V   Ne = {n.sum():8.3f}   neq evals = {nev:4d}"
                  f"   [{time.time()-t1:.1f}s/iter, {time.time()-t0:.0f}s total]", flush=True)
        if dphi < p.scf_tol:
            break
    else:
        if verbose:
            print(f"   WARNING: SCF not converged in {p.scf_maxiter} iterations (last dphi {hist[-1]:.1e} V)")
    U = pois.core_potential(phi)
    for d in devs:
        d.set_potential(U)
    return dict(devs=devs, pois=pois, phi=phi, U=U, n=n, U_lead=U_lead, hist=hist, time=time.time() - t0)


def post_process(p: NWParams, res, n_E=201, bond=True, ldos=True, verbose=True):
    devs = res["devs"]
    kT = p.kT
    lo = min(0.0, -p.V_ds) - 12 * kT
    hi = max(0.0, -p.V_ds) + 12 * kT
    Es = np.linspace(lo, hi, n_E)
    T = sum(d.g / 2 * d.transmission(Es) for d in devs)       # per spin, all valleys
    I = sum(d.current(lo, hi) for d in devs)
    out = dict(E=Es, T=T, I=float(I))
    if bond and abs(p.V_ds) > 1e-9:
        Ib = sum(d.bond_current(lo, hi) for d in devs)
        out["I_bond"] = Ib
        out["bond_dev"] = float(np.abs(Ib - I).max() / max(abs(I), 1e-30))
    if ldos:
        U = res["U"]
        sub0 = min(np.linalg.eigvalsh(d.H_cs)[0] for d in devs)
        Emin = min(U.min() + sub0 - 0.1, -p.V_ds - 0.2)
        out["E_ldos"] = np.linspace(Emin, 0.4, 80)
        out["ldos"] = sum(d.ldos_x(out["E_ldos"]) for d in devs)
        out["Ec_x"] = U.min(axis=1) + sub0
    if verbose:
        msg = f"[TS-8] I_d = {I*1e6:.4f} uA"
        if "bond_dev" in out:
            msg += f"   (bond-current conservation: max rel. deviation {out['bond_dev']:.1e})"
        print(msg, flush=True)
    return out


def run_bias(p: NWParams, verbose=True, **kw):
    res = scf(p, verbose=verbose, U_lead=kw.get("U_lead"), phi_init=kw.get("phi_init"))
    post = post_process(p, res, bond=kw.get("bond", True), ldos=kw.get("ldos", True), verbose=verbose)
    return res, post


def sweep(p: NWParams, name, values, verbose=True, bond=False):
    """Id vs V_g or V_ds, reusing the lead offset and the previous potential."""
    mask = cross_section_mask(p)
    U_lead = neutral_lead_shift(p, mask)
    phi, rows = None, []
    for v in values:
        pp = replace(p, **{name: float(v)})
        t0 = time.time()
        res = scf(pp, verbose=False, U_lead=U_lead, phi_init=phi)
        phi = res["phi"] if p.scf else None
        post = post_process(pp, res, bond=bond, ldos=False, verbose=False)
        rows.append(dict(V_g=pp.V_g, V_ds=pp.V_ds, I=post["I"], scf_iters=len(res["hist"]), time=time.time() - t0))
        if verbose:
            print(f"   V_g = {pp.V_g:+.3f} V   V_ds = {pp.V_ds:.3f} V   I_d = {post['I']*1e6:10.4f} uA"
                  f"   ({len(res['hist'])} SCF iters, {time.time()-t0:.0f} s)", flush=True)
    return rows


# =============================================================================
# 8. Hamiltonian export (ported from the old module, now block-sparse)
# =============================================================================
def device_hamiltonian_sparse(dev: Device):
    K, N = dev.Nx, dev.N
    blocks = [[None] * K for _ in range(K)]
    for i in range(K):
        blocks[i][i] = sp.csr_matrix(dev.Hd[i])
        if i + 1 < K:
            blocks[i][i + 1] = sp.csr_matrix(dev.V)
            blocks[i + 1][i] = sp.csr_matrix(dev.V.conj().T)
    return sp.bmat(blocks, format="csr")


def export_hamiltonian(dev: Device, path, fmt="npz"):
    H = device_hamiltonian_sparse(dev)
    info = dict(size=H.shape[0], nnz=int(H.nnz), orbitals_per_slice=dev.N, slices=dev.Nx,
                hermitian=bool(abs(H - H.conj().T).max() < 1e-12), hopping_x_eV=dev.tx)
    if fmt == "npz":
        sp.save_npz(path, H)
    elif fmt == "mat":
        from scipy.io import savemat
        savemat(path, {"H": H})
    elif fmt == "csv":
        C = H.tocoo()
        np.savetxt(path, np.column_stack([C.row, C.col, C.data.real]), delimiter=",",
                   header="row,col,value_eV", fmt=["%d", "%d", "%.10e"])
    else:
        raise ValueError("fmt must be npz, mat or csv")
    info["file"] = path
    return info


# =============================================================================
# 9. Six-band k.p holes (ported from negf_suite/negf_level3_kp_nanowire.py)
# =============================================================================
def lk6(kx, ky, kz, mat="Si"):
    """6x6 Luttinger-Kohn H (eV), k in 1/nm; VBM at 0, electron-energy sign (bands go down)."""
    g1, g2, g3, D = HOLE_MATERIALS[mat]
    P = E0_NM * g1 * (kx**2 + ky**2 + kz**2)
    Qm = E0_NM * g2 * (kx**2 + ky**2 - 2 * kz**2)
    R = E0_NM * np.sqrt(3) * (-g2 * (kx**2 - ky**2) + 2j * g3 * kx * ky)
    S = E0_NM * 2 * np.sqrt(3) * g3 * (kx - 1j * ky) * kz
    Rc, Sc = np.conj(R), np.conj(S)
    r2, r32 = np.sqrt(2), np.sqrt(1.5)
    H = np.array([
        [P + Qm, -S, R, 0, -S / r2, r2 * R],
        [-Sc, P - Qm, 0, R, -r2 * Qm, r32 * S],
        [Rc, 0, P - Qm, S, r32 * Sc, r2 * Qm],
        [0, Rc, Sc, P + Qm, -r2 * Rc, -Sc / r2],
        [-Sc / r2, -r2 * Qm, r32 * S, -r2 * R, P + D, 0],
        [r2 * Rc, r32 * Sc, r2 * Qm, -S / r2, 0, P + D]], dtype=complex)
    return -H


class KPWire:
    """Block-tridiagonal 6-band k.p Hamiltonian of a square wire, transport along x.
    k_i k_j -> -(1/2)(d_i d_j + d_j d_i) with central differences (exactly Hermitian);
    the inter-slice coupling V is a matrix because of the k_x k_y, k_x k_z terms."""

    def __init__(self, Ny, Nz, a, mat="Si"):
        self.Ny, self.Nz, self.a = Ny, Nz, a
        self.Ncs = Ny * Nz
        H = lambda kx, ky, kz: lk6(kx, ky, kz, mat)
        C0 = H(0, 0, 0)
        Cxx, Cyy, Czz = H(1, 0, 0) - C0, H(0, 1, 0) - C0, H(0, 0, 1) - C0
        Cxy = H(1, 1, 0) - C0 - Cxx - Cyy
        Cxz = H(1, 0, 1) - C0 - Cxx - Czz
        Cyz = H(0, 1, 1) - C0 - Cyy - Czz
        I = np.eye(self.Ncs)
        Sy = np.kron(np.eye(Ny, k=1), np.eye(Nz))
        Sz = np.kron(np.eye(Ny), np.eye(Nz, k=1))
        Dy, Dz = (Sy - Sy.T) / (2 * a), (Sz - Sz.T) / (2 * a)
        Kyy, Kzz = (2 * I - Sy - Sy.T) / a**2, (2 * I - Sz - Sz.T) / a**2
        self.H0 = (np.kron(C0, I) + np.kron(Cxx, 2 / a**2 * I) + np.kron(Cyy, Kyy)
                   + np.kron(Czz, Kzz) + np.kron(Cyz, -Dy @ Dz))
        self.V = np.kron(Cxx, -I / a**2) + np.kron(Cxy, -Dy / (2 * a)) + np.kron(Cxz, -Dz / (2 * a))

    def slice_block(self, U_cs):
        return self.H0 + np.kron(np.eye(6), np.diag(U_cs))

    def bloch(self, kx):
        return self.H0 + self.V * np.exp(1j * kx * self.a) + self.V.conj().T * np.exp(-1j * kx * self.a)

    def subbands(self, kxs, nb=8):
        return np.array([np.sort(np.linalg.eigvalsh(self.bloch(k)))[::-1][:nb] for k in kxs])


class ModeSpace:
    """Coupled mode space: M highest valence eigenvectors of the reference slice."""

    def __init__(self, wire, M, U_ref=None):
        U_ref = np.zeros(wire.Ncs) if U_ref is None else U_ref
        w, v = eigh(wire.slice_block(U_ref))
        self.Phi = v[:, np.argsort(w)[::-1][:M]]
        self.wire = wire
        self.V = self.Phi.conj().T @ wire.V @ self.Phi

    def slice_block(self, U_cs):
        return self.Phi.conj().T @ self.wire.slice_block(U_cs) @ self.Phi


class KPDevice:
    def __init__(self, wire, U_profile, U_lead_L, U_lead_R, mode_space=None, eta=1e-5):
        blk = mode_space.slice_block if mode_space else wire.slice_block
        self.V = mode_space.V if mode_space else wire.V
        self.Hd = [blk(u) for u in U_profile]
        ones = np.ones(wire.Ncs)
        self.hL00, self.hR00 = blk(U_lead_L * ones), blk(U_lead_R * ones)
        self.eta = eta

    def sigmas(self, E):
        """Sigma_L = V^+ g_L V (lead coupling V^+ going left), Sigma_R = V g_R V^+."""
        Vd = self.V.conj().T
        SigL = Vd @ sancho_rubio(E, self.hL00, Vd) @ self.V
        SigR = self.V @ sancho_rubio(E, self.hR00, self.V) @ Vd
        return SigL, SigR

    def transmission(self, Es):
        T = np.zeros(len(Es))
        for k, E in enumerate(Es):
            z = E + 1j * self.eta
            SigL, SigR = self.sigmas(z)
            _, glc = rgf(z, self.Hd, self.V, SigL, SigR, True)
            GamL, GamR = 1j * (SigL - SigL.conj().T), 1j * (SigR - SigR.conj().T)
            T[k] = np.real(np.trace(GamL @ glc[0] @ GamR @ glc[0].conj().T))
        return T


def kp_lead_fermi_level(wire, ms, N_A_cm3, kT, a):
    """Hole Fermi level of a flat p++ lead (holes of H = electrons of -H)."""
    blk = ms.slice_block if ms else wire.slice_block
    h00 = -blk(np.zeros(wire.Ncs))
    V = -(ms.V if ms else wire.V)
    p_cell = N_A_cm3 * 1e6 * (a * 1e-9) ** 3 * wire.Ncs
    I = np.eye(h00.shape[0])

    def holes(mu):
        z, w = equilibrium_contour(-mu, kT, -0.8, 20, 12, 8)
        tot = 0.0
        for zk, wk in zip(z, w):
            gl = sancho_rubio(zk, h00, V.conj().T)
            gr = sancho_rubio(zk, h00, V)
            G = inv(zk * I - h00 - V @ gl @ V.conj().T - V.conj().T @ gr @ V)
            tot += wk * np.trace(G)
        return -np.imag(tot) / np.pi

    lo, hi = -0.8, 0.3
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if holes(mid) > p_cell:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def run_kp(args):
    kT = KB * args.T / Q
    N = max(1, int(round(args.W / args.a)))
    Nx = int(round(args.L / args.a))
    wire = KPWire(N, N, args.a, args.mat)
    ms = ModeSpace(wire, args.M) if args.M > 0 else None
    Eb = wire.subbands(np.array([0.0]), 4)[0]
    t0 = time.time()
    mu = kp_lead_fermi_level(wire, ms, args.NA, kT, args.a)
    print(f"k.p {args.mat} wire {args.W} nm, {6*N*N} orbitals/slice -> "
          f"{'mode space M=' + str(args.M) if ms else 'real space'}; top subband {Eb[0]:.4f} eV; "
          f"N_A={args.NA:.1e} -> mu = {mu:.4f} eV  [{time.time()-t0:.1f}s]", flush=True)
    x = np.arange(Nx) * args.a
    L = x[-1]
    Es = np.linspace(mu - 0.35, Eb[0] + 0.05, args.nE)
    rows = []
    for Vg in args.Vg_list:
        prof = -0.8 * Vg * np.exp(-((x - L / 2) / (0.22 * L)) ** 2) - args.Vds * x / L
        dev = KPDevice(wire, np.repeat(prof[:, None], wire.Ncs, 1), 0.0, -args.Vds, ms)
        T = dev.transmission(Es)
        I = Q_OVER_H_EV * np.trapezoid(T * (fermi_r(Es, mu, kT) - fermi_r(Es, mu - args.Vds, kT)), Es)
        rows.append(dict(V_g=Vg, I=float(I)))
        print(f"   V_g = {Vg:.2f} V   I_d = {I*1e6:9.4f} uA", flush=True)
    return rows


# =============================================================================
# 10. Self-test (physics identities)
# =============================================================================
def selftest(verbose=True):
    results = []

    def check(name, ok, detail):
        results.append((name, bool(ok), detail))
        if verbose:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name:52s} {detail}", flush=True)

    rng = np.random.default_rng(1)
    N, K = 3, 5
    Hd = [(lambda A: A + A.conj().T)(rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))) for _ in range(K)]
    V = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
    SigL = -0.3j * np.eye(N) + 0.1 * rng.normal(size=(N, N))
    SigR = -0.2j * np.eye(N)
    E = 0.37 + 1e-6j
    Hf = np.zeros((N * K, N * K), complex)
    for i in range(K):
        Hf[i*N:(i+1)*N, i*N:(i+1)*N] = Hd[i]
        if i + 1 < K:
            Hf[i*N:(i+1)*N, (i+1)*N:(i+2)*N] = V
            Hf[(i+1)*N:(i+2)*N, i*N:(i+1)*N] = V.conj().T
    Hf[:N, :N] += SigL; Hf[-N:, -N:] += SigR
    G = inv(E * np.eye(N * K) - Hf)
    grd, glc = rgf(E, Hd, V, SigL, SigR, True)
    gfc = rgf_first_column(E, Hd, V, SigL, SigR)
    err = max(max(np.abs(grd[i] - G[i*N:(i+1)*N, i*N:(i+1)*N]).max(),
                  np.abs(glc[i] - G[i*N:(i+1)*N, -N:]).max(),
                  np.abs(gfc[i] - G[i*N:(i+1)*N, :N]).max()) for i in range(K))
    check("RGF diag / last col / first col == dense inverse", err < 1e-10, f"max err {err:.1e}")

    t0 = 1.0
    E = 0.8
    g = sancho_rubio(E + 1e-9j, np.array([[2 * t0]]), np.array([[-t0]]))
    ka = np.arccos(1 - E / (2 * t0))
    err = abs(t0**2 * g[0, 0] - (-t0 * np.exp(1j * ka)))
    check("Lopez-Sancho == analytic -t exp(ika)", err < 1e-6, f"err {err:.1e}")

    kT = 0.0259
    lead = Lead(np.zeros((1, 1)), 1.0, 0.0)
    Vc = -np.eye(1)
    Uc = np.array([0.0, 0.1, 0.3, 0.3, 0.1, 0.0])
    Hc = [np.array([[2 + u]]) for u in Uc]
    mu = 0.7

    def diagG(z):
        S = lead.self_energy(z)
        return np.array([gg[0, 0] for gg in rgf(z, Hc, Vc, S, S, False)[0]])

    Er = np.linspace(-0.5, 1.6, 20001)
    ref = sum(-np.imag(diagG(e + 1e-6j)) / np.pi * fermi_r(e, mu, kT) for e in Er) * (Er[1] - Er[0])
    for label, (z, w) in (("TranSIESTA contour", equilibrium_contour(mu, kT, -0.5, 24, 12, 8)),
                          ("Ozaki poles", ozaki_contour(mu, kT, 40))):
        rho = -np.imag(sum(wk * diagG(zk) for zk, wk in zip(z, w))) / np.pi
        err = np.abs(rho - ref).max()
        check(f"equilibrium density: {label} == real-axis", err < 1e-4, f"max err {err:.1e}")

    p = NWParams(W=1.5, L_s=0.6, L_g=0.9, L_d=0.6, V_ds=0.0, eta=1e-9)
    mask = cross_section_mask(p)
    d = Device(p, mask, 0.26, 0.26, 0.26, 1, 0.0, 0.0)
    d.set_potential(np.zeros((d.Nx, d.N)))
    sub = np.sort(np.linalg.eigvalsh(d.H_cs))
    T = d.transmission(np.array([sub[0] + 0.05, sub[1] + 0.05, sub[3] + 0.05]))
    check("uniform wire: T per spin = # open subbands (1,3,4)", np.allclose(T, [1, 3, 4], atol=1e-5), f"T = {np.round(T, 6)}")

    p = NWParams(W=0.9, L_s=0.9, L_g=1.8, L_d=0.9, V_ds=0.2, eta=1e-9)
    mask = cross_section_mask(p)
    s0 = np.linalg.eigvalsh(cross_section_hamiltonian(mask, p.a, 0.26, 0.26))[0]   # lowest subband
    d = Device(p, mask, 0.26, 0.26, 0.26, 1, -0.3 - s0, -0.5 - s0)
    x = np.arange(p.Nx)
    d.set_potential(np.repeat((-0.3 - s0 - 0.2 * x / (p.Nx - 1) + 0.15 * np.exp(-((x - p.Nx / 2) / 2.0) ** 2))[:, None], d.N, 1))
    worst = 0.0
    for Ek in (-0.25, -0.1, 0.02):
        j = d._bond_point(Ek)
        land = d._neq_point(Ek)[2] * (fermi_r(Ek, 0, p.kT) - fermi_r(Ek, -0.2, p.kT))
        worst = max(worst, np.abs(j - land).max() / max(abs(land), 1e-12))
    check("bond current == Landauer T (f_L - f_R), every bond", worst < 1e-5, f"max rel err {worst:.1e}")

    p = NWParams(shape="circle", W=2.4, t_ox=1.0, L_s=1.2, L_g=1.8, L_d=1.2, V_g=0.3)
    mask = cross_section_mask(p)
    pois = Poisson3D(p, mask, 0.5, 0.6)
    n = 0.5 * pois.ND_site + 1e-4
    phi0 = np.zeros(pois.dir_mask.size); phi0[pois.dir_mask] = pois.dir_val[pois.dir_mask]
    phi = pois.solve(n, phi0, 1e9)            # huge V_T -> linear Poisson, one exact solve
    res = pois.residual(phi, n)
    check("Poisson3D (circle GAA) satisfies discrete equation", res < 1e-8, f"rel residual {res:.1e}")

    g1, g2, g3, D = HOLE_MATERIALS["Si"]
    k = 0.01
    E001 = np.sort(np.linalg.eigvalsh(lk6(0, 0, k)))[::-1]
    m_hh = -E0_NM * k**2 / E001[0]
    ok = abs(m_hh - 1 / (g1 - 2 * g2)) < 2e-3 and np.allclose(lk6(0.3, -0.2, 0.5), lk6(0.3, -0.2, 0.5).conj().T)
    check("k.p: LK Hermitian, m_HH[001] = 1/(g1-2g2)", ok, f"m_HH = {m_hh:.4f} vs {1/(g1-2*g2):.4f}")

    n_fail = sum(not r[1] for r in results)
    if verbose:
        print(f"{'ALL PASSED' if n_fail == 0 else str(n_fail) + ' FAILED'} ({len(results)} checks)")
    return results


# =============================================================================
# 11. Plotting and output
# =============================================================================
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_bias(p, res, post, path):
    plt = _plt()
    x = (np.arange(p.Nx) + 0.5) * p.a
    fig, ax = plt.subplots(2, 3, figsize=(16, 8.5))
    ax[0, 0].plot(x, post["Ec_x"], label="lowest subband edge")
    ax[0, 0].axhline(0, c="k", ls="--", lw=.8, label="mu_S")
    ax[0, 0].axhline(-p.V_ds, c="k", ls=":", lw=.8, label="mu_D")
    ax[0, 0].axvspan(p.L_s, p.L_s + p.L_g, color="grey", alpha=.12, label="gate")
    ax[0, 0].set(xlabel="x (nm)", ylabel="eV", title=f"Band profile  Vg={p.V_g} V  Vds={p.V_ds} V")
    ax[0, 0].legend(fontsize=8)
    nx = res["n"].reshape(p.Nx, -1).sum(1) / p.a
    ax[0, 1].semilogy(x, nx)
    ax[0, 1].set(xlabel="x (nm)", ylabel="electrons / nm", title="Line electron density")
    ax[0, 2].plot(post["E"], post["T"])
    ax[0, 2].axvspan(-p.V_ds, 0, color="orange", alpha=.15)
    ax[0, 2].set(xlabel="E (eV)", ylabel="T(E)", title="Transmission per spin (all valleys)")
    im = ax[1, 0].imshow(np.log10(post["ldos"] + 1e-6), origin="lower", aspect="auto", cmap="inferno",
                         extent=[0, p.Nx * p.a, post["E_ldos"][0], post["E_ldos"][-1]])
    ax[1, 0].plot(x, post["Ec_x"], "c--", lw=.8)
    ax[1, 0].set(xlabel="x (nm)", ylabel="E (eV)", title="log10 LDOS(x,E)")
    fig.colorbar(im, ax=ax[1, 0])
    if "I_bond" in post:
        ax[1, 1].plot(x[:-1] + p.a / 2, post["I_bond"] * 1e6, "o-", ms=3, label="bond current")
        ax[1, 1].axhline(post["I"] * 1e6, c="r", ls="--", label="Landauer")
        ax[1, 1].legend()
    ax[1, 1].set(xlabel="x (nm)", ylabel="uA", title="Current along the wire (conservation)")
    if res["phi"] is not None:
        pois = res["pois"]
        phi = res["phi"].reshape(pois.shape)
        im = ax[1, 2].imshow(phi[p.Nx // 2].T, origin="lower", cmap="viridis")
        ax[1, 2].set(title="phi(y,z) mid-channel (V)", xlabel="y (grid)", ylabel="z (grid)")
        fig.colorbar(im, ax=ax[1, 2])
    else:
        ax[1, 2].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_sweep(rows, xkey, path, title, group=None):
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    groups = {}
    for r in rows:
        groups.setdefault(r.get(group) if group else "", []).append(r)
    for gname, rr in groups.items():
        xs = [r[xkey] for r in rr]
        Is = np.array([r["I"] for r in rr]) * 1e6
        lab = f"{group}={gname}" if group else None
        ax[0].plot(xs, Is, "o-", label=lab)
        ax[1].semilogy(xs, np.abs(Is) + 1e-9, "o-", label=lab)
    for a_ in ax:
        a_.set(xlabel=xkey + " (V)", ylabel="I_d (uA)")
        if group:
            a_.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _save(outdir, name, data):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, name)
    with open(path, "w") as f:
        json.dump(data, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    return path


# =============================================================================
# 12. Command line
# =============================================================================
def _add_device_args(ap):
    g = ap.add_argument_group("device")
    g.add_argument("--quick", action="store_true", help="small 1.8 nm x 12 nm device (fast)")
    g.add_argument("--shape", choices=["square", "circle"])
    g.add_argument("--W", type=float, help="core width / diameter (nm)")
    g.add_argument("--tox", type=float, dest="t_ox")
    g.add_argument("--Ls", type=float, dest="L_s")
    g.add_argument("--Lg", type=float, dest="L_g")
    g.add_argument("--Ld", type=float, dest="L_d")
    g.add_argument("--a", type=float, help="grid spacing (nm)")
    g.add_argument("--material", choices=list(ELECTRON_MATERIALS))
    g.add_argument("--valleys", choices=["single", "si3"])
    g.add_argument("--ND", type=float, dest="N_D", help="S/D donor density (cm^-3)")
    g.add_argument("--T", type=float)
    g.add_argument("--Vg", type=float, dest="V_g")
    g.add_argument("--Vds", type=float, dest="V_ds")
    g.add_argument("--phi-ms", type=float, dest="phi_ms")
    n = ap.add_argument_group("numerics")
    n.add_argument("--density", choices=["contour", "ozaki"], dest="density_method")
    n.add_argument("--no-scf", action="store_false", dest="scf", default=None,
                   help="analytic potential instead of NEGF-Poisson (fast, qualitative)")
    n.add_argument("--scf-tol", type=float, dest="scf_tol")
    n.add_argument("--mixing", type=float)
    n.add_argument("--workers", type=int, dest="n_workers", help="processes for the energy loop")
    ap.add_argument("--outdir", default="negf_final_out")


def _params_from(args):
    kw = dict(QUICK) if getattr(args, "quick", False) else {}
    for f in NWParams.__dataclass_fields__:
        v = getattr(args, f, None)
        if v is not None:
            kw[f] = v
    return NWParams(**kw)


def _describe(p):
    mask = cross_section_mask(p)
    print(f"Device: {p.material} {p.shape} core W={p.W} nm ({int(mask.sum())} sites/slice), {p.Nx} slices "
          f"(L_s/L_g/L_d = {p.L_s}/{p.L_g}/{p.L_d} nm), t_ox={p.t_ox} nm, valleys={p.valleys}, "
          f"{'NEGF-Poisson SCF' if p.scf else 'analytic potential'}, density={p.density_method}, "
          f"workers={p.n_workers}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Finalized NEGF code for Si nanowire (GAA) transistors",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__.split("USAGE")[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest", help="physics-identity checks")
    b = sub.add_parser("bias", help="one bias point: SCF, T(E), I, LDOS, bond current, figures")
    _add_device_args(b)
    s = sub.add_parser("idvg", help="transfer characteristic I_d(V_g)")
    _add_device_args(s)
    s.add_argument("--Vg-range", nargs=3, type=float, default=[0.0, 0.8, 5], metavar=("START", "STOP", "N"))
    o = sub.add_parser("idvd", help="output characteristics I_d(V_ds) for several V_g")
    _add_device_args(o)
    o.add_argument("--Vds-range", nargs=3, type=float, default=[0.05, 0.4, 4], metavar=("START", "STOP", "N"))
    o.add_argument("--Vg-list", nargs="+", type=float, default=[0.4, 0.6])
    m = sub.add_parser("materials", help="compare channel materials at one bias")
    _add_device_args(m)
    m.add_argument("--list", nargs="+", default=list(ELECTRON_MATERIALS))
    e = sub.add_parser("export-h", help="write the device Hamiltonian (sparse) to disk")
    _add_device_args(e)
    e.add_argument("--format", choices=["npz", "mat", "csv"], default="npz")
    k = sub.add_parser("kp", help="6-band k.p hole transport (p-FET)")
    k.add_argument("--mat", choices=list(HOLE_MATERIALS), default="Si")
    k.add_argument("--W", type=float, default=2.4)
    k.add_argument("--a", type=float, default=0.4)
    k.add_argument("--L", type=float, default=12.0)
    k.add_argument("--M", type=int, default=64, help="modes kept (0 = real space)")
    k.add_argument("--NA", type=float, default=5e20)
    k.add_argument("--T", type=float, default=300.0)
    k.add_argument("--Vds", type=float, default=0.1)
    k.add_argument("--Vg-list", nargs="+", type=float, default=[0.0, 0.15, 0.3, 0.45])
    k.add_argument("--nE", type=int, default=120)
    k.add_argument("--outdir", default="negf_final_out")
    args = ap.parse_args(argv)
    T0 = time.time()

    if args.cmd == "selftest":
        res = selftest()
        return 0 if all(r[1] for r in res) else 1

    if args.cmd == "kp":
        rows = run_kp(args)
        print("saved", _save(args.outdir, "kp_idvg.json", dict(args=vars(args), rows=rows)))
        print(f"Total wall time: {time.time()-T0:.1f} s")
        return 0

    p = _params_from(args)
    if p.scf is None:
        p.scf = True
    _describe(p)

    if args.cmd == "bias":
        res, post = run_bias(p)
        tag = f"bias_{p.shape}_Vg{p.V_g:.2f}_Vds{p.V_ds:.2f}"
        plot_bias(p, res, post, os.path.join(args.outdir, tag + ".png")) if os.makedirs(args.outdir, exist_ok=True) is None else None
        summ = dict(params=asdict(p), I_d=post["I"], U_lead=res["U_lead"], scf_history=res["hist"],
                    bond_current=post.get("I_bond"), bond_rel_dev=post.get("bond_dev"), time_s=time.time() - T0)
        np.savez(os.path.join(args.outdir, tag + ".npz"), E=post["E"], T=post["T"], U=res["U"], n=res["n"],
                 E_ldos=post["E_ldos"], ldos=post["ldos"], Ec_x=post["Ec_x"])
        print("saved", _save(args.outdir, tag + ".json", summ), "and", tag + ".png/.npz")
    elif args.cmd == "idvg":
        vals = np.linspace(args.Vg_range[0], args.Vg_range[1], int(args.Vg_range[2]))
        rows = sweep(p, "V_g", vals)
        plot_sweep(rows, "V_g", os.path.join(args.outdir, "idvg.png"), f"I_d-V_g, V_ds={p.V_ds} V") \
            if os.makedirs(args.outdir, exist_ok=True) is None else None
        print("saved", _save(args.outdir, "idvg.json", dict(params=asdict(p), rows=rows)))
    elif args.cmd == "idvd":
        vals = np.linspace(args.Vds_range[0], args.Vds_range[1], int(args.Vds_range[2]))
        rows = []
        for vg in args.Vg_list:
            print(f"--- V_g = {vg} V", flush=True)
            rows += sweep(replace(p, V_g=vg), "V_ds", vals)
        os.makedirs(args.outdir, exist_ok=True)
        plot_sweep(rows, "V_ds", os.path.join(args.outdir, "idvd.png"), "Output characteristics", group="V_g")
        print("saved", _save(args.outdir, "idvd.json", dict(params=asdict(p), rows=rows)))
    elif args.cmd == "materials":
        rows = []
        for mat in args.list:
            pm = replace(p, material=mat, valleys=p.valleys if mat == "Si" else "single")
            t0 = time.time()
            res, post = run_bias(pm, verbose=False, bond=False, ldos=False)
            rows.append(dict(material=mat, I=post["I"], U_lead=res["U_lead"], time=time.time() - t0))
            print(f"   {mat:7s} E_F-E_c,lead = {-res['U_lead']:.3f} eV   I_d = {post['I']*1e6:9.4f} uA"
                  f"   ({time.time()-t0:.0f} s)", flush=True)
        print("saved", _save(args.outdir, "materials.json", dict(params=asdict(p), rows=rows)))
    elif args.cmd == "export-h":
        mask, U_lead, devs = build_devices(p)
        pois = Poisson3D(p, mask, -U_lead, -U_lead + p.V_ds)
        for d in devs:
            d.set_potential(analytic_potential(p, U_lead, pois.n_site))
        os.makedirs(args.outdir, exist_ok=True)
        for iv, d in enumerate(devs):
            info = export_hamiltonian(d, os.path.join(args.outdir, f"hamiltonian_valley{iv}.{args.format}"), args.format)
            print("  ", info)
    print(f"Total wall time: {time.time()-T0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
