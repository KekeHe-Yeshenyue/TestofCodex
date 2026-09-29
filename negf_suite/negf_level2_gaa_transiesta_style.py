#!/usr/bin/env python3
"""
=============================================================================
 LEVEL 2  --  Self-consistent NEGF-Poisson for a Gate-All-Around (GAA) Si
              nanowire FET, organised the way TranSIESTA does it
=============================================================================

TranSIESTA (Brandbyge et al., PRB 65, 165401 (2002); Papior et al., CPC 212,
8 (2017)) is a DFT+NEGF code.  This file keeps TranSIESTA's *transport
machinery* but replaces the DFT Hamiltonian by an effective-mass
tight-binding (finite-difference) Hamiltonian for silicon, so that the whole
thing runs in minutes on a laptop.  Every step below is labelled with the
TranSIESTA concept it mirrors:

  [TS-1] Electrode calculation      -> neutral_lead_shift()
         (bulk lead, find E_F for charge neutrality; TranSIESTA does this with
          a separate periodic DFT run and stores TSHS)
  [TS-2] Electrode self-energies    -> sancho_rubio()  (Lopez-Sancho decimation,
          identical algorithm to TranSIESTA/sisl RecursiveSI and to dpnegf's
          negf/surface_green.py)
  [TS-3] Block-tridiagonal Green function  -> rgf()
          (TranSIESTA "BTD" inversion; dpnegf negf/recursive_green_cal.py;
           Anantram, Lundstrom, Nikonov, Proc. IEEE 96, 1511 (2008))
  [TS-4] Equilibrium density matrix by complex-contour integration
          -> equilibrium_contour():  circle + line + Fermi poles
          (TS.Contours.Eq  circle/line, TS.Contour.Eq.Pole).  An alternative
          Ozaki continued-fraction pole sum (what dpnegf uses) is provided
          in ozaki_contour() and cross-checked.
  [TS-5] Non-equilibrium density by real-axis integration in the bias window
          -> Device.density(): Delta_neq = Int (f_R - f_L) G Gamma_R G^+ dE/2pi
          with the Brandbyge weighting of the two equivalent expressions
          (rho = rho_eq^L + Delta^R  vs  rho_eq^R + Delta^L).
  [TS-6] Hartree potential with gate (TS.Hartree / electrostatic gating)
          -> Poisson3D: 3-D finite-volume Poisson, epsilon(r), GAA gate as a
          Dirichlet boundary, Gummel-Newton linearisation.
  [TS-7] SCF loop with mixing              -> scf()
  [TS-8] tbtrans post-processing: T(E), I, LDOS on a fine real-axis grid
          -> Device.transmission(), Device.current(), Device.ldos()

Physical model
  * Si conduction band, effective-mass approximation.  Optionally all three
    pairs of Delta valleys with anisotropic masses (m_l = 0.916, m_t = 0.19)
    for a [100] wire, or a single isotropic valley for speed.  Spin 2.
  * Square Si core (W x W) surrounded by SiO2 (t_ox) and a metal gate on all
    four sides = gate-all-around.  Source/drain extensions are n++ doped and
    continue as semi-infinite leads.  Hard-wall boundary at Si/SiO2 for the
    wavefunction (no penetration), full dielectric treatment in Poisson.

Requirements  python >= 3.9, numpy, scipy, matplotlib.  No compiler.
Run           python negf_level2_gaa_transiesta_style.py --quick     (~2 min)
              python negf_level2_gaa_transiesta_style.py --workers 4 (full size, ~10-20 min)
              python negf_level2_gaa_transiesta_style.py --help
Hardware      quick: any laptop (< 1 GB RAM, ~2 min).  Default size: ~8 min per bias point on 4 cores (470 s measured).
=============================================================================
"""
import os
# Pin the BLAS library to ONE thread *before* numpy is imported.  The matrices here are
# small (<= a few hundred), where threaded BLAS gains nothing, and with --workers > 1 the
# competing OpenBLAS thread pools of the worker processes slow everything down by 100x.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import time
import numpy as np
from numpy.linalg import inv
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.linalg import eigh_tridiagonal
from dataclasses import dataclass, field
from multiprocessing import Pool
from scipy.integrate import quad_vec

# ----------------------------------------------------------------------------- constants
HBAR = 1.054571817e-34; Q = 1.602176634e-19; M0 = 9.1093837015e-31
KB = 1.380649e-23; EPS0 = 8.8541878128e-12
G0 = 2 * Q**2 / (2 * np.pi * HBAR)            # 2e^2/h = 77.48 uS

def fermi(E, mu, kT):
    """Fermi function, works for real or complex E (needed on the contour), scalar or array."""
    scalar = np.ndim(E) == 0
    x = np.atleast_1d(np.asarray((np.asarray(E) - mu) / kT, dtype=complex))
    xr = x.real
    out = np.empty_like(x)
    big = xr > 40; small = xr < -40; mid = ~(big | small)
    out[big] = 0.0; out[small] = 1.0
    out[mid] = 1.0 / (1.0 + np.exp(x[mid]))
    return complex(out[0]) if scalar else out


# ============================================================================ parameters
@dataclass
class GAAParams:
    # geometry (nm)
    W: float = 2.4          # Si core width = height
    t_ox: float = 1.0       # oxide thickness (all four sides)
    L_s: float = 4.0        # source extension (n++)
    L_g: float = 8.0        # gate length (channel, undoped)
    L_d: float = 4.0        # drain extension (n++)
    a: float = 0.3          # grid spacing (nm) for both H and Poisson
    # material
    eps_si: float = 11.7
    eps_ox: float = 3.9
    N_D: float = 1e20       # cm^-3 donor density in S/D extensions and leads
    valleys: str = "single" # "single" (m*=0.26 isotropic, g=6? no: g_v=1 for speed) or "si3" (3 Delta-valley pairs)
    T: float = 300.0
    # bias
    V_g: float = 0.4
    V_ds: float = 0.3
    phi_ms: float = 0.45    # flat-band / work-function offset (V).  The gate electrode is
                            # held at phi_gate = phi_S + V_g - phi_ms, i.e. V_g is measured
                            # w.r.t. the source lead; V_g = phi_ms would align the channel
                            # band with the n++ lead band (fully ON).  Threshold ~ 0.35 V.
    # numerics
    eta: float = 1e-4       # device broadening on the real axis (eV); resonances narrower than
                            # this are broadened -- needed for a finite real-axis quadrature
    n_circle: int = 24; n_line: int = 12; n_pole: int = 8
    neq_tol: float = 1e-5   # absolute tolerance (electrons/site) of the adaptive real-axis integral
    density_method: str = "contour"   # "contour" (TranSIESTA) or "ozaki" (dpnegf)
    ozaki_M: int = 60
    scf_tol: float = 2e-3   # V
    scf_maxiter: int = 40
    mixing: float = 0.6
    n_workers: int = 1

    @property
    def kT(self): return KB * self.T / Q
    @property
    def valley_list(self):
        if self.valleys == "si3":      # [100] wire: (m_x, m_y, m_z, degeneracy)
            return [(0.916, 0.19, 0.19, 2), (0.19, 0.916, 0.19, 2), (0.19, 0.19, 0.916, 2)]
        return [(0.26, 0.26, 0.26, 1)]  # cheap isotropic single valley (spin still 2)


# ============================================================================ [TS-2] surface GF
def sancho_rubio(E, h00, h01, eta=0.0, tol=1e-10, maxit=300):
    """
    Lopez-Sancho / Sancho-Rubio decimation for the surface Green function of a
    semi-infinite lead with principal-layer Hamiltonian h00 and inter-layer
    coupling h01 (pointing *into* the lead).  Returns g_s (N x N).
    Same algorithm as TranSIESTA, sisl.RecursiveSI and dpnegf surface_green().
    E may be complex.
    """
    N = h00.shape[0]; Ez = (E + 1j * eta) * np.eye(N)
    eps_s = h00.astype(complex).copy(); eps = eps_s.copy()
    alpha = h01.astype(complex).copy(); beta = alpha.conj().T.copy()
    for _ in range(maxit):
        g = inv(Ez - eps)
        agb = alpha @ g @ beta
        eps_s = eps_s + agb
        eps = eps + agb + beta @ g @ alpha
        alpha = alpha @ g @ alpha; beta = beta @ g @ beta
        if np.abs(alpha).max() < tol and np.abs(beta).max() < tol:
            break
    return inv(Ez - eps_s)


# ============================================================================ [TS-3] RGF
def rgf(E, Hd, V, SigL, SigR, need_lastcol=True):
    """
    Recursive Green function for a block-tridiagonal H (uniform block size).
      Hd   : list of K diagonal blocks (N x N) (potential already included)
      V    : coupling block H_{i,i+1} (N x N), H_{i+1,i} = V^+
      SigL : self-energy added to block 0 ;  SigR : added to block K-1
    Returns
      grd    : list of diagonal blocks G_{ii}
      gr_lc  : list of last-column blocks G_{i,K-1}   (None if not requested)
    Algorithm: Anantram-Lundstrom-Nikonov (2008), eqs. B1-B6, as in dpnegf.
    """
    K = len(Hd); N = Hd[0].shape[0]; I = np.eye(N)
    Vd = V.conj().T
    gL = [None] * K
    gL[0] = inv(E * I - Hd[0] - SigL)
    for i in range(1, K):
        D = E * I - Hd[i] - Vd @ gL[i - 1] @ V
        if i == K - 1: D -= SigR
        gL[i] = inv(D)
    grd = [None] * K; grd[-1] = gL[-1]
    gr_lc = [None] * K if need_lastcol else None
    if need_lastcol: gr_lc[-1] = gL[-1]
    for i in range(K - 2, -1, -1):
        gV = gL[i] @ V
        grd[i] = gL[i] + gV @ grd[i + 1] @ Vd @ gL[i]
        if need_lastcol: gr_lc[i] = gV @ gr_lc[i + 1]
    return grd, gr_lc


# ============================================================================ [TS-4] contours
def equilibrium_contour(mu, kT, E_min, n_circle=24, n_line=12, n_pole=8):
    """
    TranSIESTA-style equilibrium contour for  rho_eq = -1/pi Im Int f(E) G(E) dE.
    Returns arrays (z, w) such that   rho_eq = -1/pi * Im[ sum_k w_k G(z_k) ].
       * circle from E_min (real axis) up to  P = mu - 10kT + i*gamma
       * horizontal line from P to mu + 25kT + i*gamma      (f ~ e^-25 there)
       * Fermi poles  z_nu = mu + i pi kT (2nu+1),  nu < n_pole, residue -kT
    with gamma = 2 pi kT n_pole (exactly between pole n_pole-1 and n_pole).
    Derivation:  Int_real f G dE = Int_C f G dz + 2 pi i sum_nu Res_nu,
                 Res_nu = -kT G(z_nu)   (G analytic in the upper half plane).
    """
    gamma = 2 * np.pi * kT * n_pole
    P = mu - 10 * kT
    # circle through E_min and P + i*gamma with centre c on the real axis
    c = ((P**2 + gamma**2) - E_min**2) / (2 * (P - E_min))
    R = c - E_min
    th1 = np.arctan2(gamma, P - c)
    x, wx = np.polynomial.legendre.leggauss(n_circle)          # theta in [th1, pi]
    th = 0.5 * (np.pi - th1) * x + 0.5 * (np.pi + th1); wth = 0.5 * (np.pi - th1) * wx
    z_c = c + R * np.exp(1j * th); dz_c = 1j * R * np.exp(1j * th) * wth
    # path direction: from E_min (theta=pi) to P (theta=th1)  -> reverse sign
    w_c = -fermi(z_c, mu, kT) * dz_c
    # line at height gamma = 2 pi kT n_pole.  NOTE: exp(i*2*pi*n_pole) = 1, so on this
    # line f(E + i gamma) = f(E) is the *real* step-like Fermi function -> a plain
    # Gauss-Legendre rule would be poor.  TranSIESTA uses a Gauss-Fermi rule; here we
    # split into three panels: [P, mu-3kT] GL, [mu-3kT, mu+3kT] GL, tail Gauss-Laguerre.
    n_a = max(2, n_line // 3); n_b = max(2, n_line // 3); n_c = max(2, n_line - n_a - n_b)
    x, wx = np.polynomial.legendre.leggauss(n_a)
    ra = 0.5 * (mu - 3 * kT - P) * x + 0.5 * (mu - 3 * kT + P); wa = 0.5 * (mu - 3 * kT - P) * wx
    x, wx = np.polynomial.legendre.leggauss(n_b)
    rb = 3 * kT * x + mu; wb = 3 * kT * wx
    xl, wl = np.polynomial.laguerre.laggauss(n_c)               # Int_0^inf e^-x g(x) dx
    rc = mu + 3 * kT + kT * xl; wc = kT * wl * np.exp(xl)          # weights for Int g(s) ds
    re = np.concatenate([ra, rb, rc]); wre = np.concatenate([wa, wb, wc])
    z_l = re + 1j * gamma; w_l = fermi(z_l, mu, kT) * wre
    # poles
    nu = np.arange(n_pole)
    z_p = mu + 1j * np.pi * kT * (2 * nu + 1); w_p = np.full(n_pole, -2j * np.pi * kT)
    return np.concatenate([z_c, z_l, z_p]), np.concatenate([w_c, w_l, w_p])


def ozaki_poles(M):
    """Ozaki (PRB 75, 035123 (2007)) poles z_p and residues R_p of the continued-
    fraction Fermi function  f(x) = 1/2 - sum_p R_p [1/(x - i z_p) + 1/(x + i z_p)],
    computed as eigen-decomposition of the tridiagonal Ozaki matrix (as in dpnegf)."""
    N = 2 * M
    off = np.array([0.5 / np.sqrt((2 * n - 1) * (2 * n + 1)) for n in range(1, N)])
    ev, evec = eigh_tridiagonal(np.zeros(N), off, select='i', select_range=(N // 2, N - 1))
    z = np.flip(1.0 / ev); R = np.flip(np.abs(evec[0, :])**2 / (4 * ev**2))
    return z, R

def ozaki_contour(mu, kT, M=60, R_big=1e4):
    """
    dpnegf-style alternative to equilibrium_contour(): only points on the
    imaginary axis.  rho_eq = -1/pi Im[ sum_k w_k G(z_k) ] with
       z_0 = mu + i R_big,  w_0 = -(i pi / 2) * (i R_big)        (0th moment / half filling)
       z_p = mu + i kT zeta_p,  w_p = -2 pi i kT R_p                (Ozaki poles)
    """
    zeta, Rp = ozaki_poles(M)
    z = np.concatenate([[mu + 1j * R_big], mu + 1j * kT * zeta])
    w = np.concatenate([[-(1j * np.pi / 2) * (1j * R_big)], -2j * np.pi * kT * Rp])
    return z, w


# ============================================================================ Hamiltonian
def cross_section_hamiltonian(Ny, Nz, a_m, my, mz):
    """2-D finite-difference kinetic energy on the Si core (hard wall). eV."""
    ty = HBAR**2 / (2 * my * M0 * a_m**2) / Q; tz = HBAR**2 / (2 * mz * M0 * a_m**2) / Q
    N = Ny * Nz; H = np.zeros((N, N))
    idx = lambda iy, iz: iy * Nz + iz
    for iy in range(Ny):
        for iz in range(Nz):
            i = idx(iy, iz); H[i, i] = 2 * ty + 2 * tz
            if iy + 1 < Ny: j = idx(iy + 1, iz); H[i, j] = H[j, i] = -ty
            if iz + 1 < Nz: j = idx(iy, iz + 1); H[i, j] = H[j, i] = -tz
    return H


class Lead:
    """[TS-1]/[TS-2] Semi-infinite n++ lead = periodic repetition of one slice."""
    def __init__(self, H_cs, tx, U_lead=0.0):
        N = H_cs.shape[0]
        self.h00 = H_cs + (2 * tx + U_lead) * np.eye(N)
        self.h01 = -tx * np.eye(N)            # coupling into the lead
        self.N = N
        self._cache = {}
    def surface_gf(self, E):
        return sancho_rubio(E, self.h00, self.h01)
    def self_energy(self, E):
        """Sigma = V^+ g_s V  with V = h01 (Hermitian here).  Cached per energy: the
        leads are fixed during the SCF, so the contour points hit the cache."""
        key = complex(E)
        if key not in self._cache:
            if len(self._cache) > 4000: self._cache.clear()
            g = self.surface_gf(E)
            self._cache[key] = self.h01.conj().T @ g @ self.h01
        return self._cache[key]
    def bulk_gf_diag(self, E):
        """Bulk (infinite-lead) Green function of one slice: (E - h00 - Sig_L - Sig_R)^-1."""
        s = self.self_energy(E)
        return inv(E * np.eye(self.N) - self.h00 - 2 * s)


def valley_hoppings(p, mx, my, mz):
    a_m = p.a * 1e-9
    tx = HBAR**2 / (2 * mx * M0 * a_m**2) / Q
    return a_m, tx


# ============================================================================ Device
class Device:
    """
    All NEGF quantities for the GAA wire at a given potential U(x,y,z) on the Si core.
    One instance per valley; densities/currents are summed over valleys by the caller.
    """
    def __init__(self, p: GAAParams, mx, my, mz, g_v, U_lead_S, U_lead_D):
        self.p = p; self.g = g_v * 2                     # valley x spin degeneracy
        a_m, tx = valley_hoppings(p, mx, my, mz)
        self.Ny = self.Nz = int(round(p.W / p.a))
        self.Nx = int(round((p.L_s + p.L_g + p.L_d) / p.a))
        self.N = self.Ny * self.Nz; self.tx = tx
        self.H_cs = cross_section_hamiltonian(self.Ny, self.Nz, a_m, my, mz)
        self.V = -tx * np.eye(self.N)
        self.leadL = Lead(self.H_cs, tx, U_lead_S)
        self.leadR = Lead(self.H_cs, tx, U_lead_D)
        self.mu_L = 0.0; self.mu_R = -p.V_ds
        self.U = None
        # lowest point of the equilibrium contour: well below any state (fixed once -> cacheable)
        self.E_min = min(U_lead_S, U_lead_D) - 1.5

    def set_potential(self, U):
        """U: (Nx, N) electron potential energy on the Si core (eV)."""
        self.U = np.asarray(U); self.Hd = [self.H_cs + np.diag(2 * self.tx + self.U[i]) for i in range(self.Nx)]

    # --- one Green-function evaluation -----------------------------------------------
    def _G(self, E, need_lastcol):
        SigL = self.leadL.self_energy(E); SigR = self.leadR.self_energy(E)
        grd, gr_lc = rgf(E, self.Hd, self.V, SigL, SigR, need_lastcol)
        return grd, gr_lc, SigL, SigR

    def _diagG(self, z):
        grd, _, _, _ = self._G(z, False)
        return np.concatenate([np.diag(g) for g in grd])                # complex, length Nx*N

    def _neq_point(self, E):
        """Real-axis point: returns diag of A_L, A_R and T(E)."""
        grd, gr_lc, SigL, SigR = self._G(E + 1j * self.p.eta, True)
        GamL = 1j * (SigL - SigL.conj().T); GamR = 1j * (SigR - SigR.conj().T)
        A_R = np.concatenate([np.real(np.diag(g @ GamR @ g.conj().T)) for g in gr_lc])
        A = np.concatenate([np.real(np.diag(1j * (g - g.conj().T))) for g in grd])
        A_L = A - A_R
        T = np.real(np.trace(GamL @ gr_lc[0] @ GamR @ gr_lc[0].conj().T))
        return A_L, A_R, T

    # --- [TS-4]/[TS-5] density -------------------------------------------------------------
    def _eq_points(self, mu):
        p = self.p
        if self.U.min() - 0.2 < self.E_min:            # potential dipped very deep: move E_min down
            self.E_min = self.U.min() - 0.5
        if p.density_method == "ozaki":
            return ozaki_contour(mu, p.kT, p.ozaki_M)
        return equilibrium_contour(mu, p.kT, self.E_min, p.n_circle, p.n_line, p.n_pole)

    def density(self, pool=None):
        """
        Electron density per site (Nx*N,), summed over spin/valley degeneracy g.
        rho = w * (rho_eq^L + Delta^R) + (1-w) * (rho_eq^R + Delta^L)     [Brandbyge 2002]
        (`pool` is ignored; parallelism is controlled by GAAParams.n_workers.)
        """
        p = self.p; kT = p.kT
        with _worker_pool(self, p.n_workers) as pmap:
            zL, wL = self._eq_points(self.mu_L)
            rho_eq_L = -np.imag(_sum_weighted(pmap, zL, wL)) / np.pi
            if abs(p.V_ds) < 1e-9:
                self._n_neq_eval = 0
                return self.g * rho_eq_L
            zR, wR = self._eq_points(self.mu_R)
            rho_eq_R = -np.imag(_sum_weighted(pmap, zR, wR)) / np.pi
            # non-equilibrium window on the real axis: adaptive Gauss-Kronrod (quad_vec) because
            # the integrand contains narrow quasi-bound-state resonances that a fixed rule would miss.
            lo, hi = min(self.mu_L, self.mu_R) - 10 * kT, max(self.mu_L, self.mu_R) + 10 * kT
            val, err, info = quad_vec(_neq_integrand, lo, hi, epsabs=p.neq_tol, epsrel=1e-4, norm='max',
                                      limit=400, workers=pmap, full_output=True)
            self._n_neq_eval = info.neval
        n_site = self.Nx * self.N
        dR, dL = val[:n_site], val[n_site:]           # dR corrects rho_eq^L, dL corrects rho_eq^R
        rhoL = rho_eq_L + dR; rhoR = rho_eq_R + dL
        wgt = dR**2 / (dL**2 + dR**2 + 1e-30)            # Brandbyge weighting (elementwise)
        return self.g * (wgt * rhoL + (1 - wgt) * rhoR)

    # --- [TS-8] tbtrans-like post-processing --------------------------------------------
    def transmission(self, Es, pool=None):
        with _worker_pool(self, self.p.n_workers) as pmap:
            return np.array([r[2] for r in pmap(_work_neq, list(Es))])

    def current(self, Es, T=None, pool=None):
        """I = g (q/h) Int T (f_L - f_R) dE   (g includes spin).  T on the grid Es is
        returned for plotting; the integral itself uses adaptive quadrature."""
        if T is None: T = self.transmission(Es)
        with _worker_pool(self, self.p.n_workers) as pmap:
            val, err = quad_vec(_current_integrand, Es[0], Es[-1], epsabs=1e-6, epsrel=1e-4, limit=400, workers=pmap)
        return self.g * Q / (2 * np.pi * HBAR) * Q * val, T

    def ldos_x(self, Es, pool=None):
        """LDOS(x, E) summed over the cross-section  (1/eV per slice)."""
        with _worker_pool(self, self.p.n_workers) as pmap:
            diags = pmap(_work_diag, [E + 1j * self.p.eta for E in Es])
        out = np.array([-np.imag(d).reshape(self.Nx, self.N).sum(1) / np.pi for d in diags])
        return self.g * out


# --- helpers for (optional) multiprocessing over energy points --------------------------
# The Device is handed to the worker processes once per density() call through the Pool
# initializer (fork snapshot), so each worker sees the *current* potential.  All functions
# mapped over energies are module-level (picklable) and read the device from _DEV.
_DEV = None
def _init_worker(dev):
    global _DEV; _DEV = dev
def _work_diag(z): return _DEV._diagG(z)
def _work_neq(E): return _DEV._neq_point(E)
def _neq_integrand(E):
    d = _DEV; kT = d.p.kT
    A_L, A_R, T = d._neq_point(E)
    fL, fR = fermi(E, d.mu_L, kT).real, fermi(E, d.mu_R, kT).real
    return np.concatenate([(fR - fL) * A_R, (fL - fR) * A_L]) / (2 * np.pi)
def _current_integrand(E):
    d = _DEV; kT = d.p.kT
    return d._neq_point(E)[2] * (fermi(E, d.mu_L, kT).real - fermi(E, d.mu_R, kT).real)

class _worker_pool:
    """Context manager yielding a map-like callable: multiprocessing Pool.map or serial map."""
    def __init__(self, dev, n_workers):
        self.dev, self.n = dev, n_workers; self.pool = None
    def __enter__(self):
        _init_worker(self.dev)
        if self.n > 1:
            self.pool = Pool(self.n, initializer=_init_worker, initargs=(self.dev,))
            return self.pool.map
        return lambda fn, xs: list(map(fn, xs))
    def __exit__(self, *a):
        if self.pool: self.pool.close(); self.pool.join()

def _sum_weighted(pmap, zs, ws):
    diags = pmap(_work_diag, list(zs))
    return sum(w * d for w, d in zip(ws, diags))


# ============================================================================ [TS-1] lead E_F
def neutral_lead_shift(p: GAAParams):
    """
    Find U_lead (band offset of the n++ leads, eV, relative to mu_S = 0) such
    that the lead is charge neutral: sum_valleys electrons/site = N_D a^3.
    This is TranSIESTA's "electrode" step: bulk electronic structure, then the
    Fermi level of the semi-infinite electrode fixes the boundary condition.
    """
    a_m = p.a * 1e-9; nD_site = p.N_D * 1e6 * a_m**3
    Ny = Nz = int(round(p.W / p.a)); N = Ny * Nz
    leads = []
    for (mx, my, mz, g) in p.valley_list:
        _, tx = valley_hoppings(p, mx, my, mz)
        leads.append((Lead(cross_section_hamiltonian(Ny, Nz, a_m, my, mz), tx, 0.0), 2 * g))
    def n_of_mu(mu):                      # electrons per site with lead band bottom at U=0
        n = 0.0
        for lead, g in leads:
            z, w = equilibrium_contour(mu, p.kT, -0.5, p.n_circle, p.n_line, p.n_pole)
            tot = sum(wk * np.trace(lead.bulk_gf_diag(zk)) for zk, wk in zip(z, w))
            n += g * (-np.imag(tot) / np.pi) / N
        return n
    lo, hi = -0.5, 3.0
    for _ in range(40):                   # bisection on mu (lead frame)
        mid = 0.5 * (lo + hi)
        if n_of_mu(mid) > nD_site: hi = mid
        else: lo = mid
    mu_lead = 0.5 * (lo + hi)
    return -mu_lead                       # U_lead such that mu_S (=0) - U_lead = mu_lead


# ============================================================================ [TS-6] Poisson
class Poisson3D:
    """
    Finite-volume Poisson  div(eps grad phi) = -rho  on a box that contains the
    Si core plus the oxide shell.  Gate = Dirichlet on the outer y/z faces for
    x in the gate window; source/drain ends = Dirichlet (lead potentials);
    other outer faces = Neumann.  Gummel/Newton: n(phi) = n_old exp((phi-phi_old)/V_T).
    """
    def __init__(self, p: GAAParams, phi_S, phi_D):
        self.p = p; a = p.a * 1e-9; self.a = a
        n_ox = int(round(p.t_ox / p.a)); Nc = int(round(p.W / p.a))
        self.Nx = int(round((p.L_s + p.L_g + p.L_d) / p.a)); self.Ny = self.Nz = Nc + 2 * n_ox
        self.n_ox = n_ox; self.Nc = Nc
        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz; self.shape = (Nx, Ny, Nz)
        idx = np.arange(Nx * Ny * Nz).reshape(Nx, Ny, Nz)
        # core mask & eps
        core = np.zeros((Nx, Ny, Nz), bool); core[:, n_ox:n_ox + Nc, n_ox:n_ox + Nc] = True
        self.core = core; self.core_idx = idx[core]           # ordering: x, then y, then z  (matches Device U)
        eps = np.where(core, p.eps_si, p.eps_ox) * EPS0
        # Dirichlet: gate faces in gate window, and x-ends
        ix_g0 = int(round(p.L_s / p.a)); ix_g1 = int(round((p.L_s + p.L_g) / p.a))
        dir_mask = np.zeros((Nx, Ny, Nz), bool); dir_val = np.zeros((Nx, Ny, Nz))
        phi_gate = phi_S + p.V_g - p.phi_ms
        for face in (np.s_[ix_g0:ix_g1, 0, :], np.s_[ix_g0:ix_g1, -1, :], np.s_[ix_g0:ix_g1, :, 0], np.s_[ix_g0:ix_g1, :, -1]):
            dir_mask[face] = True; dir_val[face] = phi_gate
        dir_mask[0] = True; dir_val[0] = phi_S; dir_mask[-1] = True; dir_val[-1] = phi_D
        self.dir_mask = dir_mask.ravel(); self.dir_val = dir_val.ravel()
        # Laplacian with harmonic-mean eps on faces
        rows, cols, vals = [], [], []
        def add(i, j, v): rows.append(i); cols.append(j); vals.append(v)
        for ax, n_ax in enumerate((Nx, Ny, Nz)):
            sl_lo = [slice(None)] * 3; sl_hi = [slice(None)] * 3
            sl_lo[ax] = slice(0, n_ax - 1); sl_hi[ax] = slice(1, n_ax)
            e_lo = eps[tuple(sl_lo)]; e_hi = eps[tuple(sl_hi)]
            e_face = (2 * e_lo * e_hi / (e_lo + e_hi)).ravel() / a**2
            i_lo = idx[tuple(sl_lo)].ravel(); i_hi = idx[tuple(sl_hi)].ravel()
            add(i_lo, i_hi, e_face); add(i_hi, i_lo, e_face); add(i_lo, i_lo, -e_face); add(i_hi, i_hi, -e_face)
        rows = np.concatenate(rows); cols = np.concatenate(cols); vals = np.concatenate(vals)
        L = sp.csr_matrix((vals, (rows, cols)), shape=(idx.size, idx.size))
        # impose Dirichlet rows
        keep = ~self.dir_mask
        D = sp.diags(keep.astype(float)); L = D @ L
        L = L + sp.diags(self.dir_mask.astype(float))
        self.L = L.tocsr()
        # doping (fixed charge) per site on the core: S/D extensions only
        nd = np.zeros((Nx, Ny, Nz)); nd_site = p.N_D * 1e6 * a**3
        nd[:ix_g0][core[:ix_g0]] = nd_site; nd[ix_g1:][core[ix_g1:]] = nd_site
        self.ND_site = nd.ravel()[self.core_idx]

    def solve(self, n_site, phi_old, V_T):
        """Newton iterations for phi (V) given electron density per core site n_site (>0)."""
        a = self.a; p = self.p
        phi = phi_old.copy(); nall = self.dir_mask.size
        n_old_full = np.zeros(nall); n_old_full[self.core_idx] = n_site
        nd_full = np.zeros(nall); nd_full[self.core_idx] = self.ND_site
        core_full = np.zeros(nall, bool); core_full[self.core_idx] = True
        for it in range(80):
            expo = np.clip((phi - phi_old) / V_T, -20, 20)
            n_lin = np.where(core_full, n_old_full * np.exp(expo), 0.0)
            rho = Q * (nd_full - n_lin) / a**3                       # C/m^3
            F = self.L @ phi + np.where(self.dir_mask, 0.0, rho)
            F[self.dir_mask] = phi[self.dir_mask] - self.dir_val[self.dir_mask]
            dn_dphi = np.where(self.dir_mask, 0.0, -Q * n_lin / (V_T * a**3))
            J = self.L + sp.diags(dn_dphi)
            dphi = spla.spsolve(J.tocsc(), -F)
            dphi = np.clip(dphi, -2 * V_T, 2 * V_T)                 # damped Newton (<= 2 kT/q per step)
            phi = phi + dphi
            if np.abs(dphi).max() < 1e-8: break
        return phi

    def core_potential(self, phi):
        """Electron potential energy U = -phi (eV) on the core, shaped (Nx, Nc*Nc)."""
        return (-phi[self.core_idx]).reshape(self.Nx, self.Nc * self.Nc)


# ============================================================================ [TS-7] SCF
def scf(p: GAAParams, verbose=True, U_lead=None, phi_init=None):
    t0 = time.time()
    if U_lead is None:
        U_lead = neutral_lead_shift(p)
    phi_S = -U_lead; phi_D = phi_S + p.V_ds
    if verbose:
        print(f"[TS-1] lead: N_D={p.N_D:.1e} cm^-3 -> band offset U_lead = {U_lead:+.4f} eV (E_F - E_c,lead = {-U_lead:.3f} eV)  [{time.time()-t0:.1f}s]", flush=True)
    pois = Poisson3D(p, phi_S, phi_D)
    devs = [Device(p, mx, my, mz, g, U_lead, U_lead - p.V_ds) for (mx, my, mz, g) in p.valley_list]
    # initial potential: linear S->D in x, on the whole box
    if phi_init is None:
        xg = np.arange(pois.Nx) / (pois.Nx - 1)
        phi = np.broadcast_to((phi_S + (phi_D - phi_S) * xg)[:, None, None], pois.shape).ravel().copy()
        phi[pois.dir_mask] = pois.dir_val[pois.dir_mask]
    else:
        phi = phi_init.copy(); phi[pois.dir_mask] = pois.dir_val[pois.dir_mask]
    V_T = p.kT
    hist = []
    for it in range(p.scf_maxiter):
        t1 = time.time()
        U = pois.core_potential(phi)
        n = np.zeros(pois.Nx * pois.Nc**2)
        for d in devs:
            d.set_potential(U); n += d.density()
        phi_new = pois.solve(n, phi, V_T)
        dphi = np.abs(phi_new - phi).max()
        phi = phi + p.mixing * (phi_new - phi)
        hist.append(dphi)
        if verbose:
            nev = sum(getattr(d, '_n_neq_eval', 0) for d in devs)
            print(f"   SCF {it:2d}  max|dphi| = {dphi:.2e} V   Ne = {n.sum():8.3f}   neq evals = {nev:4d}   [{time.time()-t1:.1f}s/iter, {time.time()-t0:.0f}s total]", flush=True)
        if dphi < p.scf_tol: break
    U = pois.core_potential(phi)
    for d in devs: d.set_potential(U)
    return dict(devs=devs, pois=pois, phi=phi, U=U, n=n, U_lead=U_lead, hist=hist, time=time.time() - t0)


def post_process(p, res, n_E=201, verbose=True, pool=None):
    """[TS-8] tbtrans-like: T(E), current, LDOS along x."""
    devs = res['devs']; kT = p.kT
    lo = min(devs[0].mu_L, devs[0].mu_R) - 12 * kT; hi = max(devs[0].mu_L, devs[0].mu_R) + 12 * kT
    Es = np.linspace(lo, hi, n_E)
    I = 0.0; T_tot = np.zeros_like(Es)
    for d in devs:
        Ii, T = d.current(Es); I += Ii; T_tot += T
    if verbose: print(f"[TS-8] I_ds = {I*1e6:.4f} uA   (G = {I/max(p.V_ds,1e-9)/G0:.3f} G0 if linear)")
    return Es, T_tot, I


def id_vg_sweep(p: GAAParams, Vgs, verbose=True):
    out = []
    U_lead = neutral_lead_shift(p); phi = None
    for Vg in Vgs:
        pp = GAAParams(**{**p.__dict__, 'V_g': Vg})
        res = scf(pp, verbose=False, U_lead=U_lead, phi_init=phi); phi = res['phi']
        Es, T, I = post_process(pp, res, verbose=False)
        out.append(I)
        if verbose: print(f"   V_g = {Vg:+.2f} V   I_d = {I*1e6:9.4f} uA   ({len(res['hist'])} SCF iters, {res['time']:.0f} s)")
    return np.array(out)


# ============================================================================ main
def plot_single(p, res, Es, T):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    pois = res['pois']; U = res['U']
    x = np.arange(pois.Nx) * p.a
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    Ec_x = U.min(axis=1)                            # lowest core potential per slice
    sub0 = np.linalg.eigvalsh(res['devs'][0].H_cs)[0]
    ax[0, 0].plot(x, Ec_x + sub0, label="lowest subband edge E_c(x)")
    ax[0, 0].axhline(0, c="k", ls="--", lw=.8, label="mu_S"); ax[0, 0].axhline(-p.V_ds, c="k", ls=":", lw=.8, label="mu_D")
    ax[0, 0].set_xlabel("x (nm)"); ax[0, 0].set_ylabel("eV"); ax[0, 0].legend(); ax[0, 0].set_title(f"Band profile  Vg={p.V_g} V, Vds={p.V_ds} V")
    n_x = res['n'].reshape(pois.Nx, -1).sum(1) / (p.a * 1e-9) * 1e-9
    ax[0, 1].semilogy(x, n_x); ax[0, 1].set_xlabel("x (nm)"); ax[0, 1].set_ylabel("electrons / nm"); ax[0, 1].set_title("Line density")
    ax[1, 0].plot(Es, T); ax[1, 0].set_xlabel("E (eV)"); ax[1, 0].set_ylabel("T(E)"); ax[1, 0].set_title("Transmission (sum over valleys x spin)")
    ax[1, 0].axvspan(-p.V_ds, 0, alpha=.15, color="orange")
    Eld = np.linspace(min(Ec_x.min() + sub0 - 0.1, -p.V_ds - 0.2), 0.4, 80)
    ld = sum(d.ldos_x(Eld) for d in res['devs'])
    im = ax[1, 1].imshow(np.log10(ld + 1e-6), origin="lower", aspect="auto", extent=[x[0], x[-1], Eld[0], Eld[-1]], cmap="inferno")
    ax[1, 1].plot(x, Ec_x + sub0, "c--", lw=.8); ax[1, 1].set_xlabel("x (nm)"); ax[1, 1].set_ylabel("E (eV)"); ax[1, 1].set_title("log10 LDOS(x,E)")
    fig.colorbar(im, ax=ax[1, 1]); fig.tight_layout(); fig.savefig("level2_gaa_single_bias.png", dpi=130)
    print("   saved level2_gaa_single_bias.png")
    # cross-section potential in the middle of the gate
    fig2, ax2 = plt.subplots(figsize=(4.5, 4))
    phi = res['phi'].reshape(pois.shape); im = ax2.imshow(phi[pois.Nx // 2].T, origin="lower", cmap="viridis",
                                                        extent=[0, pois.Ny * p.a, 0, pois.Nz * p.a])
    ax2.set_title("phi(y,z) mid-channel (V)"); ax2.set_xlabel("y (nm)"); ax2.set_ylabel("z (nm)"); fig2.colorbar(im, ax=ax2)
    fig2.tight_layout(); fig2.savefig("level2_gaa_cross_section_phi.png", dpi=130); print("   saved level2_gaa_cross_section_phi.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Level-2: TranSIESTA-style NEGF-Poisson for a GAA Si nanowire")
    ap.add_argument("--quick", action="store_true", help="small device, single valley, one bias point")
    ap.add_argument("--valleys", default=None, choices=["single", "si3"])
    ap.add_argument("--Vg", type=float, default=None); ap.add_argument("--Vds", type=float, default=None)
    ap.add_argument("--density", default=None, choices=["contour", "ozaki"])
    ap.add_argument("--workers", type=int, default=1, help="processes for the energy loop")
    ap.add_argument("--sweep", action="store_true", help="Id-Vg sweep (slow)")
    args = ap.parse_args()
    p = GAAParams()
    if args.quick:
        p = GAAParams(W=1.8, L_s=3.0, L_g=6.0, L_d=3.0, n_circle=16, n_line=9, n_pole=6, scf_maxiter=30)
    if args.valleys: p.valleys = args.valleys
    if args.Vg is not None: p.V_g = args.Vg
    if args.Vds is not None: p.V_ds = args.Vds
    if args.density: p.density_method = args.density
    p.n_workers = args.workers
    Nc = int(round(p.W / p.a)); Nx = int(round((p.L_s + p.L_g + p.L_d) / p.a))
    print(f"Device: Si core {p.W}x{p.W} nm ({Nc}x{Nc}={Nc*Nc} orbitals/slice), {Nx} slices, t_ox={p.t_ox} nm, valleys={p.valleys}")
    print(f"        density method = {p.density_method}, eq points = {p.n_circle}+{p.n_line}+{p.n_pole}, neq: adaptive (tol {p.neq_tol}), eta = {p.eta} eV, workers = {p.n_workers}")
    T0 = time.time()
    res = scf(p)
    Es, T, I = post_process(p, res)
    plot_single(p, res, Es, T)
    print(f"Single bias point total: {time.time()-T0:.1f} s")
    if args.sweep:
        Vgs = np.linspace(0.0, 0.8, 5)
        Ids = id_vg_sweep(p, Vgs)
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 4)); ax.semilogy(Vgs, np.abs(Ids) * 1e6, "o-")
        ax.set_xlabel("V_g (V)"); ax.set_ylabel("I_d (uA)"); ax.set_title(f"GAA Si NW, Vds={p.V_ds} V"); fig.tight_layout(); fig.savefig("level2_gaa_IdVg.png", dpi=130)
        print("   saved level2_gaa_IdVg.png")
    print(f"Total wall time: {time.time()-T0:.1f} s")
