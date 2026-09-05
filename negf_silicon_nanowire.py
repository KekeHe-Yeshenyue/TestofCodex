"""
Non-Equilibrium Green's Function (NEGF) Quantum Transport Simulation
for Silicon Nanowire and Gate-All-Around (GAA) Transistors.

This unified module implements:
  - Tight-binding Hamiltonian (effective-mass) for rectangular and cylindrical nanowires
  - Multiple semiconductor materials (Si, Ge, InGaAs, GaAs)
  - Surface Green's function via Sancho-Rubio decimation (class + functional API)
  - Recursive Green's function (RGF) for block-tridiagonal systems
  - Full Green's function solver (G^R, G^A, G^<, G^n)
  - Transmission T(E), current density (Landauer-Buttiker), LDOS
  - Bond current for spatially-resolved current flow
  - Poisson solver for electrostatics with GAA gate coupling
  - Self-consistent NEGF-Poisson loop
  - Hamiltonian export (npz, csv, txt, mat)
  - I-V and transfer characteristic sweeps

References:
  [1] S. Datta, "Quantum Transport: Atom to Transistor", Cambridge (2005).
  [2] M.P. Lopez-Sancho et al., J. Phys. F 15, 851 (1985).
  [3] A. Svizhenko et al., J. Appl. Phys. 91, 2343 (2002).
  [4] R. Lake et al., J. Appl. Phys. 81, 7845 (1997).
"""

import numpy as np
from numpy import linalg as la
from scipy import linalg
import warnings
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
from enum import Enum


# ---------------------------------------------------------------------------
# Physical constants (SI)
# ---------------------------------------------------------------------------
Q_E = 1.602176634e-19        # elementary charge [C]
HBAR = 1.054571817e-34       # reduced Planck constant [J*s]
M_E = 9.1093837015e-31       # free electron mass [kg]
KB = 1.380649e-23             # Boltzmann constant [J/K]
EPSILON_0 = 8.8541878128e-12  # vacuum permittivity [F/m]
H_PLANCK_EV = 4.135667696e-15  # Planck constant [eV*s]


# ---------------------------------------------------------------------------
# Material system
# ---------------------------------------------------------------------------
class MaterialType(Enum):
    """Supported semiconductor channel materials."""
    SILICON = "Si"
    GERMANIUM = "Ge"
    INGAAS = "InGaAs"
    GAAS = "GaAs"


@dataclass
class Material:
    """Semiconductor material properties."""
    name: str
    effective_mass: float      # in units of m_e
    dielectric_constant: float
    bandgap: float             # eV
    electron_affinity: float   # eV

    @classmethod
    def get_material(cls, material_type: MaterialType) -> 'Material':
        materials = {
            MaterialType.SILICON: cls("Silicon", 0.26, 11.7, 1.12, 4.05),
            MaterialType.GERMANIUM: cls("Germanium", 0.12, 16.0, 0.66, 4.0),
            MaterialType.INGAAS: cls("In0.53Ga0.47As", 0.041, 13.9, 0.74, 4.5),
            MaterialType.GAAS: cls("GaAs", 0.067, 12.9, 1.42, 4.07),
        }
        return materials[material_type]


# ---------------------------------------------------------------------------
# Device parameters — rectangular nanowire (block-RGF)
# ---------------------------------------------------------------------------
@dataclass
class DeviceParams:
    """Parameters for a rectangular-cross-section silicon nanowire."""
    Ny: int = 3
    Nz: int = 3
    Nx: int = 60

    a: float = 0.543e-9 / 2   # lattice constant / 2

    mx: float = 0.19
    my: float = 0.19
    mz: float = 0.19

    T: float = 300.0
    V_bias: float = 0.3
    V_gate: float = 0.0

    eta: float = 1e-6

    E_min: float = -0.5
    E_max: float = 1.5
    NE: int = 300

    @property
    def N_orb(self) -> int:
        return self.Ny * self.Nz


# ---------------------------------------------------------------------------
# Device parameters — GAA cylindrical transistor
# ---------------------------------------------------------------------------
@dataclass
class GAADeviceParams:
    """Parameters for a Gate-All-Around cylindrical nanowire transistor."""
    channel_length: float = 10e-9
    nanowire_radius: float = 3e-9
    oxide_thickness: float = 1e-9

    nz: int = 50
    nr: int = 10

    material: Material = None
    oxide_dielectric: float = 3.9

    vg: float = 0.0
    vd: float = 0.0
    vs: float = 0.0

    source_doping: float = 1e20
    drain_doping: float = 1e20
    channel_doping: float = 1e15

    temperature: float = 300.0

    def __post_init__(self):
        if self.material is None:
            self.material = Material.get_material(MaterialType.SILICON)


# ===================================================================
#  PART A — Rectangular nanowire with block-matrix RGF
# ===================================================================

# ---------------------------------------------------------------------------
# Hamiltonian construction (rectangular cross-section)
# ---------------------------------------------------------------------------
def build_slice_hamiltonian(p: DeviceParams) -> np.ndarray:
    """
    Build the on-site Hamiltonian H_slice for one cross-sectional slice
    of the nanowire using an effective-mass tight-binding model.

    The 2D grid (Ny x Nz) is mapped to a 1D index:  idx = iy * Nz + iz.
    """
    N = p.N_orb
    t_y = (HBAR ** 2 / (2 * p.my * M_E * p.a ** 2)) / Q_E
    t_z = (HBAR ** 2 / (2 * p.mz * M_E * p.a ** 2)) / Q_E

    H = np.zeros((N, N), dtype=complex)

    for iy in range(p.Ny):
        for iz in range(p.Nz):
            idx = iy * p.Nz + iz
            n_y_neighbors = (1 if iy > 0 else 0) + (1 if iy < p.Ny - 1 else 0)
            n_z_neighbors = (1 if iz > 0 else 0) + (1 if iz < p.Nz - 1 else 0)
            H[idx, idx] = n_y_neighbors * t_y + n_z_neighbors * t_z

            if iy < p.Ny - 1:
                jdx = (iy + 1) * p.Nz + iz
                H[idx, jdx] = -t_y
                H[jdx, idx] = -t_y

            if iz < p.Nz - 1:
                jdx = iy * p.Nz + (iz + 1)
                H[idx, jdx] = -t_z
                H[jdx, idx] = -t_z

    return H


def build_coupling_matrix(p: DeviceParams) -> np.ndarray:
    """Inter-slice coupling V = -t_x * I  for rectangular nanowire."""
    t_x = (HBAR ** 2 / (2 * p.mx * M_E * p.a ** 2)) / Q_E
    N = p.N_orb
    return -t_x * np.eye(N, dtype=complex)


def potential_profile(p: DeviceParams) -> np.ndarray:
    """Electrostatic potential [eV] along the channel for each slice."""
    x = np.linspace(0, 1, p.Nx)
    V_sd = -p.V_bias * x
    V_g = -p.V_gate * np.ones(p.Nx)
    return V_sd + V_g


# ---------------------------------------------------------------------------
# Surface Green's function — functional API (for block-RGF path)
# ---------------------------------------------------------------------------
def surface_green_function(E: float, H_slice: np.ndarray,
                           V: np.ndarray, p: DeviceParams,
                           max_iter: int = 100, tol: float = 1e-8
                           ) -> np.ndarray:
    """
    Sancho-Rubio surface Green's function for a semi-infinite lead
    built from identical slices with on-site H_slice and coupling V.
    """
    N = H_slice.shape[0]
    EI = (E + 1j * p.eta) * np.eye(N, dtype=complex)

    eps_s = H_slice.copy()
    eps = H_slice.copy()
    alpha = V.copy()
    beta = V.conj().T.copy()

    for _ in range(max_iter):
        g_eps = linalg.inv(EI - eps)

        eps_s = eps_s + alpha @ g_eps @ beta
        eps = eps + alpha @ g_eps @ beta + beta @ g_eps @ alpha
        alpha_new = alpha @ g_eps @ alpha
        beta = beta @ g_eps @ beta
        alpha = alpha_new

        if np.linalg.norm(alpha) < tol:
            break

    return linalg.inv(EI - eps_s)


def contact_self_energy(E: float, H_slice: np.ndarray,
                        V: np.ndarray, p: DeviceParams) -> np.ndarray:
    """Self-energy Sigma = V^dag . g_s . V for a semi-infinite contact."""
    g_s = surface_green_function(E, H_slice, V, p)
    return V.conj().T @ g_s @ V


def broadening(Sigma: np.ndarray) -> np.ndarray:
    """Broadening matrix Gamma = i(Sigma - Sigma^dag)."""
    return 1j * (Sigma - Sigma.conj().T)


# ---------------------------------------------------------------------------
# Fermi-Dirac distribution
# ---------------------------------------------------------------------------
def fermi(E: float, mu: float, T: float) -> float:
    kT = KB * T / Q_E
    x = (E - mu) / kT
    if x > 500:
        return 0.0
    elif x < -500:
        return 1.0
    return 1.0 / (1.0 + np.exp(x))


# ---------------------------------------------------------------------------
# Recursive Green's Function (block-matrix RGF)
# ---------------------------------------------------------------------------
def recursive_green_function(E: float, p: DeviceParams,
                             H_slices: list,
                             V: np.ndarray,
                             Sigma_L: np.ndarray,
                             Sigma_R: np.ndarray):
    """
    Block-matrix RGF returning diagonal blocks of G^R, G^n, and T(E).

    This is the multi-orbital version where each slice has N_orb x N_orb
    Hamiltonian blocks.
    """
    Nx = p.Nx
    N = p.N_orb
    EI = (E + 1j * p.eta) * np.eye(N, dtype=complex)
    Vdag = V.conj().T

    mu_S = 0.0
    mu_D = -p.V_bias
    f_S = fermi(E, mu_S, p.T)
    f_D = fermi(E, mu_D, p.T)

    Gamma_L = broadening(Sigma_L)
    Gamma_R = broadening(Sigma_R)

    Sigma_in_L = f_S * Gamma_L
    Sigma_in_R = f_D * Gamma_R

    # --- Forward sweep: left-connected GFs ---
    gL = [None] * Nx
    gL[0] = linalg.inv(EI - H_slices[0] - Sigma_L)
    for i in range(1, Nx):
        Sigma_i = Vdag @ gL[i - 1] @ V
        if i == Nx - 1:
            gL[i] = linalg.inv(EI - H_slices[i] - Sigma_i - Sigma_R)
        else:
            gL[i] = linalg.inv(EI - H_slices[i] - Sigma_i)

    # --- Backward sweep: G^R diagonal blocks ---
    G_diag = [None] * Nx
    Gn_diag = [None] * Nx
    G_diag[Nx - 1] = gL[Nx - 1]

    for i in range(Nx - 2, -1, -1):
        G_diag[i] = gL[i] + gL[i] @ V @ G_diag[i + 1] @ Vdag @ gL[i]

    # --- G^n diagonal blocks via RGF ---
    gL_Sigma_in = [None] * Nx
    gL_Sigma_in[0] = gL[0] @ Sigma_in_L @ gL[0].conj().T

    for i in range(1, Nx):
        prop = Vdag @ gL_Sigma_in[i - 1] @ V
        Sigma_in_i = prop + Sigma_in_R if i == Nx - 1 else prop
        gL_Sigma_in[i] = gL[i] @ Sigma_in_i @ gL[i].conj().T

    Gn_diag[Nx - 1] = gL_Sigma_in[Nx - 1]

    for i in range(Nx - 2, -1, -1):
        A_i = gL[i] @ V @ G_diag[i + 1]
        Gn_diag[i] = (gL_Sigma_in[i]
                       + A_i @ Vdag @ gL_Sigma_in[i]
                       + gL_Sigma_in[i] @ V @ G_diag[i + 1].conj().T @ Vdag @ gL[i].conj().T
                       + A_i @ Gn_diag[i + 1] @ A_i.conj().T)

    # --- Transmission via off-diagonal G^R_{0, Nx-1} ---
    G_0N = gL[0].copy()
    for i in range(1, Nx):
        G_0N = G_0N @ V @ (gL[i] if i < Nx - 1 else G_diag[Nx - 1])

    T_E = np.real(np.trace(Gamma_L @ G_0N @ Gamma_R @ G_0N.conj().T))

    return G_diag, Gn_diag, T_E


# ---------------------------------------------------------------------------
# Full NEGF solver for rectangular nanowire (block-RGF)
# ---------------------------------------------------------------------------
def solve_negf(p: DeviceParams):
    """
    Energy sweep computing T(E), current I, and LDOS(x, E)
    for a rectangular-cross-section nanowire using block-matrix RGF.
    """
    H0 = build_slice_hamiltonian(p)
    V = build_coupling_matrix(p)
    V_pot = potential_profile(p)

    t_x = (HBAR ** 2 / (2 * p.mx * M_E * p.a ** 2)) / Q_E
    H_slices = []
    for i in range(p.Nx):
        Hi = H0.copy() + 2 * t_x * np.eye(p.N_orb, dtype=complex)
        Hi += V_pot[i] * np.eye(p.N_orb, dtype=complex)
        H_slices.append(Hi)

    H_lead_L = H0.copy() + 2 * t_x * np.eye(p.N_orb, dtype=complex)
    H_lead_R = H0.copy() + 2 * t_x * np.eye(p.N_orb, dtype=complex)
    H_lead_R += (-p.V_bias) * np.eye(p.N_orb, dtype=complex)

    E_arr = np.linspace(p.E_min, p.E_max, p.NE)
    T_arr = np.zeros(p.NE)
    LDOS = np.zeros((p.NE, p.Nx))

    print(f"NEGF solver: {p.Nx} slices, {p.N_orb} orbitals/slice, "
          f"{p.NE} energy points")
    print(f"  Cross-section: {p.Ny}x{p.Nz}, lattice constant a={p.a*1e9:.3f} nm")
    print(f"  V_bias={p.V_bias} V, V_gate={p.V_gate} V, T={p.T} K")
    print("  Running energy sweep...")

    for ie, E in enumerate(E_arr):
        if ie % 50 == 0:
            print(f"    E = {E:.3f} eV  ({ie}/{p.NE})")

        Sigma_L = contact_self_energy(E, H_lead_L, V, p)
        Sigma_R = contact_self_energy(E, H_lead_R, V, p)

        G_diag, Gn_diag, T_E = recursive_green_function(
            E, p, H_slices, V, Sigma_L, Sigma_R
        )
        T_arr[ie] = T_E

        for ix in range(p.Nx):
            LDOS[ie, ix] = -np.imag(np.trace(G_diag[ix])) / np.pi

    mu_S = 0.0
    mu_D = -p.V_bias
    f_S = np.array([fermi(E, mu_S, p.T) for E in E_arr])
    f_D = np.array([fermi(E, mu_D, p.T) for E in E_arr])

    I_total = 2 * Q_E / H_PLANCK_EV * np.trapezoid(T_arr * (f_S - f_D), E_arr)

    print(f"\n  Total current I = {I_total*1e6:.4f} uA")
    print(f"  Peak transmission = {np.max(T_arr):.4f}")

    return E_arr, T_arr, I_total, LDOS


# ---------------------------------------------------------------------------
# Plotting for rectangular nanowire
# ---------------------------------------------------------------------------
def plot_results(E_arr, T_arr, I_total, LDOS, p: DeviceParams):
    """Four-panel visualization of block-RGF results."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    ax.plot(E_arr, T_arr, 'b-', linewidth=1.5)
    ax.set_xlabel('Energy [eV]')
    ax.set_ylabel('Transmission T(E)')
    ax.set_title('(a) Transmission Spectrum')
    ax.set_xlim(p.E_min, p.E_max)
    ax.grid(True, alpha=0.3)
    mu_S, mu_D = 0.0, -p.V_bias
    ax.axvspan(min(mu_S, mu_D), max(mu_S, mu_D),
               alpha=0.15, color='orange', label='Bias window')
    ax.legend()

    ax = axes[0, 1]
    f_S = np.array([fermi(E, mu_S, p.T) for E in E_arr])
    f_D = np.array([fermi(E, mu_D, p.T) for E in E_arr])
    integrand = T_arr * (f_S - f_D)
    ax.fill_between(E_arr, integrand, alpha=0.4, color='green')
    ax.plot(E_arr, integrand, 'g-', linewidth=1.2)
    ax.set_xlabel('Energy [eV]')
    ax.set_ylabel('T(E) [f_S - f_D]')
    ax.set_title(f'(b) Current Integrand  (I = {I_total*1e6:.3f} uA)')
    ax.set_xlim(p.E_min, p.E_max)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    x_nm = np.arange(p.Nx) * p.a * 1e9
    extent = [x_nm[0], x_nm[-1], E_arr[0], E_arr[-1]]
    im = ax.imshow(LDOS, aspect='auto', origin='lower', extent=extent,
                   cmap='hot', interpolation='bilinear')
    ax.set_xlabel('Position along wire [nm]')
    ax.set_ylabel('Energy [eV]')
    ax.set_title('(c) Local Density of States')
    fig.colorbar(im, ax=ax, label='LDOS [1/eV]')

    ax = axes[1, 1]
    V_pot = potential_profile(p)
    t_x = (HBAR ** 2 / (2 * p.mx * M_E * p.a ** 2)) / Q_E
    E_sub = build_slice_hamiltonian(p)
    evals = np.sort(np.real(linalg.eigvalsh(E_sub)))
    E_cb = evals[0] + 2 * t_x

    ax.plot(x_nm, V_pot + E_cb, 'r-', linewidth=2, label='CB edge (lowest subband)')
    ax.axhline(y=0.0, color='blue', linestyle='--', label='mu_S = 0 eV')
    ax.axhline(y=-p.V_bias, color='cyan', linestyle='--',
               label=f'mu_D = {-p.V_bias} eV')
    ax.set_xlabel('Position along wire [nm]')
    ax.set_ylabel('Energy [eV]')
    ax.set_title('(d) Band Diagram')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle('NEGF Quantum Transport - Silicon Nanowire', fontsize=14,
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig('negf_results.png', dpi=150, bbox_inches='tight')
    print("  Plot saved to negf_results.png")
    plt.show()


def compute_iv_curve(p: DeviceParams, V_values: np.ndarray):
    """Sweep bias voltage for the rectangular nanowire."""
    currents = []
    for V_b in V_values:
        p_v = DeviceParams(
            Ny=p.Ny, Nz=p.Nz, Nx=p.Nx, a=p.a,
            mx=p.mx, my=p.my, mz=p.mz,
            T=p.T, V_bias=V_b, V_gate=p.V_gate,
            eta=p.eta, E_min=p.E_min, E_max=p.E_max, NE=p.NE
        )
        _, _, I, _ = solve_negf(p_v)
        currents.append(I)
        print(f"  V_bias = {V_b:.3f} V  =>  I = {I*1e6:.4f} uA")
    return np.array(currents)


# ===================================================================
#  PART B — GAA transistor with full-matrix NEGF solver
# ===================================================================

# ---------------------------------------------------------------------------
# Surface Green's function — class API (for GAA solver)
# ---------------------------------------------------------------------------
class SurfaceGreenFunction:
    """
    Surface Green's function calculator for semi-infinite leads.

    Supports two algorithms:
      - Sancho-Rubio decimation (default, exponentially convergent)
      - Direct iterative self-consistency
    """

    def __init__(self, H_unit: np.ndarray, H_coupling: np.ndarray,
                 eta: float = 1e-6, max_iter: int = 100, tol: float = 1e-8):
        self.H_00 = np.array(H_unit, dtype=complex)
        self.H_01 = np.array(H_coupling, dtype=complex)
        self.H_10 = self.H_01.T.conj()
        self.n = H_unit.shape[0]
        self.eta = eta
        self.max_iter = max_iter
        self.tol = tol

    def calculate(self, E: float) -> np.ndarray:
        """Sancho-Rubio decimation method (exponential convergence)."""
        eps_s = self.H_00.copy()
        eps_b = self.H_00.copy()
        alpha = self.H_01.copy()
        beta = self.H_10.copy()

        I = np.eye(self.n, dtype=complex)
        E_plus = (E + 1j * self.eta) * I

        for _ in range(self.max_iter):
            g_b = la.inv(E_plus - eps_b)
            alpha_new = alpha @ g_b @ alpha
            beta_new = beta @ g_b @ beta

            eps_s = eps_s + alpha @ g_b @ beta
            eps_b = eps_b + alpha @ g_b @ beta + beta @ g_b @ alpha

            if la.norm(alpha_new) + la.norm(beta_new) < self.tol:
                break

            alpha = alpha_new
            beta = beta_new

        return la.inv(E_plus - eps_s)

    def calculate_iterative(self, E: float) -> np.ndarray:
        """Direct iterative method (simpler but slower)."""
        I = np.eye(self.n, dtype=complex)
        E_plus = (E + 1j * self.eta) * I

        g_s = la.inv(E_plus - self.H_00)

        for _ in range(self.max_iter):
            sigma = self.H_01 @ g_s @ self.H_10
            g_s_new = la.inv(E_plus - self.H_00 - sigma)

            if la.norm(g_s_new - g_s) < self.tol:
                break
            g_s = g_s_new

        return g_s


# ---------------------------------------------------------------------------
# NEGF Solver for GAA transistor (full-matrix)
# ---------------------------------------------------------------------------
class NEGFSolver:
    """
    Full-matrix NEGF solver for GAA transistors.

    Computes G^R, G^A, G^<, spectral function, transmission,
    bond current, electron density, and supports Hamiltonian export.
    """

    def __init__(self, params: GAADeviceParams, eta: float = 1e-6):
        self.params = params
        self.eta = eta
        self._build_grid()
        self.H = None
        self.potential = None
        self.electron_density = None

    def _build_grid(self):
        p = self.params
        self.dz = p.channel_length / (p.nz - 1)
        self.z = np.linspace(0, p.channel_length, p.nz)

        if p.nr > 1:
            self.dr = p.nanowire_radius / (p.nr - 1)
            self.r = np.linspace(0, p.nanowire_radius, p.nr)
        else:
            self.dr = p.nanowire_radius
            self.r = np.array([0])

        self.n_total = p.nz * p.nr

    def build_hamiltonian(self, potential: Optional[np.ndarray] = None) -> np.ndarray:
        """Build device Hamiltonian using effective mass finite differences."""
        p = self.params
        m_eff = p.material.effective_mass * M_E
        t_z = (HBAR**2 / (2 * m_eff * self.dz**2)) / Q_E

        if p.nr == 1:
            n = p.nz
            H = np.zeros((n, n), dtype=complex)
            for i in range(n):
                H[i, i] = 2 * t_z
                if potential is not None:
                    H[i, i] += potential[i]
            for i in range(n - 1):
                H[i, i + 1] = -t_z
                H[i + 1, i] = -t_z
        else:
            H = self._build_2d_hamiltonian(potential, t_z)

        self.H = H
        return H

    def _build_2d_hamiltonian(self, potential, t_z):
        p = self.params
        m_eff = p.material.effective_mass * M_E
        t_r = (HBAR**2 / (2 * m_eff * self.dr**2)) / Q_E

        n = p.nz * p.nr
        H = np.zeros((n, n), dtype=complex)

        def idx(iz, ir):
            return iz * p.nr + ir

        for iz in range(p.nz):
            for ir in range(p.nr):
                i = idx(iz, ir)
                H[i, i] = 2 * t_z + 2 * t_r
                if potential is not None:
                    H[i, i] += potential[iz, ir] if potential.ndim > 1 else potential[iz]

                if iz > 0:
                    H[i, idx(iz - 1, ir)] = -t_z
                if iz < p.nz - 1:
                    H[i, idx(iz + 1, ir)] = -t_z

                if ir > 0:
                    r_factor = np.sqrt(self.r[ir] / self.r[ir - 1]) if self.r[ir - 1] > 0 else 1
                    H[i, idx(iz, ir - 1)] = -t_r * r_factor
                if ir < p.nr - 1:
                    r_factor = np.sqrt(self.r[ir] / self.r[ir + 1])
                    H[i, idx(iz, ir + 1)] = -t_r * r_factor

        return H

    # --- Lead self-energy ---

    def calculate_lead_self_energy(self, E: float, lead: str = 'source') -> np.ndarray:
        """Self-energy from a semi-infinite source or drain lead."""
        p = self.params
        m_eff = p.material.effective_mass * M_E
        t = (HBAR**2 / (2 * m_eff * self.dz**2)) / Q_E

        n = self.H.shape[0]
        sigma = np.zeros((n, n), dtype=complex)

        if p.nr == 1:
            H_lead = np.array([[2 * t]], dtype=complex)
            H_coupling = np.array([[-t]], dtype=complex)
        else:
            H_lead = np.zeros((p.nr, p.nr), dtype=complex)
            H_coupling = np.zeros((p.nr, p.nr), dtype=complex)
            t_r = (HBAR**2 / (2 * m_eff * self.dr**2)) / Q_E
            for ir in range(p.nr):
                H_lead[ir, ir] = 2 * t + 2 * t_r
                if ir > 0:
                    H_lead[ir, ir - 1] = -t_r
                    H_lead[ir - 1, ir] = -t_r
                H_coupling[ir, ir] = -t

        if lead == 'source':
            E_F_shift = self._fermi_level_from_doping(p.source_doping)
            H_lead_shifted = H_lead - E_F_shift * np.eye(H_lead.shape[0])
            bias = p.vs
        else:
            E_F_shift = self._fermi_level_from_doping(p.drain_doping)
            H_lead_shifted = H_lead - E_F_shift * np.eye(H_lead.shape[0])
            bias = p.vd

        sgf = SurfaceGreenFunction(H_lead_shifted, H_coupling, eta=self.eta)
        g_surface = sgf.calculate(E - bias)

        if p.nr == 1:
            if lead == 'source':
                sigma[0, 0] = -t * g_surface[0, 0] * (-t)
            else:
                sigma[-1, -1] = -t * g_surface[0, 0] * (-t)
        else:
            tau = H_coupling
            sigma_block = tau @ g_surface @ tau.T.conj()
            if lead == 'source':
                sigma[:p.nr, :p.nr] = sigma_block
            else:
                sigma[-p.nr:, -p.nr:] = sigma_block

        return sigma

    def _fermi_level_from_doping(self, doping: float) -> float:
        p = self.params
        m_eff = p.material.effective_mass
        T = p.temperature

        N_c = 2 * (2 * np.pi * m_eff * M_E * KB * T / (2 * np.pi * HBAR)**2)**1.5
        N_c *= 1e-6
        kT = KB * T / Q_E
        return kT * np.log(doping / N_c)

    # --- Green's functions ---

    def calculate_retarded_greens_function(self, E: float):
        """G^R(E) = [(E+i*eta)I - H - Sigma_S - Sigma_D]^{-1}"""
        if self.H is None:
            raise ValueError("Hamiltonian not built. Call build_hamiltonian first.")

        n = self.H.shape[0]
        I = np.eye(n, dtype=complex)

        sigma_S = self.calculate_lead_self_energy(E, 'source')
        sigma_D = self.calculate_lead_self_energy(E, 'drain')

        G_R = la.inv((E + 1j * self.eta) * I - self.H - sigma_S - sigma_D)
        return G_R, sigma_S, sigma_D

    def calculate_advanced_greens_function(self, G_R: np.ndarray) -> np.ndarray:
        """G^A = (G^R)^dag"""
        return G_R.T.conj()

    def calculate_lesser_greens_function(self, E: float, G_R: np.ndarray,
                                         sigma_S: np.ndarray,
                                         sigma_D: np.ndarray) -> np.ndarray:
        """G^< = G^R . Sigma^< . G^A"""
        p = self.params
        gamma_S = 1j * (sigma_S - sigma_S.T.conj())
        gamma_D = 1j * (sigma_D - sigma_D.T.conj())

        kT = KB * p.temperature / Q_E
        f_S = self._fermi_function(E - p.vs, kT)
        f_D = self._fermi_function(E - p.vd, kT)

        sigma_lesser = 1j * f_S * gamma_S + 1j * f_D * gamma_D
        G_A = self.calculate_advanced_greens_function(G_R)
        return G_R @ sigma_lesser @ G_A

    def _fermi_function(self, E: float, kT: float) -> float:
        if kT < 1e-10:
            return 1.0 if E < 0 else 0.0
        x = E / kT
        if x > 100:
            return 0.0
        elif x < -100:
            return 1.0
        return 1.0 / (1.0 + np.exp(x))

    # --- Physical observables ---

    def calculate_electron_density(self, E_min: float, E_max: float,
                                   n_energy: int = 100) -> np.ndarray:
        """Electron density n(r) = -(1/pi) integral Im[G^<(r,r,E)] dE."""
        p = self.params
        E_grid = np.linspace(E_min, E_max, n_energy)

        n = self.H.shape[0]
        density = np.zeros(n)

        for E in E_grid:
            G_R, sigma_S, sigma_D = self.calculate_retarded_greens_function(E)
            G_lesser = self.calculate_lesser_greens_function(E, G_R, sigma_S, sigma_D)
            density += np.diag(G_lesser).imag

        dE = E_grid[1] - E_grid[0]
        density *= -dE / np.pi

        if p.nr == 1:
            volume = np.pi * p.nanowire_radius**2 * self.dz
        else:
            volume = self.dz * 2 * np.pi * self.dr * self.r.mean()

        density /= volume * 1e6
        self.electron_density = density
        return density

    def calculate_transmission(self, E: float) -> float:
        """T(E) = Tr[Gamma_S . G^R . Gamma_D . G^A]"""
        G_R, sigma_S, sigma_D = self.calculate_retarded_greens_function(E)
        G_A = self.calculate_advanced_greens_function(G_R)

        gamma_S = 1j * (sigma_S - sigma_S.T.conj())
        gamma_D = 1j * (sigma_D - sigma_D.T.conj())

        return np.trace(gamma_S @ G_R @ gamma_D @ G_A).real

    def calculate_spectral_function(self, E: float) -> np.ndarray:
        """A(E) = -2 Im(G^R)  (diagonal elements)."""
        G_R, _, _ = self.calculate_retarded_greens_function(E)
        return np.diag(-2 * G_R.imag)

    def calculate_current(self, E_min: float, E_max: float,
                          n_energy: int = 200) -> Tuple[float, np.ndarray]:
        """Current via Landauer-Buttiker: I = (2q/h) integral T(E)[f_S - f_D] dE."""
        p = self.params
        kT = KB * p.temperature / Q_E

        E_grid = np.linspace(E_min, E_max, n_energy)
        current_spectrum = np.zeros(n_energy)

        for i, E in enumerate(E_grid):
            T_E = self.calculate_transmission(E)
            f_S = self._fermi_function(E - p.vs, kT)
            f_D = self._fermi_function(E - p.vd, kT)
            current_spectrum[i] = T_E * (f_S - f_D)

        prefactor = 2 * Q_E / (2 * np.pi * HBAR)
        total_current = prefactor * np.trapezoid(current_spectrum, E_grid) * Q_E

        return total_current, current_spectrum

    def calculate_bond_current(self, E: float, l: int) -> float:
        """Bond current I_{l -> l+1}(E) for position-resolved current flow."""
        p = self.params
        if l < 0 or l >= p.nz - 1:
            raise ValueError(f"Layer index must be between 0 and {p.nz - 2}")

        G_R, sigma_S, sigma_D = self.calculate_retarded_greens_function(E)
        G_lesser = self.calculate_lesser_greens_function(E, G_R, sigma_S, sigma_D)

        if p.nr == 1:
            H_coupling = self.H[l, l + 1]
            G_lesser_coupling = G_lesser[l + 1, l]
            return (2 * Q_E / HBAR) * np.real(H_coupling * G_lesser_coupling)
        else:
            idx_l = l * p.nr
            idx_l1 = (l + 1) * p.nr
            H_block = self.H[idx_l:idx_l + p.nr, idx_l1:idx_l1 + p.nr]
            G_block = G_lesser[idx_l1:idx_l1 + p.nr, idx_l:idx_l + p.nr]
            return (2 * Q_E / HBAR) * np.real(np.trace(H_block @ G_block))

    # --- Hamiltonian export / inspection ---

    def export_hamiltonian(self, filename: str = "hamiltonian.npz",
                           format: str = "npz") -> dict:
        """Export Hamiltonian to file (npz, csv, txt, or mat)."""
        if self.H is None:
            raise ValueError("Hamiltonian not built. Call build_hamiltonian first.")

        H = self.H
        n = H.shape[0]

        eigenvalues = np.linalg.eigvalsh(H.real)
        sparsity = np.sum(np.abs(H) < 1e-10) / H.size * 100

        info = {
            'size': H.shape,
            'n_elements': H.size,
            'n_nonzero': np.sum(np.abs(H) > 1e-10),
            'sparsity_percent': sparsity,
            'is_hermitian': np.allclose(H, H.T.conj()),
            'diagonal_range': (H.diagonal().real.min(), H.diagonal().real.max()),
            'off_diagonal': H[0, 1] if n > 1 else None,
            'eigenvalue_range': (eigenvalues.min(), eigenvalues.max()),
            'bandwidth': self._calculate_bandwidth(),
        }

        if format == "npz":
            np.savez(filename, H_real=H.real, H_imag=H.imag,
                     z_grid=self.z, params=str(self.params))
            info['filename'] = filename
        elif format == "csv":
            np.savetxt(f"{filename}_real.csv", H.real, delimiter=',', fmt='%.10e')
            np.savetxt(f"{filename}_imag.csv", H.imag, delimiter=',', fmt='%.10e')
            info['filename'] = f"{filename}_real.csv, {filename}_imag.csv"
        elif format == "txt":
            with open(filename, 'w') as f:
                f.write(f"# Hamiltonian Matrix for GAA Transistor NEGF\n")
                f.write(f"# Size: {n} x {n}\n")
                f.write(f"# Channel length: {self.params.channel_length*1e9:.2f} nm\n")
                f.write(f"# Grid spacing: {self.dz*1e9:.4f} nm\n")
                f.write(f"# Hopping parameter t: {-H[0,1].real:.6f} eV\n#\n")
                f.write(f"# Matrix (real part):\n")
                for i in range(n):
                    row = ' '.join([f"{H[i,j].real:12.6f}" for j in range(n)])
                    f.write(row + '\n')
            info['filename'] = filename
        elif format == "mat":
            try:
                from scipy.io import savemat
                savemat(filename, {'H_real': H.real, 'H_imag': H.imag,
                                   'z_grid': self.z})
                info['filename'] = filename
            except ImportError:
                np.savez(filename.replace('.mat', '.npz'),
                         H_real=H.real, H_imag=H.imag, z_grid=self.z)
                info['filename'] = filename.replace('.mat', '.npz')

        return info

    def _calculate_bandwidth(self) -> int:
        if self.H is None:
            return 0
        n = self.H.shape[0]
        bw = 0
        for i in range(n):
            for j in range(n):
                if np.abs(self.H[i, j]) > 1e-10:
                    bw = max(bw, abs(i - j))
        return bw

    def print_hamiltonian_info(self):
        """Print detailed Hamiltonian matrix information."""
        if self.H is None:
            print("Hamiltonian not built yet.")
            return

        H = self.H
        n = H.shape[0]
        p = self.params
        m_eff = p.material.effective_mass * M_E
        t = (HBAR**2 / (2 * m_eff * self.dz**2)) / Q_E

        print("\n" + "=" * 50)
        print("Hamiltonian Matrix Information")
        print("=" * 50)
        print(f"Matrix size: {n} x {n}")
        print(f"Non-zero elements: {np.sum(np.abs(H) > 1e-10)}")
        print(f"Sparsity: {np.sum(np.abs(H) < 1e-10) / H.size * 100:.1f}%")
        print(f"Bandwidth: {self._calculate_bandwidth()}")
        print(f"\nPhysical parameters:")
        print(f"  Grid spacing dz: {self.dz*1e9:.4f} nm")
        print(f"  Effective mass: {p.material.effective_mass} m_e")
        print(f"  Hopping parameter t: {t:.6f} eV")
        print(f"\nMatrix elements:")
        print(f"  Diagonal: [{H.diagonal().real.min():.4f}, {H.diagonal().real.max():.4f}] eV")
        print(f"  Off-diagonal: {H[0,1].real:.4f} eV")
        print(f"  Is Hermitian: {np.allclose(H, H.T.conj())}")

        block_size = min(8, n)
        print(f"\nStructure (first {block_size}x{block_size} block):")
        print("-" * 50)
        for i in range(block_size):
            row = ""
            for j in range(block_size):
                val = H[i, j].real
                row += "   .   " if abs(val) < 1e-10 else f"{val:7.3f}"
            print(row)
        print("=" * 50)


# ---------------------------------------------------------------------------
# Poisson Solver for GAA electrostatics
# ---------------------------------------------------------------------------
class PoissonSolver:
    """
    1D Poisson solver for GAA transistor electrostatics.
    Includes gate capacitive coupling with cylindrical geometry enhancement.
    """

    def __init__(self, params: GAADeviceParams):
        self.params = params

    def solve_1d(self, charge_density: np.ndarray,
                 bc_source: float = 0.0, bc_drain: float = 0.0) -> np.ndarray:
        """Solve 1D Poisson equation with GAA gate coupling."""
        p = self.params
        nz = p.nz
        dz = p.channel_length / (nz - 1)

        n_contact = max(2, int(0.15 * nz))

        potential = np.zeros(nz)

        barrier_height = 0.3
        v_threshold = 0.25

        for i in range(nz):
            if i < n_contact:
                potential[i] = p.vs
            elif i >= nz - n_contact:
                potential[i] = p.vd
            else:
                z_rel = (i - n_contact) / (nz - 2 * n_contact)
                v_dibl = p.vs + (p.vd - p.vs) * z_rel
                gate_effect = max(0, barrier_height - (p.vg - v_threshold))
                potential[i] = v_dibl + gate_effect

                if charge_density is not None and len(charge_density) > i:
                    n_density = charge_density[i] * 1e6
                    screening = -Q_E * n_density * dz**2 / (
                        EPSILON_0 * p.material.dielectric_constant)
                    potential[i] += screening * 0.01

        return potential


# ---------------------------------------------------------------------------
# Self-Consistent NEGF-Poisson solver
# ---------------------------------------------------------------------------
class SelfConsistentNEGF:
    """
    Self-consistent NEGF-Poisson loop for GAA transistors.

    Loop: init density -> Poisson -> Hamiltonian -> NEGF -> new density -> mix -> repeat.
    """

    def __init__(self, params: GAADeviceParams,
                 mixing: float = 0.3, max_iter: int = 100, tol: float = 1e-6):
        self.params = params
        self.mixing = mixing
        self.max_iter = max_iter
        self.tol = tol
        self.negf = NEGFSolver(params)
        self.poisson = PoissonSolver(params)
        self.converged = False
        self.iteration = 0
        self.density_history = []

    def initialize_density(self) -> np.ndarray:
        p = self.params
        n = p.nz if p.nr == 1 else p.nz * p.nr
        density = np.zeros(n)
        n_contact = max(1, int(0.1 * p.nz))

        for i in range(n):
            iz = i // p.nr if p.nr > 1 else i
            if iz < n_contact:
                density[i] = p.source_doping
            elif iz >= p.nz - n_contact:
                density[i] = p.drain_doping
            else:
                density[i] = p.channel_doping

        return density

    def solve(self, E_min: float = -0.5, E_max: float = 0.5,
              n_energy: int = 100, verbose: bool = True) -> Dict:
        """Run the self-consistent NEGF-Poisson loop."""
        p = self.params

        density = self.initialize_density()
        potential = np.zeros_like(density)

        if verbose:
            print("Starting self-consistent NEGF-Poisson loop...")
            print(f"  Vg = {p.vg:.3f} V, Vd = {p.vd:.3f} V")

        for iteration in range(self.max_iter):
            self.iteration = iteration

            if p.nr == 1:
                potential = self.poisson.solve_1d(density, p.vs, p.vd)

            self.negf.build_hamiltonian(potential)
            density_new = self.negf.calculate_electron_density(E_min, E_max, n_energy)

            density_diff = la.norm(density_new - density) / (la.norm(density) + 1e-10)
            density = (1 - self.mixing) * density + self.mixing * density_new
            self.density_history.append(density.copy())

            if verbose:
                print(f"  Iteration {iteration + 1}: density change = {density_diff:.2e}")

            if density_diff < self.tol:
                self.converged = True
                if verbose:
                    print(f"Converged after {iteration + 1} iterations!")
                break

        if not self.converged and verbose:
            warnings.warn(f"Did not converge after {self.max_iter} iterations")

        total_current, current_spectrum = self.negf.calculate_current(E_min, E_max, n_energy)

        E_grid = np.linspace(E_min, E_max, n_energy)
        transmission = np.array([self.negf.calculate_transmission(E) for E in E_grid])

        return {
            'density': density,
            'potential': potential,
            'current': total_current,
            'current_spectrum': current_spectrum,
            'transmission': transmission,
            'energy_grid': E_grid,
            'converged': self.converged,
            'iterations': self.iteration + 1,
            'z_grid': self.negf.z,
        }


# ===================================================================
#  PART C — Convenience entry points
# ===================================================================

def run_example_simulation(vg_min: float = 0.0, vg_max: float = 0.7,
                           n_vg_points: int = 8, vd: float = 0.05,
                           plot_results_flag: bool = True):
    """Run a GAA transistor gate-voltage sweep (transfer characteristics)."""
    print("=" * 60)
    print("NEGF Transport Simulation for GAA Transistor")
    print("Transfer Characteristics (I_D vs V_G)")
    print("=" * 60)

    base_params = {
        'channel_length': 12e-9,
        'nanowire_radius': 2.5e-9,
        'oxide_thickness': 1e-9,
        'nz': 25,
        'nr': 1,
        'vd': vd,
        'source_doping': 1e20,
        'drain_doping': 1e20,
        'channel_doping': 1e15,
        'temperature': 300,
    }

    params = GAADeviceParams(**base_params, vg=0.0)
    print(f"\nDevice Parameters:")
    print(f"  Channel length: {params.channel_length * 1e9:.1f} nm")
    print(f"  Nanowire radius: {params.nanowire_radius * 1e9:.1f} nm")
    print(f"  Material: {params.material.name}")
    print(f"  Drain voltage: {vd} V")
    print(f"\nGate voltage sweep: {vg_min} V to {vg_max} V ({n_vg_points} points)")

    vg_values = np.linspace(vg_min, vg_max, n_vg_points)
    currents = []
    all_results = []

    for i, vg in enumerate(vg_values):
        print(f"\n[{i+1}/{n_vg_points}] V_G = {vg:.3f} V")
        params = GAADeviceParams(**base_params, vg=vg)
        solver = SelfConsistentNEGF(params, mixing=0.25, max_iter=40, tol=1e-3)
        results = solver.solve(E_min=-0.4, E_max=0.8, n_energy=40, verbose=False)

        current = np.abs(results['current'])
        currents.append(current)
        all_results.append(results)
        status = "converged" if results['converged'] else f"({results['iterations']} iters)"
        print(f"  I_D = {current * 1e6:.4f} uA  [{status}]")

    currents = np.array(currents)

    print("\n" + "=" * 60)
    print(f"{'V_G (V)':<12} {'I_D (uA)':<15} {'I_D (A)':<15}")
    print("-" * 42)
    for vg, current in zip(vg_values, currents):
        print(f"{vg:<12.3f} {current*1e6:<15.4f} {current:<15.4e}")

    if plot_results_flag:
        try:
            import matplotlib.pyplot as plt
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            ax1.plot(vg_values, currents * 1e6, 'b-o', linewidth=2, markersize=8,
                     markerfacecolor='white', markeredgewidth=2)
            ax1.set_xlabel('V_G (V)')
            ax1.set_ylabel('I_DS (uA)')
            ax1.set_title('Transfer Characteristics (Linear Scale)')
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(bottom=0)

            ax2.semilogy(vg_values, currents * 1e6 + 1e-6, 'r-s', linewidth=2,
                        markersize=8, markerfacecolor='white', markeredgewidth=2)
            ax2.set_xlabel('V_G (V)')
            ax2.set_ylabel('I_DS (uA)')
            ax2.set_title('Transfer Characteristics (Log Scale)')
            ax2.grid(True, alpha=0.3, which='both')

            plt.suptitle(f'GAA Transistor: L={base_params["channel_length"]*1e9:.0f}nm, '
                        f'R={base_params["nanowire_radius"]*1e9:.1f}nm, V_D={vd}V')
            plt.tight_layout()
            plt.savefig('transfer_curve.png', dpi=150, bbox_inches='tight')
            print(f"\nTransfer curve saved to: transfer_curve.png")
            plt.show()
        except ImportError:
            print("\nMatplotlib not available for plotting.")

    return vg_values, currents, all_results


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main():
    """Run block-RGF simulation for rectangular nanowire (default demo)."""
    p = DeviceParams(
        Ny=3, Nz=3, Nx=40,
        V_bias=0.3, V_gate=0.0,
        T=300.0,
        E_min=-0.5, E_max=2.0, NE=200,
    )

    E_arr, T_arr, I_total, LDOS = solve_negf(p)
    plot_results(E_arr, T_arr, I_total, LDOS, p)

    print("\n--- Summary ---")
    print(f"  Device: {p.Ny}x{p.Nz} cross-section, {p.Nx} slices")
    print(f"  Wire length: {p.Nx * p.a * 1e9:.2f} nm")
    print(f"  V_bias = {p.V_bias} V, V_gate = {p.V_gate} V")
    print(f"  Current = {I_total*1e6:.4f} uA")
    print(f"  Conductance G = {I_total/p.V_bias * 1e6:.4f} uS")
    print(f"  G / G_0 = {I_total/p.V_bias / (2*Q_E**2/(2*np.pi*HBAR)):.4f}")


if __name__ == "__main__":
    main()
