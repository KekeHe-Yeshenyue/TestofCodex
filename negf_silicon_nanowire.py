"""
Non-Equilibrium Green's Function (NEGF) Quantum Transport Simulation
for Silicon Nanowire Field-Effect Transistors.

This module implements:
  - Tight-binding Hamiltonian for a silicon nanowire (effective mass approx.)
  - Surface Green's function via iterative Sancho-Rubio method
  - Recursive Green's function (RGF) algorithm for layer-by-layer inversion
  - Transmission coefficient T(E)
  - Current density via Landauer-Büttiker formula
  - Local density of states (LDOS)

References:
  [1] S. Datta, "Quantum Transport: Atom to Transistor", Cambridge (2005).
  [2] M.P. Lopez-Sancho et al., J. Phys. F 15, 851 (1985).
  [3] A. Svizhenko et al., J. Appl. Phys. 91, 2343 (2002).
"""

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Physical constants (SI)
# ---------------------------------------------------------------------------
Q_E = 1.602176634e-19       # elementary charge [C]
HBAR = 1.054571817e-34      # reduced Planck constant [J·s]
M_E = 9.1093837015e-31      # free electron mass [kg]
KB = 1.380649e-23            # Boltzmann constant [J/K]


# ---------------------------------------------------------------------------
# Device parameters
# ---------------------------------------------------------------------------
@dataclass
class DeviceParams:
    """Parameters describing the silicon nanowire device."""
    # Nanowire cross-section grid
    Ny: int = 3              # grid points across y (width)
    Nz: int = 3              # grid points across z (height)
    Nx: int = 60             # number of layers along transport (x)

    # Lattice spacing
    a: float = 0.543e-9 / 2  # lattice constant / 2  ≈ 0.27 nm

    # Effective masses for silicon (in units of m_e)
    mx: float = 0.19         # longitudinal effective mass (transport dir.)
    my: float = 0.19         # transverse
    mz: float = 0.19         # transverse

    # Temperature
    T: float = 300.0         # Kelvin

    # Bias
    V_bias: float = 0.3      # source-drain bias [V]

    # Gate potential profile (simple linear ramp for demo)
    V_gate: float = 0.0      # gate voltage [V]

    # Broadening for Green's function
    eta: float = 1e-6        # small imaginary part [eV]

    # Energy grid
    E_min: float = -0.5      # [eV]
    E_max: float = 1.5       # [eV]
    NE: int = 300            # number of energy points

    @property
    def N_orb(self) -> int:
        """Number of orbitals per cross-sectional slice."""
        return self.Ny * self.Nz


# ---------------------------------------------------------------------------
# Hamiltonian construction
# ---------------------------------------------------------------------------
def build_slice_hamiltonian(p: DeviceParams) -> np.ndarray:
    """
    Build the on-site Hamiltonian H_slice for one cross-sectional slice
    of the nanowire using an effective-mass tight-binding model.

    The 2D grid (Ny × Nz) is mapped to a 1D index:  idx = iy * Nz + iz.

    On-site energy:  e0 = t_y + t_z  (from kinetic energy discretisation)
        with  t_y = hbar^2 / (2 m_y a^2),  t_z = hbar^2 / (2 m_z a^2)
    Off-diagonal hopping within slice: -t_y (y-neighbours), -t_z (z-neighbours)
    """
    N = p.N_orb
    # Hopping energies [J] -> [eV]
    t_y = (HBAR ** 2 / (2 * p.my * M_E * p.a ** 2)) / Q_E
    t_z = (HBAR ** 2 / (2 * p.mz * M_E * p.a ** 2)) / Q_E

    H = np.zeros((N, N), dtype=complex)

    for iy in range(p.Ny):
        for iz in range(p.Nz):
            idx = iy * p.Nz + iz
            # On-site: sum of 2D kinetic energy contributions
            n_y_neighbors = (1 if iy > 0 else 0) + (1 if iy < p.Ny - 1 else 0)
            n_z_neighbors = (1 if iz > 0 else 0) + (1 if iz < p.Nz - 1 else 0)
            H[idx, idx] = n_y_neighbors * t_y + n_z_neighbors * t_z

            # y-direction hopping
            if iy < p.Ny - 1:
                jdx = (iy + 1) * p.Nz + iz
                H[idx, jdx] = -t_y
                H[jdx, idx] = -t_y

            # z-direction hopping
            if iz < p.Nz - 1:
                jdx = iy * p.Nz + (iz + 1)
                H[idx, jdx] = -t_z
                H[jdx, idx] = -t_z

    return H


def build_coupling_matrix(p: DeviceParams) -> np.ndarray:
    """
    Build the inter-slice coupling matrix V (hopping between adjacent
    slices along the transport direction x).

    V_{ij} = -t_x  delta_{ij}   (nearest-neighbour, same orbital index)
    """
    t_x = (HBAR ** 2 / (2 * p.mx * M_E * p.a ** 2)) / Q_E
    N = p.N_orb
    V = -t_x * np.eye(N, dtype=complex)
    return V


def potential_profile(p: DeviceParams) -> np.ndarray:
    """
    Electrostatic potential along the channel [eV] for each slice.

    Returns array of shape (Nx,).  A simple model:
      - Linear source-drain drop
      - Gate modulation (uniform shift for simplicity)
    """
    x = np.linspace(0, 1, p.Nx)
    # Linear drop from 0 to -qV_bias across the channel
    V_sd = -p.V_bias * x
    # Gate shifts the band edge
    V_g = -p.V_gate * np.ones(p.Nx)
    return V_sd + V_g


# ---------------------------------------------------------------------------
# Surface Green's function  (Sancho-Rubio iterative method)
# ---------------------------------------------------------------------------
def surface_green_function(E: float, H_slice: np.ndarray,
                           V: np.ndarray, p: DeviceParams,
                           max_iter: int = 100, tol: float = 1e-8
                           ) -> np.ndarray:
    """
    Compute the surface Green's function g_s of a semi-infinite lead
    using the Sancho-Rubio iterative scheme [2].

    The lead is modelled as an infinite repetition of identical slices
    with on-site block H_slice and inter-slice coupling V.

    Algorithm:
        Initialise:
            epsilon_s = H_slice
            epsilon   = H_slice
            alpha     = V
            beta      = V†
        Iterate:
            g_epsilon = (E·I - epsilon)^{-1}
            epsilon_s <- epsilon_s + alpha · g_epsilon · beta
            epsilon   <- epsilon   + alpha · g_epsilon · beta
                                   + beta  · g_epsilon · alpha
            alpha     <- alpha · g_epsilon · alpha
            beta      <- beta  · g_epsilon · beta
        Converge when ||alpha|| < tol.
        Surface GF:  g_s = (E·I - epsilon_s)^{-1}
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

    g_s = linalg.inv(EI - eps_s)
    return g_s


# ---------------------------------------------------------------------------
# Self-energies from contacts
# ---------------------------------------------------------------------------
def contact_self_energy(E: float, H_slice: np.ndarray,
                        V: np.ndarray, p: DeviceParams) -> np.ndarray:
    """
    Self-energy of a semi-infinite contact:
        Σ = V† · g_s · V

    where g_s is the surface Green's function of the lead.
    """
    g_s = surface_green_function(E, H_slice, V, p)
    Sigma = V.conj().T @ g_s @ V
    return Sigma


def broadening(Sigma: np.ndarray) -> np.ndarray:
    """Broadening (coupling) matrix:  Γ = i (Σ - Σ†)"""
    return 1j * (Sigma - Sigma.conj().T)


# ---------------------------------------------------------------------------
# Recursive Green's Function (RGF) algorithm
# ---------------------------------------------------------------------------
def recursive_green_function(E: float, p: DeviceParams,
                             H_slices: list[np.ndarray],
                             V: np.ndarray,
                             Sigma_L: np.ndarray,
                             Sigma_R: np.ndarray):
    """
    Recursive Green's function algorithm to obtain:
      - G^R (retarded GF) diagonal blocks
      - G^n (electron correlation function) diagonal blocks

    for the device Hamiltonian consisting of Nx slices.

    Parameters
    ----------
    E       : energy [eV]
    p       : DeviceParams
    H_slices: list of Nx on-site Hamiltonian blocks (N_orb × N_orb)
    V       : inter-slice coupling (same for all slices here)
    Sigma_L : left contact self-energy  (added to slice 0)
    Sigma_R : right contact self-energy (added to slice Nx-1)

    Returns
    -------
    G_diag  : list of diagonal blocks G^R_{ii}, i = 0..Nx-1
    Gn_diag : list of diagonal blocks G^n_{ii}
    T_E     : transmission at energy E
    """
    Nx = p.Nx
    N = p.N_orb
    EI = (E + 1j * p.eta) * np.eye(N, dtype=complex)
    Vdag = V.conj().T

    # Fermi functions for source and drain
    mu_S = 0.0
    mu_D = -p.V_bias
    f_S = fermi(E, mu_S, p.T)
    f_D = fermi(E, mu_D, p.T)

    Gamma_L = broadening(Sigma_L)
    Gamma_R = broadening(Sigma_R)

    # In-scattering from contacts (coherent transport, no scattering inside)
    Sigma_in_L = f_S * Gamma_L
    Sigma_in_R = f_D * Gamma_R

    # ------------------------------------------------------------------
    # Forward sweep: build left-connected Green's functions g^R_L[i]
    # ------------------------------------------------------------------
    gL = [None] * Nx  # left-connected GF for each slice

    # First slice includes left self-energy
    gL[0] = linalg.inv(EI - H_slices[0] - Sigma_L)

    for i in range(1, Nx):
        Sigma_i = Vdag @ gL[i - 1] @ V
        if i == Nx - 1:
            gL[i] = linalg.inv(EI - H_slices[i] - Sigma_i - Sigma_R)
        else:
            gL[i] = linalg.inv(EI - H_slices[i] - Sigma_i)

    # ------------------------------------------------------------------
    # The full retarded GF of the last slice = gL[Nx-1]
    # ------------------------------------------------------------------
    G_diag = [None] * Nx
    Gn_diag = [None] * Nx

    G_diag[Nx - 1] = gL[Nx - 1]

    # ------------------------------------------------------------------
    # Backward sweep: obtain all diagonal blocks of G^R and G^n
    # ------------------------------------------------------------------
    # G^n for the last slice
    # Sigma_in at last slice = Sigma_in_R + contribution from left
    # For a full RGF we propagate Sigma_in through the device.
    # Simplified: we compute G^n from the full GF diagonal blocks.
    # ------------------------------------------------------------------

    # Backward sweep for G^R diagonal blocks
    for i in range(Nx - 2, -1, -1):
        G_diag[i] = gL[i] + gL[i] @ V @ G_diag[i + 1] @ Vdag @ gL[i]

    # ------------------------------------------------------------------
    # Compute G^n diagonal blocks (electron correlation)
    # Forward sweep to accumulate in-scattering
    # ------------------------------------------------------------------
    # G^n = G^R · Σ^in · G^A  where Σ^in = Σ^in_L + Σ^in_R (at boundaries)
    # Using RGF for G^n:

    # Left-connected Σ^in propagation
    Sigma_in_left = [np.zeros((N, N), dtype=complex)] * Nx
    Sigma_in_left[0] = Sigma_in_L

    gL_Sigma_in = [None] * Nx
    gL_Sigma_in[0] = gL[0] @ Sigma_in_left[0] @ gL[0].conj().T

    for i in range(1, Nx):
        prop = Vdag @ gL_Sigma_in[i - 1] @ V
        if i == Nx - 1:
            Sigma_in_i = prop + Sigma_in_R
        else:
            Sigma_in_i = prop
        gL_Sigma_in[i] = gL[i] @ Sigma_in_i @ gL[i].conj().T

    # The last slice G^n
    Gn_diag[Nx - 1] = gL_Sigma_in[Nx - 1]

    # Backward sweep for G^n (using connection formula)
    for i in range(Nx - 2, -1, -1):
        A_i = gL[i] @ V @ G_diag[i + 1]
        Gn_diag[i] = (gL_Sigma_in[i]
                       + A_i @ Vdag @ gL_Sigma_in[i]
                       + gL_Sigma_in[i] @ V @ G_diag[i + 1].conj().T @ Vdag @ gL[i].conj().T
                       + A_i @ Gn_diag[i + 1] @ A_i.conj().T)

    # ------------------------------------------------------------------
    # Transmission:  T(E) = Tr[ Γ_L · G^R · Γ_R · G^A ]
    # Use Fisher-Lee relation via the full device GF at boundaries.
    # We need G^R_{0, Nx-1}. Build it from the RGF:
    # ------------------------------------------------------------------
    # G_{0,N-1} = gL[0] · V · gL[1] · V · ... · gL[N-2] · V · G_{N-1,N-1}
    # But more efficiently, just use the recursive relation:
    G_0N = gL[0].copy()
    for i in range(1, Nx):
        if i < Nx - 1:
            G_0N = G_0N @ V @ gL[i]
        else:
            G_0N = G_0N @ V @ G_diag[Nx - 1]

    T_E = np.real(np.trace(Gamma_L @ G_0N @ Gamma_R @ G_0N.conj().T))

    return G_diag, Gn_diag, T_E


# ---------------------------------------------------------------------------
# Fermi-Dirac distribution
# ---------------------------------------------------------------------------
def fermi(E: float, mu: float, T: float) -> float:
    """Fermi-Dirac distribution f(E, mu, T)."""
    kT = KB * T / Q_E  # in eV
    x = (E - mu) / kT
    # Guard against overflow
    if x > 500:
        return 0.0
    elif x < -500:
        return 1.0
    return 1.0 / (1.0 + np.exp(x))


# ---------------------------------------------------------------------------
# Full NEGF solver
# ---------------------------------------------------------------------------
def solve_negf(p: DeviceParams):
    """
    Main NEGF solver.  Sweeps over energy to compute:
      - Transmission T(E)
      - Current I  (Landauer formula)
      - Local density of states LDOS(x, E)

    Returns
    -------
    E_arr       : energy array [eV]
    T_arr       : transmission array
    I_total     : total current [A]
    LDOS        : array (NE, Nx) — LDOS summed over cross-section orbitals
    """
    # Build Hamiltonian components
    H0 = build_slice_hamiltonian(p)
    V = build_coupling_matrix(p)
    V_pot = potential_profile(p)

    # On-site Hamiltonian for each slice (add electrostatic potential)
    t_x = (HBAR ** 2 / (2 * p.mx * M_E * p.a ** 2)) / Q_E
    H_slices = []
    for i in range(p.Nx):
        Hi = H0.copy()
        # Add x-direction kinetic energy contribution (2 t_x on diagonal)
        Hi += 2 * t_x * np.eye(p.N_orb, dtype=complex)
        # Add electrostatic potential
        Hi += V_pot[i] * np.eye(p.N_orb, dtype=complex)
        H_slices.append(Hi)

    # Lead Hamiltonians (same as first/last slice without potential drop)
    H_lead_L = H0.copy() + 2 * t_x * np.eye(p.N_orb, dtype=complex)
    H_lead_R = H0.copy() + 2 * t_x * np.eye(p.N_orb, dtype=complex)
    # The drain lead is shifted by -V_bias
    H_lead_R += (-p.V_bias) * np.eye(p.N_orb, dtype=complex)

    E_arr = np.linspace(p.E_min, p.E_max, p.NE)
    T_arr = np.zeros(p.NE)
    LDOS = np.zeros((p.NE, p.Nx))

    print(f"NEGF solver: {p.Nx} slices, {p.N_orb} orbitals/slice, "
          f"{p.NE} energy points")
    print(f"  Cross-section: {p.Ny}×{p.Nz}, lattice constant a={p.a*1e9:.3f} nm")
    print(f"  V_bias={p.V_bias} V, V_gate={p.V_gate} V, T={p.T} K")
    print("  Running energy sweep...")

    for ie, E in enumerate(E_arr):
        if ie % 50 == 0:
            print(f"    E = {E:.3f} eV  ({ie}/{p.NE})")

        # Contact self-energies
        Sigma_L = contact_self_energy(E, H_lead_L, V, p)
        Sigma_R = contact_self_energy(E, H_lead_R, V, p)

        # RGF
        G_diag, Gn_diag, T_E = recursive_green_function(
            E, p, H_slices, V, Sigma_L, Sigma_R
        )

        T_arr[ie] = T_E

        # LDOS at each slice:  A(E) = i(G^R - G^A) = -2 Im(G^R)
        for ix in range(p.Nx):
            LDOS[ie, ix] = -np.imag(np.trace(G_diag[ix])) / np.pi

    # ------------------------------------------------------------------
    # Current via Landauer formula:
    #   I = (q/h) ∫ T(E) [f_S(E) - f_D(E)] dE
    # ------------------------------------------------------------------
    mu_S = 0.0
    mu_D = -p.V_bias
    dE = E_arr[1] - E_arr[0]

    f_S = np.array([fermi(E, mu_S, p.T) for E in E_arr])
    f_D = np.array([fermi(E, mu_D, p.T) for E in E_arr])

    # Spin degeneracy factor = 2
    I_total = 2 * (Q_E / (2 * np.pi * HBAR / Q_E)) * np.trapz(
        T_arr * (f_S - f_D), E_arr
    )
    # Simpler: I = (2q/h) ∫ T(E)(f_S - f_D) dE
    # h in eV·s:  h = 2π ħ,  ħ = 6.582e-16 eV·s  => h = 4.136e-15 eV·s
    h_eV = 4.135667696e-15  # [eV·s]
    I_total = 2 * Q_E / h_eV * np.trapz(T_arr * (f_S - f_D), E_arr)

    print(f"\n  Total current I = {I_total*1e6:.4f} μA")
    print(f"  Peak transmission = {np.max(T_arr):.4f}")

    return E_arr, T_arr, I_total, LDOS


# ---------------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------------
def plot_results(E_arr, T_arr, I_total, LDOS, p: DeviceParams):
    """Generate publication-quality plots of NEGF results."""

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # (a) Transmission vs Energy
    ax = axes[0, 0]
    ax.plot(E_arr, T_arr, 'b-', linewidth=1.5)
    ax.set_xlabel('Energy [eV]')
    ax.set_ylabel('Transmission T(E)')
    ax.set_title('(a) Transmission Spectrum')
    ax.set_xlim(p.E_min, p.E_max)
    ax.grid(True, alpha=0.3)

    # Shade the bias window
    mu_S, mu_D = 0.0, -p.V_bias
    ax.axvspan(min(mu_S, mu_D), max(mu_S, mu_D),
               alpha=0.15, color='orange', label='Bias window')
    ax.legend()

    # (b) Transmission × (f_S - f_D)  — the current integrand
    ax = axes[0, 1]
    f_S = np.array([fermi(E, mu_S, p.T) for E in E_arr])
    f_D = np.array([fermi(E, mu_D, p.T) for E in E_arr])
    integrand = T_arr * (f_S - f_D)
    ax.fill_between(E_arr, integrand, alpha=0.4, color='green')
    ax.plot(E_arr, integrand, 'g-', linewidth=1.2)
    ax.set_xlabel('Energy [eV]')
    ax.set_ylabel('T(E) [f_S - f_D]')
    ax.set_title(f'(b) Current Integrand  (I = {I_total*1e6:.3f} μA)')
    ax.set_xlim(p.E_min, p.E_max)
    ax.grid(True, alpha=0.3)

    # (c) LDOS map
    ax = axes[1, 0]
    x_nm = np.arange(p.Nx) * p.a * 1e9
    extent = [x_nm[0], x_nm[-1], E_arr[0], E_arr[-1]]
    im = ax.imshow(LDOS, aspect='auto', origin='lower', extent=extent,
                   cmap='hot', interpolation='bilinear')
    ax.set_xlabel('Position along wire [nm]')
    ax.set_ylabel('Energy [eV]')
    ax.set_title('(c) Local Density of States')
    fig.colorbar(im, ax=ax, label='LDOS [1/eV]')

    # (d) Potential profile + band diagram
    ax = axes[1, 1]
    V_pot = potential_profile(p)
    t_x = (HBAR ** 2 / (2 * p.mx * M_E * p.a ** 2)) / Q_E
    # Conduction band edge approximation (bottom of 2D subband)
    E_sub = build_slice_hamiltonian(p)
    evals = np.sort(np.real(linalg.eigvalsh(E_sub)))
    E_cb = evals[0] + 2 * t_x  # lowest subband + x kinetic offset

    ax.plot(x_nm, V_pot + E_cb, 'r-', linewidth=2, label='CB edge (lowest subband)')
    ax.axhline(y=0.0, color='blue', linestyle='--', label=f'μ_S = 0 eV')
    ax.axhline(y=-p.V_bias, color='cyan', linestyle='--',
               label=f'μ_D = {-p.V_bias} eV')
    ax.set_xlabel('Position along wire [nm]')
    ax.set_ylabel('Energy [eV]')
    ax.set_title('(d) Band Diagram')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle('NEGF Quantum Transport — Silicon Nanowire', fontsize=14,
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig('negf_results.png', dpi=150, bbox_inches='tight')
    print("  Plot saved to negf_results.png")
    plt.show()


# ---------------------------------------------------------------------------
# I-V characteristic sweep
# ---------------------------------------------------------------------------
def compute_iv_curve(p: DeviceParams, V_values: np.ndarray):
    """Sweep bias voltage and compute I-V curve."""
    currents = []
    for V in V_values:
        p_v = DeviceParams(
            Ny=p.Ny, Nz=p.Nz, Nx=p.Nx, a=p.a,
            mx=p.mx, my=p.my, mz=p.mz,
            T=p.T, V_bias=V, V_gate=p.V_gate,
            eta=p.eta, E_min=p.E_min, E_max=p.E_max, NE=p.NE
        )
        _, _, I, _ = solve_negf(p_v)
        currents.append(I)
        print(f"  V_bias = {V:.3f} V  =>  I = {I*1e6:.4f} μA")
    return np.array(currents)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    """Run the NEGF simulation with default parameters."""
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
    print(f"  Current = {I_total*1e6:.4f} μA")
    print(f"  Conductance G = {I_total/p.V_bias * 1e6:.4f} μS")
    print(f"  G / G_0 = {I_total/p.V_bias / (2*Q_E**2/(2*np.pi*HBAR)):.4f}")


if __name__ == "__main__":
    main()
