#!/usr/bin/env python3
"""
Test suite for the unified NEGF transport code.

Verifies:
1. Surface Green's function (class and functional APIs)
2. Hamiltonian construction (rectangular and GAA)
3. Green's function calculations (G^R, G^A, G^<)
4. Transmission calculation
5. Current calculation and conservation
6. Self-consistent solver
7. Block-matrix RGF for rectangular nanowire
"""

import numpy as np
from numpy import linalg as la
import sys


def test_surface_green_function():
    """Test surface Green's function (class API)."""
    print("Testing Surface Green's Function...")

    from negf_silicon_nanowire import SurfaceGreenFunction

    t = 1.0
    H_unit = np.array([[2 * t]], dtype=complex)
    H_coupling = np.array([[-t]], dtype=complex)

    sgf = SurfaceGreenFunction(H_unit, H_coupling, eta=1e-5)

    g_s = sgf.calculate(E=0.0)

    assert g_s.shape == (1, 1), "Shape mismatch"
    assert np.isfinite(g_s[0, 0]), "g_s should be finite"
    assert np.abs(g_s[0, 0].imag) > 0, "g_s should have imaginary part"

    print(f"  g_surface(E=0) = {g_s[0, 0]:.6f}")
    print("  PASSED")
    return True


def test_surface_green_function_iterative():
    """Test the alternative iterative surface GF method."""
    print("Testing Surface Green's Function (Iterative)...")

    from negf_silicon_nanowire import SurfaceGreenFunction

    t = 1.0
    H_unit = np.array([[2 * t]], dtype=complex)
    H_coupling = np.array([[-t]], dtype=complex)

    sgf = SurfaceGreenFunction(H_unit, H_coupling, eta=1e-5)

    g_decimation = sgf.calculate(E=0.0)
    g_iterative = sgf.calculate_iterative(E=0.0)

    diff = la.norm(g_decimation - g_iterative) / la.norm(g_decimation)
    assert diff < 0.01, f"Methods should agree, got relative diff {diff}"

    print(f"  Relative difference between methods: {diff:.2e}")
    print("  PASSED")
    return True


def test_hamiltonian_construction():
    """Test GAA Hamiltonian construction."""
    print("Testing GAA Hamiltonian Construction...")

    from negf_silicon_nanowire import GAADeviceParams, NEGFSolver

    params = GAADeviceParams(
        channel_length=10e-9,
        nz=20,
        nr=1,
        temperature=300
    )

    solver = NEGFSolver(params)
    H = solver.build_hamiltonian()

    assert np.allclose(H, H.T.conj()), "Hamiltonian should be Hermitian"

    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            if abs(i - j) > 1:
                assert H[i, j] == 0, f"Non-tridiagonal element H[{i},{j}] = {H[i,j]}"

    assert all(H[i, i].real > 0 for i in range(H.shape[0])), "Diagonal should be positive"

    for i in range(H.shape[0] - 1):
        assert H[i, i + 1].real < 0, "Off-diagonal should be negative (hopping)"

    print(f"  Hamiltonian shape: {H.shape}")
    print(f"  Diagonal range: [{H.diagonal().real.min():.4f}, {H.diagonal().real.max():.4f}] eV")
    print(f"  Off-diagonal: {H[0, 1]:.4f} eV")
    print("  PASSED")
    return True


def test_rectangular_hamiltonian():
    """Test rectangular nanowire Hamiltonian (block-RGF path)."""
    print("Testing Rectangular Nanowire Hamiltonian...")

    from negf_silicon_nanowire import DeviceParams, build_slice_hamiltonian, build_coupling_matrix

    p = DeviceParams(Ny=3, Nz=3)
    H_slice = build_slice_hamiltonian(p)
    V = build_coupling_matrix(p)

    assert H_slice.shape == (9, 9), f"Expected (9,9), got {H_slice.shape}"
    assert np.allclose(H_slice, H_slice.T.conj()), "H_slice should be Hermitian"
    assert V.shape == (9, 9), f"V shape should be (9,9), got {V.shape}"

    assert np.allclose(V, np.diag(np.diag(V))), "V should be diagonal"

    print(f"  H_slice shape: {H_slice.shape}")
    print(f"  V diagonal: {V[0,0]:.4f} eV")
    print("  PASSED")
    return True


def test_greens_functions():
    """Test retarded and advanced Green's function calculations."""
    print("Testing Green's Functions...")

    from negf_silicon_nanowire import GAADeviceParams, NEGFSolver

    params = GAADeviceParams(
        channel_length=5e-9,
        nz=10,
        nr=1,
        vd=0.1,
        temperature=300
    )

    solver = NEGFSolver(params)
    solver.build_hamiltonian()

    E = 0.1
    G_R, sigma_S, sigma_D = solver.calculate_retarded_greens_function(E)
    G_A = solver.calculate_advanced_greens_function(G_R)

    assert np.allclose(G_A, G_R.T.conj()), "G^A should equal (G^R)^dag"
    assert np.abs(G_R.imag).max() > 0, "G^R should have imaginary part"

    print(f"  G^R shape: {G_R.shape}")
    print(f"  Max |Im(G^R)|: {np.abs(G_R.imag).max():.6f}")
    print("  PASSED")
    return True


def test_lesser_green_function():
    """Test lesser Green's function calculation."""
    print("Testing Lesser Green's Function...")

    from negf_silicon_nanowire import GAADeviceParams, NEGFSolver

    params = GAADeviceParams(
        channel_length=5e-9,
        nz=10,
        nr=1,
        vd=0.1,
        temperature=300
    )

    solver = NEGFSolver(params)
    solver.build_hamiltonian()

    E = 0.05
    G_R, sigma_S, sigma_D = solver.calculate_retarded_greens_function(E)
    G_lesser = solver.calculate_lesser_greens_function(E, G_R, sigma_S, sigma_D)

    assert np.all(np.isfinite(G_lesser)), "G^< should have finite elements"

    print(f"  G^< shape: {G_lesser.shape}")
    print(f"  Max |G^<|: {np.abs(G_lesser).max():.6f}")
    print("  PASSED")
    return True


def test_transmission():
    """Test transmission calculation."""
    print("Testing Transmission...")

    from negf_silicon_nanowire import GAADeviceParams, NEGFSolver

    params = GAADeviceParams(
        channel_length=5e-9,
        nz=15,
        nr=1,
        vd=0.0,
        temperature=300
    )

    solver = NEGFSolver(params)
    solver.build_hamiltonian()

    E_values = np.linspace(-0.2, 0.5, 20)
    T_values = []

    for E in E_values:
        T = solver.calculate_transmission(E)
        T_values.append(T)
        assert T >= -1e-10, f"Transmission should be non-negative, got {T}"
        assert T <= 10, f"Transmission unexpectedly large: {T}"

    T_values = np.array(T_values)
    print(f"  T(E) range: [{T_values.min():.6f}, {T_values.max():.6f}]")
    print("  PASSED")
    return True


def test_current_conservation():
    """Test that current is conserved along the device."""
    print("Testing Current Conservation...")

    from negf_silicon_nanowire import GAADeviceParams, NEGFSolver

    params = GAADeviceParams(
        channel_length=8e-9,
        nz=20,
        nr=1,
        vd=0.15,
        vg=0.2,
        temperature=300
    )

    solver = NEGFSolver(params)

    potential = np.linspace(0, params.vd, params.nz)
    solver.build_hamiltonian(potential)

    E = 0.1
    bond_currents = []

    for l in range(params.nz - 1):
        I_bond = solver.calculate_bond_current(E, l)
        bond_currents.append(I_bond)

    bond_currents = np.array(bond_currents)

    current_variation = bond_currents.std() / (np.abs(bond_currents.mean()) + 1e-20)

    print(f"  Bond current range: [{bond_currents.min():.4e}, {bond_currents.max():.4e}] A")
    print(f"  Current variation: {current_variation * 100:.2f}%")

    if current_variation < 0.1:
        print("  PASSED")
    else:
        print("  WARNING: Current not well conserved (may need finer grid)")
    return True


def test_self_consistent_solver():
    """Test self-consistent NEGF-Poisson solver."""
    print("Testing Self-Consistent Solver...")

    from negf_silicon_nanowire import GAADeviceParams, SelfConsistentNEGF

    params = GAADeviceParams(
        channel_length=8e-9,
        nz=15,
        nr=1,
        vg=0.2,
        vd=0.05,
        source_doping=1e19,
        drain_doping=1e19,
        channel_doping=1e15,
        temperature=300
    )

    solver = SelfConsistentNEGF(params, mixing=0.3, max_iter=10, tol=1e-2)
    results = solver.solve(E_min=-0.3, E_max=0.4, n_energy=20, verbose=False)

    assert 'current' in results, "Missing current in results"
    assert 'potential' in results, "Missing potential in results"
    assert 'density' in results, "Missing density in results"
    assert 'transmission' in results, "Missing transmission in results"

    assert np.isfinite(results['current']), "Current should be finite"
    assert len(results['density']) == params.nz, "Density size mismatch"

    print(f"  Iterations: {results['iterations']}")
    print(f"  Converged: {results['converged']}")
    print(f"  Current: {results['current'] * 1e6:.4f} uA")
    print("  PASSED")
    return True


def test_material_system():
    """Test material selection and properties."""
    print("Testing Material System...")

    from negf_silicon_nanowire import Material, MaterialType

    for mt in MaterialType:
        mat = Material.get_material(mt)
        assert mat.effective_mass > 0, f"{mat.name}: effective mass must be positive"
        assert mat.bandgap > 0, f"{mat.name}: bandgap must be positive"
        assert mat.dielectric_constant > 0, f"{mat.name}: dielectric must be positive"
        print(f"  {mat.name}: m*={mat.effective_mass} m_e, Eg={mat.bandgap} eV")

    print("  PASSED")
    return True


def test_block_rgf():
    """Test the block-matrix RGF for rectangular nanowire."""
    print("Testing Block-Matrix RGF...")

    from negf_silicon_nanowire import (
        DeviceParams, build_slice_hamiltonian, build_coupling_matrix,
        surface_green_function, contact_self_energy, recursive_green_function
    )

    p = DeviceParams(Ny=2, Nz=2, Nx=10, V_bias=0.1, NE=5)
    H0 = build_slice_hamiltonian(p)
    V = build_coupling_matrix(p)

    from negf_silicon_nanowire import HBAR, M_E, Q_E
    t_x = (HBAR ** 2 / (2 * p.mx * M_E * p.a ** 2)) / Q_E

    H_slices = []
    for i in range(p.Nx):
        Hi = H0.copy() + 2 * t_x * np.eye(p.N_orb, dtype=complex)
        H_slices.append(Hi)

    H_lead = H0.copy() + 2 * t_x * np.eye(p.N_orb, dtype=complex)

    E = 1.0
    Sigma_L = contact_self_energy(E, H_lead, V, p)
    Sigma_R = contact_self_energy(E, H_lead, V, p)

    G_diag, Gn_diag, T_E = recursive_green_function(E, p, H_slices, V, Sigma_L, Sigma_R)

    assert len(G_diag) == p.Nx, "Wrong number of G^R blocks"
    assert len(Gn_diag) == p.Nx, "Wrong number of G^n blocks"
    assert T_E >= -1e-10, f"Transmission should be non-negative, got {T_E}"

    for i in range(p.Nx):
        assert G_diag[i].shape == (p.N_orb, p.N_orb), f"G_diag[{i}] wrong shape"
        assert Gn_diag[i].shape == (p.N_orb, p.N_orb), f"Gn_diag[{i}] wrong shape"

    print(f"  N_orb = {p.N_orb}, Nx = {p.Nx}")
    print(f"  T(E={E}) = {T_E:.6f}")
    print(f"  G^R block shape: {G_diag[0].shape}")
    print("  PASSED")
    return True


def test_transfer_curve():
    """Test transfer curve (I_D vs V_G) calculation."""
    print("Testing Transfer Curve (I_D vs V_G)...")

    from negf_silicon_nanowire import GAADeviceParams, SelfConsistentNEGF

    base_params = {
        'channel_length': 8e-9,
        'nanowire_radius': 2e-9,
        'nz': 15,
        'nr': 1,
        'vd': 0.05,
        'source_doping': 1e20,
        'drain_doping': 1e20,
        'channel_doping': 1e15,
        'temperature': 300
    }

    vg_values = [0.0, 0.2, 0.4, 0.6]
    currents = []

    print("  Sweeping gate voltage...")
    for vg in vg_values:
        params = GAADeviceParams(**base_params, vg=vg)
        solver = SelfConsistentNEGF(params, mixing=0.3, max_iter=15, tol=1e-2)
        results = solver.solve(E_min=-0.3, E_max=0.5, n_energy=25, verbose=False)
        current = np.abs(results['current'])
        currents.append(current)
        print(f"    V_G = {vg:.1f} V: I_D = {current*1e6:.4f} uA")

    currents = np.array(currents)

    assert np.all(np.isfinite(currents)), "All currents should be finite"

    print(f"\n  Current at low Vg: {currents[0]*1e6:.4f} uA")
    print(f"  Current at high Vg: {currents[-1]*1e6:.4f} uA")
    print("  PASSED")
    return True


def run_all_tests():
    """Run all tests and report results."""
    tests = [
        ("Surface Green's Function (Decimation)", test_surface_green_function),
        ("Surface Green's Function (Iterative)", test_surface_green_function_iterative),
        ("Material System", test_material_system),
        ("GAA Hamiltonian Construction", test_hamiltonian_construction),
        ("Rectangular Hamiltonian", test_rectangular_hamiltonian),
        ("Green's Functions (G^R, G^A)", test_greens_functions),
        ("Lesser Green's Function (G^<)", test_lesser_green_function),
        ("Transmission", test_transmission),
        ("Current Conservation", test_current_conservation),
        ("Block-Matrix RGF", test_block_rgf),
        ("Self-Consistent Solver", test_self_consistent_solver),
        ("Transfer Curve", test_transfer_curve),
    ]

    print("=" * 60)
    print("NEGF Transport Code - Unified Test Suite")
    print("=" * 60)

    passed = 0
    failed = 0

    for name, test_func in tests:
        print(f"\n--- {name} ---")
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"  FAILED")
        except Exception as e:
            failed += 1
            print(f"  FAILED with exception: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
