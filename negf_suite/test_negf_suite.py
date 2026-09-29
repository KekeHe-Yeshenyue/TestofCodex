#!/usr/bin/env python3
"""
Tests for the three-level NEGF suite.   Run:  python test_negf_suite.py   (or pytest)
Each test is a physics identity, not a regression number:
  * analytic 1-D self-energy == Lopez-Sancho decimation
  * RGF diagonal/last-column blocks == dense inverse
  * TranSIESTA contour and Ozaki pole sum == brute-force real-axis integral
  * uniform wire transmission == number of open subbands (integer)
  * LK k.p bulk masses == analytic Luttinger expressions; discretised H Hermitian
  * k.p mode space converges to real space
  * Fermi function: real/complex, scalar/array
"""
import numpy as np, sys, time
from numpy.linalg import inv

import negf_level1_datta_basics as L1
import negf_level2_gaa_transiesta_style as L2
import negf_level3_kp_nanowire as L3


def test_level1_self_energy_matches_sancho_rubio():
    for E in (0.3, 1.0, 5.0, -0.2, 7.5):          # inside band, below band, above band
        sig_a = L1.lead_self_energy(E, 0.0)
        g = L1.numeric_sancho_rubio(E, np.array([[2 * L1.t0]]), np.array([[-L1.t0]]), eta=1e-9)
        sig_n = L1.t0**2 * g[0, 0]
        assert abs(sig_a - sig_n) < 1e-6, (E, sig_a, sig_n)
        assert sig_a.imag <= 1e-12                # retarded
    print("  level1 analytic Sigma == Lopez-Sancho          OK")


def test_level1_uniform_wire_T_is_one():
    Np = 30; U = np.zeros(Np)
    E = np.linspace(0.2, 4 * L1.t0 - 0.2, 50)
    T, _, _, _ = L1.sweep(U, 0.1, 0.1, E)
    assert np.allclose(T, 1.0, atol=1e-6)
    print("  level1 uniform wire T(E)=1                     OK")


def test_level1_density_counts_states():
    # a uniform wire fully below the Fermi level (mu far above the band) holds exactly 2 electrons/site
    Np = 20; U = np.zeros(Np)
    E = np.linspace(-0.2, 4 * L1.t0 + 0.2, 6000)
    _, _, n, _ = L1.sweep(U, 50.0, 50.0, E)
    assert np.allclose(n, 2.0, atol=0.02), n[:3]
    print("  level1 Int A/2pi dE = 1 state per site           OK")


def test_level2_rgf_equals_dense_inverse():
    rng = np.random.default_rng(0); N, K = 3, 5
    Hd = [(lambda A: A + A.conj().T)(rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))) for _ in range(K)]
    V = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
    SigL = -0.3j * np.eye(N) + 0.1 * rng.normal(size=(N, N)); SigR = -0.2j * np.eye(N)
    E = 0.37 + 1e-6j
    Hfull = np.zeros((N * K, N * K), complex)
    for i in range(K):
        Hfull[i*N:(i+1)*N, i*N:(i+1)*N] = Hd[i]
        if i + 1 < K:
            Hfull[i*N:(i+1)*N, (i+1)*N:(i+2)*N] = V; Hfull[(i+1)*N:(i+2)*N, i*N:(i+1)*N] = V.conj().T
    Hfull[:N, :N] += SigL; Hfull[-N:, -N:] += SigR
    G = inv(E * np.eye(N * K) - Hfull)
    grd, glc = L2.rgf(E, Hd, V, SigL, SigR, True)
    for i in range(K):
        assert np.allclose(grd[i], G[i*N:(i+1)*N, i*N:(i+1)*N], atol=1e-10)
        assert np.allclose(glc[i], G[i*N:(i+1)*N, -N:], atol=1e-10)
    print("  level2 RGF blocks == dense inverse (matrix V)  OK")


def test_level2_contours_vs_brute_force():
    kT = 0.0259; t = 1.0
    lead = L2.Lead(np.zeros((1, 1)), t, 0.0); V = -t * np.eye(1)
    U = np.array([0.0, 0.1, 0.3, 0.3, 0.1, 0.0]); Hd = [np.array([[2 * t + u]]) for u in U]; mu = 0.7
    def diagG(z):
        S = lead.self_energy(z); grd, _ = L2.rgf(z, Hd, V, S, S, False); return np.array([g[0, 0] for g in grd])
    E = np.linspace(-0.5, 1.6, 20001); dE = E[1] - E[0]
    ref = sum(-np.imag(diagG(e + 1e-6j)) / np.pi * L2.fermi(e, mu, kT).real for e in E) * dE
    z, w = L2.equilibrium_contour(mu, kT, -0.5, 24, 12, 8)
    rho_c = -np.imag(sum(wk * diagG(zk) for zk, wk in zip(z, w))) / np.pi
    z, w = L2.ozaki_contour(mu, kT, 40)
    rho_o = -np.imag(sum(wk * diagG(zk) for zk, wk in zip(z, w))) / np.pi
    assert np.abs(rho_c - ref).max() < 1e-4, np.abs(rho_c - ref).max()
    assert np.abs(rho_o - ref).max() < 1e-4, np.abs(rho_o - ref).max()
    print(f"  level2 contour ({np.abs(rho_c-ref).max():.1e}) & Ozaki ({np.abs(rho_o-ref).max():.1e}) == brute force   OK")


def test_level2_fermi():
    assert abs(L2.fermi(0.0, 0.0, 0.025) - 0.5) < 1e-12
    assert L2.fermi(np.array([-5.0, 5.0]), 0.0, 0.025).real.tolist() == [1.0, 0.0]
    assert abs(L2.fermi(0.1 + 0.2j, 0.0, 0.025)) < 1.0
    print("  level2 fermi() scalar/array/complex             OK")


def test_level2_uniform_gaa_wire_T_integer():
    p = L2.GAAParams(W=1.5, L_s=1.5, L_g=1.5, L_d=1.5, V_ds=0.0, eta=1e-9)   # tiny eta -> exact integers
    d = L2.Device(p, 0.26, 0.26, 0.26, 1, 0.0, 0.0)
    d.set_potential(np.zeros((d.Nx, d.N)))
    sub = np.sort(np.linalg.eigvalsh(d.H_cs))
    Es = np.array([sub[0] + 0.05, sub[1] + 0.05, sub[3] + 0.05])   # sub[1]==sub[2] degenerate
    T = d.transmission(Es)
    assert np.allclose(T, [1, 3, 4], atol=1e-5), T
    print("  level2 uniform GAA wire T = # open subbands     OK")


def test_level3_lk_hermitian_and_masses():
    g1, g2, g3, D = L3.MATERIALS["Si"]
    H = L3.lk6(0.3, -0.2, 0.5); assert np.allclose(H, H.conj().T)
    k = 0.01
    E001 = np.sort(np.linalg.eigvalsh(L3.lk6(0, 0, k)))[::-1]
    E111 = np.sort(np.linalg.eigvalsh(L3.lk6(k/3**.5, k/3**.5, k/3**.5)))[::-1]
    m = lambda E: -L3.E0 * k**2 / E
    assert abs(m(E001[0]) - 1/(g1 - 2*g2)) < 2e-3 and abs(m(E001[2]) - 1/(g1 + 2*g2)) < 3e-3
    assert abs(m(E111[0]) - 1/(g1 - 2*g3)) < 3e-3
    assert abs(np.sort(np.linalg.eigvalsh(L3.lk6(0, 0, 0)))[0] + D) < 1e-12      # split-off at -Delta
    w = L3.KPWire(4, 4, 0.4); Hb = w.bloch(0.7); assert np.allclose(Hb, Hb.conj().T)
    print("  level3 LK Hamiltonian Hermitian, masses correct  OK")


def test_level3_uniform_wire_integer_T_and_modespace():
    w = L3.KPWire(5, 5, 0.4); Eb = w.subbands(np.array([0.0]), 4)[0]
    ms = L3.ModeSpace(w, np.zeros(w.Ncs), 40)
    Es = np.array([Eb[0] - 0.01, Eb[2] - 0.01])
    T = L3.KPDevice(w, np.zeros((5, w.Ncs)), 0, 0, ms, eta=1e-9).transmission(Es)
    assert np.allclose(T, np.round(T), atol=2e-2), T      # 2e-2: M=40 mode truncation
    # mode space vs real space with a barrier
    U = np.tile((-0.1 * np.exp(-((np.arange(8) - 3.5) / 2.0)**2))[:, None], (1, w.Ncs))
    Es = np.linspace(Eb[0] - 0.15, Eb[0] + 0.01, 8)
    Trs = L3.KPDevice(w, U, 0, 0, None).transmission(Es)
    Tms = L3.KPDevice(w, U, 0, 0, L3.ModeSpace(w, np.zeros(w.Ncs), 90)).transmission(Es)
    assert np.abs(Trs - Tms).max() < 0.05, np.abs(Trs - Tms).max()
    print(f"  level3 uniform T integer; mode space (M=90) vs real space {np.abs(Trs-Tms).max():.3f}  OK")


if __name__ == "__main__":
    t0 = time.time(); fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try: fn()
            except Exception as e:
                fails += 1; print(f"  {name} FAILED: {e!r}")
    print(f"{'ALL PASSED' if not fails else str(fails)+' FAILED'}   ({time.time()-t0:.1f} s)")
    sys.exit(1 if fails else 0)
