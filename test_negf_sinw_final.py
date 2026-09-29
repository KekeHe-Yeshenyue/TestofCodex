#!/usr/bin/env python3
"""
Tests for negf_sinw_final.py.   Run:  python test_negf_sinw_final.py   (or: pytest -q test_negf_sinw_final.py)

Fast tests (~20 s) check physics identities; set NEGF_SLOW=1 to also run a full
self-consistent bias point (~2 min) and compare it with the independent
negf_suite/negf_level2_gaa_transiesta_style.py implementation.
"""
import os
import sys
import time

import numpy as np

import negf_sinw_final as F


def test_selftest_identities():
    """RGF vs dense inverse, Lopez-Sancho vs analytic, contour/Ozaki vs real axis,
    integer transmission, bond current vs Landauer, Poisson residual, k.p masses."""
    res = F.selftest(verbose=False)
    failed = [(n, d) for n, ok, d in res if not ok]
    assert not failed, failed


def test_circle_mask_and_hamiltonian():
    p = F.NWParams(shape="circle", W=2.4, a=0.3)
    m = F.cross_section_mask(p)
    assert m.shape == (8, 8) and m.sum() < 64 and not m[0, 0] and m[3, 4]
    H = F.cross_section_hamiltonian(m, p.a, 0.26, 0.26)
    assert np.allclose(H, H.T) and H.shape == (m.sum(), m.sum())
    # hard-wall circle ground state lies between inscribed and circumscribed squares
    e0 = np.linalg.eigvalsh(H)[0]
    sq = lambda w: 2 * F.E0_NM / 0.26 * (np.pi / w) ** 2
    assert sq(2.4 + 0.3) * 0.9 < e0 < sq(2.4 / np.sqrt(2)) * 1.3, e0


def test_lead_neutrality_monotonic_in_doping():
    p = F.NWParams(W=1.8)
    m = F.cross_section_mask(p)
    u1 = F.neutral_lead_shift(F.NWParams(W=1.8, N_D=5e19), m)
    u2 = F.neutral_lead_shift(F.NWParams(W=1.8, N_D=2e20), m)
    assert u2 < u1 < 0, (u1, u2)                      # more donors -> Fermi level higher in the band


def test_lead_fermi_level_light_mass_materials():
    """Regression: InGaAs (m* = 0.041) has its first subband ~4 eV up in a 1.8 nm wire;
    the Fermi-level search must follow it (a fixed bracket used to clip at 3 eV)."""
    for mat in ("Si", "InGaAs"):
        p = F.NWParams(**F.QUICK, material=mat)
        m = F.cross_section_mask(p)
        mx, my, mz, _ = p.valley_list[0]
        H = F.cross_section_hamiltonian(m, p.a, my, mz)
        e_sub = np.linalg.eigvalsh(H)[0]
        ef = -F.neutral_lead_shift(p, m)
        assert e_sub < ef < e_sub + 0.5, (mat, e_sub, ef)


def test_poisson_gate_controls_channel():
    p = F.NWParams(W=1.8, L_s=1.5, L_g=3.0, L_d=1.5, V_g=0.8)
    m = F.cross_section_mask(p)
    phi_hi = F.Poisson3D(p, m, 0.5, 0.5)
    p2 = F.NWParams(W=1.8, L_s=1.5, L_g=3.0, L_d=1.5, V_g=0.0)
    phi_lo = F.Poisson3D(p2, m, 0.5, 0.5)
    n = np.zeros(phi_hi.n_site * p.Nx)
    out = []
    for P in (phi_hi, phi_lo):
        phi0 = np.zeros(P.dir_mask.size); phi0[P.dir_mask] = P.dir_val[P.dir_mask]
        phi = P.solve(n, phi0, 1e9)
        out.append(P.core_potential(phi)[p.Nx // 2].mean())
    # U = -phi: a higher gate voltage lowers the electron potential energy mid-channel
    assert out[0] < out[1], out


def test_export_hamiltonian_roundtrip(tmp_path=None):
    import scipy.sparse as sp
    import tempfile
    d = tmp_path or tempfile.mkdtemp(prefix="negf_test_")
    p = F.NWParams(W=0.9, L_s=0.6, L_g=0.6, L_d=0.6)
    m = F.cross_section_mask(p)
    dev = F.Device(p, m, 0.26, 0.26, 0.26, 1, 0.0, 0.0)
    dev.set_potential(np.zeros((dev.Nx, dev.N)))
    info = F.export_hamiltonian(dev, os.path.join(str(d), "h.npz"), "npz")
    H = sp.load_npz(info["file"])
    assert info["hermitian"] and H.shape == (dev.Nx * dev.N,) * 2


def test_no_scf_bias_runs_and_current_positive():
    p = F.NWParams(**{**F.QUICK, "scf": False, "V_g": 0.6, "V_ds": 0.2, "W": 1.2, "L_s": 1.5, "L_g": 3.0, "L_d": 1.5})
    res, post = F.run_bias(p, verbose=False, ldos=False)
    # bond current is conserved to O(eta): eta = 1e-4 eV acts as a weak sink (~1-2 %)
    assert post["I"] > 0 and post["bond_dev"] < 5e-2, (post["I"], post["bond_dev"])


def test_kp_mode_space_converges():
    w = F.KPWire(5, 5, 0.4)
    Eb = w.subbands(np.array([0.0]), 4)[0]
    U = np.tile((-0.1 * np.exp(-((np.arange(8) - 3.5) / 2.0) ** 2))[:, None], (1, w.Ncs))
    Es = np.linspace(Eb[0] - 0.15, Eb[0] + 0.01, 8)
    Trs = F.KPDevice(w, U, 0, 0, None).transmission(Es)
    Tms = F.KPDevice(w, U, 0, 0, F.ModeSpace(w, 90)).transmission(Es)
    assert np.abs(Trs - Tms).max() < 0.05


def test_slow_scf_regression_vs_level2():
    """Same device and numerics as negf_suite level 2 --quick --Vg 0.5: I_d = 2.9919 uA there."""
    if not os.environ.get("NEGF_SLOW"):
        print("    (skipped; set NEGF_SLOW=1)")
        return
    p = F.NWParams(**{**F.QUICK, "V_g": 0.5, "n_workers": int(os.environ.get("NEGF_WORKERS", "1"))})
    res, post = F.run_bias(p, verbose=False, ldos=False)
    assert abs(post["I"] * 1e6 - 2.9919) < 0.01, post["I"]
    assert post["bond_dev"] < 5e-2


if __name__ == "__main__":
    t0, fails = time.time(), 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            t = time.time()
            try:
                fn()
                print(f"  PASS  {name}  ({time.time()-t:.1f} s)")
            except Exception as e:
                fails += 1
                print(f"  FAIL  {name}: {e!r}")
    print(f"{'ALL PASSED' if not fails else str(fails) + ' FAILED'}   ({time.time()-t0:.1f} s)")
    sys.exit(1 if fails else 0)
