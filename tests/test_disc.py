"""Disc basis-expansion tests (Milestone 5).

Covers the Miyamoto-Nagai primitives, the Gram-matrix quadrature, and the full
DiscPotential pipeline on a synthetic exponential disc: agreement with direct
summation, the advantage over the spheroidal HO basis for a thin disc, and
integrate -> analyse -> classify running on the disc potential.

    python tests/test_disc.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lanfear as lf  # noqa: E402
from lanfear import _core  # noqa: E402
from lanfear.disc_potential import DiscPotential, _mn_density, _mn_potential  # noqa: E402
from lanfear.orbits import OrbitResults, SUMMARY_COLUMNS  # noqa: E402


def exponential_disc(n=200_000, Rd=3.0, z0=0.3, m_total=1e10, seed=7):
    rng = np.random.default_rng(seed)
    R = rng.gamma(2.0, Rd, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    z = z0 * np.arctanh(rng.uniform(-1, 1, n))
    pos = np.stack([R * np.cos(phi), R * np.sin(phi), z], axis=1)
    ps = lf.ParticleSystem(
        pos=pos,
        vel=np.zeros((n, 3)),
        mass=np.full(n, m_total / n),
        ids=np.arange(n),
        species=np.full(n, "STAR"),
    )
    ps.prepare(centre="shrinking_sphere")
    return ps


def test_mn_primitives():
    """The numpy MN helpers match the C++ ones bit-for-bit."""
    for x, y, z, a, b in [
        (1.0, 0.5, 0.7, 1.5, 0.3),
        (2.0, 0, 0.3, 0.5, 0.1),
        (0.1, -0.2, 0.05, 2.0, 0.5),
    ]:
        assert (
            abs(_mn_potential(x, y, z, a, b) - _core.mn_potential(x, y, z, a, b))
            < 1e-12
        )
        R = np.hypot(x, y)
        assert abs(_mn_density(R, z, a, b) - _core.mn_density(R, z, a, b)) < 1e-12
    print("MN primitives (numpy == C++): OK")


def test_gram_matrix():
    """Gram quadrature is converged, symmetric, and matches scipy."""
    a = np.array([0.3, 1.0, 2.5])
    b = np.array([0.1, 0.3, 0.3])
    g300 = DiscPotential._gram_matrix(a, b, n_quad=300)
    g600 = DiscPotential._gram_matrix(a, b, n_quad=600)
    assert np.allclose(g300, g600, rtol=1e-4)  # converged in n_quad
    assert np.allclose(g300, g300.T, rtol=1e-6)  # symmetric
    try:
        from scipy import integrate

        f = lambda zz, R: (
            2
            * 2
            * np.pi
            * R
            * _mn_density(R, zz, a[1], b[1])
            * _mn_potential(R, 0, zz, a[1], b[1])
        )
        truth, _ = integrate.dblquad(
            f, 1e-6, 200, lambda R: 1e-6, lambda R: 200, epsabs=1e-9, epsrel=1e-7
        )
        assert abs(g300[1, 1] - truth) / abs(truth) < 2e-3
        print(
            f"Gram matrix: converged, symmetric, matches scipy "
            f"({g300[1,1]:.5f} vs {truth:.5f})"
        )
    except ImportError:
        print("Gram matrix: converged and symmetric (scipy unavailable)")


def test_eval_exact():
    """Evaluating with a unit coefficient reproduces the analytic MN potential."""
    a = np.array([0.3, 1.0, 2.5])
    b = np.array([0.3, 0.3, 0.3])
    core = _core.DiscPotential(a, b)
    core.set_coefficients(np.array([0.0, 1.0, 0.0]))
    err = max(
        abs(core.potential(*p) - _mn_potential(*p, a[1], b[1]))
        for p in [(1.0, 0, 0), (2.0, 0, 0.5), (0.5, 0.5, 0.3), (3.0, 0, 1.0)]
    )
    assert err < 1e-12, err
    print("eval with unit coefficient == analytic MN: OK")


def test_exponential_disc_validation():
    ps = exponential_disc()
    disc = lf.DiscPotential.from_particles(ps, n_radial=10, n_vert=3, rcond=1e-4)
    total = np.sum(disc.core.coefficients)
    v = disc.validate(n_points=3000)
    print(f"  sum(c)={total:.3f}, {v}")
    assert 0.85 < total < 1.15  # monopole ~ total mass
    assert v.median < 0.02  # < 2% agreement
    return ps, disc


def test_beats_ho_in_plane():
    """For a thin disc the disc basis beats the spheroidal HO basis in-plane."""
    ps = exponential_disc(z0=0.09, seed=1)  # z0/Rd = 0.03, very thin
    disc = lf.DiscPotential.from_particles(ps, n_radial=12, n_vert=3)
    ho = lf.Potential.from_particles(ps, n_max=16, l_max=8)
    pos_ho = ps.field.pos / ps.scale_radius
    m_ho = ps.field.mass / ps.field.mass.sum()

    Rg = np.linspace(0.3, 4, 12)
    pts = np.stack([Rg, np.zeros_like(Rg), np.zeros_like(Rg)], axis=1)  # in-plane
    d = pts[:, None, :] - pos_ho[None, :, :]
    direct = -np.sum(
        m_ho[None, :] / np.sqrt(np.einsum("pij,pij->pi", d, d) + 0.03**2), axis=1
    )
    err_disc = np.median(
        np.abs(disc.core.potential_batch(pts) - direct) / np.abs(direct)
    )
    err_ho = np.median(np.abs(ho.core.potential_batch(pts) - direct) / np.abs(direct))
    print(f"  thin disc in-plane: disc={err_disc:.2%}  HO={err_ho:.2%}")
    assert err_disc < err_ho


def test_pipeline_and_pickle():
    """integrate + analyse + classify on the disc potential, plus pickle."""
    import pickle

    ps = exponential_disc(n=60_000, seed=1)
    disc = lf.DiscPotential.from_particles(ps, n_radial=10, n_vert=3)

    # pickle round-trip of the C++ core.
    core2 = pickle.loads(pickle.dumps(disc.core))
    assert (
        abs(disc.core.potential(1.3, 0.2, -0.4) - core2.potential(1.3, 0.2, -0.4))
        < 1e-12
    )

    # Circular in-plane initial conditions (HO units) -> disc loop orbits.
    pos_ho = ps.field.pos[:1500] / ps.scale_radius
    acc = disc.acceleration(ps.field.pos[:1500])
    Rc = np.hypot(pos_ho[:, 0], pos_ho[:, 1])
    aR = -(acc[:, 0] * pos_ho[:, 0] + acc[:, 1] * pos_ho[:, 1]) / np.maximum(Rc, 1e-6)
    vc = np.sqrt(np.maximum(aR * Rc, 1e-9))
    ph = np.arctan2(pos_ho[:, 1], pos_ho[:, 0])
    vel_ho = np.stack([-vc * np.sin(ph), vc * np.cos(ph), np.zeros(len(vc))], axis=1)
    states = np.concatenate([pos_ho, vel_ho], axis=1)

    summ, fund, lines = disc.core.analyse_batch(
        states, n_periods=20, n_samples=2048, n_lines=4
    )
    res = OrbitResults(
        ids=np.arange(len(states)),
        summary=summ,
        columns=SUMMARY_COLUMNS,
        time_unit=disc.time_unit,
        length_unit=disc.scale_radius,
        n_periods=20,
        n_samples=2048,
        initial_radius=np.linalg.norm(states[:, :3], axis=1),
        fundamentals=fund,
        lines=lines,
    )
    ok = res.ok
    assert np.mean(ok) > 0.95
    assert np.median(summ[ok, 5]) < 1e-5  # energy_drift column
    cl = res.classify()
    counts = cl.counts()
    # A flattened disc of near-circular orbits is dominated by short-axis tubes
    # and (1:1:1) rosettes.
    n_sat = counts.get("short_axis_tube", 0) + counts.get("rosette", 0)
    print(f"  disc pipeline: {int(ok.sum())} ok, families={counts}")
    assert n_sat > 0.8 * int(ok.sum())
    print("pipeline + pickle: OK")


if __name__ == "__main__":
    print("== MN primitives ==")
    test_mn_primitives()
    print("== Gram matrix ==")
    test_gram_matrix()
    print("== eval exactness ==")
    test_eval_exact()
    print("== exponential disc ==")
    test_exponential_disc_validation()
    print("== disc vs HO (thin) ==")
    test_beats_ho_in_plane()
    print("== pipeline + pickle ==")
    test_pipeline_and_pickle()
    print("\nALL DISC TESTS PASSED")
