"""handoff/breakdown_2d.py (Boyd's full Kn_GLL on the SU2 mesh) and handoff/report.py,
on synthetic fields with known answers, plus one real SU2 solution if present."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial import Delaunay

from handoff import breakdown_2d as b2d
from handoff import report
from sparta.gas import Gas, load_gas

N2 = Gas(species="N2", mass=4.6518e-26, d_ref=4.11e-10, omega=0.74, T_ref=273.15, alpha=1.36, rotdof=2)
ROOT = Path(__file__).resolve().parents[1]


def fake_field(x, r, rho, T, u, v=None):
    """Su2Field stand-in on a structured grid of points (what kn_field and plane_lines use)."""
    X, R = np.meshgrid(x, r, indexing="ij")
    xy = np.column_stack([X.ravel(), R.ravel()])
    f = lambda q: np.broadcast_to(q(X, R) if callable(q) else q, X.shape).ravel().astype(float)
    nodes = dict(rho=f(rho), T=f(T), u=f(u), v=f(v if v is not None else 0.0))
    return SimpleNamespace(xy=xy, nodes=nodes, tri=Delaunay(xy))


def test_node_gradients_exact_for_linear_fields():
    rng = np.random.default_rng(1)
    xy = rng.uniform(0, 1, (400, 2))
    tri = Delaunay(xy)
    vals = np.column_stack([3 * xy[:, 0] - 2 * xy[:, 1] + 1, 0.5 * xy[:, 1]])
    g = b2d.node_gradients(xy, tri.simplices, vals)
    np.testing.assert_allclose(g[:, 0], np.tile([3, -2], (400, 1)), atol=1e-9)
    np.testing.assert_allclose(g[:, 1], np.tile([0, 0.5], (400, 1)), atol=1e-9)


@pytest.mark.parametrize("axis", [0, 1])
def test_exponential_density_gives_lambda_over_L_in_x_and_r(axis):
    L, T = 1e-3, 100.0
    x = np.linspace(0, 4e-3, 81); r = np.linspace(0, 4e-3, 81)
    rho = (lambda X, R: 1e-3 * np.exp(-(X if axis == 0 else R) / L))
    fld = fake_field(x, r, rho, T, u=1000.0)
    kn, dom = b2d.kn_field(fld, N2)
    lam = N2.mean_free_path(fld.nodes["rho"] / N2.mass, T)
    inner = (fld.xy > 2e-4).all(1) & (fld.xy < 3.8e-3).all(1)
    # first-order error of linear triangle gradients: cell/L = 0.05 here -> about 1%
    np.testing.assert_allclose(kn[inner], lam[inner] / L, rtol=0.02)
    assert (dom[inner] == 0).all()  # density term dominates


def test_speed_gradient_uses_sound_speed_at_rest():
    # gas at rest except a gentle velocity ramp: |grad V| / a, not / V (which would blow up)
    x = np.linspace(0, 1e-3, 41); r = np.linspace(0, 1e-3, 41)
    fld = fake_field(x, r, 1e-3, 300.0, u=lambda X, R: 10.0 * X / 1e-3)
    kn, dom = b2d.kn_field(fld, N2)
    a = b2d.sound_speed(N2, 300.0)
    lam = N2.mean_free_path(1e-3 / N2.mass, 300.0)
    np.testing.assert_allclose(kn, lam * 1e4 / a, rtol=1e-6)
    assert (dom == 2).all()


def test_uniform_flow_has_no_breakdown_and_quantile_ignores_a_thin_edge():
    x = np.linspace(0, 5e-3, 21); r = np.linspace(0, 5e-3, 101)
    fld = fake_field(x, r, 1e-3, 100.0, u=800.0)
    xs, kn = b2d.knudsen_profile(fld, N2)
    assert np.all(kn < 1e-9) and b2d.first_breakdown(xs, kn, 0.01) is None
    # a broken-down outer edge carrying 0.5% of the mass flow: below a 1% quantile, above 0.1%
    kn_row = np.where(np.arange(100) >= 95, 1.0, 1e-4)[None, :]
    w = np.where(np.arange(100) >= 95, 0.001, 0.995 / 95)[None, :]
    assert b2d.mass_quantile(kn_row, w, 0.01)[0] == pytest.approx(1e-4)
    assert b2d.mass_quantile(kn_row, w, 0.001)[0] == pytest.approx(1.0)


def test_plane_lines_mass_flow():
    x = np.linspace(0, 1e-3, 5); r = np.linspace(0, 2e-3, 201)
    rho, u = 2e-3, 500.0
    fld = fake_field(x, r, rho, 100.0, u=u)
    kn = np.zeros(len(fld.xy))
    rr, kl, w, mdot = b2d.plane_lines(fld, kn, np.array([5e-4]))
    assert mdot[0] == pytest.approx(rho * u * np.pi * 2e-3**2, rel=2e-3)


def test_report_flags_a_broken_down_plane(tmp_path):
    # jet core r < 1 mm dense and slow-varying, outer annulus thin with a steep density drop
    x = np.linspace(0, 4e-3, 41); r = np.linspace(0, 3e-3, 121)
    rho = lambda X, R: 1e-2 * np.exp(-(R / 1e-3) ** 6) + 1e-7
    fld = fake_field(x, r, rho, 100.0, u=lambda X, R: 800.0 * np.exp(-(R / 1.5e-3) ** 2))
    info = dict(case="synthetic", gas="N2", vtu="none", module="handoff.breakdown_2d", thr=0.01,
                opts={}, mode="forced", D=1e-3, x_c=2e-3, x_break=None, profile=None)
    plane = dict(n=np.array([1e20, 1e22]), T=np.array([100.0, 100.0]), u=np.array([0.0, 800.0]))
    flows = dict(mdot_net=1e-5, mdot_inward=1.1e-5)
    txt = report.write(tmp_path, fld, N2, info, plane, flows, mass_fraction=0.01)
    assert (tmp_path / "report.txt").read_text() == txt
    assert "Checks:" in txt and "[ok] SU2 mass flow" in txt
    assert "[!!]" in txt and "has Kn_GLL(2D) > 0.01" in txt
    assert (tmp_path / "report.png").is_file()


@pytest.mark.skipif(not (ROOT / "runs/cd_nozzle_n2/su2/flow.vtu").is_file(), reason="no N2 nozzle SU2 run")
def test_n2_nozzle_axis_is_continuum_but_outer_jet_is_not():
    from lib.su2_field import Su2Field
    fld = Su2Field(ROOT / "runs/cd_nozzle_n2/su2/flow.vtu", xmin=-1e-9)
    gas = load_gas("n2")
    kn, _ = b2d.kn_field(fld, gas)
    _, kl, w, _ = b2d.plane_lines(fld, kn, np.array([5e-3]))
    assert kl[0, 0] < 0.01  # axis
    frac = (w * (kl > 0.05)).sum() / w.sum()
    assert 0.05 < frac < 0.6  # the nozzle boundary layer carries a sizeable share
