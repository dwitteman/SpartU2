"""Tests for handoff/breakdown_gll.py: synthetic profiles with known Kn_GLL, plus one SU2 run."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from handoff import breakdown_gll
from sparta.gas import Gas, load_gas

HE = Gas(species="He", mass=6.6342e-27, d_ref=2.33e-10, omega=0.66, T_ref=273.15, alpha=1.0)
ROOT = Path(__file__).resolve().parents[1]


def test_exponential_density_gives_lambda_over_L():
    # rho = rho0 exp(-x/L)  ->  |drho/dx|/rho = 1/L  ->  Kn = lambda / L
    L, rho0, T = 1e-3, 1e-2, 300.0
    x = np.linspace(0, 5e-3, 5001)
    rho = rho0 * np.exp(-x / L)
    kn = breakdown_gll.kn_gll(x, rho, np.full_like(x, T), HE)
    lam = HE.mean_free_path(rho / HE.mass, T)
    np.testing.assert_allclose(kn[1:-1], lam[1:-1] / L, rtol=1e-5)


def test_free_jet_power_law():
    # rho ~ x^-2 at constant T  ->  |drho/dx|/rho = 2/x  ->  Kn = 2 lambda / x, growing like x
    x = np.geomspace(1e-4, 1e-2, 4001)
    rho = 1e-8 * x**-2
    T = np.full_like(x, 50.0)
    kn = breakdown_gll.kn_gll(x, rho, T, HE)
    lam = HE.mean_free_path(rho / HE.mass, T)
    np.testing.assert_allclose(kn[1:-1], 2 * lam[1:-1] / x[1:-1], rtol=1e-4)
    assert np.all(np.diff(kn) > 0)


def test_uniform_flow_has_zero_kn():
    x = np.linspace(0, 1, 11)
    kn = breakdown_gll.kn_gll(x, np.full(11, 1e-3), np.full(11, 300.0), HE)
    np.testing.assert_allclose(kn, 0, atol=1e-15)  # round-off from linspace spacing


def test_mean_free_path_scales_with_T_exponent():
    # lambda ~ T^(omega - 1/2) at fixed n; at T_ref it is 1/(sqrt2 pi d^2 n)
    n = 1e22
    assert HE.mean_free_path(n, HE.T_ref) == pytest.approx(1 / (np.sqrt(2) * np.pi * HE.d_ref**2 * n))
    ratio = HE.mean_free_path(n, 4 * HE.T_ref) / HE.mean_free_path(n, HE.T_ref)
    assert ratio == pytest.approx(4 ** (HE.omega - 0.5))


@pytest.mark.parametrize("x", [np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.5]), np.array([1.0])])
def test_rejects_bad_x(x):
    with pytest.raises(ValueError):
        breakdown_gll.kn_gll(x, np.ones_like(x), np.ones_like(x), HE)


def test_first_breakdown():
    x = np.linspace(0, 1, 11)
    assert breakdown_gll.first_breakdown(x, x, 0.45) == pytest.approx(0.5)
    assert breakdown_gll.first_breakdown(x, x, 0.5) == pytest.approx(0.6)  # strictly greater
    assert breakdown_gll.first_breakdown(x, x, 2.0) is None
    # non-monotonic: the first crossing wins, even if Kn drops again afterwards
    assert breakdown_gll.first_breakdown(x, [0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0], 0.05) == pytest.approx(0.2)


def test_kn_at_interpolates():
    assert breakdown_gll.kn_at(np.array([0.0, 1.0]), np.array([0.0, 0.2]), 0.25) == pytest.approx(0.05)


def test_axis_profile_selects_sorts_and_dedups():
    xy = np.array([[2.0, 0.0], [0.0, 0.0], [1.0, 0.5], [1.0, 0.0], [2.0, 0.0]])
    field = SimpleNamespace(xy=xy, nodes={"rho": np.array([3.0, 1.0, 9.0, 2.0, 3.0]),
                                          "T": np.array([30.0, 10.0, 90.0, 20.0, 30.0])})
    x, rho, T = breakdown_gll.axis_profile(field)
    np.testing.assert_array_equal(x, [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(rho, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(T, [10.0, 20.0, 30.0])


VTU = ROOT / "runs" / "ASDEX_capil_01" / "su2" / "flow.vtu"


@pytest.mark.skipif(not VTU.is_file(), reason="needs the ASDEX_capil_01 SU2 solution")
def test_su2_capillary_breaks_down_between_4_and_6_D():
    from lib.su2_field import Su2Field

    D = 0.4e-3
    x, kn = breakdown_gll.axis_knudsen(Su2Field(VTU, xmin=-1e-9), load_gas("he.species"))
    xb = breakdown_gll.first_breakdown(x, kn, 0.05)  # the published 0.05 criterion, whatever the default
    assert xb is not None and 4 * D < xb < 6 * D
    assert breakdown_gll.kn_at(x, kn, 4 * D) < 0.05
    near = (x > 0.5 * D) & (x < 4 * D)  # free-jet region: Kn grows downstream
    assert np.all(np.diff(kn[near]) > -0.1 * kn[near][1:])
