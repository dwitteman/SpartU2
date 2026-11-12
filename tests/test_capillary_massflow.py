"""Capillary mass flow: lib/capillary.py's analytical models, and SU2 runs checked against them.

The SU2 mass flow tested here is the flow through the capillary, integrated from the
flow.vtu nodes across tube sections. It is not SU2's "total" Avg_Massflow, which is
inlet + outlet (the net imbalance of the whole domain).

Expected ordering (see lib/capillary.py): SU2 has an isothermal 300 K wall and
resolves the developing entrance flow, so its mass flow should lie below the
adiabatic, fully developed Fanno value and close to the isothermal model with the
Hagenbach entrance loss.
"""
import re
import tomllib
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad

from lib import capillary as cap

ROOT = Path(__file__).resolve().parents[1]
HE = cap.IdealGas.from_su2("he")
CASE = dict(D=0.4e-3, L=66e-3, p0=1e5, T0=300.15)  # the ASDEX capillary

# SU2 runs used as references: converged, 1 bar. Missing runs are skipped.
REFERENCE_RUNS = ["ASDEX_capil_01", "ASDEX_capil_02", "capillary_400um_1bar"]
K_ENTRANCE = 1.25        # Hagenbach factor, laminar circular tube (Shah & London 1978)
TOL_CONSERVATION = 0.02  # max spread of mdot between tube sections
TOL_SU2_LOG = 0.01       # vtu-integrated inlet mdot vs SU2's own inlet Avg_Massflow
TOL_ISOTHERMAL = 0.05    # SU2 vs isothermal + entrance model


class ConstMuGas(cap.IdealGas):
    """Gas with constant viscosity, so the friction factor is constant along the tube."""

    def mu(self, T):
        return np.full_like(np.asarray(T, float), self.mu_ref)


# ---- analytical model -----------------------------------------------------------------

def test_sutherland_reference_point():
    assert HE.mu(HE.T_ref) == pytest.approx(HE.mu_ref)
    assert HE.mu(400.0) > HE.mu(300.0) > HE.mu(200.0)


@pytest.mark.parametrize("gamma", [1.4, 5 / 3])
@pytest.mark.parametrize("M", [0.05, 0.2, 0.5, 0.9])
def test_fanno_ode_integrates_to_closed_form(gamma, M):
    val, _ = quad(lambda m: cap.fanno_dfdm(m, gamma), M, 1.0)
    assert val == pytest.approx(cap.fanno_length(M, gamma), rel=1e-8)


def test_choked_orifice_matches_textbook_formula():
    # mdot* = A p0 sqrt(gamma/(R T0)) (2/(gamma+1))^((gamma+1)/(2(gamma-1)))
    g = HE.gamma
    expect = (cap.area(CASE["D"]) * CASE["p0"] * np.sqrt(g / (HE.R * CASE["T0"]))
              * (2 / (g + 1)) ** ((g + 1) / (2 * (g - 1))))
    assert cap.choked_orifice(CASE["D"], CASE["p0"], CASE["T0"], HE) == pytest.approx(expect, rel=1e-12)


def test_fanno_constant_viscosity_matches_closed_form():
    # constant mu -> constant f_D = 64/Re, so f_D L/D must equal fanno_length(M1)
    gas = ConstMuGas(HE.gamma, HE.R, 2e-5, 273.15, 0.0)
    r = cap.fanno_laminar(CASE["D"], CASE["L"], CASE["p0"], CASE["T0"], gas)
    f = 64 / r["Re_in"]
    assert r["Re_in"] == pytest.approx(r["Re_exit"])
    assert f * CASE["L"] / CASE["D"] == pytest.approx(cap.fanno_length(r["M1"], gas.gamma), rel=1e-6)


def test_isothermal_constant_viscosity_matches_closed_form():
    r = cap.isothermal_laminar(CASE["D"], CASE["L"], CASE["p0"], CASE["T0"], HE)
    f = 64 / r["Re"]
    assert r["M1"] < 1 / np.sqrt(HE.gamma)
    assert f * CASE["L"] / CASE["D"] == pytest.approx(cap.isothermal_length(r["M1"], HE.gamma), rel=1e-6)


def test_short_tube_approaches_choked_orifice():
    mdot_star = cap.choked_orifice(CASE["D"], CASE["p0"], CASE["T0"], HE)
    short = cap.fanno_laminar(CASE["D"], 1e-6, CASE["p0"], CASE["T0"], HE)
    assert short["mdot"] < mdot_star
    assert short["mdot"] == pytest.approx(mdot_star, rel=0.02)


def test_mass_flow_trends():
    args = (CASE["D"], CASE["L"], CASE["p0"], CASE["T0"], HE)
    base = cap.fanno_laminar(*args)["mdot"]
    longer = cap.fanno_laminar(CASE["D"], 2 * CASE["L"], CASE["p0"], CASE["T0"], HE)["mdot"]
    higher_p = cap.fanno_laminar(CASE["D"], CASE["L"], 2 * CASE["p0"], CASE["T0"], HE)["mdot"]
    with_entrance = cap.fanno_laminar(*args, k_entrance=K_ENTRANCE)["mdot"]
    assert longer < base < higher_p
    assert with_entrance < base
    # laminar, low inlet Mach: f L/D ~ 1/(gamma M1^2) and f ~ 1/mdot give mdot ~ p0^2
    assert 3.0 < higher_p / base < 4.0


def test_isothermal_wall_at_T0_gives_less_flow_than_adiabatic():
    # heating keeps the gas hot (higher viscosity) and chokes earlier (M = 1/sqrt(gamma))
    args = (CASE["D"], CASE["L"], CASE["p0"], CASE["T0"], HE)
    assert cap.isothermal_laminar(*args)["mdot"] < cap.fanno_laminar(*args)["mdot"]


# ---- SU2 runs -------------------------------------------------------------------------

def _su2_run(name):
    d = ROOT / "runs" / name / "su2"
    if not (d / "flow.vtu").is_file() or not (d / "case.toml").is_file():
        pytest.skip(f"no SU2 solution for {name}")
    with open(d / "case.toml", "rb") as f:
        case = tomllib.load(f)
    from lib.su2_field import Su2Field

    g, gs = case["geometry"], case["gas"]
    D, L = g["throat_diam_mm"] * 1e-3, g["throat_length_mm"] * 1e-3
    x, mdot = cap.tube_massflows(Su2Field(d / "flow.vtu"), D / 2, np.linspace(-L, 0, 12))
    gas = cap.IdealGas.from_su2(Path(gs["species_file"]).stem)
    return dict(dir=d, D=D, L=L, p0=gs["plen_pres_pa"], T0=gs["plen_temp_K"],
                Tw=case["su2"]["wall_temp_K"], gas=gas, x=x, mdot=mdot)


def _su2_log_massflows(run_dir):
    """Last screen line's Avg_Massflow values from su2.log, keyed total + MARKER_ANALYZE names."""
    log, cfg = run_dir / "su2.log", run_dir / "su2.cfg"
    if not log.is_file():
        return None
    markers = re.search(r"^MARKER_ANALYZE=\s*\(([^)]*)\)", cfg.read_text(), re.M)
    names = ["total"] + [m.strip() for m in markers.group(1).split(",")] if markers else None
    header, last = None, None
    for line in log.read_text(errors="replace").splitlines():
        if "Avg_Massflow" in line:
            header = [c.strip() for c in line.strip("|").split("|")]
        elif header and re.match(r"^\|\s*\d+\|", line):
            last = [c.strip() for c in line.strip("|").split("|")]
    if not (header and last and names):
        return None
    vals = [float(v) for h, v in zip(header, last) if h == "Avg_Massflow"]
    return dict(zip(names, vals)) if len(vals) == len(names) else None


@pytest.fixture(scope="module", params=REFERENCE_RUNS)
def su2(request):
    return _su2_run(request.param)


def test_su2_mass_conserved_along_tube(su2):
    m = su2["mdot"]
    assert np.ptp(m) / m.mean() < TOL_CONSERVATION, dict(zip(np.round(su2["x"] * 1e3, 1), m))


def test_su2_inlet_matches_su2_log(su2):
    d = su2["dir"]
    if (d / "su2.log").is_file() and (d / "su2.log").stat().st_mtime < (d / "flow.vtu").stat().st_mtime - 60:
        pytest.skip("su2.log is older than flow.vtu, so it is from a different run")
    log = _su2_log_massflows(d)
    if not log or "inlet" not in log:
        pytest.skip("no inlet Avg_Massflow column in su2.log")
    assert su2["mdot"][0] == pytest.approx(log["inlet"], rel=TOL_SU2_LOG)


def test_su2_below_adiabatic_fanno(su2):
    fanno = cap.fanno_laminar(su2["D"], su2["L"], su2["p0"], su2["T0"], su2["gas"])
    assert su2["mdot"].mean() < fanno["mdot"]


def test_su2_matches_isothermal_with_entrance_loss(su2):
    iso = cap.isothermal_laminar(su2["D"], su2["L"], su2["p0"], su2["T0"], su2["gas"],
                                 T_wall=su2["Tw"], k_entrance=K_ENTRANCE)
    assert su2["mdot"].mean() == pytest.approx(iso["mdot"], rel=TOL_ISOTHERMAL)
