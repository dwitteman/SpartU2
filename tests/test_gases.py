"""Gas data consistency between SU2 (su2/setup.py GASES) and SPARTA (sparta/data/*.species/.vss)."""
import numpy as np
import pytest

from lib.capillary import IdealGas
from sparta.gas import DATA, KB, load_gas
from su2.setup import GASES

AMU = 1.66053907e-27


@pytest.mark.parametrize("stem", sorted(GASES))
def test_every_su2_gas_has_sparta_data(stem):
    assert (DATA / f"{stem}.species").is_file() and (DATA / f"{stem}.vss").is_file()
    load_gas(f"{stem}.species")  # also checks the species IDs in the two files agree


@pytest.mark.parametrize("stem", sorted(GASES))
def test_gas_constant_matches_molecular_mass(stem):
    gas = load_gas(f"{stem}.species")
    # 0.5%: he.species/ar.species store molmass ~0.2% below molwt * amu
    assert GASES[stem]["R"] == pytest.approx(KB / gas.mass, rel=5e-3)


@pytest.mark.parametrize("stem", ["he", "ar"])
def test_su2_viscosity_matches_vss_near_room_temperature(stem):
    # he/ar Sutherland constants are anchored at 273 K; d2 and n2 are fitted over a wide range
    T = np.linspace(250, 400, 31)
    su2, vss = IdealGas.from_su2(stem).mu(T), load_gas(f"{stem}.species").viscosity(T)
    assert np.abs(su2 / vss - 1).max() < 0.06


@pytest.mark.parametrize("stem, T_low, tol", [("d2", 20, 0.075), ("n2", 50, 0.05)])
def test_viscosity_fit_covers_nozzle_temperatures(stem, T_low, tol):
    # minimax-fitted down to T_low (d2 7.2%, n2 4.7% worst case), room temperature included
    # (he/ar only match near 273 K; see su2/setup.py)
    T = np.geomspace(T_low, 400, 200)
    su2, vss = IdealGas.from_su2(stem).mu(T), load_gas(f"{stem}.species").viscosity(T)
    assert np.abs(su2 / vss - 1).max() < tol


def test_d2_against_measured_viscosity_and_molecular_data():
    gas = load_gas("d2.species")
    assert gas.viscosity(293.15) == pytest.approx(1.26e-5, rel=0.02)  # measured D2, ~12.6 uPa s
    assert gas.mass == pytest.approx(4.0282 * AMU, rel=1e-3)
    assert GASES["d2"]["gamma"] == pytest.approx(1.4)


def test_n2_against_measured_viscosity_and_molecular_data():
    gas = load_gas("n2.species")
    assert gas.viscosity(300.0) == pytest.approx(1.78e-5, rel=0.02)  # measured N2, ~17.8 uPa s
    assert gas.viscosity(200.0) == pytest.approx(1.29e-5, rel=0.03)
    assert gas.mass == pytest.approx(28.0134 * AMU, rel=1e-3)
    assert GASES["n2"]["gamma"] == pytest.approx(1.4)


@pytest.mark.parametrize("stem", sorted(GASES))
def test_relaxation_numbers_are_probabilities(stem):
    # SPARTA's RotRel/VibRel columns are 1/Z (relax if rotrel >= uniform random), not Z
    row = next(r for r in (l.split("#")[0].split() for l in (DATA / f"{stem}.species").read_text().splitlines()) if r)
    rotdof, rotrel, vibdof, vibrel = int(row[3]), float(row[4]), int(row[5]), float(row[6])
    assert 0 <= rotrel <= 1 and 0 <= vibrel <= 1
    if rotdof:
        assert rotrel > 0
