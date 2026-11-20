"""sparta/chamber.py: pumped walls and background reservoir for the DSMC chamber (no SPARTA run)."""
import math
import re

import pytest

from sparta import chamber as sc
from sparta.gas import KB, load_gas
from sparta.setup import in_continue, in_run

N2 = load_gas("n2.species")
BOX = (-0.01, 0.12, 0.0, 0.06)
MDOT = 1.1e-6
IO = dict(out_every=1000, avg_steps=500, restart_every=5000)


def case(**sparta):
    return {"su2": {"wall_temp_K": 293.15}, "sparta": sparta}


def commands(text):
    return [l.split()[0] for l in text.splitlines() if l.strip() and not l.startswith("#")]


def test_outflow_is_the_default_and_unchanged():
    ch = sc.chamber_model(case(), N2, MDOT, BOX)
    assert ch["mode"] == "outflow" and ch["boundary"] == "o ao p"
    assert in_run("c", BOX, "N2", "n2", 1e17, 1e-8, 10000, chamber=ch, **IO) == \
        in_run("c", BOX, "N2", "n2", 1e17, 1e-8, 10000, **IO)


def test_pump_balance_both_ways():
    A = math.pi * 0.06**2 + 2 * math.pi * 0.06 * 0.13
    assert sc.pumped_area(BOX) == pytest.approx(A)
    N_in, cbar = MDOT / N2.mass, N2.mean_speed(293.15)
    ch = sc.chamber_model(case(chamber="pump", pump_pres_pa=0.125), N2, MDOT, BOX)
    assert ch["f"] == pytest.approx(N_in / (0.125 / (KB * 293.15) * cbar / 4 * A))
    back = sc.chamber_model(case(chamber="pump", pump_fraction=ch["f"]), N2, MDOT, BOX)
    assert back["P"] == pytest.approx(0.125)
    assert ch["boundary"] == "os as p" and ch["fill"]["n"] == pytest.approx(0.125 / (KB * 293.15))


@pytest.mark.parametrize("sparta", [dict(chamber="pump"),                                   # neither
                                    dict(chamber="pump", pump_fraction=0.1, pump_pres_pa=1),  # both
                                    dict(chamber="pump", pump_fraction=1.5),
                                    dict(chamber="pump", pump_pres_pa=1e-6),                  # f > 1
                                    dict(chamber="reservoir"),                                # no pressure
                                    dict(chamber="reservoir", background_pres_pa=0.1, background_gas={"ar": 0}),
                                    dict(chamber="vacuum")])
def test_bad_settings_stop(sparta):
    with pytest.raises(SystemExit):
        sc.chamber_model(case(**sparta), N2, MDOT, BOX)


def test_reservoir_composition_and_renamed_beam_species():
    ch = sc.chamber_model(case(chamber="reservoir", background_pres_pa=0.2,
                               background_gas={"ar": 3, "n2": 1}), N2, MDOT, BOX)
    assert [(b["id"], b["frac"]) for b in ch["bg"]] == [("Ar", 0.75), ("N2bg", 0.25)]
    assert ch["n_bg"] == pytest.approx(0.2 / (KB * 293.15))
    files = sc.species_files(ch, "n2")
    ids = [l.split()[0] for l in files["gas.species"].splitlines() if not l.startswith("#")]
    assert ids == ["N2", "Ar", "N2bg"]
    vss = {l.split()[0]: l.split()[1:] for l in files["gas.vss"].splitlines() if not l.startswith("#")}
    assert vss["N2bg"] == vss["N2"]  # same gas, own ID
    # default background: the beam gas
    ch = sc.chamber_model(case(chamber="reservoir", background_pres_pa=0.1), N2, MDOT, BOX)
    assert [b["id"] for b in ch["bg"]] == ["N2bg"]


def test_pump_inputs():
    ch = sc.chamber_model(case(chamber="pump", pump_fraction=0.12), N2, MDOT, BOX)
    run = in_run("c", BOX, "N2", "n2", 1e17, 1e-8, 10000, chamber=ch, **IO)
    assert "boundary        os as p" in run
    assert re.search(r"surf_react\s+pump global 0.12 0.0", run)
    assert "bound_modify    xhi yhi collide pump react pump" in run
    assert "create_particles fill n 0" in run
    assert "emit/face/file gas xlo" in run  # one species: inflow mixture unchanged
    cont = in_continue("c", "n2", 10000, chamber=ch, species="N2", **IO)
    assert "bound_modify    xhi yhi collide pump react pump" in cont  # models are not in the restart
    assert "create_particles" not in cont  # no refill on a continuation
    c = commands(run)
    assert c.index("bound_modify") < c.index("create_particles") < c.index("run")


def test_reservoir_inputs_with_a_second_species():
    ch = sc.chamber_model(case(chamber="reservoir", background_pres_pa=0.125, background_gas={"ar": 1}),
                          N2, MDOT, BOX)
    run = in_run("c", BOX, "N2", "n2", 1e17, 1e-8, 10000, chamber=ch, **IO)
    assert "species         gas.species N2 Ar" in run
    assert "mixture         beam N2" in run and "emit/face/file beam xlo" in run
    assert "fix             bg emit/face bg xhi yhi" in run
    assert "collide         vss gas gas.vss" in run
    assert "boundary        o ao p" in run
    # per-species columns after the existing ones
    assert "flow.*.dump columns after g_lambda: nrho_N2 u_N2 v_N2 nrho_Ar u_Ar v_Ar T_N2 T_Ar" in run
    assert "g_su2_nrho g_lambda f_avg[6]" in run and "f_avg[13]" in run and "f_avg[14]" not in run
    cont = in_continue("c", "n2", 10000, chamber=ch, species="N2", **IO)
    assert "fix             bg emit/face bg xhi yhi" in cont and "collide         vss gas gas.vss" in cont
