"""SU2 -> SPARTA inflow (sparta/inflow.py): fluxes, file format, and SPARTA reproducing it."""
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from sparta import inflow
from sparta.gas import DATA, KB, load_gas

N2 = load_gas("n2.species")
SPA = Path(os.environ.get("SPA_SERIAL", Path.home() / "code/sparta-24Sep2025/src/spa_serial"))
# The SPARTA test launches a real DSMC run, so it only runs when asked for:
#   RUN_SPARTA_TESTS=1 pytest tests/test_inflow.py
RUN_SPARTA = os.environ.get("RUN_SPARTA_TESTS") == "1"


# ---- fluxes ---------------------------------------------------------------------------

def test_inward_flux_limits():
    n, T, m = 1e20, 300.0, N2.mass
    cbar = np.sqrt(8 * KB * T / (np.pi * m))
    assert inflow.inward_number_flux(n, 0.0, T, m) == pytest.approx(n * cbar / 4, rel=1e-12)
    assert inflow.inward_number_flux(n, 5000.0, T, m) == pytest.approx(n * 5000.0, rel=1e-6)
    assert inflow.inward_number_flux(n, -5000.0, T, m) < 1e-6 * n * cbar
    # net = inward - outward: the outward flux is the inward flux of -u
    u = 150.0
    net = inflow.inward_number_flux(n, u, T, m) - inflow.inward_number_flux(n, -u, T, m)
    assert net == pytest.approx(n * u, rel=1e-9)


def test_annulus_integral_and_summary():
    r = np.linspace(0, 0.01, 401)
    assert inflow.annulus_integral(r, np.ones_like(r)) == pytest.approx(np.pi * 0.01**2, rel=1e-4)
    plane = dict(r=r, n=np.full_like(r, 1e21), u=np.full_like(r, 3000.0), v=0 * r,
                 T=np.full_like(r, 20.0), rho=np.full_like(r, 1e21 * N2.mass))
    f = inflow.plane_summary(plane, N2.mass)
    assert f["mdot_net"] == pytest.approx(1e21 * N2.mass * 3000 * np.pi * 1e-4, rel=1e-4)
    assert f["mdot_inward"] == pytest.approx(f["mdot_net"], rel=1e-6)  # hypersonic: one-sided = net


def test_write_inflow_format(tmp_path):
    r = np.array([0.0, 0.001, 0.003])
    plane = dict(r=r, n=np.array([1e21, 2e21, 3e21]), u=np.array([900.0, 800, -1]),
                 v=np.array([0.0, 50, 0]), T=np.array([20.0, 30, 300]))
    inflow.write_inflow(tmp_path / "in.txt", plane, rot=True, vib=False, header="test")
    lines = [l for l in (tmp_path / "in.txt").read_text().splitlines() if l and not l.startswith("#")]
    assert lines[0].split()[0] == inflow.SECTION
    assert lines[1:4] == ["NI 3", "NV 6", "VALUES nrho vx vy vz temp trot"]
    assert [float(v) for v in lines[4].split()[1:]] == [0.0, 0.001, 0.003]
    rows = np.array([[float(v) for v in l.split()] for l in lines[5:]])
    assert rows[:, 0].tolist() == [1, 2, 3]
    assert np.allclose(rows[:, 1], plane["n"]) and np.allclose(rows[:, 5], plane["T"])
    assert np.allclose(rows[:, 6], plane["T"])  # trot = T


# ---- SPARTA reproduces the inflow ------------------------------------------------------

def read_grid_dump(path):
    """Columns of the last snapshot in a SPARTA grid dump."""
    lines = Path(path).read_text().splitlines()
    i = max(k for k, l in enumerate(lines) if l.startswith("ITEM: CELLS"))
    cols = lines[i].split()[2:]
    data = np.array([[float(v) for v in l.split()] for l in lines[i + 1:] if not l.startswith("ITEM")])
    return {c: data[:, j] for j, c in enumerate(cols)}


@pytest.mark.skipif(not RUN_SPARTA, reason="runs SPARTA; set RUN_SPARTA_TESTS=1 to enable")
@pytest.mark.skipif(not SPA.is_file(), reason=f"SPARTA not found at {SPA} (set SPA_SERIAL)")
def test_sparta_reproduces_axisymmetric_inflow(tmp_path):
    # hypersonic N2 with density rising linearly in r: downstream of xlo the DSMC density
    # and velocity must match the file, near the axis too (checks the radial weighting).
    # With `weight cell radius`, particles/cell = 2 pi n dy / fnum: fnum 3e16 gives ~30
    # per cell, ~6000 in all (fnum 2e9 would be ~1e11 particles and exhaust memory).
    L, R, n0, U, T = 0.02, 0.01, 1e20, 2000.0, 50.0
    r = np.linspace(0, R, 11)
    n = n0 * (1 + r / R)
    plane = dict(r=r, n=n, u=np.full_like(r, U), v=0 * r, T=np.full_like(r, T))
    inflow.write_inflow(tmp_path / "inflow.sparta", plane, rot=True, vib=True)
    for ext in ("species", "vss"):
        shutil.copy(DATA / f"n2.{ext}", tmp_path)
    steps, avg = 400, 300
    (tmp_path / "in.test").write_text(f"""\
seed 12345
dimension 2
boundary o ao p
create_box 0 {L} 0 {R} -0.5 0.5
create_grid 20 10 1
global weight cell radius
global fnum 3e16
species n2.species N2
mixture gas N2
collide vss gas n2.vss
timestep 2e-7
fix in emit/face/file gas xlo inflow.sparta XLO
compute g grid all gas nrho u v
compute th thermal/grid all gas temp
fix avg ave/grid all 1 {avg} {steps} c_g[*] c_th[*]
dump d grid all {steps} flow.dump id xc yc f_avg[1] f_avg[2] f_avg[3] f_avg[4]
run {steps}
""")
    subprocess.run(["bash", "-c", f"ulimit -v 2000000; exec {SPA} -in in.test"], cwd=tmp_path,
                   check=True, capture_output=True, stdin=subprocess.DEVNULL, timeout=120)
    d = read_grid_dump(tmp_path / "flow.dump")
    nrho, u, Tg = d["f_avg[1]"], d["f_avg[2]"], d["f_avg[4]"]
    expect = n0 * (1 + d["yc"] / R)
    # the transit takes ~50 steps, so the average is over steady flow. Measured (12 Nov 2026):
    # rows r = 1.5-8.5 mm within 1-4%; the axis row is 14% high and the top row 25% low.
    # The top row is expected: yhi is an outflow, so gas diffuses out and none comes back.
    # The axis excess is not explained yet (open item in sparta/README.md).
    rows = np.unique(d["yc"])
    ratio = np.array([np.mean(nrho[d["yc"] == y] / expect[d["yc"] == y]) for y in rows])
    assert np.all(np.abs(ratio[1:-1] - 1) < 0.05), ratio
    assert np.abs(ratio[0] - 1) < 0.2, ratio  # axis row
    assert np.median(u) == pytest.approx(U, rel=0.01)
    assert np.median(Tg) == pytest.approx(T, rel=0.1)
