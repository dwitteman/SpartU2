"""Mesh sequencing: su2/setup.py --coarsen, and run_case.sh COARSEN / su2_refine (no solver run)."""
import copy
import os
import subprocess
import sys
from pathlib import Path

import pytest

from lib.common import load_case
from su2.setup import COUNT_KEYS, coarsen_case, run_dir_name

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "examples" / "cd_nozzle_n2.toml"


def test_coarsen_scales_counts_and_sizes():
    c = copy.deepcopy(load_case(CASE))
    c["su2"]["lip_relax_cell_um"] = [1.0, 10.0]
    k = coarsen_case(c, 4)
    for key in COUNT_KEYS:
        if key in c["su2"]:
            assert k["su2"][key] == max(2, round(c["su2"][key] / 4))
    assert k["su2"]["first_cell_um"] == 4 * c["su2"]["first_cell_um"]
    assert k["su2"]["lip_relax_cell_um"] == [4.0, 40.0]
    assert c["su2"]["lip_relax_cell_um"] == [1.0, 10.0]  # the original is untouched
    assert coarsen_case(c, 1)["su2"] == c["su2"]


def test_coarsen_keeps_at_least_two_cells_and_scales_the_default_throat_cell():
    c = copy.deepcopy(load_case(CASE))
    c["su2"]["n_radial_tube"] = 3
    c["su2"].pop("throat_cell_um", None)
    k = coarsen_case(c, 8)
    assert k["su2"]["n_radial_tube"] == 2
    R_t_um = c["geometry"]["throat_diam_mm"] * 1e3 / 2
    assert k["su2"]["throat_cell_um"] == pytest.approx(8 * R_t_um / 20)  # cd_nozzle default R_t/20


def test_coarsen_rejects_zero():
    with pytest.raises(SystemExit):
        coarsen_case(load_case(CASE), 0)


def test_run_dir_names():
    assert run_dir_name() == "su2" and run_dir_name(1) == "su2"
    assert run_dir_name(4) == "su2_coarse4"


def run_case(tmp_path, **env):
    e = {"PATH": os.environ["PATH"], "PY": sys.executable, "RUNS": str(tmp_path), "NP": "1",
         "CASE": str(CASE), **env}
    return subprocess.run(["bash", ROOT / "run_case.sh"], cwd=ROOT, capture_output=True, text=True, env=e)


def test_su2_refine_needs_from_and_a_coarse_restart(tmp_path):
    r = run_case(tmp_path, STAGES="su2_refine")
    assert r.returncode != 0 and "set FROM" in r.stderr
    r = run_case(tmp_path, STAGES="su2_refine", FROM="4")
    assert r.returncode != 0 and "su2_coarse4/restart_flow.dat" in r.stderr
