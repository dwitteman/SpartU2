"""su2/setup.py convergence settings: mass-flow Cauchy tolerance scaled with the case."""
import copy
import re
from pathlib import Path

import pytest

from lib.common import load_case
from su2.setup import ideal_massflow, massflow_cauchy_eps, su2_config

ROOT = Path(__file__).resolve().parents[1]
N2_4KPA = ROOT / "examples" / "cd_nozzle_n2.toml"


def case(p0):
    c = copy.deepcopy(load_case(N2_4KPA))
    c["gas"]["plen_pres_pa"] = p0
    return c


def test_ideal_massflow_n2():
    # 0.6847 p0 A* / sqrt(R T0) for gamma = 1.4: N2, 0.5 mm throat, 4225 Pa, 300.15 K
    assert ideal_massflow(case(4225)) == pytest.approx(1.903e-6, rel=1e-3)


def test_tolerance_is_relative_to_the_flow():
    lo, hi = case(10.0), case(2e5)
    assert massflow_cauchy_eps(hi) / massflow_cauchy_eps(lo) == pytest.approx(2e4)
    hi["su2"]["conv_massflow_rel"] = 1e-5
    assert massflow_cauchy_eps(hi) == pytest.approx(1e-5 * ideal_massflow(hi) / 0.1)


def test_config_uses_the_field_name_su2_recognises():
    txt = su2_config(case(4225), "x.toml")
    conv = dict(re.findall(r"^(CONV_\w+)= (.+)$", txt, re.M))
    assert conv["CONV_FIELD"] == "SURFACE_MASSFLOW"  # "Avg_Massflow" is only the CSV header: ignored
    assert "CONV_RESIDUAL_MINVAL" not in conv  # only applies to residual fields
    assert float(conv["CONV_CAUCHY_EPS"]) == pytest.approx(massflow_cauchy_eps(case(4225)), rel=1e-3)
