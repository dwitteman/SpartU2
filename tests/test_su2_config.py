"""su2/setup.py: ideal mass flow and the convergence settings in su2.cfg."""
import copy
import re
from pathlib import Path

import pytest

from lib.common import load_case
from su2.setup import ideal_massflow, su2_config

ROOT = Path(__file__).resolve().parents[1]
N2_4KPA = ROOT / "examples" / "cd_nozzle_n2.toml"


def case(p0):
    c = copy.deepcopy(load_case(N2_4KPA))
    c["gas"]["plen_pres_pa"] = p0
    return c


def test_ideal_massflow_n2():
    # 0.6847 p0 A* / sqrt(R T0) for gamma = 1.4: N2, 0.5 mm throat, 4225 Pa, 300.15 K
    assert ideal_massflow(case(4225)) == pytest.approx(1.903e-6, rel=1e-3)


def test_config_converges_on_the_density_residual():
    txt = su2_config(case(4225), "x.toml")
    conv = dict(re.findall(r"^(CONV_\w+)= (.+)$", txt, re.M))
    assert conv["CONV_FIELD"] == "RMS_DENSITY"  # SU2's name; the CSV header "rms[Rho]" is ignored
    assert float(conv["CONV_RESIDUAL_MINVAL"]) == -12  # log10 of the residual, in SI units here
    assert int(conv["CONV_STARTITER"]) > 0
    assert not any(k.startswith("CONV_CAUCHY") for k in conv)  # only for coefficient fields
