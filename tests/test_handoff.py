"""handoff/handoff.py: coupling-plane choice, handoff.toml round trip, and the bash driver."""
import subprocess
from pathlib import Path

import pytest

from handoff.handoff import choose_plane, read_handoff

D = 0.5e-3
ROOT = Path(__file__).resolve().parents[1]


def test_auto_plane_is_margin_upstream_of_breakdown():
    assert choose_plane("auto", D, x_break=6 * D, margin_D=1.0) == pytest.approx(5 * D)
    assert choose_plane("auto", D, x_break=6 * D, margin_D=2.5) == pytest.approx(3.5 * D)


def test_forced_plane_ignores_breakdown():
    assert choose_plane(10, D, x_break=6 * D) == pytest.approx(10 * D)
    assert choose_plane(4.0, D, x_break=None) == pytest.approx(4 * D)


def test_auto_without_breakdown_or_room_stops():
    with pytest.raises(SystemExit, match="no breakdown point"):
        choose_plane("auto", D, x_break=None)
    with pytest.raises(SystemExit, match="within couple_margin_D"):
        choose_plane("auto", D, x_break=0.5 * D, margin_D=1.0)
    with pytest.raises(SystemExit, match="outside the chamber"):
        choose_plane(100, D, x_break=None, x_max=0.01)


def test_read_handoff(tmp_path):
    with pytest.raises(SystemExit, match="run handoff/handoff.py first"):
        read_handoff(tmp_path)
    (tmp_path / "handoff").mkdir()
    (tmp_path / "handoff" / "handoff.toml").write_text(
        '[handoff]\nsu2_vtu = "/x/flow.vtu"\nmode = "auto"\nx_couple_m = 0.0025\n'
        "mdot_net = 7.8e-05       # kg/s\n")
    h = read_handoff(tmp_path)
    assert h["x_couple_m"] == 0.0025 and h["mode"] == "auto" and h["mdot_net"] == 7.8e-5


def test_run_case_script_parses_and_needs_a_case():
    script = ROOT / "run_case.sh"
    assert subprocess.run(["bash", "-n", script]).returncode == 0
    r = subprocess.run(["bash", script], cwd=ROOT, capture_output=True, text=True,
                       env={"PATH": "/usr/bin:/bin"})
    assert r.returncode != 0 and "set CASE" in r.stderr
