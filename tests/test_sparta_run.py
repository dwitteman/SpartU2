"""sparta/run.py memory guards, tested with small Python child processes (never SPARTA)."""
import io
import sys

import pytest

from sparta import run as srun

ALLOC = "import time; x = bytearray({mb} * 2**20); x[::4096] = b'1' * len(x[::4096]); time.sleep({s})"


def test_watchdog_kills_process_over_the_limit():
    cmd = [sys.executable, "-c", ALLOC.format(mb=200, s=10)]
    rc, peak, killed = srun.run_watched(cmd, ".", io.StringIO(), mem_limit=100e6, poll_s=0.1)
    assert killed and rc != 0
    assert peak > 100e6


def test_watchdog_leaves_small_process_alone():
    log = io.StringIO()
    cmd = [sys.executable, "-c", ALLOC.format(mb=10, s=0.3) + "; print('done')"]
    rc, peak, killed = srun.run_watched(cmd, ".", log, mem_limit=500e6, poll_s=0.05)
    assert rc == 0 and not killed
    assert "done" in log.getvalue()


def test_particle_estimate_is_read_from_setup_log(tmp_path):
    assert srun.setup_particles(tmp_path) is None
    (tmp_path / "setup.log").write_text("fnum 9.25e+16\n  steady-state particles ~ 5e+06 (~1 GB at 200 B each)\n")
    assert srun.setup_particles(tmp_path) == pytest.approx(5e6)


def test_refuses_to_start_when_estimate_exceeds_limit(tmp_path):
    (tmp_path / "in.sparta").write_text("# not run\n")
    (tmp_path / "case.toml").write_text("[sparta]\nnprocs = 1\n")
    (tmp_path / "setup.log").write_text("  steady-state particles ~ 1e+09\n")  # ~200 GB
    with pytest.raises(SystemExit, match="exceeds the limit"):
        srun.run(tmp_path, max_mem_gb=1.0)


def test_latest_restart_and_restart_needs_a_file(tmp_path):
    assert srun.latest_restart(tmp_path) is None
    for st in (5000, 10000, 900):
        (tmp_path / f"restart.{st}.sparta").write_text("")
    (tmp_path / "restart.final.sparta").write_text("")  # not numeric: ignored
    assert srun.latest_restart(tmp_path) == 10000
    empty = tmp_path / "empty"; empty.mkdir()
    (empty / "case.toml").write_text("[sparta]\nnprocs = 1\n")
    with pytest.raises(SystemExit, match="nothing to continue"):
        srun.run(empty, restart=True)
    (tmp_path / "case.toml").write_text("[sparta]\nnprocs = 1\n")
    with pytest.raises(SystemExit, match="not after the latest restart"):
        srun.run(tmp_path, restart=True, target=10000)
