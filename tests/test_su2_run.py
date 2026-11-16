"""su2/run.py restart handling: su2.cfg edits and archiving (never runs SU2)."""
import pytest

from su2 import run as srun

CFG = "SOLVER= NAVIER_STOKES\nRESTART_SOL= YES\nITER= 15000\nMESH_FILENAME= mesh.su2\n"


def test_set_cfg_rewrites_keys_in_place(tmp_path):
    cfg = tmp_path / "su2.cfg"; cfg.write_text(CFG)
    srun.set_cfg(cfg, RESTART_SOL="NO", ITER=20000)
    assert cfg.read_text() == "SOLVER= NAVIER_STOKES\nRESTART_SOL= NO\nITER= 20000\nMESH_FILENAME= mesh.su2\n"
    with pytest.raises(SystemExit, match="NOT_A_KEY"):
        srun.set_cfg(cfg, NOT_A_KEY=1)


def test_archive_numbers_successive_copies(tmp_path):
    (tmp_path / "history.csv").write_text("a")
    (tmp_path / "flow.vtu").write_text("b")
    assert srun.archive(tmp_path).name == "1"
    (tmp_path / "history.csv").write_text("c")
    d = srun.archive(tmp_path)
    assert d.name == "2" and (d / "history.csv").read_text() == "c" and (d / "flow.vtu").is_file()
    assert (tmp_path / "archive/1/history.csv").read_text() == "a"


def test_restart_without_restart_file_stops(tmp_path):
    (tmp_path / "su2.cfg").write_text(CFG)
    with pytest.raises(SystemExit, match="restart_flow.dat"):
        srun.run(tmp_path, nprocs=1, restart=True)
