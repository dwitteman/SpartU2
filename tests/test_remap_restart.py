"""su2/remap_restart.py: SU2 binary restart read/write and interpolation onto a new mesh."""
import numpy as np
import pytest

from su2 import remap_restart as rr

NAMES = ["x", "y", "Density", "Momentum_x", "Momentum_y", "Energy"]


def linear(xy):
    x, y = xy.T
    return np.column_stack([1 + x, 2 * y, x - y, 3 + x + y])


def write_mesh(path, xy):
    lines = ["NDIME= 2", "NELEM= 0", f"NPOIN= {len(xy)}"]
    lines += [f"{x:.17g} {y:.17g} {i}" for i, (x, y) in enumerate(xy)]
    path.write_text("\n".join(lines) + "\n")


def test_round_trip(tmp_path):
    xy = np.random.default_rng(0).random((50, 2))
    data = np.column_stack([xy, linear(xy)])
    rr.write_restart(tmp_path / "r.dat", NAMES, data)
    names, back = rr.read_restart(tmp_path / "r.dat")
    assert names == NAMES and np.array_equal(back, data)
    assert (tmp_path / "r.dat").stat().st_size == 20 + 33 * 6 + 8 * data.size


def test_remap_is_exact_for_linear_fields(tmp_path):
    g = np.linspace(0, 1, 11)
    old = np.array([(x, y) for x in g for y in g])
    new = np.random.default_rng(1).random((200, 2))
    new[:4] = [(0, 0), (1, 1), (0, 1), (1, 0)]  # corners, on the hull
    rr.write_restart(tmp_path / "old.dat", NAMES, np.column_stack([old, linear(old)]))
    write_mesh(tmp_path / "mesh.su2", new)
    rr.remap(tmp_path / "old.dat", tmp_path / "mesh.su2", tmp_path / "new.dat")
    names, out = rr.read_restart(tmp_path / "new.dat")
    assert names == NAMES
    assert np.array_equal(out[:, :2], new)  # new node order and coordinates
    assert out[:, 2:] == pytest.approx(linear(new), abs=1e-12)


def test_rejects_non_binary(tmp_path):
    (tmp_path / "r.csv").write_text('"PointID","x","y"\n')
    with pytest.raises(SystemExit):
        rr.read_restart(tmp_path / "r.csv")
