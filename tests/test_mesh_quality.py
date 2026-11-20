"""su2/mesh_quality.py: per-cell metrics and the mesh_quality.txt report."""
import numpy as np
import pytest

from su2 import mesh_quality as mq


def grid(xs, ys):
    """Structured quad mesh on the tensor grid xs x ys: (points, quads)."""
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    pts = np.column_stack([X.ravel(), Y.ravel()])
    nx, ny = len(xs), len(ys)
    idx = lambda i, j: i * ny + j  # noqa: E731
    quads = np.array([[idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)]
                      for i in range(nx - 1) for j in range(ny - 1)])
    return pts, quads


def write_su2(path, pts, quads, marks):
    lines = ["NDIME= 2", f"NELEM= {len(quads)}"]
    lines += [f"9 {' '.join(map(str, q))} {k}" for k, q in enumerate(quads)]
    lines += [f"NPOIN= {len(pts)}"] + [f"{x:.17g} {y:.17g} {k}" for k, (x, y) in enumerate(pts)]
    lines += [f"NMARK= {len(marks)}"]
    for tag, el in marks.items():
        lines += [f"MARKER_TAG= {tag}", f"MARKER_ELEMS= {len(el)}"] + [f"3 {a} {b}" for a, b in el]
    path.write_text("\n".join(lines) + "\n")


def test_square_cells_are_perfect():
    m = mq.cell_metrics(*grid(np.linspace(0, 1, 6), np.linspace(0, 1, 6)))
    assert m["aspect"] == pytest.approx(1)
    assert m["angle_min"] == pytest.approx(90) and m["angle_max"] == pytest.approx(90)
    assert m["jump"] == pytest.approx(1)
    assert m["area"].sum() == pytest.approx(1)


def test_stretching_and_jump():
    xs = np.concatenate([np.arange(0, 1.0, 0.1), np.arange(1.0, 2.01, 0.2)])  # dx 0.1 then 0.2
    m = mq.cell_metrics(*grid(xs, [0, 0.01, 0.02]))
    assert m["aspect"].max() == pytest.approx(20)  # 0.2 x 0.01
    assert m["aspect"].min() == pytest.approx(10)
    assert m["jump"].max() == pytest.approx(2)  # the cells either side of x = 1
    assert np.sum(m["jump"] > 1.5) == 4


def test_skewed_cell_angles():
    pts = np.array([[0, 0], [1, 0], [1.5, 1], [0.5, 1]], float)  # parallelogram
    m = mq.cell_metrics(pts, np.array([[0, 1, 2, 3]]))
    a = np.degrees(np.arctan2(1, 0.5))
    assert m["angle_min"][0] == pytest.approx(a) and m["angle_max"][0] == pytest.approx(180 - a)


def test_report_regions_and_checks(tmp_path):
    # "nozzle" x in [-1, 0], chamber x in [0, 10]; wall along r = 1 ending at the lip (0, 1);
    # thin cells (aspect 2000) in the chamber trip the aspect check only
    xs = np.concatenate([np.linspace(-1, 0, 11), np.linspace(0, 10, 11)[1:]])
    ys = np.concatenate([np.linspace(0, 1, 6), [1 + 5e-4], np.linspace(1, 3, 6)[1:]])
    pts, quads = grid(xs, ys)
    wall = [(i * len(ys) + 5, (i + 1) * len(ys) + 5) for i in range(10)]
    write_su2(tmp_path / "mesh.su2", pts, quads, {"wall": np.array(wall)})
    out, flagged = mq.write(tmp_path / "mesh.su2")
    text = out.read_text()
    assert out.name == "mesh_quality.txt"
    assert "lip at (0, 1e+03) mm" in text
    for region in ("nozzle (x < 0 mm)", "chamber, r < lip", "chamber, r > lip"):
        assert region in text
    assert "[!!] aspect ratio" in text and "[ok] corner angles" in text
    assert len(flagged) == 2  # aspect ratio, and the size jump into the thin row
    assert any("aspect ratio" in f for f in flagged) and any("size jump" in f for f in flagged)


def test_rejects_triangles(tmp_path):
    (tmp_path / "m.su2").write_text("NDIME= 2\nNELEM= 1\n5 0 1 2 0\nNPOIN= 3\n0 0 0\n1 0 1\n0 1 2\n")
    with pytest.raises(SystemExit):
        mq.read_su2(tmp_path / "m.su2")
