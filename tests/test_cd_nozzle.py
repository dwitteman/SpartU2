"""Converging-diverging nozzle geometry (su2/geom/cd_nozzle.py): wall profile and gmsh mesh."""
import copy
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from su2.geom import capillary, cd_nozzle

ROOT = Path(__file__).resolve().parents[1]
CASE = {
    "geometry": dict(noz_type="cd_nozzle", throat_diam_mm=0.5, inlet_diam_mm=3.0, exit_diam_mm=2.0,
                     conv_half_angle_deg=30, div_half_angle_deg=15, throat_rc_D=1.0,
                     throat_length_mm=0.0, dim=2, symmetry="axial",
                     chamb_length_cm=1.0, chamb_height_cm=0.5),
    "su2": dict(first_cell_um=1.0, throat_cell_um=10.0, n_radial_tube=20, n_axial_conv=40,
                n_axial_div=60, n_axial_ext=40, n_radial_ext=30),
}


def case(**geometry):
    c = copy.deepcopy(CASE)
    c["geometry"].update(geometry)
    return c


# ---- wall profile ---------------------------------------------------------------------

@pytest.mark.parametrize("L_t", [0.0, 0.3])
def test_profile_end_points_and_throat(L_t):
    p = cd_nozzle.profile(case(throat_length_mm=L_t))
    assert p.r(p.x_in) == pytest.approx(1.5e-3, rel=1e-12)
    assert p.r(0.0) == pytest.approx(1.0e-3, rel=1e-12)
    x = np.linspace(p.x_in, 0, 20001)
    r = p.r(x)
    assert r.min() == pytest.approx(0.25e-3, rel=1e-6)  # sampled; exact at the throat below
    assert p.r(p.x_tu) == pytest.approx(0.25e-3, rel=1e-12)
    assert p.x_td - p.x_tu == pytest.approx(L_t * 1e-3)
    assert np.all(r[(x >= p.x_tu) & (x <= p.x_td)] == pytest.approx(0.25e-3))
    assert p.area_ratio == pytest.approx(16.0)


def test_profile_is_smooth_with_cone_angles():
    # value and slope continuous at every junction; slopes equal the cone angles
    p = cd_nozzle.profile(case(throat_length_mm=0.2))
    eps = 1e-9
    slope = lambda x: (p.r(x + eps) - p.r(x - eps)) / (2 * eps)  # noqa: E731
    for xj in (p.x_c, p.x_tu, p.x_td, p.x_d):
        assert p.r(xj - 1e-12) == pytest.approx(p.r(xj + 1e-12), abs=1e-14 + 2e-12)  # slope x gap
        assert slope(xj - 10 * eps) == pytest.approx(slope(xj + 10 * eps), abs=1e-3)
    assert slope((p.x_in + p.x_c) / 2) == pytest.approx(-math.tan(math.radians(30)), rel=1e-6)
    assert slope(p.x_d / 2) == pytest.approx(math.tan(math.radians(15)), rel=1e-6)


def test_profile_monotonic_either_side_of_throat():
    p = cd_nozzle.profile(CASE)
    x = np.linspace(p.x_in, 0, 5001)
    r = p.r(x)
    assert np.all(np.diff(r[x <= p.x_tu]) <= 0)
    assert np.all(np.diff(r[x >= p.x_td]) >= 0)


@pytest.mark.parametrize("bad, match", [
    (dict(inlet_diam_mm=0.6), "inlet radius"),
    (dict(exit_diam_mm=0.52), "exit radius"),
    (dict(conv_half_angle_deg=90), "conv_half_angle_deg"),
    (dict(div_half_angle_deg=0), "div_half_angle_deg"),
])
def test_profile_rejects_impossible_geometry(bad, match):
    with pytest.raises(SystemExit, match=match):
        cd_nozzle.profile(case(**bad))


# ---- gmsh mesh ------------------------------------------------------------------------

def read_su2(path):
    """(points, quads, {marker: line elements}) from an SU2 mesh file."""
    lines = Path(path).read_text().splitlines()
    pts, quads, marks, i = None, None, {}, 0
    while i < len(lines):
        s = lines[i].strip()
        if s.startswith("NELEM="):
            n = int(s.split("=")[1])
            quads = np.array([[int(v) for v in l.split()[1:5]] for l in lines[i + 1:i + 1 + n]])
            i += n
        elif s.startswith("NPOIN="):
            n = int(s.split("=")[1].split()[0])
            pts = np.array([[float(v) for v in l.split()[:2]] for l in lines[i + 1:i + 1 + n]])
            i += n
        elif s.startswith("MARKER_TAG="):
            tag, n = s.split("=")[1].strip(), int(lines[i + 1].split("=")[1])
            marks[tag] = np.array([[int(v) for v in l.split()[1:3]] for l in lines[i + 2:i + 2 + n]])
            i += n + 1
        i += 1
    return pts, quads, marks


def mesh(tmp_path, gen, c):
    if not shutil.which("gmsh"):
        pytest.skip("gmsh not on PATH")
    geo, su2 = tmp_path / "mesh.geo", tmp_path / "mesh.su2"
    geo.write_text(gen.geo(c, source="test"))
    subprocess.run(["gmsh", str(geo), "-2", "-format", "su2", "-o", str(su2)],
                   check=True, capture_output=True)
    return read_su2(su2)


@pytest.mark.parametrize("L_t", [0.0, 0.3])
def test_cd_mesh(tmp_path, L_t):
    c = case(throat_length_mm=L_t)
    p = cd_nozzle.profile(c)
    pts, quads, marks = mesh(tmp_path, cd_nozzle, c)
    s = c["su2"]
    n_throat = round(L_t * 1e-3 / (s["throat_cell_um"] * 1e-6)) if L_t else 0
    nx_noz = s["n_axial_conv"] + n_throat + s["n_axial_div"]
    expect = s["n_radial_tube"] * (nx_noz + s["n_axial_ext"]) + s["n_radial_ext"] * s["n_axial_ext"]
    assert len(quads) == expect
    assert {k: len(v) for k, v in marks.items()} == {
        "inlet": s["n_radial_tube"], "wall": nx_noz, "faceplate": s["n_radial_ext"],
        "axis": nx_noz + s["n_axial_ext"],
        "outlet": s["n_radial_tube"] + s["n_radial_ext"] + s["n_axial_ext"]}

    # every quad valid: same orientation, not degenerate, no extreme skew
    q = pts[quads]
    area = 0.5 * np.sum(q[..., 0] * np.roll(q[..., 1], -1, 1) - np.roll(q[..., 0], -1, 1) * q[..., 1], 1)
    assert np.all(area > 0) or np.all(area < 0)
    e0, e1 = q - np.roll(q, 1, 1), np.roll(q, -1, 1) - q
    ang = np.degrees(np.arccos(np.clip(np.sum(-e0 * e1, 2)
                                       / (np.linalg.norm(e0, axis=2) * np.linalg.norm(e1, axis=2)), -1, 1)))
    assert 30 < ang.min() and ang.max() < 150

    # wall nodes on the exact profile (spline error well below the first cell)
    w = np.unique(marks["wall"].ravel())
    assert np.abs(pts[w, 1] - p.r(pts[w, 0])).max() < 0.05 * s["first_cell_um"] * 1e-6
    assert pts[w, 0].min() == pytest.approx(p.x_in) and pts[w, 0].max() == pytest.approx(0, abs=1e-15)

    # first cell at the wall on the inlet line is first_cell_um
    r_in = np.sort(pts[np.unique(marks["inlet"].ravel()), 1])
    assert r_in[-1] - r_in[-2] == pytest.approx(s["first_cell_um"] * 1e-6, rel=1e-3)


def test_capillary_mesh_still_builds(tmp_path):
    c = {"geometry": dict(throat_diam_mm=0.4, throat_length_mm=5, chamb_length_cm=1, chamb_height_cm=0.5),
         "su2": dict(first_cell_um=1.0, n_radial_tube=10, n_axial_tube=40, n_axial_ext=30, n_radial_ext=20)}
    pts, quads, marks = mesh(tmp_path, capillary, c)
    assert len(quads) == 10 * 40 + 10 * 30 + 20 * 30
    assert set(marks) == {"inlet", "wall", "faceplate", "axis", "outlet"}
