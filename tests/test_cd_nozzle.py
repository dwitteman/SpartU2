"""Converging-diverging nozzle geometry (su2/geom/cd_nozzle.py): wall profile and gmsh mesh."""
import copy
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from su2.geom import capillary, cd_nozzle
from su2.mesh_quality import read_su2

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


# ---- chamber lip-line relaxation ---------------------------------------------------------

def relaxed(x=(0.5, 0.75, 1.0), cells=(1.0, 10.0, 50.0)):
    c = case()
    c["su2"].update(lip_relax_x=list(x), lip_relax_cell_um=list(cells))
    return c


def test_chamber_stations_snap_to_axial_nodes():
    st = cd_nozzle.chamber_stations(relaxed(), 0.01, 1e-6, 40)
    assert [s[2] for s in st] == pytest.approx([1e-6, 1e-6, 1e-5, 5e-5])
    assert st[0][:2] == (0, 0.0) and st[-1][:2] == (40, 0.01)
    g = cd_nozzle.growth_rate(0.01, 40, 1e-6)
    for i, x, _ in st:  # each station is a node of the unsplit geometric distribution
        assert x == pytest.approx(1e-6 * (g**i - 1) / (g - 1), rel=1e-6)
    # no keys: one block, first_cell everywhere
    assert cd_nozzle.chamber_stations(case(), 0.01, 1e-6, 40) == [(0, 0.0, 1e-6), (40, 0.01, 1e-6)]
    # last station short of the outlet: the cell is held constant to the outlet
    st = cd_nozzle.chamber_stations(relaxed((0.5, 0.75), (1.0, 10.0)), 0.01, 1e-6, 40)
    assert st[-1][1:] == pytest.approx((0.01, 1e-5))


@pytest.mark.parametrize("x, cells", [((0.5, 0.75), (1.0,)), ((0.75, 0.5), (1.0, 10.0)),
                                      ((0.5, 1.5), (1.0, 10.0)), ((0.5, 0.75), (1.0, 0.0)),
                                      ((0.9, 0.9001), (1.0, 10.0))])
def test_chamber_stations_reject_bad_keys(x, cells):
    with pytest.raises(SystemExit):
        cd_nozzle.chamber_stations(relaxed(x, cells), 0.01, 1e-6, 40)


def test_relaxed_mesh(tmp_path):
    pts, quads, marks = mesh(tmp_path, cd_nozzle, relaxed())
    (tmp_path / "base").mkdir()
    base = mesh(tmp_path / "base", cd_nozzle, case())
    # same cells and boundary faces as the unrelaxed mesh
    assert len(quads) == len(base[1])
    assert {k: len(v) for k, v in marks.items()} == {k: len(v) for k, v in base[2].items()}
    q = pts[quads]
    area = 0.5 * np.sum(q[..., 0] * np.roll(q[..., 1], -1, 1) - np.roll(q[..., 0], -1, 1) * q[..., 1], 1)
    assert np.all(area > 0) or np.all(area < 0)

    # radial cell at the lip line: first_cell_um at the exit plane, the last value at the outlet
    R_e = 1e-3

    def lip_cell(x0):
        r = np.sort(pts[np.abs(pts[:, 0] - x0) < 1e-12, 1])
        k = np.argmin(np.abs(r - R_e))
        return r[k + 1] - r[k], r[k] - r[k - 1]
    assert lip_cell(0.0)[0] == pytest.approx(1e-6, rel=1e-3)
    x_out = pts[np.unique(marks["outlet"].ravel()), 0].max()
    assert lip_cell(x_out) == pytest.approx((5e-5, 5e-5), rel=1e-3)


def test_exit_cell_sets_the_axial_cell_at_the_exit_plane(tmp_path):
    c = case()
    c["su2"]["exit_cell_um"] = 10.0
    pts, quads, marks = mesh(tmp_path, cd_nozzle, c)
    ax = np.sort(pts[np.unique(marks["axis"].ravel()), 0])
    k = np.argmin(np.abs(ax))  # the exit plane on the axis
    assert ax[k + 1] - ax[k] == pytest.approx(10e-6, rel=1e-3)  # chamber side
    assert ax[k] - ax[k - 1] == pytest.approx(10e-6, rel=1e-3)  # nozzle side
    r_in = np.sort(pts[np.unique(marks["inlet"].ravel()), 1])
    assert r_in[-1] - r_in[-2] == pytest.approx(1e-6, rel=1e-3)  # wall cell still first_cell_um
