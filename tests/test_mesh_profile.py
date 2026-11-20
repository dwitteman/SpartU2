"""su2/mesh.py (general mesher) and su2/geom/profile.py (wall from a table of points)."""
import copy
import shutil
import subprocess

import numpy as np
import pytest

from su2 import mesh
from su2.geom import capillary, cd_nozzle, profile
from su2.mesh_quality import read_su2

BASE = {
    "geometry": dict(noz_type="profile", throat_diam_mm=0.5, chamb_length_cm=1.0, chamb_height_cm=0.5),
    "su2": dict(first_cell_um=1.0, n_radial_tube=10, n_axial_conv=20, n_axial_div=20,
                n_axial_ext=20, n_radial_ext=15),
}


def case(tmp_path, rows, **geometry):
    f = tmp_path / "wall.csv"
    f.write_text("x_mm, r_mm  # header\n" + "\n".join(f"{x}, {r}" for x, r in rows) + "\n")
    c = copy.deepcopy(BASE)
    c["geometry"].update(profile_file=str(f), **geometry)
    return c


def build(tmp_path, c):
    if not shutil.which("gmsh"):
        pytest.skip("gmsh not on PATH")
    geo = tmp_path / "m.geo"
    geo.write_text(mesh.geo(c, mesh.provider(c).shape(c), "test"))
    subprocess.run(["gmsh", str(geo), "-2", "-format", "su2", "-o", str(tmp_path / "m.su2")],
                   check=True, capture_output=True)
    return read_su2(tmp_path / "m.su2")


def test_read_profile_shifts_to_the_exit_and_converts_units(tmp_path):
    f = tmp_path / "p.txt"
    f.write_text("# comment\nx r\n10 1.0\n12\t0.5\n15;0.75\n")
    x, r = profile.read_profile(f)
    assert x == pytest.approx([-5e-3, -3e-3, 0.0]) and r == pytest.approx([1e-3, 0.5e-3, 0.75e-3])
    x, _ = profile.read_profile(f, "m")
    assert x == pytest.approx([-5, -3, 0])


@pytest.mark.parametrize("text", ["1 1\n", "0 1\n0 2\n", "0 1\n1 -1\n", "0 1\nbad line\n"])
def test_read_profile_rejects_bad_files(tmp_path, text):
    f = tmp_path / "p.txt"
    f.write_text(text)
    with pytest.raises(SystemExit):
        profile.read_profile(f)


def test_sections_for_each_kind_of_wall(tmp_path):
    cd = profile.shape(case(tmp_path, [(0, 1.0), (2, 0.25), (5, 1.0)]))
    assert [(s.n, s.fine) for s in cd.sections] == [(20, "end"), (10, "start"), (10, "end")]
    assert [s.x0 for s in cd.sections] + [cd.sections[-1].x1] == pytest.approx([-5e-3, -3e-3, -1.5e-3, 0])
    conv = profile.shape(case(tmp_path, [(0, 1.0), (5, 0.25)]))
    assert [(s.n, s.fine) for s in conv.sections] == [(20, "end")]
    tube = profile.shape(case(tmp_path, [(0, 0.25), (5, 0.25)]))
    assert [(s.n, s.fine) for s in tube.sections] == [(20, "end")]  # throat = the exit
    div = profile.shape(case(tmp_path, [(0, 0.25), (5, 1.0)]))
    assert [s.fine for s in div.sections] == ["start", "end"]
    # the wall passes through the points without overshoot
    assert cd.r(np.array([-5e-3, -3e-3, 0.0])) == pytest.approx([1e-3, 0.25e-3, 1e-3])
    xs = np.linspace(-5e-3, 0, 501)
    assert cd.r(xs).min() >= 0.25e-3 - 1e-15 and cd.r(xs).max() <= 1e-3 + 1e-15


def test_profile_mesh_builds_with_the_standard_markers(tmp_path):
    c = case(tmp_path, [(0, 1.0), (1, 0.6), (2, 0.25), (3, 0.5), (5, 1.0)])
    pts, quads, marks = build(tmp_path, c)
    s = c["su2"]
    n_noz = s["n_axial_conv"] + s["n_axial_div"]
    assert len(quads) == s["n_radial_tube"] * (n_noz + s["n_axial_ext"]) + s["n_radial_ext"] * s["n_axial_ext"]
    assert set(marks) == {"inlet", "wall", "faceplate", "axis", "outlet"}
    sh = profile.shape(c)
    w = np.unique(marks["wall"].ravel())
    assert np.abs(pts[w, 1] - sh.r(pts[w, 0])).max() < 0.05 * s["first_cell_um"] * 1e-6


def test_profile_reproduces_an_analytic_nozzle(tmp_path):
    # the cd_nozzle example wall, sampled into a file: the same throat and exit, close wall
    c0 = copy.deepcopy(BASE)
    c0["geometry"].update(noz_type="cd_nozzle", inlet_diam_mm=3.0, exit_diam_mm=2.0, conv_half_angle_deg=30,
                          div_half_angle_deg=15, throat_rc_D=1.0, throat_length_mm=0.0)
    p = cd_nozzle.profile(c0)
    x = np.linspace(p.x_in, 0, 300)
    c = case(tmp_path, list(zip(x * 1e3, p.r(x) * 1e3)))
    sh = profile.shape(c)
    xs = np.linspace(p.x_in, 0, 2000)
    assert np.abs(sh.r(xs) - p.r(xs)).max() < 1e-3 * p.R_t
    assert sh.sections[0].x1 == pytest.approx(p.x_td, abs=(x[1] - x[0]))  # throat found


def test_shapes_must_be_contiguous_and_end_at_the_exit():
    bad = mesh.Shape(r=lambda x: 0.001 + 0 * np.asarray(x),
                     sections=[mesh.Section(-2e-3, -1e-3, 5), mesh.Section(-0.9e-3, 0.0, 5)])
    with pytest.raises(SystemExit):
        mesh.check_shape(bad)
    with pytest.raises(SystemExit):
        mesh.check_shape(mesh.Shape(r=lambda x: 0.001 + 0 * np.asarray(x),
                                    sections=[mesh.Section(-2e-3, -1e-3, 5)]))


def test_capillary_takes_the_chamber_options_too(tmp_path):
    c = {"geometry": dict(noz_type="capillary", throat_diam_mm=0.4, throat_length_mm=5,
                          chamb_length_cm=1, chamb_height_cm=0.5),
         "su2": dict(first_cell_um=1.0, exit_cell_um=10.0, n_radial_tube=10, n_axial_tube=40,
                     n_axial_ext=30, n_radial_ext=20, lip_relax_x=[0.5, 1.0], lip_relax_cell_um=[1.0, 10.0])}
    if not shutil.which("gmsh"):
        pytest.skip("gmsh not on PATH")
    geo = tmp_path / "m.geo"
    geo.write_text(capillary.geo(c, "test"))
    subprocess.run(["gmsh", str(geo), "-2", "-format", "su2", "-o", str(tmp_path / "m.su2")],
                   check=True, capture_output=True)
    pts, quads, marks = read_su2(tmp_path / "m.su2")
    assert len(quads) == 10 * 40 + 10 * 30 + 20 * 30
    ax = np.sort(pts[np.unique(marks["axis"].ravel()), 0])
    k = np.argmin(np.abs(ax))
    assert ax[k + 1] - ax[k] == pytest.approx(10e-6, rel=1e-3)  # exit_cell_um on the chamber side
