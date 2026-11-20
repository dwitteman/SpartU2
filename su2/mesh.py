#!/usr/bin/env python3
"""Axisymmetric structured mesh for any nozzle wall r(x) exhausting into a cylindrical chamber.

Usage: python su2/mesh.py CASE.toml [--out DIR] [--coarsen N]
Writes DIR/mesh.geo, mesh.su2 (gmsh) and mesh_quality.txt; DIR defaults to runs/<name>/mesh/.
su2/setup.py calls the same code for the su2 run directory.

The geometry comes from a provider in su2/geom/<noz_type>.py, which only describes the
shape: a Shape with the wall radius r(x) (exit plane at x = 0, inlet at x_in < 0) and the
axial sections of the nozzle, each with its cell count and where its cells cluster. This
module turns any Shape into blocks:

  +------------------------------- top (outlet) ------------------+
  |                                                               |
  faceplate                 chamber block B                    outlet
  |                                                               |
 _+ lip ----------------------------------------------------------+
  /wall\\___    _____/ wall                                        |
 |  section | section | ... |         chamber block A            outlet
 +----------+---------+-----+-------------------------------------+
 inlet               axis                       axis

  - nozzle: one block per section, n_radial_tube cells from the wall (first_cell_um) to the axis;
  - chamber: blocks A (axis .. lip radius) and B (lip .. chamber radius), split along x at the
    lip_relax stations (chamber_stations); exit_cell_um is the axial cell either side of the
    exit plane.
Markers: inlet, wall, faceplate, axis, outlet.

A provider is a module with `shape(case) -> Shape`. The wall must be single-valued, r(x) > 0
from the inlet to the exit; anything that can be sampled that way (an analytic profile, a
table of points, a curve taken from a CAD file) can be meshed.
"""
import argparse
import importlib
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

if __package__ in (None, ""):  # allow `python su2/mesh.py` as well as `-m su2.mesh`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.common import die, fmt, get, growth_rate, load_case, log

N_SPLINE = 200  # interior points per curved wall section


@dataclass
class Section:
    """One nozzle block between two x stations."""
    x0: float
    x1: float
    n: int                 # axial cells
    fine: str = "none"     # "start", "end" or "none": the end the cells cluster toward
    h: float = None        # first axial cell at the fine end [m]; None with "exit" = exit_cell_um
    straight: bool = False  # the wall is a straight line here (no spline)


@dataclass
class Shape:
    """A nozzle described for the mesher."""
    r: Callable            # wall radius [m] at x [m]; exit plane at x = 0
    sections: list         # Sections from the inlet to the exit (x = 0), contiguous
    name: str = ""         # for the mesh.geo header
    notes: list = field(default_factory=list)  # extra header lines

    @property
    def x_in(self):
        return self.sections[0].x0

    @property
    def R_exit(self):
        return float(self.r(0.0))


def provider(case):
    """The su2/geom module for [geometry] noz_type."""
    noz = get(case, "geometry", "noz_type")
    try:
        return importlib.import_module(f"su2.geom.{noz}")
    except ModuleNotFoundError:
        die(f"no geometry provider for noz_type={noz} (su2/geom/{noz}.py)")


def check_shape(s: Shape):
    if not s.sections:
        die("geometry provider returned no sections")
    for a, b in zip(s.sections, s.sections[1:]):
        if a.x1 != b.x0:
            die(f"geometry sections are not contiguous at x = {a.x1}")
    if s.sections[-1].x1 != 0.0:
        die("the last geometry section must end at the exit plane, x = 0")
    for sec in s.sections:
        if not sec.x1 > sec.x0 or sec.n < 1:
            die(f"bad geometry section {sec}")
    xs = np.linspace(s.x_in, 0.0, 2001)
    if not np.all(np.asarray(s.r(xs)) > 0):
        die("the wall radius must be positive from the inlet to the exit")


class _Geo:
    """Minimal .geo writer with automatic entity numbering."""

    def __init__(self):
        self.lines, self.np, self.nc = [], 0, 0

    def point(self, x, y):
        self.np += 1
        self.lines.append(f"Point({self.np}) = {{{fmt(float(x))}, {fmt(float(y))}, 0}};")
        return self.np

    def line(self, a, b, comment=""):
        self.nc += 1
        self.lines.append(f"Line({self.nc}) = {{{a}, {b}}};" + (f"  // {comment}" if comment else ""))
        return self.nc

    def spline(self, pts, comment=""):
        self.nc += 1
        self.lines.append(f"Spline({self.nc}) = {{{', '.join(map(str, pts))}}};"
                          + (f"  // {comment}" if comment else ""))
        return self.nc

    def transfinite(self, curve, n, prog):
        self.lines.append(f"Transfinite Curve{{{curve}}} = {n + 1} Using Progression {prog:.8f};")

    def text(self):
        return "\n".join(self.lines) + "\n"


def chamber_stations(case: dict, Lx: float, h: float, nx_e: int, h_x: float = None) -> list:
    """Chamber x stations and the radial first cell at the lip line at each: [(node, x, cell)].

    [su2] lip_relax_x (fractions of the chamber length) and lip_relax_cell_um (cell size at each)
    coarsen the lip-line cell downstream, where 1 um cells stretched to mm lengths are needlessly
    anisotropic. Upstream of the first station the cell is first_cell_um; between stations it
    changes smoothly (gmsh's transfinite interpolation); past the last it stays constant. Each
    station is moved to the nearest node of the axial distribution, so the axial cells are
    unchanged. Without the keys there is one chamber block, first_cell_um everywhere.
    h is the lip-line cell at the exit plane, h_x the first axial cell (default h).
    """
    h_x = h if h_x is None else h_x
    fr = get(case, "su2", "lip_relax_x", [])
    cells = get(case, "su2", "lip_relax_cell_um", [])
    if len(fr) != len(cells):
        die("su2.lip_relax_x and su2.lip_relax_cell_um need the same number of entries")
    if any(not 0 < f <= 1 for f in fr) or list(fr) != sorted(set(fr)):
        die("su2.lip_relax_x must be increasing fractions of the chamber length in (0, 1]")
    if any(c <= 0 for c in cells):
        die("su2.lip_relax_cell_um must be positive")
    g = growth_rate(Lx, nx_e, h_x)
    i = np.arange(nx_e + 1)
    xn = i * 1.0 if g == 1.0 else (g**i - 1) / (g - 1)
    xn = Lx * xn / xn[-1]  # the axial node positions gmsh will produce
    st = [(0, 0.0, h)]
    for f, c in zip(fr, cells):
        k = int(np.argmin(np.abs(xn - f * Lx)))
        if k <= st[-1][0]:
            die(f"su2.lip_relax_x = {f} lands on the same chamber node as the station before it; "
                f"spread the stations out or raise n_axial_ext")
        st.append((k, float(xn[k]), c * 1e-6))
    if st[-1][0] < nx_e:
        st.append((nx_e, Lx, st[-1][2]))
    return st


def geo(case: dict, shape: Shape, source: str = "") -> str:
    """The .geo text for a Shape and the case's [geometry] chamber and [su2] mesh keys."""
    check_shape(shape)
    r = shape.r
    R_e = shape.R_exit
    Lx = get(case, "geometry", "chamb_length_cm") * 1e-2
    Rext = get(case, "geometry", "chamb_height_cm") * 1e-2
    h = get(case, "su2", "first_cell_um") * 1e-6
    h_x = get(case, "su2", "exit_cell_um", h * 1e6) * 1e-6  # axial cell either side of the exit plane
    nr = get(case, "su2", "n_radial_tube")
    nx_e = get(case, "su2", "n_axial_ext")
    nr_e = get(case, "su2", "n_radial_ext")
    if not Rext > R_e:
        die("geometry.chamb_height_cm must exceed the nozzle exit radius")

    secs = shape.sections
    stations = [s.x0 for s in secs] + [0.0]
    G = _Geo()
    axis_pts = [G.point(x, 0) for x in stations]
    wall_pts = [G.point(x, r(x)) for x in stations]

    # radial lines (wall -> axis), graded from the wall
    radial = []
    for k, x in enumerate(stations):
        c = G.line(wall_pts[k], axis_pts[k], "inlet" if k == 0 else ("exit plane" if x == 0 else ""))
        G.transfinite(c, nr, growth_rate(float(r(x)), nr, h))
        radial.append(c)

    axis_c, wall_c, surfaces = [], [], []
    for k, s in enumerate(secs):
        rev = s.fine == "end"  # orient curves so they start at the fine end
        a0, a1 = (axis_pts[k + 1], axis_pts[k]) if rev else (axis_pts[k], axis_pts[k + 1])
        ca = G.line(a0, a1, "axis, nozzle")
        w0, w1 = (wall_pts[k + 1], wall_pts[k]) if rev else (wall_pts[k], wall_pts[k + 1])
        if s.straight:
            cw = G.line(w0, w1, "wall")
        else:
            xs = np.linspace(s.x0, s.x1, N_SPLINE + 2)[1:-1]
            seq = [wall_pts[k]] + [G.point(x, r(x)) for x in xs] + [wall_pts[k + 1]]
            cw = G.spline(seq[::-1] if rev else seq, "wall")
        hf = s.h if s.h is not None else h_x
        prog = 1.0 if s.fine == "none" else growth_rate(s.x1 - s.x0, s.n, hf)
        G.transfinite(ca, s.n, prog)
        G.transfinite(cw, s.n, prog)
        sg = 1 if not rev else -1
        # loop A_k -> A_k+1 -> W_k+1 -> W_k -> A_k
        G.lines.append(f"Curve Loop({k + 1}) = {{{sg * ca}, {-radial[k + 1]}, {-sg * cw}, {radial[k]}}};")
        G.lines.append(f"Plane Surface({k + 1}) = {{{k + 1}}};")
        axis_c.append(ca)
        wall_c.append(cw)
        surfaces.append(k + 1)

    # chamber: blocks A (axis .. lip radius) and B (lip .. chamber radius), split along x at
    # the lip_relax stations; the radial first cell at the lip line can grow at each station
    st = chamber_stations(case, Lx, h, nx_e, h_x)
    g_xe = growth_rate(Lx, nx_e, h_x)
    ax_p, lip_p, top_p = [axis_pts[-1]], [wall_pts[-1]], [G.point(0, Rext)]
    for _, x, _ in st[1:]:
        ax_p.append(G.point(x, 0))
        lip_p.append(G.point(x, R_e))
        top_p.append(G.point(x, Rext))
    last = len(st) - 1
    lo, up = [radial[-1]], []  # radial lines at each station: lip -> axis (A), lip -> top (B)
    for k, (_, x, h_lip) in enumerate(st):
        if k > 0:
            c = G.line(lip_p[k], ax_p[k], "outlet, lower" if k == last else "chamber station")
            G.transfinite(c, nr, growth_rate(R_e, nr, h_lip))
            lo.append(c)
        c = G.line(lip_p[k], top_p[k],
                   "faceplate" if k == 0 else ("outlet, upper" if k == last else "chamber station"))
        G.transfinite(c, nr_e, growth_rate(Rext - R_e, nr_e, h_lip))
        up.append(c)
    ax_c, top_c = [], []
    for k in range(last):
        n = st[k + 1][0] - st[k][0]
        c_ax = G.line(ax_p[k], ax_p[k + 1], "axis, chamber")
        c_lip = G.line(lip_p[k], lip_p[k + 1], "lip line extension")
        c_top = G.line(top_p[k], top_p[k + 1], "top")
        for c in (c_ax, c_lip, c_top):
            G.transfinite(c, n, g_xe)
        nA, nB = len(surfaces) + 1, len(surfaces) + 2
        G.lines.append(f"Curve Loop({nA}) = {{{c_ax}, {-lo[k + 1]}, {-c_lip}, {lo[k]}}};")
        G.lines.append(f"Plane Surface({nA}) = {{{nA}}};")
        G.lines.append(f"Curve Loop({nB}) = {{{c_lip}, {up[k + 1]}, {-c_top}, {-up[k]}}};")
        G.lines.append(f"Plane Surface({nB}) = {{{nB}}};")
        surfaces += [nA, nB]
        ax_c.append(c_ax)
        top_c.append(c_top)

    sl = ", ".join(map(str, surfaces))
    join = lambda cs: ", ".join(map(str, cs))  # noqa: E731
    G.lines += [
        f"Transfinite Surface{{{sl}}};",
        f"Recombine Surface{{{sl}}};",
        "",
        f'Physical Curve("inlet") = {{{radial[0]}}};',
        f'Physical Curve("wall") = {{{join(wall_c)}}};',
        f'Physical Curve("faceplate") = {{{up[0]}}};',
        f'Physical Curve("axis") = {{{join(axis_c + ax_c)}}};',
        f'Physical Curve("outlet") = {{{join([lo[-1], up[-1]] + top_c)}}};',
        f'Physical Surface("fluid") = {{{sl}}};',
    ]
    head = [
        f"// Generated by su2/mesh.py ({shape.name}) from {source}",
        *[f"// {n}" for n in shape.notes],
        f"// nozzle {fmt(-shape.x_in)} m long, exit radius {fmt(R_e)} m; chamber {fmt(Lx)} x {fmt(Rext)} m",
        "// sections: " + "; ".join(f"x {fmt(s.x0)}..{fmt(s.x1)} m, {s.n} cells"
                                    + ("" if s.fine == "none" else f", fine at {s.fine}")
                                    for s in secs),
        f"// first cell {fmt(h)} m (wall, lip), axial cell at the exit plane {fmt(h_x)} m",
        "// lip-line first cell along the chamber: "
        + ", ".join(f"{fmt(h_lip)} m at x = {fmt(x)} m" for _, x, h_lip in st),
        "",
    ]
    return "\n".join(head) + G.text()


def write_mesh(case: dict, case_path, out_dir) -> Path:
    """mesh.geo, mesh.su2 (gmsh, log in gmsh.log) and mesh_quality.txt in out_dir; returns mesh.su2.
    Flagged quality checks are printed as warnings."""
    from su2 import mesh_quality  # here, not at the top: mesh_quality has no geometry dependence

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shape = provider(case).shape(case)
    geo_file = out_dir / "mesh.geo"
    geo_file.write_text(geo(case, shape, source=str(case_path)))

    gmsh_log = out_dir / "gmsh.log"
    su2 = out_dir / "mesh.su2"
    with open(gmsh_log, "w") as f:
        r = subprocess.run(["gmsh", str(geo_file), "-2", "-format", "su2", "-o", str(su2)],
                           stdout=f, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        print("\n".join(gmsh_log.read_text().splitlines()[-10:]), file=sys.stderr)
        die(f"gmsh failed, see {gmsh_log}")
    with open(su2) as f:
        nelem = next((m.group(1) for line in f if (m := re.match(r"NELEM=\s*(\d+)", line))), "?")
    log(f"mesh: {nelem} cells")

    qfile, flagged = mesh_quality.write(su2)
    for w in flagged:
        log(f"WARNING: {w}")
    log(f"wrote {qfile}")
    return su2


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("case", help="case file")
    ap.add_argument("--out", help="output directory (default: runs/<name>/mesh/)")
    ap.add_argument("--coarsen", type=int, default=1, help="N-times coarser mesh (as su2/setup.py --coarsen)")
    a = ap.parse_args(argv)
    case = load_case(Path(a.case).resolve())
    if a.coarsen != 1:
        from su2.setup import coarsen_case
        case = coarsen_case(case, a.coarsen)
    out = a.out or Path("runs") / get(case, "case", "name") / "mesh"
    write_mesh(case, Path(a.case).resolve(), out)


if __name__ == "__main__":
    main()
