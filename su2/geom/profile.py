"""Geometry provider: any nozzle wall from a table of points (noz_type = "profile").

[geometry]
noz_type = "profile"
profile_file = "my_nozzle.csv"   # x, r of the wall, inlet to exit; relative to the case file
profile_units = "mm"             # "mm" (default) or "m"
throat_diam_mm = 0.5             # still the reference length D for the handoff and SPARTA

The file has two columns, axial position and wall radius, from the inlet to the exit (a header
line and # comments are skipped; commas, tabs or spaces). x must increase; it is shifted so the
last point, the exit, is at x = 0. The wall between the points is a monotone cubic (PCHIP)
through them, so it passes through every point without overshooting; add points where the
wall curves sharply.

Sections (su2/mesh.py builds the blocks), with the throat at the narrowest point (the last one,
if a stretch of the wall is equally narrow):
  inlet -> throat    n_axial_conv cells clustered at the throat (throat_cell_um, default R_t/20)
  throat -> exit     n_axial_div cells in two sections, clustered at the throat and at the exit
                     (exit_cell_um, default first_cell_um)
A wall that narrows all the way to the exit (or a straight tube) is one section of n_axial_conv
cells clustered at the exit; one that only widens is the two diverging sections.

This is the route for CAD geometry too: anything that yields the wall as (x, r) points of an
axisymmetric half-section can be written to such a file.
"""
import re
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

from lib.common import case_file, die, get, log
from su2.mesh import Section, Shape, geo as _mesh_geo

UNITS = {"mm": 1e-3, "m": 1.0}


def read_profile(path, units="mm"):
    """(x, r) [m] of a wall profile file, x shifted so the last point is at 0."""
    path = Path(path)
    if not path.is_file():
        die(f"profile file {path} not found")
    if units not in UNITS:
        die(f"geometry.profile_units = {units!r}: use \"mm\" or \"m\"")
    rows = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = [p for p in re.split(r"[,\s;]+", line) if p]
        try:
            rows.append([float(parts[0]), float(parts[1])])
        except (ValueError, IndexError):
            if rows:
                die(f"{path}: can't read the line {line!r}")
            continue  # header
    if len(rows) < 2:
        die(f"{path}: needs at least two (x, r) points")
    xr = np.array(rows) * UNITS[units]
    x, r = xr[:, 0], xr[:, 1]
    if not np.all(np.diff(x) > 0):
        die(f"{path}: x must increase from the inlet to the exit (a single-valued wall r(x))")
    if not np.all(r > 0):
        die(f"{path}: the wall radius must be positive")
    return x - x[-1], r


def shape(case: dict) -> Shape:
    g = "geometry"
    path = case_file(case, get(case, g, "profile_file"))
    x, r = read_profile(path, get(case, g, "profile_units", "mm"))
    wall = PchipInterpolator(x, r, extrapolate=False)
    r_of = lambda xx: wall(np.clip(xx, x[0], 0.0))  # noqa: E731 (clip guards round-off at the ends)

    tol = 1e-9 * r.max()
    i_t = int(np.flatnonzero(r <= r.min() + tol)[-1])  # narrowest point (the last, if a stretch is)
    x_t, R_t = float(x[i_t]), float(r[i_t])
    D = get(case, g, "throat_diam_mm") * 1e-3
    if abs(2 * R_t - D) > 0.01 * D:
        log(f"WARNING: geometry.throat_diam_mm = {D*1e3:g} mm but the profile's narrowest diameter is "
            f"{2*R_t*1e3:.4g} mm; D is the length unit of the handoff and SPARTA (x_couple_D etc.)")
    h_t = get(case, "su2", "throat_cell_um", R_t / 20 * 1e6) * 1e-6
    nx_c = get(case, "su2", "n_axial_conv")
    secs = []
    if i_t == len(x) - 1:  # narrows to the exit (or straight): one section clustered at the exit
        secs.append(Section(x[0], 0.0, nx_c, "end", None))
    else:
        nx_d = get(case, "su2", "n_axial_div")
        if nx_d < 2:
            die("su2.n_axial_div must be at least 2 (it is split between two sections)")
        if i_t > 0:
            secs.append(Section(x[0], x_t, nx_c, "end", h_t))
        x_mid = x_t / 2
        secs += [Section(x_t, x_mid, nx_d // 2, "start", h_t),
                 Section(x_mid, 0.0, nx_d - nx_d // 2, "end", None)]
    return Shape(r=r_of, sections=secs, name="profile", notes=[
        f"profile {path.name}: {len(x)} points, inlet D = {2*r[0]:.6g} m, throat D = {2*R_t:.6g} m "
        f"at x = {x_t:.6g} m, exit D = {2*r[-1]:.6g} m"])


def geo(case: dict, source: str = "") -> str:
    """The .geo text for this case (su2/mesh.py)."""
    return _mesh_geo(case, shape(case), source)
