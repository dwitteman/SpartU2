"""gmsh .geo for an axisymmetric conical converging-diverging nozzle exhausting into a chamber.

x is the symmetry axis, y is radius, exit plane at x = 0. Units: metres.

Wall profile, upstream to downstream:
  inlet (radius R_in) -> cone at conv_half_angle -> circular arc of radius rc into the
  throat (R_t) -> optional straight throat of length L_t -> circular arc of radius rc
  -> cone at div_half_angle -> exit (R_e) at x = 0.

The nozzle is split into structured blocks at stations along x:
  conv    inlet -> throat start     axial cells clustered at the throat
  throat  straight throat (if any)  uniform, ~throat_cell
  div1    throat end -> midpoint     clustered at the throat
  div2    midpoint -> exit           clustered at the exit plane (exit_cell, default first_cell)
then the chamber blocks A (r < R_e) and B (lip -> chamber radius), as in capillary.py:

  +------------------------------- top (outlet) -----------------+
  |                                                              |
  faceplate                     block B                       outlet
  |                                                              |
 _+ lip ---------------------------------------------------------+
  /wall\\___    _____/ wall                                       |
 |  conv |thr| div1 | div2 |            block A                outlet
 +-------+---+------+------+-------------------------------------+
 inlet          axis                     axis

Every radial line is graded so its first cell at the wall is first_cell_um.
Markers: inlet, wall, faceplate, axis, outlet (same as capillary.py).
"""
import math
from dataclasses import dataclass

import numpy as np

from lib.common import die, fmt, get, growth_rate

N_SPLINE = 200  # interior points per curved wall segment


@dataclass
class Profile:
    """Nozzle wall r(x) [m]; exit plane at x = 0, everything upstream negative."""
    R_in: float
    R_t: float
    R_e: float
    rc: float
    th_c: float  # converging half-angle, rad
    th_d: float  # diverging half-angle, rad
    L_t: float
    x_in: float = 0.0
    x_tu: float = 0.0  # throat start (end of converging arc)
    x_td: float = 0.0  # throat end (start of diverging arc)
    x_c: float = 0.0   # converging cone/arc tangent point
    x_d: float = 0.0   # diverging arc/cone tangent point

    def __post_init__(self):
        R_t, rc = self.R_t, self.rc
        r_c = R_t + rc * (1 - math.cos(self.th_c))  # radius at converging tangent point
        r_d = R_t + rc * (1 - math.cos(self.th_d))  # radius at diverging tangent point
        if self.R_in <= r_c:
            die(f"inlet radius {self.R_in*1e3:g} mm must exceed {r_c*1e3:.4g} mm "
                f"(throat + converging arc); increase inlet_diam_mm or reduce throat_rc_D")
        if self.R_e <= r_d:
            die(f"exit radius {self.R_e*1e3:g} mm must exceed {r_d*1e3:.4g} mm "
                f"(throat + diverging arc); increase exit_diam_mm or reduce throat_rc_D")
        # lay out from the exit (x = 0) upstream
        len_div_cone = (self.R_e - r_d) / math.tan(self.th_d)
        self.x_d = -len_div_cone
        self.x_td = self.x_d - rc * math.sin(self.th_d)
        self.x_tu = self.x_td - self.L_t
        self.x_c = self.x_tu - rc * math.sin(self.th_c)
        self.x_in = self.x_c - (self.R_in - r_c) / math.tan(self.th_c)

    @property
    def length(self):
        return -self.x_in

    @property
    def area_ratio(self):
        return (self.R_e / self.R_t) ** 2

    def r(self, x):
        """Wall radius at x (array or scalar)."""
        x = np.asarray(x, float)
        rc, R_t = self.rc, self.R_t
        out = np.empty_like(x)
        conv_cone = x < self.x_c
        conv_arc = (x >= self.x_c) & (x < self.x_tu)
        throat = (x >= self.x_tu) & (x <= self.x_td)
        div_arc = (x > self.x_td) & (x <= self.x_d)
        div_cone = x > self.x_d
        r_c = R_t + rc * (1 - math.cos(self.th_c))
        r_d = R_t + rc * (1 - math.cos(self.th_d))
        out[conv_cone] = r_c + (self.x_c - x[conv_cone]) * math.tan(self.th_c)
        out[conv_arc] = R_t + rc - np.sqrt(np.maximum(rc**2 - (x[conv_arc] - self.x_tu) ** 2, 0))
        out[throat] = R_t
        out[div_arc] = R_t + rc - np.sqrt(np.maximum(rc**2 - (x[div_arc] - self.x_td) ** 2, 0))
        out[div_cone] = r_d + (x[div_cone] - self.x_d) * math.tan(self.th_d)
        return out if out.ndim else float(out)


def profile(case: dict) -> Profile:
    g = "geometry"
    R_t = get(case, g, "throat_diam_mm") * 1e-3 / 2
    th_c, th_d = get(case, g, "conv_half_angle_deg"), get(case, g, "div_half_angle_deg")
    for name, th in (("conv_half_angle_deg", th_c), ("div_half_angle_deg", th_d)):
        if not 0 < th < 90:
            die(f"geometry.{name} must be between 0 and 90 degrees")
    return Profile(R_in=get(case, g, "inlet_diam_mm") * 1e-3 / 2, R_t=R_t,
                   R_e=get(case, g, "exit_diam_mm") * 1e-3 / 2,
                   rc=get(case, g, "throat_rc_D", 1.0) * 2 * R_t,
                   th_c=math.radians(th_c), th_d=math.radians(th_d),
                   L_t=get(case, g, "throat_length_mm", 0.0) * 1e-3)


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


def geo(case: dict, source: str = "") -> str:
    """Return the .geo file text for this case. `source` is noted in the header."""
    p = profile(case)
    Lx = get(case, "geometry", "chamb_length_cm") * 1e-2
    Rext = get(case, "geometry", "chamb_height_cm") * 1e-2
    h = get(case, "su2", "first_cell_um") * 1e-6
    h_x = get(case, "su2", "exit_cell_um", h * 1e6) * 1e-6  # axial cell either side of the exit plane
    h_t = get(case, "su2", "throat_cell_um", p.R_t / 20 * 1e6) * 1e-6
    nr = get(case, "su2", "n_radial_tube")
    nx_c = get(case, "su2", "n_axial_conv")
    nx_d = get(case, "su2", "n_axial_div")
    nx_e = get(case, "su2", "n_axial_ext")
    nr_e = get(case, "su2", "n_radial_ext")
    if not Rext > p.R_e:
        die("geometry.chamb_height_cm must exceed the nozzle exit radius")
    if nx_d < 2:
        die("su2.n_axial_div must be at least 2 (it is split between two blocks)")

    x_mid = p.x_td / 2
    # stations and the blocks between them: (x_start, x_end, cells, fine end, fine size)
    blocks = [(p.x_in, p.x_tu, nx_c, "end", h_t)]
    if p.L_t > 0:
        blocks.append((p.x_tu, p.x_td, max(1, round(p.L_t / h_t)), "none", None))
    blocks += [(p.x_td, x_mid, nx_d // 2, "start", h_t), (x_mid, 0.0, nx_d - nx_d // 2, "end", h_x)]
    stations = [b[0] for b in blocks] + [0.0]

    G = _Geo()
    axis_pts = [G.point(x, 0) for x in stations]
    wall_pts = [G.point(x, p.r(x)) for x in stations]

    # radial lines (wall -> axis), graded from the wall
    radial = []
    for k, x in enumerate(stations):
        c = G.line(wall_pts[k], axis_pts[k], "inlet" if k == 0 else ("exit plane" if x == 0 else ""))
        G.transfinite(c, nr, growth_rate(p.r(x), nr, h))
        radial.append(c)

    axis_c, wall_c, surfaces = [], [], []
    for k, (x0, x1, n, fine, hf) in enumerate(blocks):
        rev = fine == "end"  # orient curves so they start at the fine end
        a0, a1 = (axis_pts[k + 1], axis_pts[k]) if rev else (axis_pts[k], axis_pts[k + 1])
        ca = G.line(a0, a1, "axis, nozzle")
        xs = np.linspace(x0, x1, N_SPLINE + 2)[1:-1]
        if p.L_t > 0 and fine == "none":
            cw = G.line(*((wall_pts[k + 1], wall_pts[k]) if rev else (wall_pts[k], wall_pts[k + 1])),
                        "wall, throat")
        else:
            inner = [G.point(x, p.r(x)) for x in xs]
            seq = [wall_pts[k]] + inner + [wall_pts[k + 1]]
            cw = G.spline(seq[::-1] if rev else seq, "wall")
        prog = 1.0 if fine == "none" else growth_rate(x1 - x0, n, hf)
        G.transfinite(ca, n, prog)
        G.transfinite(cw, n, prog)
        s = 1 if not rev else -1
        # loop A_k -> A_k+1 -> W_k+1 -> W_k -> A_k
        G.lines.append(f"Curve Loop({k + 1}) = {{{s * ca}, {-radial[k + 1]}, {-s * cw}, {radial[k]}}};")
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
        lip_p.append(G.point(x, p.R_e))
        top_p.append(G.point(x, Rext))
    last = len(st) - 1
    lo, up = [radial[-1]], []  # radial lines at each station: lip -> axis (A), lip -> top (B)
    for k, (_, x, h_lip) in enumerate(st):
        if k > 0:
            c = G.line(lip_p[k], ax_p[k], "outlet, lower" if k == last else "chamber station")
            G.transfinite(c, nr, growth_rate(p.R_e, nr, h_lip))
            lo.append(c)
        c = G.line(lip_p[k], top_p[k],
                   "faceplate" if k == 0 else ("outlet, upper" if k == last else "chamber station"))
        G.transfinite(c, nr_e, growth_rate(Rext - p.R_e, nr_e, h_lip))
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
        f"// Generated by su2/geom/cd_nozzle.py from {source}",
        f"// conical CD nozzle: D_in = {fmt(2*p.R_in)} m, D_t = {fmt(2*p.R_t)} m, D_e = {fmt(2*p.R_e)} m, "
        f"A_e/A_t = {p.area_ratio:.4g}, length {fmt(p.length)} m",
        f"// throat rc = {fmt(p.rc)} m, straight throat {fmt(p.L_t)} m; domain {fmt(Lx)} x {fmt(Rext)} m",
        f"// first cell {fmt(h)} m (wall, lip), axial cell at the exit plane {fmt(h_x)} m, "
        f"throat axial cell {fmt(h_t)} m",
        "// lip-line first cell along the chamber: "
        + ", ".join(f"{fmt(h_lip)} m at x = {fmt(x)} m" for _, x, h_lip in st),
        "",
    ]
    return "\n".join(head) + G.text()
