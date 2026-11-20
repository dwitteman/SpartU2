"""Geometry provider: axisymmetric conical converging-diverging nozzle (noz_type = "cd_nozzle").

x is the symmetry axis, y is radius, exit plane at x = 0. Units: metres.

Wall profile, upstream to downstream:
  inlet (radius R_in) -> cone at conv_half_angle -> circular arc of radius rc into the
  throat (R_t) -> optional straight throat of length L_t -> circular arc of radius rc
  -> cone at div_half_angle -> exit (R_e) at x = 0.

The nozzle is split into sections at stations along x (su2/mesh.py builds the blocks):
  conv    inlet -> throat start     axial cells clustered at the throat (throat_cell_um)
  throat  straight throat (if any)  uniform, ~throat_cell
  div1    throat end -> midpoint     clustered at the throat
  div2    midpoint -> exit           clustered at the exit plane (exit_cell_um, default first_cell_um)
"""
import math
from dataclasses import dataclass

import numpy as np

from lib.common import die, get, growth_rate  # noqa: F401 (growth_rate re-exported for tests)
from su2.mesh import Section, Shape, chamber_stations, geo as _mesh_geo  # noqa: F401

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


def shape(case: dict) -> Shape:
    """The nozzle as sections for su2/mesh.py."""
    p = profile(case)
    h_t = get(case, "su2", "throat_cell_um", p.R_t / 20 * 1e6) * 1e-6
    nx_c = get(case, "su2", "n_axial_conv")
    nx_d = get(case, "su2", "n_axial_div")
    if nx_d < 2:
        die("su2.n_axial_div must be at least 2 (it is split between two blocks)")
    x_mid = p.x_td / 2
    secs = [Section(p.x_in, p.x_tu, nx_c, "end", h_t)]
    if p.L_t > 0:
        secs.append(Section(p.x_tu, p.x_td, max(1, round(p.L_t / h_t)), "none", straight=True))
    secs += [Section(p.x_td, x_mid, nx_d // 2, "start", h_t),
             Section(x_mid, 0.0, nx_d - nx_d // 2, "end", None)]  # None: exit_cell_um
    return Shape(r=p.r, sections=secs, name="cd_nozzle", notes=[
        f"conical CD nozzle: D_in = {2*p.R_in:.6g} m, D_t = {2*p.R_t:.6g} m, D_e = {2*p.R_e:.6g} m, "
        f"A_e/A_t = {p.area_ratio:.4g}",
        f"throat rc = {p.rc:.6g} m, straight throat {p.L_t:.6g} m, throat axial cell {h_t:.6g} m"])


def geo(case: dict, source: str = "") -> str:
    """The .geo text for this case (su2/mesh.py)."""
    return _mesh_geo(case, shape(case), source)
