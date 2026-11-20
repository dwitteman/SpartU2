"""Geometry provider: straight capillary tube (noz_type = "capillary").

A tube of diameter throat_diam_mm and length throat_length_mm, exit plane at x = 0, as one
section of n_axial_tube cells clustered at the exit (exit_cell_um, default first_cell_um).
su2/mesh.py adds the chamber:

  +--------------- top (outlet) --------------+
  |                                           |
 faceplate              block B             outlet
  |                                           |
  +---- tube wall ----+ lip ------------------+
  |                   |                       |
 inlet   tube block   |       block A       outlet
  |                   |                       |
  +------ axis -------+------- axis ----------+

Markers: inlet, wall, faceplate, axis, outlet
"""
import numpy as np

from lib.common import die, get
from su2.mesh import Section, Shape, geo as _mesh_geo


def shape(case: dict) -> Shape:
    D = get(case, "geometry", "throat_diam_mm") * 1e-3
    L = get(case, "geometry", "throat_length_mm") * 1e-3
    if not (D > 0 and L > 0):
        die("geometry.throat_diam_mm and throat_length_mm must be positive")
    R = D / 2
    nx_t = get(case, "su2", "n_axial_tube")
    return Shape(r=lambda x: np.full(np.shape(x), R) if np.ndim(x) else R,
                 sections=[Section(-L, 0.0, nx_t, "end", None, straight=True)],
                 name="capillary", notes=[f"capillary: D = {D:.6g} m, L = {L:.6g} m"])


def geo(case: dict, source: str = "") -> str:
    """The .geo text for this case (su2/mesh.py)."""
    return _mesh_geo(case, shape(case), source)
