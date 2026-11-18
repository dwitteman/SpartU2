"""Nozzle (or capillary) walls for a DSMC domain that starts inside the nozzle.

When the coupling plane x_c is upstream of the nozzle exit (x_c < 0), the SPARTA box
[x_c, Lx] x [0, R] contains a solid region: the nozzle body above the wall r_w(x) for
x_c <= x < 0. It is written as one closed 2D outline (read_surf format), extended past the
box so that `read_surf ... clip` trims it at the box edges:

    (0, R_big) <------------------- (x_c - d, R_big)      outside the box: clipped
        |                                  ^
        | faceplate (x = 0)                |              outside the box: clipped
        v                                  |
    (0, r_w(0)) --- wall, upstream ---> (x_c - d, r_w)

SPARTA's outward normal of a line p1 -> p2 is z x (p2 - p1), so walking this outline
(wall upstream, then up, across and down the faceplate) puts every normal on the flow side.
"""
from pathlib import Path

import numpy as np

from lib.common import die


def wall_profile(field, x_c):
    """(x, r) of the wall from x_c to the exit: the SU2 outline nodes in that range, plus the
    two ends, increasing in x."""
    e = field._edges
    if e is None:
        die("the SU2 field has no mesh cells, so the nozzle wall is unknown")
    nodes = np.unique(e.ravel())
    x = field.xy[nodes, 0]
    x = np.unique(np.concatenate([[x_c, 0.0], x[(x > x_c) & (x < 0)]]))
    # merge points closer than 1 nm (float32 copies of one SU2 node), keeping both ends
    keep = np.concatenate([[True], np.diff(x) > 1e-9])
    keep[-1] = True
    keep[-2] &= x[-1] - x[-2] > 1e-9
    x = x[keep]
    r = field.r_top(x)
    # at x = 0 r_top is the chamber height; the wall ends at the exit lip just before it
    r[-1] = field.r_top([-1e-12 * max(1.0, abs(x_c))])[0]
    if np.isnan(r).any():
        die(f"no nozzle wall found between x = {x_c:.4g} m and the exit")
    return x, r


def outline(x_w, r_w, box):
    """Closed outline of the nozzle body, as a list of (x, y) points (the last joins the first)."""
    xlo, xhi, ylo, yhi = box
    pad = 0.02 * max(xhi - xlo, yhi - ylo)
    # extend the wall straight past the box, along its first segment
    slope = (r_w[1] - r_w[0]) / (x_w[1] - x_w[0])
    x0 = xlo - pad
    # exit lip -> upstream; the wall point on the plane itself is left out (the extension passes
    # through it), or clipping would leave a near-zero-length line at the box edge
    pts = [(x, r) for x, r in zip(x_w[:0:-1], r_w[:0:-1])]
    pts.append((x0, r_w[0] + slope * (x0 - x_w[0])))
    pts += [(x0, yhi + pad), (0.0, yhi + pad)]                # up, across, faceplate back down
    return pts


def write_surf(path, pts, header=""):
    """Write a closed 2D outline as a read_surf file (points + lines, p_i -> p_i+1)."""
    n = len(pts)
    lines = [f"# {h}" for h in header.splitlines()] + [
        "", f"{n} points", f"{n} lines", "", "Points", ""]
    lines += [f"{i + 1} {x:.12g} {y:.12g}" for i, (x, y) in enumerate(pts)]
    lines += ["", "Lines", ""]
    lines += [f"{i + 1} {i + 1} {(i + 1) % n + 1}" for i in range(n)]
    Path(path).write_text("\n".join(lines) + "\n")


def solid_test(x_w, r_w):
    """f(x, y) -> True inside the nozzle body (x_w[0] <= x < 0 and y above the wall)."""
    def inside(x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        return (x >= x_w[0]) & (x < 0) & (y > np.interp(x, x_w, r_w))
    return inside
