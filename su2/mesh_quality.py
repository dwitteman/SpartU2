#!/usr/bin/env python3
"""Mesh quality report for a 2D quadrilateral SU2 mesh: mesh_quality.txt.

Usage: python su2/mesh_quality.py runs/<name>/su2/mesh.su2   # prints it, writes mesh_quality.txt beside it
su2/setup.py writes it after meshing and prints the flagged checks.

Per cell:
  aspect ratio  longest / shortest edge
  corner angle  the four interior angles (90 = square)
  size jump     largest area ratio to a cell sharing an edge (1 = same size)
Summarised for the nozzle (x < 0), the chamber below the lip radius (the jet core) and the
chamber above it. The lip is the downstream end of the "wall" marker.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.common import die

# warning thresholds
MAX_ASPECT = 1000.0
ANGLE_RANGE = (45.0, 135.0)
MAX_JUMP = 1.5

_QUAD = 9  # SU2/VTK element type


def read_su2(path):
    """(points [n, 2], quads [m, 4], {marker: line elements [k, 2]}) of a 2D SU2 mesh."""
    lines = Path(path).read_text().splitlines()
    pts, quads, marks, i = None, None, {}, 0
    while i < len(lines):
        s = lines[i].strip()
        if s.startswith("NDIME=") and int(s.split("=")[1]) != 2:
            die(f"{path}: only 2D meshes are supported")
        if s.startswith("NELEM="):
            n = int(s.split("=")[1])
            el = [l.split() for l in lines[i + 1:i + 1 + n]]
            types = {int(e[0]) for e in el}
            if types - {_QUAD}:
                die(f"{path}: only quadrilateral meshes are supported (element types {sorted(types)})")
            quads = np.array([[int(v) for v in e[1:5]] for e in el])
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
    if pts is None or quads is None:
        die(f"{path}: no NPOIN or NELEM section")
    return pts, quads, marks


def cell_metrics(pts, quads) -> dict:
    """Aspect ratio, min/max corner angle, size jump, area and centroid of every quad."""
    q = pts[quads]                                   # [m, 4, 2]
    e = np.roll(q, -1, axis=1) - q                   # edge k: corner k -> k+1
    L = np.linalg.norm(e, axis=2)
    prev = -np.roll(e, 1, axis=1)                    # corner k -> k-1
    cos = np.sum(prev * e, axis=2) / (np.linalg.norm(prev, axis=2) * L)
    ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    area = 0.5 * np.abs(np.sum(q[..., 0] * np.roll(q[..., 1], -1, 1)
                               - np.roll(q[..., 0], -1, 1) * q[..., 1], axis=1))

    # cells sharing an edge: sort all edges by their node pair, equal neighbours are a pair
    ed = np.sort(np.stack([quads, np.roll(quads, -1, axis=1)], axis=2).reshape(-1, 2), axis=1)
    key = ed[:, 0].astype(np.int64) * len(pts) + ed[:, 1]
    order = np.argsort(key, kind="stable")
    k, cell = key[order], np.repeat(np.arange(len(quads)), 4)[order]
    d = np.flatnonzero(k[1:] == k[:-1])
    a, b = cell[d], cell[d + 1]
    r = np.maximum(area[a] / area[b], area[b] / area[a])
    jump = np.ones(len(quads))
    np.maximum.at(jump, a, r)
    np.maximum.at(jump, b, r)
    return dict(aspect=L.max(1) / L.min(1), angle_min=ang.min(1), angle_max=ang.max(1),
                jump=jump, area=area, xy=q.mean(1))


def lip(pts, marks):
    """(x, r) of the lip: the most downstream node of the wall marker, or None."""
    if "wall" not in marks:
        return None
    w = np.unique(marks["wall"].ravel())
    return pts[w[np.argmax(pts[w, 0])]]


def _loc(xy):
    return f"({xy[0] * 1e3:.3g}, {xy[1] * 1e3:.3g}) mm"


def analyse(path):
    """Return (report text, list of flagged checks)."""
    pts, quads, marks = read_su2(path)
    m = cell_metrics(pts, quads)
    x, r = m["xy"].T
    tip = lip(pts, marks)
    regions = [("all", np.ones(len(quads), bool))]
    if tip is not None:
        x0, r0 = tip
        regions += [(f"nozzle (x < {x0 * 1e3:g} mm)", x < x0),
                    ("chamber, r < lip (jet core)", (x >= x0) & (r < r0)),
                    ("chamber, r > lip", (x >= x0) & (r >= r0))]

    out = [f"Mesh quality: {path}",
           f"{len(quads)} quadrilaterals, {len(pts)} nodes"
           + (f"; lip at {_loc(tip)}" if tip is not None else ""),
           "",
           "aspect ratio = longest / shortest edge; angle = interior corner angles (90 = square);",
           "size jump = largest area ratio to a neighbouring cell (1 = same size).",
           "",
           f"{'region':30s} {'cells':>7s} | {'aspect ratio':^22s} | {'angle [deg]':^11s} | {'size jump':^11s}",
           f"{'':30s} {'':>7s} | {'median':>6s} {'99%':>7s} {'max':>7s} | {'min':>5s} {'max':>5s} | {'99%':>5s} {'max':>5s}"]
    for name, sel in regions:
        if not sel.any():
            continue
        ar, j = m["aspect"][sel], m["jump"][sel]
        out.append(f"{name:30s} {sel.sum():7d} | {np.median(ar):6.3g} {np.percentile(ar, 99):7.4g} "
                   f"{ar.max():7.4g} | {m['angle_min'][sel].min():5.1f} {m['angle_max'][sel].max():5.1f} | "
                   f"{np.percentile(j, 99):5.3g} {j.max():5.3g}")

    worst_angle = np.maximum(90 - m["angle_min"], m["angle_max"] - 90)
    out += ["", "Worst cells (x, r):"]
    for label, v, f in (("aspect ratio", m["aspect"], "{:.4g}"), ("size jump", m["jump"], "{:.3g}"),
                        ("angle off 90 deg", worst_angle, "{:.1f}")):
        top = np.argsort(v)[::-1][:3]
        out.append(f"  {label:17s} " + ";  ".join(f"{f.format(v[i])} at {_loc(m['xy'][i])}" for i in top))

    lo, hi = ANGLE_RANGE
    bad_ar = m["aspect"] > MAX_ASPECT
    bad_ang = (m["angle_min"] < lo) | (m["angle_max"] > hi)
    bad_j = m["jump"] > MAX_JUMP
    checks = [
        (not bad_ar.any(), f"aspect ratio <= {MAX_ASPECT:g}",
         f"{bad_ar.sum()} cells above, worst {m['aspect'].max():.4g} at {_loc(m['xy'][np.argmax(m['aspect'])])}"),
        (not bad_ang.any(), f"corner angles within {lo:g}-{hi:g} deg",
         f"{bad_ang.sum()} cells outside, worst {worst_angle.max() + 90:.1f} deg "
         f"at {_loc(m['xy'][np.argmax(worst_angle)])}"),
        (not bad_j.any(), f"size jump <= {MAX_JUMP:g}",
         f"{bad_j.sum()} cells above, worst {m['jump'].max():.3g} at {_loc(m['xy'][np.argmax(m['jump'])])}"),
    ]
    out += ["", "Checks:"]
    flagged = []
    for ok, what, detail in checks:
        out.append(f"  [{'ok' if ok else '!!'}] {what}" + ("" if ok else f": {detail}"))
        if not ok:
            flagged.append(f"mesh: {what} fails: {detail}")
    out += ["", "SU2 prints its own (dual control volume) quality table near the top of su2.log."]
    return "\n".join(out) + "\n", flagged


def write(mesh_path, out_path=None):
    """Write mesh_quality.txt (default: beside the mesh). Returns (path, flagged checks)."""
    text, flagged = analyse(mesh_path)
    out = Path(out_path) if out_path else Path(mesh_path).with_name("mesh_quality.txt")
    out.write_text(text)
    return out, flagged


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mesh", help="mesh.su2")
    ap.add_argument("-o", "--out", help="report file (default: mesh_quality.txt beside the mesh)")
    a = ap.parse_args(argv)
    out, _ = write(a.mesh, a.out)
    print(out.read_text(), end="")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
