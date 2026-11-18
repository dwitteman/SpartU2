"""Read an SU2 solution (ParaView .vtu) and sample it at arbitrary points.

Only the SU2 output written by su2/setup.py is supported: binary appended
data, raw encoding, UInt64 block headers. In the axisymmetric case, x is the
axis and y is the radius.

The field covers the whole SU2 domain by default: the nozzle or capillary (x < 0)
as well as the chamber. Interpolation uses a Delaunay triangulation of the SU2 nodes,
with the triangles that fall outside the mesh (outside the nozzle wall, where the
domain is not convex) masked out, so nothing is interpolated or differentiated across
a wall.
"""
import re
from pathlib import Path

import numpy as np
from scipy.interpolate import NearestNDInterpolator
from scipy.spatial import Delaunay

from lib.common import die

_DTYPES = {"Float32": "<f4", "Float64": "<f8", "Int32": "<i4", "Int64": "<i8", "UInt8": "u1"}
_ARRAY = re.compile(rb'<DataArray type="(\w+)" Name="(\w*)" NumberOfComponents= ?"(\d+)" '
                    rb'offset="(\d+)" format="appended"/>')
_VTK_TRIANGLE, _VTK_QUAD = 5, 9


def read_vtu(path) -> dict:
    """Return {name: array} for every DataArray; the point coordinates are under 'Points'."""
    raw = Path(path).read_bytes()
    tag = b'<AppendedData encoding="raw">'
    start = raw.find(tag)
    if start < 0 or b'header_type="UInt64"' not in raw[:start]:
        die(f"{path}: expected binary appended .vtu with UInt64 headers (SU2 PARAVIEW output)")
    data = raw.index(b"_", start + len(tag)) + 1  # appended block starts after '_'

    arrays = {}
    for typ, name, ncomp, offset in _ARRAY.findall(raw[:start]):
        pos = data + int(offset)
        if pos + 8 > len(raw) or pos + 8 + int(np.frombuffer(raw, "<u8", 1, pos)[0]) > len(raw):
            die(f"{path} is truncated ({len(raw)} bytes): SU2 was probably stopped while "
                f"writing it. Rewrite it from restart_flow.dat with one SU2 iteration "
                f"(RESTART_SOL= YES, ITER= 1).")
        nbytes = int(np.frombuffer(raw, "<u8", 1, pos)[0])
        a = np.frombuffer(raw, _DTYPES[typ.decode()], nbytes // np.dtype(_DTYPES[typ.decode()]).itemsize,
                          pos + 8)
        ncomp = int(ncomp)
        arrays[name.decode() or "Points"] = a.reshape(-1, ncomp) if ncomp > 1 else a
    return arrays


def vtu_cells(arrays) -> list:
    """The 2D cells of a read_vtu() result, as a list of node-index arrays (triangles and quads)."""
    conn, offs, types = arrays["connectivity"], arrays["offsets"], arrays["types"]
    starts = np.concatenate([[0], offs[:-1]])
    cells = []
    for t, n in ((_VTK_TRIANGLE, 3), (_VTK_QUAD, 4)):
        sel = types == t
        if sel.any():
            cells.append(conn[starts[sel, None] + np.arange(n)].astype(np.int64))
    return cells


def boundary_edges(cells) -> np.ndarray:
    """(E, 2) node pairs of the edges that belong to exactly one cell: the domain outline."""
    edges = np.concatenate([np.stack([c, np.roll(c, -1, axis=1)], axis=-1).reshape(-1, 2) for c in cells])
    key = np.sort(edges, axis=1)
    uniq, count = np.unique(key, axis=0, return_counts=True)
    return uniq[count == 1]


def points_inside(pts, xy, edges, chunk=4096) -> np.ndarray:
    """True for points inside the polygon(s) outlined by edges (even-odd ray casting along +x)."""
    a, b = xy[edges[:, 0]], xy[edges[:, 1]]
    out = np.zeros(len(pts), bool)
    for i in range(0, len(pts), chunk):
        p = pts[i:i + chunk, None, :]
        straddle = (a[None, :, 1] > p[..., 1]) != (b[None, :, 1] > p[..., 1])
        with np.errstate(divide="ignore", invalid="ignore"):
            xc = a[None, :, 0] + (p[..., 1] - a[None, :, 1]) * (b[None, :, 0] - a[None, :, 0]) / (
                b[None, :, 1] - a[None, :, 1])
        out[i:i + chunk] = (straddle & (p[..., 0] < xc)).sum(axis=1) % 2 == 1
    return out


class Su2Field:
    """SU2 flow field, linearly interpolated, restricted to the region x >= xmin.

    Attributes: xy (N, 2) node positions; nodes {rho [kg/m3], T [K], u, v [m/s]};
    tri, the Delaunay triangulation of the nodes; simplices, its triangles inside the
    SU2 domain (use these for gradients and plots); r_top(x), the largest radius of the
    domain at x (the nozzle or capillary wall for x < 0, the chamber height after).
    """

    FIELDS = ("Density", "Temperature")

    def __init__(self, vtu_path, xmin=-np.inf):
        a = read_vtu(vtu_path)
        for f in self.FIELDS + ("Velocity",):
            if f not in a:
                die(f"{vtu_path}: no '{f}' field (VOLUME_OUTPUT must include PRIMITIVE)")
        xy = a["Points"][:, :2].astype(float)
        vals = np.column_stack([a["Density"], a["Temperature"],
                                a["Velocity"][:, 0], a["Velocity"][:, 1]]).astype(float)
        cells = vtu_cells(a) if "connectivity" in a else None
        keep = xy[:, 0] >= xmin
        if cells is not None and not keep.all():
            new = np.cumsum(keep) - 1
            cells = [new[c[keep[c].all(axis=1)]] for c in cells]
        self._build(xy[keep], dict(zip(("rho", "T", "u", "v"), vals[keep].T)), cells)

    @classmethod
    def from_arrays(cls, xy, nodes, cells=None):
        """A field from node positions, node values {rho, T, u, v} and optional mesh cells
        (lists of node-index arrays). Without cells the domain is the convex hull of xy."""
        self = cls.__new__(cls)
        self._build(np.asarray(xy, float), {k: np.asarray(v, float) for k, v in nodes.items()}, cells)
        return self

    def _build(self, xy, nodes, cells):
        self.xy, self.nodes = xy, nodes
        self.tri = Delaunay(xy)
        self.inside = np.ones(len(self.tri.simplices), bool)
        if cells:
            self._edges = boundary_edges(cells)
            # Delaunay fills the convex hull; a triangle outside the mesh has all its nodes on
            # the domain outline (there are no nodes outside it), so only those need testing.
            on_edge = np.zeros(len(xy), bool)
            on_edge[self._edges.ravel()] = True
            cand = np.nonzero(on_edge[self.tri.simplices].all(axis=1))[0]
            centroid = xy[self.tri.simplices[cand]].mean(axis=1)
            self.inside[cand] = points_inside(centroid, xy, self._edges)
        else:
            self._edges = None
        self.simplices = self.tri.simplices[self.inside]
        self._near = NearestNDInterpolator(xy, np.column_stack([nodes[k] for k in ("rho", "T", "u", "v")]))

    def r_top(self, x) -> np.ndarray:
        """Largest radius inside the domain at each x (nan outside the domain's x range)."""
        x = np.atleast_1d(np.asarray(x, float))
        if self._edges is None:
            return np.where((x >= self.xy[:, 0].min()) & (x <= self.xy[:, 0].max()), self.xy[:, 1].max(), np.nan)
        a, b = self.xy[self._edges[:, 0]], self.xy[self._edges[:, 1]]
        out = np.full(len(x), np.nan)
        for i in range(0, len(x), 512):
            xi = x[i:i + 512, None]
            lo, hi = np.minimum(a[:, 0], b[:, 0]), np.maximum(a[:, 0], b[:, 0])
            span = (xi >= lo) & (xi <= hi)
            with np.errstate(divide="ignore", invalid="ignore"):
                t = np.where(hi > lo, (xi - a[:, 0]) / (b[:, 0] - a[:, 0]), 0.0)
            r = np.where(hi > lo, a[:, 1] + t * (b[:, 1] - a[:, 1]), np.maximum(a[:, 1], b[:, 1]))
            r = np.where(span, r, -np.inf).max(axis=1)
            out[i:i + 512] = np.where(np.isfinite(r), r, np.nan)
        return out

    def interpolate(self, values, pts) -> np.ndarray:
        """Linear interpolation of node values (N,) or (N, k) at points (M, 2); nan outside the domain."""
        values = np.asarray(values, float)
        flat = values.reshape(len(self.xy), -1)
        pts = np.asarray(pts, float).reshape(-1, 2)
        s = self.tri.find_simplex(pts)
        ok = s >= 0
        ok[ok] = self.inside[s[ok]]
        out = np.full((len(pts), flat.shape[1]), np.nan)
        if ok.any():
            T = self.tri.transform[s[ok]]
            b = np.einsum("ijk,ik->ij", T[:, :2], pts[ok] - T[:, 2])
            bary = np.column_stack([b, 1 - b.sum(axis=1)])
            out[ok] = np.einsum("ij,ijk->ik", bary, flat[self.tri.simplices[s[ok]]])
        return out if values.ndim > 1 else out[:, 0]

    def sample(self, x, y) -> dict:
        """Interpolate rho, T, u, v at points (x, y). Returns arrays of the same shape.
        Points outside the domain take the nearest node's values."""
        x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
        pts = np.column_stack([x.ravel(), y.ravel()])
        vals = self.interpolate(np.column_stack([self.nodes[k] for k in ("rho", "T", "u", "v")]), pts)
        bad = np.isnan(vals[:, 0])
        if bad.any():
            vals[bad] = self._near(pts[bad])
        return {k: vals[:, i].reshape(x.shape) for i, k in enumerate(("rho", "T", "u", "v"))}
