"""Read an SU2 solution (ParaView .vtu) and sample it at arbitrary points.

Only the SU2 output written by su2/setup.py is supported: binary appended
data, raw encoding, UInt64 block headers. In the axisymmetric case, x is the
axis and y is the radius.
"""
import re
from pathlib import Path

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

from lib.common import die

_DTYPES = {"Float32": "<f4", "Float64": "<f8", "Int32": "<i4", "Int64": "<i8", "UInt8": "u1"}
_ARRAY = re.compile(rb'<DataArray type="(\w+)" Name="(\w*)" NumberOfComponents= ?"(\d+)" '
                    rb'offset="(\d+)" format="appended"/>')


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
        nbytes = int(np.frombuffer(raw, "<u8", 1, pos)[0])
        a = np.frombuffer(raw, _DTYPES[typ.decode()], nbytes // np.dtype(_DTYPES[typ.decode()]).itemsize,
                          pos + 8)
        ncomp = int(ncomp)
        arrays[name.decode() or "Points"] = a.reshape(-1, ncomp) if ncomp > 1 else a
    return arrays


class Su2Field:
    """SU2 flow field, linearly interpolated, restricted to the region x >= xmin.

    Attributes after sampling: rho [kg/m3], T [K], u, v [m/s] (axial, radial).
    """

    FIELDS = ("Density", "Temperature")

    def __init__(self, vtu_path, xmin=-np.inf):
        a = read_vtu(vtu_path)
        for f in self.FIELDS + ("Velocity",):
            if f not in a:
                die(f"{vtu_path}: no '{f}' field (VOLUME_OUTPUT must include PRIMITIVE)")
        xy = a["Points"][:, :2].astype(float)
        keep = xy[:, 0] >= xmin
        self.xy = xy[keep]
        vals = np.column_stack([a["Density"][keep], a["Temperature"][keep],
                                a["Velocity"][keep, 0], a["Velocity"][keep, 1]]).astype(float)
        self.nodes = dict(zip(("rho", "T", "u", "v"), vals.T))
        self._lin = LinearNDInterpolator(self.xy, vals)  # Delaunay of the SU2 nodes
        self._near = NearestNDInterpolator(self.xy, vals)  # fallback just outside the hull

    def sample(self, x, y) -> dict:
        """Interpolate rho, T, u, v at points (x, y). Returns arrays of the same shape."""
        x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
        pts = np.column_stack([x.ravel(), y.ravel()])
        vals = self._lin(pts)
        bad = np.isnan(vals[:, 0])
        if bad.any():
            vals[bad] = self._near(pts[bad])
        return {k: vals[:, i].reshape(x.shape) for i, k in enumerate(("rho", "T", "u", "v"))}
