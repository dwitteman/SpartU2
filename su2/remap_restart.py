#!/usr/bin/env python3
"""Interpolate an SU2 binary restart onto a new mesh of the same domain.

Usage: python su2/remap_restart.py OLD_RESTART NEW_MESH [-o OUT]
  OLD_RESTART  restart_flow.dat from the old mesh (SU2 binary; it holds its own x, y)
  NEW_MESH     mesh.su2 to restart on
  OUT          output restart (default: restart_flow.dat next to NEW_MESH)

Every field after x, y (the conserved variables) is interpolated linearly on the Delaunay
triangulation of the old nodes. New nodes outside it (round-off on the boundary) take the
nearest old node. Use it when a remesh (e.g. su2.lip_relax_*) would otherwise throw away a
converged solution; the new mesh must cover the same domain.

The SU2 binary restart is: 5 int32 (535532, n_fields, n_points, 0, 0), n_fields names of
33 bytes, then n_points x n_fields float64, one row per mesh node in mesh.su2 order.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.common import die, log

MAGIC = 535532
NAME_LEN = 33


def read_restart(path):
    """Return (names, data[n_points, n_fields]) of an SU2 binary restart."""
    b = Path(path).read_bytes()
    head = np.frombuffer(b[:20], np.int32) if len(b) >= 20 else np.zeros(1, np.int32)
    if head[0] != MAGIC:
        die(f"{path} is not an SU2 binary restart (ASCII .csv restarts are not supported)")
    nf, npt = int(head[1]), int(head[2])
    start = 20 + NAME_LEN * nf
    if len(b) < start + 8 * nf * npt:
        die(f"{path} is truncated: {len(b)} bytes for {npt} points x {nf} fields")
    names = [b[20 + NAME_LEN * i:20 + NAME_LEN * (i + 1)].split(b"\0")[0].decode() for i in range(nf)]
    data = np.frombuffer(b, np.float64, nf * npt, start).reshape(npt, nf)
    return names, data


def write_restart(path, names, data):
    npt, nf = data.shape
    with open(path, "wb") as f:
        f.write(np.array([MAGIC, nf, npt, 0, 0], np.int32).tobytes())
        for n in names:
            f.write(n.encode()[:NAME_LEN - 1].ljust(NAME_LEN, b"\0"))
        f.write(np.ascontiguousarray(data, np.float64).tobytes())


def mesh_points(path):
    """Node coordinates of a 2D mesh.su2, in node order."""
    with open(path) as f:
        for line in f:
            if line.startswith("NDIME=") and int(line.split("=")[1]) != 2:
                die(f"{path}: only 2D meshes are supported")
            if line.startswith("NPOIN="):
                n = int(line.split("=")[1].split()[0])
                pts = np.array([next(f).split()[:2] for _ in range(n)], float)
                return pts
    die(f"{path}: no NPOIN section")


def remap(old_restart, new_mesh, out):
    names, old = read_restart(old_restart)
    if names[:2] != ["x", "y"]:
        die(f"{old_restart}: expected x, y as the first fields, got {names[:2]}")
    new_xy = mesh_points(new_mesh)
    vals = old[:, 2:]
    new = LinearNDInterpolator(old[:, :2], vals)(new_xy)
    miss = np.isnan(new).any(axis=1)
    if miss.any():
        new[miss] = NearestNDInterpolator(old[:, :2], vals)(new_xy[miss])
    write_restart(out, names, np.column_stack([new_xy, new]))
    log(f"{old_restart}: {len(old)} nodes -> {out}: {len(new_xy)} nodes "
        f"({miss.sum()} from the nearest node)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("old_restart")
    ap.add_argument("new_mesh")
    ap.add_argument("-o", "--out", help="output restart (default: restart_flow.dat beside NEW_MESH)")
    a = ap.parse_args(argv)
    out = Path(a.out) if a.out else Path(a.new_mesh).with_name("restart_flow.dat")
    if out.resolve() == Path(a.old_restart).resolve():
        die("the output would overwrite the old restart; pass -o or use a different directory")
    remap(a.old_restart, a.new_mesh, out)


if __name__ == "__main__":
    main()
