"""Hierarchical SPARTA grid, refined so every cell is smaller than the local mean free path.

Level 1 is a uniform nx x ny grid over the box. A cell is split 2x2 into
level k+1 while its size exceeds lambda_min / cells_per_mfp, where lambda_min
is the smallest mean free path found in the cell. That minimum is taken over
the cell centre and corners (interpolated) and over every SU2 node inside the
cell, so features narrower than the cell are not missed.

Cells are written as a SPARTA read_grid file, using SPARTA's cell ID encoding
(see Grid::id_find_child in grid_id.cpp): the level-k child index (1-based,
ix fastest) is shifted left by the total bits of all parent levels.
"""
from dataclasses import dataclass, field

import numpy as np

from lib.common import die

# SPARTA's default build (SPARTA_BIG) uses 32-bit unsigned cellint, but read_grid
# parses IDs with atoi, so stay within 31 bits. A -DSPARTA_BIGBIG build allows 64.
ID_BITS = 31

# sample offsets within a cell, in cell widths: centre first, then 4 corners
_OX = np.array([0.5, 0.0, 1.0, 0.0, 1.0])
_OY = np.array([0.5, 0.0, 0.0, 1.0, 1.0])


def id_bits(n: int) -> int:
    """Bits to store child indices 1..n (port of Grid::id_bits)."""
    nstore, nbits = 1, 1
    while nstore < n:
        nstore = 2 * nstore + 1
        nbits += 1
    return nbits


def max_depth(nx: int, ny: int, max_bits: int = ID_BITS) -> int:
    """Deepest level reachable from an nx x ny level-1 grid with 2x2 refinement."""
    return 1 + (max_bits - id_bits(nx * ny)) // id_bits(4)


@dataclass
class Grid:
    box: tuple       # (xlo, xhi, ylo, yhi)
    nx: int          # level-1 cells in x
    ny: int          # level-1 cells in y
    level: np.ndarray = None  # per leaf cell: level (1 = coarsest)
    i: np.ndarray = None      # per leaf: x index in the uniform grid of its level
    j: np.ndarray = None      # per leaf: y index
    data: dict = field(default_factory=dict)  # per leaf: lam, tau, speed, n, T
    unresolved: int = 0       # leaves still too big at the depth limit
    worst: float = 1.0        # largest (cell size * cells_per_mfp / lambda) among them

    @property
    def nlevels(self) -> int:
        return int(self.level.max())

    def size(self):
        """(dx, dy) of each leaf cell."""
        f = 2.0 ** (self.level - 1)
        xlo, xhi, ylo, yhi = self.box
        return (xhi - xlo) / (self.nx * f), (yhi - ylo) / (self.ny * f)

    def centres(self):
        dx, dy = self.size()
        return self.box[0] + (self.i + 0.5) * dx, self.box[2] + (self.j + 0.5) * dy

    def ids(self) -> np.ndarray:
        """SPARTA cell IDs of the leaves."""
        b1, b2 = id_bits(self.nx * self.ny), id_bits(4)
        lev = self.level
        i1, j1 = self.i >> (lev - 1), self.j >> (lev - 1)
        ids = (j1 * self.nx + i1 + 1).astype(np.uint64)
        for k in range(2, self.nlevels + 1):
            m = lev >= k
            s = np.where(m, lev - k, 0)
            ichild = ((self.j >> s) & 1) * 2 + ((self.i >> s) & 1) + 1
            ids[m] |= ichild[m].astype(np.uint64) << np.uint64(b1 + b2 * (k - 2))
        return ids

    def write(self, path, custom: dict):
        """Write a read_grid file. `custom` = {name: per-leaf array}, appended as columns."""
        ids = self.ids()
        cols = [ids.astype(np.int64)] + list(custom.values())
        fmt = ["%d"] + ["%.6g"] * len(custom)
        header = [
            "# SPARTA grid file written by SpartU2 sparta/setup.py",
            f"# custom columns: {' '.join(custom)}" if custom else "#",
            "",
            f"{len(ids)} cells",
            f"{self.nlevels} levels",
            f"{self.nx} {self.ny} 1 level-1",
            *[f"2 2 1 level-{k}" for k in range(2, self.nlevels + 1)],
            "",
            "Cells",
            "",
        ]
        with open(path, "w") as f:
            f.write("\n".join(header) + "\n")
            np.savetxt(f, np.column_stack(cols), fmt=fmt)


def _group(key, values, reduce):
    """Reduce `values` over equal `key`s. Returns (unique keys, reduced values)."""
    order = np.argsort(key, kind="stable")
    k = key[order]
    uniq, first = np.unique(k, return_index=True)
    return uniq, reduce.reduceat(values[order], first)


def _lookup(uniq, red, keys, fill):
    """Reduced value for each of `keys`, or `fill` where the key has no nodes."""
    if not len(uniq):
        return np.full(len(keys), fill)
    pos = np.minimum(np.searchsorted(uniq, keys), len(uniq) - 1)
    return np.where(uniq[pos] == keys, red[pos], fill)


def build(box, nx, ny, sampler, nodes, cells_per_mfp, depth_limit) -> Grid:
    """Refine an nx x ny level-1 grid over box down to at most depth_limit levels.

    sampler(x, y) -> dict with arrays lam, tau, speed, n, T at those points.
    nodes = dict x, y, lam, tau, speed for SU2 nodes inside the box.
    """
    xlo, xhi, ylo, yhi = box
    I, J = (a.ravel() for a in np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij"))
    out = {k: [] for k in ("level", "i", "j", "lam", "tau", "speed", "n", "T")}
    unresolved, worst = 0, 1.0

    for level in range(1, depth_limit + 1):
        f = 2 ** (level - 1)
        dx, dy = (xhi - xlo) / (nx * f), (yhi - ylo) / (ny * f)

        s = sampler(xlo + (I[:, None] + _OX) * dx, ylo + (J[:, None] + _OY) * dy)
        lam, tau, speed = s["lam"].min(1), s["tau"].min(1), s["speed"].max(1)

        # fold in SU2 nodes that fall inside each cell
        nkey = (np.clip(((nodes["y"] - ylo) / dy).astype(np.int64), 0, ny * f - 1) * (nx * f)
                + np.clip(((nodes["x"] - xlo) / dx).astype(np.int64), 0, nx * f - 1))
        ckey = J.astype(np.int64) * (nx * f) + I
        for name, red, fill in (("lam", np.minimum, np.inf), ("tau", np.minimum, np.inf),
                                ("speed", np.maximum, 0.0)):
            u, r = _group(nkey, nodes[name], red)
            cur = {"lam": lam, "tau": tau, "speed": speed}[name]
            cur[:] = red(cur, _lookup(u, r, ckey, fill))

        ratio = max(dx, dy) * cells_per_mfp / lam
        split = ratio > 1
        if level == depth_limit:
            unresolved = int(split.sum())
            worst = float(ratio[split].max()) if unresolved else 1.0
            split[:] = False

        keep = ~split
        for k, v in (("level", np.full(keep.sum(), level)), ("i", I[keep]), ("j", J[keep]),
                     ("lam", lam[keep]), ("tau", tau[keep]), ("speed", speed[keep]),
                     ("n", s["n"][keep, 0]), ("T", s["T"][keep, 0])):
            out[k].append(v)

        if not split.any():
            break
        # children of split cells: (2i+a, 2j+b)
        I = (2 * I[split][:, None] + np.array([0, 1, 0, 1])).ravel()
        J = (2 * J[split][:, None] + np.array([0, 0, 1, 1])).ravel()

    cat = {k: np.concatenate(v) for k, v in out.items()}
    if not len(cat["level"]):
        die("grid has no cells")
    return Grid(box, nx, ny, cat.pop("level"), cat.pop("i"), cat.pop("j"), cat,
                unresolved, worst)
