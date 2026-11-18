#!/usr/bin/env python3
"""Boyd's full gradient-length Knudsen number over the whole SU2 field.

At every SU2 node (Boyd, Chen & Chandler 1995; Wang & Boyd 2003):

    Kn_GLL = lambda * max( |grad rho| / rho,  |grad T| / T,  |grad V| / max(V, a) )

with lambda the local VSS mean free path, V the flow speed and a the speed of sound.
Dividing the speed gradient by max(V, a) instead of V keeps it finite where the gas is
nearly at rest. Gradients are taken on the SU2 mesh itself: the exact gradient of the
linear interpolant on each Delaunay triangle, averaged to the nodes weighted by area.

The coupling plane needs one number per axial station x. A plume always has broken-down
gas somewhere on any plane (the lip, the edge of the jet, the near-vacuum background), so the
station value is the Kn_GLL exceeded by a fraction `mass_fraction` (default 1%) of the mass
flow through the plane. first_breakdown(x, kn, 0.05) is then the first plane where more than
1% of the jet passes through gas with Kn_GLL > 0.05.

Options come from the case file's [breakdown] section, e.g. mass_fraction = 0.01.

Usage: python handoff/breakdown_2d.py CASE.toml [--runs RUNS_DIR] [--su2 FLOW.vtu]
prints the station profile and where breakdown happens.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import LinearNDInterpolator

if __package__ in (None, ""):  # allow `python handoff/breakdown_2d.py` as well as `-m`
    sys.path[0] = str(Path(__file__).resolve().parents[1])

from handoff.breakdown_gll import KN_BREAKDOWN, axis_profile, first_breakdown, kn_at  # noqa: F401  (API)

KB = 1.380649e-23
MASS_FRACTION = 0.01  # default fraction of the plane's mass flow allowed above the threshold
QUANTITIES = ("rho", "T", "V")  # order of the terms in the max, for reporting which one dominates


def node_gradients(xy, simplices, values):
    """Gradients at the nodes of a triangulation: (N, k, 2) for values of shape (N, k).

    Exact on each triangle for a linear field; node value = area-weighted mean over the
    triangles around it. Degenerate (zero-area) triangles are ignored."""
    values = np.asarray(values, float).reshape(len(xy), -1)
    p = xy[simplices]
    d1, d2 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    det = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]  # twice the signed area
    ok = np.abs(det) > 1e-14 * np.abs(det).max()
    s, d1, d2, det = simplices[ok], d1[ok], d2[ok], det[ok]
    f = values[s]
    df1, df2 = f[:, 1] - f[:, 0], f[:, 2] - f[:, 0]
    gx = (df1 * d2[:, 1, None] - df2 * d1[:, 1, None]) / det[:, None]
    gy = (df2 * d1[:, 0, None] - df1 * d2[:, 0, None]) / det[:, None]
    area = 0.5 * np.abs(det)
    num = np.zeros((len(xy), values.shape[1], 2))
    den = np.zeros(len(xy))
    for j in range(3):
        np.add.at(num, s[:, j], area[:, None, None] * np.stack([gx, gy], axis=-1))
        np.add.at(den, s[:, j], area)
    return num / np.maximum(den, 1e-300)[:, None, None]


def sound_speed(gas, T):
    gamma = (5 + gas.rotdof) / (3 + gas.rotdof)  # vibration frozen
    return np.sqrt(gamma * KB * np.asarray(T, float) / gas.mass)


def kn_field(field, gas):
    """(kn, dominant) at every SU2 node: Boyd's Kn_GLL, and which term (index into
    QUANTITIES) sets it."""
    nd = field.nodes
    rho, T = nd["rho"], nd["T"]
    V = np.hypot(nd["u"], nd["v"])
    g = node_gradients(field.xy, field.tri.simplices, np.column_stack([rho, T, V]))
    ratio = np.hypot(g[..., 0], g[..., 1]) / np.column_stack([rho, T, np.maximum(V, sound_speed(gas, T))])
    lam = gas.mean_free_path(rho / gas.mass, T)
    return lam * ratio.max(axis=1), ratio.argmax(axis=1)


def plane_lines(field, kn, xs, n_r=400):
    """Kn_GLL, mass-flow weights and net mass flow on radial lines x = xs[i].

    Returns r (n_r,), kn (len(xs), n_r), w (len(xs), n_r) = rho * max(u, 0) * 2 pi r dr
    (inward mass flow per point), mdot (len(xs),) net rho*u through each plane [kg/s]."""
    r_max = field.xy[:, 1].max()
    r = np.unique(np.concatenate([np.linspace(0, r_max, n_r), np.geomspace(1e-6 * r_max, r_max, n_r)]))
    interp = LinearNDInterpolator(field.tri, np.column_stack([kn, field.nodes["rho"], field.nodes["u"]]))
    X, R = np.meshgrid(xs, r, indexing="ij")
    vals = interp(np.column_stack([X.ravel(), R.ravel()])).reshape(len(xs), len(r), 3)
    vals = np.nan_to_num(vals)
    dr = np.diff(r, prepend=r[0]) / 2 + np.diff(r, append=r[-1]) / 2  # trapezoid weights
    flux = vals[..., 1] * vals[..., 2] * 2 * np.pi * r * dr
    return r, vals[..., 0], np.maximum(flux, 0), flux.sum(axis=1)


def mass_quantile(kn, w, fraction):
    """Per row: the Kn exceeded by `fraction` of the weight w (nan where there is no flow)."""
    out = np.full(len(kn), np.nan)
    for i, (k, wi) in enumerate(zip(kn, w)):
        tot = wi.sum()
        if tot <= 0:
            continue
        o = np.argsort(k)[::-1]
        cum = np.cumsum(wi[o]) / tot
        out[i] = k[o][min(np.searchsorted(cum, fraction), len(o) - 1)]
    return out


def stations(field, n_max=400):
    """Axial stations: the SU2 axis node positions (clustered like the mesh), at most n_max."""
    x = axis_profile(field)[0]
    return x[np.unique(np.linspace(0, len(x) - 1, min(n_max, len(x))).round().astype(int))]


def knudsen_profile(field, gas, mass_fraction=MASS_FRACTION):
    """(x, kn): per axial station, the Boyd Kn_GLL exceeded by mass_fraction of the mass
    flow through that plane. Same API as breakdown_gll.knudsen_profile."""
    kn, _ = kn_field(field, gas)
    xs = stations(field)
    _, kl, w, _ = plane_lines(field, kn, xs)
    q = mass_quantile(kl, w, mass_fraction)
    ok = np.isfinite(q)
    return xs[ok], q[ok]


def main(argv=None):
    from lib.common import get, load_case, log
    from lib.su2_field import Su2Field
    from sparta.gas import load_gas

    ap = argparse.ArgumentParser(description="Print the plane Kn_GLL (Boyd, 2D) profile of an SU2 solution.")
    ap.add_argument("case", help="case file")
    ap.add_argument("--runs", default="runs", help="runs directory (default: ./runs)")
    ap.add_argument("--su2", help="SU2 flow.vtu (default: RUNS/<name>/su2/flow.vtu)")
    args = ap.parse_args(argv)

    case = load_case(args.case)
    D = get(case, "geometry", "throat_diam_mm") * 1e-3
    thr = get(case, "sparta", "kn_breakdown", KN_BREAKDOWN)
    frac = case.get("breakdown", {}).get("mass_fraction", MASS_FRACTION)
    vtu = args.su2 or Path(args.runs) / get(case, "case", "name") / "su2" / "flow.vtu"
    gas = load_gas(get(case, "gas", "species_file"))
    x, kn = knudsen_profile(Su2Field(vtu, xmin=-1e-9), gas, frac)

    print(f"Kn_GLL exceeded by {frac:.0%} of the mass flow through each plane")
    print(f"{'x/D':>8} {'Kn_GLL':>10}")
    for xd in (0.5, 1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 40, 80):
        if xd * D <= x[-1]:
            print(f"{xd:8g} {kn_at(x, kn, xd * D):10.3g}")
    xb = first_breakdown(x, kn, thr)
    log(f"first plane with more than {frac:.0%} of the mass flow above Kn_GLL = {thr}: x = {xb / D:.3g} D"
        if xb is not None else f"no plane has more than {frac:.0%} of its mass flow above {thr}")


if __name__ == "__main__":
    main()
