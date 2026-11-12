#!/usr/bin/env python3
"""Continuum breakdown parameter, used to place the NS-DSMC coupling plane.

The gradient-length local Knudsen number (Boyd, Chen & Chandler 1995) is

    Kn_GLL = lambda * |d rho / dx| / rho

where lambda is the local mean free path. Navier-Stokes is trusted where
Kn_GLL < KN_BREAKDOWN, and the coupling plane must sit upstream of the first
point where that fails.

The core functions take plain arrays, so they can be tested without an SU2
file. axis_profile() and axis_knudsen() connect them to an SU2 solution.

Usage: python handoff/breakdown_gll.py [CASE.toml] [--runs RUNS_DIR] [--su2 FLOW.vtu]
prints the axis Kn_GLL profile and where breakdown happens.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):  # allow `python handoff/breakdown_gll.py` as well as `-m`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

KN_BREAKDOWN = 0.05  # default threshold (Boyd, Chen & Chandler 1995)


def kn_gll(x, rho, T, gas):
    """Kn_GLL along a line, from density rho [kg/m3] and temperature T [K] at points x [m].

    x must be strictly increasing. Only the gradient along x is used, which is the
    full gradient on the symmetry axis.
    """
    x, rho, T = (np.asarray(a, float) for a in (x, rho, T))
    if x.ndim != 1 or len(x) < 2 or not np.all(np.diff(x) > 0):
        raise ValueError("x must be a 1-D, strictly increasing array with at least 2 points")
    lam = gas.mean_free_path(rho / gas.mass, T)
    return lam * np.abs(np.gradient(rho, x)) / rho


def first_breakdown(x, kn, threshold=KN_BREAKDOWN):
    """First x where kn > threshold, or None if it never does."""
    over = np.nonzero(np.asarray(kn) > threshold)[0]
    return float(x[over[0]]) if len(over) else None


def kn_at(x, kn, x0):
    """Kn_GLL interpolated at x0."""
    return float(np.interp(x0, x, kn))


def axis_profile(field, tol=1e-12):
    """(x, rho, T) at the SU2 nodes on the axis (|y| < tol), sorted by x, duplicates removed.

    Uses the nodes themselves: differentiating the linear interpolant is noisy,
    because its slope jumps at every SU2 cell edge.
    """
    on = np.abs(field.xy[:, 1]) < tol
    x, idx = np.unique(field.xy[on, 0], return_index=True)
    return x, field.nodes["rho"][on][idx], field.nodes["T"][on][idx]


def axis_knudsen(field, gas):
    """(x, Kn_GLL) on the axis of an SU2 solution."""
    x, rho, T = axis_profile(field)
    return x, kn_gll(x, rho, T, gas)


def main(argv=None):
    from lib.common import get, load_case, log
    from lib.su2_field import Su2Field
    from sparta.gas import load_gas

    ap = argparse.ArgumentParser(description="Print the axis Kn_GLL profile of an SU2 solution.")
    ap.add_argument("case", nargs="?", default="config.toml", help="case file (default: config.toml)")
    ap.add_argument("--runs", default="runs", help="runs directory (default: ./runs)")
    ap.add_argument("--su2", help="SU2 flow.vtu (default: RUNS/<name>/su2/flow.vtu)")
    args = ap.parse_args(argv)

    case = load_case(args.case)
    D = get(case, "geometry", "throat_diam_mm") * 1e-3
    thr = get(case, "sparta", "kn_breakdown", KN_BREAKDOWN)
    vtu = args.su2 or Path(args.runs) / get(case, "case", "name") / "su2" / "flow.vtu"
    gas = load_gas(get(case, "gas", "species_file"))
    x, kn = axis_knudsen(Su2Field(vtu, xmin=-1e-9), gas)

    print(f"{'x/D':>8} {'Kn_GLL':>10}")
    for xd in (0.5, 1, 2, 3, 4, 5, 6, 8, 10, 15, 20):
        if xd * D <= x[-1]:
            print(f"{xd:8g} {kn_at(x, kn, xd * D):10.3g}")
    xb = first_breakdown(x, kn, thr)
    log(f"Kn_GLL first exceeds {thr} at x = {xb / D:.3g} D" if xb is not None
        else f"Kn_GLL stays below {thr} on the axis")


if __name__ == "__main__":
    main()
