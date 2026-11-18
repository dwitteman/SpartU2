#!/usr/bin/env python3
"""SU2 -> SPARTA handoff: place the coupling plane and write the DSMC inflow.

Usage: python handoff/handoff.py CASE.toml [--runs RUNS_DIR] [--su2 FLOW.vtu]

Reads the SU2 solution (RUNS/<name>/su2/flow.vtu by default) and writes RUNS/<name>/handoff/:
  handoff.toml   coupling plane, Kn_GLL there, mass flows, the SU2 file used (read by sparta/setup.py)
  inflow.sparta  SU2 n, v, T on the plane, for SPARTA's fix emit/face/file
  kn_profile.csv x [m], x/D and the breakdown module's Kn along x, to plot or check by hand
  report.txt     sanity-check report: Kn along the jet and across the plane, mass flows, checks
  report.png     the same as plots (skipped if matplotlib is missing)

The coupling plane comes from [sparta] x_couple_D:
  "auto" (default)  couple_margin_D throat diameters (default 1) upstream of the first point
                    where the breakdown module's Kn exceeds kn_breakdown; stops if that point is
                    inside the nozzle or capillary (the breakdown profile covers them too);
  a number          forces the plane there, in throat diameters past the nozzle exit.
The module is [sparta] breakdown_module (handoff.breakdown_gll: axis only; handoff.breakdown_2d:
Boyd's full Kn_GLL over each plane); its options go in a [breakdown] section.
"""
import argparse
import importlib
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):  # allow `python handoff/handoff.py` as well as `-m handoff.handoff`
    # replace (not prepend to) the script's folder, or this file would shadow the handoff package
    sys.path[0] = str(Path(__file__).resolve().parents[1])

from handoff import report
from lib.common import die, get, load_case, log
from lib.su2_field import Su2Field
from sparta import inflow
from sparta.gas import load_gas

# Breakdown-parameter implementation, chosen by [sparta] breakdown_module. Any module
# with these names works, so different versions can sit side by side in handoff/.
DEFAULT_BREAKDOWN = "handoff.breakdown_gll"
BREAKDOWN_API = ("KN_BREAKDOWN", "knudsen_profile", "kn_at", "first_breakdown")


def load_breakdown(name: str):
    """Import a breakdown module by dotted name and check it has the expected functions."""
    try:
        mod = importlib.import_module(name)
    except ModuleNotFoundError as e:
        die(f"breakdown_module '{name}' not found ({e})")
    missing = [a for a in BREAKDOWN_API if not hasattr(mod, a)]
    if missing:
        die(f"breakdown_module '{name}' is missing: {', '.join(missing)}")
    return mod


def choose_plane(x_couple_D, D, x_break, margin_D=1.0, x_max=np.inf):
    """Coupling plane x [m] (0 = nozzle exit).

    x_couple_D: "auto" or a number of throat diameters D [m]; x_break: first breakdown point
    [m] or None, negative if it is inside the nozzle or capillary. "auto" puts the plane margin_D
    diameters upstream of x_break. The plane must be in the chamber (x >= 0): the DSMC domain
    starts at the plane and has no nozzle walls."""
    if x_couple_D == "auto":
        if x_break is None:
            die("Kn_GLL stays below the threshold on the whole SU2 axis, so there is no breakdown "
                "point to couple at: set [sparta] x_couple_D to a number to force the plane")
        x_c = x_break - margin_D * D
        if x_break < 0:
            die(f"the gas breaks down inside the nozzle, at x = {x_break / D:.3g} D "
                f"({x_break * 1e3:.4g} mm before the exit), but the DSMC domain can only start in the "
                f"chamber: force [sparta] x_couple_D (as close to the exit as you accept), and see "
                f"report.txt for how far the breakdown reaches")
        if x_c < 0:
            die(f"breakdown at {x_break / D:.3g} D is within couple_margin_D = {margin_D:g} D of the "
                f"nozzle exit: lower couple_margin_D or force x_couple_D")
    else:
        x_c = float(x_couple_D) * D
    if not 0 <= x_c < x_max:
        die(f"coupling plane x = {x_c * 1e3:.4g} mm is outside the chamber [0, {x_max * 1e3:.4g}) mm")
    return x_c


def read_handoff(run_root) -> dict:
    """handoff.toml written by handoff() in run_root/handoff/ (run_root = RUNS/<name>)."""
    f = Path(run_root) / "handoff" / "handoff.toml"
    if not f.is_file():
        die(f"{f} not found: run handoff/handoff.py first")
    return load_case(f)["handoff"]


def handoff(case_path, runs_dir="runs", su2_vtu=None) -> Path:
    """Write RUNS/<name>/handoff/ from the SU2 solution. Returns that directory."""
    case_path = Path(case_path).resolve()
    case = load_case(case_path)
    name = get(case, "case", "name")
    D = get(case, "geometry", "throat_diam_mm") * 1e-3
    Lx = get(case, "geometry", "chamb_length_cm") * 1e-2
    Rext = get(case, "geometry", "chamb_height_cm") * 1e-2

    vtu = (Path(su2_vtu) if su2_vtu else Path(runs_dir) / name / "su2" / "flow.vtu").resolve()
    if not vtu.is_file():
        die(f"{vtu} not found: run SU2 first or pass --su2")
    gas = load_gas(get(case, "gas", "species_file"))
    log(f"reading SU2 solution {vtu}")
    field = Su2Field(vtu)  # whole domain: nozzle/capillary (x < 0) and chamber

    # ---- breakdown profile and the coupling plane -------------------------------
    module = get(case, "sparta", "breakdown_module", DEFAULT_BREAKDOWN)
    bd = load_breakdown(module)
    thr = get(case, "sparta", "kn_breakdown", bd.KN_BREAKDOWN)
    opts = case.get("breakdown", {})
    try:
        xa, kn = bd.knudsen_profile(field, gas, **opts)
    except TypeError as e:
        die(f"[breakdown] options {opts} don't fit {module}: {e}")
    x_break = bd.first_breakdown(xa, kn, thr)
    mode = get(case, "sparta", "x_couple_D", "auto")
    margin = get(case, "sparta", "couple_margin_D", 1.0)
    x_c = choose_plane(mode, D, x_break, margin, Lx)
    kn_c = bd.kn_at(xa, kn, x_c)

    # ---- SU2 state on the plane ------------------------------------------------
    plane = inflow.sample_plane(field, gas, x_c, inflow.plane_radii(field, x_c, Rext))
    flows = inflow.plane_summary(plane, gas.mass)

    out = Path(runs_dir) / name / "handoff"
    out.mkdir(parents=True, exist_ok=True)
    inflow.write_inflow(out / "inflow.sparta", plane, rot=gas.rotdof > 0, vib=gas.vibdof > 0,
                        header=f"Generated by handoff/handoff.py from {case_path.name} ({vtu})\n"
                               f"SU2 state on x = {x_c:.10g} m: nrho [1/m3], v [m/s], T [K]")
    np.savetxt(out / "kn_profile.csv", np.column_stack([xa, xa / D, kn]), delimiter=",",
               header="x_m,x_D,kn", comments="", fmt="%.6g")
    xb = "" if x_break is None else f"x_breakdown_m = {x_break:.10g}\n"
    (out / "handoff.toml").write_text(
        f"# Written by handoff/handoff.py from {case_path.name}; read by sparta/setup.py\n"
        f"[handoff]\n"
        f'su2_vtu = "{vtu}"\n'
        f'mode = "{"auto" if mode == "auto" else "forced"}"\n'
        f"x_couple_m = {x_c:.10g}\n"
        f"x_couple_D = {x_c / D:.6g}\n"
        f"kn_at_plane = {kn_c:.6g}\n"
        f"kn_threshold = {thr:g}\n"
        f"{xb}"
        f"mdot_net = {flows['mdot_net']:.6g}       # kg/s through the plane (SU2)\n"
        f"mdot_inward = {flows['mdot_inward']:.6g}    # kg/s emitted by SPARTA (one-sided flux)\n"
        f"u_max = {float(plane['u'].max()):.6g}\n")

    log(f"{module}: Kn first exceeds {thr} at "
        + (f"x = {x_break / D:.3g} D" + (" (inside the nozzle)" if x_break < 0 else "")
           if x_break is not None else "no point in the SU2 domain"))
    log(f"coupling plane ({'auto' if mode == 'auto' else 'forced'}): x = {x_c * 1e3:.4g} mm = "
        f"{x_c / D:.3g} D, Kn_GLL there {kn_c:.3g}")
    log(f"inflow: {len(plane['r'])} points; SU2 net mass flow {flows['mdot_net']:.4g} kg/s, "
        f"emitted (one-sided) {flows['mdot_inward']:.4g} kg/s")
    if x_break is not None and x_c >= x_break:
        log(f"WARNING: the forced plane is past the breakdown point ({x_break / D:.3g} D): "
            f"SU2 is not reliable there")

    # ---- sanity-check report ---------------------------------------------------
    info = dict(case=name, gas=gas.species, vtu=vtu, module=module, thr=thr, opts=opts,
                mode="auto" if mode == "auto" else "forced", D=D, x_c=x_c, x_break=x_break,
                profile=(xa, kn))
    txt = report.write(out, field, gas, info, plane, flows,
                       opts.get("mass_fraction", report.b2d.MASS_FRACTION))
    flagged = [l.strip() for l in txt.splitlines() if l.strip().startswith("[!!]")]
    for l in flagged:
        log(f"CHECK {l}")
    log(f"report: {out / 'report.txt'}" + (f" ({len(flagged)} checks flagged)" if flagged else " (all checks ok)"))
    log(f"wrote {out}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("case", help="case file")
    ap.add_argument("--runs", default="runs", help="runs directory (default: ./runs)")
    ap.add_argument("--su2", help="SU2 flow.vtu (default: RUNS/<name>/su2/flow.vtu)")
    args = ap.parse_args(argv)
    handoff(args.case, args.runs, args.su2)


if __name__ == "__main__":
    main()
