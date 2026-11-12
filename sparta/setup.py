#!/usr/bin/env python3
"""Build a SPARTA run directory: DSMC domain and a grid sized from the SU2 solution.

Usage: python sparta/setup.py [CASE.toml] [--runs RUNS_DIR] [--su2 FLOW.vtu]
Needs a finished SU2 run (RUNS_DIR/<name>/su2/flow.vtu by default) and writes
RUNS_DIR/<name>/sparta/.

The DSMC domain starts at the coupling plane x = x_couple_D * D, downstream of
the nozzle exit, and fills the rest of the chamber. Its grid is refined so
that every cell is smaller than the local mean free path / cells_per_mfp.
The SU2 state on the plane becomes the DSMC inflow (inflow.sparta, see
sparta/inflow.py); in.sparta runs the coupled DSMC case, in.grid only loads the grid.
"""
import argparse
import importlib
import shutil
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):  # allow `python sparta/setup.py` as well as `-m sparta.setup`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.common import die, get, load_case, log
from lib.su2_field import Su2Field
from sparta import grid as sgrid
from sparta import inflow
from sparta.gas import DATA, load_gas


# Breakdown-parameter implementation, chosen by [sparta] breakdown_module. Any module
# with these names works, so different versions can sit side by side in handoff/.
DEFAULT_BREAKDOWN = "handoff.breakdown_gll"
BREAKDOWN_API = ("KN_BREAKDOWN", "axis_knudsen", "kn_at", "first_breakdown")


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


def _auto(case, key, default="auto"):
    """[sparta] key that may be a number or "auto"; returns None for auto."""
    v = get(case, "sparta", key, default)
    return None if v == "auto" else float(v)


def setup(case_path, runs_dir="runs", su2_vtu=None) -> Path:
    case_path = Path(case_path).resolve()
    case = load_case(case_path)
    name = get(case, "case", "name")
    if (get(case, "geometry", "dim"), get(case, "geometry", "symmetry")) != (2, "axial"):
        die("only dim=2, symmetry=axial is implemented so far")

    D = get(case, "geometry", "throat_diam_mm") * 1e-3
    Lx = get(case, "geometry", "chamb_length_cm") * 1e-2
    Rext = get(case, "geometry", "chamb_height_cm") * 1e-2
    x_c = get(case, "sparta", "x_couple_D") * D
    nx, ny = get(case, "sparta", "base_grid")[:2]
    k = get(case, "sparta", "cells_per_mfp", 3)
    ppc_target = get(case, "sparta", "ppc_target", 20)
    tau_frac = get(case, "sparta", "dt_tau_frac", 0.2)
    cell_frac = get(case, "sparta", "dt_cell_frac", 0.5)
    if not 0 <= x_c < Lx:
        die(f"x_couple_D puts the coupling plane at {x_c*1e3:.3g} mm, outside the chamber")

    vtu = Path(su2_vtu) if su2_vtu else Path(runs_dir) / name / "su2" / "flow.vtu"
    if not vtu.is_file():
        die(f"{vtu} not found: run SU2 first (su2/setup.py, su2/run.py) or pass --su2")

    gas = load_gas(get(case, "gas", "species_file"))
    log(f"reading SU2 solution {vtu}")
    field = Su2Field(vtu, xmin=-1e-9)  # chamber only (x >= exit plane)

    def props(rho, T, u, v):
        n = rho / gas.mass
        lam = gas.mean_free_path(n, T)
        cbar = gas.mean_speed(T)
        return dict(n=n, T=T, lam=lam, tau=lam / cbar, speed=np.hypot(u, v) + cbar)

    def sampler(x, y):
        return props(**field.sample(x, y))

    box = (x_c, Lx, 0.0, Rext)
    inside = field.xy[:, 0] >= x_c
    nodes = props(**{q: a[inside] for q, a in field.nodes.items()})
    nodes.update(x=field.xy[inside, 0], y=field.xy[inside, 1])

    depth = sgrid.max_depth(nx, ny)
    if "max_levels" in case["sparta"]:
        depth = min(depth, get(case, "sparta", "max_levels"))
    log(f"refining {nx} x {ny} base grid to dx <= lambda/{k}, at most {depth} levels")
    g = sgrid.build(box, nx, ny, sampler, nodes, k, depth)

    # ---- particles and timestep ---------------------------------------------
    dx, dy = g.size()
    d = g.data
    # with `global weight cell radius`, particles per cell = 2 pi n dy / fnum
    ndy = 2 * np.pi * d["n"] * dy
    ref = g.level > 1 if (g.level > 1).any() else np.ones_like(g.level, bool)
    fnum_ppc = np.percentile(ndy[ref], 10) / ppc_target
    # memory budget: never more than max_particles at steady state (~100-200 bytes each)
    max_particles = float(get(case, "sparta", "max_particles", 5e6))
    fnum_mem = ndy.sum() / max_particles
    fnum_rec = max(fnum_ppc, fnum_mem)
    dt_rec = float(np.min(np.minimum(tau_frac * d["tau"],
                                     cell_frac * np.minimum(dx, dy) / d["speed"])))
    fnum = _auto(case, "fnum") or fnum_rec
    dt = _auto(case, "timestep_s") or dt_rec
    ppc = ndy / fnum

    # ---- continuum breakdown along the axis ---------------------------------
    breakdown = load_breakdown(get(case, "sparta", "breakdown_module", DEFAULT_BREAKDOWN))
    kn_thr = get(case, "sparta", "kn_breakdown", breakdown.KN_BREAKDOWN)
    xa, kn = breakdown.axis_knudsen(field, gas)
    kn_c = breakdown.kn_at(xa, kn, x_c)
    x_break = breakdown.first_breakdown(xa, kn, kn_thr)

    # ---- inflow at the coupling plane ---------------------------------------
    plane = inflow.sample_plane(field, gas, x_c, inflow.plane_radii(field, x_c, Rext))
    flows = inflow.plane_summary(plane, gas.mass)
    run_steps = int(get(case, "sparta", "run_steps", 1000))
    avg_steps = int(get(case, "sparta", "average_steps", max(run_steps // 2, 1)))
    if not 0 < avg_steps <= run_steps:
        die("sparta.average_steps must be between 1 and run_steps")

    # ---- write run directory ------------------------------------------------
    run_dir = Path(runs_dir) / name / "sparta"
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(case_path, run_dir / "case.toml")
    stem = Path(get(case, "gas", "species_file")).stem
    for ext in ("species", "vss"):
        shutil.copy(DATA / f"{stem}.{ext}", run_dir / f"{stem}.{ext}")
    g.write(run_dir / "grid.sparta", {"lambda": d["lam"], "su2_nrho": d["n"]})
    (run_dir / "in.grid").write_text(in_grid(case_path.name, box, gas.species, stem, fnum, dt))
    inflow.write_inflow(run_dir / "inflow.sparta", plane, rot=gas.rotdof > 0, vib=gas.vibdof > 0,
                        header=f"Generated by sparta/setup.py from {case_path.name} ({vtu})\n"
                               f"SU2 state on x = {x_c:.10g} m: nrho [1/m3], v [m/s], T [K]")
    (run_dir / "in.sparta").write_text(in_run(case_path.name, box, gas.species, stem, fnum, dt,
                                              run_steps, avg_steps))

    # ---- report -------------------------------------------------------------
    per_level = np.bincount(g.level)[1:]
    rep = [
        f"case {name}: {gas.species}, D = {D*1e3:g} mm",
        f"domain x = [{x_c*1e3:.4g}, {Lx*1e3:.4g}] mm (coupling plane at {x_c/D:g} D), "
        f"r = [0, {Rext*1e3:.4g}] mm",
        f"grid: {len(g.level)} cells over {g.nlevels} levels "
        f"({', '.join(f'L{i+1}:{c}' for i, c in enumerate(per_level))})",
        f"  cell size {np.min(dx)*1e6:.3g} um (finest) to {np.max(dx)*1e6:.3g} um (base)",
        f"  mean free path in domain {d['lam'].min()*1e6:.3g} um to {d['lam'].max()*1e3:.3g} mm",
        f"fnum     {fnum:.3g}" + ("" if fnum == fnum_rec else f"  (recommended {fnum_rec:.3g})")
        + (f"  [raised from {fnum_ppc:.3g} to keep <= {max_particles:.3g} particles]"
           if fnum_mem > fnum_ppc else ""),
        f"  particles/cell: refined cells median {np.median(ppc[ref]):.3g}, "
        f"10th pct {np.percentile(ppc[ref], 10):.3g}; all cells min {ppc.min():.3g}",
        f"  steady-state particles ~ {ppc.sum():.3g} (~{ppc.sum() * 200 / 1e9:.2g} GB at 200 B each)",
        f"timestep {dt:.3g} s" + ("" if dt == dt_rec else f"  (recommended {dt_rec:.3g})"),
        f"inflow: {len(plane['r'])} points on the plane; SU2 net mass flow {flows['mdot_net']:.4g} kg/s, "
        f"emitted (one-sided) {flows['mdot_inward']:.4g} kg/s",
        f"  plane: n {plane['n'].min():.3g} to {plane['n'].max():.3g} /m3, "
        f"T {plane['T'].min():.3g} to {plane['T'].max():.3g} K, u {plane['u'].min():.3g} to {plane['u'].max():.3g} m/s",
        f"run: {run_steps} steps ({run_steps * dt:.3g} s), averaged over the last {avg_steps}; "
        f"one transit of the box at the max inflow speed takes {(Lx - x_c) / max(plane['u'].max(), 1.0) / dt:.3g} steps",
        f"axis Kn_GLL at coupling plane: {kn_c:.3g} "
        + (f"(continuum breaks down at x ~ {x_break/D:.3g} D)" if x_break is not None
           else "(below breakdown everywhere on the axis)"),
    ]
    for line in rep:
        log(line)
    warn = []
    if g.unresolved:
        warn.append(f"{g.unresolved} cells still exceed lambda/{k} at the level limit "
                    f"(worst by {g.worst:.2g}x): coarsen base_grid, or move the coupling "
                    f"plane downstream")
    if x_break is not None and x_c >= x_break:
        warn.append(f"axis Kn_GLL first exceeds {kn_thr} at {x_break/D:.3g} D, upstream of "
                    f"the coupling plane: SU2 is not reliable there, use x_couple_D < "
                    f"{x_break/D:.3g}")
    if ppc.sum() > 1.01 * max_particles:
        warn.append(f"fnum = {fnum:.3g} gives ~{ppc.sum():.3g} particles, over max_particles = "
                    f"{max_particles:.3g}: this may not fit in memory")
    if fnum_mem > fnum_ppc:
        warn.append(f"the particle budget (max_particles = {max_particles:.3g}) leaves the refined "
                    f"cells {np.percentile(ppc[ref], 10):.2g} particles (10th pct) instead of "
                    f"{ppc_target}: results will be noisier; shrink the domain or raise max_particles")
    if fnum > 2 * fnum_rec or dt > 2 * dt_rec:
        warn.append("fnum/timestep are well above the recommended values")
    for w in warn:
        log(f"WARNING: {w}")
    (run_dir / "setup.log").write_text("\n".join(rep + [f"WARNING: {w}" for w in warn]) + "\n")
    log(f"wrote {run_dir}")
    return run_dir


def in_grid(case_name, box, species, stem, fnum, dt) -> str:
    xlo, xhi, ylo, yhi = box
    return f"""\
# Generated by sparta/setup.py from {case_name} -- edit the case file, not this.
# Mesh check: loads the DSMC domain and grid, no flow yet. The inflow from SU2 at
# the coupling plane (xlo) still needs to be added.
#   mpirun -n N spa_mpi -in in.grid    ->  grid.dump (cells, SU2 mean free path)

seed            12345
dimension       2

# xlo = coupling plane (inflow + outflow), ylo = axis, yhi/xhi = chamber outflow
boundary        o ao p
create_box      {xlo:.10g} {xhi:.10g} {ylo:.10g} {yhi:.10g} -0.5 0.5
read_grid       grid.sparta custom lambda float 0 custom su2_nrho float 0
balance_grid    rcb cell

global          weight cell radius
global          fnum {fnum:.4g}
species         {stem}.species {species}
mixture         gas {species}
collide         vss gas {stem}.vss
timestep        {dt:.4g}

dump            cells grid all 1 grid.dump id xlo ylo xhi yhi g_lambda g_su2_nrho
run             0
"""


def in_run(case_name, box, species, stem, fnum, dt, run_steps, avg_steps) -> str:
    """Coupled DSMC run: SU2 inflow on xlo, flow field averaged over the last avg_steps."""
    xlo, xhi, ylo, yhi = box
    return f"""\
# Generated by sparta/setup.py from {case_name} -- edit the case file, not this.
# DSMC of the chamber downstream of the coupling plane (xlo), fed by the SU2 state there.
#   mpirun -n N spa_mpi -in in.sparta   ->  log.sparta, flow.<step>.dump

seed            12345
dimension       2

# xlo = coupling plane (inflow + outflow), ylo = axis, yhi/xhi = chamber outflow
boundary        o ao p
create_box      {xlo:.10g} {xhi:.10g} {ylo:.10g} {yhi:.10g} -0.5 0.5
read_grid       grid.sparta custom lambda float 0 custom su2_nrho float 0
balance_grid    rcb cell

global          weight cell radius
global          fnum {fnum:.4g}
species         {stem}.species {species}
mixture         gas {species}
collide         vss gas {stem}.vss
timestep        {dt:.4g}

fix             inflow emit/face/file gas xlo inflow.sparta XLO

compute         g grid all gas nrho massrho u v
compute         th thermal/grid all gas temp
fix             avg ave/grid all 1 {avg_steps} {run_steps} c_g[*] c_th[*]
dump            flow grid all {run_steps} flow.*.dump id xc yc f_avg[1] f_avg[2] f_avg[3] f_avg[4] f_avg[5] g_su2_nrho g_lambda

stats           {max(run_steps // 50, 1)}
stats_style     step cpu np nattempt ncoll
fix             lb balance {max(run_steps // 20, 1)} 1.1 rcb part
run             {run_steps}
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("case", nargs="?", default="config.toml", help="case file (default: config.toml)")
    ap.add_argument("--runs", default="runs", help="runs directory (default: ./runs)")
    ap.add_argument("--su2", help="SU2 flow.vtu (default: RUNS/<name>/su2/flow.vtu)")
    args = ap.parse_args(argv)
    setup(args.case, args.runs, args.su2)


if __name__ == "__main__":
    main()
