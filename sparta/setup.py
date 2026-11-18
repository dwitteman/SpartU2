#!/usr/bin/env python3
"""Build a SPARTA run directory: DSMC domain and a grid sized from the SU2 solution.

Usage: python sparta/setup.py CASE.toml [--runs RUNS_DIR]
Needs the handoff (RUNS_DIR/<name>/handoff/, from handoff/handoff.py) and writes
RUNS_DIR/<name>/sparta/.

The DSMC domain starts at the coupling plane chosen by the handoff and fills the
rest of the chamber. If the plane is inside the nozzle or capillary (x_c < 0), the
domain includes the rest of it: the nozzle wall and the faceplate are written to
nozzle.surf, read with `read_surf ... clip`, and are diffuse walls at
[su2] wall_temp_K. Its grid is refined so that every cell is smaller than the
local mean free path / cells_per_mfp, using the SU2 solution the handoff read.
The handoff's inflow.sparta is copied in; in.sparta runs the coupled DSMC case,
in.continue continues it from a restart file, in.grid only loads the grid.
"""
import argparse
import math
import shutil
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):  # allow `python sparta/setup.py` as well as `-m sparta.setup`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from handoff.handoff import read_handoff
from lib.common import die, get, load_case, log
from lib.su2_field import Su2Field
from sparta import grid as sgrid
from sparta import surf as ssurf
from sparta.gas import DATA, load_gas


def _auto(case, key, default="auto"):
    """[sparta] key that may be a number or "auto"; returns None for auto."""
    v = get(case, "sparta", key, default)
    return None if v == "auto" else float(v)


def setup(case_path, runs_dir="runs") -> Path:
    case_path = Path(case_path).resolve()
    case = load_case(case_path)
    name = get(case, "case", "name")
    if (get(case, "geometry", "dim"), get(case, "geometry", "symmetry")) != (2, "axial"):
        die("only dim=2, symmetry=axial is implemented so far")

    D = get(case, "geometry", "throat_diam_mm") * 1e-3
    Lx = get(case, "geometry", "chamb_length_cm") * 1e-2
    Rext = get(case, "geometry", "chamb_height_cm") * 1e-2
    ho = read_handoff(Path(runs_dir) / name)
    x_c = ho["x_couple_m"]
    nx, ny = get(case, "sparta", "base_grid")[:2]
    k = get(case, "sparta", "cells_per_mfp", 3)
    ppc_target = get(case, "sparta", "ppc_target", 20)
    tau_frac = get(case, "sparta", "dt_tau_frac", 0.2)
    cell_frac = get(case, "sparta", "dt_cell_frac", 0.5)
    if not x_c < Lx:
        die(f"handoff coupling plane x = {x_c*1e3:.3g} mm is past the chamber: rerun handoff/handoff.py")

    vtu = Path(ho["su2_vtu"])
    inflow_file = Path(runs_dir) / name / "handoff" / "inflow.sparta"
    if not vtu.is_file() or not inflow_file.is_file():
        die(f"{vtu} or {inflow_file} is missing: rerun handoff/handoff.py")

    gas = load_gas(get(case, "gas", "species_file"))
    log(f"reading SU2 solution {vtu}")
    field = Su2Field(vtu)  # whole SU2 domain: the nozzle wall comes from its outline
    box = (x_c, Lx, 0.0, Rext)

    # nozzle walls inside the box: the solid region gets no refinement and no particles
    walls = None
    solid = lambda x, y: np.zeros(np.shape(x), bool)
    if x_c < 0:
        x_w, r_w = ssurf.wall_profile(field, x_c)
        walls = dict(T=float(get(case, "su2", "wall_temp_K")), pts=ssurf.outline(x_w, r_w, box),
                     r_plane=float(r_w[0]), r_exit=float(r_w[-1]), n_wall=len(x_w) - 1)
        solid = ssurf.solid_test(x_w, r_w)

    def props(rho, T, u, v):
        n = rho / gas.mass
        lam = gas.mean_free_path(n, T)
        cbar = gas.mean_speed(T)
        return dict(n=n, T=T, lam=lam, tau=lam / cbar, speed=np.hypot(u, v) + cbar)

    def sampler(x, y):
        p = props(**field.sample(x, y))
        s = solid(x, y)  # inside the nozzle body: no gas
        if s.any():
            p.update(n=np.where(s, 0.0, p["n"]), T=np.where(s, 0.0, p["T"]),
                     lam=np.where(s, np.inf, p["lam"]), tau=np.where(s, np.inf, p["tau"]),
                     speed=np.where(s, 0.0, p["speed"]))
        return p

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
    gas_cell = d["n"] > 0  # cells whose centre is in the flow (not inside the nozzle body)
    ref = gas_cell & (g.level > 1) if (gas_cell & (g.level > 1)).any() else gas_cell
    fnum_ppc = np.percentile(ndy[ref], 10) / ppc_target
    # memory budget: never more than max_particles at steady state (~100-200 bytes each)
    max_particles = float(get(case, "sparta", "max_particles", 5e6))
    fnum_mem = ndy.sum() / max_particles
    fnum_rec = max(fnum_ppc, fnum_mem)
    with np.errstate(divide="ignore"):
        dt_rec = float(np.min(np.minimum(tau_frac * d["tau"],
                                         cell_frac * np.minimum(dx, dy) / d["speed"])))
    fnum = _auto(case, "fnum") or fnum_rec
    dt = _auto(case, "timestep_s") or dt_rec
    ppc = ndy / fnum

    # ---- run length and output ---------------------------------------------
    run_steps = int(get(case, "sparta", "run_steps", 1000))
    hold = adapt_hold(case)
    hold["images"] = hold_images(case, box, hold["steps"], float(d["n"].max()))
    # dumps fall on multiples of output_every counted from step 0, so with a hold it must
    # divide both the hold and the run for every averaged block (and the last) to be written
    out_every = int(get(case, "sparta", "output_every",
                        math.gcd(hold["steps"], run_steps) if hold["steps"] else run_steps))
    if hold["steps"] and (hold["steps"] % out_every or run_steps % out_every):
        die(f"sparta.output_every = {out_every} must divide hold_steps = {hold['steps']} and "
            f"run_steps = {run_steps} (e.g. {math.gcd(hold['steps'], run_steps)})")
    avg_steps = int(get(case, "sparta", "average_steps", max(out_every // 2, 1)))
    restart_every = int(get(case, "sparta", "restart_every", run_steps))
    if not 0 < avg_steps <= out_every:
        die("sparta.average_steps must be between 1 and output_every")
    if restart_every <= 0 or out_every <= 0:
        die("sparta.output_every and restart_every must be positive")
    if run_steps < 2:
        die("sparta.run_steps must be at least 2")

    # ---- write run directory ------------------------------------------------
    run_dir = Path(runs_dir) / name / "sparta"
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(case_path, run_dir / "case.toml")
    stem = Path(get(case, "gas", "species_file")).stem
    for ext in ("species", "vss"):
        shutil.copy(DATA / f"{stem}.{ext}", run_dir / f"{stem}.{ext}")
    # cells inside the nozzle body have no gas: lambda 0 there (SPARTA cannot read inf)
    g.write(run_dir / "grid.sparta", {"lambda": np.where(np.isfinite(d["lam"]), d["lam"], 0.0), "su2_nrho": d["n"]})
    if walls:
        ssurf.write_surf(run_dir / "nozzle.surf", walls["pts"],
                         header=f"Generated by sparta/setup.py from {case_path.name}: nozzle wall from the "
                                f"coupling plane to the exit, and the faceplate.\nClosed outline, clipped "
                                f"to the box by read_surf ... clip.")
    elif (run_dir / "nozzle.surf").exists():
        (run_dir / "nozzle.surf").unlink()  # from an earlier setup with the plane in the nozzle
    (run_dir / "in.grid").write_text(in_grid(case_path.name, box, gas.species, stem, fnum, dt, walls))
    shutil.copy(inflow_file, run_dir / "inflow.sparta")
    io = dict(out_every=out_every, avg_steps=avg_steps, restart_every=restart_every)
    (run_dir / "in.sparta").write_text(in_run(case_path.name, box, gas.species, stem, fnum, dt,
                                              run_steps, hold=hold, walls=walls, **io))
    (run_dir / "in.continue").write_text(in_continue(case_path.name, stem, run_steps, walls=walls, **io))

    # ---- report -------------------------------------------------------------
    per_level = np.bincount(g.level)[1:]
    rep = [
        f"case {name}: {gas.species}, D = {D*1e3:g} mm",
        f"domain x = [{x_c*1e3:.4g}, {Lx*1e3:.4g}] mm (coupling plane at {x_c/D:.4g} D), "
        f"r = [0, {Rext*1e3:.4g}] mm",
        (f"walls: nozzle from r = {walls['r_plane']*1e3:.4g} mm at the plane to {walls['r_exit']*1e3:.4g} mm "
         f"at the exit ({walls['n_wall']} lines), and the faceplate; diffuse at {walls['T']:g} K (nozzle.surf)"
         if walls else "walls: none (the plane is in the chamber; xlo is open)"),
        f"grid: {len(g.level)} cells over {g.nlevels} levels "
        f"({', '.join(f'L{i+1}:{c}' for i, c in enumerate(per_level))})",
        f"  cell size {np.min(dx)*1e6:.3g} um (finest) to {np.max(dx)*1e6:.3g} um (base)",
        f"  mean free path in domain {d["lam"].min()*1e6:.3g} um to {d["lam"][gas_cell].max()*1e3:.3g} mm",
        f"fnum     {fnum:.3g}" + ("" if fnum == fnum_rec else f"  (recommended {fnum_rec:.3g})")
        + (f"  [raised from {fnum_ppc:.3g} to keep <= {max_particles:.3g} particles]"
           if fnum_mem > fnum_ppc else ""),
        f"  particles/cell: refined cells median {np.median(ppc[ref]):.3g}, "
        f"10th pct {np.percentile(ppc[ref], 10):.3g}; all gas cells min {ppc[gas_cell].min():.3g}"
        + (f"; {int((~gas_cell).sum())} cells inside the nozzle body" if walls else ""),
        f"  steady-state particles ~ {ppc.sum():.3g} (~{ppc.sum() * 200 / 1e9:.2g} GB at 200 B each)",
        f"timestep {dt:.3g} s" + ("" if dt == dt_rec else f"  (recommended {dt_rec:.3g})"),
        f"inflow (from the handoff, {ho['mode']} plane): SU2 net mass flow {ho['mdot_net']:.4g} kg/s, "
        f"emitted (one-sided) {ho['mdot_inward']:.4g} kg/s; axis Kn_GLL at the plane {ho['kn_at_plane']:.3g}",
        (f"hold: {hold['steps']} steps ({hold['steps'] * dt:.3g} s) without output, then "
         + (f"adapt the grid to the particle count averaged over the last {hold['window']} steps "
            f"(refine > {hold['refine']:g}, coarsen < {hold['coarsen']:g} particles)"
            if hold["adapt"] else "no grid adaptation")
         if hold["steps"] else "hold: none"),
        (f"images: image.<step>.ppm every {hold['images']['every']} steps of the hold, "
         f"{hold['images']['width']} x {hold['images']['height']} px, "
         + ("particles (red on black)" if hold["images"]["style"] == "particles" else
            f"log10 n from {hold['images']['lo']:g} to {hold['images']['hi']:g}")
         if hold["images"] else "images: none"),
        f"run: {run_steps} steps ({run_steps * dt:.3g} s); output every {out_every} (averaged over the "
        f"last {avg_steps}); restart file every {restart_every}; "
        f"one transit of the box at the max inflow speed takes {(Lx - x_c) / max(ho['u_max'], 1.0) / dt:.3g} steps",
    ]
    for line in rep:
        log(line)
    warn = []
    if g.unresolved:
        warn.append(f"{g.unresolved} cells still exceed lambda/{k} at the level limit "
                    f"(worst by {g.worst:.2g}x): use a finer base_grid, or a SPARTA build with "
                    f"64-bit cell IDs (-DSPARTA_BIGBIG) and a higher max_levels")
    if "x_breakdown_m" in ho and x_c > ho["x_breakdown_m"]:
        warn.append(f"Kn_GLL first exceeds {ho['kn_threshold']} at {ho['x_breakdown_m']/D:.3g} D, "
                    f"upstream of the coupling plane: SU2 is not reliable there, use "
                    f"x_couple_D < {ho['x_breakdown_m']/D:.3g} or \"auto\"")
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


def _walls(walls) -> str:
    """Nozzle wall + faceplate surface and its collision model (empty without walls)."""
    if not walls:
        return ""
    return f"""
# nozzle wall (coupling plane -> exit) and faceplate, diffuse at the SU2 wall temperature
read_surf       nozzle.surf clip
surf_collide    wall diffuse {walls['T']:g} 1.0
surf_modify     all collide wall
"""


def _walls_restart(walls) -> str:
    """After read_restart: the surfaces are in the restart file, their collision model is not."""
    if not walls:
        return ""
    return f"""surf_collide    wall diffuse {walls['T']:g} 1.0
surf_modify     all collide wall
"""


def in_grid(case_name, box, species, stem, fnum, dt, walls=None) -> str:
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
# gridcut 0: each rank keeps only its own cells and their neighbours, not a copy of the
# whole grid (the default -1). Needs the clumped assignment that balance_grid rcb gives.
global          gridcut 0.0
read_grid       grid.sparta custom lambda float 0 custom su2_nrho float 0
balance_grid    rcb cell
{_walls(walls)}
global          weight cell radius
global          fnum {fnum:.4g}
species         {stem}.species {species}
mixture         gas {species}
collide         vss gas {stem}.vss
timestep        {dt:.4g}

dump            cells grid all 1 grid.dump id xlo ylo xhi yhi g_lambda g_su2_nrho
run             0
"""


def adapt_hold(case) -> dict:
    """[sparta] hold/adapt settings: a transient hold of hold_steps, then (optionally) a
    one-time adapt_grid on the particle count averaged over the last adapt_window steps."""
    ppc = get(case, "sparta", "ppc_target", 20)
    h = dict(steps=int(get(case, "sparta", "hold_steps", 0)),
             adapt=bool(get(case, "sparta", "adapt", True)),
             window=int(get(case, "sparta", "adapt_window", 1000)),
             refine=float(get(case, "sparta", "adapt_refine_count", 4 * ppc)),
             coarsen=float(get(case, "sparta", "adapt_coarsen_count", ppc)))
    if h["steps"] < 0:
        die("sparta.hold_steps must be >= 0")
    if h["steps"] and h["adapt"]:
        if not 0 < h["window"] <= h["steps"] or h["steps"] % h["window"]:
            die("sparta.adapt_window must be positive and divide hold_steps")
        if not h["coarsen"] < h["refine"]:
            die("sparta.adapt_coarsen_count must be below adapt_refine_count")
    return h


def hold_images(case, box, hold_steps, n_max):
    """[sparta] image_* settings for snapshots of the transient hold, or None.

    Picture size: in 2D the SPARTA box is 1 m deep in z, and dump image scales the picture to
    2 x the largest box dimension, so without a zoom the domain fills a few % of the frame.
    zoom = 2 max(Lx, Ly, 1 m) / (height of the picture in metres) makes the box fill it."""
    every = int(get(case, "sparta", "image_every", 0))
    if not every:
        return None
    style = get(case, "sparta", "image_style", "particles")
    if style not in ("particles", "density"):
        die('sparta.image_style must be "particles" or "density"')
    if every < 0 or not hold_steps:
        die("sparta.image_every needs a positive value and a transient hold (hold_steps > 0)")
    xlo, xhi, ylo, yhi = box
    Lx, Ly = xhi - xlo, yhi - ylo
    width = int(get(case, "sparta", "image_width", 1600))
    height = 2 * round(width * Ly / Lx / 2)  # even, for movie encoders
    zoom = 0.99 * 2 * max(Lx, Ly, 1.0) / max(Ly, Lx * height / width)
    avg = int(get(case, "sparta", "image_avg_steps", min(100, every)))
    if not 0 < avg <= every:
        die("sparta.image_avg_steps must be between 1 and image_every")
    hi_auto = math.ceil(math.log10(max(n_max, 1.0)))
    lo, hi = (float(v) for v in get(case, "sparta", "image_log_range", [hi_auto - 5, hi_auto]))
    if not lo < hi:
        die("sparta.image_log_range must be [low, high] with low < high (log10 of 1/m3)")
    # particles: one dot each, image_pdiam_px pixels across (pdiam is in metres)
    px = Ly / height
    pdiam = float(get(case, "sparta", "image_pdiam_px", 1.5)) * px
    return dict(style=style, every=every, avg=avg, width=width, height=height, zoom=zoom, lo=lo, hi=hi,
                pdiam=pdiam, sdiam=2 * px / min(Lx, Ly))


def _images_on(img, walls=None) -> str:
    """Start the hold snapshots as PPM images: the particles (stock SPARTA look: red dots on
    black, walls grey), or log10 number density per cell, averaged."""
    if img.get("style", "particles") == "particles":
        surf = f" &\n                surf one {img['sdiam']:.3g}" if walls else ""
        return f"""# ---- images of the transient hold -------------------------------------------------------
# image.<step>.ppm every {img['every']} steps: every particle as a red dot ({img['pdiam']:.3g} m across) on black{", nozzle wall grey" if walls else ""}.
# PPM because this SPARTA build has no JPEG/PNG; convert with ImageMagick (sparta/README.md).
# zoom: the 2D box is 1 m deep in z and the picture is scaled to that, so zoom fits it to the domain.
dump            img image all {img['every']} image.*.ppm type type pdiam {img['pdiam']:.4g} &
                size {img['width']} {img['height']} zoom {img['zoom']:.6g} box no 0.0{surf}
dump_modify     img pad 7 backcolor black

"""
    return f"""# ---- images of the transient hold -------------------------------------------------------
# image.<step>.ppm every {img['every']} steps: log10 number density per cell, averaged over the
# {img['avg']} steps before each image, white = 1e{img['lo']:g} /m3 or less, navy = 1e{img['hi']:g} /m3 or more.
# PPM because this SPARTA build has no JPEG/PNG; convert with ImageMagick (sparta/README.md).
# zoom: the 2D box is 1 m deep in z and the picture is scaled to that, so zoom fits it to the domain.
compute         img grid all gas nrho
variable        img_lgn grid log(c_img[1]+{10 ** img['lo']:.4g})
fix             img ave/grid all 1 {img['avg']} {img['every']} v_img_lgn
dump            img image all {img['every']} image.*.ppm type type particle no grid f_img &
                size {img['width']} {img['height']} zoom {img['zoom']:.6g} box no 0.0
dump_modify     img pad 7 backcolor white &
                cmap grid {img['lo']:g} {img['hi']:g} cf 0.0 5 min white 0.25 lightsteelblue 0.5 cornflowerblue 0.75 royalblue max navy

"""


def _images_off(img) -> str:
    """End of the hold: stop the images (before the grid changes)."""
    if img.get("style", "particles") == "particles":
        return "# end of the hold: stop the images\nundump          img\n\n"
    return """# end of the hold: stop the images (before the grid changes)
undump          img
unfix           img
variable        img_lgn delete
uncompute       img

"""


def _averaging(out_every, avg_steps) -> str:
    """Time-averaged flow field and its dump."""
    return f"""\
compute         g grid all gas nrho massrho u v
compute         th thermal/grid all gas temp
fix             avg ave/grid all 1 {avg_steps} {out_every} c_g[*] c_th[*]
dump            flow grid all {out_every} flow.*.dump id xc yc f_avg[1] f_avg[2] f_avg[3] f_avg[4] f_avg[5] g_su2_nrho g_lambda
"""


def _housekeeping(restart_every, nlog) -> str:
    """Stats, load balancing and restart files."""
    return f"""\
stats           {nlog}
stats_style     step cpu np nattempt ncoll
fix             lb balance {max(nlog * 2, 1)} 1.1 rcb part
restart         {restart_every} restart.*.sparta
"""


_FIRST_STEP_NOTE = """\
# One step before the averaged output is defined: SPARTA always dumps on the first step of
# a run that falls on a multiple of the dump interval, which would write an empty average.
run             1
"""


def _hold_block(hold, walls=None) -> str:
    """Transient hold, then optional one-time grid adaptation to where the particles are."""
    if not hold["steps"]:
        return ""
    img = hold.get("images")
    on, off = (_images_on(img, walls), _images_off(img)) if img else ("", "")
    txt = f"""\
{on}# ---- transient hold: let the jet fill the box before averaging -------------------------
"""
    if not hold["adapt"]:
        return txt + f"run             {hold['steps']}\n\n{off}"
    return txt + f"""\
run             {hold['steps'] - hold['window']}

# particle count per cell, averaged over the last {hold['window']} steps of the hold
compute         hold_n grid all gas n
fix             hold_n ave/grid all 1 {hold['window']} {hold['window']} c_hold_n[1]
run             {hold['window']}

{off}# ---- adapt the grid to the particles ----------------------------------------------------
# refine cells holding > {hold['refine']:g} particles, merge sibling cells holding < {hold['coarsen']:g} in total.
# Refinement stops at the cell-ID bit limit (cells already at the finest level stay);
# new cells get 0 in the SU2 columns g_su2_nrho, g_lambda.
adapt_grid      all refine coarsen value f_hold_n {hold['refine']:g} {hold['coarsen']:g} combine sum
unfix           hold_n
uncompute       hold_n
balance_grid    rcb part
write_restart   restart.*.sparta

"""


def in_run(case_name, box, species, stem, fnum, dt, run_steps, out_every, avg_steps, restart_every,
           hold=None, walls=None) -> str:
    """Coupled DSMC run from an empty box: SU2 inflow on xlo; an optional transient hold
    with grid adaptation; then run_steps with averaged output every out_every steps;
    restart files every restart_every steps and at the end."""
    hold = hold or dict(steps=0, adapt=False)
    xlo, xhi, ylo, yhi = box
    return f"""\
# Generated by sparta/setup.py from {case_name} -- edit the case file, not this.
# DSMC downstream of the coupling plane (xlo), fed by the SU2 state there.
#   mpirun -n N spa_mpi -in in.sparta   ->  log.sparta, flow.<step>.dump, restart.<step>.sparta
# Continue later with in.continue.

seed            12345
dimension       2

# xlo = coupling plane (inflow + outflow), ylo = axis, yhi/xhi = chamber outflow
boundary        o ao p
create_box      {xlo:.10g} {xhi:.10g} {ylo:.10g} {yhi:.10g} -0.5 0.5
# gridcut 0: each rank keeps only its own cells and their neighbours, not a copy of the
# whole grid (the default -1). Needs the clumped assignment that balance_grid rcb gives.
global          gridcut 0.0
read_grid       grid.sparta custom lambda float 0 custom su2_nrho float 0
balance_grid    rcb cell
{_walls(walls)}
global          weight cell radius
global          fnum {fnum:.4g}
species         {stem}.species {species}
mixture         gas {species}
collide         vss gas {stem}.vss
timestep        {dt:.4g}

fix             inflow emit/face/file gas xlo inflow.sparta XLO
{_housekeeping(restart_every, max(out_every // 10, 1))}
{_hold_block(hold, walls)}# ---- run with averaged output ------------------------------------------------------------
{_FIRST_STEP_NOTE}
{_averaging(out_every, avg_steps)}
run             {run_steps - 1}
write_restart   restart.*.sparta
"""


def in_continue(case_name, stem, run_steps, out_every, avg_steps, restart_every, walls=None) -> str:
    """Continue a run from its latest restart file, up to timestep `target`."""
    return f"""\
# Generated by sparta/setup.py from {case_name} -- edit the case file, not this.
# Continue the DSMC run from the latest restart.<step>.sparta, up to timestep `target`:
#   mpirun -n N spa_mpi -in in.continue -var target 30000
# Works with a different N than the original run. The restart file holds the grid (as
# adapted after the hold), particles, species, mixture, global settings (fnum, weight,
# gridcut), timestep size and step number; everything else is defined again here.
# A continuation does not repeat the hold or the grid adaptation: continue from a restart
# written after the adaptation (the one at the end of the hold or later).

variable        target index {2 * run_steps}
variable        seed index 23456      # use a new seed for each continuation
seed            ${{seed}}

read_restart    restart.*.sparta balance rcb cell
collide         vss gas {stem}.vss
{_walls_restart(walls)}fix             inflow emit/face/file gas xlo inflow.sparta XLO
{_housekeeping(restart_every, max(out_every // 10, 1))}
{_FIRST_STEP_NOTE}
{_averaging(out_every, avg_steps)}
run             ${{target}} upto
write_restart   restart.*.sparta
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("case", help="case file")
    ap.add_argument("--runs", default="runs", help="runs directory (default: ./runs)")
    args = ap.parse_args(argv)
    setup(args.case, args.runs)


if __name__ == "__main__":
    main()
