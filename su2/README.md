# SU2 stage

Solves laminar Navier–Stokes for the capillary flow and the near-field jet. The result
(`flow.vtu`) is used by the [handoff](../handoff/README.md) to place the coupling plane and
write the DSMC inflow, and by the [SPARTA stage](../sparta/README.md) to size the DSMC grid.

Run all commands from the repo root. For the overall workflow, see the
[main README](../README.md).

## Set up

```bash
python su2/setup.py my_case.toml            # default case file: config.toml
python su2/setup.py my_case.toml --runs /scratch/runs
```

This does five things:
1. Creates `runs/<name>/su2/` and copies the case file into it as `case.toml`.
2. Writes `mesh.geo` using the geometry generator `su2/geom/<noz_type>.py`.
3. Runs gmsh to produce `mesh.su2`. gmsh's output goes to `gmsh.log`, and the cell count is printed.
4. Writes `mesh_quality.txt` ([below](#mesh-quality)) and prints any checks it fails as warnings.
5. Writes `su2.cfg`: boundary conditions, gas properties and solver settings.

> `su2.cfg` and `mesh.geo` are generated files. For a one-off experiment, copy `su2.cfg` to
> a new name inside the run directory and run SU2 on that copy by hand. Running setup again
> overwrites `su2.cfg` and the mesh.

### Restarting

To continue an unconverged run from its `restart_flow.dat`:

```bash
CASE=my_case.toml STAGES=su2_restart SU2_ITER=20000 bash run_case.sh   # 20000 more iterations
```

- **What the stage changes:** `run_case.sh` sets `RESTART_SOL` (and `ITER`, if `SU2_ITER` is
  given) in `su2.cfg` with `sed`. The plain `su2` stage sets `RESTART_SOL= NO`, so it always
  starts from the initial state, whatever an earlier restart left in `su2.cfg`.
- **Archive:** before a restart, the current `flow.vtu`, `history.csv`, `su2.log`,
  `restart_flow.dat` and `su2.cfg` are copied to `archive/<n>/`. You can compare the new
  solution with the old one, or go back.
- **Iteration count:** SU2 counts iterations from 0 again after a restart, so `SU2_ITER` is the
  number for this run, not a total.
- **Don't run setup between restarts:** it remeshes, so the restart file no longer fits.
- **Stepping the back pressure down:** `BACK_PRES=<Pa>` sets the outlet pressure
  (`MARKER_OUTLET`, and `FREESTREAM_PRESSURE`, which only sets the initial state) in `su2.cfg`
  for the `su2` and `su2_restart` stages. Start with `su2` at a high back pressure, then
  `su2_restart` at lower ones, each from the previous solution. `[su2] back_pres_pa` in the
  case file only applies at `su2_setup`; set it to the final value so the case file records it.

### Starting from a coarse mesh

A new case converges much faster on a coarse mesh, and its solution is a far better starting
point for the full mesh than gas at rest: on the N₂ nozzle at half resolution, a start from the
quarter-mesh solution began at `rms[Rho]` −6.2 with the CFL climbing, while a cold start sat at
−4.9 with the CFL at its 0.5 floor.

```bash
export CASE=my_case.toml
COARSEN=4 STAGES="su2_setup su2" SU2_ITER=20000 bash run_case.sh   # runs/<name>/su2_coarse4/
COARSEN=2 FROM=4 STAGES="su2_setup su2_refine" bash run_case.sh   # optional middle level
FROM=2 STAGES="su2_setup su2_refine" bash run_case.sh             # full mesh, runs/<name>/su2/
```

- **`COARSEN=N`** (or `python su2/setup.py CASE --coarsen N`) divides every cell count in `[su2]`
  by N (at least 2 cells) and multiplies the cell sizes (`first_cell_um`, `exit_cell_um`, `throat_cell_um`,
  `lip_relax_cell_um`) by N, so the blocks and grading match the full mesh. It builds and runs
  in `runs/<name>/su2_coarse<N>/`; the full mesh in `su2/` is untouched. `su2_restart` and
  `BACK_PRES` work there too, so a low back pressure can be stepped down on the coarse mesh.
- **`su2_refine`** with `FROM=N` archives the current run (as `su2_restart` does), interpolates
  `su2_coarse<N>/restart_flow.dat` onto this mesh with `su2/remap_restart.py`, and runs SU2
  from it. Run `su2_setup` first (or in the same `STAGES`) so the finer mesh exists.
- **Levels:** 2× and 4× work for the N₂ nozzle (22,700 and 5,680 cells against 90,800). At 8×
  the nozzle has 5 radial cells and `mesh_quality.txt` flags a 26× size jump at the lip: too
  coarse to be a useful start.
- **The coarse solution is only a start.** Its boundary layers and shear layers are smeared;
  converge the full mesh before the handoff, which reads `su2/flow.vtu`.

`su2/remap_restart.py OLD_RESTART NEW_MESH [-o OUT]` can also be run by hand. It reads SU2
binary restarts (which hold their own x, y) and interpolates the conserved variables linearly
on the Delaunay triangulation of the old nodes; the new mesh must cover the same domain. gmsh
numbers the nodes differently for each mesh, so a restart file never fits another mesh as is.

## Run

```bash
CASE=my_case.toml STAGES=su2 NP=8 bash run_case.sh   # or qsub -v CASE=...,STAGES=su2 run_case.sh
cd runs/<name>/su2 && mpirun -np 8 SU2_CFD su2.cfg > su2.log 2>&1   # the same, by hand
```

SU2's screen output goes to `su2.log`; it is block-buffered, so follow progress in
`history.csv`. SU2 writes `flow.vtu` and `restart_flow.dat` every 500 iterations. A job
killed while writing (walltime, `qdel`) can leave `flow.vtu` truncated; the handoff then
stops with a message. Rewrite it from `restart_flow.dat` with `SU2_ITER=1` and the
`su2_restart` stage (this overwrites `history.csv`; the earlier one is in `archive/<n>/`).

## Results

| File | Contents |
|---|---|
| `flow.vtu` | volume solution for ParaView (coordinates, conserved and primitive variables) |
| `surface_flow.csv` | values on the plotted markers (`wall`, and `faceplate` when it is a wall) |
| `history.csv` | residuals, CFL, and mass flow through `inlet` and `outlet` per iteration |
| `restart_flow.dat` | restart file, written every 500 iterations |
| `su2.log` / `gmsh.log` | solver and mesher output |
| `mesh_quality.txt` | mesh quality report, written by setup ([Mesh quality](#mesh-quality)) |

### Reading the screen output

```
|  Inner_Iter| Time(sec)| Time(sec)| rms[Rho]| rms[RhoE]| Avg CFL|Avg_Massflow|Avg_Massflow|Avg_Massflow|
|       49999|   ...    |   ...    |-7.182616| ...      | ...    |  2.9169e-06|  2.9243e-06| -7.4131e-09|
                                                                     total        inlet       outlet
```

The two `Time(sec)` columns are the time for this iteration and the running average.
`SURFACE_MASSFLOW` gives three columns, all in kg/s:
- **Total:** the sum over the `MARKER_ANALYZE` markers.
- **Inlet** and **outlet:** one column each, in the order the markers are listed.

The total always equals inlet + outlet, which is how you can tell the columns apart. The sign
follows the boundary normal: flow into the domain at `inlet` is positive, and flow out through
`outlet` is negative. In an axisymmetric run the 2πr factor is included, so the two can be
compared directly.

`history.csv` only stores the total (`Avg_Massflow`).

### Checking convergence

- **Automatic stop:** the run stops when the density residual `rms[Rho]` (log10) falls below
  −12 (`CONV_FIELD= RMS_DENSITY`, `CONV_RESIDUAL_MINVAL= -12`), or at `ITER`. Two details:
  - **Field name:** it must be SU2's name, `RMS_DENSITY`. The CSV header names (`rms[Rho]`,
    `Avg_Massflow`) are ignored: SU2 prints "Ignoring Convergence Field(s)" and runs every iteration.
  - **Units:** with `REF_DIMENSIONALIZATION= DIMENSIONAL` the residual is in SI units, so −12 is
    not a relative drop. A low-pressure case (density ~1e-6 kg/m³ in the chamber) reaches it
    sooner than its flow has settled; a dense one may never get there. Use −12 as a backstop,
    and judge convergence by the drop from the start of the run and by the checks below. Change
    `CONV_RESIDUAL_MINVAL` in `su2.cfg` if a case needs another level.
- **Residual:** `rms[Rho]` should fall. A residual that has levelled off means the run is
  stuck, not converged.
- **Mass balance:** at steady state, what comes in must go out. The outlet column should be
  about −(inlet), and the total should go to about 0. If the total is still close to the
  inlet value, gas is still building up inside the domain.
- **Jet:** the jet should leave the tube. On the axis, the exit plane should be choked
  (M ≈ 1) and M should rise quickly downstream.

A run that starts from rest needs many iterations before the flow front gets out of a long
capillary. If the chamber is still at back pressure, run more iterations, or restart from a
converged solution on a coarser mesh.

A steady inlet mass flow does not mean the run has converged. Also look for:
- **A stalled near-wall layer.**
  - **Symptom:** in a nozzle, the cells next to the diverging wall can stay at the initial
    back pressure (0.1 Pa) while the core beside them is at tens of Pa.
  - **Why:** their viscous local timestep, Δt ≈ ρΔy²/μ, is about 1e-13 s for 1 µm cells at
    0.1 Pa, so they barely move. `Min DT` in `history.csv` of about 1e-14 to 1e-13 is the
    sign.
  - **What helps:** more iterations shrink the layer slowly. In the N₂ nozzle it went from
    20 to 13 µm thick at the exit over 6000 more iterations.
  - **Better (not implemented yet):** start the domain at a higher pressure than the back
    pressure.
- **Core still changing.** Compare the axis pressure between two snapshots. In the N₂
  nozzle it was still rising 30% at x = −10 mm after 15,000 iterations.

To compare two snapshots, copy `flow.vtu` aside before a restart.

---

## Model

### Geometry and mesh (`noz_type = "capillary"`)

The model is 2D axisymmetric: x runs along the axis, y is the radius, and the capillary exit
plane is at x = 0. The mesh is built from three structured quad blocks:

```
  +--------------- top (outlet) --------------+
  |                                           |
faceplate              block B             outlet
  |                                           |
  +---- tube wall ----+ lip -------------------+
  |                   |                       |
inlet   tube block    |       block A       outlet
  |                   |                       |
  +------ axis -------+ exit ----- axis ------+
  x = -L              x = 0                    x = chamb_length
```

Cells get smaller toward the tube wall, the exit plane and the lip. The first cell at each of
these is `first_cell_um` wide, and the growth ratios are solved to fit; they are written in the
`mesh.geo` header. If a cell count is already coarse enough to be uniform at that size, the
ratio is 1.

### Geometry and mesh (`noz_type = "cd_nozzle"`)

A conical converging–diverging nozzle, with its exit plane at x = 0 and the same chamber
blocks as the capillary. Upstream to downstream, the wall is:
- the inlet, at diameter `inlet_diam_mm`;
- a cone at `conv_half_angle_deg`;
- a circular arc of radius `throat_rc_D`·D_t into the throat at `throat_diam_mm`;
- an optional straight throat, `throat_length_mm`;
- a second arc, then a cone at `div_half_angle_deg` out to `exit_diam_mm`.

`su2/geom/cd_nozzle.py` builds the profile (`profile(case)`, which also gives the station
positions and the area ratio) and writes the blocks:

| Block | Axial cells | Clustered at |
|---|---|---|
| converging (inlet → throat) | `n_axial_conv` | throat (`throat_cell_um`) |
| straight throat, if any | L_t / `throat_cell_um` | uniform |
| diverging, first half | `n_axial_div`/2 | throat (`throat_cell_um`) |
| diverging, second half | `n_axial_div`/2 | exit plane (`exit_cell_um`, default `first_cell_um`) |
| chamber A, B | `n_axial_ext` | exit plane |

Every radial line has `n_radial_tube` cells, and its first cell at the wall is
`first_cell_um`. Curved walls are gmsh splines through 200 samples of the exact profile, so
the wall nodes lie within about 10 nm of it. The geometry stops with a clear message if it
can't be built, for example when the inlet or exit is too small for the throat arcs.
`examples/cd_nozzle_n2.toml` is a complete example case.

**Thin cells in the chamber** (optional keys). By default every cell next to the wall, the exit
plane and the lip is `first_cell_um` thin, and two rows of these continue through the chamber
as cells that are millimetres long: along the lip line out to the outlet, and up the faceplate
to the chamber top. At very low back pressure (0.125 Pa for the N₂ nozzle) the gas viscosity is
so large (ν ~ 10 m²/s) that SU2 hardly updates such cells: they stay frozen, as a pressure
ridge or a trough at the initial pressure, and the run can't settle (the inlet and outlet mass
flows keep drifting apart). Nothing physical there is that thin: the shear layer at the exit is
already about a millimetre thick. Two keys coarsen them:

```toml
exit_cell_um = 10.0                           # axial cell either side of the exit plane
lip_relax_x = [0.02, 0.1, 0.3, 1.0]           # stations, as fractions of the chamber length
lip_relax_cell_um = [1.0, 10.0, 50.0, 100.0]  # radial cell at the lip line at each station
```

- **`exit_cell_um`** (default `first_cell_um`) sets the axial cell on both sides of the exit
  plane: the end of the diverging section and the start of the chamber, including the column
  up the faceplate. The radial cells at the walls and the lip stay `first_cell_um`.
- **`lip_relax_x` / `lip_relax_cell_um`** set the radial cell at the lip line along the chamber.
  The chamber is split into blocks at the stations; each station moves to the nearest axial
  node, so the axial cells, the cell count and the markers don't change. Upstream of the first
  station the cell is `first_cell_um`; between stations gmsh's transfinite interpolation ramps
  it smoothly; past the last station it stays constant. The `mesh.geo` header lists the
  stations. A value above the uniform spacing (lip radius / `n_radial_tube` below the lip,
  (chamber radius − lip radius) / `n_radial_ext` above it) just gives uniform cells.
- **Choosing values:** check `mesh_quality.txt`. For the N₂ nozzle's 39,800-cell mesh, the values
  above take the worst aspect ratio from 4341 to 784 (in the nozzle), leave no size jump above
  1.5, and keep corner angles within 52–128°. Starting the lip relaxation at half the chamber
  length instead left the frozen band upstream of it; a 50 µm exit cell brings back a 1.8×
  jump mid-way along the diverging section.
- Both change the mesh, so an old `restart_flow.dat` no longer fits; interpolate it with
  `su2/remap_restart.py` ([Starting from a coarse mesh](#starting-from-a-coarse-mesh)).

### Mesh quality

`su2_setup` writes `mesh_quality.txt` next to the mesh (`su2/mesh_quality.py`). For an
existing mesh: `python su2/mesh_quality.py runs/<name>/su2/mesh.su2`. Per cell it measures:
- **aspect ratio:** longest / shortest edge;
- **corner angles:** the four interior angles (90° = square);
- **size jump:** the largest area ratio to a cell sharing an edge (1 = same size).

It summarises these (median, 99th percentile, extremes) for the nozzle, the chamber below the
lip radius (the jet core) and the chamber above it, lists the three worst cells for each with
their (x, r), and checks:

| Check | Limit (constants in `su2/mesh_quality.py`) |
|---|---|
| aspect ratio | ≤ 1000 |
| corner angles | 45–135° |
| size jump | ≤ 1.5 |

A failed check is a warning, not an error: boundary-layer cells are meant to be thin. Look at
where the worst cells are. Long thin cells far from any wall or shear layer, or a size jump
inside the nozzle, are worth fixing. SU2 also prints its own table (orthogonality, control-volume
aspect ratio, volume ratio) near the top of `su2.log`, as worst values only.

For the N₂ nozzle (`examples/cd_n2_4kpa_0p125pa.toml`), the report found what the thin-cell
keys above now fix:
- **aspect ratio up to 4341 at the faceplate top** (x ≈ 0.5 µm, r ≈ 58 mm): the 1 µm axial
  cells at the exit plane ran all the way up the faceplate, where the radial cells are 2 mm;
- **aspect ratio ~3000 along the lip line** out to the outlet;
- **size jump 1.69 at x ≈ −18 mm**, mid-way along the diverging section, where its two blocks
  (each coarsest at that end) meet. A 10 µm `exit_cell_um` makes their end sizes match.

### Boundary conditions

| Marker | SU2 condition | Value |
|---|---|---|
| `inlet` | total conditions (plenum) | `plen_pres_pa`, `plen_temp_K`, flow along +x |
| `wall` | isothermal no-slip | `wall_temp_K` |
| `faceplate` | isothermal wall, or symmetry/slip if `faceplate = "sym"` | `wall_temp_K` |
| `axis` | symmetry | |
| `outlet` | static pressure | `back_pres_pa` |

The domain starts filled with gas at rest at `back_pres_pa` and `wall_temp_K`.

> SU2 can't use a back pressure of zero. Use a small value, such as 0.1 Pa. The flow leaving
> through most of the outlet is supersonic, so this value hardly affects the solution.

### Gas

The species comes from the stem of `species_file`: `he.species` gives `he`. Gas properties
are in the `GASES` table in `su2/setup.py`: γ, R, Sutherland viscosity, and a constant
Prandtl number. Each gas also needs `sparta/data/<gas>.species` and `.vss` files;
`tests/test_gases.py` checks that the two codes agree.

| Gas | γ | Viscosity fit to the DSMC (VSS) model |
|---|---|---|
| `he`, `ar` | 5/3 | anchored at 273 K: within 6% at 250–400 K, but 16% low at 100 K and up to 60% low at 20 K |
| `d2` | 1.4 | minimax over 20–400 K: within 7.2% everywhere in that range |
| `n2` | 1.4 | minimax over 50–400 K: within 4.7% (below ~50 K an N₂ expansion condenses anyway) |

SU2's compressible solver only allows Sutherland or constant viscosity, so it can't follow
the VSS power law μ ∝ T^ω exactly.

For `d2`, γ = 1.4 assumes rotation is active. Below about 50 K, rotation freezes and the
real γ rises towards 5/3, which this model doesn't capture.

### Solver

The flow is laminar Navier–Stokes, solved to steady state with implicit Euler time stepping.
It uses the Roe scheme with MUSCL reconstruction and the Venkatakrishnan limiter, starts at
CFL 1, and lets the CFL adapt up to 100.

---

## Case file: `[su2]`

```toml
[su2]
back_pres_pa = 0.1             # outlet static pressure, and the initial pressure
wall_temp_K = 300.15           # tube wall / faceplate temperature, and the initial temperature
faceplate = "wall"             # "wall" (isothermal) or "sym" (slip); optional, default "wall"
first_cell_um = 1.0            # first cell size at wall, exit plane and lip
# exit_cell_um = 10.0          # cd_nozzle: axial cell either side of the exit plane; optional, default first_cell_um
n_radial_tube = 40             # cells across the tube radius
n_axial_tube = 800             # cells along the tube
n_axial_ext = 300              # cells from the exit plane to the outlet
n_radial_ext = 200             # cells from the lip to the outer radius
# lip_relax_x = [0.02, 0.1, 0.3, 1.0]           # cd_nozzle: coarsen the lip-line cell downstream; optional
# lip_relax_cell_um = [1.0, 10.0, 50.0, 100.0]  # radial cell at the lip line at each station
iterations = 15000             # maximum iterations
# ext_length_D, ext_radius_D   # currently unused: [geometry] chamb_* set the domain size
```

For `noz_type = "cd_nozzle"`, the tube keys are replaced by:

```toml
throat_cell_um = 10.0          # axial cell size at the throat; optional, default R_t/20
n_radial_tube = 40             # cells from wall to axis inside the nozzle
n_axial_conv = 120             # inlet -> throat
n_axial_div = 200              # throat -> exit (split over two blocks)
```

The `[case]`, `[geometry]` and `[gas]` sections are described in the
[main README](../README.md#case-file).

## Extending

- **New nozzle shape:** add `su2/geom/<name>.py` with a function
  `geo(case: dict, source: str) -> str` that returns the gmsh `.geo` text. It must define the
  physical curves `inlet`, `wall`, `faceplate`, `axis` and `outlet`, and a physical surface.
  Then set `noz_type = "<name>"`.
- **New gas:** add an entry to `GASES` in `su2/setup.py`, keyed by the species file stem.
  Add the matching SPARTA files too; see [sparta/README.md](../sparta/README.md#extending).
