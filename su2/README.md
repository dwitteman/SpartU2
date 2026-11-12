# SU2 stage

Solves laminar Navier–Stokes for the capillary flow and the near-field jet. The result
(`flow.vtu`) is used by the [SPARTA stage](../sparta/README.md) to size the DSMC grid, and
later to provide its inflow.

Run all commands from the repo root. For the overall workflow, see the
[main README](../README.md).

## Set up

```bash
python su2/setup.py my_case.toml            # default case file: config.toml
python su2/setup.py my_case.toml --runs /scratch/runs
```

This does four things:
1. Creates `runs/<name>/su2/` and copies the case file into it as `case.toml`.
2. Writes `mesh.geo` using the geometry generator `su2/geom/<noz_type>.py`.
3. Runs gmsh to produce `mesh.su2`. gmsh's output goes to `gmsh.log`, and the cell count is printed.
4. Writes `su2.cfg`: boundary conditions, gas properties and solver settings.

> `su2.cfg` and `mesh.geo` are generated files. For a one-off experiment, copy `su2.cfg` to
> a new name inside the run directory and run SU2 on that copy by hand. Running setup again
> overwrites `su2.cfg` and the mesh.

### Restarting

Setup always writes `RESTART_SOL= NO`, and takes `ITER` from `[su2] iterations`. There is no
restart option in the case file yet. To continue an unconverged run from its
`restart_flow.dat`:

1. In `runs/<name>/su2/su2.cfg`, set `RESTART_SOL= YES` and raise `ITER` if needed.
2. Run `python su2/run.py runs/<name>/su2` again.
3. Don't run setup again in between, or it will undo these edits.

`su2.log` is overwritten on each run. Each restart begins counting iterations from 0 again.

## Run

```bash
python su2/run.py runs/<name>/su2           # MPI ranks default to [su2] nprocs
python su2/run.py runs/<name>/su2 -n 8
```

SU2's output appears on screen and is also saved to `su2.log`. Ctrl-C stops the MPI job.
The script exits with SU2's exit code.

## Results

| File | Contents |
|---|---|
| `flow.vtu` | volume solution for ParaView (coordinates, conserved and primitive variables) |
| `surface_flow.csv` | values on the plotted markers (`wall`, and `faceplate` when it is a wall) |
| `history.csv` | residuals, CFL, and mass flow through `inlet` and `outlet` per iteration |
| `restart_flow.dat` | restart file, written every 500 iterations |
| `su2.log` / `gmsh.log` | solver and mesher output |

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

- **Residual:** `rms[Rho]` should fall. The run stops early once it reaches 1e-10. A residual
  that has levelled off means the run is stuck, not converged.
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
| diverging, second half | `n_axial_div`/2 | exit plane (`first_cell_um`) |
| chamber A, B | `n_axial_ext` | exit plane |

Every radial line has `n_radial_tube` cells, and its first cell at the wall is
`first_cell_um`. Curved walls are gmsh splines through 200 samples of the exact profile, so
the wall nodes lie within about 10 nm of it. The geometry stops with a clear message if it
can't be built, for example when the inlet or exit is too small for the throat arcs.
`examples/cd_nozzle_n2.toml` is a complete example case.

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
n_radial_tube = 40             # cells across the tube radius
n_axial_tube = 800             # cells along the tube
n_axial_ext = 300              # cells from the exit plane to the outlet
n_radial_ext = 200             # cells from the lip to the outer radius
iterations = 15000             # maximum iterations
nprocs = 20                    # default number of MPI ranks for run.py; optional, default 1
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
