# SPARTA stage

Builds the DSMC domain downstream of the coupling plane, a grid sized from the
[SU2 solution](../su2/README.md), the SU2 inflow on the coupling plane, and a SPARTA run
input.

**Status:**
- **Inflow:** checked against SPARTA on a synthetic case (see below).
- **`in.sparta`:** written but not yet run on a real case.

Run all commands from the repo root. For the overall workflow, see the
[main README](../README.md).

## Set up

```bash
python sparta/setup.py my_case.toml               # reads runs/<name>/su2/flow.vtu
python sparta/setup.py my_case.toml --su2 other/flow.vtu
python sparta/setup.py my_case.toml --runs /scratch/runs
```

This needs a converged SU2 solution and numpy/scipy (`pip install -r requirements.txt`).
It prints a summary, also saved to `setup.log`:
- domain, cells per level and cell sizes;
- fnum, particle count and memory estimate, and timestep;
- the inflow: number of points, and the SU2 net and emitted mass flow through the plane;
- run length compared with one transit of the box;
- the Knudsen number at the coupling plane, and any warnings.

Then check that SPARTA reads the grid:

```bash
cd runs/<name>/sparta && spa_serial -in in.grid   # writes grid.dump
```

`grid.dump` lists each cell's bounds, with the SU2 mean free path and density. It is a quick
way to plot the grid.

## Run

```bash
cd runs/<name>/sparta && mpirun -n N spa_mpi -in in.sparta   # -> log.sparta, flow.<step>.dump
```

> **Check `setup.log` first.** The steady-state particle count and memory estimate are
> printed there. The estimate assumes the DSMC density matches SU2's. If SU2's chamber
> hasn't filled yet, the real DSMC count will be higher. On a 16 GB WSL machine, keep
> `max_particles` at a few million.

`in.sparta`:
- **Inflow:** emits the inflow on `xlo` every step.
- **Averaging:** averages the cell density, mass density, velocity and temperature over the
  last `average_steps` of `run_steps`.
- **Output:** writes them to `flow.<run_steps>.dump`, next to SU2's density and mean free
  path for comparison.
- **Load balance:** rebalances every `run_steps`/20 steps.

The default `run_steps = 1000` is a smoke test. Steady state needs at least a few box
transits. `setup.log` gives the number of steps for one transit at the fastest inflow speed;
for large chambers it is tens of thousands.

## Outputs (`runs/<name>/sparta/`)

| File | Contents |
|---|---|
| `grid.sparta` | grid file for `read_grid`, with two extra columns: SU2 mean free path `lambda` and number density `su2_nrho` |
| `in.grid` | a small SPARTA input that loads the domain and grid and dumps them; there is no flow |
| `inflow.sparta` | the SU2 state on the coupling plane, as a `fix emit/face/file` table |
| `in.sparta` | the coupled DSMC run: grid, gas, inflow, averaging and dump |
| `<gas>.species`, `<gas>.vss` | copied from `sparta/data/` |
| `setup.log` | setup summary and warnings |
| `case.toml` | copy of the case file |

---

## How it works

### Domain

| | |
|---|---|
| x | from the coupling plane x_c = `x_couple_D`·D to `chamb_length_cm` |
| r | from 0 to `chamb_height_cm` |
| `xlo` | coupling plane: open, and carries the SU2 inflow |
| `ylo` | axis (`a`, axisymmetric) |
| `xhi`, `yhi` | open, matching the SU2 outlet |

### Grid

The grid is hierarchical. Level 1 is a uniform `base_grid` mesh. A cell is split 2×2 while
it is larger than λ/`cells_per_mfp`, where λ is the smallest VSS mean free path in the cell.
λ comes from the SU2 density and temperature, read at the cell centre and corners and at every
SU2 node inside the cell, so features narrower than a cell are not missed.

The grid is written as a SPARTA `read_grid` file, using SPARTA's cell-ID encoding
(`sparta/grid.py`). SPARTA's default build stores cell IDs in 32 bits. Setup stays within 31,
because `read_grid` parses IDs with `atoi`. Each 2×2 level costs 3 bits, so a 200×100 base
grid (15 bits) allows 6 levels.

If setup warns that some cells are still too big, you can:
- **Coarsen `base_grid`:** fewer bits for level 1 leaves room for more levels.
- **Move the coupling plane downstream:** the mean free path is larger there.
- **Rebuild SPARTA with `-DSPARTA_BIGBIG`:** this gives 64-bit cell IDs. Then raise
  `ID_BITS` in `sparta/grid.py`.

### fnum, timestep and memory

The input uses `global weight cell radius`, so a particle's weight is fnum·r·Δx. The
number of particles in a cell is then 2π·n·Δr/fnum, independent of radius.

fnum therefore has units of 1/m², not "molecules per particle". In a small test box it
is still around 1e16. A value like 1e9 would ask for about 1e11 particles and exhaust
memory.

- **fnum:** chosen so that 90% of the refined cells hold at least `ppc_target` particles,
  then raised if needed so the steady-state count stays below `max_particles` (default 5e6).
  - **Memory:** setup prints the count and the memory it needs, at about 200 bytes per particle.
  - **Warnings:** setup warns when the budget costs particles per cell, or when a hand-set
    fnum exceeds the budget.
- **Timestep:** min(0.2·τ_collision, 0.5·cell transit time) over all cells.

Either can be set directly in the case file. Setup warns if a value you set is more than twice
the recommended one.

Cells in the background gas hold very few particles (≪ 1). At 0.1 Pa that region is
essentially vacuum.

### Inflow at the coupling plane

`sparta/inflow.py` samples the SU2 solution along x = x_c:
- **Radii:** the SU2 node radii of the nearest node column, plus 200 uniform points, so the
  plume edge keeps SU2's resolution.
- **Table:** writes nrho, vx, vy, vz = 0 and temp to `inflow.sparta`, plus trot and tvib
  (both = T) for gases that have those modes. SU2 assumes local equilibrium.

SPARTA (`fix emit/face/file ... xlo inflow.sparta XLO`):
- **Interpolation:** interpolates the table linearly to the midpoint of each cell face on
  `xlo`.
- **Emission:** emits the one-sided Maxwellian flux for that state through the face annulus
  π(r_hi² − r_lo²), divided by the cell weight.
- **Stagnant or slow gas** on the plane still emits its thermal flux n·c̄/4. The emitted mass
  flow is therefore larger than SU2's net ρu·2πr. The excess leaves again through `xlo`.

Setup reports both mass flows. In the N₂ nozzle case, the 0.1 Pa background over the
0.6 m plane emits about 1.5e-4 kg/s, twice the jet.

**Check** (`tests/test_inflow.py`, opt-in):
- **Case:** a hypersonic N₂ stream at 2000 m/s and 50 K, with density rising linearly in r,
  in a 2 × 1 cm axisymmetric box.
- **Result:** SPARTA reproduces the density within 1–4% from r = 1.5 to 8.5 mm. The velocity
  is exact and the temperature is within 1%.
- **Two edge effects:**
  - **Outermost row (r = 9.5 mm):** 25% low. This is expected: `yhi` is an outflow, so gas
    diffuses out and none comes back.
  - **Row on the axis (r < 1 mm):** 14% high. **Not explained yet.** Check this before
    trusting near-axis DSMC densities. Possible causes are SPARTA's axisymmetric particle
    weighting (clone/delete) or how it places emitted particles on the innermost face.

### Where to put the coupling plane

Setup computes the gradient-length Knudsen number Kn_GLL = λ|dρ/dx|/ρ on the SU2 axis
nodes, and warns if the coupling plane is downstream of the point where it first goes above
`kn_breakdown` (default 0.05). Past that point the NS solution can't be trusted as an inflow.
For the 0.4 mm He capillary at 1 bar, breakdown is at about 4.7 D, so `x_couple_D = 4`.

To look at the profile without building a grid:

```bash
python handoff/breakdown_gll.py my_case.toml       # or --su2 path/to/flow.vtu
```

> **The check only looks at the axis.** In a nozzle plume, continuum breakdown starts off
> the axis:
> - in the expansion around the lip;
> - in the outer boundary layer, where λ approaches the layer thickness;
> - in the near-vacuum background, where λ is centimetres at 0.1 Pa.
>
> In the N₂ nozzle (13 mm exit), the axis stays continuum well past 100 mm, but every plane
> downstream of the exit has broken-down regions at larger r. A 2D criterion (a new
> `handoff/` version) is the next step.

> Downstream of breakdown the SU2 field is outside NS validity, and the SU2 jet collapses to
> background density within a few more diameters. A real free jet stays far denser there.
> Away from the coupling plane, the grid from setup is only a first guess. In the production
> run, let SPARTA refine it (`fix adapt` / `adapt_grid`) to the actual DSMC density.

---

## Case file: `[sparta]`

```toml
[sparta]
x_couple_D = 4                 # coupling plane / DSMC xlo, in throat diameters past the exit
base_grid = [200, 100]         # level-1 cells (x, r)
cells_per_mfp = 3              # refine until cell size <= lambda / this; optional
ppc_target = 20                # particles per refined cell, used for fnum; optional
fnum = "auto"                  # or a number; optional
timestep_s = "auto"            # or a number; optional
# dt_tau_frac = 0.2            # timestep <= this x mean collision time; optional
# dt_cell_frac = 0.5           # timestep <= this x cell transit time; optional
# kn_breakdown = 0.05          # Kn_GLL threshold for the coupling-plane check; optional
# breakdown_module = "handoff.breakdown_gll"  # which breakdown version to use; optional
# max_levels = 5               # optional cap below the cell-ID bit limit
# max_particles = 5e6          # cap on steady-state particles for the automatic fnum (~200 B each); optional
# run_steps = 1000             # steps in in.sparta; optional
# average_steps = 500          # average over the last this many steps; optional, default run_steps/2
nprocs = 20                    # not used yet (for the future sparta/run.py)
```

The `[case]`, `[geometry]` and `[gas]` sections are described in the
[main README](../README.md#case-file).

## Code

| File | Contents |
|---|---|
| `setup.py` | command-line entry point: domain, grid, fnum/dt, inflow, Kn check, writes the run directory |
| `inflow.py` | samples SU2 on the coupling plane, writes the `fix emit/face/file` table, Maxwellian fluxes |
| `grid.py` | refinement (`build`), SPARTA cell IDs, `read_grid` file writer |
| `gas.py` | reads `.species`/`.vss` files; VSS mean free path, viscosity and mean thermal speed |
| `data/` | `he`, `ar`, `d2` and `n2` species and VSS files |
| `../handoff/breakdown_gll.py` | Kn_GLL breakdown parameter (default `breakdown_module`); see [handoff/README.md](../handoff/README.md) |
| `../lib/su2_field.py` | SU2 `.vtu` reader and linear interpolator (used for the grid and the inflow) |

## Extending

- **New gas:** add `data/<gas>.species` and `data/<gas>.vss`, one species per file, with
  matching IDs. Add the SU2 entry too; see [su2/README.md](../su2/README.md#extending).
  In the `.species` file, `RotRel` and `VibRel` are relaxation *probabilities* per
  collision (1/Z, so Z_rot = 5 is `0.2`), not collision numbers.
- **Grid adaptation (next step):** let SPARTA refine the grid to its own density
  (`adapt_grid` / `fix adapt`). Away from the coupling plane, the grid from setup is only a
  first guess.
