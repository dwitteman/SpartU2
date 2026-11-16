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
python sparta/run.py runs/<name>/sparta            # ranks from [sparta] nprocs -> sparta.log, flow.<step>.dump
python sparta/run.py runs/<name>/sparta -n 4 --max-mem-gb 4
python sparta/run.py runs/<name>/sparta -n 1 --input in.grid   # just load the grid
```

`run.py` has two memory guards, so a run can't exhaust RAM and take WSL down:
- **Before the run:** it refuses to start if the particle estimate in `setup.log` (about
  200 B per particle) exceeds the limit. Pass `--force` to override.
- **During the run:** it checks the total resident memory of all SPARTA ranks every second,
  and kills them all if it goes over the limit.

The limit is `--max-mem-gb`, or `[sparta] max_mem_gb`, or else half the machine's RAM.
Binaries come from `SPA_MPI`/`SPA_SERIAL`, then `PATH`, then
`~/code/sparta-24Sep2025/src`.

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

### Transient hold and grid adaptation

With `hold_steps` set (for example 25000), `in.sparta` runs in three phases:

1. **Hold:** `hold_steps` steps from an empty box with no averaging or dumps, so the jet can
   fill the domain. Stats and restart files are still written. During the last
   `adapt_window` steps (default 1000), the particle count in each cell is averaged.
2. **Adapt:** a single `adapt_grid` based on that averaged count. A restart file is written
   afterwards.
   - **Refine:** cells holding more than `adapt_refine_count` particles (default
     4 × `ppc_target`) are split 2×2.
   - **Coarsen:** groups of 4 sibling cells holding fewer than `adapt_coarsen_count`
     in total (default `ppc_target`) are merged.
   - **Rebalance:** the grid is then rebalanced across processes (`balance_grid rcb part`).
3. **Run:** `run_steps` more steps with averaged output every `output_every` steps.

The total length is `hold_steps` + `run_steps`.

`output_every` is counted from step 0, so with a hold it must divide both `hold_steps` and
`run_steps`. That way every averaged block, including the last, is written. By default it
is their greatest common divisor, for example 5000 for a 25000-step hold and a 10000-step run.

Limits of the adaptation:
- **Refinement depth:** refinement can't go deeper than the 32-bit cell-ID limit. A
  200×100 base grid with 6 levels is already there, so cells at the finest level stay as
  they are. In practice you mainly get:
  - empty background regions coarsened;
  - coarse cells that the plume has filled refined.

  To refine further, use a coarser `base_grid` or build SPARTA with 64-bit cell IDs (see Grid).
- **SU2 columns:** new cells get 0 in the SU2 columns of the dump (`g_su2_nrho`, `g_lambda`).
- **Weights:** particle weights are recomputed for new cells. Particles already in a split
  cell keep their weight until they move, which settles within a few steps.
- **Continuations:** `in.continue` doesn't repeat the hold or the adaptation. Continue from a
  restart written after it (the one at the end of the hold, or later).
- **Set `adapt = false`** to keep the hold without adapting.

Check in `log.sparta` after the hold:
- the `adapt_grid` summary (cells refined and coarsened, and the new cell count);
- that the particle count had levelled off before the hold ended.

### Continuing a run

`in.sparta` writes a restart file `restart.<step>.sparta` every `restart_every` steps and
at the end. `in.continue` picks up from the latest one and runs up to an absolute timestep:

```bash
python sparta/run.py runs/<name>/sparta --restart --target 30000   # memory-guarded, picks a new seed
mpirun -n N spa_mpi -in in.continue -var target 30000              # by hand
mpirun -n N spa_mpi -in in.continue -var target 50000 -var seed 777  # by hand, new seed
```

Without `--target`, `run.py --restart` runs `run_steps` past the latest restart.

**Continuing after SU2 has changed:** `in.continue` reads `inflow.sparta` fresh, so you can
update the inflow and continue the DSMC rather than starting from an empty box. Restart or
rerun SU2, run `sparta/setup.py` to rewrite `inflow.sparta`, then `run.py --restart`
(from Python: `su2_run(d, restart=True, iterations=...)`, `sparta_setup(case)`, then
`sparta_run(s, restart=True, target=...)`). The DSMC keeps the grid from
its restart file. A new grid only takes effect in a fresh run (`--sparta new`).

- **What carries over:** the restart file holds the grid, particles, species, mixture, global
  settings (fnum, weight, gridcut), timestep size and step number.
- **What `in.continue` redefines:** the collision model, the inflow and all output, with the
  same settings as `in.sparta`.
- **Process count:** it works with a different number of processes; `balance rcb cell`
  regroups the cells on reading.
- **Averages restart:** the `fix ave/grid` averages start again. The first dump after
  continuing comes at the next multiple of `output_every`.
- **First step:** `in.continue` takes one step before defining the dump. SPARTA always dumps on
  the first step of a run when it falls on a multiple of the dump interval, which would
  overwrite the last `flow.<step>.dump` with an empty average.
- **Disk:** each restart file is roughly 100 bytes per particle plus about 100 per cell, so
  several hundred MB for the N₂ nozzle. Delete old ones; `read_restart` only needs the latest.
- **Not portable:** restart files are binary. Continue on the same kind of machine.

The default `run_steps = 1000` is a smoke test. Steady state needs at least a few box
transits. `setup.log` gives the number of steps for one transit at the fastest inflow speed;
for large chambers it is tens of thousands.

## Outputs (`runs/<name>/sparta/`)

| File | Contents |
|---|---|
| `grid.sparta` | grid file for `read_grid`, with two extra columns: SU2 mean free path `lambda` and number density `su2_nrho` |
| `in.grid` | a small SPARTA input that loads the domain and grid and dumps them; there is no flow |
| `inflow.sparta` | the SU2 state on the coupling plane, as a `fix emit/face/file` table |
| `in.sparta` | the coupled DSMC run: grid, gas, inflow, averaging, dumps and restart files |
| `in.continue` | continues from the latest `restart.<step>.sparta` up to `-var target <step>` |
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
# hold_steps = 25000           # transient hold before averaging, then adapt the grid; optional, default 0
# adapt = true                 # adapt the grid after the hold; optional
# adapt_window = 1000          # average the particle count over the last this many hold steps; must divide hold_steps
# adapt_refine_count = 80      # refine cells with more particles than this; optional, default 4 x ppc_target
# adapt_coarsen_count = 20     # merge sibling cells with fewer in total; optional, default ppc_target
# output_every = 1000          # dump the averaged flow every this many steps; optional,
#                              # default run_steps, or gcd(hold_steps, run_steps) with a hold
# average_steps = 500          # average over the last this many steps before each dump; optional, default output_every/2
# restart_every = 5000         # write restart.<step>.sparta every this many steps; optional, default run_steps
# max_mem_gb = 6               # sparta/run.py memory limit; optional, default half the RAM
nprocs = 20                    # MPI ranks for sparta/run.py
```

The `[case]`, `[geometry]` and `[gas]` sections are described in the
[main README](../README.md#case-file).

## Code

| File | Contents |
|---|---|
| `setup.py` | command-line entry point: domain, grid, fnum/dt, inflow, Kn check, writes the run directory |
| `run.py` | runs SPARTA under mpirun with the memory guards; output tee'd to `sparta.log` |
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
