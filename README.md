# SpartU2: NS–DSMC coupling

Simulates a gas jet expanding from a capillary or a converging–diverging nozzle into a
low-pressure chamber with two codes:

- **SU2** (Navier–Stokes, continuum) solves the dense flow inside the capillary or nozzle and just past its exit.
- **SPARTA** (DSMC) takes over downstream of a coupling plane, where the plume becomes rarefied.

```
         x = -L            x = 0      x = x_c                     x = chamb_length
plenum ─► +=================+----------:----------------------------+  r = chamb_height
          |    capillary    |          :                            |
          +=================+----------:----------------------------+  r = 0 (axis)
          |<------------------- SU2 (whole chamber) --------------->|
                                       |<----- SPARTA (DSMC) ------>|
```

SU2 solves the whole chamber, but its solution is only used upstream of the coupling plane
x_c, where the flow is still continuum. The DSMC domain starts at x_c.

The package does three things, each a small Python step that writes plain files for the next:

1. **SU2 setup:** mesh and `su2.cfg` for the nozzle and chamber.
2. **Handoff:** finds where the continuum breaks down in the SU2 solution, places the
   coupling plane, and writes the SU2 density, velocity and temperature there as the DSMC inflow.
3. **SPARTA setup:** DSMC grid sized from the SU2 solution, and the SPARTA input files.

Python never launches a solver. The bash script `run_case.sh` runs the Python steps and
launches SU2 and SPARTA with MPI, on a PBS cluster or on a workstation.

**Status:**
- **SU2 stage:** complete for the capillary and the CD nozzle.
- **Handoff and SPARTA stage:** working; the N₂ CD nozzle has been run for 26,000 DSMC steps.
  SPARTA reproduces the SU2 inflow within 1–4% away from the axis and the outer boundary,
  and the plume matches SU2 within 2–13% on the axis over the first 25 mm.

| Stage | Details |
|---|---|
| SU2 | [su2/README.md](su2/README.md): mesh, boundary conditions, solver, outputs |
| Handoff | [handoff/README.md](handoff/README.md): breakdown parameter, coupling plane, inflow |
| SPARTA | [sparta/README.md](sparta/README.md): domain, grid refinement, fnum/timestep and memory, hold and grid adaptation |

---

## Requirements

| Tool | Notes |
|---|---|
| Python ≥ 3.11 | the SU2 scripts use only the standard library; the SPARTA stage also needs numpy and scipy (`pip install -r requirements.txt`, already installed in `.venv`) |
| [gmsh](https://gmsh.info) | must be on `PATH` and able to write the `su2` mesh format |
| SU2 (`SU2_CFD`) | tested with v8.5.0; must be on `PATH`, or set `SU2_CFD=/path/to/SU2_CFD` |
| SPARTA (`spa_mpi` / `spa_serial`) | tested with the 24 Sep 2025 release, default build (32-bit cell IDs) |
| MPI (`mpirun`) | `run_case.sh` launches both solvers with it; set `MPIRUN` to change the launcher |

## Layout

```
config.toml          case file: one per simulation (copy and edit)
examples/            example case files (cd_nozzle_n2.toml)
requirements.txt     Python packages for the SPARTA stage and tests
pytest.ini           test configuration
tests/               all tests + test plots   -> tests/README.md
run_case.sh          bash driver: runs the stages and launches the solvers (PBS or local)
runnit.py            the Python setup steps, one line each, for interactive use
handoff/             handoff.py + breakdown-parameter versions   -> handoff/README.md
lib/common.py        shared helpers: TOML access, logging, mesh grading
lib/su2_field.py     read an SU2 flow.vtu and interpolate it at any point
lib/capillary.py     analytical capillary mass flow (Fanno / isothermal, laminar) and SU2 tube mass flow
su2/                 SU2 stage: setup.py, geom/   -> su2/README.md
sparta/              SPARTA stage: setup.py, grid.py, inflow.py, gas.py, data/   -> sparta/README.md
runs/<case>/su2/     generated run directories, one per case name
runs/<case>/handoff/
runs/<case>/sparta/
Archive/             older bash versions of the SU2 scripts, kept for reference only
```

---

## Workflow

Everything runs from the repo root through `run_case.sh`. Pick the stages with `STAGES`:

```bash
# PBS: the whole chain in one job
qsub -v CASE=my_case.toml run_case.sh
# PBS: only the DSMC part, after checking the SU2 solution
qsub -v CASE=my_case.toml,STAGES="handoff sparta_setup sparta" run_case.sh
# workstation: same script, ranks from NP
CASE=my_case.toml STAGES="su2_setup su2" NP=8 bash run_case.sh
```

| Stage | What it does | Writes |
|---|---|---|
| `su2_setup` | `python su2/setup.py CASE` | `runs/<name>/su2/`: mesh, `su2.cfg` |
| `su2` | `mpirun SU2_CFD su2.cfg` from the initial state | `flow.vtu`, `restart_flow.dat`, `history.csv`, `su2.log` |
| `su2_restart` | the same from `restart_flow.dat`; earlier results copied to `su2/archive/<n>/` | |
| `su2_refine` | the same from a coarser run's solution (`FROM=N`, `su2_coarse<N>/`), interpolated onto this mesh | |
| `handoff` | `python handoff/handoff.py CASE` | `runs/<name>/handoff/`: `handoff.toml`, `inflow.sparta`, `report.txt`/`.png` |
| `sparta_setup` | `python sparta/setup.py CASE` | `runs/<name>/sparta/`: grid, `in.sparta`, `in.continue`, `setup.log` |
| `sparta` | `mpirun spa_mpi -in in.sparta` | `log.sparta`, `flow.<step>.dump`, `restart.<step>.sparta` |
| `sparta_restart` | `mpirun spa_mpi -in in.continue` from the newest restart, up to `TARGET` | `log.<from>-<to>.sparta`, more dumps and restarts |

Each line of the table is a single command, so a failed stage can be rerun by hand with
the command in the second column. The script stops at the first failing stage. Its header
lists the settings: `NP`, `MPIRUN`, `SU2_ITER`, `TARGET`, `RUNS` and the executable names.
`COARSEN=N` runs the SU2 stages on an N-times coarser mesh in `su2_coarse<N>/`, to start
the full mesh from with `su2_refine` ([su2/README.md](su2/README.md#starting-from-a-coarse-mesh)).
For a PBS job, set the cores and walltime in the `#PBS` lines (Torque syntax, with the
PBS Pro form in a comment).

Read `runs/<name>/handoff/report.txt` (and `report.png`) after the handoff: it checks the
coupling plane against the full 2D breakdown criterion and the SU2 mass flow, and marks
problems with `[!!]` ([handoff/README.md](handoff/README.md#the-report)).

Check SU2 convergence (`history.csv`) before the handoff. The coupling plane, the inflow and
the SPARTA grid all come from `flow.vtu`, and an unconverged solution gives meaningless ones.
If SU2 is stopped while writing, `flow.vtu` can be truncated; the handoff then says so.

Before the `sparta` stage, read the particle count and memory estimate in
`runs/<name>/sparta/setup.log` and request at least that much memory.
`max_particles` (default 5e6, about 1 GB) caps the automatic `fnum`; see
[sparta/README.md](sparta/README.md#fnum-timestep-and-memory). On a workstation, a hand-set
`fnum` that is too small can exhaust memory; under WSL, a `memory=` limit in `.wslconfig`
keeps that from taking Windows down.

Every setup script copies the case file into its run directory as `case.toml`, so each run
records its own inputs. A run with the same `[case] name` overwrites the earlier one.

Files that setup generates, such as `su2.cfg`, `mesh.geo`, `grid.sparta` and `in.grid`, start
with a "generated" comment. Change the case file and run setup again instead of editing them.

---

## Case file

A case is a single TOML file. Each stage reads its own section, which is documented in
that stage's README:

- `[su2]`: see [su2/README.md](su2/README.md#case-file-su2)
- `[sparta]`: see [sparta/README.md](sparta/README.md#case-file-sparta)

These sections are shared by both stages:

```toml
[case]
name = "ASDEX_capil_02"        # run directory name

[geometry]
noz_type = "capillary"         # or "cd_nozzle", or "profile" (wall from an (x, r) table); -> su2/geom/<noz_type>.py, meshed by su2/mesh.py
throat_diam_mm = 0.4           # capillary inner diameter D; many lengths are given in units of D
throat_length_mm = 66          # capillary length L
dim = 2                        # only 2 / "axial" is implemented
symmetry = "axial"
chamb_length_cm = 5            # chamber length downstream of the exit (both domains end here)
chamb_height_cm = 2.5          # chamber radius (must be > D/2)

[gas]
species_file = "he.species"    # the file stem picks the gas in both codes: "he", "ar", "d2" or "n2"
plen_pres_pa = 1.0e5           # plenum (inlet total) pressure
plen_temp_K = 300.15           # plenum (inlet total) temperature
```

A CD nozzle replaces the capillary's `throat_length_mm` tube with these keys (see
[su2/README.md](su2/README.md#geometry-and-mesh-noz_type--cd_nozzle) and
`examples/cd_nozzle_n2.toml`):

```toml
noz_type = "cd_nozzle"
throat_diam_mm = 0.5           # D_t
inlet_diam_mm = 3.0
exit_diam_mm = 2.0
conv_half_angle_deg = 30
div_half_angle_deg = 15
throat_rc_D = 1.0              # throat blending arcs, in D_t; optional, default 1
throat_length_mm = 0.0         # straight throat section; optional, default 0
```

**Adding a gas** takes two entries, one per code:
- **SU2:** an entry in `GASES` in `su2/setup.py`.
- **SPARTA:** `<gas>.species` and `<gas>.vss` files in `sparta/data/`.

Keep the SU2 Sutherland fit consistent with the VSS parameters.

---

## From Python

The three setup steps are plain functions, for use in a notebook or `runnit.py`:

```python
from su2.setup import setup as su2_setup
from handoff.handoff import handoff
from sparta.setup import setup as sparta_setup

su2_setup("my_case.toml")      # -> runs/<name>/su2/ (remeshes: don't call it before an SU2 restart)
handoff("my_case.toml")        # -> runs/<name>/handoff/ (needs su2/flow.vtu)
sparta_setup("my_case.toml")   # -> runs/<name>/sparta/ (needs the handoff; overwrites in.sparta edits)
```

`runnit.py` lists these one per line; comment lines in or out. The solvers are then started
with `run_case.sh`.

## Tests

```bash
.venv/bin/python -m pytest          # runs everything in tests/ (pytest.ini puts the repo root on the path)
```

`tests/test_breakdown_gll.py` checks the breakdown parameter against profiles with known
answers: exponential, power-law and uniform density. It also checks the threshold search and
the extraction of axis nodes. One test runs on a real SU2 solution
(`runs/ASDEX_capil_01`) and is skipped if that file is missing.
`tests/test_capillary_massflow.py` checks the SU2 capillary mass flow against Fanno and
isothermal friction flow. `tests/test_handoff.py` checks the coupling-plane choice and
that `run_case.sh` parses. `tests/test_inflow.py` checks the SU2 → SPARTA inflow. Its one
test that runs SPARTA is off unless you set `RUN_SPARTA_TESTS=1`. See
[tests/README.md](tests/README.md).

## Not done yet

- Particle weighting that survives grid adaptation: with `global weight cell radius`,
  coarsening cells doubles the weight of the particles in them (see
  [sparta/README.md](sparta/README.md#transient-hold-and-grid-adaptation))
- A separate SPARTA domain size: DSMC currently fills the whole SU2 chamber, which is
  expensive for large chambers
- Coupling for a jet that is partly rarefied at the exit: the 2D breakdown criterion
  (`handoff.breakdown_2d`) shows the N₂ nozzle's outer boundary layer is already transitional
  at the exit plane ([handoff/README.md](handoff/README.md#the-report))
- An SU2 initial-pressure option: starting the domain at the 0.1 Pa back pressure can
  leave a stalled low-density layer on the nozzle wall ([su2/README.md](su2/README.md#checking-convergence))
- 3D, or geometries that are not axisymmetric
