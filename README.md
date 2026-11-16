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

**Status:**
- **SU2 stage:** complete for the capillary and the CD nozzle.
- **SPARTA stage:** builds the DSMC domain, a grid sized from the SU2 solution, the SU2
  inflow on the coupling plane (`inflow.sparta`) and a run input (`in.sparta`).
  - **Tested:** SPARTA reproduces a given axisymmetric inflow, within 1–4% away from the
    axis and the outer boundary ([sparta/README.md](sparta/README.md#inflow-at-the-coupling-plane)).
  - **Not tested:** `in.sparta` for a real case has not been run yet.

| Stage | Details |
|---|---|
| SU2 | [su2/README.md](su2/README.md): mesh, boundary conditions, solver, outputs |
| SPARTA | [sparta/README.md](sparta/README.md): domain, grid refinement, fnum/timestep and memory, inflow, where to put the coupling plane |

---

## Requirements

| Tool | Notes |
|---|---|
| Python ≥ 3.11 | the SU2 scripts use only the standard library; the SPARTA stage also needs numpy and scipy (`pip install -r requirements.txt`, already installed in `.venv`) |
| [gmsh](https://gmsh.info) | must be on `PATH` and able to write the `su2` mesh format |
| SU2 (`SU2_CFD`) | tested with v8.5.0; must be on `PATH`, or set `SU2_CFD=/path/to/SU2_CFD` |
| SPARTA (`spa_mpi` / `spa_serial`) | tested with the 24 Sep 2025 release, default build (32-bit cell IDs) |
| MPI (`mpirun`) | used to launch SU2 |

## Layout

```
config.toml          case file: one per simulation (copy and edit)
examples/            example case files (cd_nozzle_n2.toml)
requirements.txt     Python packages for the SPARTA stage and tests
pytest.ini           test configuration
tests/               all tests + test plots   -> tests/README.md
handoff/             breakdown-parameter versions   -> handoff/README.md
lib/common.py        shared helpers: TOML access, logging, mesh grading
lib/su2_field.py     read an SU2 flow.vtu and interpolate it at any point
lib/capillary.py     analytical capillary mass flow (Fanno / isothermal, laminar) and SU2 tube mass flow
su2/                 SU2 stage: setup.py, run.py, geom/   -> su2/README.md
sparta/              SPARTA stage: setup.py, run.py, grid.py, inflow.py, gas.py, data/   -> sparta/README.md
runs/<case>/su2/     generated run directories, one per case name
runs/<case>/sparta/
Archive/             older bash versions of the SU2 scripts, kept for reference only
```

---

## Workflow

Run all commands from the repo root.

```bash
cp config.toml my_case.toml                      # 1. new case: edit [case] name and the inputs
python su2/setup.py my_case.toml                 # 2. SU2 mesh + config -> runs/<name>/su2/
python su2/run.py runs/<name>/su2                # 3. run SU2 (MPI ranks from [su2] nprocs)
python sparta/setup.py my_case.toml              # 4. DSMC domain, grid, inflow -> runs/<name>/sparta/
cd runs/<name>/sparta && spa_serial -in in.grid  # 5. check that SPARTA loads the grid
python sparta/run.py runs/<name>/sparta           # 6. coupled DSMC run, with a memory guard
```

Or edit `runnit.py` and run `python runnit.py`. It lists one line per action; comment
lines in or out to choose which stages run, from scratch or continued (see
[From Python](#from-python)). Both stages can continue from restart files; see
[su2/README.md](su2/README.md#restarting) and
[sparta/README.md](sparta/README.md#continuing-a-run).

Before step 6, read the particle count and memory estimate in `setup.log`. `max_particles`
(default 5e6, about 1 GB) caps the automatic `fnum`. A hand-set `fnum` that is too small
can exhaust memory and take WSL down with it. `sparta/run.py` refuses to start if the
estimate exceeds the memory limit (default half the RAM), and kills SPARTA if its actual
memory use goes over. See
[sparta/README.md](sparta/README.md#fnum-timestep-and-memory).

Every setup script copies the case file into its run directory as `case.toml`, so each run
records its own inputs. A run with the same `[case] name` overwrites the earlier one.

Files that setup generates, such as `su2.cfg`, `mesh.geo`, `grid.sparta` and `in.grid`, start
with a "generated" comment. Change the case file and run setup again instead of editing them.

Check that SU2 has converged before step 4. The SPARTA grid is sized from SU2's
`flow.vtu`, and an unconverged solution gives a meaningless grid.

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
noz_type = "capillary"         # or "cd_nozzle"; -> su2/geom/<noz_type>.py
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

Each stage can be called as a function as well as from the command line. A coupling
driver can chain them:

```python
from su2.setup import setup as su2_setup
from su2.run import run as su2_run
from sparta.setup import setup as sparta_setup
from sparta.run import run as sparta_run

d = su2_setup("my_case.toml")          # remeshes and resets su2.cfg (no restart)
if su2_run(d) != 0:
    raise SystemExit("SU2 failed")
s = sparta_setup("my_case.toml")
sparta_run(s)                          # memory-guarded; returns SPARTA's exit code
```

To continue instead of starting over, call the run functions on the existing run
directories with `restart=True`:

```python
su2_run("runs/<name>/su2", restart=True, iterations=20000)    # 20000 more SU2 iterations
sparta_setup("my_case.toml")           # only if SU2 changed: new inflow.sparta (overwrites in.sparta edits)
sparta_run("runs/<name>/sparta", nprocs=18, max_mem_gb=8, restart=True, target=30000)  # DSMC up to step 30000
```

`su2_run` returns 0 even if SU2 reached its iteration limit without converging. Check
convergence before trusting the SPARTA inputs. `su2_setup` remeshes, so don't call it
before an SU2 restart.

## Tests

```bash
.venv/bin/python -m pytest          # runs everything in tests/ (pytest.ini puts the repo root on the path)
```

`tests/test_breakdown_gll.py` checks the breakdown parameter against profiles with known
answers: exponential, power-law and uniform density. It also checks the threshold search and
the extraction of axis nodes. One test runs on a real SU2 solution
(`runs/ASDEX_capil_01`) and is skipped if that file is missing.
`tests/test_capillary_massflow.py` checks the SU2 capillary mass flow against Fanno and
isothermal friction flow. `tests/test_inflow.py` checks the SU2 → SPARTA inflow. Its one
test that runs SPARTA is off unless you set `RUN_SPARTA_TESTS=1`. See
[tests/README.md](tests/README.md).

## Not done yet

- A first real coupled SPARTA run (`in.sparta` is written but untested on a real case),
  and grid adaptation to the DSMC density
- A separate SPARTA domain size: DSMC currently fills the whole SU2 chamber, which is
  expensive for large chambers
- A breakdown criterion off the axis: the current one only looks at the axis, but in a
  nozzle plume breakdown starts at the lip and in the boundary layer
  ([sparta/README.md](sparta/README.md#where-to-put-the-coupling-plane))
- An SU2 initial-pressure option: starting the domain at the 0.1 Pa back pressure can
  leave a stalled low-density layer on the nozzle wall ([su2/README.md](su2/README.md#checking-convergence))
- 3D, or geometries that are not axisymmetric
