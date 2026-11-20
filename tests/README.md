# Tests

All the project's tests live here.

```bash
.venv/bin/python -m pytest              # run everything (pytest.ini: testpaths = tests)
.venv/bin/python -m pytest -v           # one line per test
.venv/bin/python -m pytest -rs          # show why a test was skipped
RUN_SPARTA_TESTS=1 .venv/bin/python -m pytest tests/test_inflow.py   # include the SPARTA run
```

No test runs SPARTA unless `RUN_SPARTA_TESTS=1` is set. The SPARTA test is small, about
6000 particles and under a second, and it runs under a 2 GB memory limit (`ulimit -v`).
It is still opt-in, so a plain `pytest` never launches DSMC.

`pytest.ini` puts the repo root on the import path, so tests import project code directly,
for example `from handoff import breakdown_gll` or `from sparta.gas import Gas`.

| File | Tests |
|---|---|
| `test_breakdown_gll.py` | `handoff/breakdown_gll.py`: Kn_GLL breakdown parameter |
| `test_breakdown_2d.py` | `handoff/breakdown_2d.py`: mesh gradients exact for linear fields; Kn = λ/L for exponential density in x and r; speed term divided by the sound speed at rest; mass-flow quantile and plane mass flow; the report flags a broken-down plane and writes both files; on the N₂ nozzle, axis continuum but outer jet not (skipped without that run) |
| `test_cd_nozzle.py` | `su2/geom/cd_nozzle.py`: wall profile (ends, throat, smoothness, cone angles, bad input) and gmsh mesh (cell counts, boundary markers, cell validity and skew, wall nodes on the profile, first wall cell); the lip-line relaxation (stations on axial nodes, bad keys rejected, same cell count, lip cell at the exit and outlet); `exit_cell_um` sets the axial cell either side of the exit plane; also that the capillary mesh still builds |
| `test_mesh_quality.py` | `su2/mesh_quality.py`: square cells are perfect, known aspect ratio and size jump on a stretched grid, skewed-cell angles, the report's regions, lip and checks, triangles rejected |
| `test_su2_config.py` | `su2/setup.py`: ideal choked mass flow, and the convergence settings (`CONV_FIELD= RMS_DENSITY` with a residual level, no Cauchy keys) |
| `test_su2_coarsen.py` | mesh sequencing: `--coarsen` scales cell counts and sizes (minimum 2 cells, default throat cell), run directory names; `run_case.sh su2_refine` needs `FROM` and the coarse restart |
| `test_remap_restart.py` | `su2/remap_restart.py`: binary restart round trip, linear fields interpolated exactly onto new nodes, non-binary files rejected |
| `test_gases.py` | each SU2 gas has SPARTA data with matching IDs; R matches the molecular mass; SU2 viscosity matches VSS (he/ar near room temperature, d2 over 20–400 K, n2 over 50–400 K); D2 and N2 against measured viscosity; RotRel/VibRel are probabilities (1/Z) |
| `test_capillary_massflow.py` | `lib/capillary.py` analytical capillary flow, and SU2 capillary mass flow against it |
| `test_handoff.py` | `handoff/handoff.py`: automatic and forced coupling plane, errors when there is no breakdown or no room, reading `handoff.toml`; `run_case.sh` parses and refuses to start without `CASE` |
| `test_sparta_inputs.py` | `sparta/setup.py` input generation: the hold, grid adaptation and averaging in the right order; hold images (size, zoom, only during the hold); continuation input (string checks, never runs SPARTA) |
| `test_sparta_chamber.py` | `sparta/chamber.py`: outflow default leaves `in.sparta` unchanged; pump fraction ↔ chamber pressure balance; bad settings stop; reservoir composition and the renamed beam species; pump and reservoir commands in `in.sparta` and `in.continue`, per-species dump columns |
| `test_inflow.py` | `sparta/inflow.py`: Maxwellian fluxes, plane mass flows, inflow file format; one opt-in test runs SPARTA on the inflow |
| `plot_test_breakdown_gll.py` | not a test; one diagnostic plot per test in `test_breakdown_gll.py` |
| `plots/` | output of the plot script |

## Naming

- **Test files:** call them `test_<module>.py`, after the module they cover. pytest
  collects any file named `test_*.py` or `*_test.py`.
- **Other files here:** helpers and plot scripts must not match those patterns, or pytest
  will import them on every run. `plot_test_<module>.py` is safe.

## test_breakdown_gll.py

Checks the breakdown parameter against cases with known answers:
- **Known profiles:** exponential, free-jet x⁻², and uniform density.
- **Mean-free-path scaling** with temperature.
- **Bad input:** non-increasing `x` must raise ValueError.
- **Threshold search** (`first_breakdown`) and interpolation (`kn_at`).
- **Axis-node extraction** (`axis_profile`), using a fake field.

One test runs on a real SU2 solution (`runs/ASDEX_capil_01/su2/flow.vtu`) and is skipped if
that file is missing.

To test another breakdown version, add a `test_<version>.py` that imports it, for example
`from handoff import breakdown_2d`.

## test_capillary_massflow.py

Compares the SU2 mass flow through the capillary with the analytical models in
`lib/capillary.py`. Both models use an isentropic entrance from the plenum, laminar friction
f = 64/Re with Sutherland viscosity, and a choked exit:

| Model | Wall | Chokes at |
|---|---|---|
| `fanno_laminar` | adiabatic (Fanno flow) | M = 1 |
| `isothermal_laminar` | isothermal at the wall temperature | M = 1/√γ |

Both models take an optional entrance loss, `k_entrance`. It accounts for the developing
flow near the inlet (Hagenbach factor, about 1.25 for a circular tube).

**Model checks:**
- **Closed forms:** the Fanno equation integrates to its closed form, and constant-viscosity
  runs reproduce the textbook relations.
- **Orifice limits:** the choked-orifice formula is right, and a very short tube approaches it.
- **Trends:** mass flow falls with tube length and with the entrance loss, and rises roughly
  as p₀² (laminar flow).
- **Wall temperature:** an isothermal wall gives less flow than an adiabatic one.

**SU2 checks**, on the runs in `REFERENCE_RUNS` (missing runs are skipped):
- **Mass flow measured from the solution:** ρu·2πr is integrated over the node columns at 12
  sections along the tube, read straight from `flow.vtu`. This is the flow through the
  capillary. It is not SU2's "total" `Avg_Massflow`, which is inlet + outlet.
- **Conservation:** the sections agree within 2%.
- **Matches SU2's own number:** the inlet section is within 1% of the inlet `Avg_Massflow`
  in `su2.log`. This is skipped if `su2.log` is older than `flow.vtu`, because then it came
  from a different run.
- **Below the adiabatic limit:** the flow is lower than the fully developed Fanno value.
- **Close to the best model:** the flow is within 5% of the isothermal model with the
  entrance loss. This is the closest model, because SU2's wall is isothermal and SU2
  resolves the entrance region.

For the 0.4 mm × 66 mm He capillary at 1 bar:

| | ṁ (kg/s) | SU2 / model |
|---|---|---|
| choked orifice (no friction) | 1.156e-5 | 0.25 |
| Fanno, fully developed | 3.233e-6 | 0.90 |
| Fanno + entrance loss | 3.093e-6 | 0.94 |
| isothermal, fully developed | 3.074e-6 | 0.95 |
| isothermal + entrance loss | 2.956e-6 | 0.99 |
| SU2 (`ASDEX_capil_02`) | 2.915e-6 | |

`capillary_100um` and `capillary_400um` aren't references. The first is a 300-iteration
smoke run. The second doesn't conserve mass along the tube.

## test_inflow.py

Checks the SU2 → SPARTA inflow (`sparta/inflow.py`):
- **Maxwellian flux limits:** n·c̄/4 at rest, n·u when hypersonic, about 0 for strong
  outflow, and inward − outward = n·u.
- **Plane integrals:** the annulus integral, and net vs emitted mass flow for a uniform
  hypersonic plane.
- **File format:** the `fix emit/face/file` table (section name, NI/NV/VALUES/IMESH lines,
  rows, trot = T).
- **SPARTA reproduces the inflow** (opt-in, `RUN_SPARTA_TESTS=1`):
  - **Case:** hypersonic N₂, 2000 m/s and 50 K, with density linear in r, in a 2 × 1 cm
    axisymmetric box.
  - **Interior rows:** density within 5% (measured 1–4%).
  - **Axis row:** within 20% (measured 14% high; open item in
    [sparta/README.md](../sparta/README.md#inflow-at-the-coupling-plane)).
  - **Flow:** median velocity within 1%, temperature within 10%.

  Set `SPA_SERIAL` if `spa_serial` is not at `~/code/sparta-24Sep2025/src/spa_serial`.

## Test plots

```bash
python tests/plot_test_breakdown_gll.py                    # default version, ASDEX_capil_01
python tests/plot_test_breakdown_gll.py --su2 runs/ASDEX_capil_01/su2/flow.vtu runs/ASDEX_capil_02/su2/flow.vtu
python tests/plot_test_breakdown_gll.py --module handoff.breakdown_gll --module handoff.breakdown_2d
python tests/plot_test_breakdown_gll.py --out /tmp/plots --case my_case.toml
```

This writes `plots/<test name>.png`, one per test. Each plot shows the input data, the result
from each version, and the expected answer (black dashed line or cross), so you can see how
close a result is to failing.
- **Synthetic tests:** seven tests use data generated in the script, mirroring the parameters
  in `test_breakdown_gll.py`. Keep the two in sync.
- **SU2 test:** plots Kn_GLL, ρ and T along the axis of each `--su2` file. `--case` (default
  `config.toml`) supplies D, the gas and the coupling plane.

Each plot holds up to three series: versions × SU2 files. Versions that lack a helper a
test needs, such as `kn_gll`, are left out of that plot.
