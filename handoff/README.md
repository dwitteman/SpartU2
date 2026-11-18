# Handoff: SU2 → SPARTA

The step between the two solvers. It reads the SU2 solution, finds where the continuum
breaks down, places the coupling plane, writes the SU2 state on that plane as the DSMC
inflow, and writes a report to sanity-check all of it. The
[SPARTA setup](../sparta/README.md) reads its output.

```bash
python handoff/handoff.py my_case.toml                     # stage "handoff" in run_case.sh
python handoff/handoff.py my_case.toml --su2 other/flow.vtu
python handoff/handoff.py my_case.toml --runs /scratch/runs
```

It reads `runs/<name>/su2/flow.vtu` (or `--su2`) and writes `runs/<name>/handoff/`:

| File | Contents |
|---|---|
| `handoff.toml` | coupling plane (m and D, auto or forced), Kn there and the breakdown point, SU2 net and emitted mass flow, the SU2 file used. `sparta/setup.py` reads it |
| `inflow.sparta` | SU2 n, velocity and T on the plane, a `fix emit/face/file` table (see [sparta/README.md](../sparta/README.md#inflow-at-the-coupling-plane)) |
| `report.txt` | the sanity-check report (below) |
| `report.png` | the same as four plots; skipped with a note if matplotlib isn't installed |
| `kn_profile.csv` | x [m], x/D and the breakdown module's Kn along x, to plot or check by hand |

It takes a few seconds on one core and never launches a solver. Run it again whenever
SU2 changes, then `sparta/setup.py`.

## Breakdown criteria

Two versions, chosen with `breakdown_module`:

| Module | Kn_GLL | One number per x, used to place the plane |
|---|---|---|
| `handoff.breakdown_gll` (default) | λ\|dρ/dx\|/ρ, density only, on the SU2 axis nodes | Kn on the axis |
| `handoff.breakdown_2d` | Boyd's full form at every SU2 node: λ·max(\|∇ρ\|/ρ, \|∇T\|/T, \|∇V\|/max(V, a)) | the Kn exceeded by `mass_fraction` (default 1%) of the mass flow through the plane |

In `breakdown_2d`:
- **Gradients** come from the SU2 mesh itself: the exact gradient of the linear interpolant on each
  Delaunay triangle, averaged to the nodes by area.
- **Speed term:** dividing by max(V, a), with a the speed of sound, keeps it finite in gas at rest.
- **Placing the plane:** every plume plane has some broken-down gas (the lip, the jet edge, the
  background), so a plain maximum over the plane would always fail. Instead, `first_breakdown`
  finds the first plane where more than `mass_fraction` of the jet's mass flow passes through
  gas over the threshold.

## The report

`report.txt` always uses the full 2D Kn_GLL as an independent check, whichever module placed
the plane. It contains:
- **The choice:** which module placed the plane, the threshold, the breakdown point and the plane.
- **A table along the jet** at fixed x/D: axis Kn (density only), axis Kn (2D), the largest Kn
  on the plane, the share of the plane's mass flow over the threshold, and the SU2 mass flow.
- **At the coupling plane:** Kn on the axis and its maximum; the radius beyond which Kn is over
  the threshold; which term (ρ, T or V) sets Kn there; the SU2 net and SPARTA emitted mass flow.
- **Checks**, each marked `[ok]` or `[!!]` (flagged ones are also printed by the handoff):
  - the plane is upstream of the breakdown point;
  - at most `mass_fraction` of the plane's mass flow is over the threshold;
  - the SU2 mass flow at the plane is within 5% of the flow at the nozzle exit;
  - SPARTA's one-sided emission is less than 1.5× the SU2 net flow (otherwise a dense background
    sits on the plane);
  - the plane state is physical (n ≥ 0, T > 1 K), with its ranges.

`report.png` has four panels:
- a Kn map of the near field, with the threshold contour and the plane;
- Kn along the jet for the axis (both versions), the plane maximum and the module's profile;
- Kn across the plane;
- the cumulative mass flow across the plane, with the part over the threshold shaded.

**The N₂ CD nozzle** (13 mm exit), with the 0.01 threshold:
- **Axis:** Kn_GLL is about 1e-3 on the axis, so the axis is continuum.
- **Outer jet:** beyond r ≈ 2.9 mm the nozzle's boundary layer is hot and thin (T 100–200 K,
  n ~ 1e21 /m³, λ 0.25–0.7 mm), with Kn 0.1–0.5 driven by the velocity and density gradients.
- **Mass flow over the threshold:** 42% at 10 D, and about 20–25% even at the 0.05 threshold.
  It is like this from the exit plane on.

So with `breakdown_2d` and `"auto"` there is no continuum plane to couple at. The outer part
of the jet would need DSMC from inside the nozzle.

## Case file

```toml
[sparta]
x_couple_D = "auto"            # or a number: force the plane, in throat diameters past the exit
couple_margin_D = 1            # with "auto": plane this many D upstream of breakdown; optional
kn_breakdown = 0.01            # Kn_GLL threshold; optional, default the module's KN_BREAKDOWN (0.01)
breakdown_module = "handoff.breakdown_gll"   # or "handoff.breakdown_2d"; optional

[breakdown]                    # options passed to the module; optional
mass_fraction = 0.01           # breakdown_2d: share of the plane's mass flow allowed over the threshold
```

- **`"auto"`** (the default) puts the plane `couple_margin_D` throat diameters upstream of the
  module's first breakdown point. It stops with a message if there is none, or if it is closer
  to the exit than the margin.
- **A number** forces the plane there; the handoff warns if that is past the breakdown point.
- **Threshold:** the default 0.01 follows Bi et al.; Boyd, Chen & Chandler (1995) used 0.05.

## Code

| File | Contents |
|---|---|
| `handoff.py` | command-line entry point: breakdown profile, coupling plane (`choose_plane`), inflow, report; `read_handoff` for the SPARTA setup |
| `breakdown_gll.py` | density-only Kn_GLL on the axis (default module) |
| `breakdown_2d.py` | Boyd's full Kn_GLL on the SU2 mesh (`kn_field`), Kn and mass flow on radial lines (`plane_lines`), mass-flow quantile |
| `report.py` | `report.txt` and `report.png` |

The plane sampling and the inflow file format are in `sparta/inflow.py`.

## Writing a new breakdown version

Copy one of the modules to a new file named after the method, change the calculation, and
select it with `breakdown_module`. A module must define:

| Name | Signature | Returns |
|---|---|---|
| `KN_BREAKDOWN` | constant | default threshold |
| `knudsen_profile` | `(field, gas, **options)` | `(x, kn)`: one value per axial station, sorted by x; options come from `[breakdown]` |
| `kn_at` | `(x, kn, x0)` | Kn at x0 |
| `first_breakdown` | `(x, kn, threshold)` | first x where Kn > threshold, or `None` |

`field` is a `lib.su2_field.Su2Field`:
- node coordinates are in `field.xy`;
- node values are in `field.nodes["rho"]`, `["T"]`, `["u"]` and `["v"]`;
- the Delaunay triangulation is `field.tri`.

`gas` is a `sparta.gas.Gas`, which provides `mean_free_path(n, T)` and `mean_speed(T)`. If a
module is missing any of the four names, the handoff stops and says which ones.

Give each version its own `tests/test_<name>.py`; see [tests/README.md](../tests/README.md).

To print a version's profile on its own:

```bash
python handoff/breakdown_gll.py my_case.toml   # axis Kn_GLL
python handoff/breakdown_2d.py my_case.toml    # mass-flow quantile of the 2D Kn_GLL per plane
```
