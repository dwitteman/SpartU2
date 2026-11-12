# Breakdown parameter versions

This folder holds the continuum-breakdown parameter used to place the NS–DSMC coupling
plane, kept apart from the rest of the code so that different versions can be tried side by
side.
Tests and test plots are in [tests/](../tests/README.md).

| File | Contents |
|---|---|
| `breakdown_gll.py` | gradient-length Knudsen number Kn_GLL = λ\|dρ/dx\|/ρ on the SU2 axis nodes (default) |

`breakdown_gll` only looks at the axis. In nozzle plumes, breakdown starts off the axis:
- at the lip;
- in the outer boundary layer;
- in the near-vacuum background.

A 2D version that also uses the T and |V| gradients (Boyd's full Kn_GLL) is the obvious
next one to add. `sparta/README.md` has the details.

## Choosing a version

`sparta/setup.py` imports the module named in the case file:

```toml
[sparta]
breakdown_module = "handoff.breakdown_gll"   # default
kn_breakdown = 0.05                          # optional; default is the module's KN_BREAKDOWN
```

## Writing a new version

Copy `breakdown_gll.py` to a new file named after the method, for example
`breakdown_2d.py`, and change the calculation. To work with `sparta/setup.py`, a module must define:

| Name | Signature | Returns |
|---|---|---|
| `KN_BREAKDOWN` | constant | default threshold |
| `axis_knudsen` | `(field, gas)` | `(x, kn)`: arrays sorted by x along the axis |
| `kn_at` | `(x, kn, x0)` | Kn at x0 |
| `first_breakdown` | `(x, kn, threshold)` | first x where Kn > threshold, or `None` |

`field` is a `lib.su2_field.Su2Field`, with node coordinates in `field.xy` and node values in
`field.nodes["rho"]`, `["T"]`, `["u"]` and `["v"]`. `gas` is a `sparta.gas.Gas`, which provides
`mean_free_path(n, T)` and `mean_speed(T)`. If a module is missing any of the four names,
setup stops and says which ones.

Give each version its own `tests/test_<name>.py`; see [tests/README.md](../tests/README.md).

To run a version on its own:

```bash
python handoff/breakdown_gll.py my_case.toml   # prints the axis Kn_GLL profile
```
