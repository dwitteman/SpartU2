#!/usr/bin/env python3
"""One diagnostic plot per test in tests/test_breakdown_gll.py.

Each figure shows the data the test uses, the result from each breakdown version,
and (where there is one) the exact answer, so a pass/fail can be seen rather than
taken on trust. Seven tests use synthetic data generated here; the SU2 test uses
real SU2 output, and any number of flow.vtu files can be given.

Usage:
  python tests/plot_test_breakdown_gll.py                                  # default version + ASDEX_capil_01
  python tests/plot_test_breakdown_gll.py --module handoff.breakdown_gll --module handoff.breakdown_2d
  python tests/plot_test_breakdown_gll.py --su2 runs/ASDEX_capil_01/su2/flow.vtu runs/ASDEX_capil_02/su2/flow.vtu
  python tests/plot_test_breakdown_gll.py --out /tmp/plots --case my_case.toml

Writes <out>/<test name>.png (default out: tests/plots/).
The synthetic test parameters mirror test_breakdown_gll.py; keep the two in sync.
"""
import argparse
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from lib.common import get, load_case, log  # noqa: E402
from sparta.gas import Gas, load_gas  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HE = Gas(species="He", mass=6.6342e-27, d_ref=2.33e-10, omega=0.66, T_ref=273.15, alpha=1.0)

# ---- style: reference palette, light mode ---------------------------------------
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # fixed order; validated all-pairs for 3 slots
REF = dict(color=INK, ls=(0, (4, 3)), lw=1.4, zorder=5)  # exact answer, drawn on top
LINE = dict(lw=1.8)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
})


def colour(i):
    if i >= len(SERIES):
        raise SystemExit(f"more than {len(SERIES)} series in one plot: compare fewer "
                         f"versions/SU2 files at once")
    return SERIES[i]


def name(mod):
    return mod.__name__.rsplit(".", 1)[-1]


def with_fn(mods, fn):
    """Versions that provide function `fn` (only the setup API is required of a version)."""
    return [m for m in mods if hasattr(m, fn)]


def finish(fig, out, test, note=""):
    if note:
        fig.text(0.01, 0.01, note, color=INK2, fontsize=9, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.04 if note else 0, 1, 1))
    path = out / f"{test}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log(f"wrote {path}")


def rel_err(a, b):
    return np.abs(a - b) / np.abs(b)


# ---- one function per test ------------------------------------------------------------

def plot_exponential(mods, out):
    L, rho0, T = 1e-3, 1e-2, 300.0
    x = np.linspace(0, 5e-3, 5001)
    rho = rho0 * np.exp(-x / L)
    exact = HE.mean_free_path(rho / HE.mass, T) / L
    fig, (a, b) = plt.subplots(2, 1, figsize=(7, 6), sharex=True, height_ratios=[2, 1])
    a.plot(x * 1e3, exact, label="exact: λ / L", **REF)
    worst = []
    for i, m in enumerate(with_fn(mods, "kn_gll")):
        kn = m.kn_gll(x, rho, np.full_like(x, T), HE)
        a.plot(x * 1e3, kn, color=colour(i), label=name(m), **LINE)
        e = rel_err(kn[1:-1], exact[1:-1])
        b.plot(x[1:-1] * 1e3, e, color=colour(i), **LINE)
        worst.append(f"{name(m)} {e.max():.1e}")
    a.set(yscale="log", ylabel="Kn_GLL", title="Exponential density ρ = ρ₀ exp(−x/L), L = 1 mm")
    a.legend()
    b.axhline(1e-5, color=MUTED, lw=1, ls=":")
    b.text(x[0] * 1e3, 1e-5, "tolerance 1e-5", color=INK2, va="bottom", ha="left", fontsize=9)
    b.set(yscale="log", xlabel="x [mm]", ylabel="relative error")
    finish(fig, out, "test_exponential_density_gives_lambda_over_L",
           "max relative error (end points excluded): " + ", ".join(worst))


def plot_power_law(mods, out):
    x = np.geomspace(1e-4, 1e-2, 4001)
    rho = 1e-8 * x**-2
    T = np.full_like(x, 50.0)
    exact = 2 * HE.mean_free_path(rho / HE.mass, T) / x
    fig, (a, b) = plt.subplots(2, 1, figsize=(7, 6), sharex=True, height_ratios=[2, 1])
    a.plot(x * 1e3, exact, label="exact: 2λ / x", **REF)
    worst = []
    for i, m in enumerate(with_fn(mods, "kn_gll")):
        kn = m.kn_gll(x, rho, T, HE)
        a.plot(x * 1e3, kn, color=colour(i), label=name(m), **LINE)
        e = rel_err(kn[1:-1], exact[1:-1])
        b.plot(x[1:-1] * 1e3, e, color=colour(i), **LINE)
        mono = "increasing" if np.all(np.diff(kn) > 0) else "NOT increasing"
        worst.append(f"{name(m)} {e.max():.1e} ({mono})")
    a.set(xscale="log", yscale="log", ylabel="Kn_GLL",
          title="Free-jet density ρ ∝ x⁻², T = 50 K: Kn grows like x")
    a.legend()
    b.axhline(1e-4, color=MUTED, lw=1, ls=":")
    b.text(x[0] * 1e3, 1e-4, "tolerance 1e-4", color=INK2, va="bottom", ha="left", fontsize=9)
    b.set(xscale="log", yscale="log", xlabel="x [mm]", ylabel="relative error")
    finish(fig, out, "test_free_jet_power_law",
           "max relative error (end points excluded): " + ", ".join(worst))


def plot_uniform(mods, out):
    x = np.linspace(0, 1, 11)
    fig, a = plt.subplots(figsize=(7, 3.8))
    worst = []
    for i, m in enumerate(with_fn(mods, "kn_gll")):
        kn = m.kn_gll(x, np.full(11, 1e-3), np.full(11, 300.0), HE)
        a.plot(x, kn, color=colour(i), marker="o", ms=6, label=name(m), **LINE)
        worst.append(f"{name(m)} {np.abs(kn).max():.1e}")
    a.axhline(1e-15, color=MUTED, lw=1, ls=":")
    a.text(0, 1e-15, "tolerance 1e-15", color=INK2, va="bottom", ha="left", fontsize=9)
    a.set(ylim=(-0.2e-15, 1.3e-15), xlabel="x [m]", ylabel="Kn_GLL",
          title="Uniform flow: Kn should be 0 (non-zero values are round-off)")
    a.legend()
    finish(fig, out, "test_uniform_flow_has_zero_kn", "max |Kn|: " + ", ".join(worst))


def plot_mfp_scaling(out):
    n = 1e22
    T = np.geomspace(20, 1500, 200)
    ratio = HE.mean_free_path(n, T) / HE.mean_free_path(n, HE.T_ref)
    fig, a = plt.subplots(figsize=(7, 4))
    a.plot(T, (T / HE.T_ref) ** (HE.omega - 0.5), label=f"exact: (T/T_ref)^(ω−½), ω = {HE.omega}", **REF)
    a.plot(T, ratio, color=colour(0), label="sparta.gas.Gas.mean_free_path", **LINE)
    T4 = 4 * HE.T_ref
    a.plot([HE.T_ref, T4], [1, HE.mean_free_path(n, T4) / HE.mean_free_path(n, HE.T_ref)],
           ls="none", marker="o", ms=7, color=colour(0), markeredgecolor=SURFACE, mew=1.5)
    for t, lab in ((HE.T_ref, "T_ref"), (T4, "4 T_ref")):
        a.annotate(lab, (t, (t / HE.T_ref) ** (HE.omega - 0.5)), xytext=(6, -12),
                   textcoords="offset points", color=INK2, fontsize=9)
    a.set(xscale="log", xlabel="T [K]", ylabel="λ(T) / λ(T_ref) at fixed n",
          title="Mean free path temperature scaling (He VSS); the two tested points marked")
    a.legend()
    lam0 = HE.mean_free_path(n, HE.T_ref)
    hs = 1 / (np.sqrt(2) * np.pi * HE.d_ref**2 * n)
    finish(fig, out, "test_mean_free_path_scales_with_T_exponent",
           f"λ(T_ref) = {lam0:.4e} m vs hard-sphere 1/(√2 π d² n) = {hs:.4e} m at n = 1e22 m⁻³")


def plot_rejects_bad_x(mods, out):
    cases = [("x = [0, 0, 1]  (repeated point)", np.array([0.0, 0.0, 1.0])),
             ("x = [1, 0.5]  (decreasing)", np.array([1.0, 0.5])),
             ("x = [1]  (single point)", np.array([1.0]))]
    ms = with_fn(mods, "kn_gll")
    cells = []
    for _, x in cases:
        row = []
        for m in ms:
            try:
                m.kn_gll(x, np.ones_like(x), np.ones_like(x), HE)
                row.append("accepted (test fails)")
            except ValueError:
                row.append("ValueError (pass)")
            except Exception as e:  # noqa: BLE001
                row.append(f"{type(e).__name__} (test fails)")
        cells.append(row)
    fig, a = plt.subplots(figsize=(8, 1.4 + 0.45 * len(cases)))
    a.axis("off")
    t = a.table(cellText=cells, rowLabels=[c for c, _ in cases], colLabels=[name(m) for m in ms],
                loc="center", cellLoc="left", rowLoc="left")
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.scale(1, 1.5)
    for cell in t.get_celld().values():
        cell.set_edgecolor(GRID)
        cell.set_facecolor(SURFACE)
    a.set_title("Bad x input must raise ValueError (outcome per version)")
    finish(fig, out, "test_rejects_bad_x")


def plot_first_breakdown(mods, out):
    x = np.linspace(0, 1, 11)
    spike = np.array([0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0])
    ms = with_fn(mods, "first_breakdown")
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4))
    a.plot(x, x, color=INK2, marker="o", ms=5, lw=1.2, label="Kn = x")
    for thr, expect, ls in ((0.45, 0.5, ":"), (0.5, 0.6, "--")):
        a.axhline(thr, color=MUTED, lw=1, ls=ls, label=f"threshold {thr} → expect x = {expect}")
    for i, m in enumerate(ms):
        for thr in (0.45, 0.5):
            xb = m.first_breakdown(x, x, thr)
            if xb is not None:
                a.plot(xb, xb, ls="none", marker="o", ms=10, mfc="none", mew=2, color=colour(i),
                       label=name(m) if thr == 0.45 else None)
    above = ", ".join(f"{name(m)} → {m.first_breakdown(x, x, 2.0)}" for m in ms)
    a.set(ylim=(0, 1.05), xlabel="x", ylabel="Kn", title="Rising Kn: first point above threshold")
    a.legend(loc="upper left")
    b.plot(x, spike, color=INK2, marker="o", ms=5, lw=1.2, label="Kn with an early spike")
    b.axhline(0.05, color=MUTED, lw=1, ls=":")
    b.text(1.0, 0.05, "threshold 0.05", color=INK2, va="bottom", ha="right", fontsize=9)
    for i, m in enumerate(ms):
        xb = m.first_breakdown(x, spike, 0.05)
        if xb is not None:
            b.plot(xb, np.interp(xb, x, spike), ls="none", marker="o", ms=10, mfc="none", mew=2,
                   color=colour(i), label=f"{name(m)}: x = {xb:g}")
    b.set(ylim=(-0.005, 0.13), xlabel="x", ylabel="Kn", title="Spike: first crossing wins (expect 0.2)")
    b.legend(loc="upper right")
    finish(fig, out, "test_first_breakdown",
           f"circles = x returned by each version.  Threshold 2.0 (never reached, expect None): {above}")


def plot_kn_at(mods, out):
    xs, ks = np.array([0.0, 1.0]), np.array([0.0, 0.2])
    fig, a = plt.subplots(figsize=(6, 4))
    a.plot(xs, ks, color=INK2, marker="o", ms=6, lw=1.2, label="data points")
    a.plot(0.25, 0.05, ls="none", marker="+", ms=14, mew=2, color=INK, label="expected (0.25, 0.05)")
    for i, m in enumerate(with_fn(mods, "kn_at")):
        a.plot(0.25, m.kn_at(xs, ks, 0.25), ls="none", marker="o", ms=10, mfc="none", mew=2,
               color=colour(i), label=f"{name(m)}: {m.kn_at(xs, ks, 0.25):g}")
    a.set(xlabel="x", ylabel="Kn", title="kn_at: linear interpolation at x = 0.25")
    a.legend()
    finish(fig, out, "test_kn_at_interpolates")


def plot_axis_profile(mods, out):
    xy = np.array([[2.0, 0.0], [0.0, 0.0], [1.0, 0.5], [1.0, 0.0], [2.0, 0.0]])
    rho = np.array([3.0, 1.0, 9.0, 2.0, 3.0])
    T = np.array([30.0, 10.0, 90.0, 20.0, 30.0])
    field = SimpleNamespace(xy=xy, nodes={"rho": rho, "T": T})
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4))
    on = xy[:, 1] == 0
    a.plot(xy[~on, 0], xy[~on, 1], ls="none", marker="o", ms=9, color=MUTED, label="off-axis node")
    a.plot(xy[on, 0], xy[on, 1], ls="none", marker="o", ms=9, color=INK2, label="axis node")
    for pos in np.unique(xy, axis=0):
        ks = [k for k in range(len(xy)) if (xy[k] == pos).all()]
        lab = ", ".join(f"#{k}" for k in ks) + f"  ρ={rho[ks[0]]:g}" + ("  (duplicate)" if len(ks) > 1 else "")
        a.annotate(lab, pos, xytext=(8, 6), textcoords="offset points", color=INK2, fontsize=8)
    a.set(xlim=(-0.3, 2.8), ylim=(-0.2, 0.8), xlabel="x", ylabel="y (radius)",
          title="Fake field: 5 nodes, one duplicate, one off axis")
    a.legend(loc="upper left")
    b.plot([0, 1, 2], [1, 2, 3], label="expected ρ(x)", marker="+", ms=14, mew=2, color=INK, ls="--", lw=1)
    for i, m in enumerate(with_fn(mods, "axis_profile")):
        px, pr, _ = m.axis_profile(field)
        b.plot(px, pr, color=colour(i), marker="o", ms=10, mfc="none", mew=2, label=name(m), **LINE)
    b.set(xlabel="x", ylabel="ρ", title="axis_profile: axis only, sorted, duplicates removed")
    b.legend(loc="upper left")
    finish(fig, out, "test_axis_profile_selects_sorts_and_dedups")


def plot_su2(mods, vtus, case, out):
    from lib.su2_field import Su2Field

    D = get(case, "geometry", "throat_diam_mm") * 1e-3
    x_c = get(case, "sparta", "x_couple_D", None)
    gas = load_gas(get(case, "gas", "species_file"))
    series = [(m, v) for v in vtus for m in mods]
    fig, (a, b, c) = plt.subplots(3, 1, figsize=(8, 9), sharex=True, height_ratios=[2, 1, 1])
    a.axvspan(4, 6, color=GRID, alpha=0.6, lw=0)
    a.text(5, 0.02, "test\nwindow\n4–6 D", color=INK2, ha="center", va="bottom", fontsize=8,
           transform=a.get_xaxis_transform())
    notes = []
    for i, (m, v) in enumerate(series):
        field = Su2Field(v, xmin=-1e-9)
        run = Path(v).resolve().parent.parent.name
        lab = f"{run} / {name(m)}" if len(mods) > 1 else run
        x, kn = m.axis_knudsen(field, gas)
        thr = m.KN_BREAKDOWN
        keep = x > 0
        a.plot(x[keep] / D, kn[keep], color=colour(i), label=lab, **LINE)
        xb = m.first_breakdown(x, kn, thr)
        if xb is not None:
            a.plot(xb / D, m.kn_at(x, kn, xb), ls="none", marker="o", ms=8, color=colour(i),
                   markeredgecolor=SURFACE, mew=1.5)
        notes.append(f"{lab}: breakdown at {xb / D:.3g} D" if xb is not None else f"{lab}: no breakdown")
        on = np.abs(field.xy[:, 1]) < 1e-12
        o = np.argsort(field.xy[on, 0])
        xa = field.xy[on, 0][o]
        k2 = xa > 0
        b.plot(xa[k2] / D, field.nodes["rho"][on][o][k2], color=colour(i), **LINE)
        c.plot(xa[k2] / D, field.nodes["T"][on][o][k2], color=colour(i), **LINE)
    a.axhline(mods[0].KN_BREAKDOWN, color=MUTED, lw=1, ls=":")
    a.text(1.0, mods[0].KN_BREAKDOWN, f" threshold {mods[0].KN_BREAKDOWN}", color=INK2,
           va="center", ha="left", fontsize=9, transform=a.get_yaxis_transform())
    for ax in (a, b, c):
        if x_c is not None:
            ax.axvline(x_c, color=INK2, lw=1, ls="--")
    if x_c is not None:
        a.text(x_c, 0.98, f"coupling plane {x_c:g} D ", color=INK2, va="top", ha="right", fontsize=9,
               transform=a.get_xaxis_transform())
    a.set(xscale="log", yscale="log", ylabel="Kn_GLL",
          title="SU2 axis: breakdown parameter, density and temperature")
    a.legend(loc="upper left")
    b.set(yscale="log", ylabel="ρ [kg/m³]")
    c.set(yscale="log", ylabel="T [K]", xlabel="x / D  (distance downstream of the exit)")
    finish(fig, out, "test_su2_capillary_breaks_down_between_4_and_6_D",
           "dots = first breakdown point.  " + ";  ".join(notes))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--module", action="append", help="breakdown version(s) to plot "
                    "(repeatable; default handoff.breakdown_gll)")
    ap.add_argument("--su2", nargs="+", help="SU2 flow.vtu file(s) for the SU2 test "
                    "(default runs/ASDEX_capil_01/su2/flow.vtu)")
    ap.add_argument("--case", default=str(ROOT / "config.toml"),
                    help="case file for D, gas and coupling plane in the SU2 plot (default config.toml)")
    ap.add_argument("--out", default=str(ROOT / "tests" / "plots"), help="output directory")
    args = ap.parse_args(argv)

    mods = [importlib.import_module(m) for m in (args.module or ["handoff.breakdown_gll"])]
    if len(mods) > len(SERIES):
        raise SystemExit(f"at most {len(SERIES)} versions at once")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    plot_exponential(mods, out)
    plot_power_law(mods, out)
    plot_uniform(mods, out)
    plot_mfp_scaling(out)
    plot_rejects_bad_x(mods, out)
    plot_first_breakdown(mods, out)
    plot_kn_at(mods, out)
    plot_axis_profile(mods, out)

    vtus = args.su2 or [ROOT / "runs" / "ASDEX_capil_01" / "su2" / "flow.vtu"]
    missing = [v for v in vtus if not Path(v).is_file()]
    if missing:
        log(f"skipping the SU2 plot: not found {', '.join(map(str, missing))}")
    else:
        plot_su2(mods, vtus, load_case(args.case), out)


if __name__ == "__main__":
    main()
