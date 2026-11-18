"""Breakdown report for the handoff: a text summary with pass/check flags, and a figure.

Written to RUNS/<name>/handoff/report.txt and report.png by handoff/handoff.py. It always
uses the full 2D Kn_GLL (breakdown_2d) as an independent check, whichever breakdown module
placed the plane. The figure is skipped (with a note) if matplotlib is not installed.
"""
from pathlib import Path

import numpy as np

from handoff import breakdown_2d as b2d
from handoff import breakdown_gll as gll

STATIONS_D = (0, 0.5, 1, 2, 3, 5, 7.5, 10, 15, 20, 30, 50, 80, 120, 200)
NOZZLE_FRACTIONS = (1, 0.75, 0.5, 0.25, 0.1)  # nozzle/capillary rows, as fractions of its length before the exit
MDOT_TOL = 0.05  # mass-flow conservation between the exit and the plane


def analyse(field, gas, x_c, thr):
    """Everything the report needs, from the SU2 field. Plain dict of arrays."""
    kn, dom = b2d.kn_field(field, gas)
    xs = b2d.stations(field)
    if not np.any(np.isclose(xs, x_c)):
        xs = np.sort(np.append(xs, x_c))
    r, kl, w, mdot = b2d.plane_lines(field, kn, xs)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = (w * (kl > thr)).sum(axis=1) / w.sum(axis=1)
    xa, kn_axis_rho = gll.axis_knudsen(field, gas)
    ic = int(np.argmin(np.abs(xs - x_c)))
    i0 = int(np.argmin(np.abs(xs)))  # nozzle exit
    # throat: the narrowest point of a nozzle (none for a straight capillary)
    x_throat = None
    if xs[0] < 0:
        inner = xs[xs < 0]
        rt = field.r_top(inner)
        r_min = np.nanmin(rt)
        if r_min < 0.98 * min(rt[0], rt[-1]):
            flat = inner[rt <= r_min * 1.001]  # a straight throat section: take its middle
            x_throat = float(0.5 * (flat[0] + flat[-1]))
    # dominant term at the plane: nodes of the SU2 column nearest the plane
    cols = np.unique(field.xy[:, 0])
    col = np.abs(field.xy[:, 0] - cols[np.argmin(np.abs(cols - x_c))]) < 1e-12
    over = col & (kn > thr)
    dom_counts = np.bincount(dom[over], minlength=3) if over.any() else np.zeros(3, int)
    return dict(kn=kn, dom=dom, xs=xs, r=r, kl=kl, w=w, mdot=mdot, frac=frac, xa=xa,
                kn_axis_rho=kn_axis_rho, ic=ic, i0=i0, x_throat=x_throat, dom_counts=dom_counts)


def checks(a, x_c, x_break, flows, plane, mass_fraction, thr):
    """List of (ok, message) sanity checks."""
    ic, out = a["ic"], []
    out.append((x_break is None or x_c < x_break,
                "plane is upstream of the breakdown point used to place it" if x_break is None or x_c < x_break
                else "plane is DOWNSTREAM of the breakdown point: SU2 is not reliable there"))
    f = a["frac"][ic]
    out.append((f <= mass_fraction,
                f"{f:.2%} of the mass flow through the plane has Kn_GLL(2D) > {thr} "
                f"(limit {mass_fraction:.0%})"))
    m0, mc = a["mdot"][a["i0"]], a["mdot"][ic]
    dev = abs(mc / m0 - 1) if m0 else np.inf
    out.append((dev <= MDOT_TOL,
                f"SU2 mass flow: {m0:.4g} kg/s at the exit, {mc:.4g} kg/s at the plane "
                f"({dev:.1%} apart, limit {MDOT_TOL:.0%})"))
    ratio = flows["mdot_inward"] / flows["mdot_net"] if flows["mdot_net"] else np.inf
    out.append((ratio < 1.5,
                f"SPARTA emits {ratio:.2f}x the SU2 net mass flow (one-sided flux includes the "
                f"thermal back-flow of slow gas; large values mean a dense background on the plane)"))
    out.append((plane["n"].min() >= 0 and plane["T"].min() > 1.0,
                f"plane state: n {plane['n'].min():.3g} to {plane['n'].max():.3g} /m3, "
                f"T {plane['T'].min():.3g} to {plane['T'].max():.3g} K, "
                f"u {plane['u'].min():.4g} to {plane['u'].max():.4g} m/s"))
    return out


def text(a, info, chk):
    """The report as a string. info: case, gas, vtu, module, thr, opts, mode, D, x_c, x_break."""
    D, x_c, thr = info["D"], info["x_c"], info["thr"]
    xs, ic = a["xs"], a["ic"]
    rc = a["r"][ic]
    i_r = a["kl"][ic] > thr
    r_first = rc[np.argmax(i_r)] if i_r.any() else None
    xb = info["x_break"]
    where_b = ""
    if xb is not None and xb < 0:
        dt = None if a["x_throat"] is None else (xb - a["x_throat"]) / D
        where_b = " (inside the nozzle" + (
            "" if dt is None else f", {abs(dt):.3g} D {'after' if dt >= 0 else 'before'} the throat middle") + ")"
    dom = a["dom_counts"]
    lines = [
        f"Breakdown report: {info['case']} ({info['gas']})",
        f"SU2 solution: {info['vtu']}",
        f"plane placed by {info['module']}, threshold Kn_GLL = {thr}"
        + (f", options {info['opts']}" if info["opts"] else ""),
        f"breakdown point from that module: "
        + (f"x = {xb * 1e3:.4g} mm = {xb / D:.3g} D{where_b}" if xb is not None
           else "none in the SU2 domain"),
        "x = 0 is the nozzle exit; x < 0 is inside the nozzle or capillary, where each plane"
        " ends at the wall",
        f"coupling plane ({info['mode']}): x = {x_c * 1e3:.4g} mm = {x_c / D:.3g} D",
        "",
        "Per plane (2D = Boyd's Kn_GLL with rho, T and |V| gradients; mass fraction = share of the",
        "plane's inward mass flow passing through gas with Kn_GLL(2D) > threshold):",
        f"  {'x/D':>7} {'':7} {'axis Kn(rho)':>13} {'axis Kn(2D)':>12} {'plane max':>10} {'mass frac':>10} {'mdot kg/s':>11}",
    ]
    rows = {}
    if xs[0] < 0:
        for f in NOZZLE_FRACTIONS:
            rows[int(np.argmin(np.abs(xs - f * xs[0])))] = "inlet" if f == 1 else ""
        if a["x_throat"] is not None:
            rows[int(np.argmin(np.abs(xs - a["x_throat"])))] = "throat"
    for xd in STATIONS_D:
        if xd * D > xs[-1]:
            break
        rows.setdefault(int(np.argmin(np.abs(xs - xd * D))), "")
    rows[a["i0"]] = "exit"
    for i in sorted(rows):
        lines.append(f"  {xs[i] / D:7.3g} {rows[i]:7} {gll.kn_at(a['xa'], a['kn_axis_rho'], xs[i]):13.3g} "
                     f"{a['kl'][i, 0]:12.3g} {a['kl'][i].max():10.3g} {a['frac'][i]:10.2%} {a['mdot'][i]:11.4g}")
    lines += [
        "",
        "At the coupling plane:",
        f"  Kn_GLL(2D) on the axis {a['kl'][ic, 0]:.3g}, largest on the plane {a['kl'][ic].max():.3g} "
        f"at r = {rc[np.argmax(a['kl'][ic])] * 1e3:.4g} mm",
        f"  Kn_GLL(2D) > {thr} from r = {r_first * 1e3:.4g} mm outward" if r_first is not None
        else f"  Kn_GLL(2D) <= {thr} across the whole plane",
        f"  term setting Kn where it is over the threshold (SU2 nodes): "
        + ", ".join(f"{q} {n}" for q, n in zip(b2d.QUANTITIES, dom)) if dom.sum() else
        "  no SU2 node on the plane is over the threshold",
        f"  SU2 net mass flow {info['mdot_net']:.4g} kg/s; SPARTA one-sided emission {info['mdot_inward']:.4g} kg/s",
        "",
        "Checks:",
    ]
    lines += [f"  [{'ok' if ok else '!!'}] {msg}" for ok, msg in chk]
    return "\n".join(lines) + "\n"


def figure(path, a, info):
    """report.png: Kn map, Kn along x, Kn and mass flow across the plane. False if no matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.tri import Triangulation
    except ImportError:
        return False
    INK, MUTED, GRID, SURF = "#1f1f1e", "#6b6b68", "#e4e4e0", "#fcfcfb"
    BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
    plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                         "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    D, x_c, thr, xb = info["D"], info["x_c"], info["thr"], info["x_break"]
    xy, ic, r = info["xy"], a["ic"], a["r"][a["ic"]]
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)

    # (a) Kn map, near field
    ext = min(xy[:, 0].max(), max(3 * x_c, 20 * D))
    # include the nozzle from its throat (or the last ext of a capillary) when there is one
    x_lo = 0.0 if xy[:, 0].min() >= 0 else max(xy[:, 0].min(), a["x_throat"] if a["x_throat"] is not None else -ext)
    rext = min(xy[:, 1].max(), 0.6 * ext)
    m = ax[0, 0]
    tri = Triangulation(xy[:, 0] * 1e3, xy[:, 1] * 1e3, info["simplices"])
    lk = np.log10(np.clip(a["kn"], 1e-5, 10))
    pc = m.tripcolor(tri, lk, shading="gouraud", cmap="Blues", vmin=-5, vmax=1, rasterized=True)
    m.tricontour(tri, lk, levels=[np.log10(thr)], colors=ORANGE, linewidths=1.2)
    m.axvline(x_c * 1e3, color=INK, ls=":", lw=1.2)  # coupling plane
    if x_lo < 0:
        m.axvline(0, color=MUTED, ls="--", lw=0.8)  # nozzle exit
    if xb is not None and x_lo <= xb <= ext:
        m.axvline(xb * 1e3, color=ORANGE, ls=":", lw=1.2)  # breakdown point
    m.set_xlim(x_lo * 1e3, ext * 1e3); m.set_ylim(0, rext * 1e3); m.grid(False)
    m.set_xlabel("x [mm]"); m.set_ylabel("r [mm]")
    m.set_title(f"Kn_GLL (2D) around the exit (dashed); orange line: Kn = {thr}", loc="left")
    fig.colorbar(pc, ax=m, shrink=0.9, pad=0.01, label="log10 Kn_GLL")

    # (b) along x
    p = ax[0, 1]
    p.plot(a["xa"] / D, a["kn_axis_rho"], color=BLUE, lw=1.6, label="axis, density only (breakdown_gll)")
    p.plot(a["xs"] / D, a["kl"][:, 0], color=GREEN, lw=1.6, label="axis, 2D (rho, T, |V|)")
    p.plot(a["xs"] / D, np.nanmax(a["kl"], axis=1), color=ORANGE, lw=1.6, label="largest on the plane, 2D")
    if info["profile"] is not None:
        p.plot(info["profile"][0] / D, info["profile"][1], color=INK, lw=1.2, ls="--",
               label=f"used to place the plane ({info['module'].split('.')[-1]})")
    p.axhline(thr, color=MUTED, ls="--", lw=1)
    p.axvline(x_c / D, color=INK, ls=":", lw=1.2, label="coupling plane")
    if xb is not None:
        p.axvline(xb / D, color=ORANGE, ls=":", lw=1.2, label="breakdown point")
    if a["xs"][0] < 0:
        p.axvline(0, color=MUTED, ls="--", lw=0.8, label="nozzle exit")
        if a["x_throat"] is not None:
            p.axvline(a["x_throat"] / D, color=MUTED, ls="-.", lw=0.8, label="throat")
    # linear through the nozzle (and as far into the plume), logarithmic beyond
    p.set_yscale("log"); p.set_xscale("symlog", linthresh=max(1.0, -a["xs"][0] / D))
    p.set_xlim(a["xs"][0] / D, a["xs"][-1] / D)
    p.set_xlabel("x / D"); p.set_ylabel("Kn_GLL")
    p.set_title("Kn_GLL along the nozzle and jet", loc="left")
    p.legend(frameon=False, fontsize=7.5, loc="lower right")

    # (c) across the plane: Kn
    q = ax[1, 0]
    q.plot(r * 1e3, a["kl"][ic], color=ORANGE, lw=1.6)
    q.axhline(thr, color=MUTED, ls="--", lw=1)
    q.set_yscale("log"); q.set_xlabel("r [mm]"); q.set_ylabel("Kn_GLL (2D)")
    q.set_title(f"Across the coupling plane, x = {x_c / D:.3g} D", loc="left")

    # (d) across the plane: cumulative mass flow
    c = ax[1, 1]
    w = a["w"][ic]
    cum = np.cumsum(w) / w.sum() if w.sum() > 0 else np.zeros_like(w)
    c.plot(r * 1e3, cum, color=BLUE, lw=1.6, label="inward mass flow inside r")
    over = a["kl"][ic] > thr
    if over.any():
        c.fill_between(r * 1e3, 0, 1, where=over, color=ORANGE, alpha=0.15, lw=0, label=f"Kn_GLL > {thr}")
    c.set_ylim(0, 1.02); c.set_xlabel("r [mm]"); c.set_ylabel("fraction of the plane's mass flow")
    c.set_xscale("log"); c.set_xlim(max(r[1], 1e-3 * r[-1]) * 1e3, r[-1] * 1e3)
    c.set_title(f"Mass flow across the plane: {a['frac'][ic]:.1%} over the threshold", loc="left")
    c.legend(frameon=False, fontsize=7.5, loc="lower right")

    fig.suptitle(f"Breakdown report: {info['case']} ({info['gas']})", x=0.01, ha="left", fontsize=11)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def write(out_dir, field, gas, info, plane, flows, mass_fraction):
    """Write report.txt (and report.png) to out_dir. Returns the report text."""
    a = analyse(field, gas, info["x_c"], info["thr"])
    chk = checks(a, info["x_c"], info["x_break"], flows, plane, mass_fraction, info["thr"])
    info = dict(info, mdot_net=flows["mdot_net"], mdot_inward=flows["mdot_inward"],
                xy=field.xy, simplices=field.simplices)
    txt = text(a, info, chk)
    out_dir = Path(out_dir)
    if not figure(out_dir / "report.png", a, info):
        txt += "(report.png skipped: matplotlib is not installed)\n"
    (out_dir / "report.txt").write_text(txt)
    return txt
