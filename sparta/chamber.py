"""Chamber boundary model for the DSMC run: what happens at the far end (xhi) and outer wall (yhi).

[sparta] chamber =
  "outflow"    (default) both faces remove every particle that reaches them.
  "pump"       both faces are diffuse walls at pump_temp_K that remove (pump) a fraction
               pump_fraction of the molecules hitting them, as in Bi et al.; the chamber pressure
               settles where the pumped rate equals the jet's inflow. pump_pres_pa instead sets
               the fraction from that balance: f = N_in / (n cbar/4 A).
  "reservoir"  both faces stay open and inject still background gas (background_gas, mole
               fractions) at background_pres_pa and background_temp_K: an open boundary to an
               infinite reservoir, which holds the chamber at that pressure and composition.

pump and reservoir start the chamber filled at the target state (fill = false to start empty).
A background species that is the beam gas gets its own SPARTA ID (e.g. N2bg), so beam and
background molecules are counted separately. With more than one species, the beam inflow uses
its own mixture "beam" and the flow dumps get per-species columns.
"""
import math

from lib.common import die, get
from sparta.gas import DATA, KB, load_gas

MODES = ("outflow", "pump", "reservoir")


def _species_rows(stem, new_id=None):
    """(species line, vss line) for one species from sparta/data, optionally renamed."""
    out = []
    for ext in ("species", "vss"):
        rows = [l.split("#", 1)[0].split() for l in (DATA / f"{stem}.{ext}").read_text().splitlines()]
        rows = [r for r in rows if r]
        if len(rows) != 1:
            die(f"sparta/data/{stem}.{ext}: expected one species")
        r = rows[0]
        if new_id:
            r = [new_id] + r[1:]
        out.append(" ".join(r))
    return out


def pumped_area(box) -> float:
    """Area [m2] of the pumped faces in the axisymmetric 3D chamber: the end disc (xhi) and
    the outer cylinder (yhi)."""
    xlo, xhi, _, R = box
    return math.pi * R**2 + 2 * math.pi * R * (xhi - xlo)


def chamber_model(case, gas, mdot, box) -> dict:
    """The [sparta] chamber settings, checked, with everything the input files need.
    gas: the beam Gas; mdot: SU2 net mass flow into the DSMC domain [kg/s]; box: (xlo, xhi, ylo, yhi)."""
    mode = get(case, "sparta", "chamber", "outflow")
    if mode not in MODES:
        die(f"sparta.chamber = {mode!r}: use one of {', '.join(MODES)}")
    T_wall = float(get(case, "su2", "wall_temp_K"))
    ch = dict(mode=mode, beam=gas.species, bg=[], n_bg=0.0, boundary="o ao p", fill=None)
    if mode == "outflow":
        return ch

    fill = bool(get(case, "sparta", "fill", True))
    if mode == "pump":
        T = float(get(case, "sparta", "pump_temp_K", T_wall))
        cbar = gas.mean_speed(T)
        A = pumped_area(box)
        N_in = mdot / gas.mass
        has_f, has_p = "pump_fraction" in case["sparta"], "pump_pres_pa" in case["sparta"]
        if has_f == has_p:
            die("sparta.chamber = \"pump\" needs one of pump_fraction or pump_pres_pa")
        if has_p:
            P = float(get(case, "sparta", "pump_pres_pa"))
            if P <= 0:
                die("sparta.pump_pres_pa must be positive")
            f = N_in / (P / (KB * T) * cbar / 4 * A)
            if not 0 < f <= 1:
                die(f"sparta.pump_pres_pa = {P:g} Pa needs a pumped fraction of {f:.3g}: the jet "
                    f"({mdot:.3g} kg/s) cannot hold that pressure with these walls")
        else:
            f = float(get(case, "sparta", "pump_fraction"))
            if not 0 < f <= 1:
                die("sparta.pump_fraction must be in (0, 1]")
            P = 4 * N_in * KB * T / (f * cbar * A)
        ch.update(boundary="os as p", T=T, f=f, P=P, n_bg=P / (KB * T), A=A, N_in=N_in)
        if fill:  # beam gas at the expected chamber pressure
            ch["fill"] = dict(mix="fill", species=[(gas.species, 1.0)], n=P / (KB * T), T=T)
        return ch

    # reservoir
    P = float(get(case, "sparta", "background_pres_pa"))
    T = float(get(case, "sparta", "background_temp_K", T_wall))
    if P <= 0 or T <= 0:
        die("sparta.background_pres_pa and background_temp_K must be positive")
    comp = get(case, "sparta", "background_gas", {gas.species.lower(): 1.0})
    total = sum(comp.values())
    if not comp or total <= 0 or any(v < 0 for v in comp.values()):
        die("sparta.background_gas must give positive mole fractions, e.g. { ar = 1.0 }")
    bg = []
    for stem, frac in comp.items():
        g = load_gas(f"{stem.lower()}.species")
        sid = g.species + "bg" if g.species == gas.species else g.species
        bg.append(dict(id=sid, stem=stem.lower(), frac=frac / total, gas=g))
    ch.update(bg=bg, P=P, T=T, n_bg=P / (KB * T))
    if fill:
        ch["fill"] = dict(mix="bg", n=P / (KB * T), T=T)
    return ch


def multi(ch) -> bool:
    return bool(ch and ch["bg"])


def species_ids(ch, beam_id):
    return [beam_id] + [b["id"] for b in ch["bg"]] if multi(ch) else [beam_id]


def species_files(ch, beam_stem):
    """{file name: text} of the combined species and VSS files when there is a background gas."""
    rows = [_species_rows(beam_stem)] + [_species_rows(b["stem"], b["id"]) for b in ch["bg"]]
    head = "# Generated by sparta/setup.py: the beam gas, then the background gas (sparta/data)\n"
    return {"gas.species": head + "\n".join(r[0] for r in rows) + "\n",
            "gas.vss": head + "\n".join(r[1] for r in rows) + "\n"}


def gas_block(ch, species, stem) -> str:
    """species / mixture / collide commands. One species: as before (mixture gas). With a
    background gas: mixture gas = all species, beam = the inflow, bg = the background."""
    if not multi(ch):
        lines = [f"species         {stem}.species {species}", f"mixture         gas {species}"]
        if ch and ch.get("fill"):
            fl = ch["fill"]
            lines.append(f"mixture         {fl['mix']} {species} nrho {fl['n']:.6g} temp {fl['T']:g} vstream 0 0 0")
        lines.append(f"collide         vss gas {stem}.vss")
        return "\n".join(lines) + "\n"
    ids = species_ids(ch, species)
    lines = [f"species         gas.species {' '.join(ids)}",
             f"mixture         gas {' '.join(ids)}",
             f"mixture         beam {species}",
             f"mixture         bg {' '.join(b['id'] for b in ch['bg'])} nrho {ch['n_bg']:.6g} "
             f"temp {ch['T']:g} vstream 0 0 0"]
    lines += [f"mixture         bg {b['id']} frac {b['frac']:.6g}" for b in ch["bg"]]
    lines.append("collide         vss gas gas.vss")
    return "\n".join(lines) + "\n"


def inflow_mixture(ch) -> str:
    return "beam" if multi(ch) else "gas"


def boundary_cmds(ch) -> str:
    """Pumped walls or the background reservoir, after the gas is defined (also after read_restart)."""
    if not ch or ch["mode"] == "outflow":
        return ""
    if ch["mode"] == "pump":
        return (f"# chamber end (xhi) and outer wall (yhi): diffuse walls at {ch['T']:g} K that pump\n"
                f"# (remove) {ch['f']:.4g} of the molecules hitting them; expected chamber pressure "
                f"{ch['P']:.4g} Pa\n"
                f"surf_collide    pump diffuse {ch['T']:g} 1.0\n"
                f"surf_react      pump global {ch['f']:.6g} 0.0\n"
                f"bound_modify    xhi yhi collide pump react pump\n")
    comp = ", ".join(f"{b['id']} {b['frac']:.3g}" for b in ch["bg"])
    return (f"# chamber end (xhi) and outer wall (yhi): open to a reservoir of still background gas\n"
            f"# ({comp}) at {ch['P']:.4g} Pa and {ch['T']:g} K\n"
            f"fix             bg emit/face bg xhi yhi\n")


def fill_cmds(ch) -> str:
    """Start the chamber at the target state (in.sparta only)."""
    if not ch or not ch.get("fill"):
        return ""
    fl = ch["fill"]
    return (f"# start the chamber filled at the target state, n = {fl['n']:.4g} /m3, {fl['T']:g} K\n"
            f"create_particles {fl['mix']} n 0\n")


def species_columns(ch, species):
    """Extra averaged columns with several species: per species n, u, v, then T; and labels."""
    if not multi(ch):
        return "", "", []
    ids = species_ids(ch, species)
    ns = len(ids)
    computes = ("compute         gs grid all species nrho u v\n"
                "compute         ts thermal/grid all species temp\n")
    labels = [f"{q}_{sid}" for sid in ids for q in ("nrho", "u", "v")] + [f"T_{sid}" for sid in ids]
    return computes, " c_gs[*] c_ts[*]", labels


def report(ch) -> str:
    if ch["mode"] == "outflow":
        return "chamber: outflow (xhi, yhi remove every particle)"
    if ch["mode"] == "pump":
        return (f"chamber: pumped walls (xhi, yhi; {ch['A']:.4g} m2) at {ch['T']:g} K removing "
                f"{ch['f']:.4g} of hits; jet {ch['N_in']:.4g} molecules/s -> chamber ~{ch['P']:.4g} Pa"
                + ("; starts filled" if ch.get("fill") else "; starts empty"))
    comp = ", ".join(f"{b['id']} {b['frac']:.3g}" for b in ch["bg"])
    return (f"chamber: reservoir of {comp} at {ch['P']:.4g} Pa, {ch['T']:g} K on xhi, yhi"
            + ("; starts filled" if ch.get("fill") else "; starts empty"))
