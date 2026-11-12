"""SU2 -> SPARTA handoff: the inflow at the coupling plane (the DSMC box face xlo).

The SU2 solution is sampled along x = x_c and written as a `fix emit/face/file` table:
number density, velocity and temperature at mesh points in r. SPARTA interpolates it
linearly to each grid-cell face on xlo and emits the one-sided Maxwellian flux for
that state (Bird 1994, eq. 4.22), so particles enter wherever the local state sends
them in, including the thermal back-diffusion from a stagnant or slow region.

The core functions take plain arrays, so they can be tested without an SU2 file.
"""
from math import erf
from pathlib import Path

import numpy as np

from lib.common import die

KB = 1.380649e-23
SECTION = "XLO"  # boundary-ID of the section in the inflow file


def plane_radii(field, x_c, r_max, n_min=200):
    """Mesh radii for the inflow table: the SU2 node radii near x = x_c, plus a uniform
    set of n_min points, clipped to [0, r_max]. Keeps SU2's resolution of the plume edge."""
    xs = np.unique(field.xy[:, 0])
    x_near = xs[np.argmin(np.abs(xs - x_c))]
    r = field.xy[np.abs(field.xy[:, 0] - x_near) < 1e-12, 1]
    r = np.concatenate([r, np.linspace(0.0, r_max, n_min)])
    r = np.unique(np.clip(r, 0.0, r_max))
    return r[np.concatenate([[True], np.diff(r) > 1e-12 * max(r_max, 1e-30)])]


def sample_plane(field, gas, x_c, r):
    """SU2 state on x = x_c at radii r: dict of n [1/m3], u, v [m/s], T [K], rho [kg/m3]."""
    s = field.sample(np.full_like(r, x_c), r)
    rho = np.maximum(s["rho"], 0.0)
    T = np.maximum(s["T"], 1e-3)
    return dict(r=r, n=rho / gas.mass, u=s["u"], v=np.where(r > 0, s["v"], 0.0), T=T, rho=rho)


def inward_number_flux(n, u, T, mass):
    """One-sided Maxwellian number flux [1/m2/s] through a plane with normal +x
    (Bird 1994, eq. 4.22): what fix emit/face injects for drift u along the normal."""
    n, u, T = (np.asarray(a, float) for a in (n, u, T))
    cmp_ = np.sqrt(2 * KB * T / mass)
    s = u / cmp_
    erf_s = np.vectorize(erf)(s)
    return n * cmp_ / (2 * np.sqrt(np.pi)) * (np.exp(-s**2) + np.sqrt(np.pi) * s * (1 + erf_s))


def annulus_integral(r, q):
    """Integral of q(r) * 2 pi r dr over the plane (trapezoidal in r)."""
    return float(np.trapezoid(q * 2 * np.pi * r, r))


def plane_summary(plane, mass):
    """Mass flows [kg/s] through the plane: SU2's net rho*u, and the inward flux SPARTA
    will emit (larger than net where gas also leaves through xlo, e.g. stagnant regions)."""
    r = plane["r"]
    net = annulus_integral(r, plane["rho"] * plane["u"])
    inward = annulus_integral(r, mass * inward_number_flux(plane["n"], plane["u"], plane["T"], mass))
    return dict(mdot_net=net, mdot_inward=inward)


def write_inflow(path, plane, rot=True, vib=True, header=""):
    """Write the fix emit/face/file table. trot/tvib are set to T (local equilibrium,
    as SU2 assumes) for species that have those modes."""
    r = plane["r"]
    if not (np.all(np.diff(r) > 0) and r[0] >= 0):
        die("inflow radii must be increasing and >= 0")
    cols = [("nrho", plane["n"]), ("vx", plane["u"]), ("vy", plane["v"]),
            ("vz", np.zeros_like(r)), ("temp", plane["T"])]
    if rot:
        cols.append(("trot", plane["T"]))
    if vib:
        cols.append(("tvib", plane["T"]))
    lines = [f"# {line}" for line in header.splitlines()] + [
        "",
        f"{SECTION}  SU2 state on the coupling plane",
        f"NI {len(r)}",
        f"NV {len(cols)}",
        "VALUES " + " ".join(name for name, _ in cols),
        "IMESH " + " ".join(f"{v:.10g}" for v in r),
        "",
    ]
    for i in range(len(r)):
        lines.append(f"{i + 1} " + " ".join(f"{vals[i]:.8g}" for _, vals in cols))
    Path(path).write_text("\n".join(lines) + "\n")
