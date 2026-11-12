"""Analytical mass flow through a capillary, for checking SU2.

Model: plenum (p0, T0) -> isentropic entrance -> constant-area tube with laminar
wall friction -> choked exit. The exit is choked whenever the back pressure is below
the exit critical pressure, which holds for every case here (p_back ~ 0.1 Pa).

Two limits for the tube are provided, because SU2's tube wall is isothermal while
textbook Fanno flow is adiabatic:

  fanno_laminar      adiabatic wall (Fanno flow), chokes at M = 1. Viscosity follows
                     the local static temperature, which falls along the tube.
  isothermal_laminar isothermal flow at the wall temperature, chokes at M = 1/sqrt(gamma).

Both use the fully developed laminar friction factor f_D = 64/Re (Darcy), with
Re = 4 mdot / (pi D mu). The developing-flow entrance region adds pressure drop on
top of that; k_entrance adds it as an incremental loss of K dynamic heads at the
inlet (Hagenbach factor; K ~ 1.25 for a circular tube, Shah & London 1978), i.e. an
extra friction length K / f_D = K Re / 64 diameters. k_entrance = 0 (default) gives
the textbook fully developed result.

Friction relations (Darcy f_D = 4 x Fanning f):
  Fanno:      f_D dx/D = 2 (1 - M^2) / (gamma M^3 (1 + (gamma-1)/2 M^2)) dM
  isothermal: f_D L*/D = (1 - gamma M^2) / (gamma M^2) + ln(gamma M^2)
"""
from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq


@dataclass
class IdealGas:
    gamma: float
    R: float       # J/(kg K)
    mu_ref: float  # Pa s, Sutherland reference viscosity
    T_ref: float   # K
    S: float       # K, Sutherland constant

    def mu(self, T):
        """Sutherland viscosity [Pa s]. S = 0 gives mu ~ T^1.5; use mu_const() for a constant."""
        return self.mu_ref * (T / self.T_ref) ** 1.5 * (self.T_ref + self.S) / (T + self.S)

    @classmethod
    def from_su2(cls, species: str) -> "IdealGas":
        """The continuum gas model su2/setup.py writes into su2.cfg."""
        from su2.setup import GASES

        g = GASES[species]
        return cls(g["gamma"], g["R"], g["mu_ref"], g["T_ref"], g["S"])


def area(D):
    return np.pi * D**2 / 4


def isentropic_massflux(M, p0, T0, gas):
    """Mass flux rho*u [kg/(m2 s)] at Mach M after isentropic expansion from p0, T0."""
    g = gas.gamma
    return p0 * M * np.sqrt(g / (gas.R * T0)) * (1 + (g - 1) / 2 * M**2) ** (-(g + 1) / (2 * (g - 1)))


def choked_orifice(D, p0, T0, gas):
    """Ideal choked mass flow with no friction: the upper bound for any tube length."""
    return area(D) * isentropic_massflux(1.0, p0, T0, gas)


def fanno_dfdm(M, gamma):
    """f_D d(x/D) / dM for Fanno flow (positive for M < 1)."""
    return 2 * (1 - M**2) / (gamma * M**3 * (1 + (gamma - 1) / 2 * M**2))


def fanno_length(M, gamma):
    """Closed-form f_D L*/D from Mach M to choking at M = 1, for constant f_D."""
    g = gamma
    return (1 - M**2) / (g * M**2) + (g + 1) / (2 * g) * np.log((g + 1) * M**2 / (2 + (g - 1) * M**2))


def isothermal_length(M, gamma):
    """Closed-form f_D L*/D from Mach M to choking at M = 1/sqrt(gamma), isothermal flow."""
    return (1 - gamma * M**2) / (gamma * M**2) + np.log(gamma * M**2)


def _solve(length_D, LD, Mmax):
    """Inlet Mach M1 in (0, Mmax) such that length_D(M1) = L/D (length_D falls with M1)."""
    lo, hi = 1e-6, Mmax * (1 - 1e-9)
    if length_D(lo) < LD:
        raise ValueError("tube too long for the solver's Mach range")
    return brentq(lambda M: length_D(M) - LD, lo, hi, xtol=1e-14, rtol=1e-12)


def fanno_laminar(D, L, p0, T0, gas, k_entrance=0.0) -> dict:
    """Choked adiabatic laminar flow. Returns mdot [kg/s], inlet Mach M1, inlet/exit Re, exit T."""
    g, A = gas.gamma, area(D)

    def T_of(M):
        return T0 / (1 + (g - 1) / 2 * M**2)

    def length_D(M1):
        """Physical tube length [diameters] that chokes with inlet Mach M1: the friction
        length from M1 to M = 1, less the equivalent length used up by the entrance loss."""
        mdot = A * isentropic_massflux(M1, p0, T0, gas)
        # f_D = 64/Re = 16 pi D mu / mdot, so d(x/D) = fanno_dfdm dM / f_D.
        # Integrate over s = ln M (dM = M ds): smooth even for very small M1.
        val, _ = quad(lambda s: fanno_dfdm(np.exp(s), g) * np.exp(s) * mdot
                      / (16 * np.pi * D * gas.mu(T_of(np.exp(s)))), np.log(M1), 0.0, limit=200)
        re_in = 4 * mdot / (np.pi * D * gas.mu(T_of(M1)))
        return val - k_entrance * re_in / 64

    M1 = _solve(length_D, L / D, 1.0)
    mdot = A * isentropic_massflux(M1, p0, T0, gas)
    Re = lambda T: 4 * mdot / (np.pi * D * gas.mu(T))  # noqa: E731
    return dict(mdot=mdot, M1=M1, Re_in=Re(T_of(M1)), Re_exit=Re(T_of(1.0)), T_exit=T_of(1.0))


def isothermal_laminar(D, L, p0, T0, gas, T_wall=None, k_entrance=0.0) -> dict:
    """Choked isothermal laminar flow at T_wall (default T0), isentropic entrance from p0, T0."""
    g, A = gas.gamma, area(D)
    T = T0 if T_wall is None else T_wall

    def mdot_of(M1):
        p1 = p0 * (1 + (g - 1) / 2 * M1**2) ** (-g / (g - 1))
        return A * p1 * M1 * np.sqrt(g / (gas.R * T))

    def length_D(M1):
        f = 16 * np.pi * D * gas.mu(T) / mdot_of(M1)
        return (isothermal_length(M1, g) - k_entrance) / f

    M1 = _solve(length_D, L / D, 1 / np.sqrt(g))
    mdot = mdot_of(M1)
    return dict(mdot=mdot, M1=M1, Re=4 * mdot / (np.pi * D * gas.mu(T)), T=T)


def tube_massflows(field, R, xs, rtol=1e-6):
    """Mass flow [kg/s] through tube cross-sections at the node columns nearest each x in xs.

    Integrates rho*u*2*pi*r over the SU2 nodes of one radial grid line (r <= R),
    using the node values directly (no interpolation). field is a lib.su2_field.Su2Field
    that includes the tube (xmin below the tube inlet). Returns (x_used, mdot) arrays.
    """
    xy, rho, u = field.xy, field.nodes["rho"], field.nodes["u"]
    axis = np.unique(xy[np.abs(xy[:, 1]) < rtol * R, 0])
    cols = np.unique([axis[np.argmin(np.abs(axis - x))] for x in np.atleast_1d(xs)])
    out = []
    for xc in cols:
        on = (np.abs(xy[:, 0] - xc) <= rtol * R) & (xy[:, 1] <= R * (1 + rtol))
        o = np.argsort(xy[on, 1])
        r = xy[on, 1][o]
        out.append(np.trapezoid(rho[on][o] * u[on][o] * 2 * np.pi * r, r))
    return cols, np.array(out)
