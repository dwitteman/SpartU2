"""SPARTA species / VSS files and kinetic-theory estimates used for grid sizing."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lib.common import die

KB = 1.380649e-23
DATA = Path(__file__).resolve().parent / "data"


@dataclass
class Gas:
    species: str  # SPARTA species ID, e.g. "He"
    mass: float   # kg
    d_ref: float  # VSS reference diameter, m
    omega: float  # viscosity-temperature exponent
    T_ref: float  # K
    alpha: float  # VSS scattering parameter
    rotdof: int = 0  # rotational degrees of freedom (from the .species file)
    vibdof: int = 0  # vibrational degrees of freedom

    def mean_free_path(self, n, T):
        """VHS/VSS mean free path [m] for number density n [1/m3] and temperature T [K]."""
        return 1.0 / (np.sqrt(2) * np.pi * self.d_ref**2 * n * (self.T_ref / T) ** (self.omega - 0.5))

    def viscosity(self, T):
        """VSS viscosity [Pa s] (Bird 1994): reduces to VHS for alpha = 1."""
        a, w = self.alpha, self.omega
        mu_ref = (5 * (a + 1) * (a + 2) * np.sqrt(self.mass * KB * self.T_ref / np.pi)
                  / (4 * a * (5 - 2 * w) * (7 - 2 * w) * self.d_ref**2))
        return mu_ref * (np.asarray(T, float) / self.T_ref) ** w

    def mean_speed(self, T):
        """Mean thermal speed sqrt(8kT/(pi m)) [m/s]."""
        return np.sqrt(8 * KB * T / (np.pi * self.mass))


def _rows(path: Path):
    """Non-comment, non-blank lines of a SPARTA data file, split into tokens."""
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            yield line.split()


def load_gas(species_file: str) -> Gas:
    """Read <stem>.species and <stem>.vss from sparta/data/ (single-species files)."""
    stem = Path(species_file).stem
    sp, vss = DATA / f"{stem}.species", DATA / f"{stem}.vss"
    for f in (sp, vss):
        if not f.is_file():
            die(f"missing {f}")
    (srow,), (vrow,) = list(_rows(sp)), list(_rows(vss))  # one species per file for now
    if srow[0] != vrow[0]:
        die(f"species ID mismatch: {sp.name} has {srow[0]}, {vss.name} has {vrow[0]}")
    return Gas(species=srow[0], mass=float(srow[2]), d_ref=float(vrow[1]),
               omega=float(vrow[2]), T_ref=float(vrow[3]), alpha=float(vrow[4]),
               rotdof=int(srow[3]), vibdof=int(srow[5]))
