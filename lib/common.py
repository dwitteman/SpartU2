"""Shared helpers for the SpartU2 setup/run scripts (SU2 and DSMC sides)."""
import time
import tomllib
from pathlib import Path

_MISSING = object()


def die(msg: str):
    """Print an error to stderr and exit with status 1."""
    raise SystemExit(f"ERROR: {msg}")


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_case(path) -> dict:
    """Read a case .toml file."""
    path = Path(path)
    if not path.is_file():
        die(f"case file {path} not found")
    with open(path, "rb") as f:
        case = tomllib.load(f)
    case["_path"] = str(path.resolve())  # so files named in the case resolve next to it
    return case


def case_file(case: dict, name: str) -> Path:
    """A file named in the case: absolute, or relative to the case file's directory."""
    p = Path(name)
    if p.is_absolute() or "_path" not in case:
        return p
    return Path(case["_path"]).parent / p


def get(case: dict, section: str, key: str, default=_MISSING):
    """case[section][key], dying with a readable message if it is missing."""
    try:
        return case[section][key]
    except KeyError:
        if default is _MISSING:
            die(f"missing [{section}] {key} in case file")
        return default


def fmt(x) -> str:
    """Compact number formatting for generated input files (like printf %.10g)."""
    return f"{x:.10g}" if isinstance(x, float) else str(x)


def growth_rate(length: float, n: int, first: float) -> float:
    """Geometric ratio r such that n cells starting at size `first` span `length`.

    Returns 1.0 (uniform) if `first` is already >= the uniform spacing.
    """
    if first * n >= length:
        return 1.0
    lo, hi = 1 + 1e-12, 2.0
    for _ in range(200):
        r = 0.5 * (lo + hi)
        if first * (r**n - 1) / (r - 1) > length:
            hi = r
        else:
            lo = r
    return r
