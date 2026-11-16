#!/usr/bin/env python3
"""Run SU2 in a directory made by su2/setup.py.

Usage: python su2/run.py RUN_DIR [-n NPROCS] [--restart] [--iter N]
NPROCS defaults to [su2] nprocs. Set SU2_CFD to override the solver binary.
Output is tee'd to RUN_DIR/su2.log.

--restart continues from RUN_DIR/restart_flow.dat (RESTART_SOL= YES); without it the run
starts from the initial state (RESTART_SOL= NO), whatever su2.cfg said before. --iter sets
ITER, the number of iterations for this run (SU2 counts from 0 again after a restart).
Before a restart, the previous flow.vtu, history.csv, su2.log and restart_flow.dat are
copied to RUN_DIR/archive/<n>/ so the earlier state can be compared or recovered.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):  # allow `python su2/run.py` as well as `-m su2.run`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.common import die, get, load_case, log

ARCHIVED = ("flow.vtu", "history.csv", "su2.log", "restart_flow.dat", "su2.cfg")


def set_cfg(cfg: Path, **opts):
    """Set KEY= value lines in su2.cfg in place (each key must already be there)."""
    lines = cfg.read_text().splitlines()
    for key, val in opts.items():
        hits = [i for i, l in enumerate(lines) if re.match(rf"\s*{key}\s*=", l)]
        if not hits:
            die(f"{key} not found in {cfg}")
        for i in hits:
            lines[i] = f"{key}= {val}"
    cfg.write_text("\n".join(lines) + "\n")


def archive(run_dir: Path) -> Path:
    """Copy the current results to run_dir/archive/<n>/ (n = 1, 2, ...)."""
    root = run_dir / "archive"
    n = 1 + max((int(d.name) for d in root.glob("*") if d.name.isdigit()), default=0)
    dest = root / str(n)
    dest.mkdir(parents=True)
    for f in ARCHIVED:
        if (run_dir / f).is_file():
            shutil.copy2(run_dir / f, dest / f)
    return dest


def run(run_dir, nprocs=None, restart=False, iterations=None) -> int:
    """Run SU2_CFD under mpirun in run_dir. Returns the solver's exit code.

    restart: continue from restart_flow.dat (previous results archived first).
    iterations: ITER for this run; default keeps the value in su2.cfg."""
    run_dir = Path(run_dir).resolve()
    cfg = run_dir / "su2.cfg"
    if not cfg.is_file():
        die(f"{cfg} not found; run su2/setup.py first")
    if restart:
        if not (run_dir / "restart_flow.dat").is_file():
            die(f"no restart_flow.dat in {run_dir}: nothing to continue from")
        log(f"archived previous results to {archive(run_dir)}")
    opts = {"RESTART_SOL": "YES" if restart else "NO"}
    if iterations is not None:
        opts["ITER"] = int(iterations)
    set_cfg(cfg, **opts)
    log(("restarting from restart_flow.dat" if restart else "starting from the initial state")
        + (f", {iterations} iterations" if iterations is not None else ""))
    if nprocs is None:
        nprocs = get(load_case(run_dir / "case.toml"), "su2", "nprocs", 1)

    su2 = os.environ.get("SU2_CFD") or shutil.which("SU2_CFD")
    if not su2:
        die("SU2_CFD not on PATH (or set SU2_CFD=/path/to/SU2_CFD)")

    log(f"running {su2} on {nprocs} ranks in {run_dir} (log: su2.log)")
    cmd = ["mpirun", "-n", str(nprocs), su2, "su2.cfg"]
    with open(run_dir / "su2.log", "w") as logf, \
            subprocess.Popen(cmd, cwd=run_dir, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1) as proc:
        try:
            for line in proc.stdout:  # tee to screen and su2.log
                sys.stdout.write(line)
                logf.write(line)
        except KeyboardInterrupt:
            proc.terminate()
            proc.wait()
            log("interrupted")
    if proc.returncode != 0:
        log(f"SU2 exited with code {proc.returncode}, see {run_dir / 'su2.log'}")
    return proc.returncode


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir", help="directory containing su2.cfg and case.toml")
    ap.add_argument("-n", "--nprocs", type=int, help="MPI ranks (default: [su2] nprocs)")
    ap.add_argument("--restart", action="store_true", help="continue from restart_flow.dat")
    ap.add_argument("--iter", type=int, help="iterations for this run (sets ITER in su2.cfg)")
    args = ap.parse_args(argv)
    sys.exit(run(args.run_dir, args.nprocs, args.restart, args.iter))


if __name__ == "__main__":
    main()
