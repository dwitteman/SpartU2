#!/usr/bin/env python3
"""Run SU2 in a directory made by su2/setup.py.

Usage: python su2/run.py RUN_DIR [-n NPROCS]   (NPROCS defaults to [su2] nprocs)
Set SU2_CFD to override the solver binary. Output is tee'd to RUN_DIR/su2.log.
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):  # allow `python su2/run.py` as well as `-m su2.run`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.common import die, get, load_case, log


def run(run_dir, nprocs=None) -> int:
    """Run SU2_CFD under mpirun in run_dir. Returns the solver's exit code."""
    run_dir = Path(run_dir).resolve()
    if not (run_dir / "su2.cfg").is_file():
        die(f"{run_dir / 'su2.cfg'} not found; run su2/setup.py first")
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
    args = ap.parse_args(argv)
    sys.exit(run(args.run_dir, args.nprocs))


if __name__ == "__main__":
    main()
