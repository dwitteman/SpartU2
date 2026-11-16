#!/usr/bin/env python3
"""Run SPARTA in a directory made by sparta/setup.py, with a memory guard.

Usage: python sparta/run.py RUN_DIR [-n NPROCS] [--restart --target STEP] [--input in.sparta]
                            [--max-mem-gb GB] [--force]
NPROCS defaults to [sparta] nprocs. Output is tee'd to RUN_DIR/sparta.log.

--restart continues from the latest restart.<step>.sparta with in.continue, up to timestep
--target (default: the latest restart step + [sparta] run_steps).
Set SPA_MPI (or SPA_SERIAL, used for -n 1) to override the binary.

Two guards keep a run from exhausting memory (and taking WSL down with it):
  before: the particle estimate in setup.log must fit in the memory limit (--force skips);
  during: the total resident memory of all SPARTA ranks is polled, and the run is
          killed if it exceeds the limit.
The limit is [sparta] max_mem_gb, or --max-mem-gb, default half the machine's RAM.
"""
import argparse
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

if __package__ in (None, ""):  # allow `python sparta/run.py` as well as `-m sparta.run`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.common import die, get, load_case, log

BYTES_PER_PARTICLE = 200  # same estimate as setup.log
SPARTA_SRC = Path.home() / "code/sparta-24Sep2025/src"


def mem_total():
    """Machine RAM in bytes (from /proc/meminfo)."""
    m = re.search(r"MemTotal:\s+(\d+) kB", Path("/proc/meminfo").read_text())
    return int(m.group(1)) * 1024


def group_rss(pgid):
    """Total resident memory [bytes] of all processes in process group pgid."""
    total = 0
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            stat = (d / "stat").read_text()
            fields = stat[stat.rindex(")") + 2:].split()  # after "pid (comm) "
            if int(fields[2]) != pgid:  # field 5 of /proc/pid/stat: pgrp
                continue
            m = re.search(r"VmRSS:\s+(\d+) kB", (d / "status").read_text())
            total += int(m.group(1)) * 1024 if m else 0
        except (OSError, ValueError):
            continue  # process exited while we looked
    return total


def setup_particles(run_dir):
    """Steady-state particle estimate from setup.log, or None."""
    f = Path(run_dir) / "setup.log"
    if not f.is_file():
        return None
    m = re.search(r"steady-state particles ~ ([\d.eE+-]+)", f.read_text())
    return float(m.group(1)) if m else None


def run_watched(cmd, cwd, logf, mem_limit, poll_s=1.0):
    """Run cmd in its own process group, tee its output to stdout and logf, and kill the
    whole group if its resident memory exceeds mem_limit bytes.
    Returns (exit code, peak memory in bytes, killed_for_memory)."""
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, text=True, bufsize=1, start_new_session=True)
    state = dict(peak=0, killed=False)

    def watch():
        while proc.poll() is None:
            rss = group_rss(proc.pid)
            state["peak"] = max(state["peak"], rss)
            if rss > mem_limit:
                state["killed"] = True
                os.killpg(proc.pid, signal.SIGKILL)
                return
            time.sleep(poll_s)

    t = threading.Thread(target=watch, daemon=True)
    t.start()
    try:
        for line in proc.stdout:
            sys.stdout.write(line)
            logf.write(line)
    except KeyboardInterrupt:
        os.killpg(proc.pid, signal.SIGTERM)
        log("interrupted")
    proc.wait()
    t.join(timeout=2 * poll_s)
    return proc.returncode, state["peak"], state["killed"]


def find_binary(nprocs):
    if nprocs == 1:
        cands = [os.environ.get("SPA_SERIAL"), shutil.which("spa_serial"), SPARTA_SRC / "spa_serial"]
    else:
        cands = [os.environ.get("SPA_MPI"), shutil.which("spa_mpi"), SPARTA_SRC / "spa_mpi"]
    for c in cands:
        if c and Path(c).is_file():
            return str(c)
    die(f"SPARTA binary not found ({'SPA_SERIAL/spa_serial' if nprocs == 1 else 'SPA_MPI/spa_mpi'})")


def latest_restart(run_dir):
    """Step of the newest restart.<step>.sparta in run_dir, or None."""
    steps = [int(m.group(1)) for f in Path(run_dir).glob("restart.*.sparta")
             if (m := re.fullmatch(r"restart\.(\d+)\.sparta", f.name))]
    return max(steps, default=None)


def run(run_dir, nprocs=None, input_file=None, max_mem_gb=None, force=False,
        restart=False, target=None) -> int:
    """Run SPARTA under mpirun in run_dir with the memory guards. Returns the exit code.

    restart: continue from the latest restart file with in.continue, up to timestep target."""
    run_dir = Path(run_dir).resolve()
    case = load_case(run_dir / "case.toml")
    extra = []
    if restart:
        step = latest_restart(run_dir)
        if step is None:
            die(f"no restart.<step>.sparta in {run_dir}: nothing to continue from")
        if target is None:
            target = step + int(get(case, "sparta", "run_steps", 1000))
        if target <= step:
            die(f"target {target} is not after the latest restart (step {step})")
        input_file = input_file or "in.continue"
        extra = ["-var", "target", str(int(target)), "-var", "seed", str(10007 + step)]
        log(f"continuing from restart.{step}.sparta up to step {target}")
    input_file = input_file or "in.sparta"
    if not (run_dir / input_file).is_file():
        die(f"{run_dir / input_file} not found; run sparta/setup.py first")
    if nprocs is None:
        nprocs = int(get(case, "sparta", "nprocs", 1))
    if max_mem_gb is None:
        max_mem_gb = get(case, "sparta", "max_mem_gb", mem_total() / 2 / 1e9)
    limit = float(max_mem_gb) * 1e9

    npart = setup_particles(run_dir)
    if npart is not None:
        need = npart * BYTES_PER_PARTICLE
        log(f"setup.log estimate: {npart:.3g} particles, ~{need / 1e9:.2g} GB; limit {limit / 1e9:.3g} GB")
        if need > limit and not force:
            die(f"estimated memory {need / 1e9:.2g} GB exceeds the limit {limit / 1e9:.3g} GB: raise "
                f"fnum or lower [sparta] max_particles and rerun sparta/setup.py (or --force)")
    else:
        log("no particle estimate in setup.log; relying on the memory watchdog only")

    spa = find_binary(nprocs)
    cmd = ([spa, "-in", input_file] if nprocs == 1
           else ["mpirun", "-n", str(nprocs), spa, "-in", input_file]) + extra
    log(f"running {' '.join(cmd)} in {run_dir} (log: sparta.log, memory limit {limit / 1e9:.3g} GB)")
    with open(run_dir / "sparta.log", "w") as logf:
        rc, peak, killed = run_watched(cmd, run_dir, logf, limit)
    log(f"peak memory {peak / 1e9:.3g} GB")
    if killed:
        log(f"KILLED: SPARTA exceeded the memory limit {limit / 1e9:.3g} GB")
        return 137
    if rc != 0:
        log(f"SPARTA exited with code {rc}, see {run_dir / 'sparta.log'}")
    return rc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir", help="directory containing in.sparta and case.toml")
    ap.add_argument("-n", "--nprocs", type=int, help="MPI ranks (default: [sparta] nprocs)")
    ap.add_argument("--input", help="input file (default: in.sparta, or in.continue with --restart; in.grid to check the grid)")
    ap.add_argument("--restart", action="store_true", help="continue from the latest restart.<step>.sparta")
    ap.add_argument("--target", type=int, help="with --restart: run up to this timestep")
    ap.add_argument("--max-mem-gb", type=float, help="memory limit (default: [sparta] max_mem_gb, else half the RAM)")
    ap.add_argument("--force", action="store_true", help="run even if setup.log's estimate exceeds the limit")
    args = ap.parse_args(argv)
    sys.exit(run(args.run_dir, args.nprocs, args.input, args.max_mem_gb, args.force, args.restart, args.target))


if __name__ == "__main__":
    main()
