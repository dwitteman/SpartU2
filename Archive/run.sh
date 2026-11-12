#!/usr/bin/env bash
# Run SU2 in a directory made by su2/setup.sh.
# Usage: su2/run.sh RUN_DIR [NPROCS]   (NPROCS defaults to [su2] nprocs)
# Set SU2_CFD to override the solver binary.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../lib/common.sh"

dir=${1:?usage: run.sh RUN_DIR [NPROCS]}
[[ -f $dir/su2.cfg ]] || die "$dir/su2.cfg not found; run su2/setup.sh first"
np=${2:-$(toml_get "$dir/case.toml" su2 nprocs 1)}
su2=${SU2_CFD:-$(command -v SU2_CFD || true)}
[[ -n $su2 ]] || die "SU2_CFD not on PATH (or set SU2_CFD=/path/to/SU2_CFD)"

cd "$dir"
log "running $su2 on $np ranks in $PWD (log: su2.log)"
mpirun -n "$np" "$su2" su2.cfg 2>&1 | tee su2.log
