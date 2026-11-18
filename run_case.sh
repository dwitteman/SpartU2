#!/bin/bash
# SpartU2 pipeline: SU2 continuum solution -> handoff -> SPARTA DSMC.
#
# PBS:    qsub -v CASE=examples/cd_nozzle_n2.toml run_case.sh
#         qsub -v CASE=examples/cd_nozzle_n2.toml,STAGES="handoff sparta_setup sparta" run_case.sh
# Local:  CASE=examples/cd_nozzle_n2.toml STAGES="handoff sparta_setup" NP=18 bash run_case.sh
#
# Stages, run in the order listed in STAGES (default: su2_setup su2 handoff sparta_setup sparta):
#   su2_setup       mesh + su2.cfg                                  -> runs/<name>/su2/
#   su2             SU2_CFD from the initial state                  -> flow.vtu, restart_flow.dat
#   su2_restart     SU2_CFD from restart_flow.dat (old results copied to su2/archive/<n>/ first)
#   handoff         coupling plane + inflow.sparta from flow.vtu    -> runs/<name>/handoff/
#   sparta_setup    grid + in.sparta + in.continue                  -> runs/<name>/sparta/
#   sparta          SPARTA from an empty box (hold, grid adaptation, averaged run)
#   sparta_restart  continue SPARTA from the newest restart.<step>.sparta up to step TARGET
#
# Settings (environment variables, or qsub -v NAME=value,...):
#   CASE      case file (required)
#   STAGES    stages to run, space separated
#   NP        MPI ranks (default: lines in $PBS_NODEFILE, else nproc)
#   MPIRUN    launcher (default: "mpirun -np $NP")
#   SU2_ITER  ITER for su2 / su2_restart (default: keep the value in su2.cfg)
#   BACK_PRES outlet back pressure [Pa] for su2 / su2_restart (default: keep the value in su2.cfg),
#             e.g. step it down over several su2_restart runs to converge a low back pressure
#   CFL_MAX   largest adaptive CFL for su2 / su2_restart (default: keep su2.cfg's, 100 from setup);
#             lower it (e.g. 20) when a back-pressure step makes SU2 diverge
#   TARGET    last step for sparta_restart (default: newest restart step + [sparta] run_steps)
#   RUNS      runs directory (default: runs)
#   PY, SU2_CFD, SPARTA   executables (default: python3, SU2_CFD, spa_mpi)
#
# Each stage stops the script on failure. Solver output: su2/su2.log, sparta/log.sparta
# (a continuation writes sparta/log.<from>-<to>.sparta). Memory: check the particle
# estimate in sparta/setup.log against the memory you request from PBS.

#PBS -N spartu2
#PBS -l nodes=1:ppn=20
#PBS -l walltime=24:00:00
#PBS -j oe
# PBS Pro uses instead:  #PBS -l select=1:ncpus=20:mpiprocs=20

set -euo pipefail
cd "${PBS_O_WORKDIR:-$PWD}"

: "${CASE:?set CASE to a case file, e.g. CASE=examples/cd_nozzle_n2.toml}"
STAGES=${STAGES:-"su2_setup su2 handoff sparta_setup sparta"}
if [[ -z ${NP:-} ]]; then
    if [[ -n ${PBS_NODEFILE:-} ]]; then NP=$(wc -l < "$PBS_NODEFILE"); else NP=$(nproc); fi
fi
MPIRUN=${MPIRUN:-"mpirun -np $NP"}
RUNS=${RUNS:-runs}
PY=${PY:-python3}
SU2_CFD=${SU2_CFD:-SU2_CFD}
SPARTA=${SPARTA:-spa_mpi}

# case_value SECTION KEY DEFAULT: a value from the case file
case_value() {
    "$PY" -c 'import sys, tomllib
c = tomllib.load(open(sys.argv[1], "rb"))
print(c.get(sys.argv[2], {}).get(sys.argv[3], sys.argv[4]))' "$CASE" "$1" "$2" "$3"
}
NAME=$(case_value case name "")
SU2_DIR=$RUNS/$NAME/su2
SPA_DIR=$RUNS/$NAME/sparta

# su2_set KEY VALUE: set "KEY= VALUE" in su2.cfg
su2_set() { sed -i "s/^$1=.*/$1= $2/" "$SU2_DIR/su2.cfg"; }

for stage in $STAGES; do
    echo "=== $stage: case $NAME, $NP ranks, $(date '+%F %T') ==="
    case $stage in
        su2_setup)
            "$PY" su2/setup.py "$CASE" --runs "$RUNS" ;;
        su2 | su2_restart)
            if [[ $stage == su2_restart ]]; then
                [[ -f $SU2_DIR/restart_flow.dat ]] || { echo "ERROR: no $SU2_DIR/restart_flow.dat" >&2; exit 1; }
                n=1; while [[ -d $SU2_DIR/archive/$n ]]; do n=$((n + 1)); done
                mkdir -p "$SU2_DIR/archive/$n"
                for f in flow.vtu history.csv su2.log restart_flow.dat su2.cfg; do
                    if [[ -f $SU2_DIR/$f ]]; then cp -p "$SU2_DIR/$f" "$SU2_DIR/archive/$n/"; fi
                done
                su2_set RESTART_SOL YES
            else
                su2_set RESTART_SOL NO
            fi
            if [[ -n ${SU2_ITER:-} ]]; then su2_set ITER "$SU2_ITER"; fi
            if [[ -n ${BACK_PRES:-} ]]; then  # outlet BC; FREESTREAM_PRESSURE only sets the initial state
                su2_set MARKER_OUTLET "( outlet, $BACK_PRES )"
                su2_set FREESTREAM_PRESSURE "$BACK_PRES"
            fi
            if [[ -n ${CFL_MAX:-} ]]; then su2_set CFL_ADAPT_PARAM "( 0.5, 1.2, 0.5, $CFL_MAX )"; fi
            grep -E '^(RESTART_SOL|ITER|MARKER_OUTLET|CFL_ADAPT_PARAM)=' "$SU2_DIR/su2.cfg"
            (cd "$SU2_DIR" && $MPIRUN "$SU2_CFD" su2.cfg > su2.log 2>&1) ;;
        handoff)
            "$PY" handoff/handoff.py "$CASE" --runs "$RUNS" ;;
        sparta_setup)
            "$PY" sparta/setup.py "$CASE" --runs "$RUNS" ;;
        sparta)
            (cd "$SPA_DIR" && $MPIRUN "$SPARTA" -in in.sparta -screen none) ;;
        sparta_restart)
            step=$(cd "$SPA_DIR" && ls restart.*.sparta 2>/dev/null \
                   | sed -n 's/^restart\.\([0-9]*\)\.sparta$/\1/p' | sort -n | tail -1 || true)
            [[ -n $step ]] || { echo "ERROR: no restart.<step>.sparta in $SPA_DIR" >&2; exit 1; }
            target=${TARGET:-$((step + $(case_value sparta run_steps 1000)))}
            echo "continuing from restart.$step.sparta up to step $target"
            (cd "$SPA_DIR" && $MPIRUN "$SPARTA" -in in.continue -var target "$target" \
                -var seed $((10007 + step)) -log "log.$step-$target.sparta" -screen none) ;;
        *)
            echo "ERROR: unknown stage '$stage'" >&2; exit 1 ;;
    esac
done
echo "=== done $(date '+%F %T') ==="
