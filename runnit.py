# Python setup steps, one line per action; comment out what you don't need.
# The solvers are launched from bash: see run_case.sh, which runs these same steps too.
from handoff.handoff import handoff
from sparta.setup import setup as sparta_setup
from su2.setup import setup as su2_setup

case = "examples/cd_nozzle_n2.toml"

#su2_setup(case)          # new SU2 mesh + su2.cfg -> runs/<name>/su2/ (remeshes)
#   then: CASE=... STAGES="su2" bash run_case.sh
handoff(case)             # coupling plane + inflow.sparta from su2/flow.vtu -> runs/<name>/handoff/
sparta_setup(case)        # grid + in.sparta -> runs/<name>/sparta/ (overwrites in.sparta edits)
#   then: CASE=... STAGES="sparta" bash run_case.sh
