import sys
from su2.setup import setup as su2_setup
from su2.run import run as su2_run
from sparta.setup import setup as sparta_setup
from sparta.run import run as sparta_run

case = "examples/cd_nozzle_n2.toml"
#d = "runs/cd_nozzle_n2/su2"        # SU2 run directory
#s = "runs/cd_nozzle_n2/sparta"     # SPARTA run directory

#d = su2_setup(case)                              # new SU2 mesh + config (remeshes)
#su2_run(d)                                       # SU2 from scratch
#su2_run(d, restart=True, iterations=20000)       # continue SU2 from restart_flow.dat

s = sparta_setup(case)                           # new grid + inflow.sparta (overwrites in.sparta edits)
#sys.exit(sparta_run(s, nprocs=18, max_mem_gb=8))                              # DSMC from an empty box
sys.exit(sparta_run(s, nprocs=18, max_mem_gb=30, restart=False, target=80000))   # continue DSMC to step 30000

