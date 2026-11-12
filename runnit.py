from su2.setup import setup as su2_setup
from su2.run import run as su2_run
from sparta.setup import setup as sparta_setup

d = su2_setup("examples/cd_nozzle_n2.toml")
su2_run(d)
#sparta_setup("my_case.toml")
