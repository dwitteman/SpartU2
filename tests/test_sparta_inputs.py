"""SPARTA input generation in sparta/setup.py: hold, grid adaptation, continuation.
String checks only; never runs SPARTA."""
import re

import pytest

from sparta.setup import adapt_hold, in_continue, in_run

BOX = (0.005, 0.12, 0.0, 0.06)
IO = dict(out_every=5000, avg_steps=2500, restart_every=10000)


def commands(text):
    """Non-comment, non-blank lines, whitespace-normalised."""
    return [" ".join(l.split()) for l in text.splitlines() if l.strip() and not l.lstrip().startswith("#")]


def index_of(cmds, pattern):
    return next(i for i, c in enumerate(cmds) if re.match(pattern, c))


def test_adapt_hold_defaults_and_checks():
    h = adapt_hold({"sparta": {"hold_steps": 25000, "ppc_target": 20}})
    assert h == dict(steps=25000, adapt=True, window=1000, refine=80.0, coarsen=20.0)
    assert adapt_hold({"sparta": {}})["steps"] == 0
    with pytest.raises(SystemExit, match="divide hold_steps"):
        adapt_hold({"sparta": {"hold_steps": 25000, "adapt_window": 3000}})
    with pytest.raises(SystemExit, match="below adapt_refine_count"):
        adapt_hold({"sparta": {"hold_steps": 25000, "adapt_refine_count": 10, "adapt_coarsen_count": 10}})


def test_run_with_hold_orders_hold_adapt_then_averaging():
    hold = adapt_hold({"sparta": {"hold_steps": 25000, "ppc_target": 20}})
    c = commands(in_run("case.toml", BOX, "N2", "n2", 1e17, 1e-8, 10000, hold=hold, **IO))
    order = [r"run 24000$", r"fix hold_n ave/grid all 1 1000 1000 c_hold_n\[1\]$", r"run 1000$",
             r"adapt_grid all refine coarsen value f_hold_n 80 20 combine sum$", r"unfix hold_n$",
             r"balance_grid rcb part$", r"write_restart", r"run 1$", r"fix avg ave/grid",
             r"dump flow grid all 5000", r"run 9999$"]
    idx = [index_of(c, p) for p in order]
    assert idx == sorted(idx)
    assert c[-1] == "write_restart restart.*.sparta"  # after the hold and at the end
    # total = hold + run_steps
    assert sum(int(m.group(1)) for x in c if (m := re.fullmatch(r"run (\d+)", x))) == 25000 + 10000


def test_run_without_hold_has_no_adaptation():
    c = commands(in_run("case.toml", BOX, "N2", "n2", 1e17, 1e-8, 10000, **IO))
    assert not any("adapt_grid" in x or "hold_n" in x for x in c)
    assert [x for x in c if x.startswith("run ")] == ["run 1", "run 9999"]


def test_hold_without_adapt_just_runs():
    hold = adapt_hold({"sparta": {"hold_steps": 20000, "adapt": False}})
    c = commands(in_run("case.toml", BOX, "N2", "n2", 1e17, 1e-8, 10000, hold=hold, **IO))
    assert not any("adapt_grid" in x for x in c)
    assert [x for x in c if x.startswith("run ")] == ["run 20000", "run 1", "run 9999"]


def test_continue_reads_restart_and_does_not_adapt():
    c = commands(in_continue("case.toml", "n2", 10000, **IO))
    assert c.index("read_restart restart.*.sparta balance rcb cell") < index_of(c, r"fix avg")
    assert not any("adapt_grid" in x for x in c)
    assert c[-2:] == ["run ${target} upto", "write_restart restart.*.sparta"]
