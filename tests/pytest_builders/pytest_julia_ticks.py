"""
Julia builder: fixed delays and SAMPLE IF TRUE with any solver.

They move by whole steps of TIME STEP, the model's ticks, as Vensim.  The
translated model records them at the ticks (PySD.jl's ``PySDTicks``): a solver
stops at every tick, where each delay's input and each sample's value are
recorded, and they are held in between.  With Euler steps of TIME STEP the
values are Vensim's (and Python PySD's); other solvers run the same model.

The MTK backend does this (the DSL backend has its own tests).
"""
import json
import os
import subprocess
import warnings

import numpy as np
import pytest

from tests.pytest_builders.pytest_julia_reference import _julia_unavailable

CONTROL = """
FINAL TIME = 10
	~	Month
	~		|

INITIAL TIME = 0
	~	Month
	~		|

SAVEPER = 1
	~	Month
	~		|

TIME STEP = 0.5
	~	Month
	~		|
"""

# a fixed delay and a sample of inputs that depend on the time only, and a
# stock fed by both
STEPWISE = """{UTF-8}
input = STEP(1, 1) + Time / 4
	~	Dmnl
	~		|

delayed = DELAY FIXED(input, 2, 0)
	~	Dmnl
	~		|

sampled = SAMPLE IF TRUE(MODULO(Time, 2) = 0, input, 0)
	~	Dmnl
	~		|

stock = INTEG(delayed + sampled - stock / 2, 0)
	~	Dmnl
	~		|
""" + CONTROL

CHANGING_DELAY = """{UTF-8}
input = STEP(1, 1)
	~	Dmnl
	~		|

delay time = 2 + Time / 10
	~	Month
	~		|

delayed = DELAY N(input, delay time, 0, 3)
	~	Dmnl
	~		|
""" + CONTROL

PLAIN = """{UTF-8}
input = Time / 10
	~	Dmnl
	~		|

smoothed = SMOOTH(input, 2)
	~	Dmnl
	~		|

stock = INTEG(smoothed, 0)
	~	Dmnl
	~		|
""" + CONTROL


def _translate(tmp_path, text, backend, name="model"):
    from pysd import translate_to_julia
    path = tmp_path / backend / f"{name}.mdl"
    path.parent.mkdir(exist_ok=True)
    path.write_text(text, encoding="utf-8")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        jl = translate_to_julia(path, backend=backend)
    return jl, jl.read_text(encoding="utf-8")


# --- what the translation writes ---------------------------------------------

@pytest.mark.parametrize("backend", ["mtk"])
def test_records_delays_and_samples_at_the_ticks(tmp_path, backend):
    _, code = _translate(tmp_path, STEPWISE, backend)
    assert "const _ticks = PySDTicks(initial_time, time_step)" in code
    assert "pysd_fixed_delay(PySDRecord(_ticks, 1, 4), 0.0, " in code
    assert "pysd_sampled(PySDRecord(_ticks, 2, 0), " in code
    # no stocks of their own, and any solver
    assert "_df_pipe_" not in code and "_sit_" not in code
    assert "_check_euler" not in code
    # run_model and the problem stop at the ticks
    assert "_ticks_callback(" in code and "pysd_free_ticks!(_ticks)" in code


def test_mtk_records_the_observed_values(tmp_path):
    _, code = _translate(tmp_path, STEPWISE, "mtk")
    assert "observed(sys, [_df_in_delayed, sampled])" in code
    assert "prob = remake(prob; callback = _ticks_callback())" in code


@pytest.mark.parametrize("backend", ["mtk"])
def test_records_a_changing_delay_time(tmp_path, backend):
    _, code = _translate(tmp_path, CHANGING_DELAY, backend)
    assert "_dlt_in_delayed" in code
    assert "pysd_fixed_delay(PySDRecord(_ticks, 1, 1), " in code and "pysd_fixed_delay(PySDRecord(_ticks, 1, 2), " in code
    assert "_check_euler" not in code


@pytest.mark.parametrize("backend", ["mtk"])
def test_other_models_have_no_ticks(tmp_path, backend):
    _, code = _translate(tmp_path, PLAIN, backend)
    assert "_ticks" not in code


# --- in Julia --------------------------------------------------------------------

def _require_julia(backend):
    reason = _julia_unavailable(backend)
    if reason:
        if os.environ.get("PYSD_JULIA_REQUIRED"):
            pytest.fail(reason)
        pytest.skip(reason)


# the values at every month, of each solver's run (from the solution, by symbol)
_VALUES = {
    "mtk": """
vals_of(sol, n) = (v = sol[getproperty(@__MODULE__, Symbol(n))];
                  [v[i] for (i, t) in enumerate(sol.t) if isinteger(t)])
""",
}


@pytest.mark.julia
@pytest.mark.parametrize("backend", ["mtk"])
def test_any_solver(tmp_path, backend):
    _require_julia(backend)
    import pysd
    jl, _ = _translate(tmp_path, STEPWISE, backend)
    script = f"""
include({json.dumps(str(jl))})
using OrdinaryDiffEq
{_VALUES[backend]}
function report(label, sol)
    for n in ("delayed", "sampled", "stock")
        println("@", label, ":", n, "=", join(vals_of(sol, n), ","))
    end
end
report("euler", run_model())
report("half", run_model(dt = time_step / 2))
report("rk4", run_model(solver = RK4()))
report("tsit5", run_model(solver = Tsit5(), adaptive = true, abstol = 1e-12, reltol = 1e-12))
"""
    path = tmp_path / "run.jl"
    path.write_text(script, encoding="utf-8")
    run = subprocess.run(["julia", "--startup-file=no", str(path)],
                         capture_output=True, text=True, timeout=1800)
    assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-3000:]
    out = {}
    for line in run.stdout.splitlines():
        if line.startswith("@"):
            name, values = line[1:].split("=", 1)
            out[name] = np.array([float(x) for x in values.split(",")])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        py = pysd.read_vensim(str(tmp_path / backend / "model.mdl")).run(
            return_columns=["delayed", "sampled", "stock"])
    # Euler steps of TIME STEP: Python PySD's values
    for n in ("delayed", "sampled", "stock"):
        np.testing.assert_allclose(out[f"euler:{n}"], py[n].values, rtol=1e-12, atol=1e-12)
    # the delay and the sample move at the ticks: the same with any solver
    for label in ("half", "rk4", "tsit5"):
        for n in ("delayed", "sampled"):
            np.testing.assert_array_equal(out[f"{label}:{n}"], out[f"euler:{n}"])
    # the stock: halving Euler's step brings it closer to the ODE's solution
    exact = out["tsit5:stock"]
    err = lambda s: np.max(np.abs(s - exact))
    assert err(out["half:stock"]) < err(out["euler:stock"]) * 0.6
    assert err(out["rk4:stock"]) < err(out["half:stock"])


# --- switches on the time, and equations written for steps of TIME STEP ---------

SWITCHES = """{UTF-8}
stepped = INTEG(STEP(1, 2), 0)
	~	Dmnl
	~		|

pulsed = INTEG(2 * PULSE(1, 0.5), 0)
	~	Dmnl
	~		|

year met = INTEG(IF THEN ELSE(Time = 2, Time / TIME STEP, 0), 0)
	~	Dmnl
	~		|

ahead = lookahead(Time + TIME STEP)
	~	Dmnl
	~		|

lookahead(
	[(0,0)-(10,10)],(0,0),(10,10))
	~	Dmnl
	~		|
""".replace("FINAL TIME = 10", "") + CONTROL.replace("FINAL TIME = 10", "FINAL TIME = 4")


@pytest.mark.parametrize("backend", ["mtk"])
def test_switches_on_the_time_stop_at_the_ticks(tmp_path, backend):
    _, code = _translate(tmp_path, SWITCHES, backend)
    assert "const _ticks = PySDTicks(initial_time, time_step)" in code
    assert "pysd_step(pysd_tick_time(_ticks, t), 1.0, 2.0)" in code
    assert "_ticks_callback(" in code
    # no fixed delays or samples: nothing to record
    assert "_record_ticks!(u, t, p = " in code and "pysd_record!" not in code


@pytest.mark.parametrize("backend", ["mtk"])
def test_equations_written_for_steps_of_time_step_are_named(tmp_path, backend):
    _, code = _translate(tmp_path, SWITCHES, backend)
    assert "function _warn_stepwise(solver, dt)" in code
    assert "    _warn_stepwise(solver, dt)\n" in code
    assert "1 use TIME STEP as a value (a move or a rate per step): year_met" in code
    assert "1 test the time for equality (true for one Euler step, at one instant here): year_met" in code
    # a time given to a lookup (Time + TIME STEP) is a time, not one of these
    assert "ahead" not in code.split("function _warn_stepwise")[1].split("end")[0]


@pytest.mark.parametrize("backend", ["mtk"])
def test_models_without_them_have_neither(tmp_path, backend):
    _, code = _translate(tmp_path, PLAIN.replace("STEP(1, 1)", "Time / 10"), backend)
    assert "_ticks" not in code and "_warn_stepwise" not in code


@pytest.mark.julia
@pytest.mark.parametrize("backend", ["mtk"])
def test_switches_are_exact_with_any_solver(tmp_path, backend):
    _require_julia(backend)
    jl, _ = _translate(tmp_path, SWITCHES, backend)
    final = {"mtk": "sol[getproperty(@__MODULE__, Symbol(n))][end]"}[backend]
    script = f"""
include({json.dumps(str(jl))})
using OrdinaryDiffEq, Logging
final(sol, n) = {final}
for (label, kw) in (("euler", (;)), ("rk4", (; solver = RK4())), ("half", (; dt = time_step / 2)),
                    ("bs3", (; solver = BS3(), adaptive = true)))
    buf = IOBuffer()
    sol = with_logger(ConsoleLogger(buf, Logging.Warn)) do
        run_model(; kw...)
    end
    println("@", label, ":stepped=", final(sol, "stepped"))
    println("@", label, ":pulsed=", final(sol, "pulsed"))
    println("@", label, ":warned=", occursin("year_met", String(take!(buf))) ? 1 : 0)
end
"""
    path = tmp_path / "run_switches.jl"
    path.write_text(script, encoding="utf-8")
    run = subprocess.run(["julia", "--startup-file=no", str(path)],
                         capture_output=True, text=True, timeout=1800)
    assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-3000:]
    out = dict(line[1:].split("=", 1) for line in run.stdout.splitlines() if line.startswith("@"))
    for label in ("euler", "rk4", "half", "bs3"):
        # a step at 2 and a pulse from 1 to 1.5 of 2: at the ticks, exactly
        assert abs(float(out[f"{label}:stepped"]) - 2.0) < 1e-12, (label, out)
        assert abs(float(out[f"{label}:pulsed"]) - 1.0) < 1e-12, (label, out)
    # Euler with steps of TIME STEP is Vensim's: no warning; the others warn
    assert out["euler:warned"] == "0"
    assert out["rk4:warned"] == "1" and out["half:warned"] == "1"


# --- RANDOM: one draw per tick, the same in every run -----------------------------

RANDOMS = """{UTF-8}
dim: a, b
	~	~	|

noise = RANDOM UNIFORM(0, 1, 7)
	~	Dmnl
	~		|

shocks[dim] = 10 + RANDOM NORMAL(-5, 5, 0, 1, 3)
	~	Dmnl
	~		|

stock = INTEG(noise, 0)
	~	Dmnl
	~		|
""" + CONTROL


@pytest.mark.parametrize("backend", ["mtk"])
def test_random_calls_are_drawn_at_the_ticks(tmp_path, backend):
    _, code = _translate(tmp_path, RANDOMS, backend)
    # standard draws, once per tick, from each call's stream (its number and its seed)
    assert "0.0 + (1.0 - 0.0) * pysd_tick_uniform(PySDRecord(_ticks, 1000010, 0), t)" in code
    # a normal truncated to [-5, 5], as Python PySD's
    assert "pysd_tick_normal(PySDRecord(_ticks, 2000009, 0, _i0), pysd_xidz(-5.0 - 0.0, 1.0, -Inf), pysd_xidz(5.0 - 0.0, 1.0, Inf), t)" in code
    assert "pysd_random_" not in code
    assert "_ticks_callback(" in code


@pytest.mark.julia
@pytest.mark.parametrize("backend", ["mtk"])
def test_random_draws_hold_between_ticks_and_repeat(tmp_path, backend):
    _require_julia(backend)
    jl, _ = _translate(tmp_path, RANDOMS, backend)
    script = f"""
include({json.dumps(str(jl))})
using OrdinaryDiffEq
{_VALUES[backend]}
function report(label, sol)
    for n in ("noise", "stock")
        println("@", label, ":", n, "=", join(vals_of(sol, n), ","))
    end
end
report("euler", run_model())
report("again", run_model())
report("half", run_model(dt = time_step / 2))
report("rk4", run_model(solver = RK4()))
"""
    path = tmp_path / "run_random.jl"
    path.write_text(script, encoding="utf-8")
    run = subprocess.run(["julia", "--startup-file=no", str(path)],
                         capture_output=True, text=True, timeout=1800)
    assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-3000:]
    out = {}
    for line in run.stdout.splitlines():
        if line.startswith("@"):
            name, values = line[1:].split("=", 1)
            out[name] = np.array([float(x) for x in values.split(",")])
    noise = out["euler:noise"]
    assert np.all((0 <= noise) & (noise < 1)) and len(np.unique(noise)) > 5
    # the same draws in every run, with any solver; the stock of a noise held
    # between ticks is the same with any of them
    for label in ("again", "half", "rk4"):
        np.testing.assert_array_equal(out[f"{label}:noise"], noise)
        np.testing.assert_allclose(out[f"{label}:stock"], out["euler:stock"], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("backend", ["mtk"])
def test_every_random_function(tmp_path, backend):
    _, code = _translate(tmp_path, """{UTF-8}
my seed = 5
	~	Dmnl
	~		|

unit = RANDOM 0 1()
	~	Dmnl
	~		|

waits = RANDOM EXPONENTIAL(1, 9, 2, 3, 4)
	~	Dmnl
	~		|

seeded by a name = RANDOM UNIFORM(0, 1, my seed)
	~	Dmnl
	~		|
""" + CONTROL, backend)
    assert "pysd_tick_uniform(PySDRecord(_ticks, 1000003, 0), t)" in code       # no seed: 0
    # (min, max, shift, stretch, seed): from max(min, shift), stretch times an exponential, below max
    assert "max(1.0, 2.0) + 3.0 * pysd_tick_exponential(PySDRecord(_ticks, 2000010, 0), pysd_xidz(9.0 - max(1.0, 2.0), 3.0, Inf), t)" in code
    # a seed that isn't a number counts as 0
    assert "pysd_tick_uniform(PySDRecord(_ticks, 3000009, 0), t)" in code
