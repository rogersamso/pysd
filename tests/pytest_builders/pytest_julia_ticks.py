"""
Julia builder: fixed delays and SAMPLE IF TRUE with any solver.

They move by whole steps of TIME STEP, the model's ticks, as Vensim.  The
translated model records them at the ticks (PySD.jl's ``PySDTicks``): a solver
stops at every tick, where each delay's input and each sample's value are
recorded, and they are held in between.  With Euler steps of TIME STEP the
values are Vensim's (and Python PySD's); other solvers run the same model.

Both the ODE and the MTK backends do this.
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
input = STEP(1, 1)
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

@pytest.mark.parametrize("backend", ["ode", "mtk"])
def test_records_delays_and_samples_at_the_ticks(tmp_path, backend):
    _, code = _translate(tmp_path, STEPWISE, backend)
    assert "const _ticks = PySDTicks(initial_time, time_step)" in code
    assert "pysd_fixed_delay(_ticks, 1, 4, 0.0, t)" in code
    assert "pysd_sampled(_ticks, 2, " in code
    # no stocks of their own, and any solver
    assert "_df_pipe_" not in code and "_sit_" not in code
    assert "_check_euler" not in code
    # run_model and the problem stop at the ticks
    assert "_ticks_callback(" in code and "pysd_free_ticks!(_ticks)" in code


def test_ode_records_what_model_gives(tmp_path):
    _, code = _translate(tmp_path, STEPWISE, "ode")
    assert "pysd_record!(_ticks, 1, hasproperty(v, :_df_in_delayed) ? v._df_in_delayed : p._df_in_delayed)" in code
    assert "pysd_record!(_ticks, 2, hasproperty(v, :sampled) ? v.sampled : p.sampled)" in code
    assert "prob = ODEProblem(rhs!, u0, tspan, _problem_p(_params); callback = _ticks_callback())" in code


def test_mtk_records_the_observed_values(tmp_path):
    _, code = _translate(tmp_path, STEPWISE, "mtk")
    assert "observed(sys, [_df_in_delayed, sampled])" in code
    assert "prob = remake(prob; callback = _ticks_callback())" in code


@pytest.mark.parametrize("backend", ["ode", "mtk"])
def test_records_a_changing_delay_time(tmp_path, backend):
    _, code = _translate(tmp_path, CHANGING_DELAY, backend)
    assert "_dlt_in_delayed" in code
    assert "pysd_fixed_delay(_ticks, 1, 1, " in code and "pysd_fixed_delay(_ticks, 1, 2, " in code
    assert "_check_euler" not in code


@pytest.mark.parametrize("backend", ["ode", "mtk"])
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


# the values at every month, of each solver's run (ODE: from observe; MTK: from
# the solution, by symbol)
_VALUES = {
    "ode": """
vals_of(sol, n) = (obs = [observe(u, t) for (u, t) in zip(sol.u, sol.t)];
                  [obs[i][n] for (i, t) in enumerate(sol.t) if isinteger(t)])
""",
    "mtk": """
vals_of(sol, n) = (v = sol[getproperty(@__MODULE__, Symbol(n))];
                  [v[i] for (i, t) in enumerate(sol.t) if isinteger(t)])
""",
}


@pytest.mark.julia
@pytest.mark.parametrize("backend", ["ode", "mtk"])
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
