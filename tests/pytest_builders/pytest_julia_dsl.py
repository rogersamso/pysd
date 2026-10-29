"""
Julia builder, DSL backend: what the translation writes for small models.

These tests only translate (no Julia needed): each writes a small Vensim
model, translates it with ``backend="dsl"`` and checks the Julia code.
"""
import re

import pytest

import pysd
from pysd.builders.julia.julia_model_builder import PYSD_JL_VERSION

CONTROL = r"""
********************************************************
	.Control
********************************************************~
		Simulation Control Parameters
	|

FINAL TIME  = 10
	~	Month
	~	The final time for the simulation.
	|

INITIAL TIME  = 0
	~	Month
	~	The initial time for the simulation.
	|

SAVEPER  = 1
	~	Month [0,?]
	~	The frequency with which output is stored.
	|

TIME STEP  = 0.5
	~	Month [0,?]
	~	The time step for the simulation.
	|

\\\---/// Sketch information - do not modify anything except names
V300  Do not put anything below this section - it will be ignored
"""


def translate(tmp_path, equations, name="small_model"):
    """Translate a model made of ``equations`` (Vensim text): its folder and
    all its Julia code, joined."""
    path = tmp_path / f"{name}.mdl"
    path.write_text("{UTF-8}\n" + equations + CONTROL, encoding="utf-8")
    main = pysd.translate_to_julia(str(path), backend="dsl")
    folder = main.parent
    code = "\n".join(f.read_text(encoding="utf-8") for f in sorted(folder.glob("*.jl")))
    return folder, code


def test_folder_is_a_package(tmp_path):
    folder, _ = translate(tmp_path, """
stock = INTEG(1, 0)
	~	Dmnl
	~		|
""", name="my_small_model")
    project = (folder / "Project.toml").read_text(encoding="utf-8")
    assert 'name = "MySmallModel"' in project
    assert re.search(r'^PySD = "[0-9a-f-]{36}"$', project, re.M)
    # the PySD.jl the translation is written for, at least
    assert f'[compat]\nPySD = "{PYSD_JL_VERSION}"' in project
    source = (folder / "src" / "MySmallModel.jl").read_text(encoding="utf-8")
    assert "module MySmallModel" in source
    assert '@precompile_model model = include("../my_small_model.jl")' in source


def test_data_without_equation_comes_from_parents(tmp_path):
    _, code = translate(tmp_path, """
outside temperature:INTERPOLATE:
	~	Dmnl
	~	Given by another model's run.
	|

warming = outside temperature * 2
	~	Dmnl
	~		|
""")
    assert "outside_temperature = data(parents, :outside_temperature)" in code
    assert "warming ~ outside_temperature(t) * 2" in code
    assert "parents = ()" in code                     # an input, given to the run


def test_inline_lookup_is_a_lookup_of_its_own(tmp_path):
    _, code = translate(tmp_path, """
x = Time
	~	Dmnl
	~		|

y = WITH LOOKUP(x, ([(0,0)-(10,1)],(0,0),(10,1)))
	~	Dmnl
	~		|
""")
    assert "y_lookup = piecewise_linear(0 => 0, 10 => 1),\n" in code
    assert '[unit = NoUnits, description = "The lookup of y."]' in code   # y has no description
    assert "y ~ y_lookup(x)" in code


def test_fixed_delay_of_a_constant(tmp_path):
    _, code = translate(tmp_path, """
c = 5
	~	Dmnl
	~		|

y = DELAY FIXED(c, 2, 1)
	~	Dmnl
	~		|
""")
    assert "y ~ c(t - 2; init = 1)" in code


def test_fixed_delay_initial_value_as_the_delayed_array(tmp_path):
    # the initial value is the delayed reference itself: the whole array, as it,
    # also when the reference renames a dimension (another range of the same
    # elements, as x is defined element by element)
    _, code = translate(tmp_path, """
A: a1, a2
	~	
	~		|

B: a1, a2
	~	
	~		|

x[a1] = 1
	~~|
x[a2] = 2
	~	Dmnl
	~		|

y[B] = DELAY FIXED(x[B], 1, x[B])
	~	Dmnl
	~		|
""")
    m = re.search(r"^\s*y ~ (.*)$", code, re.M)
    assert m, code
    assert "x(t - 1; init = x)" in m.group(1), m.group(1)
    assert "set(" in m.group(1)                        # renamed after the delay


def test_units_module_registers_itself(tmp_path):
    folder, _ = translate(tmp_path, """
stock = INTEG(1, 0)
	~	widgets
	~		|
""")
    units = (folder / "units.jl").read_text(encoding="utf-8")
    assert "function __init__()" in units              # when a package loads it
    assert units.rstrip().endswith("Unitful.register(ModelUnits)")   # and when it's defined


def test_sum_over_several_dimensions_at_once(tmp_path):
    # one sum over both, as Python adds them (not one sum inside the other)
    _, code = translate(tmp_path, """
A: a1, a2
	~	
	~		|

B: b1, b2, b3
	~	
	~		|

x[A, B] = 1
	~	Dmnl
	~		|

total = SUM(x[A!, B!])
	~	Dmnl
	~		|

by_a[A] = SUM(x[A, B!])
	~	Dmnl
	~		|
""")
    assert re.search(r"total ~ sum_over\(x, \(a, b\)\)", code), code
    assert re.search(r"by_a ~ sum_over\(x, b\)", code), code


def test_functions_and_time_settings(tmp_path):
    _, code = translate(tmp_path, """
pulses = PULSE TRAIN(1, 2, 5, 10)
	~	Dmnl
	~		|

steps = QUANTUM(Time, 3)
	~	Dmnl
	~		|

left = FINAL TIME - Time
	~	Month
	~		|
""")
    assert "pulses ~ pysd_pulse_train(t, 1, 5, 2, 10)" in code     # start, interval, width, end
    assert "steps ~ pysd_quantum(t, 3)" in code
    assert "left ~ 10 - t" in code                                  # FINAL TIME: its value


def test_step_and_ramp_with_times_of_an_array(tmp_path):
    _, code = translate(tmp_path, """
A: a1, a2
	~	
	~		|

start[A] = 1, 3
	~	Month
	~		|

s[A] = STEP(2, start[A])
	~	Dmnl
	~		|

r[A] = RAMP(1, start[A], 8)
	~	Dmnl
	~		|
""")
    assert re.search(r"s ~ 2 \.\* \(t \.≥ start\)", code), code
    assert re.search(r"r ~ 1 \.\* clamp\.\(t \.- start, 0, 8 \.- start\)", code), code


def test_unknown_time_unit_is_declared(tmp_path):
    # a time unit Unitful doesn't know (XMILE's "Time", for instance) is declared
    # in units.jl, as any other unknown unit
    path = tmp_path / "turns.mdl"
    path.write_text("{UTF-8}\nx = 1\n\t~\tDmnl\n\t~\t\t|\n" + CONTROL.replace("Month", "Turn"), encoding="utf-8")
    main = pysd.translate_to_julia(str(path), backend="dsl")
    units = (main.parent / "units.jl").read_text(encoding="utf-8")
    assert "@refunit Turn" in units
