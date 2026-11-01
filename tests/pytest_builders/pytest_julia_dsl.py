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


# --- names, labels and units -----------------------------------------------

def test_names_that_clash_get_a_number(tmp_path):
    # `end` is reserved in Julia (end_var), and `end var` cleans to the same
    _, code = translate(tmp_path, """
end = 1
	~	Dmnl
	~		|

end var = 2
	~	Dmnl
	~		|

total = end + end var
	~	Dmnl
	~		|
""")
    assert re.search(r"total ~ end_var \+ end_var_1|total ~ end_var_1 \+ end_var", code)


def test_numeric_elements_are_symbols(tmp_path):
    _, code = translate(tmp_path, """
layer: (l1-l3)
	~	~		|

depth: 1, 2, 3
	~	~		|

thickness[depth] = 1, 2, 3
	~	m
	~		|
""")
    assert 'Symbol("1"), Symbol("2"), Symbol("3")' in code


def test_helpers():
    from pysd.builders.julia.dsl_builder import number, translate_units
    assert number(True) == "true" and number(False) == "false"
    # units it can't read are left without a unit
    assert translate_units("Month @ 2") == (None, [])


# --- time settings -----------------------------------------------------------

def test_time_step_as_a_fraction_and_saveper_by_default(tmp_path):
    path = tmp_path / "fraction.mdl"
    path.write_text("{UTF-8}\nstock = INTEG(1, 0)\n\t~\tDmnl\n\t~\t\t|\n\n" + CONTROL
                    .replace("TIME STEP  = 0.5", "TIME STEP  = 1/4")
                    .replace("SAVEPER  = 1", "SAVEPER  = TIME STEP"), encoding="utf-8")
    main = pysd.translate_to_julia(str(path), backend="dsl")
    text = main.read_text(encoding="utf-8")
    assert "time = 0:0.25:10, saveat = 0.25," in text


# --- parameters and expressions ----------------------------------------------

def test_parameter_in_parts_with_except(tmp_path):
    _, code = translate(tmp_path, """
dim: a, b, c
	~	~		|

x[dim] :EXCEPT: [a] = 1
	~	Dmnl
	~		|

x[a] = 2
	~	Dmnl
	~		|
""")
    assert "x[dim = At(:a)]" in code or "x[dim = Not(At(:a))]" in code


def test_excel_constant_in_a_variable_that_changes_with_time(tmp_path):
    _, code = translate(tmp_path, """
growth = GET DIRECT CONSTANTS('inputs.xlsx', 'Sheet1', 'B2') * Time
	~	Dmnl
	~		|
""")
    assert 'description = "Read from Excel for growth."' in code
    assert re.search(r"growth ~ growth_input \* t", code)


def test_initial_of_a_constant_is_the_constant(tmp_path):
    _, code = translate(tmp_path, """
k = 3
	~	Dmnl
	~		|

start = INITIAL(k)
	~	Dmnl
	~		|

stock = INTEG(start, 0)
	~	Dmnl
	~		|
""")
    assert "∂t(stock) ~ start" in code
    assert re.search(r"start ~ k\b|start = k\b", code)


def test_unary_plus(tmp_path):
    _, code = translate(tmp_path, """
a = 2
	~	Dmnl
	~		|

b = +a * Time
	~	Dmnl
	~		|
""")
    assert re.search(r"b ~ a \* t", code)


def test_pulse_with_times_of_an_array(tmp_path):
    _, code = translate(tmp_path, """
dim: a, b
	~	~		|

start[dim] = 1, 2
	~	Month
	~		|

p[dim] = PULSE(start[dim], 1)
	~	Dmnl
	~		|
""")
    assert "(start .≤ t .< start .+ 1)" in code


# --- what isn't translated yet: an error that names the equation -------------

@pytest.mark.parametrize("equations, message", [
    ("""
a = 1
	~	Dmnl
	~		|

b = 2
	~	Dmnl
	~		|

late = DELAY FIXED(a + b, 2, 0)
	~	Dmnl
	~		|
""", "late: DELAY FIXED of an expression"),
    ("""
a: a1, a2
	~	~		|

b: b1, b2
	~	~		|

x[a] = 1, 2
	~	Dmnl
	~		|

y[b] = x[b]
	~	Dmnl
	~		|
""", r"y: x\[b\]: elements it hasn.t"),
    ("""
a: a1, a2
	~	~		|

b: b1, b2
	~	~		|

c: c1, c2
	~	~		|

m[b, c] = 1, 0; 0, 1;
	~	Dmnl
	~		|

inverse[a, b, c] = INVERT MATRIX(m[b, c], 2)
	~	Dmnl
	~		|
""", "inverse: INVERT MATRIX over 3 dimensions"),
])
def test_not_translated_yet(tmp_path, equations, message):
    with pytest.raises(Exception, match=message):
        translate(tmp_path, equations)


def test_random_functions(tmp_path):
    _, code = translate(tmp_path, """
dim: a, b
	~	~		|

noise = RANDOM UNIFORM(0, 1, 7)
	~	Dmnl
	~		|

shocks[dim] = 10 + RANDOM NORMAL(-5, 5, 0, 1, 3)
	~	Dmnl
	~		|

waits = RANDOM EXPONENTIAL(0, 9, 1, 2, 4)
	~	Dmnl
	~		|

fraction = RANDOM 0 1()
	~	Dmnl
	~		|
""")
    assert "noise ~ random_uniform(0, 1; seed = 7)" in code
    # an equation for all of an array: a draw per element
    assert "random_normal(-5, 5, 0, 1; seed = 3, dims = dim)" in code
    assert "waits ~ random_exponential(0, 9, 1, 2; seed = 4)" in code
    assert "fraction ~ random_0_1()" in code


def test_forecast_active_initial_and_get_time_value(tmp_path):
    _, code = translate(tmp_path, """
input = 1 + Time
	~	Dmnl
	~		|

forecast = FORECAST(input, 2, 3)
	~	Dmnl
	~		|

buffered = ACTIVE INITIAL(input * 2, 5) + 1
	~	Dmnl
	~		|

ahead = GET TIME VALUE(0, 0.5, 0)
	~	Month
	~		|
""")
    # FORECAST as Python PySD's: a smooth that starts at input / (1 + initial trend)
    assert "smooth(input, 2; init = input / (1 + 0))" in code
    assert re.search(r"forecast ~ input \* \(1 \+ zidz\(input - smooth", code)
    # ACTIVE INITIAL inside an expression: a variable of its own
    assert "buffered_active_initial ~ input * 2" in code
    assert "buffered_active_initial(t₀) ~ 5" in code
    assert "buffered ~ buffered_active_initial + 1" in code
    assert "ahead ~ t + 0.5" in code


def test_a_function_not_translated_stops_the_model(tmp_path):
    # as Python PySD: a warning, and a call that stops the model, naming it
    with pytest.warns(UserWarning, match="VECTOR SELECT.*will crash"):
        _, code = translate(tmp_path, """
dim: a, b
	~	~		|

x[dim] = 1, 2
	~	Dmnl
	~		|

pick = VECTOR SELECT(x[dim!], x[dim!], 0, 0, 0)
	~	Dmnl
	~		|
""")
    assert 'pysd_not_implemented("VECTOR SELECT"' in code
