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
    # start, interval, width, end
    assert "pulses ~ (1 ≤ t < 10) & (mod(t - 1 + 1e-6, 5) < 2)" in code
    assert "steps ~ 3 * trunc(t / 3)" in code
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


def test_time_settings_that_change_during_the_run(tmp_path):
    path = tmp_path / "changing.mdl"
    path.write_text("{UTF-8}\nstock = INTEG(1, 0)\n\t~\tDmnl\n\t~\t\t|\n\n"
                    "rate = TIME STEP * 2\n\t~\tDmnl\n\t~\t\t|\n\n" + CONTROL
                    .replace("FINAL TIME  = 10", "FINAL TIME  = IF THEN ELSE(stock > 5, Time, 10)")
                    .replace("TIME STEP  = 0.5", "TIME STEP  = IF THEN ELSE(Time < 4, 1, 0.5)")
                    .replace("SAVEPER  = 1", "SAVEPER  = TIME STEP"), encoding="utf-8")
    main = pysd.translate_to_julia(str(path), backend="dsl")
    text = main.read_text(encoding="utf-8")
    assert "time = (0, :final_time), dt = :time_step, saveat = :saveper," in text
    code = "\n".join(f.read_text(encoding="utf-8") for f in main.parent.glob("*.jl"))
    # the settings are variables; TIME STEP in an equation is its variable, not Δt
    assert "final_time ~ ifelse(stock > 5, t, 10)" in code
    assert "time_step ~ ifelse(t < 4, 1, 0.5)" in code
    assert "saveper ~ time_step" in code
    assert "rate ~ time_step * 2" in code
    names = (main.parent / "names.tsv").read_text(encoding="utf-8")
    assert "FINAL TIME\tfinal_time" in names


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


def test_delay_fixed_of_an_expression(tmp_path):
    _, code = translate(tmp_path, """
a = Time
	~	Dmnl
	~		|

late = DELAY FIXED(a + 1, 2, 0)
	~	Dmnl
	~		|
""")
    # a variable of its own, delayed
    assert "late_delayed_expression ~ a + 1" in code
    assert "late ~ late_delayed_expression(t - 2; init = 0)" in code


def test_get_time_value_of_the_initial_time_and_the_clock(tmp_path):
    with pytest.warns(UserWarning, match="GET TIME VALUE.*will crash"):
        folder, code = translate(tmp_path, """
from start = GET TIME VALUE(1, 2, 0)
	~	Month
	~		|

year now = GET TIME VALUE(2, 0, 1)
	~	Dmnl
	~		|

weekday now = GET TIME VALUE(2, 0, 5) + 1
	~	Dmnl
	~		|

month of time = GET TIME VALUE(0, 0, 3)
	~	Dmnl
	~		|
""")
    assert "from_start ~ t₀ + 2" in code
    # the computer's clock, with Julia's Dates (Vensim's measures; 0 is Sunday)
    assert "year_now ~ year(now())" in code
    assert "weekday_now ~ dayofweek(now()) % 7 + 1" in code
    assert "using PySD, DimensionalData, Dates" in code
    assert 'Dates = "ade2ca70-3891-5945-98fb-dc099432e06a"' in (folder / "Project.toml").read_text()
    # as Python PySD: a calendar measure of the model's time stops the model
    assert re.search(r"month_of_time ~ error\(\"Not implemented function 'GET TIME VALUE\(0, …, 3\)'", code)


def test_a_function_not_translated_stops_the_model(tmp_path):
    # as Python PySD: a warning, and a call that stops the model, naming it
    with pytest.warns(UserWarning, match="LOOKUP FORWARD.*will crash"):
        _, code = translate(tmp_path, """
table(
	[(0,0)-(2,2)],(0,0),(1,1),(2,2))
	~	Dmnl
	~		|

ahead = LOOKUP FORWARD(table, Time)
	~	Dmnl
	~		|
""")
    assert 'error("Not implemented function \'LOOKUP FORWARD\'' in code


def test_vector_functions_and_game(tmp_path):
    _, code = translate(tmp_path, """
region: north, south
	~	~		|

tech: coal, gas, solar
	~	~		|

rank: first, second, third
	~	~		|

cost[region, tech] = 3, 1, 2; 2, 3, 1;
	~	Dmnl
	~		|

available[tech] = 1, 0, 1
	~	Dmnl
	~		|

order[region, rank] = VECTOR SORT ORDER(cost[region, tech], 1)
	~	Dmnl
	~		|

ranks[region, tech] = VECTOR RANK(cost[region, tech], 0)
	~	Dmnl
	~		|

sorted[region, rank] = VECTOR REORDER(cost[region, tech], order[region, rank])
	~	Dmnl
	~		|

cheapest[region] = VECTOR SELECT(available[tech!], cost[region, tech!], -1, 8, 0)
	~	Dmnl
	~		|

cheapest anywhere = VECTOR SELECT(available[tech!], cost[region!, tech!], -1, 8, 0)
	~	Dmnl
	~		|

played = GAME(cheapest anywhere * 2)
	~	Dmnl
	~		|
""")
    # along the last subscript, the results along the variable's last, by position
    assert "order ~ vector_sort_order(set(cost, :tech => Dim{:rank}(rank_elements)), 1, rank)" in code
    assert "ranks ~ vector_rank(cost, 0, tech)" in code
    assert "sorted ~ vector_reorder(set(cost, :tech => Dim{:rank}(rank_elements)), order, rank)" in code
    # over the ! subscripts, which the result no longer has
    assert "cheapest ~ vector_select(available, cost, tech, -1, 8, 0)" in code
    assert "cheapest_anywhere ~ vector_select(available, cost, (tech, region), -1, 8, 0)" in code
    assert "played ~ cheapest_anywhere * 2" in code


def test_vector_select_over_a_subscript_kept_elsewhere(tmp_path):
    # dim2 is kept in the selection and summed over (!) in the input: two dimensions
    _, code = translate(tmp_path, """
dim1: a1, a2
	~	~		|

dim2: b1, b2
	~	~		|

sel[dim1, dim2] = 1, 0; 0, 1;
	~	Dmnl
	~		|

input[dim2] = 3, 4
	~	Dmnl
	~		|

out[dim2] = VECTOR SELECT(sel[dim1!, dim2], input[dim2!], -1, 0, 0)
	~	Dmnl
	~		|
""")
    assert "out ~ vector_select(sel, set(input, :dim2 => :dim2_selected), (dim1, :dim2_selected), -1, 0, 0)" in code


def test_data_in_parts_over_two_subscripts(tmp_path):
    _, code = translate(tmp_path, """
region: north, south
	~	~		|

product: x, y
	~	~		|

company: c1, c2
	~	~		|

revenue[north, x, company]:RAW::=
	GET DIRECT DATA('input.xlsx', 'Sheet1', 'time', 'B2') ~~|
revenue[north, y, company]:RAW::=
	GET DIRECT DATA('input.xlsx', 'Sheet1', 'time', 'C2') ~~|
revenue[south, x, company]:RAW::=
	GET DIRECT DATA('input.xlsx', 'Sheet1', 'time', 'D2') ~~|
revenue[south, y, company]:RAW::=
	GET DIRECT DATA('input.xlsx', 'Sheet1', 'time', 'E2')
	~	Dmnl
	~		|
""")
    # a vector per region, of a vector per product
    assert re.search(r"revenue\[region, product, company\] = \[\[data\(\w+, \"time\", \"B2\"; method = :raw\), "
                     r"data\(\w+, \"time\", \"C2\"; method = :raw\)\], \[data", code)


def test_allocate_functions(tmp_path):
    _, code = translate(tmp_path, """
region: north, south
	~	~		|

branch: b1, b2, b3
	~	~		|

pprofile: ptype, ppriority, pwidth, pextra
	~	~		|

request[region, branch] = 5, 3, 2; 1, 4, 6;
	~	Dmnl
	~		|

priority[branch] = 3, 1, 2
	~	Dmnl
	~		|

supply[region] = 6, 20
	~	Dmnl
	~		|

profiles[branch, pprofile] = 1, 3, 1, 0; 1, 1, 1, 0; 1, 2, 1, 0;
	~	Dmnl
	~		|

shipped[region, branch] = ALLOCATE BY PRIORITY(request[region, branch], priority[branch], ELMCOUNT(branch), 1, supply[region])
	~	Dmnl
	~		|

available[region, branch] = ALLOCATE AVAILABLE(request[region, branch], profiles[branch, ptype], 6)
	~	Dmnl
	~		|
""")
    # along the last subscript; the width and the supply over the others
    assert "shipped ~ allocate_by_priority(request, priority, 1, supply, branch)" in code
    # the profiles written from their first element (ptype): the whole range
    assert "available ~ allocate_available(request, profiles, 6, branch, pprofile)" in code


def test_macros_are_written_in_place(tmp_path):
    _, code = translate(tmp_path, """
:MACRO: ACCUMULATE(input, start)
ACCUMULATE = INTEG(input, start)
	~	input
	~		|

:END OF MACRO:
a = 1
	~	Dmnl
	~		|

first = ACCUMULATE(a, 0)
	~	Dmnl
	~		|

second = ACCUMULATE(2 * a, 5)
	~	Dmnl
	~		|
""")
    # one copy of the macro per call: its own stock, its parameters replaced
    assert "∂t(accumulate_output) ~ a" in code
    assert "accumulate_output(t₀) ~ 0" in code
    assert "∂t(accumulate_2_output) ~ 2a" in code or "∂t(accumulate_2_output) ~ 2 * a" in code
    assert "accumulate_2_output(t₀) ~ 5" in code
    assert "first ~ accumulate_output" in code and "second ~ accumulate_2_output" in code
    assert 'unit = u"input"' not in code


def test_except_narrower_at_several_positions(tmp_path):
    _, code = translate(tmp_path, """
store: s1, s2, s3
	~	~		|

product: p1, p2
	~	~		|

x[store, product] :EXCEPT: [s3, p2] = 1
	~	Dmnl
	~		|

x[s3, p2] = 2
	~	Dmnl
	~		|
""")
    # rectangular parts: every store but s3, then s3 for every product but p2
    assert "x[store = Not(At(:s3))] ~ 1" in code
    assert re.search(r"x\[store = At\(\[?:s3\]?\), product = (Not\(At\(:p2\)\)|At\(\[:p1\]\))\] ~ 1", code)
    assert "x[store = At(:s3), product = At(:p2)] ~ 2" in code


def test_initial_of_an_expression(tmp_path):
    _, code = translate(tmp_path, """
start = INITIAL(Time * 2)
	~	Dmnl
	~		|
""")
    assert "start_initial_expression ~ t * 2" in code
    assert "start ~ start_initial_expression(t₀)" in code


def test_subscripts_and_elements_as_values(tmp_path):
    _, code = translate(tmp_path, """
dim: a, b, c
	~	~		|

other: c, a, b
	~	~		|

same[dim, other] = IF THEN ELSE(dim = other, 1, 0)
	~	Dmnl
	~		|

is b[dim] = IF THEN ELSE(dim = b, 1, 0)
	~	Dmnl
	~		|
""")
    # as Python PySD: the positions in the main range, so equal elements are equal
    assert "other_position[other] = [3, 1, 2]" in code
    assert "is_b ~ ifelse.(dim_position .== 2, 1, 0)" in code


def test_numbers_and_operators_as_vensim(tmp_path):
    _, code = translate(tmp_path, """
x = -2^2
	~	Dmnl
	~		|

y = MODULO(Time - 5, 3)
	~	Dmnl
	~		|

z = IF THEN ELSE(Time > 5, :NA:, Time)
	~	Dmnl
	~		|
""")
    assert "x ~ -2 ^ 2" in code                  # -(2^2), as Vensim
    assert "y ~ rem(t - 5, 3)" in code           # the sign of the first argument
    assert "NaN" in code


def test_tabbed_array_in_a_part(tmp_path):
    _, code = translate(tmp_path, """
row: r1, r2
	~	~		|

col: c1, c2, c3
	~	~		|

layer: l1, l2
	~	~		|

x[row, col, l1] = TABBED ARRAY(
	1	2	3
	4	5	6)
	~	Dmnl
	~		|

x[row, col, l2] = 0
	~	Dmnl
	~		|
""")
    assert "[[1 2 3; 4 5 6], 0]" in code


def test_data_in_parts_of_several_shapes(tmp_path):
    _, code = translate(tmp_path, """
dim: d1, d2
	~	~		|

adim: a1, a2
	~	~		|

v[d1, a1] := GET DIRECT DATA('input.xlsx', 'Sheet1', 'time', 'B2')
	~	Dmnl
	~		|

v[d2, a1] = 2
	~	Dmnl
	~		|

v[dim, a2] := GET DIRECT DATA('input.xlsx', 'Sheet1', 'time', 'C2')
	~	Dmnl
	~		|

k[dim] = GET DIRECT CONSTANTS('input.xlsx', 'Sheet1', 'D5')
	~	Dmnl
	~		|
""")
    # one equation per part, each data a parameter of its own over the part's ranges
    assert "v_d1_a1 = data(" in code and "v_a2[dim] = data(" in code
    assert "v[dim = At(:d1), adim = At(:a1)] ~ v_d1_a1(t)" in code
    assert "v[adim = At(:a2)] ~ v_a2(t)" in code
    assert "v(t)[dim, adim]" in code
    # constants from a cell: the block of the variable's size
    assert '"D5:E5"' in code
