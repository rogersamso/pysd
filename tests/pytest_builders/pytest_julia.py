"""
Unit tests for the Julia/ModelingToolkit builder.

Tests are organised into classes that mirror the modules they exercise:

* ``TestJuliaNamespaceManager``  — namespace.py
* ``TestJuliaASTVisitor``         — julia_expressions_builder.py
* ``TestInlineLookupRegistry``    — julia_expressions_builder.py
* ``TestLookupHelpers``           — julia_expressions_builder.py
* ``TestJuliaSectionBuilder``     — julia_model_builder.py (element processing)
* ``TestJuliaModelBuilder``       — julia_model_builder.py (end-to-end)
* ``TestModularBuild``            — modular file generation
* ``TestTranslateToJulia``        — pysd.translate_to_julia entry point
"""
from pathlib import Path

import pytest

from pysd.builders.julia.namespace import JuliaNamespaceManager, JULIA_KEYWORDS
from pysd.builders.julia.julia_expressions_builder import (
    JuliaASTVisitor,
    InlineLookupRegistry,
    HELPER_IMPLEMENTATIONS,
    format_number,
    format_vector,
    lookup_interpolation_code,
)
from pysd.builders.julia.julia_model_builder import (
    JuliaModelBuilder,
    JuliaSectionBuilder,
    _path_to_eq_var,
)
from pysd.translators.structures.abstract_expressions import (
    ArithmeticStructure,
    CallStructure,
    GameStructure,
    InitialStructure,
    InlineLookupsStructure,
    IntegStructure,
    LogicStructure,
    LookupsStructure,
    ReferenceStructure,
    SmoothStructure,
    DelayStructure,
)
from pysd.translators.structures.abstract_model import (
    AbstractComponent,
    AbstractControlElement,
    AbstractElement,
    AbstractLookup,
    AbstractModel,
    AbstractSection,
    AbstractSubscriptRange,
    AbstractUnchangeableConstant,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_section(
    elements=None,
    subscripts=(),
    split=False,
    views_dict=None,
    path=None,
):
    """Return a minimal AbstractSection suitable for JuliaSectionBuilder."""
    if path is None:
        path = Path("test_model.mdl")
    return AbstractSection(
        name="__main__",
        path=path,
        type="main",
        params=[],
        returns=[],
        subscripts=tuple(subscripts),
        elements=tuple(elements or []),
        constraints=tuple(),
        test_inputs=tuple(),
        split=split,
        views_dict=views_dict,
    )


def _make_component(ast, comp_type="Auxiliary", subtype="Normal"):
    comp = AbstractComponent(subscripts=[[], []], ast=ast)
    comp.type = comp_type
    comp.subtype = subtype
    return comp


def _make_constant_component(value):
    comp = AbstractUnchangeableConstant(subscripts=[[], []], ast=value)
    return comp


def _make_element(name, ast, comp_class=None, units="", docs=""):
    if comp_class is AbstractUnchangeableConstant:
        comp = AbstractUnchangeableConstant(subscripts=[[], []], ast=ast)
    else:
        comp = AbstractComponent(subscripts=[[], []], ast=ast)
    return AbstractElement(name=name, components=[comp], units=units, documentation=docs)


def _make_lookup_element(name, xs, ys, itp_type="interpolate"):
    lut_ast = LookupsStructure(x=xs, y=ys, x_limits=(xs[0], xs[-1]),
                               y_limits=(ys[0], ys[-1]), type=itp_type)
    comp = AbstractLookup(subscripts=[[], []], ast=lut_ast)
    return AbstractElement(name=name, components=[comp])


def _make_stock_element(name, flow_ast, initial_ast):
    ast = IntegStructure(flow=flow_ast, initial=initial_ast)
    comp = AbstractComponent(subscripts=[[], []], ast=ast)
    return AbstractElement(name=name, components=[comp])


def _make_control_element(name, value):
    comp = AbstractUnchangeableConstant(subscripts=[[], []], ast=value)
    return AbstractControlElement(name=name, components=[comp])


def _section_builder_from_elements(elements, path=None, split=False, views_dict=None):
    section = _make_section(elements, path=path or Path("test_model.mdl"),
                            split=split, views_dict=views_dict)
    return JuliaSectionBuilder(section)


def _visitor_with_namespace(names=None):
    ns = JuliaNamespaceManager()
    for n in (names or []):
        ns.add_to_namespace(n)
    registry = InlineLookupRegistry()
    needed = set()
    return JuliaASTVisitor(ns, registry, needed), ns, registry, needed


# ===========================================================================
# JuliaNamespaceManager
# ===========================================================================

class TestJuliaNamespaceManager:

    def test_time_pre_registered(self):
        ns = JuliaNamespaceManager()
        assert ns.get("Time") == "t"

    def test_add_simple_name(self):
        ns = JuliaNamespaceManager()
        ident = ns.add_to_namespace("Population")
        assert ident == "population"
        assert ns.get("Population") == "population"

    def test_add_name_with_spaces(self):
        ns = JuliaNamespaceManager()
        ident = ns.add_to_namespace("Birth Rate")
        assert ident == "birth_rate"

    def test_add_name_with_special_chars(self):
        ns = JuliaNamespaceManager()
        ident = ns.add_to_namespace("var-n")
        assert ident == "var_n"

    def test_case_insensitive_lookup(self):
        ns = JuliaNamespaceManager()
        ns.add_to_namespace("Population")
        assert ns.get("population") == "population"
        assert ns.get("POPULATION") == "population"
        assert ns.get("PoPuLaTiOn") == "population"

    def test_idempotent_registration(self):
        ns = JuliaNamespaceManager()
        id1 = ns.add_to_namespace("Alpha")
        id2 = ns.add_to_namespace("Alpha")
        assert id1 == id2

    def test_keyword_avoidance(self):
        ns = JuliaNamespaceManager()
        for kw in ("end", "begin", "if", "for", "while", "module"):
            ident = ns.add_to_namespace(kw)
            assert ident not in JULIA_KEYWORDS, f"'{ident}' is a Julia keyword"

    def test_collision_resolution(self):
        ns = JuliaNamespaceManager()
        # Both "Birth Rate" and "birth rate" map to the same clean form
        id1 = ns.add_to_namespace("Birth Rate")
        id2 = ns.add_to_namespace("birth rate")
        assert id1 != id2
        assert id1 == "birth_rate"
        assert id2 == "birth_rate_1"

    def test_leading_digit(self):
        ns = JuliaNamespaceManager()
        ident = ns.add_to_namespace("1st var")
        assert ident[0].isalpha() or ident[0] == "_"

    def test_unknown_name_returns_none(self):
        ns = JuliaNamespaceManager()
        assert ns.get("nonexistent") is None

    @pytest.mark.parametrize("name,expected", [
        ("GDP", "gdp"),
        ("CO2 emissions", "co2_emissions"),
        ("Net__Flow", "net_flow"),
        ("x", "x"),
    ])
    def test_various_names(self, name, expected):
        ns = JuliaNamespaceManager()
        assert ns.add_to_namespace(name) == expected


# ===========================================================================
# format_number / format_vector helpers
# ===========================================================================

class TestFormatHelpers:

    @pytest.mark.parametrize("value,expected", [
        (1.0, "1.0"),
        (0.5, "0.5"),
        (float("inf"), "Inf"),
        (float("-inf"), "-Inf"),
        (float("nan"), "NaN"),
        (3, "3.0"),
    ])
    def test_format_number(self, value, expected):
        assert format_number(value) == expected

    def test_format_vector(self):
        result = format_vector((0.0, 50.0, 100.0))
        assert result == "[0.0, 50.0, 100.0]"


# ===========================================================================
# InlineLookupRegistry
# ===========================================================================

class TestInlineLookupRegistry:

    def test_register_returns_unique_names(self):
        reg = InlineLookupRegistry()
        n1 = reg.register((0.0, 1.0), (0.0, 1.0), "interpolate")
        n2 = reg.register((0.0, 2.0), (0.0, 4.0), "interpolate")
        assert n1 != n2

    def test_register_increments_counter(self):
        reg = InlineLookupRegistry()
        n1 = reg.register((0.0,), (0.0,), "interpolate")
        n2 = reg.register((0.0,), (0.0,), "interpolate")
        assert n1 == "_inline_lookup_1"
        assert n2 == "_inline_lookup_2"

    def test_entries_returns_all_registered(self):
        reg = InlineLookupRegistry()
        reg.register((0.0, 1.0), (0.0, 2.0), "interpolate")
        reg.register((0.0, 5.0), (0.0, 10.0), "interpolate")
        assert len(reg.entries) == 2


# ===========================================================================
# lookup_interpolation_code
# ===========================================================================

class TestLookupInterpolationCode:

    def test_basic_output(self):
        const_decl, func_decl = lookup_interpolation_code(
            "my_lut", (0.0, 1.0, 2.0), (0.0, 5.0, 10.0), "interpolate"
        )
        assert "LinearInterpolation" in const_decl
        assert "my_lut_itp" in const_decl
        # DataInterpolations: ys first, xs second
        assert "[0.0, 5.0, 10.0]" in const_decl   # ys
        assert "[0.0, 1.0, 2.0]" in const_decl    # xs
        assert func_decl == "my_lut(x) = my_lut_itp(x)"

    def test_const_keyword_present(self):
        const_decl, _ = lookup_interpolation_code("lut", (1.0,), (2.0,), "extrapolate")
        assert const_decl.startswith("const ")


# ===========================================================================
# JuliaASTVisitor
# ===========================================================================

class TestJuliaASTVisitor:

    # --- numeric literals ---------------------------------------------------

    def test_integer_literal(self):
        v, *_ = _visitor_with_namespace()
        assert v.visit(3) == "3.0"

    def test_float_literal(self):
        v, *_ = _visitor_with_namespace()
        assert v.visit(0.5) == "0.5"

    def test_inf_literal(self):
        v, *_ = _visitor_with_namespace()
        assert v.visit(float("inf")) == "Inf"

    def test_none_becomes_zero(self):
        v, *_ = _visitor_with_namespace()
        assert v.visit(None) == "0.0"

    # --- arithmetic ---------------------------------------------------------

    @pytest.mark.parametrize("ops,args,expected", [
        (["+"], [1.0, 2.0], "(1.0 + 2.0)"),
        (["-"], [5.0, 3.0], "(5.0 - 3.0)"),
        (["*"], [2.0, 4.0], "(2.0 * 4.0)"),
        (["/"], [6.0, 3.0], "(6.0 / 3.0)"),
        (["^"], [2.0, 8.0], "(2.0 ^ 8.0)"),
    ])
    def test_binary_arithmetic(self, ops, args, expected):
        v, *_ = _visitor_with_namespace()
        node = ArithmeticStructure(operators=ops, arguments=args)
        assert v.visit(node) == expected

    def test_unary_negation(self):
        v, *_ = _visitor_with_namespace()
        node = ArithmeticStructure(operators=["-"], arguments=[3.0])
        assert v.visit(node) == "(-3.0)"

    def test_chained_arithmetic(self):
        v, *_ = _visitor_with_namespace()
        node = ArithmeticStructure(operators=["+", "*"], arguments=[1.0, 2.0, 3.0])
        result = v.visit(node)
        assert "1.0" in result and "2.0" in result and "3.0" in result

    # --- logic --------------------------------------------------------------

    @pytest.mark.parametrize("vensim_op,julia_op", [
        ("=", "=="),
        ("<>", "!="),
        ("<", "<"),
        (">", ">"),
        ("<=", "<="),
        (">=", ">="),
        (":AND:", "&&"),
        (":OR:", "||"),
    ])
    def test_logic_operators(self, vensim_op, julia_op):
        v, *_ = _visitor_with_namespace()
        node = LogicStructure(operators=[vensim_op], arguments=[1.0, 0.0])
        assert julia_op in v.visit(node)

    def test_unary_not(self):
        v, *_ = _visitor_with_namespace()
        node = LogicStructure(operators=[":NOT:"], arguments=[1.0])
        result = v.visit(node)
        assert "!" in result

    # --- references ---------------------------------------------------------

    def test_known_reference(self):
        v, ns, *_ = _visitor_with_namespace(["Population"])
        node = ReferenceStructure(reference="Population")
        assert v.visit(node) == "population"

    def test_case_insensitive_reference(self):
        v, ns, *_ = _visitor_with_namespace(["Birth Rate"])
        node = ReferenceStructure(reference="birth rate")
        assert v.visit(node) == "birth_rate"

    def test_unknown_reference_warns(self):
        v, *_ = _visitor_with_namespace()
        node = ReferenceStructure(reference="Unknown Var")
        with pytest.warns(UserWarning, match="not found in namespace"):
            result = v.visit(node)
        assert isinstance(result, str)

    # --- built-in function calls --------------------------------------------

    @pytest.mark.parametrize("vensim_name,julia_name", [
        ("ABS", "abs"),
        ("EXP", "exp"),
        ("LN", "log"),
        ("SQRT", "sqrt"),
        ("SIN", "sin"),
        ("COS", "cos"),
        ("TAN", "tan"),
        ("ARCSIN", "asin"),
        ("ARCCOS", "acos"),
        ("ARCTAN", "atan"),
        ("MIN", "min"),
        ("MAX", "max"),
        ("MODULO", "mod"),
        ("INTEGER", "trunc"),
    ])
    def test_builtin_functions(self, vensim_name, julia_name):
        v, *_ = _visitor_with_namespace()
        node = CallStructure(
            function=ReferenceStructure(reference=vensim_name),
            arguments=(1.0,),
        )
        assert julia_name in v.visit(node)

    def test_if_then_else(self):
        v, *_ = _visitor_with_namespace()
        node = CallStructure(
            function=ReferenceStructure(reference="IF THEN ELSE"),
            arguments=(1.0, 2.0, 3.0),
        )
        assert "ifelse" in v.visit(node)

    def test_unknown_function_warns(self):
        v, *_ = _visitor_with_namespace()
        node = CallStructure(
            function=ReferenceStructure(reference="SOME_UNKNOWN_FUNC"),
            arguments=(1.0,),
        )
        with pytest.warns(UserWarning, match="Unknown Vensim function"):
            result = v.visit(node)
        assert "some_unknown_func" in result

    # --- helper functions registered in needed_helpers ----------------------

    @pytest.mark.parametrize("func_name", ["XIDZ", "ZIDZ", "PULSE", "RAMP", "STEP"])
    def test_helper_functions_registered(self, func_name):
        v, _, _, needed = _visitor_with_namespace()
        node = CallStructure(
            function=ReferenceStructure(reference=func_name),
            arguments=(1.0, 2.0, 3.0),
        )
        v.visit(node)
        helper_name = f"_{func_name.lower()}"
        assert helper_name in needed

    def test_pulse_prepends_t(self):
        v, *_ = _visitor_with_namespace()
        node = CallStructure(
            function=ReferenceStructure(reference="PULSE"),
            arguments=(10.0, 1.0),
        )
        result = v.visit(node)
        assert result.startswith("_pulse(t,")

    def test_ramp_prepends_t(self):
        v, *_ = _visitor_with_namespace()
        node = CallStructure(
            function=ReferenceStructure(reference="RAMP"),
            arguments=(0.1, 5.0),
        )
        result = v.visit(node)
        assert result.startswith("_ramp(t,")

    # --- InitialStructure / GameStructure -----------------------------------

    def test_initial_structure_returns_inner(self):
        v, *_ = _visitor_with_namespace()
        node = InitialStructure(initial=42.0)
        assert v.visit(node) == "42.0"

    def test_game_structure_returns_inner(self):
        v, *_ = _visitor_with_namespace()
        node = GameStructure(expression=7.0)
        assert v.visit(node) == "7.0"

    # --- inline lookups -----------------------------------------------------

    def test_inline_lookup_registers(self):
        v, _, registry, _ = _visitor_with_namespace(["x_var"])
        lut = LookupsStructure(
            x=(0.0, 1.0, 2.0),
            y=(0.0, 5.0, 10.0),
            x_limits=(0.0, 2.0),
            y_limits=(0.0, 10.0),
            type="interpolate",
        )
        node = InlineLookupsStructure(
            argument=ReferenceStructure(reference="x_var"),
            lookups=lut,
        )
        result = v.visit(node)
        assert len(registry.entries) == 1
        assert "_inline_lookup_1" in result

    def test_inline_lookup_call_includes_arg(self):
        v, ns, registry, _ = _visitor_with_namespace(["input"])
        lut = LookupsStructure(
            x=(0.0, 1.0),
            y=(0.0, 2.0),
            x_limits=(0.0, 1.0),
            y_limits=(0.0, 2.0),
            type="interpolate",
        )
        node = InlineLookupsStructure(
            argument=ReferenceStructure(reference="input"),
            lookups=lut,
        )
        result = v.visit(node)
        assert "input" in result


# ===========================================================================
# JuliaSectionBuilder — element processing
# ===========================================================================

class TestJuliaSectionBuilderElements:

    # --- stocks -----------------------------------------------------------

    def test_stock_creates_ode_equation(self):
        flow = ArithmeticStructure(operators=["-"], arguments=[
            ReferenceStructure("Births"), ReferenceStructure("Deaths")
        ])
        pop_elem = _make_stock_element("Population", flow, 1000.0)
        # Register the referenced variables so the visitor can resolve them
        births_elem = _make_element("Births", 10.0)
        deaths_elem = _make_element("Deaths", 5.0)
        sb = _section_builder_from_elements([pop_elem, births_elem, deaths_elem])
        sb.build_section()

        assert any("@variables population(t)" in d for d in sb.stock_decls)
        assert any("D(population)" in e for e in sb.built_elements["population"][0])
        assert any("population =>" in u for u in sb.u0_entries)

    def test_stock_initial_value_in_u0(self):
        elem = _make_stock_element("Capital", 5.0, 100.0)
        sb = _section_builder_from_elements([elem])
        sb.build_section()
        assert any("capital => 100.0" in u for u in sb.u0_entries)

    # --- constants / parameters -------------------------------------------

    def test_constant_creates_parameter(self):
        comp = AbstractUnchangeableConstant(subscripts=[[], []], ast=0.03)
        elem = AbstractElement(name="Birth Rate", components=[comp])
        sb = _section_builder_from_elements([elem])
        sb.build_section()
        assert any("@parameters birth_rate = 0.03" in d for d in sb.param_decls)

    def test_auxiliary_creates_variable_and_equation(self):
        rhs = ArithmeticStructure(operators=["*"], arguments=[
            ReferenceStructure("population"), ReferenceStructure("birth_rate_param")
        ])
        # Register the references so namespace resolves them
        elem_pop = _make_stock_element("Population", 0.0, 100.0)
        comp_br = AbstractUnchangeableConstant(subscripts=[[], []], ast=0.03)
        elem_br = AbstractElement(name="birth rate param", components=[comp_br])
        comp_aux = AbstractComponent(subscripts=[[], []], ast=rhs)
        elem_births = AbstractElement(name="Births", components=[comp_aux])

        sb = _section_builder_from_elements([elem_pop, elem_br, elem_births])
        sb.build_section()

        assert any("@variables births(t)" in d for d in sb.aux_decls)
        assert any("births ~" in e for eqs, _ in sb.built_elements.values() for e in eqs)

    # --- lookup tables ----------------------------------------------------

    def test_named_lookup_registers_interpolant(self):
        elem = _make_lookup_element(
            "Effect Table", (0.0, 0.5, 1.0), (0.0, 0.8, 1.0)
        )
        sb = _section_builder_from_elements([elem])
        sb.build_section()

        assert any("effect_table_itp" in d for d in sb.lookup_const_decls)
        assert any("effect_table(x)" in f for f in sb.lookup_func_decls)

    def test_named_lookup_no_equation_generated(self):
        elem = _make_lookup_element("LUT", (0.0, 1.0), (0.0, 2.0))
        sb = _section_builder_from_elements([elem])
        sb.build_section()
        # lookup elements produce no ODE/algebraic equations
        assert sb.built_elements["lut"][0] == []

    # --- control variables ------------------------------------------------

    def test_control_vars_stored_not_emitted_as_params(self):
        elems = [
            _make_control_element("INITIAL TIME", 0.0),
            _make_control_element("FINAL TIME", 100.0),
            _make_control_element("TIME STEP", 0.25),
            _make_control_element("SAVEPER", 0.25),
        ]
        sb = _section_builder_from_elements(elems)
        sb.build_section()

        # Control vars should NOT appear as @parameters
        assert not any("initial_time" in d for d in sb.param_decls)
        assert sb.control_vals["initial_time"] == "0.0"
        assert sb.control_vals["final_time"] == "100.0"
        assert sb.control_vals["time_step"] == "0.25"

    # --- smooth expansion -------------------------------------------------

    def test_smooth1_expands_to_ode_and_aux(self):
        flow_ast = SmoothStructure(
            input=5.0, smooth_time=3.0, initial=5.0, order=1
        )
        comp = AbstractComponent(subscripts=[[], []], ast=flow_ast)
        elem = AbstractElement(name="Smooth Output", components=[comp])
        sb = _section_builder_from_elements([elem])
        sb.build_section()

        all_eqs = [e for eqs, _ in sb.built_elements.values() for e in eqs]
        assert any("D(_lv1_smooth_output)" in e for e in all_eqs)
        assert any("smooth_output ~" in e for e in all_eqs)
        assert any("_lv1_smooth_output(t)" in d for d in sb.stock_decls)

    def test_smooth3_produces_three_levels(self):
        flow_ast = SmoothStructure(
            input=5.0, smooth_time=3.0, initial=5.0, order=3
        )
        comp = AbstractComponent(subscripts=[[], []], ast=flow_ast)
        elem = AbstractElement(name="Smooth3", components=[comp])
        sb = _section_builder_from_elements([elem])
        sb.build_section()

        assert sum(1 for d in sb.stock_decls if "_lv" in d and "smooth3" in d) == 3

    # --- delay expansion --------------------------------------------------

    def test_delay1_expands_to_ode_and_aux(self):
        delay_ast = DelayStructure(
            input=10.0, delay_time=2.0, initial=10.0, order=1
        )
        comp = AbstractComponent(subscripts=[[], []], ast=delay_ast)
        elem = AbstractElement(name="Delayed Value", components=[comp])
        sb = _section_builder_from_elements([elem])
        sb.build_section()

        all_eqs = [e for eqs, _ in sb.built_elements.values() for e in eqs]
        assert any("D(_dl1_delayed_value)" in e for e in all_eqs)
        assert any("delayed_value ~" in e for e in all_eqs)

    def test_delay3_produces_three_levels(self):
        delay_ast = DelayStructure(
            input=5.0, delay_time=6.0, initial=5.0, order=3
        )
        comp = AbstractComponent(subscripts=[[], []], ast=delay_ast)
        elem = AbstractElement(name="Delay3", components=[comp])
        sb = _section_builder_from_elements([elem])
        sb.build_section()

        assert sum(1 for d in sb.stock_decls if "_dl" in d and "delay3" in d) == 3


# ===========================================================================
# JuliaModelBuilder — end-to-end file generation
# ===========================================================================

class TestJuliaModelBuilder:

    def _minimal_model(self, tmp_path):
        """Build an AbstractModel with one stock and one parameter."""
        birth_rate_comp = AbstractUnchangeableConstant(subscripts=[[], []], ast=0.03)
        birth_rate_elem = AbstractElement(name="Birth Rate", components=[birth_rate_comp])

        flow_ast = ArithmeticStructure(
            operators=["*"],
            arguments=[ReferenceStructure("Population"), ReferenceStructure("Birth Rate")],
        )
        pop_ast = IntegStructure(flow=flow_ast, initial=1000.0)
        pop_comp = AbstractComponent(subscripts=[[], []], ast=pop_ast)
        pop_elem = AbstractElement(name="Population", components=[pop_comp])

        control_elems = [
            _make_control_element("INITIAL TIME", 0.0),
            _make_control_element("FINAL TIME", 100.0),
            _make_control_element("TIME STEP", 1.0),
            _make_control_element("SAVEPER", 1.0),
        ]

        section = _make_section(
            elements=[birth_rate_elem, pop_elem] + control_elems,
            path=tmp_path / "my_model.mdl",
        )
        return AbstractModel(
            original_path=tmp_path / "my_model.mdl",
            sections=(section,),
        )

    def test_build_model_returns_jl_path(self, tmp_path):
        model = self._minimal_model(tmp_path)
        path = JuliaModelBuilder(model).build_model()
        assert path.suffix == ".jl"
        assert path.exists()

    def test_output_contains_using_mtk(self, tmp_path):
        model = self._minimal_model(tmp_path)
        path = JuliaModelBuilder(model).build_model()
        content = path.read_text()
        assert "using ModelingToolkit" in content

    def test_output_contains_stock_declaration(self, tmp_path):
        model = self._minimal_model(tmp_path)
        path = JuliaModelBuilder(model).build_model()
        content = path.read_text()
        assert "@variables population(t)" in content

    def test_output_contains_parameter_declaration(self, tmp_path):
        model = self._minimal_model(tmp_path)
        path = JuliaModelBuilder(model).build_model()
        content = path.read_text()
        assert "@parameters birth_rate = 0.03" in content

    def test_output_contains_ode_equation(self, tmp_path):
        model = self._minimal_model(tmp_path)
        path = JuliaModelBuilder(model).build_model()
        content = path.read_text()
        assert "D(population)" in content

    def test_output_contains_u0(self, tmp_path):
        model = self._minimal_model(tmp_path)
        path = JuliaModelBuilder(model).build_model()
        content = path.read_text()
        assert "population => 1000.0" in content

    def test_output_contains_ode_system(self, tmp_path):
        model = self._minimal_model(tmp_path)
        path = JuliaModelBuilder(model).build_model()
        content = path.read_text()
        assert "ODESystem" in content
        assert "structural_simplify" in content

    def test_output_contains_run_model_function(self, tmp_path):
        model = self._minimal_model(tmp_path)
        path = JuliaModelBuilder(model).build_model()
        content = path.read_text()
        assert "function run_model(" in content

    def test_control_vars_emitted(self, tmp_path):
        model = self._minimal_model(tmp_path)
        path = JuliaModelBuilder(model).build_model()
        content = path.read_text()
        assert "initial_time = 0.0" in content
        assert "final_time   = 100.0" in content
        assert "time_step    = 1.0" in content

    def test_lookup_table_emitted(self, tmp_path):
        lut_ast = LookupsStructure(
            x=(0.0, 1.0, 2.0),
            y=(0.0, 0.5, 1.0),
            x_limits=(0.0, 2.0),
            y_limits=(0.0, 1.0),
            type="interpolate",
        )
        lut_comp = AbstractLookup(subscripts=[[], []], ast=lut_ast)
        lut_elem = AbstractElement(name="Effect LUT", components=[lut_comp])
        control_elems = [
            _make_control_element("INITIAL TIME", 0.0),
            _make_control_element("FINAL TIME", 10.0),
            _make_control_element("TIME STEP", 0.1),
            _make_control_element("SAVEPER", 0.1),
        ]
        section = _make_section(
            elements=[lut_elem] + control_elems,
            path=tmp_path / "lut_model.mdl",
        )
        model = AbstractModel(
            original_path=tmp_path / "lut_model.mdl",
            sections=(section,),
        )
        path = JuliaModelBuilder(model).build_model()
        content = path.read_text()
        assert "LinearInterpolation" in content
        assert "effect_lut_itp" in content
        assert "DataInterpolations" in content

    def test_helper_functions_emitted(self, tmp_path):
        pulse_ast = CallStructure(
            function=ReferenceStructure(reference="PULSE"),
            arguments=(10.0, 2.0),
        )
        comp = AbstractComponent(subscripts=[[], []], ast=pulse_ast)
        elem = AbstractElement(name="Pulse Signal", components=[comp])
        control_elems = [
            _make_control_element("INITIAL TIME", 0.0),
            _make_control_element("FINAL TIME", 20.0),
            _make_control_element("TIME STEP", 0.1),
            _make_control_element("SAVEPER", 0.1),
        ]
        section = _make_section(
            elements=[elem] + control_elems,
            path=tmp_path / "pulse_model.mdl",
        )
        model = AbstractModel(
            original_path=tmp_path / "pulse_model.mdl",
            sections=(section,),
        )
        path = JuliaModelBuilder(model).build_model()
        content = path.read_text()
        assert "_pulse(" in content


# ===========================================================================
# Modular build
# ===========================================================================

class TestModularBuild:

    def _two_view_model(self, tmp_path):
        """Model with two views: 'Sector A' (population) and 'Sector B' (capital)."""
        pop_elem = _make_stock_element("Population", 1.0, 100.0)
        cap_elem = _make_stock_element("Capital", 2.0, 500.0)
        control_elems = [
            _make_control_element("INITIAL TIME", 0.0),
            _make_control_element("FINAL TIME", 50.0),
            _make_control_element("TIME STEP", 0.5),
            _make_control_element("SAVEPER", 0.5),
        ]
        views_dict = {
            "Sector A": {"Population"},
            "Sector B": {"Capital"},
        }
        section = _make_section(
            elements=[pop_elem, cap_elem] + control_elems,
            path=tmp_path / "split_model.mdl",
            split=True,
            views_dict=views_dict,
        )
        return AbstractModel(
            original_path=tmp_path / "split_model.mdl",
            sections=(section,),
        )

    def test_main_file_created(self, tmp_path):
        model = self._two_view_model(tmp_path)
        path = JuliaModelBuilder(model).build_model()
        assert path.exists()

    def test_module_files_created(self, tmp_path):
        model = self._two_view_model(tmp_path)
        JuliaModelBuilder(model).build_model()
        modules_dir = tmp_path / "modules_split_model"
        assert modules_dir.exists()
        jl_files = list(modules_dir.glob("*.jl"))
        assert len(jl_files) == 2

    def test_main_file_has_include_statements(self, tmp_path):
        model = self._two_view_model(tmp_path)
        path = JuliaModelBuilder(model).build_model()
        content = path.read_text()
        assert "include(" in content

    def test_main_file_concatenates_eq_vectors(self, tmp_path):
        model = self._two_view_model(tmp_path)
        path = JuliaModelBuilder(model).build_model()
        content = path.read_text()
        # The main file should reference the module equation vectors
        assert "eqs = [" in content

    def test_module_files_contain_eq_var(self, tmp_path):
        model = self._two_view_model(tmp_path)
        JuliaModelBuilder(model).build_model()
        modules_dir = tmp_path / "modules_split_model"
        for jl_file in modules_dir.glob("*.jl"):
            content = jl_file.read_text()
            assert "_eqs = Equation[" in content

    def test_all_declarations_in_main_file(self, tmp_path):
        """Variable declarations must be in main file so modules can reference them."""
        model = self._two_view_model(tmp_path)
        path = JuliaModelBuilder(model).build_model()
        content = path.read_text()
        assert "@variables population(t)" in content
        assert "@variables capital(t)" in content


# ===========================================================================
# _path_to_eq_var
# ===========================================================================

class TestPathToEqVar:

    @pytest.mark.parametrize("parts,expected", [
        (["modules_model", "Sector A"], "sector_a_eqs"),
        (["modules_model", "Sector A", "Sub1"], "sector_a_sub1_eqs"),
        (["modules_model", "Demographics"], "demographics_eqs"),
        (["modules_model", "sector-b"], "sector_b_eqs"),
    ])
    def test_conversion(self, parts, expected):
        path = Path(*parts)
        assert _path_to_eq_var(path) == expected


# ===========================================================================
# translate_to_julia entry point (integration, Vensim .mdl)
# ===========================================================================

class TestTranslateToJulia:

    def test_unsupported_format_raises(self, tmp_path):
        fake = tmp_path / "model.xyz"
        fake.write_text("dummy")
        from pysd import translate_to_julia
        with pytest.raises(ValueError, match="Unsupported model format"):
            translate_to_julia(fake)

    def test_vensim_model_produces_jl_file(self, tmp_path):
        """End-to-end smoke test with the split_model fixture."""
        import shutil
        src = Path("tests/more-tests/split_model/test_split_model.mdl")
        if not src.exists():
            pytest.skip("test-models submodule not checked out")

        dst = tmp_path / "test_split_model.mdl"
        shutil.copy(src, dst)

        from pysd import translate_to_julia
        # The model uses GET DIRECT CONSTANTS which emits an expected warning
        with pytest.warns(UserWarning):
            path = translate_to_julia(dst)
        assert path.exists()
        assert path.suffix == ".jl"
        content = path.read_text()
        assert "using ModelingToolkit" in content
        assert "ODESystem" in content
        assert "run_model" in content
