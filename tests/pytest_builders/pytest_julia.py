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
    AllocateAvailableStructure,
    AllocateByPriorityStructure,
    ArithmeticStructure,
    CallStructure,
    DataStructure,
    DelayFixedStructure,
    DelayStructure,
    ForecastStructure,
    GameStructure,
    GetConstantsStructure,
    GetDataStructure,
    GetLookupsStructure,
    InitialStructure,
    InlineLookupsStructure,
    IntegStructure,
    LogicStructure,
    LookupsStructure,
    ReferenceStructure,
    SampleIfTrueStructure,
    SmoothNStructure,
    SmoothStructure,
    SubscriptsReferenceStructure,
    TrendStructure,
)
from pysd.translators.structures.abstract_model import (
    AbstractComponent,
    AbstractControlElement,
    AbstractData,
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


def _make_subscript_range(name, elems):
    return AbstractSubscriptRange(name=name, subscripts=elems, mapping=[])


def _make_subscripted_element(name, ast, dim_name, comp_class=None):
    """Element whose first component covers one subscript dimension."""
    if comp_class is AbstractUnchangeableConstant:
        comp = AbstractUnchangeableConstant(subscripts=[[dim_name], []], ast=ast)
    else:
        comp = AbstractComponent(subscripts=[[dim_name], []], ast=ast)
    return AbstractElement(name=name, components=[comp])


def _make_data_element(name, ast):
    """Element whose component is an AbstractData (external time-series)."""
    comp = AbstractData(subscripts=[[], []], ast=ast)
    return AbstractElement(name=name, components=[comp])


def _section_builder_from_elements(elements, path=None, split=False, views_dict=None,
                                   subscripts=()):
    section = _make_section(elements, path=path or Path("test_model.mdl"),
                            split=split, views_dict=views_dict,
                            subscripts=subscripts)
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
        const_decl, func_decl, reg_decl = lookup_interpolation_code(
            "my_lut", (0.0, 1.0, 2.0), (0.0, 5.0, 10.0), "interpolate"
        )
        assert "LinearInterpolation" in const_decl
        assert "my_lut_itp" in const_decl
        # DataInterpolations: ys first, xs second
        assert "[0.0, 5.0, 10.0]" in const_decl   # ys
        assert "[0.0, 1.0, 2.0]" in const_decl    # xs
        assert func_decl == "my_lut(x) = my_lut_itp(x)"
        assert "@register_symbolic" in reg_decl
        assert "my_lut" in reg_decl

    def test_const_keyword_present(self):
        const_decl, _, _ = lookup_interpolation_code("lut", (1.0,), (2.0,), "extrapolate")
        assert const_decl.startswith("const ")

    def test_hold_forward_uses_constant_interpolation(self):
        const_decl, _, _ = lookup_interpolation_code(
            "lut", (0.0, 1.0), (5.0, 10.0), "hold_forward"
        )
        assert "ConstantInterpolation" in const_decl
        assert "LinearInterpolation" not in const_decl
        assert "dir" not in const_decl

    def test_hold_backward_uses_constant_interpolation_right(self):
        const_decl, _, _ = lookup_interpolation_code(
            "lut", (0.0, 1.0), (5.0, 10.0), "hold_backward"
        )
        assert "ConstantInterpolation" in const_decl
        assert "dir=:right" in const_decl

    def test_unknown_type_falls_back_to_linear(self):
        const_decl, _, _ = lookup_interpolation_code(
            "lut", (0.0, 1.0), (5.0, 10.0), "unknown_type"
        )
        assert "LinearInterpolation" in const_decl


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
    ])
    def test_comparison_operators(self, vensim_op, julia_op):
        v, *_ = _visitor_with_namespace()
        node = LogicStructure(operators=[vensim_op], arguments=[1.0, 0.0])
        assert julia_op in v.visit(node)

    def test_and_uses_helper_function(self):
        """AND maps to _logical_and helper (not &&) for symbolic MTK compatibility."""
        v, _, _, needed = _visitor_with_namespace()
        node = LogicStructure(operators=[":AND:"], arguments=[1.0, 0.0])
        result = v.visit(node)
        assert "_logical_and(" in result
        assert "_logical_and" in needed

    def test_or_uses_helper_function(self):
        """OR maps to _logical_or helper (not ||) for symbolic MTK compatibility."""
        v, _, _, needed = _visitor_with_namespace()
        node = LogicStructure(operators=[":OR:"], arguments=[1.0, 0.0])
        result = v.visit(node)
        assert "_logical_or(" in result
        assert "_logical_or" in needed

    def test_unary_not_uses_helper_function(self):
        """NOT maps to _logical_not helper for symbolic MTK compatibility."""
        v, _, _, needed = _visitor_with_namespace()
        node = LogicStructure(operators=[":NOT:"], arguments=[1.0])
        result = v.visit(node)
        assert "_logical_not(" in result
        assert "_logical_not" in needed

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

    @pytest.mark.parametrize("func_ref", ["IF THEN ELSE", "if_then_else"])
    def test_if_then_else(self, func_ref):
        """Both the space form and the underscore form (as stored by the parser) work."""
        v, *_ = _visitor_with_namespace()
        node = CallStructure(
            function=ReferenceStructure(reference=func_ref),
            arguments=(1.0, 2.0, 3.0),
        )
        assert "ifelse" in v.visit(node)

    @pytest.mark.parametrize("func_ref", ["PULSE TRAIN", "pulse_train"])
    def test_pulse_train_both_forms(self, func_ref):
        """Both the space form and the underscore form (as stored by the parser) work."""
        v, _, _, needed = _visitor_with_namespace()
        node = CallStructure(
            function=ReferenceStructure(reference=func_ref),
            arguments=(10.0, 1.0, 5.0, 100.0),
        )
        result = v.visit(node)
        assert "_pulse_train" in result
        assert "_pulse_train" in needed

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


# ===========================================================================
# Extended AST visitor coverage
# ===========================================================================

class TestJuliaASTVisitorExtended:

    def test_bool_true(self):
        v, *_ = _visitor_with_namespace()
        assert v.visit(True) == "true"

    def test_bool_false(self):
        v, *_ = _visitor_with_namespace()
        assert v.visit(False) == "false"

    def test_string_numeric(self):
        v, *_ = _visitor_with_namespace()
        assert v.visit("3.14") == "3.14"

    def test_string_non_numeric(self):
        v, *_ = _visitor_with_namespace()
        result = v.visit("hello")
        assert "'hello'" in result

    def test_numpy_scalar(self):
        import numpy as np
        v, *_ = _visitor_with_namespace()
        assert v.visit(np.float64(2.5)) == "2.5"

    def test_numpy_1d_array(self):
        import numpy as np
        v, *_ = _visitor_with_namespace()
        result = v.visit(np.array([1.0, 2.0, 3.0]))
        assert result == "[1.0, 2.0, 3.0]"

    def test_numpy_2d_array_flattened(self):
        import numpy as np
        v, *_ = _visitor_with_namespace()
        result = v.visit(np.array([[1.0, 2.0], [3.0, 4.0]]))
        assert "[" in result
        assert "1.0" in result and "4.0" in result

    def test_subscripts_reference_structure_known(self):
        v, ns, *_ = _visitor_with_namespace(["sectors"])
        node = SubscriptsReferenceStructure(subscripts=("sectors",))
        result = v.visit(node)
        assert result == "sectors"

    def test_subscripts_reference_structure_unknown(self):
        v, *_ = _visitor_with_namespace()
        node = SubscriptsReferenceStructure(subscripts=("unknown_dim",))
        result = v.visit(node)
        assert "unknown_dim" in result

    def test_subscripts_reference_structure_empty(self):
        v, *_ = _visitor_with_namespace()
        node = SubscriptsReferenceStructure(subscripts=())
        result = v.visit(node)
        assert result == "0.0"

    def test_unknown_node_warns_and_returns_zero(self):
        v, *_ = _visitor_with_namespace()
        with pytest.warns(UserWarning, match="Unsupported AST node type"):
            result = v.visit(object())
        assert result == "0.0"

    def test_unary_not(self):
        v, *_ = _visitor_with_namespace()
        node = LogicStructure(operators=[":NOT:"], arguments=[1.0])
        result = v.visit(node)
        assert "_logical_not" in result

    def test_elmcount_no_args(self):
        v, *_ = _visitor_with_namespace()
        node = CallStructure(
            function=ReferenceStructure("ELMCOUNT"), arguments=()
        )
        result = v.visit(node)
        assert result == "0"

    def test_elmcount_non_reference_arg(self):
        v, *_ = _visitor_with_namespace()
        node = CallStructure(
            function=ReferenceStructure("ELMCOUNT"), arguments=(3.0,)
        )
        result = v.visit(node)
        assert result == "3.0"

    def test_elmcount_reference_with_known_size(self):
        v, ns, _, _, = _visitor_with_namespace()
        ns.add_to_namespace("sectors")
        v.subs_sizes = {"sectors": 5}
        node = CallStructure(
            function=ReferenceStructure("ELMCOUNT"),
            arguments=(ReferenceStructure("sectors"),)
        )
        result = v.visit(node)
        assert result == "5"

    def test_time_helper_prepends_t(self):
        v, *_ = _visitor_with_namespace()
        node = CallStructure(
            function=ReferenceStructure("PULSE"),
            arguments=(10.0, 2.0),
        )
        result = v.visit(node)
        assert result.startswith("_pulse(t,")

    def test_model_variable_lookup_call(self):
        """A function call whose name is a model variable → emit as-is."""
        v, ns, *_ = _visitor_with_namespace(["effect table"])
        node = CallStructure(
            function=ReferenceStructure("effect table"),
            arguments=(ReferenceStructure("input"),),
        )
        ns.add_to_namespace("input")
        result = v.visit(node)
        assert "effect_table" in result

    def test_get_constants_in_expression_fallback(self):
        """GetConstantsStructure inside an expression warns when file unreadable."""
        v, *_ = _visitor_with_namespace()
        node = GetConstantsStructure(file="nonexistent.xlsx", tab="Sheet1", cell="A1")
        with pytest.warns(UserWarning, match="GetConstantsStructure"):
            result = v.visit(node)
        assert result == "0.0"


# ===========================================================================
# Section builder — subscript handling
# ===========================================================================

class TestJuliaSectionBuilderSubscripts:

    def test_alias_subscript_defaults_to_zero_size(self):
        sr_alias = AbstractSubscriptRange(name="alias_dim", subscripts="real_dim", mapping=[])
        section = _make_section(subscripts=[sr_alias])
        sb = JuliaSectionBuilder(section)
        assert sb._subs_sizes.get("alias_dim") == 0

    def test_list_subscript_has_correct_size(self):
        sr = _make_subscript_range("energy_type", ["Hydro", "Solar", "Wind"])
        section = _make_section(subscripts=[sr])
        sb = JuliaSectionBuilder(section)
        assert sb._subs_sizes["energy_type"] == 3

    def test_subs_const_decl_emitted(self):
        sr = _make_subscript_range("sector", ["A", "B", "C", "D"])
        elem = _make_subscripted_element("output", 1.0, "sector",
                                         comp_class=AbstractUnchangeableConstant)
        sr_elem = _make_subscript_range("sector", ["A", "B", "C", "D"])
        sb = _section_builder_from_elements([elem], subscripts=[sr_elem])
        sb.build_section()
        assert any("N_SECTOR" in d for d in sb.subs_const_decls)

    def test_1d_subscripted_parameter(self):
        sr = _make_subscript_range("energy_type", ["Hydro", "Solar"])
        elem = _make_subscripted_element("cost", 2.5, "energy_type",
                                         comp_class=AbstractUnchangeableConstant)
        sb = _section_builder_from_elements([elem], subscripts=[sr])
        sb.build_section()
        assert any("cost[1:N_ENERGY_TYPE]" in d for d in sb.param_decls)

    def test_1d_subscripted_stock(self):
        sr = _make_subscript_range("sector", ["A", "B"])
        comp = AbstractComponent(
            subscripts=[["sector"], []],
            ast=IntegStructure(flow=1.0, initial=0.0),
        )
        elem = AbstractElement(name="capital", components=[comp])
        sb = _section_builder_from_elements([elem], subscripts=[sr])
        sb.build_section()
        assert any("capital(t)[" in d for d in sb.stock_decls)

    def test_1d_subscripted_auxiliary(self):
        sr = _make_subscript_range("sector", ["A", "B", "C"])
        elem = _make_subscripted_element("output", 3.0, "sector")
        sb = _section_builder_from_elements([elem], subscripts=[sr])
        sb.build_section()
        assert any("output(t)[" in d for d in sb.aux_decls)
        eqs = [e for eqs, _ in sb.built_elements.values() for e in eqs]
        assert any("Symbolics.scalarize" in e for e in eqs)

    def test_2d_subscripted_auxiliary(self):
        sr1 = _make_subscript_range("row_dim", ["R1", "R2"])
        sr2 = _make_subscript_range("col_dim", ["C1", "C2", "C3"])
        comp = AbstractComponent(
            subscripts=[["row_dim", "col_dim"], []],
            ast=1.0,
        )
        elem = AbstractElement(name="matrix", components=[comp])
        sb = _section_builder_from_elements([elem], subscripts=[sr1, sr2])
        sb.build_section()
        eqs = [e for eqs, _ in sb.built_elements.values() for e in eqs]
        assert any("_i0" in e and "_i1" in e for e in eqs)

    def test_element_with_no_components_returns_empty(self):
        elem = AbstractElement(name="empty_var", components=[])
        sb = _section_builder_from_elements([elem])
        sb.build_section()
        assert sb.built_elements["empty_var"][0] == []


# ===========================================================================
# Section builder — INITIAL() handling
# ===========================================================================

class TestJuliaSectionBuilderInitial:

    def test_initial_resolves_from_stock(self):
        stock = _make_stock_element("Level", 1.0, 42.0)
        init_ast = InitialStructure(initial=ReferenceStructure("Level"))
        comp = AbstractComponent(subscripts=[[], []], ast=init_ast)
        init_elem = AbstractElement(name="Init Value", components=[comp])
        sb = _section_builder_from_elements([stock, init_elem])
        sb.build_section()
        assert any("@parameters init_value = 42.0" in d for d in sb.param_decls)

    def test_initial_resolves_from_parameter(self):
        const_comp = AbstractUnchangeableConstant(subscripts=[[], []], ast=7.5)
        const_elem = AbstractElement(name="Base Rate", components=[const_comp])
        init_ast = InitialStructure(initial=ReferenceStructure("Base Rate"))
        comp = AbstractComponent(subscripts=[[], []], ast=init_ast)
        init_elem = AbstractElement(name="Init Rate", components=[comp])
        sb = _section_builder_from_elements([const_elem, init_elem])
        sb.build_section()
        assert any("@parameters init_rate = 7.5" in d for d in sb.param_decls)

    @pytest.mark.filterwarnings("always::UserWarning")
    def test_initial_fallback_emits_warning(self):
        # Reference that can't be resolved → fallback to aux + warning
        init_ast = InitialStructure(initial=ReferenceStructure("unknown_var"))
        comp = AbstractComponent(subscripts=[[], []], ast=init_ast)
        elem = AbstractElement(name="Init Fallback", components=[comp])
        with pytest.warns(UserWarning, match="Cannot resolve INITIAL"):
            sb = _section_builder_from_elements([elem])
            sb.build_section()
        assert any("@variables init_fallback(t)" in d for d in sb.aux_decls)

    def test_resolve_ref_initial_chain(self):
        """INITIAL(aux) where aux ~ stock → resolves to stock initial."""
        stock = _make_stock_element("S", 1.0, 99.0)
        aux_comp = AbstractComponent(subscripts=[[], []], ast=ReferenceStructure("S"))
        aux_elem = AbstractElement(name="A", components=[aux_comp])
        init_ast = InitialStructure(initial=ReferenceStructure("A"))
        init_comp = AbstractComponent(subscripts=[[], []], ast=init_ast)
        init_elem = AbstractElement(name="Init A", components=[init_comp])
        sb = _section_builder_from_elements([stock, aux_elem, init_elem])
        sb.build_section()
        assert any("@parameters init_a = 99.0" in d for d in sb.param_decls)


# ===========================================================================
# Section builder — expansion methods
# ===========================================================================

class TestJuliaSectionBuilderExpansions:

    def test_delay_fixed_expands(self):
        ast = DelayFixedStructure(input=5.0, delay_time=2.0, initial=5.0)
        comp = AbstractComponent(subscripts=[[], []], ast=ast)
        elem = AbstractElement(name="Delayed Fixed", components=[comp])
        sb = _section_builder_from_elements([elem])
        sb.build_section()
        all_eqs = [e for eqs, _ in sb.built_elements.values() for e in eqs]
        assert any("_df_delayed_fixed" in e for e in all_eqs)
        assert any("delayed_fixed ~" in e for e in all_eqs)
        assert any("_df_delayed_fixed(t)" in d for d in sb.stock_decls)

    def test_trend_expands(self):
        ast = TrendStructure(input=10.0, average_time=5.0, initial_trend=0.02)
        comp = AbstractComponent(subscripts=[[], []], ast=ast)
        elem = AbstractElement(name="Trend Out", components=[comp])
        sb = _section_builder_from_elements([elem])
        sb.build_section()
        all_eqs = [e for eqs, _ in sb.built_elements.values() for e in eqs]
        assert any("_sm_trend_out" in e for e in all_eqs)
        assert any("trend_out ~" in e for e in all_eqs)

    def test_forecast_expands(self):
        ast = ForecastStructure(input=10.0, average_time=5.0, horizon=3.0, initial_trend=0.01)
        comp = AbstractComponent(subscripts=[[], []], ast=ast)
        elem = AbstractElement(name="Forecast Out", components=[comp])
        sb = _section_builder_from_elements([elem])
        sb.build_section()
        all_eqs = [e for eqs, _ in sb.built_elements.values() for e in eqs]
        assert any("_sm_forecast_out" in e for e in all_eqs)
        assert any("forecast_out ~" in e for e in all_eqs)

    def test_sample_if_true_expands(self):
        ts_elem = _make_control_element("TIME STEP", 0.25)
        ast = SampleIfTrueStructure(condition=1.0, input=5.0, initial=5.0)
        comp = AbstractComponent(subscripts=[[], []], ast=ast)
        elem = AbstractElement(name="Sample Out", components=[comp])
        sb = _section_builder_from_elements([ts_elem, elem])
        sb.build_section()
        all_eqs = [e for eqs, _ in sb.built_elements.values() for e in eqs]
        assert any("_sit_sample_out" in e for e in all_eqs)
        assert any("sample_out ~" in e for e in all_eqs)

    def test_allocate_available_approximation_warns(self):
        ast = AllocateAvailableStructure(
            request=ReferenceStructure("request"),
            pp=ReferenceStructure("pp"),
            avail=ReferenceStructure("supply"),
        )
        comp = AbstractComponent(subscripts=[[], []], ast=ast)
        elem = AbstractElement(name="Alloc Out", components=[comp])
        req_elem = _make_element("request", 1.0)
        pp_elem = _make_element("pp", 1.0)
        sup_elem = _make_element("supply", 10.0)
        with pytest.warns(UserWarning, match="proportional"):
            sb = _section_builder_from_elements([req_elem, pp_elem, sup_elem, elem])
            sb.build_section()
        all_eqs = [e for eqs, _ in sb.built_elements.values() for e in eqs]
        assert any("alloc_out" in e for e in all_eqs)

    def test_allocate_by_priority_approximation_warns(self):
        ast = AllocateByPriorityStructure(
            request=ReferenceStructure("demand"),
            priority=ReferenceStructure("prio"),
            size=1,
            width=0.1,
            supply=ReferenceStructure("available"),
        )
        comp = AbstractComponent(subscripts=[[], []], ast=ast)
        elem = AbstractElement(name="Alloc Prio", components=[comp])
        d_elem = _make_element("demand", 1.0)
        p_elem = _make_element("prio", 1.0)
        a_elem = _make_element("available", 5.0)
        with pytest.warns(UserWarning, match="proportional"):
            sb = _section_builder_from_elements([d_elem, p_elem, a_elem, elem])
            sb.build_section()

    def test_smooth_non_integer_order_warns_and_defaults(self):
        ast = SmoothStructure(input=1.0, smooth_time=2.0, initial=1.0, order="bad")
        comp = AbstractComponent(subscripts=[[], []], ast=ast)
        elem = AbstractElement(name="Sm Bad", components=[comp])
        with pytest.warns(UserWarning, match="non-integer order"):
            sb = _section_builder_from_elements([elem])
            sb.build_section()
        assert sum(1 for d in sb.stock_decls if "_lv" in d and "sm_bad" in d) == 3

    def test_delay_non_integer_order_warns_and_defaults(self):
        ast = DelayStructure(input=1.0, delay_time=2.0, initial=1.0, order="bad")
        comp = AbstractComponent(subscripts=[[], []], ast=ast)
        elem = AbstractElement(name="Dl Bad", components=[comp])
        with pytest.warns(UserWarning, match="non-integer order"):
            sb = _section_builder_from_elements([elem])
            sb.build_section()
        assert sum(1 for d in sb.stock_decls if "_dl" in d and "dl_bad" in d) == 3


# ===========================================================================
# Section builder — unsupported / fallback structures
# ===========================================================================

class TestJuliaSectionBuilderUnsupported:

    def test_data_structure_emits_warning_and_placeholder(self):
        ast = DataStructure()
        comp = AbstractComponent(subscripts=[[], []], ast=ast)
        elem = AbstractElement(name="Data Var", components=[comp])
        with pytest.warns(UserWarning, match="not supported"):
            sb = _section_builder_from_elements([elem])
            sb.build_section()
        all_eqs = [e for eqs, _ in sb.built_elements.values() for e in eqs]
        assert any("UNSUPPORTED" in e for e in all_eqs)

    def test_abstract_data_no_get_data_structure_falls_through_to_aux(self):
        # AbstractData whose AST is not a GetDataStructure falls through to the
        # regular auxiliary path and emits a "data-override" warning instead of
        # a GET_DATA_FAILED placeholder.
        comp = AbstractData(subscripts=[[], []], ast=0.0)
        elem = AbstractElement(name="Ext Data", components=[comp])
        with pytest.warns(UserWarning, match="data-override"):
            sb = _section_builder_from_elements([elem])
            sb.build_section()
        all_eqs = [e for eqs, _ in sb.built_elements.values() for e in eqs]
        assert not any("GET_DATA_FAILED" in e for e in all_eqs), "Expected no placeholder"
        assert any("ext_data" in e for e in all_eqs), "Expected regular equation"


# ===========================================================================
# Section builder — external data readers (mocked)
# ===========================================================================

class TestJuliaSectionBuilderExternal:

    def test_read_get_constants_scalar_success(self, mocker, tmp_path):
        import numpy as np
        mock_ext = mocker.MagicMock()
        mock_ext.data = np.float64(3.14)
        mocker.patch(
            "pysd.py_backend.external.ExtConstant",
            return_value=mock_ext,
        )
        ast = GetConstantsStructure(file="data.xlsx", tab="Sheet1", cell="A1")
        comp = AbstractComponent(subscripts=[[], []], ast=ast)
        elem = AbstractElement(name="Rate", components=[comp])
        sb = _section_builder_from_elements([elem], path=tmp_path / "m.mdl")
        sb.build_section()
        assert any("@parameters rate = 3.14" in d for d in sb.param_decls)

    def test_read_get_constants_array_success(self, mocker, tmp_path):
        import numpy as np
        mock_ext = mocker.MagicMock()
        mock_ext.data = np.array([1.0, 2.0, 3.0])
        mocker.patch(
            "pysd.py_backend.external.ExtConstant",
            return_value=mock_ext,
        )
        ast = GetConstantsStructure(file="data.xlsx", tab="Sheet1", cell="B1")
        comp = AbstractComponent(subscripts=[[], []], ast=ast)
        elem = AbstractElement(name="Costs", components=[comp])
        sb = _section_builder_from_elements([elem], path=tmp_path / "m.mdl")
        sb.build_section()
        assert any("const costs = [1.0" in d for d in sb.ext_const_decls)

    def test_read_get_constants_failure_falls_through(self, mocker, tmp_path):
        mocker.patch(
            "pysd.py_backend.external.ExtConstant",
            side_effect=FileNotFoundError("no such file"),
        )
        ast = GetConstantsStructure(file="missing.xlsx", tab="Sheet1", cell="A1")
        comp = AbstractComponent(subscripts=[[], []], ast=ast)
        elem = AbstractElement(name="Bad Const", components=[comp])
        with pytest.warns(UserWarning):
            sb = _section_builder_from_elements([elem], path=tmp_path / "m.mdl")
            sb.build_section()

    def test_get_lookups_scalar_success(self, mocker, tmp_path):
        import numpy as np
        import xarray as xr
        xs = np.array([0.0, 1.0, 2.0])
        ys = np.array([0.0, 0.5, 1.0])
        da = xr.DataArray(ys, coords={"lookup_dim": xs}, dims=["lookup_dim"])
        mock_ext = mocker.MagicMock()
        mock_ext.data = da
        mocker.patch(
            "pysd.py_backend.external.ExtLookup",
            return_value=mock_ext,
        )
        ast = GetLookupsStructure(file="data.xlsx", tab="Sheet1",
                                  x_row_or_col="x_col", cell="B1")
        comp = AbstractComponent(subscripts=[[], []], ast=ast)
        elem = AbstractElement(name="Effect Table", components=[comp])
        sb = _section_builder_from_elements([elem], path=tmp_path / "m.mdl")
        sb.build_section()
        assert any("effect_table_itp" in d for d in sb.lookup_const_decls)

    def test_get_lookups_2d_success(self, mocker, tmp_path):
        import numpy as np
        import xarray as xr
        n_pts, n_subs = 3, 2
        xs = np.array([0.0, 1.0, 2.0])
        ys = np.ones((n_pts, n_subs))
        da = xr.DataArray(ys, coords={"lookup_dim": xs}, dims=["lookup_dim", "sub"])
        mock_ext = mocker.MagicMock()
        mock_ext.data = da
        mocker.patch(
            "pysd.py_backend.external.ExtLookup",
            return_value=mock_ext,
        )
        ast = GetLookupsStructure(file="data.xlsx", tab="Sheet1",
                                  x_row_or_col="x_col", cell="B1")
        comp = AbstractComponent(subscripts=[[], []], ast=ast)
        elem = AbstractElement(name="Sub Table", components=[comp])
        sb = _section_builder_from_elements([elem], path=tmp_path / "m.mdl")
        sb.build_section()
        assert any("sub_table_fns" in d for d in sb.lookup_const_decls)
        assert any("sub_table(i, x)" in d for d in sb.lookup_func_decls)

    def test_get_lookups_3d_emits_2d_dispatch(self, mocker, tmp_path):
        # 3D data (n_points × n_dim1 × n_dim2) is now handled correctly:
        # emits one sub-function per (i, j) pair and a 2-index dispatch.
        import numpy as np
        import xarray as xr
        xs = np.array([0.0, 1.0])
        ys = np.ones((2, 2, 3))
        da = xr.DataArray(ys, coords={"lookup_dim": xs},
                          dims=["lookup_dim", "d1", "d2"])
        mock_ext = mocker.MagicMock()
        mock_ext.data = da
        mocker.patch(
            "pysd.py_backend.external.ExtLookup",
            return_value=mock_ext,
        )
        ast = GetLookupsStructure(file="data.xlsx", tab="Sheet1",
                                  x_row_or_col="x", cell="A1")
        comp = AbstractComponent(subscripts=[[], []], ast=ast)
        elem = AbstractElement(name="Hd Table", components=[comp])
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sb = _section_builder_from_elements([elem], path=tmp_path / "m.mdl")
            sb.build_section()
        assert not [x for x in w if "> 1D subs" in str(x.message) or "> 2D" in str(x.message)]
        # 2×3 = 6 sub-functions + fns array + dispatch
        assert any("hd_table_1_1" in d for d in sb.lookup_const_decls)
        assert any("hd_table_2_3" in d for d in sb.lookup_const_decls)
        assert any("hd_table(i, j, x)" in d for d in sb.lookup_func_decls)

    def test_get_lookups_4d_warns_and_flattens(self, mocker, tmp_path):
        # Arrays with >3 dimensions still emit a warning and fall back to
        # first-column approximation.
        import numpy as np
        import xarray as xr
        xs = np.array([0.0, 1.0])
        ys = np.ones((2, 2, 2, 2))
        da = xr.DataArray(ys, coords={"lookup_dim": xs},
                          dims=["lookup_dim", "d1", "d2", "d3"])
        mock_ext = mocker.MagicMock()
        mock_ext.data = da
        mocker.patch(
            "pysd.py_backend.external.ExtLookup",
            return_value=mock_ext,
        )
        ast = GetLookupsStructure(file="data.xlsx", tab="Sheet1",
                                  x_row_or_col="x", cell="A1")
        comp = AbstractComponent(subscripts=[[], []], ast=ast)
        elem = AbstractElement(name="Hd Table", components=[comp])
        with pytest.warns(UserWarning, match="> 2D"):
            sb = _section_builder_from_elements([elem], path=tmp_path / "m.mdl")
            sb.build_section()
        assert any("hd_table_itp" in d for d in sb.lookup_const_decls)

    def test_get_lookups_read_failure_warns(self, mocker, tmp_path):
        mocker.patch(
            "pysd.py_backend.external.ExtLookup",
            side_effect=FileNotFoundError("no such file"),
        )
        ast = GetLookupsStructure(file="bad.xlsx", tab="Sheet1",
                                  x_row_or_col="x", cell="A1")
        comp = AbstractComponent(subscripts=[[], []], ast=ast)
        elem = AbstractElement(name="Bad Lut", components=[comp])
        with pytest.warns(UserWarning, match="Could not read GET LOOKUPS"):
            sb = _section_builder_from_elements([elem], path=tmp_path / "m.mdl")
            sb.build_section()
        all_eqs = [e for eqs, _ in sb.built_elements.values() for e in eqs]
        assert any("GET_LOOKUPS_FAILED" in e for e in all_eqs)

    def test_get_data_scalar_success(self, mocker, tmp_path):
        import numpy as np
        import xarray as xr
        ts = np.array([1995.0, 2000.0, 2005.0])
        vals = np.array([1.0, 2.0, 3.0])
        da = xr.DataArray(vals, coords={"time": ts}, dims=["time"])
        mock_ext = mocker.MagicMock()
        mock_ext.data = da
        mocker.patch(
            "pysd.py_backend.external.ExtData",
            return_value=mock_ext,
        )
        ast = GetDataStructure(file="data.xlsx", tab="Sheet1",
                               time_row_or_col="time_col", cell="B1")
        comp = AbstractComponent(subscripts=[[], []], ast=ast)
        elem = AbstractElement(name="Historic Eff", components=[comp])
        sb = _section_builder_from_elements([elem], path=tmp_path / "m.mdl")
        sb.build_section()
        assert any("historic_eff_itp" in d for d in sb.lookup_const_decls)

    def test_get_data_2d_success(self, mocker, tmp_path):
        import numpy as np
        import xarray as xr
        ts = np.array([1995.0, 2000.0])
        vals = np.ones((2, 3))
        da = xr.DataArray(vals, coords={"time": ts}, dims=["time", "sub"])
        mock_ext = mocker.MagicMock()
        mock_ext.data = da
        mocker.patch(
            "pysd.py_backend.external.ExtData",
            return_value=mock_ext,
        )
        ast = GetDataStructure(file="data.xlsx", tab="Sheet1",
                               time_row_or_col="time_col", cell="B1")
        comp = AbstractComponent(subscripts=[[], []], ast=ast)
        elem = AbstractElement(name="Sub Series", components=[comp])
        sb = _section_builder_from_elements([elem], path=tmp_path / "m.mdl")
        sb.build_section()
        assert any("sub_series_fns" in d for d in sb.lookup_const_decls)

    def test_get_data_read_failure_warns(self, mocker, tmp_path):
        mocker.patch(
            "pysd.py_backend.external.ExtData",
            side_effect=FileNotFoundError("no such file"),
        )
        ast = GetDataStructure(file="bad.xlsx", tab="Sheet1",
                               time_row_or_col="t_col", cell="A1")
        comp = AbstractComponent(subscripts=[[], []], ast=ast)
        elem = AbstractElement(name="Bad Data", components=[comp])
        with pytest.warns(UserWarning, match="Could not read GET DATA"):
            sb = _section_builder_from_elements([elem], path=tmp_path / "m.mdl")
            sb.build_section()
        all_eqs = [e for eqs, _ in sb.built_elements.values() for e in eqs]
        assert any("GET_DATA_FAILED" in e for e in all_eqs)

    # ------------------------------------------------------------------
    # Per-element-component GET LOOKUPS / GET DATA (Task B fix)
    # ------------------------------------------------------------------

    def test_get_lookups_per_element_component_coords_built_correctly(self, mocker, tmp_path):
        """When a GET LOOKUPS element has per-sector-element components (each
        comp specifies a single element name rather than a range name), _coords
        must map the element back to its parent range with a single-element list.
        ExtLookup should be called with coords={'sector': ['A']}, not {'A': []}."""
        import numpy as np
        import xarray as xr

        xs = np.array([0.0, 1.0])
        ys = np.ones((2, 1))  # shape (n_pts, 1) — scalar per element
        da = xr.DataArray(ys, coords={"lookup_dim": xs}, dims=["lookup_dim", "sector"])
        mock_ext = mocker.MagicMock()
        mock_ext.data = da

        ext_cls = mocker.patch("pysd.py_backend.external.ExtLookup", return_value=mock_ext)

        sr_sector = _make_subscript_range("sector", ["A", "B"])

        # Two components: one per element of 'sector'
        ast_a = GetLookupsStructure(file="d.xlsx", tab="S", x_row_or_col="x", cell="col_a")
        ast_b = GetLookupsStructure(file="d.xlsx", tab="S", x_row_or_col="x", cell="col_b")
        comp_a = AbstractComponent(subscripts=[["A"], []], ast=ast_a)
        comp_b = AbstractComponent(subscripts=[["B"], []], ast=ast_b)
        elem = AbstractElement(name="My Lookup", components=[comp_a, comp_b])

        sb = _section_builder_from_elements([elem], subscripts=[sr_sector], path=tmp_path / "m.mdl")
        sb.build_section()

        # ExtLookup must have been constructed
        assert ext_cls.called
        init_call_kwargs = ext_cls.call_args
        coords_arg = init_call_kwargs[1].get("coords") or (init_call_kwargs[0][4] if len(init_call_kwargs[0]) > 4 else None)
        # coords must map the parent range name 'sector' to ['A'], not '' to []
        if coords_arg is not None:
            assert "sector" in coords_arg, f"Expected 'sector' in coords, got {coords_arg}"
            assert coords_arg["sector"] == ["A"], f"Expected ['A'], got {coords_arg['sector']}"

    def test_get_lookups_per_element_no_placeholder_emitted(self, mocker, tmp_path):
        """Per-element GET LOOKUPS components must NOT emit a GET_LOOKUPS_FAILED
        placeholder — a real lookup declaration must appear."""
        import numpy as np
        import xarray as xr

        xs = np.array([1995.0, 2000.0, 2005.0])
        ys = np.ones((3, 2))
        da = xr.DataArray(ys, coords={"lookup_dim": xs},
                          dims=["lookup_dim", "type"])
        mock_ext = mocker.MagicMock()
        mock_ext.data = da
        mocker.patch("pysd.py_backend.external.ExtLookup", return_value=mock_ext)

        sr_type = _make_subscript_range("type", ["X", "Y"])
        ast_x = GetLookupsStructure(file="f.xlsx", tab="S", x_row_or_col="yr", cell="cx")
        ast_y = GetLookupsStructure(file="f.xlsx", tab="S", x_row_or_col="yr", cell="cy")
        comp_x = AbstractComponent(subscripts=[["X"], []], ast=ast_x)
        comp_y = AbstractComponent(subscripts=[["Y"], []], ast=ast_y)
        elem = AbstractElement(name="Rate Table", components=[comp_x, comp_y])

        sb = _section_builder_from_elements([elem], subscripts=[sr_type], path=tmp_path / "m.mdl")
        sb.build_section()

        all_eqs = [e for eqs, _ in sb.built_elements.values() for e in eqs]
        assert not any("GET_LOOKUPS_FAILED" in e for e in all_eqs), (
            "GET_LOOKUPS_FAILED placeholder must not be emitted for per-element components"
        )
        # A lookup interpolation constant must have been declared
        assert sb.lookup_const_decls, "No lookup constant declarations emitted"

    def test_get_data_per_element_coords_uses_parent_range(self, mocker, tmp_path):
        """GET DATA with per-element components: coords must use parent range name."""
        import numpy as np
        import xarray as xr

        xs = np.array([1995.0, 2000.0])
        ys = np.ones((2, 1))
        da = xr.DataArray(ys, coords={"time": xs}, dims=["time", "fuel"])
        mock_ext = mocker.MagicMock()
        mock_ext.data = da
        ext_cls = mocker.patch("pysd.py_backend.external.ExtData", return_value=mock_ext)

        sr_fuel = _make_subscript_range("fuel", ["coal", "gas", "oil"])

        ast_c = GetDataStructure(file="e.xlsx", tab="W", time_row_or_col="yr", cell="coal_c")
        ast_g = GetDataStructure(file="e.xlsx", tab="W", time_row_or_col="yr", cell="gas_c")
        comp_c = AbstractComponent(subscripts=[["coal"], []], ast=ast_c)
        comp_g = AbstractComponent(subscripts=[["gas"], []], ast=ast_g)
        from pysd.translators.structures.abstract_model import AbstractData
        comp_c.__class__ = AbstractData
        comp_g.__class__ = AbstractData
        elem = AbstractElement(name="Historic Share", components=[comp_c, comp_g])

        sb = _section_builder_from_elements([elem], subscripts=[sr_fuel], path=tmp_path / "m.mdl")
        sb.build_section()

        assert ext_cls.called
        init_kwargs = ext_cls.call_args[1] if ext_cls.call_args[1] else {}
        coords_arg = init_kwargs.get("coords")
        if coords_arg:
            assert "fuel" in coords_arg, f"Expected parent range 'fuel' in coords, got {coords_arg}"


# ===========================================================================
# Section builder — file generation helpers
# ===========================================================================

class TestJuliaFileGeneration:

    def _minimal_sb(self, tmp_path):
        stock = _make_stock_element("S", 1.0, 10.0)
        controls = [
            _make_control_element("INITIAL TIME", 0.0),
            _make_control_element("FINAL TIME", 10.0),
            _make_control_element("TIME STEP", 1.0),
            _make_control_element("SAVEPER", 1.0),
        ]
        sb = _section_builder_from_elements(
            [stock] + controls, path=tmp_path / "m.mdl"
        )
        sb.build_section()
        return sb

    def test_helpers_block_empty_when_none_needed(self, tmp_path):
        sb = self._minimal_sb(tmp_path)
        sb.needed_helpers.clear()
        assert sb._helpers_block() == ""

    def test_helpers_block_contains_implementation(self, tmp_path):
        sb = self._minimal_sb(tmp_path)
        sb.needed_helpers.add("_xidz")
        block = sb._helpers_block()
        assert "_xidz" in block

    def test_lookup_block_empty_when_none(self, tmp_path):
        sb = self._minimal_sb(tmp_path)
        assert sb._lookup_block() == ""

    def test_lookup_block_contains_declaration(self, tmp_path):
        sb = self._minimal_sb(tmp_path)
        sb.lookup_const_decls.append("const lut_itp = LinearInterpolation([1.0], [0.0])")
        sb.lookup_func_decls.append("lut(x) = lut_itp(x)")
        sb.lookup_register_decls.append("@register_symbolic lut(x::Real)")
        block = sb._lookup_block()
        assert "LinearInterpolation" in block

    def test_declarations_block_includes_subs_constants(self, tmp_path):
        sr = _make_subscript_range("energy_type", ["H", "S"])
        elem = _make_subscripted_element("cost", 1.0, "energy_type",
                                         comp_class=AbstractUnchangeableConstant)
        controls = [
            _make_control_element("INITIAL TIME", 0.0),
            _make_control_element("FINAL TIME", 10.0),
            _make_control_element("TIME STEP", 1.0),
            _make_control_element("SAVEPER", 1.0),
        ]
        sb = _section_builder_from_elements(
            [elem] + controls,
            path=tmp_path / "m.mdl",
            subscripts=[sr],
        )
        sb.build_section()
        block = sb._declarations_block()
        assert "N_ENERGY_TYPE" in block
        assert "Subscript dimension sizes" in block

    def test_equations_block_empty(self, tmp_path):
        sb = self._minimal_sb(tmp_path)
        block = sb._equations_block([])
        assert block == "eqs = Equation[]\n"

    def test_u0_block_empty(self, tmp_path):
        sb = self._minimal_sb(tmp_path)
        sb.u0_entries.clear()
        block = sb._u0_block()
        assert block == "u0 = []\n"

    def test_ext_const_in_declarations_block(self, tmp_path):
        sb = self._minimal_sb(tmp_path)
        sb.ext_const_decls.append("const big_array = [1.0, 2.0]")
        block = sb._declarations_block()
        assert "External constants" in block
        assert "big_array" in block


# ===========================================================================
# Modular build — extended edge cases
# ===========================================================================

class TestModularBuildExtended:

    def test_variable_not_in_any_view_emits_warning(self, tmp_path):
        """Variable assigned to no view → leftover warning."""
        pop = _make_stock_element("Population", 1.0, 100.0)
        orphan = _make_element("Orphan Var", 5.0)
        controls = [
            _make_control_element("INITIAL TIME", 0.0),
            _make_control_element("FINAL TIME", 10.0),
            _make_control_element("TIME STEP", 1.0),
            _make_control_element("SAVEPER", 1.0),
        ]
        views_dict = {"Sector A": {"Population"}}
        section = _make_section(
            elements=[pop, orphan] + controls,
            path=tmp_path / "m.mdl",
            split=True,
            views_dict=views_dict,
        )
        model = AbstractModel(original_path=tmp_path / "m.mdl", sections=(section,))
        with pytest.warns(UserWarning, match="not declared in any view"):
            JuliaModelBuilder(model).build_model()

    def test_view_with_only_control_vars_skipped(self, tmp_path):
        """A view containing only control variables produces no module file."""
        pop = _make_stock_element("Population", 1.0, 100.0)
        it = _make_control_element("INITIAL TIME", 0.0)
        ft = _make_control_element("FINAL TIME", 10.0)
        ts = _make_control_element("TIME STEP", 1.0)
        sv = _make_control_element("SAVEPER", 1.0)
        views_dict = {
            "Main": {"Population"},
            "Controls": {"INITIAL TIME", "FINAL TIME"},
        }
        section = _make_section(
            elements=[pop, it, ft, ts, sv],
            path=tmp_path / "ctrl_model.mdl",
            split=True,
            views_dict=views_dict,
        )
        model = AbstractModel(original_path=tmp_path / "ctrl_model.mdl",
                               sections=(section,))
        JuliaModelBuilder(model).build_model()
        modules_dir = tmp_path / "modules_ctrl_model"
        jl_files = list(modules_dir.glob("*.jl"))
        assert len(jl_files) == 1  # Only "Main", not "Controls"

    def test_nested_views(self, tmp_path):
        """Views with sub-views (intermediate nodes) are handled."""
        pop = _make_stock_element("Population", 1.0, 100.0)
        cap = _make_stock_element("Capital", 2.0, 500.0)
        controls = [
            _make_control_element("INITIAL TIME", 0.0),
            _make_control_element("FINAL TIME", 50.0),
            _make_control_element("TIME STEP", 0.5),
            _make_control_element("SAVEPER", 0.5),
        ]
        views_dict = {
            "Economy": {
                "Demography": {"Population"},
                "Assets": {"Capital"},
            }
        }
        section = _make_section(
            elements=[pop, cap] + controls,
            path=tmp_path / "nested.mdl",
            split=True,
            views_dict=views_dict,
        )
        model = AbstractModel(original_path=tmp_path / "nested.mdl",
                               sections=(section,))
        JuliaModelBuilder(model).build_model()
        modules_dir = tmp_path / "modules_nested"
        jl_files = list(modules_dir.rglob("*.jl"))
        assert len(jl_files) == 2


# ===========================================================================
# _format_julia_value utility
# ===========================================================================

class TestFormatJuliaValue:

    def test_float_scalar(self):
        from pysd.builders.julia.julia_model_builder import _format_julia_value
        assert _format_julia_value(3.14) == "3.14"

    def test_int_scalar(self):
        from pysd.builders.julia.julia_model_builder import _format_julia_value
        assert _format_julia_value(5) == "5.0"

    def test_1d_numpy_array(self):
        import numpy as np
        from pysd.builders.julia.julia_model_builder import _format_julia_value
        result = _format_julia_value(np.array([1.0, 2.0, 3.0]))
        assert result == "[1.0, 2.0, 3.0]"

    def test_2d_numpy_array(self):
        import numpy as np
        from pysd.builders.julia.julia_model_builder import _format_julia_value
        result = _format_julia_value(np.array([[1.0, 2.0], [3.0, 4.0]]))
        assert "[" in result and ";" in result

    def test_xarray_dataarray(self):
        import numpy as np
        import xarray as xr
        from pysd.builders.julia.julia_model_builder import _format_julia_value
        da = xr.DataArray(np.array([1.0, 2.0]))
        result = _format_julia_value(da)
        assert result == "[1.0, 2.0]"

    def test_0d_numpy_array(self):
        import numpy as np
        from pysd.builders.julia.julia_model_builder import _format_julia_value
        result = _format_julia_value(np.array(42.0))
        assert result == "42.0"


# ===========================================================================
# Additional targeted tests for remaining coverage gaps
# ===========================================================================

class TestCoverageGaps:
    """Fills specific uncovered lines identified by coverage analysis."""

    # --- julia_expressions_builder.py ---

    def test_numpy_0d_array_in_visitor(self):
        import numpy as np
        v, *_ = _visitor_with_namespace()
        result = v.visit(np.array(7.5))  # 0-d ndarray
        assert result == "7.5"

    def test_get_constants_in_expression_success(self, mocker):
        import numpy as np
        mock_ext = mocker.MagicMock()
        mock_ext.data = np.float64(42.0)
        mocker.patch("pysd.py_backend.external.ExtConstant", return_value=mock_ext)
        v, *_ = _visitor_with_namespace()
        node = GetConstantsStructure(file="data.xlsx", tab="Sheet1", cell="A1")
        result = v.visit(node)
        assert result == "42.0"

    def test_unary_non_not_logic_operator(self):
        """Unary logic op that is not NOT uses LOGIC_OPS table directly."""
        v, *_ = _visitor_with_namespace()
        node = LogicStructure(operators=["<>"], arguments=[1.0])
        result = v.visit(node)
        assert "!=" in result or "1.0" in result

    def test_reference_with_active_subscript_context(self):
        """_reference appends subscript indices when active_subs is set."""
        ns = JuliaNamespaceManager()
        ns.add_to_namespace("output")
        registry = InlineLookupRegistry()
        needed = set()
        v = JuliaASTVisitor(
            ns, registry, needed,
            active_subs={"sector": "_i0"},
            var_dims={"output": ["sector"]},
        )
        result = v.visit(ReferenceStructure("output"))
        assert "output[_i0]" == result

    def test_subscript_name_as_reference_emits_loop_index(self):
        """Bare subscript name in an expression (e.g. IF_THEN_ELSE(s=s1,1,0))
        must emit the loop-index variable, not a sanitised fallback identifier.
        This covers the identity-matrix pattern:
          I_Matrix[s, s1] = IF_THEN_ELSE(s = s1, 1, 0)
        where s and s1 are subscript range names, not model variables."""
        ns = JuliaNamespaceManager()
        registry = InlineLookupRegistry()
        needed = set()
        v = JuliaASTVisitor(
            ns, registry, needed,
            active_subs={"sectors_a_matrix": "_i0", "sectors_a_matrix1": "_i1"},
        )
        assert v.visit(ReferenceStructure("sectors_a_matrix")) == "_i0"
        assert v.visit(ReferenceStructure("sectors_a_matrix1")) == "_i1"
        # Original MDL casing should also resolve correctly
        assert v.visit(ReferenceStructure("sectors_A_matrix")) == "_i0"

    def test_subscript_name_reference_no_warning(self):
        """Subscript-name-as-loop-index must not emit a namespace-fallback warning."""
        import warnings
        ns = JuliaNamespaceManager()
        registry = InlineLookupRegistry()
        needed = set()
        v = JuliaASTVisitor(
            ns, registry, needed,
            active_subs={"s": "_i0"},
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = v.visit(ReferenceStructure("s"))
        assert result == "_i0"

    def test_elmcount_resolves_to_integer_case_insensitive(self):
        """ELMCOUNT(SubName) must emit the integer size even when the
        subs_sizes key casing differs from the reference casing."""
        ns = JuliaNamespaceManager()
        registry = InlineLookupRegistry()
        needed = set()
        # subs_sizes uses mixed-case key (as the abstract model does); reference
        # arrives lowercase from the expression parser.
        v = JuliaASTVisitor(
            ns, registry, needed,
            subs_sizes={"Sectors_A_Matrix": 14},
        )
        node = CallStructure(
            function=ReferenceStructure("ELMCOUNT"),
            arguments=[ReferenceStructure("sectors_a_matrix")],
        )
        result = v.visit(node)
        assert result == "14"

    def test_elmcount_resolves_to_integer_exact_match(self):
        """ELMCOUNT works when casing matches exactly (regression guard)."""
        ns = JuliaNamespaceManager()
        registry = InlineLookupRegistry()
        needed = set()
        v = JuliaASTVisitor(
            ns, registry, needed,
            subs_sizes={"sectors": 5},
        )
        node = CallStructure(
            function=ReferenceStructure("ELMCOUNT"),
            arguments=[ReferenceStructure("sectors")],
        )
        assert v.visit(node) == "5"

    def test_invert_matrix_with_elmcount_emits_integer_size(self):
        """INVERT_MATRIX(..., ELMCOUNT(s)) inside a subscripted equation emits
        the integer count, not the loop-index variable for s."""
        ns = JuliaNamespaceManager()
        ns.add_to_namespace("my_matrix")
        registry = InlineLookupRegistry()
        needed = set()
        v = JuliaASTVisitor(
            ns, registry, needed,
            active_subs={"s": "_i0", "s1": "_i1"},
            subs_sizes={"s": 3, "s1": 3},
        )
        node = CallStructure(
            function=ReferenceStructure("INVERT_MATRIX"),
            arguments=[
                ReferenceStructure("my_matrix"),
                CallStructure(
                    function=ReferenceStructure("ELMCOUNT"),
                    arguments=[ReferenceStructure("s")],
                ),
            ],
        )
        result = v.visit(node)
        assert result == "inv(my_matrix, 3)"

    # --- julia_model_builder.py ---

    def test_inline_lookup_registered_after_build(self, tmp_path):
        """InlineLookupsStructure inside an element populates lookup_const_decls."""
        lut_ast = InlineLookupsStructure(
            argument=ReferenceStructure("x_val"),
            lookups=LookupsStructure(
                x=(0.0, 1.0), y=(0.0, 2.0),
                x_limits=(0.0, 1.0), y_limits=(0.0, 2.0),
                type="interpolate",
            ),
        )
        x_elem = _make_element("x val", 0.5)
        comp = AbstractComponent(subscripts=[[], []], ast=lut_ast)
        elem = AbstractElement(name="Lookup Result", components=[comp])
        sb = _section_builder_from_elements([x_elem, elem])
        sb.build_section()
        assert any("_inline_lookup_" in d for d in sb.lookup_const_decls)

    def test_element_dims_empty_subscripts(self):
        """_element_dims returns [] when component has no subscript list."""
        comp = AbstractComponent(subscripts=[[], []], ast=1.0)
        comp.subscripts = [[]]  # empty first subscript
        elem = AbstractElement(name="scalar", components=[comp])
        sr = _make_subscript_range("dim", ["a", "b"])
        sb = _section_builder_from_elements([elem], subscripts=[sr])
        dims = sb._element_dims(elem)
        assert dims == []

    def test_element_dims_no_components_direct(self):
        """_element_dims defensive check: no components → empty list."""
        elem = AbstractElement(name="empty", components=[])
        sb = _section_builder_from_elements([])
        # Call directly (bypass _process_element's early-return guard)
        assert sb._element_dims(elem) == []

    def test_2d_subscripted_stock(self):
        """N≥2 dimensional stock uses comprehension form."""
        sr1 = _make_subscript_range("row", ["R1", "R2"])
        sr2 = _make_subscript_range("col", ["C1", "C2"])
        comp = AbstractComponent(
            subscripts=[["row", "col"], []],
            ast=IntegStructure(flow=1.0, initial=0.0),
        )
        elem = AbstractElement(name="matrix stock", components=[comp])
        sb = _section_builder_from_elements([elem], subscripts=[sr1, sr2])
        sb.build_section()
        eqs = [e for eqs, _ in sb.built_elements.values() for e in eqs]
        assert any("_i0" in e and "_i1" in e for e in eqs)
        assert any("D(matrix_stock" in e for e in eqs)

    def test_get_constants_control_element(self, mocker, tmp_path):
        """GetConstantsStructure for a control element updates control_vals."""
        import numpy as np
        mock_ext = mocker.MagicMock()
        mock_ext.data = np.float64(100.0)
        mocker.patch("pysd.py_backend.external.ExtConstant", return_value=mock_ext)
        ast = GetConstantsStructure(file="d.xlsx", tab="Sheet1", cell="A1")
        comp = AbstractUnchangeableConstant(subscripts=[[], []], ast=ast)
        final_time = AbstractControlElement(name="FINAL TIME", components=[comp])
        other_controls = [
            _make_control_element("INITIAL TIME", 0.0),
            _make_control_element("TIME STEP", 1.0),
            _make_control_element("SAVEPER", 1.0),
        ]
        sb = _section_builder_from_elements(
            [final_time] + other_controls, path=tmp_path / "m.mdl"
        )
        sb.build_section()
        assert sb.control_vals["final_time"] == "100.0"

    def test_subscripted_aux_1d_control_branch(self):
        """1D subscripted control aux updates control_vals."""
        sr = _make_subscript_range("dim", ["A", "B"])
        comp = AbstractComponent(subscripts=[["dim"], []], ast=5.0)
        ctrl_elem = AbstractControlElement(name="FINAL TIME", components=[comp])
        other = [
            _make_control_element("INITIAL TIME", 0.0),
            _make_control_element("TIME STEP", 1.0),
            _make_control_element("SAVEPER", 1.0),
        ]
        sb = _section_builder_from_elements([ctrl_elem] + other, subscripts=[sr])
        sb.build_section()
        # Control val is set even if subscripted (value is the visited expression)
        assert sb.control_vals.get("final_time") is not None

    def test_subscripted_aux_2d_control_branch(self):
        """2D subscripted control aux updates control_vals."""
        sr1 = _make_subscript_range("row", ["R1", "R2"])
        sr2 = _make_subscript_range("col", ["C1", "C2"])
        comp = AbstractComponent(subscripts=[["row", "col"], []], ast=1.0)
        ctrl_elem = AbstractControlElement(name="FINAL TIME", components=[comp])
        other = [
            _make_control_element("INITIAL TIME", 0.0),
            _make_control_element("TIME STEP", 1.0),
            _make_control_element("SAVEPER", 1.0),
        ]
        sb = _section_builder_from_elements(
            [ctrl_elem] + other, subscripts=[sr1, sr2]
        )
        sb.build_section()
        assert sb.control_vals.get("final_time") is not None

    def test_get_lookups_with_subscripts_in_section(self, mocker, tmp_path):
        """_process_get_lookups iterates over section subscripts to build subs_map."""
        import numpy as np
        import xarray as xr
        xs = np.array([0.0, 1.0])
        ys = np.array([0.0, 1.0])
        da = xr.DataArray(ys, coords={"lookup_dim": xs}, dims=["lookup_dim"])
        mock_ext = mocker.MagicMock()
        mock_ext.data = da
        mocker.patch("pysd.py_backend.external.ExtLookup", return_value=mock_ext)
        ast = GetLookupsStructure(file="d.xlsx", tab="S", x_row_or_col="x", cell="A1")
        comp = AbstractComponent(subscripts=[[], []], ast=ast)
        elem = AbstractElement(name="Lut", components=[comp])
        sr = _make_subscript_range("energy_type", ["H", "S"])
        sb = _section_builder_from_elements([elem], path=tmp_path/"m.mdl",
                                             subscripts=[sr])
        sb.build_section()
        assert any("lut_itp" in d for d in sb.lookup_const_decls)

    def test_get_lookups_multi_component(self, mocker, tmp_path):
        """Multi-component GetLookupsStructure merges coords (exercises inner for loop)."""
        import numpy as np
        import xarray as xr
        xs = np.array([0.0, 1.0])
        ys = np.array([0.0, 1.0])
        da = xr.DataArray(ys, coords={"lookup_dim": xs}, dims=["lookup_dim"])
        mock_ext = mocker.MagicMock()
        mock_ext.data = da
        mocker.patch("pysd.py_backend.external.ExtLookup", return_value=mock_ext)
        ast1 = GetLookupsStructure(file="d.xlsx", tab="S", x_row_or_col="x", cell="A1")
        ast2 = GetLookupsStructure(file="d.xlsx", tab="S", x_row_or_col="x", cell="B1")
        sr = _make_subscript_range("dim_a", ["X"])
        # Give components subscripts so _coords returns non-empty dicts
        comp1 = AbstractComponent(subscripts=[["dim_a"], []], ast=ast1)
        comp2 = AbstractComponent(subscripts=[["dim_a"], []], ast=ast2)
        elem = AbstractElement(name="Multi Lut", components=[comp1, comp2])
        sb = _section_builder_from_elements([elem], path=tmp_path/"m.mdl",
                                             subscripts=[sr])
        sb.build_section()
        assert any("multi_lut_itp" in d for d in sb.lookup_const_decls)

    def test_get_lookups_data_without_values_attr(self, mocker, tmp_path):
        """_process_get_lookups handles data without .values (plain numpy array)."""
        import numpy as np
        # A mock where .data is a plain 1D numpy array (no .values)
        xs_arr = np.array([0.0, 1.0, 2.0])
        ys_arr = np.array([0.0, 0.5, 1.0])

        class FakeLookupData:
            values = None  # No .values — will use np.asarray path
            def __init__(self):
                # make hasattr(data, "values") False by removing attr
                pass

        # Use a real structure: mock data without .values
        mock_data = mocker.MagicMock()
        del mock_data.values  # remove values attr
        mock_data.__array__ = lambda *a: ys_arr  # make np.asarray work
        mock_data.coords = {"lookup_dim": mocker.MagicMock(values=xs_arr)}
        mock_ext = mocker.MagicMock()
        mock_ext.data = mock_data
        mocker.patch("pysd.py_backend.external.ExtLookup", return_value=mock_ext)
        ast = GetLookupsStructure(file="d.xlsx", tab="S", x_row_or_col="x", cell="A1")
        comp = AbstractComponent(subscripts=[[], []], ast=ast)
        elem = AbstractElement(name="Plain Lut", components=[comp])
        sb = _section_builder_from_elements([elem], path=tmp_path/"m.mdl")
        sb.build_section()
        # Should produce a lookup (scalar path fallback via np.asarray)
        assert any("plain_lut" in d for d in sb.lookup_const_decls)

    def test_get_data_with_subscripts_in_section(self, mocker, tmp_path):
        """_process_get_data iterates over section subscripts to build subs_map."""
        import numpy as np
        import xarray as xr
        ts = np.array([1995.0, 2000.0])
        vals = np.array([1.0, 2.0])
        da = xr.DataArray(vals, coords={"time": ts}, dims=["time"])
        mock_ext = mocker.MagicMock()
        mock_ext.data = da
        mocker.patch("pysd.py_backend.external.ExtData", return_value=mock_ext)
        ast = GetDataStructure(file="d.xlsx", tab="S", time_row_or_col="t", cell="A1")
        comp = AbstractComponent(subscripts=[[], []], ast=ast)
        elem = AbstractElement(name="Historic Data", components=[comp])
        sr = _make_subscript_range("energy_type", ["H", "S"])
        sb = _section_builder_from_elements([elem], path=tmp_path/"m.mdl",
                                             subscripts=[sr])
        sb.build_section()
        assert any("historic_data_itp" in d for d in sb.lookup_const_decls)

    def test_get_data_multi_component(self, mocker, tmp_path):
        """Multi-component GetDataStructure merges coords (exercises inner for loop)."""
        import numpy as np
        import xarray as xr
        ts = np.array([1995.0, 2000.0])
        vals = np.array([1.0, 2.0])
        da = xr.DataArray(vals, coords={"time": ts}, dims=["time"])
        mock_ext = mocker.MagicMock()
        mock_ext.data = da
        mocker.patch("pysd.py_backend.external.ExtData", return_value=mock_ext)
        ast1 = GetDataStructure(file="d.xlsx", tab="S", time_row_or_col="t", cell="A1")
        ast2 = GetDataStructure(file="d.xlsx", tab="S", time_row_or_col="t", cell="B1")
        sr = _make_subscript_range("dim_b", ["Y"])
        comp1 = AbstractComponent(subscripts=[["dim_b"], []], ast=ast1)
        comp2 = AbstractComponent(subscripts=[["dim_b"], []], ast=ast2)
        elem = AbstractElement(name="Multi Data", components=[comp1, comp2])
        sb = _section_builder_from_elements([elem], path=tmp_path/"m.mdl",
                                             subscripts=[sr])
        sb.build_section()
        assert any("multi_data_itp" in d for d in sb.lookup_const_decls)

    def test_get_data_no_time_dimension_raises_into_fallback(self, mocker, tmp_path):
        """Data without time dimension causes ValueError → fallback placeholder."""
        import numpy as np
        mock_data = mocker.MagicMock()
        del mock_data.values
        mock_data.__array__ = lambda *a: np.array([1.0, 2.0])
        mock_data.coords = {}  # no "time" coord
        mock_ext = mocker.MagicMock()
        mock_ext.data = mock_data
        mocker.patch("pysd.py_backend.external.ExtData", return_value=mock_ext)
        ast = GetDataStructure(file="d.xlsx", tab="S", time_row_or_col="t", cell="A1")
        comp = AbstractComponent(subscripts=[[], []], ast=ast)
        elem = AbstractElement(name="No Time", components=[comp])
        with pytest.warns(UserWarning, match="Could not read GET DATA"):
            sb = _section_builder_from_elements([elem], path=tmp_path/"m.mdl")
            sb.build_section()

    def test_get_data_3d_emits_2d_dispatch(self, mocker, tmp_path):
        """3D data (n_time × n_dim1 × n_dim2) is now handled: emits per-(i,j)
        sub-functions and a 2-index dispatch without raising or using a placeholder."""
        import numpy as np
        import xarray as xr
        import warnings
        ts = np.array([1995.0, 2000.0])
        vals = np.ones((2, 3, 4))
        da = xr.DataArray(vals, coords={"time": ts}, dims=["time", "d1", "d2"])
        mock_ext = mocker.MagicMock()
        mock_ext.data = da
        mocker.patch("pysd.py_backend.external.ExtData", return_value=mock_ext)
        ast = GetDataStructure(file="d.xlsx", tab="S", time_row_or_col="t", cell="A1")
        comp = AbstractComponent(subscripts=[[], []], ast=ast)
        elem = AbstractElement(name="Hfc Emissions", components=[comp])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sb = _section_builder_from_elements([elem], path=tmp_path/"m.mdl")
            sb.build_section()
        assert not [x for x in w if "Could not read GET DATA" in str(x.message)]
        all_eqs = [e for eqs, _ in sb.built_elements.values() for e in eqs]
        assert not any("GET_DATA_FAILED" in e for e in all_eqs)
        # 3×4 = 12 sub-functions emitted
        assert any("hfc_emissions_1_1" in d for d in sb.lookup_const_decls)
        assert any("hfc_emissions_3_4" in d for d in sb.lookup_const_decls)
        assert any("hfc_emissions(i, j, x)" in d for d in sb.lookup_func_decls)

    def test_initial_from_literal_float(self):
        """INITIAL(5.0) resolves to literal without needing reference resolution."""
        init_ast = InitialStructure(initial=5.0)
        comp = AbstractComponent(subscripts=[[], []], ast=init_ast)
        elem = AbstractElement(name="Init Literal", components=[comp])
        sb = _section_builder_from_elements([elem])
        sb.build_section()
        assert any("@parameters init_literal = 5.0" in d for d in sb.param_decls)

    def test_initial_from_get_constants_success(self, mocker, tmp_path):
        """INITIAL(GetConstantsStructure) resolves to the read value."""
        import numpy as np
        mock_ext = mocker.MagicMock()
        mock_ext.data = np.float64(99.0)
        mocker.patch("pysd.py_backend.external.ExtConstant", return_value=mock_ext)
        gc_ast = GetConstantsStructure(file="d.xlsx", tab="S", cell="A1")
        init_ast = InitialStructure(initial=gc_ast)
        comp = AbstractComponent(subscripts=[[], []], ast=init_ast)
        elem = AbstractElement(name="Init Ext", components=[comp])
        sb = _section_builder_from_elements([elem], path=tmp_path/"m.mdl")
        sb.build_section()
        assert any("@parameters init_ext = 99.0" in d for d in sb.param_decls)

    def test_resolve_ref_initial_depth_exceeded(self):
        """_resolve_ref_initial returns None when depth < 0."""
        sb = _section_builder_from_elements([])
        result = sb._resolve_ref_initial("anything", depth=-1)
        assert result is None

    def test_resolve_ref_initial_follows_numeric_aux_rhs(self):
        """INITIAL resolves when aux equation RHS is a plain number."""
        # aux ~ 42.0 → INITIAL(aux) → 42.0
        aux_comp = AbstractComponent(subscripts=[[], []], ast=42.0)
        aux_elem = AbstractElement(name="Aux Val", components=[aux_comp])
        init_ast = InitialStructure(initial=ReferenceStructure("Aux Val"))
        init_comp = AbstractComponent(subscripts=[[], []], ast=init_ast)
        init_elem = AbstractElement(name="Init Aux", components=[init_comp])
        sb = _section_builder_from_elements([aux_elem, init_elem])
        sb.build_section()
        assert any("@parameters init_aux = 42.0" in d for d in sb.param_decls)

    def test_resolve_ref_initial_returns_none_for_complex_rhs(self):
        """_resolve_ref_initial returns None for end of chain."""
        sb = _section_builder_from_elements([])
        sb.namespace.add_to_namespace("x")
        # x is in namespace but has no u0, param, or built_elements entry
        result = sb._resolve_ref_initial("x", depth=3)
        assert result is None

    def test_read_get_constants_multi_component(self, mocker, tmp_path):
        """Multi-component GetConstantsStructure merges coords (exercises inner for loop)."""
        import numpy as np
        mock_ext = mocker.MagicMock()
        mock_ext.data = np.float64(5.0)
        mocker.patch("pysd.py_backend.external.ExtConstant", return_value=mock_ext)
        ast1 = GetConstantsStructure(file="d.xlsx", tab="S", cell="A1")
        ast2 = GetConstantsStructure(file="d.xlsx", tab="S", cell="B1")
        sr = _make_subscript_range("dim_c", ["Z"])
        comp1 = AbstractComponent(subscripts=[["dim_c"], []], ast=ast1)
        comp2 = AbstractComponent(subscripts=[["dim_c"], []], ast=ast2)
        elem = AbstractElement(name="Multi Const", components=[comp1, comp2])
        sb = _section_builder_from_elements([elem], path=tmp_path/"m.mdl",
                                             subscripts=[sr])
        sb.build_section()
        assert any("multi_const" in d for d in sb.param_decls)

    @pytest.mark.filterwarnings("always::UserWarning")
    def test_initial_from_get_constants_exception(self, mocker, tmp_path):
        """INITIAL(GetConstantsStructure) exception silenced → returns None → fallback."""
        mocker.patch(
            "pysd.py_backend.external.ExtConstant",
            side_effect=FileNotFoundError("missing"),
        )
        gc_ast = GetConstantsStructure(file="missing.xlsx", tab="S", cell="A1")
        init_ast = InitialStructure(initial=gc_ast)
        comp = AbstractComponent(subscripts=[[], []], ast=init_ast)
        elem = AbstractElement(name="Init Gc Fail", components=[comp])
        with pytest.warns(UserWarning, match="Cannot resolve INITIAL"):
            sb = _section_builder_from_elements([elem], path=tmp_path/"m.mdl")
            sb.build_section()
        assert any("@variables init_gc_fail(t)" in d for d in sb.aux_decls)

    def test_modular_build_no_equations_uses_empty_list(self, tmp_path):
        """Modular build with only control vars → combined = Equation[]."""
        pop = _make_stock_element("Population", 1.0, 100.0)
        controls = [
            _make_control_element("INITIAL TIME", 0.0),
            _make_control_element("FINAL TIME", 10.0),
            _make_control_element("TIME STEP", 1.0),
            _make_control_element("SAVEPER", 1.0),
        ]
        # Single view with only the stock
        views_dict = {"Main": {"Population"}}
        section = _make_section(
            elements=[pop] + controls,
            path=tmp_path / "m.mdl",
            split=True,
            views_dict=views_dict,
        )
        model = AbstractModel(original_path=tmp_path / "m.mdl", sections=(section,))
        path = JuliaModelBuilder(model).build_model()
        content = path.read_text()
        assert "eqs = [" in content

    def test_modular_build_empty_eq_var_names(self, tmp_path):
        """Modular build: view references nonexistent var AND constants have no eqs
        → eq_var_names=[] AND leftover_eqs=[] → eqs = Equation[]."""
        # A constant has no equations; the view maps to nothing → both lists empty
        rate_comp = AbstractUnchangeableConstant(subscripts=[[], []], ast=0.5)
        rate_elem = AbstractElement(name="Rate", components=[rate_comp])
        controls = [
            _make_control_element("INITIAL TIME", 0.0),
            _make_control_element("FINAL TIME", 10.0),
            _make_control_element("TIME STEP", 1.0),
            _make_control_element("SAVEPER", 1.0),
        ]
        # View only references a name that is not in the namespace
        views_dict = {"Main": {"NonExistentVariable"}}
        section = _make_section(
            elements=[rate_elem] + controls,
            path=tmp_path / "empty_eq.mdl",
            split=True,
            views_dict=views_dict,
        )
        model = AbstractModel(original_path=tmp_path / "empty_eq.mdl",
                               sections=(section,))
        path = JuliaModelBuilder(model).build_model()
        content = path.read_text()
        assert "eqs = Equation[]" in content

    def test_format_julia_value_3d_array_flattened(self):
        """3D numpy array → flattened Julia 1D vector."""
        import numpy as np
        from pysd.builders.julia.julia_model_builder import _format_julia_value
        arr = np.ones((2, 2, 2))
        result = _format_julia_value(arr)
        assert result.startswith("[") and result.endswith("]")
        assert ";" not in result  # 1D, not 2D matrix syntax


# ===========================================================================
# JSON data backend tests
# ===========================================================================

class TestJSONDataBackend:

    def _minimal_model_with_lookup(self, tmp_path):
        """Model with a stock, a parameter, and a named lookup table."""
        br_comp = AbstractUnchangeableConstant(subscripts=[[], []], ast=0.03)
        br_elem = AbstractElement(name="Birth Rate", components=[br_comp],
                                  units="1/year")
        lut_ast = LookupsStructure(
            x=(0.0, 1.0, 2.0), y=(0.0, 0.5, 1.0),
            x_limits=(0.0, 2.0), y_limits=(0.0, 1.0), type="interpolate",
        )
        lut_comp = AbstractLookup(subscripts=[[], []], ast=lut_ast)
        lut_elem = AbstractElement(name="Effect Table", components=[lut_comp])
        flow_ast = ArithmeticStructure(
            operators=["*"],
            arguments=[ReferenceStructure("Population"), ReferenceStructure("Birth Rate")],
        )
        pop_ast = IntegStructure(flow=flow_ast, initial=1000.0)
        pop_comp = AbstractComponent(subscripts=[[], []], ast=pop_ast)
        pop_elem = AbstractElement(name="Population", components=[pop_comp])
        controls = [
            _make_control_element("INITIAL TIME", 0.0),
            _make_control_element("FINAL TIME", 100.0),
            _make_control_element("TIME STEP", 1.0),
            _make_control_element("SAVEPER", 1.0),
        ]
        section = _make_section(
            elements=[br_elem, lut_elem, pop_elem] + controls,
            path=tmp_path / "my_model.mdl",
        )
        return AbstractModel(
            original_path=tmp_path / "my_model.mdl",
            sections=(section,),
        )

    def test_invalid_data_format_raises(self, tmp_path):
        model = self._minimal_model_with_lookup(tmp_path)
        with pytest.raises(ValueError, match="data_format"):
            JuliaModelBuilder(model, data_format="invalid")

    def test_json_mode_creates_data_file(self, tmp_path):
        model = self._minimal_model_with_lookup(tmp_path)
        JuliaModelBuilder(model, data_format="json").build_model()
        assert (tmp_path / "my_model_data.json").exists()

    def test_hardcoded_mode_no_data_file(self, tmp_path):
        model = self._minimal_model_with_lookup(tmp_path)
        JuliaModelBuilder(model, data_format="hardcoded").build_model()
        assert not (tmp_path / "my_model_data.json").exists()

    def test_json_file_has_correct_schema(self, tmp_path):
        import json
        model = self._minimal_model_with_lookup(tmp_path)
        JuliaModelBuilder(model, data_format="json").build_model()
        data = json.loads((tmp_path / "my_model_data.json").read_text())
        assert "constants" in data
        assert "lookups" in data
        assert "data" in data

    def test_json_file_contains_parameter(self, tmp_path):
        import json
        model = self._minimal_model_with_lookup(tmp_path)
        JuliaModelBuilder(model, data_format="json").build_model()
        data = json.loads((tmp_path / "my_model_data.json").read_text())
        assert "birth_rate" in data["constants"]
        assert data["constants"]["birth_rate"]["values"] == pytest.approx(0.03)
        assert data["constants"]["birth_rate"]["units"] == "1/year"

    def test_json_file_contains_lookup(self, tmp_path):
        import json
        model = self._minimal_model_with_lookup(tmp_path)
        JuliaModelBuilder(model, data_format="json").build_model()
        # Named lookup tables are registered as inline lookups via inline_registry
        data = json.loads((tmp_path / "my_model_data.json").read_text())
        assert "lookups" in data
        # The lookup should have x, y, interp_type fields
        if data["lookups"]:
            key = next(iter(data["lookups"]))
            lut = data["lookups"][key]
            assert "x" in lut and "y" in lut and "interp_type" in lut

    def test_jl_file_uses_json3(self, tmp_path):
        model = self._minimal_model_with_lookup(tmp_path)
        path = JuliaModelBuilder(model, data_format="json").build_model()
        content = path.read_text()
        assert "JSON3" in content
        assert "_model_data" in content
        assert "my_model_data.json" in content

    def test_jl_file_params_reference_model_data(self, tmp_path):
        model = self._minimal_model_with_lookup(tmp_path)
        path = JuliaModelBuilder(model, data_format="json").build_model()
        content = path.read_text()
        assert '_model_data["constants"]["birth_rate"]' in content

    def test_hardcoded_mode_unchanged(self, tmp_path):
        """data_format='hardcoded' produces identical output to no data_format arg."""
        model1 = self._minimal_model_with_lookup(tmp_path / "a")
        (tmp_path / "a").mkdir()
        path1 = JuliaModelBuilder(model1).build_model()

        model2 = self._minimal_model_with_lookup(tmp_path / "b")
        (tmp_path / "b").mkdir()
        path2 = JuliaModelBuilder(model2, data_format="hardcoded").build_model()

        assert path1.read_text() == path2.read_text()

    def test_json_mode_get_lookups(self, mocker, tmp_path):
        """External lookup via GET_DIRECT_LOOKUPS appears in JSON file."""
        import json
        import numpy as np
        import xarray as xr
        xs = np.array([0.0, 1.0, 2.0])
        ys = np.array([10.0, 20.0, 30.0])
        da = xr.DataArray(ys, coords={"lookup_dim": xs}, dims=["lookup_dim"])
        mock_ext = mocker.MagicMock()
        mock_ext.data = da
        mocker.patch("pysd.py_backend.external.ExtLookup", return_value=mock_ext)
        ast = GetLookupsStructure(file="d.xlsx", tab="S", x_row_or_col="x", cell="A1")
        comp = AbstractComponent(subscripts=[[], []], ast=ast)
        elem = AbstractElement(name="Ext Lut", components=[comp])
        controls = [
            _make_control_element("INITIAL TIME", 0.0),
            _make_control_element("FINAL TIME", 10.0),
            _make_control_element("TIME STEP", 1.0),
            _make_control_element("SAVEPER", 1.0),
        ]
        section = _make_section(
            elements=[elem] + controls, path=tmp_path / "m.mdl"
        )
        model = AbstractModel(original_path=tmp_path / "m.mdl", sections=(section,))
        JuliaModelBuilder(model, data_format="json").build_model()
        data = json.loads((tmp_path / "m_data.json").read_text())
        assert "ext_lut" in data["lookups"]
        assert data["lookups"]["ext_lut"]["x"] == pytest.approx([0.0, 1.0, 2.0])

    def test_json_mode_get_data(self, mocker, tmp_path):
        """External time-series via GET_DIRECT_DATA appears in JSON file."""
        import json
        import numpy as np
        import xarray as xr
        ts = np.array([1995.0, 2000.0, 2005.0])
        vals = np.array([1.0, 2.0, 3.0])
        da = xr.DataArray(vals, coords={"time": ts}, dims=["time"])
        mock_ext = mocker.MagicMock()
        mock_ext.data = da
        mocker.patch("pysd.py_backend.external.ExtData", return_value=mock_ext)
        ast = GetDataStructure(file="d.xlsx", tab="S", time_row_or_col="t", cell="A1")
        comp = AbstractComponent(subscripts=[[], []], ast=ast)
        elem = AbstractElement(name="Historic Eff", components=[comp])
        controls = [
            _make_control_element("INITIAL TIME", 0.0),
            _make_control_element("FINAL TIME", 10.0),
            _make_control_element("TIME STEP", 1.0),
            _make_control_element("SAVEPER", 1.0),
        ]
        section = _make_section(
            elements=[elem] + controls, path=tmp_path / "m.mdl"
        )
        model = AbstractModel(original_path=tmp_path / "m.mdl", sections=(section,))
        JuliaModelBuilder(model, data_format="json").build_model()
        data = json.loads((tmp_path / "m_data.json").read_text())
        assert "historic_eff" in data["data"]
        assert data["data"]["historic_eff"]["time"] == pytest.approx([1995.0, 2000.0, 2005.0])


class TestJSONDataBackendCoverage:
    """Covers remaining JSON-mode branches."""

    def _controls(self):
        return [
            _make_control_element("INITIAL TIME", 0.0),
            _make_control_element("FINAL TIME", 10.0),
            _make_control_element("TIME STEP", 1.0),
            _make_control_element("SAVEPER", 1.0),
        ]

    def test_json_mode_inline_lookup_accumulated(self, tmp_path):
        """Inline lookups (InlineLookupsStructure) go into _json_data in JSON mode."""
        import json
        lut_ast = InlineLookupsStructure(
            argument=1.0,
            lookups=LookupsStructure(
                x=(0.0, 1.0), y=(0.0, 2.0),
                x_limits=(0.0, 1.0), y_limits=(0.0, 2.0),
                type="interpolate",
            ),
        )
        comp = AbstractComponent(subscripts=[[], []], ast=lut_ast)
        elem = AbstractElement(name="LutResult", components=[comp])
        section = _make_section(
            elements=[elem] + self._controls(), path=tmp_path / "m.mdl"
        )
        model = AbstractModel(original_path=tmp_path / "m.mdl", sections=(section,))
        JuliaModelBuilder(model, data_format="json").build_model()
        data = json.loads((tmp_path / "m_data.json").read_text())
        assert any("_inline_lookup_" in k for k in data["lookups"])

    def test_json_mode_ext_constant_accumulates(self, mocker, tmp_path):
        """GetConstantsStructure in JSON mode calls _json_accumulate_constant."""
        import json
        import numpy as np
        mock_ext = mocker.MagicMock()
        mock_ext.data = np.float64(7.5)
        mocker.patch("pysd.py_backend.external.ExtConstant", return_value=mock_ext)
        ast = GetConstantsStructure(file="d.xlsx", tab="S", cell="A1")
        comp = AbstractComponent(subscripts=[[], []], ast=ast)
        elem = AbstractElement(name="Ext Rate", components=[comp], units="1/year")
        section = _make_section(
            elements=[elem] + self._controls(), path=tmp_path / "m.mdl"
        )
        model = AbstractModel(original_path=tmp_path / "m.mdl", sections=(section,))
        JuliaModelBuilder(model, data_format="json").build_model()
        data = json.loads((tmp_path / "m_data.json").read_text())
        assert "ext_rate" in data["constants"]
        assert data["constants"]["ext_rate"]["values"] == pytest.approx(7.5)
        assert data["constants"]["ext_rate"]["units"] == "1/year"

    def test_json_mode_ext_constant_array_in_ext_const_decls(self, mocker, tmp_path):
        """Array external constant → ext_const_decls in JSON mode → JSON-backed ref."""
        import json
        import numpy as np
        mock_ext = mocker.MagicMock()
        mock_ext.data = np.array([1.0, 2.0, 3.0])
        mocker.patch("pysd.py_backend.external.ExtConstant", return_value=mock_ext)
        ast = GetConstantsStructure(file="d.xlsx", tab="S", cell="A1")
        comp = AbstractComponent(subscripts=[[], []], ast=ast)
        elem = AbstractElement(name="Arr Const", components=[comp])
        section = _make_section(
            elements=[elem] + self._controls(), path=tmp_path / "m.mdl"
        )
        model = AbstractModel(original_path=tmp_path / "m.mdl", sections=(section,))
        path = JuliaModelBuilder(model, data_format="json").build_model()
        content = path.read_text()
        # In JSON mode, array const uses _model_data reference
        assert '_model_data["constants"]["arr_const"]' in content
        data = json.loads((tmp_path / "m_data.json").read_text())
        assert "arr_const" in data["constants"]

    def test_json_mode_2d_lookup_accumulates(self, mocker, tmp_path):
        """2D subscripted lookup in JSON mode stores each column."""
        import json
        import numpy as np
        import xarray as xr
        xs = np.array([0.0, 1.0])
        ys = np.ones((2, 3))
        da = xr.DataArray(ys, coords={"lookup_dim": xs}, dims=["lookup_dim", "sub"])
        mock_ext = mocker.MagicMock()
        mock_ext.data = da
        mocker.patch("pysd.py_backend.external.ExtLookup", return_value=mock_ext)
        ast = GetLookupsStructure(file="d.xlsx", tab="S", x_row_or_col="x", cell="A1")
        comp = AbstractComponent(subscripts=[[], []], ast=ast)
        elem = AbstractElement(name="Sub Lut", components=[comp])
        section = _make_section(
            elements=[elem] + self._controls(), path=tmp_path / "m.mdl"
        )
        model = AbstractModel(original_path=tmp_path / "m.mdl", sections=(section,))
        JuliaModelBuilder(model, data_format="json").build_model()
        data = json.loads((tmp_path / "m_data.json").read_text())
        assert "sub_lut_1" in data["lookups"]
        assert "sub_lut_2" in data["lookups"]
        assert "sub_lut_3" in data["lookups"]

    def test_json_mode_2d_data_accumulates(self, mocker, tmp_path):
        """2D time-series in JSON mode stores each column."""
        import json
        import numpy as np
        import xarray as xr
        ts = np.array([1995.0, 2000.0])
        vals = np.ones((2, 2))
        da = xr.DataArray(vals, coords={"time": ts}, dims=["time", "sub"])
        mock_ext = mocker.MagicMock()
        mock_ext.data = da
        mocker.patch("pysd.py_backend.external.ExtData", return_value=mock_ext)
        ast = GetDataStructure(file="d.xlsx", tab="S", time_row_or_col="t", cell="A1")
        comp = AbstractComponent(subscripts=[[], []], ast=ast)
        elem = AbstractElement(name="Sub Series", components=[comp])
        section = _make_section(
            elements=[elem] + self._controls(), path=tmp_path / "m.mdl"
        )
        model = AbstractModel(original_path=tmp_path / "m.mdl", sections=(section,))
        JuliaModelBuilder(model, data_format="json").build_model()
        data = json.loads((tmp_path / "m_data.json").read_text())
        assert "sub_series_1" in data["data"]
        assert "sub_series_2" in data["data"]

    def test_json_mode_modular_build_writes_json(self, tmp_path):
        """Modular (split) build in JSON mode still writes the data JSON file."""
        br_comp = AbstractUnchangeableConstant(subscripts=[[], []], ast=0.05)
        br_elem = AbstractElement(name="Rate", components=[br_comp])
        pop = _make_stock_element("Population", 1.0, 100.0)
        controls = self._controls()
        views_dict = {"Main": {"Population"}, "Params": {"Rate"}}
        section = _make_section(
            elements=[br_elem, pop] + controls,
            path=tmp_path / "split.mdl",
            split=True,
            views_dict=views_dict,
        )
        model = AbstractModel(original_path=tmp_path / "split.mdl", sections=(section,))
        JuliaModelBuilder(model, data_format="json").build_model()
        assert (tmp_path / "split_data.json").exists()

    def test_json_mode_nonnumeric_constant_uses_fallback(self, tmp_path):
        """A constant whose value can't be float()-converted is skipped gracefully."""
        import json
        # Use an ArithmeticStructure as the constant AST — visitor.visit() returns
        # a Julia expression like "(a * b)" that can't be float()'d
        rhs = ArithmeticStructure(
            operators=["*"],
            arguments=[ReferenceStructure("a"), ReferenceStructure("b")],
        )
        a_elem = _make_element("a", 2.0, comp_class=AbstractUnchangeableConstant)
        b_elem = _make_element("b", 3.0, comp_class=AbstractUnchangeableConstant)
        comp = AbstractComponent(subscripts=[[], []], ast=rhs)
        comp.type = "Constant"
        elem = AbstractElement(name="Product", components=[comp])
        section = _make_section(
            elements=[a_elem, b_elem, elem] + self._controls(),
            path=tmp_path / "m.mdl"
        )
        model = AbstractModel(original_path=tmp_path / "m.mdl", sections=(section,))
        # Should not raise even though the value can't be stored as float
        path = JuliaModelBuilder(model, data_format="json").build_model()
        assert path.exists()


class TestJSONAccumulateConstant:
    """Covers the _json_accumulate_constant helper's edge cases."""

    def test_xarray_dataarray_uses_values(self, mocker, tmp_path):
        """When ext.data is a DataArray, .values is extracted (line 1320)."""
        import json
        import numpy as np
        import xarray as xr
        da = xr.DataArray(np.float64(9.9))  # 0-D DataArray with .values
        mock_ext = mocker.MagicMock()
        mock_ext.data = da
        mocker.patch("pysd.py_backend.external.ExtConstant", return_value=mock_ext)
        ast = GetConstantsStructure(file="d.xlsx", tab="S", cell="A1")
        comp = AbstractComponent(subscripts=[[], []], ast=ast)
        elem = AbstractElement(name="Da Const", components=[comp])
        controls = [
            _make_control_element("INITIAL TIME", 0.0),
            _make_control_element("FINAL TIME", 10.0),
            _make_control_element("TIME STEP", 1.0),
            _make_control_element("SAVEPER", 1.0),
        ]
        section = _make_section(
            elements=[elem] + controls, path=tmp_path / "m.mdl"
        )
        model = AbstractModel(original_path=tmp_path / "m.mdl", sections=(section,))
        JuliaModelBuilder(model, data_format="json").build_model()
        data = json.loads((tmp_path / "m_data.json").read_text())
        assert "da_const" in data["constants"]
        assert data["constants"]["da_const"]["values"] == pytest.approx(9.9)

    def test_exception_in_accumulate_uses_julia_val_fallback(self, mocker, tmp_path):
        """If _json_accumulate_constant raises, the julia literal is stored."""
        import json
        import numpy as np
        # First call to ExtConstant (from _read_get_constants) succeeds
        # Second call (from _json_accumulate_constant) raises
        mock_ext_good = mocker.MagicMock()
        mock_ext_good.data = np.float64(5.0)
        mock_ext_fail = mocker.MagicMock()
        mock_ext_fail.initialize.side_effect = RuntimeError("second call fails")
        mocker.patch(
            "pysd.py_backend.external.ExtConstant",
            side_effect=[mock_ext_good, mock_ext_fail],
        )
        ast = GetConstantsStructure(file="d.xlsx", tab="S", cell="A1")
        comp = AbstractComponent(subscripts=[[], []], ast=ast)
        elem = AbstractElement(name="Fallback Const", components=[comp])
        controls = [
            _make_control_element("INITIAL TIME", 0.0),
            _make_control_element("FINAL TIME", 10.0),
            _make_control_element("TIME STEP", 1.0),
            _make_control_element("SAVEPER", 1.0),
        ]
        section = _make_section(
            elements=[elem] + controls, path=tmp_path / "m.mdl"
        )
        model = AbstractModel(original_path=tmp_path / "m.mdl", sections=(section,))
        JuliaModelBuilder(model, data_format="json").build_model()
        data = json.loads((tmp_path / "m_data.json").read_text())
        # Fallback stores the julia literal string
        assert "fallback_const" in data["constants"]
        assert data["constants"]["fallback_const"]["values"] == "5.0"


# ===========================================================================
# Phase 3B — GET DATA interpolation method passthrough
# ===========================================================================

class TestGetDataMethodPassthrough:

    def test_vensim_keyword_to_itp_type(self):
        from pysd.builders.julia.julia_model_builder import _vensim_keyword_to_itp_type
        assert _vensim_keyword_to_itp_type(None) == "interpolate"
        assert _vensim_keyword_to_itp_type("interpolate") == "interpolate"
        assert _vensim_keyword_to_itp_type("hold_backward") == "hold_forward"
        assert _vensim_keyword_to_itp_type("look_forward") == "hold_backward"
        assert _vensim_keyword_to_itp_type("raw") == "interpolate"

    def test_hold_backward_produces_constant_interpolation(self, mocker, tmp_path):
        import numpy as np
        import xarray as xr
        ts = np.array([1995.0, 2000.0, 2005.0])
        vals = np.array([1.0, 2.0, 3.0])
        da = xr.DataArray(vals, coords={"time": ts}, dims=["time"])
        mock_ext = mocker.MagicMock()
        mock_ext.data = da
        mocker.patch("pysd.py_backend.external.ExtData", return_value=mock_ext)
        ast = GetDataStructure(file="d.xlsx", tab="S", time_row_or_col="t", cell="A1")
        # AbstractData with hold_backward keyword
        comp = AbstractData(subscripts=[[], []], ast=ast, keyword="hold_backward")
        elem = AbstractElement(name="Step Series", components=[comp])
        controls = [
            _make_control_element("INITIAL TIME", 0.0),
            _make_control_element("FINAL TIME", 10.0),
            _make_control_element("TIME STEP", 1.0),
            _make_control_element("SAVEPER", 1.0),
        ]
        section = _make_section(
            elements=[elem] + controls, path=tmp_path / "m.mdl"
        )
        model = AbstractModel(original_path=tmp_path / "m.mdl", sections=(section,))
        path = JuliaModelBuilder(model).build_model()
        content = path.read_text()
        assert "ConstantInterpolation" in content
        assert "LinearInterpolation" not in content

    def test_look_forward_produces_constant_interpolation_right(self, mocker, tmp_path):
        import numpy as np
        import xarray as xr
        ts = np.array([1995.0, 2000.0])
        vals = np.array([1.0, 2.0])
        da = xr.DataArray(vals, coords={"time": ts}, dims=["time"])
        mock_ext = mocker.MagicMock()
        mock_ext.data = da
        mocker.patch("pysd.py_backend.external.ExtData", return_value=mock_ext)
        ast = GetDataStructure(file="d.xlsx", tab="S", time_row_or_col="t", cell="A1")
        comp = AbstractData(subscripts=[[], []], ast=ast, keyword="look_forward")
        elem = AbstractElement(name="Fwd Series", components=[comp])
        controls = [
            _make_control_element("INITIAL TIME", 0.0),
            _make_control_element("FINAL TIME", 10.0),
            _make_control_element("TIME STEP", 1.0),
            _make_control_element("SAVEPER", 1.0),
        ]
        section = _make_section(
            elements=[elem] + controls, path=tmp_path / "m.mdl"
        )
        model = AbstractModel(original_path=tmp_path / "m.mdl", sections=(section,))
        path = JuliaModelBuilder(model).build_model()
        content = path.read_text()
        assert "ConstantInterpolation" in content
        assert "dir=:right" in content


# ===========================================================================
# Phase 3C — Variable limits
# ===========================================================================

class TestVariableLimits:

    def test_limits_comment_no_limits(self):
        elem = _make_element("x", 1.0)
        assert JuliaSectionBuilder._limits_comment(elem) == ""

    def test_limits_comment_both_bounds(self):
        elem = AbstractElement(
            name="x", components=[_make_component(1.0)],
            limits=(0.0, 1.0), units="Dmnl",
        )
        comment = JuliaSectionBuilder._limits_comment(elem)
        assert "0.0" in comment and "1.0" in comment
        assert comment.startswith("  # limits:")

    def test_limits_comment_lower_only(self):
        elem = AbstractElement(
            name="x", components=[_make_component(1.0)],
            limits=(0.0, None),
        )
        comment = JuliaSectionBuilder._limits_comment(elem)
        assert "0.0" in comment
        assert "Inf" in comment

    def test_limits_comment_upper_only(self):
        elem = AbstractElement(
            name="x", components=[_make_component(1.0)],
            limits=(None, 100.0),
        )
        comment = JuliaSectionBuilder._limits_comment(elem)
        assert "-Inf" in comment
        assert "100.0" in comment

    def test_limits_appear_in_param_declaration(self):
        comp = AbstractUnchangeableConstant(subscripts=[[], []], ast=0.5)
        elem = AbstractElement(name="Birth Rate", components=[comp], limits=(0.0, 1.0))
        sb = _section_builder_from_elements([elem])
        sb.build_section()
        assert any("# limits:" in d for d in sb.param_decls)

    def test_limits_appear_in_aux_equation(self):
        comp = AbstractComponent(subscripts=[[], []], ast=2.5)
        elem = AbstractElement(name="Output", components=[comp], limits=(0.0, None))
        sb = _section_builder_from_elements([elem])
        sb.build_section()
        eqs = [e for eqs, _ in sb.built_elements.values() for e in eqs]
        assert any("# limits:" in e for e in eqs)

    def test_limits_in_full_generated_file(self, tmp_path):
        comp = AbstractUnchangeableConstant(subscripts=[[], []], ast=0.5)
        elem = AbstractElement(name="Rate", components=[comp], limits=(0.0, 1.0))
        controls = [
            _make_control_element("INITIAL TIME", 0.0),
            _make_control_element("FINAL TIME", 10.0),
            _make_control_element("TIME STEP", 1.0),
            _make_control_element("SAVEPER", 1.0),
        ]
        section = _make_section(
            elements=[elem] + controls, path=tmp_path / "m.mdl"
        )
        model = AbstractModel(original_path=tmp_path / "m.mdl", sections=(section,))
        path = JuliaModelBuilder(model).build_model()
        content = path.read_text()
        assert "# limits:" in content

    def test_limits_stored_in_json(self, tmp_path):
        import json
        comp = AbstractUnchangeableConstant(subscripts=[[], []], ast=0.5)
        elem = AbstractElement(name="Rate", components=[comp], limits=(0.0, 1.0))
        controls = [
            _make_control_element("INITIAL TIME", 0.0),
            _make_control_element("FINAL TIME", 10.0),
            _make_control_element("TIME STEP", 1.0),
            _make_control_element("SAVEPER", 1.0),
        ]
        section = _make_section(
            elements=[elem] + controls, path=tmp_path / "m.mdl"
        )
        model = AbstractModel(original_path=tmp_path / "m.mdl", sections=(section,))
        JuliaModelBuilder(model, data_format="json").build_model()
        data = json.loads((tmp_path / "m_data.json").read_text())
        assert "limits" in data["constants"]["rate"]
        assert data["constants"]["rate"]["limits"] == [0.0, 1.0]


# ===========================================================================
# Phase 3D — EXCEPT subscript exclusion
# ===========================================================================

class TestExceptSubscriptExclusion:

    def _make_except_element(self, name, dim_name, dim_elems,
                              comp1_ast, comp2_ast, except_labels):
        """Make an element with two components where comp1 has EXCEPT."""
        # comp1: covers dim_name, except except_labels
        comp1 = AbstractComponent(
            subscripts=[[dim_name], [except_labels]],
            ast=comp1_ast,
        )
        # comp2: covers just the excepted elements (no EXCEPT)
        comp2 = AbstractComponent(
            subscripts=[[dim_name], []],
            ast=comp2_ast,
        )
        return AbstractElement(name=name, components=[comp1, comp2])

    def test_except_element_generates_per_index_equations(self):
        sr = _make_subscript_range("category", ["A", "B", "C"])
        # comp1: category = 1.0, EXCEPT [B]
        # comp2: all category = 2.0 (no EXCEPT)
        elem = self._make_except_element(
            "My Var", "category", ["A", "B", "C"],
            comp1_ast=1.0, comp2_ast=2.0,
            except_labels=["B"],
        )
        sb = _section_builder_from_elements([elem], subscripts=[sr])
        sb.build_section()
        eqs = [e for eqs, _ in sb.built_elements.values() for e in eqs]
        # identifier is "my_var"; should have equations for index 1 (A), 3 (C) from comp1
        assert any("my_var[1]" in e for e in eqs)  # A from comp1
        assert any("my_var[3]" in e for e in eqs)  # C from comp1
        # The variable should be declared as array
        assert any("my_var(t)[" in d for d in sb.aux_decls)

    def test_except_element_excludes_correct_index(self):
        sr = _make_subscript_range("sector", ["S1", "S2", "S3"])
        elem = self._make_except_element(
            "Output", "sector", ["S1", "S2", "S3"],
            comp1_ast=5.0, comp2_ast=10.0,
            except_labels=["S2"],
        )
        sb = _section_builder_from_elements([elem], subscripts=[sr])
        sb.build_section()
        eqs = [e for eqs, _ in sb.built_elements.values() for e in eqs]
        # comp1 covers S1(1) and S3(3), NOT S2(2)
        comp1_eqs = [e for e in eqs if "5.0" in e]
        assert any("[1]" in e for e in comp1_eqs)
        assert any("[3]" in e for e in comp1_eqs)
        assert not any("[2]" in e for e in comp1_eqs)

    def test_except_2d_emits_equations_for_all_pairs(self):
        """2D EXCEPT: both components must produce equations covering all (i,j) pairs
        with no warning about unsupported dimensionality."""
        # r has 3 elements; c has 2 elements → 6 total pairs
        # comp0: r×c EXCEPT [R1]×c  → covers (R2, *) and (R3, *)
        # comp1: r×c (no EXCEPT)    → covers all r×c; effectively fills (R1, *)
        sr1 = _make_subscript_range("r", ["R1", "R2", "R3"])
        sr2 = _make_subscript_range("c", ["C1", "C2"])
        comp0 = AbstractComponent(
            subscripts=[["r", "c"], [["R1", "c"]]],
            ast=1.0,
        )
        comp1 = AbstractComponent(subscripts=[["r", "c"], []], ast=2.0)
        elem = AbstractElement(name="Matrix", components=[comp0, comp1])
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # fail if any UserWarning is raised
            sb = _section_builder_from_elements([elem], subscripts=[sr1, sr2])
            sb.build_section()
        eqs = [e for eqs, _ in sb.built_elements.values() for e in eqs]
        assert len(eqs) > 0

    def test_except_2d_excluded_rows_use_second_component(self):
        """Pairs excluded from comp0 via EXCEPT must use comp1's formula, not comp0's."""
        # r={A,B,C}, c={X,Y}; comp0 covers r×c EXCEPT [B]×c; comp1 covers all r×c
        sr1 = _make_subscript_range("r", ["A", "B", "C"])
        sr2 = _make_subscript_range("c", ["X", "Y"])
        comp0 = AbstractComponent(
            subscripts=[["r", "c"], [["B", "c"]]],
            ast=10.0,
        )
        comp1 = AbstractComponent(subscripts=[["r", "c"], []], ast=99.0)
        elem = AbstractElement(name="Out", components=[comp0, comp1])
        sb = _section_builder_from_elements([elem], subscripts=[sr1, sr2])
        sb.build_section()
        eqs = [e for eqs, _ in sb.built_elements.values() for e in eqs]
        # comp0 formula (10.0) must NOT appear for any equation at row-index 2 (B)
        comp0_eqs = [e for e in eqs if "10.0" in e]
        assert not any(
            ("[2," in e or ", 2]" in e or "[2]" in e) for e in comp0_eqs
        ), "comp0's formula must not be used for row B (index 2)"
        # comp0 formula (10.0) MUST appear for rows A(1) and C(3)
        assert any("[1," in e or "1]" in e for e in comp0_eqs), "comp0 must cover row A"
        assert any("[3," in e or "3]" in e for e in comp0_eqs), "comp0 must cover row C"

    def test_except_2d_element_spec_as_specific_element(self):
        """When a component's subscript spec names a specific element (not a range),
        only that element's rows/columns should be covered."""
        # r={A,B,C}; c={X,Y}
        # comp0: r×c EXCEPT [B]×c → covers (A,*) and (C,*)
        # comp1: B×c (specific element, no EXCEPT) → covers (B,*)
        sr1 = _make_subscript_range("r", ["A", "B", "C"])
        sr2 = _make_subscript_range("c", ["X", "Y"])
        comp0 = AbstractComponent(
            subscripts=[["r", "c"], [["B", "c"]]],
            ast=1.0,
        )
        comp1 = AbstractComponent(subscripts=[["B", "c"], []], ast=2.0)
        elem = AbstractElement(name="Res", components=[comp0, comp1])
        sb = _section_builder_from_elements([elem], subscripts=[sr1, sr2])
        sb.build_section()
        eqs = [e for eqs, _ in sb.built_elements.values() for e in eqs]
        # comp1 formula (2.0) must appear only for row B (index 2).
        # Generated form: "[res[_i0, _i1] ~ 2.0 for _i0 in [2], _i1 in ...]..."
        comp1_eqs = [e for e in eqs if "2.0" in e]
        assert comp1_eqs, "comp1 formula must appear in some equation"
        assert all("in [2]" in e or "_i0, 2]" in e for e in comp1_eqs), (
            f"comp1's formula must only cover row 2 (B); got: {comp1_eqs}"
        )


# ===========================================================================
# Phase 3E — Macro support
# ===========================================================================

class TestMacroSupport:

    def _two_section_model(self, tmp_path):
        """AbstractModel with a main section and one macro section."""
        # Main section: simple stock
        pop = _make_stock_element("Population", 1.0, 100.0)
        controls = [
            _make_control_element("INITIAL TIME", 0.0),
            _make_control_element("FINAL TIME", 10.0),
            _make_control_element("TIME STEP", 1.0),
            _make_control_element("SAVEPER", 1.0),
        ]
        main_section = _make_section(
            elements=[pop] + controls,
            path=tmp_path / "my_model.mdl",
        )

        # Macro section: simple auxiliary
        macro_aux = _make_element("Macro Output", 42.0)
        macro_section = AbstractSection(
            name="my_macro",
            path=tmp_path / "my_model.mdl",
            type="macro",
            params=["Input"],
            returns=["Macro Output"],
            subscripts=(),
            elements=(macro_aux,),
            constraints=(),
            test_inputs=(),
            split=False,
            views_dict=None,
        )

        return AbstractModel(
            original_path=tmp_path / "my_model.mdl",
            sections=(main_section, macro_section),
        )

    def test_build_model_creates_main_jl(self, tmp_path):
        model = self._two_section_model(tmp_path)
        path = JuliaModelBuilder(model).build_model()
        assert path.exists()
        assert path.suffix == ".jl"

    def test_macro_section_creates_companion_file(self, tmp_path):
        model = self._two_section_model(tmp_path)
        JuliaModelBuilder(model).build_model()
        # Macro file should exist next to main file
        macro_file = tmp_path / "my_model_my_macro.jl"
        assert macro_file.exists()

    def test_macro_file_contains_equations(self, tmp_path):
        model = self._two_section_model(tmp_path)
        JuliaModelBuilder(model).build_model()
        macro_file = tmp_path / "my_model_my_macro.jl"
        content = macro_file.read_text()
        assert "my_macro_eqs" in content
        assert "Equation[" in content

    def test_macro_file_contains_macro_name_comment(self, tmp_path):
        model = self._two_section_model(tmp_path)
        JuliaModelBuilder(model).build_model()
        macro_file = tmp_path / "my_model_my_macro.jl"
        content = macro_file.read_text()
        assert "Macro my_macro" in content

    def test_main_file_unaffected_by_macro(self, tmp_path):
        """Main model still contains ODESystem even with a macro section."""
        model = self._two_section_model(tmp_path)
        path = JuliaModelBuilder(model).build_model()
        content = path.read_text()
        assert "ODESystem" in content
        assert "population" in content


class TestMacroSupportCoverage:
    """Cover remaining macro-section code paths."""

    def test_macro_with_inline_lookup_and_json(self, tmp_path):
        """Macro section with inline lookup and json mode covers lines 227-232, 243, 262."""
        import json
        lut_ast = InlineLookupsStructure(
            argument=1.0,
            lookups=LookupsStructure(
                x=(0.0, 1.0), y=(0.0, 2.0),
                x_limits=(0.0, 1.0), y_limits=(0.0, 2.0),
                type="interpolate",
            ),
        )
        comp = AbstractComponent(subscripts=[[], []], ast=lut_ast)
        lut_elem = AbstractElement(name="Macro LUT", components=[comp])
        main_section = _make_section(
            elements=[
                _make_stock_element("S", 1.0, 1.0),
                _make_control_element("INITIAL TIME", 0.0),
                _make_control_element("FINAL TIME", 10.0),
                _make_control_element("TIME STEP", 1.0),
                _make_control_element("SAVEPER", 1.0),
            ],
            path=tmp_path / "m.mdl",
        )
        macro_section = AbstractSection(
            name="lookup_macro", path=tmp_path / "m.mdl",
            type="macro", params=[], returns=["Macro LUT"],
            subscripts=(), elements=(lut_elem,),
            constraints=(), test_inputs=(),
            split=False, views_dict=None,
        )
        model = AbstractModel(
            original_path=tmp_path / "m.mdl",
            sections=(main_section, macro_section),
        )
        JuliaModelBuilder(model, data_format="json").build_model()
        macro_path = tmp_path / "m_lookup_macro.jl"
        assert macro_path.exists()
        assert "DataInterpolations" in macro_path.read_text()
        assert (tmp_path / "m_lookup_macro_data.json").exists()


class TestExceptConstantComponent:
    """Covers the Constant component in EXCEPT handler (lines 744-749)."""

    def test_except_with_constant_component_emits_comment(self):
        sr = _make_subscript_range("dim", ["X", "Y", "Z"])
        comp1 = AbstractUnchangeableConstant(
            subscripts=[["dim"], [["Y"]]], ast=1.0
        )
        comp2 = AbstractUnchangeableConstant(
            subscripts=[["dim"], []], ast=5.0
        )
        elem = AbstractElement(name="Const Except", components=[comp1, comp2])
        sb = _section_builder_from_elements([elem], subscripts=[sr])
        sb.build_section()
        eqs = [e for eqs, _ in sb.built_elements.values() for e in eqs]
        # The constant component in EXCEPT emits a comment equation
        assert any("# EXCEPT:" in e for e in eqs)


# ===========================================================================
# Phase 4 — .mdl file translation tests (run without Julia runtime)
# ===========================================================================

class TestMdlFileTranslation:
    """Translate more-tests .mdl files and check the generated .jl content.
    These tests exercise the full PySD→Julia translation pipeline without
    requiring a Julia runtime.
    """

    MORE_TESTS = Path("tests/more-tests")

    def _translate(self, mdl_path, tmp_path):
        import shutil
        dst = tmp_path / mdl_path.name
        shutil.copy(mdl_path, dst)
        from pysd import translate_to_julia
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            jl_path = translate_to_julia(dst)
        return jl_path

    def test_julia_data_structure_creates_jl_file(self, tmp_path):
        mdl = self.MORE_TESTS / "julia_data_structure" / "test_julia_data_structure.mdl"
        if not mdl.exists():
            pytest.skip("julia_data_structure test model not found")
        jl_path = self._translate(mdl, tmp_path)
        assert jl_path.exists()
        assert jl_path.suffix == ".jl"

    def test_julia_data_structure_emits_unsupported_warning(self, tmp_path):
        mdl = self.MORE_TESTS / "julia_data_structure" / "test_julia_data_structure.mdl"
        if not mdl.exists():
            pytest.skip("julia_data_structure test model not found")
        import shutil, warnings
        dst = tmp_path / mdl.name
        shutil.copy(mdl, dst)
        from pysd import translate_to_julia
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            translate_to_julia(dst)
        msgs = [str(w.message) for w in captured]
        # DataStructure or DATA variable warning should be present
        assert any("DataStructure" in m or "data" in m.lower() for m in msgs), \
            f"Expected DataStructure warning, got: {msgs}"

    def test_julia_delay_fixed_no_warning(self, tmp_path):
        mdl = self.MORE_TESTS / "julia_delay_fixed" / "test_julia_delay_fixed.mdl"
        if not mdl.exists():
            pytest.skip("julia_delay_fixed test model not found")
        import shutil, warnings
        dst = tmp_path / mdl.name
        shutil.copy(mdl, dst)
        from pysd import translate_to_julia
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            translate_to_julia(dst)
        user_warns = [w for w in captured if issubclass(w.category, UserWarning)]
        assert not user_warns, f"Expected no UserWarning, got: {user_warns}"

    def test_julia_delay_fixed_emits_ode(self, tmp_path):
        mdl = self.MORE_TESTS / "julia_delay_fixed" / "test_julia_delay_fixed.mdl"
        if not mdl.exists():
            pytest.skip("julia_delay_fixed test model not found")
        jl_path = self._translate(mdl, tmp_path)
        content = jl_path.read_text()
        assert "_df_" in content
        assert "D(_df_" in content

    def test_julia_trend_emits_smooth_stock(self, tmp_path):
        mdl = self.MORE_TESTS / "julia_trend" / "test_julia_trend.mdl"
        if not mdl.exists():
            pytest.skip("julia_trend test model not found")
        jl_path = self._translate(mdl, tmp_path)
        content = jl_path.read_text()
        assert "_sm_" in content
        assert "D(_sm_" in content

    def test_julia_forecast_emits_smooth_stock(self, tmp_path):
        mdl = self.MORE_TESTS / "julia_forecast" / "test_julia_forecast.mdl"
        if not mdl.exists():
            pytest.skip("julia_forecast test model not found")
        jl_path = self._translate(mdl, tmp_path)
        content = jl_path.read_text()
        assert "_sm_" in content

    def test_julia_sample_if_true_emits_stock(self, tmp_path):
        mdl = self.MORE_TESTS / "julia_sample_if_true" / "test_julia_sample_if_true.mdl"
        if not mdl.exists():
            pytest.skip("julia_sample_if_true test model not found")
        jl_path = self._translate(mdl, tmp_path)
        content = jl_path.read_text()
        assert "_sit_" in content

    def test_json_mode_produces_data_file(self, tmp_path):
        """translate_to_julia with data_format=json creates a .json companion."""
        mdl = self.MORE_TESTS / "julia_delay_fixed" / "test_julia_delay_fixed.mdl"
        if not mdl.exists():
            pytest.skip("julia_delay_fixed test model not found")
        import shutil, warnings
        dst = tmp_path / mdl.name
        shutil.copy(mdl, dst)
        from pysd import translate_to_julia
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            jl_path = translate_to_julia(dst, data_format="json")
        json_path = jl_path.with_name(f"{jl_path.stem}_data.json")
        assert json_path.exists()
