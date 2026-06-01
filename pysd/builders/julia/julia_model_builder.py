"""
Translates a PySD AbstractModel into a standalone Julia file that uses
ModelingToolkit.jl.  The generated file requires no PySD or Python at runtime.

Entry point::

    from pysd.builders.julia.julia_model_builder import JuliaModelBuilder
    path = JuliaModelBuilder(abstract_model).build_model()
"""
from __future__ import annotations

import itertools
import re
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from warnings import warn

from pysd._version import __version__
from pysd.translators.structures.abstract_model import (
    AbstractComponent,
    AbstractControlElement,
    AbstractData,
    AbstractElement,
    AbstractLookup,
    AbstractModel,
    AbstractSection,
    AbstractUnchangeableConstant,
)
from pysd.translators.structures.abstract_expressions import (
    AllocateAvailableStructure,
    AllocateByPriorityStructure,
    DataStructure,
    DelayFixedStructure,
    DelayNStructure,
    DelayStructure,
    ForecastStructure,
    GetConstantsStructure,
    GetDataStructure,
    GetLookupsStructure,
    InitialStructure,
    IntegStructure,
    LookupsStructure,
    ReferenceStructure,
    SampleIfTrueStructure,
    SmoothNStructure,
    SmoothStructure,
    TrendStructure,
)

from .julia_expressions_builder import (
    HELPER_IMPLEMENTATIONS,
    InlineLookupRegistry,
    JuliaASTVisitor,
    format_number,
    lookup_interpolation_code,
)
from .namespace import JuliaNamespaceManager

# Control variable identifiers produced by Vensim
_CONTROL_IDENTIFIERS = frozenset(
    {"initial_time", "final_time", "time_step", "saveper"}
)

# Structures that expand to auxiliary state variables (handled at element level)
_STATEFUL_STRUCTURES = (
    IntegStructure,
    SmoothStructure,
    SmoothNStructure,
    DelayStructure,
    DelayNStructure,
    DelayFixedStructure,
)

# Structures not yet supported — emit a warning and a placeholder equation
_UNSUPPORTED_STRUCTURES = (
    DataStructure,
)


# ---------------------------------------------------------------------------
# Top-level builder
# ---------------------------------------------------------------------------

class JuliaModelBuilder:
    """Build a standalone Julia/ModelingToolkit model from an AbstractModel.

    Parameters
    ----------
    abstract_model:
        The abstract model produced by a PySD translator.
    """

    def __init__(self, abstract_model: AbstractModel) -> None:
        self.original_path = abstract_model.original_path
        self.sections = [
            JuliaSectionBuilder(section) for section in abstract_model.sections
        ]

    def build_model(self) -> Path:
        """Translate all sections and return the path to the main ``.jl`` file."""
        for section in self.sections:
            section.build_section()
        return self.sections[0].path


# ---------------------------------------------------------------------------
# Section builder
# ---------------------------------------------------------------------------

class JuliaSectionBuilder:
    """Build one section (main model or macro) of the Julia output.

    Parameters
    ----------
    abstract_section:
        The abstract section to translate.
    """

    def __init__(self, abstract_section: AbstractSection) -> None:
        self.name: str = abstract_section.name
        self.path: Path = abstract_section.path.with_suffix(".jl")
        self.root: Path = self.path.parent
        self.model_name: str = self.path.stem
        self.split: bool = abstract_section.split
        self.views_dict: Optional[dict] = abstract_section.views_dict
        self.abstract_elements: List[AbstractElement] = list(abstract_section.elements)
        self._abstract_subscripts = abstract_section.subscripts

        self.namespace = JuliaNamespaceManager()
        self.inline_registry = InlineLookupRegistry()
        self.needed_helpers: Set[str] = set()

        # Map subscript range name → number of elements
        self._subs_sizes: Dict[str, int] = {}
        for sr in self._abstract_subscripts:
            if isinstance(sr.subscripts, list):
                self._subs_sizes[sr.name] = len(sr.subscripts)
            elif isinstance(sr.subscripts, str):
                # copy alias — resolve later if needed, default to 0
                self._subs_sizes[sr.name] = 0

        # Map subscript range name → ordered list of element labels
        self._subs_elems: Dict[str, List[str]] = {}
        for sr in self._abstract_subscripts:
            if isinstance(sr.subscripts, list):
                self._subs_elems[sr.name] = list(sr.subscripts)

        # Accumulated declarations
        self.stock_decls: List[str] = []
        self.aux_decls: List[str] = []
        self.param_decls: List[str] = []
        self.ext_const_decls: List[str] = []
        self.lookup_const_decls: List[str] = []
        self.lookup_func_decls: List[str] = []
        self.lookup_register_decls: List[str] = []
        self.subs_const_decls: List[str] = []
        self.u0_entries: List[str] = []
        # Map julia identifier -> list of dim names (for subscripted vars)
        self._var_dims: Dict[str, List[str]] = {}
        self.control_vals: Dict[str, Optional[str]] = {
            "initial_time": None,
            "final_time": None,
            "time_step": None,
            "saveper": None,
        }

        # Maps Julia identifier -> (equations, is_control_var)
        self.built_elements: Dict[str, Tuple[List[str], bool]] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def build_section(self) -> None:
        """Build the section, writing one or more ``.jl`` files."""
        # First pass: populate the namespace with all element names
        for elem in self.abstract_elements:
            self.namespace.add_to_namespace(elem.name)

        # Emit subscript size constants (const N_DIMNAME = n)
        for name, size in sorted(self._subs_sizes.items()):
            if size > 0:
                jl_name = "N_" + re.sub(r"[^a-z0-9]", "_", name.lower()).upper()
                self.subs_const_decls.append(f"const {jl_name} = {size}")

        # Second pass: process control elements first so that control_vals
        # (especially time_step) are available for constructs like SAMPLE IF TRUE.
        non_control_elems = []
        initial_elems = []
        for elem in self.abstract_elements:
            identifier = self.namespace.namespace[elem.name]
            is_control = isinstance(elem, AbstractControlElement)
            comp = elem.components[0] if elem.components else None
            if is_control:
                eqs = self._process_element(elem, identifier, is_control)
                self.built_elements[identifier] = (eqs, is_control)
            elif comp is not None and isinstance(comp.ast, InitialStructure):
                initial_elems.append((elem, identifier, is_control))
            else:
                non_control_elems.append((elem, identifier, is_control))

        # Third pass: process non-control, non-INITIAL elements
        for elem, identifier, is_control in non_control_elems:
            eqs = self._process_element(elem, identifier, is_control)
            self.built_elements[identifier] = (eqs, is_control)

        # Fourth pass: INITIAL elements (u0_entries now complete)
        for elem, identifier, is_control in initial_elems:
            eqs = self._process_element(elem, identifier, is_control)
            self.built_elements[identifier] = (eqs, is_control)

        # Register any inline lookups collected while visiting ASTs
        for lut_name, xs, ys, itp_type in self.inline_registry.entries:
            const_decl, func_decl, reg_decl = lookup_interpolation_code(lut_name, xs, ys, itp_type)
            self.lookup_const_decls.append(const_decl)
            self.lookup_func_decls.append(func_decl)
            self.lookup_register_decls.append(reg_decl)

        if self.split and self.views_dict:
            self._build_modular()
        else:
            self._build()

    # ------------------------------------------------------------------
    # Subscript helpers
    # ------------------------------------------------------------------

    def _element_dims(self, elem: "AbstractElement") -> List[Tuple[str, int]]:
        """Return ``[(dim_name, dim_size), ...]`` for *elem*'s defining subscripts.

        Uses the first component's first subscript list.  Dims with size == 0
        (unresolved aliases) are filtered out.
        """
        if not elem.components:
            return []
        comp = elem.components[0]
        if not comp.subscripts or not comp.subscripts[0]:
            return []
        dims = []
        for dim_name in comp.subscripts[0]:
            size = self._subs_sizes.get(dim_name, 0)
            if size > 0:
                dims.append((dim_name, size))
        return dims

    def _jl_n(self, dim_name: str) -> str:
        """Julia constant name for the size of a subscript dimension."""
        return "N_" + re.sub(r"[^a-z0-9]", "_", dim_name.lower()).upper()

    def _range_str(self, dims: List[Tuple[str, int]]) -> str:
        """Build ``'1:N_D0, 1:N_D1, ...'`` for array declarations."""
        return ", ".join(f"1:{self._jl_n(d)}" for d, _ in dims)

    def _idx_vars(self, ndim: int) -> List[str]:
        """Generate index variable names ``_i0, _i1, ...`` for comprehensions."""
        return [f"_i{k}" for k in range(ndim)]

    def _for_clause(self, dims: List[Tuple[str, int]], idx_vars: List[str]) -> str:
        """Build ``'_i0 in 1:N_D0, _i1 in 1:N_D1, ...'`` for comprehensions."""
        return ", ".join(
            f"{iv} in 1:{self._jl_n(d)}" for (d, _), iv in zip(dims, idx_vars)
        )

    def _nd_visitor(self, dims: List[Tuple[str, int]], idx_vars: List[str]) -> "JuliaASTVisitor":
        """Return a visitor with active subscript index context for N dims."""
        active_subs = {d: iv for (d, _), iv in zip(dims, idx_vars)}
        return JuliaASTVisitor(
            self.namespace, self.inline_registry, self.needed_helpers,
            active_subs=active_subs, var_dims=self._var_dims,
            subs_sizes=self._subs_sizes, root=self.root,
        )

    def _nd_u0_entries(
        self, identifier: str, dims: List[Tuple[str, int]], init_expr: str
    ) -> None:
        """Append per-element u0 entries for an N-dimensional stock."""
        ranges = [range(1, size + 1) for _, size in dims]
        for idx_combo in itertools.product(*ranges):
            idx_str = ", ".join(str(i) for i in idx_combo)
            self.u0_entries.append(f"{identifier}[{idx_str}] => {init_expr}")

    # ------------------------------------------------------------------
    # Element processing
    # ------------------------------------------------------------------

    def _process_element(
        self,
        elem: AbstractElement,
        identifier: str,
        is_control: bool,
    ) -> List[str]:
        """Return the equation string(s) for *elem*.

        Variable/parameter declarations and initial conditions are registered
        as side-effects on ``self``.
        """
        if not elem.components:
            return []

        comp = elem.components[0]
        ast = comp.ast

        # Determine subscript dimensionality for this element
        dims = self._element_dims(elem)
        ndim = len(dims)

        # Register var dims for use by visitors in 2D contexts
        if ndim > 0 and not is_control:
            self._var_dims[identifier] = [d for d, _ in dims]

        # Scalar visitor (no active subscript context)
        visitor = JuliaASTVisitor(
            self.namespace, self.inline_registry, self.needed_helpers,
            subs_sizes=self._subs_sizes, root=self.root,
        )

        # ---- Named lookup table ----------------------------------------
        if isinstance(comp, AbstractLookup) and isinstance(ast, LookupsStructure):
            const_decl, func_decl, reg_decl = lookup_interpolation_code(
                identifier, ast.x, ast.y, ast.type
            )
            self.lookup_const_decls.append(const_decl)
            self.lookup_func_decls.append(func_decl)
            self.lookup_register_decls.append(reg_decl)
            return []

        # ---- INITIAL() — freeze inner expression at t=0 ----------------
        # Vensim's INITIAL(x) returns the value of x at t=0.  We implement
        # this as a @parameters constant equal to x's initial condition.
        if isinstance(ast, InitialStructure):
            val = self._resolve_initial_value(ast.initial)
            if val is not None:
                if not is_control:
                    self.param_decls.append(f"@parameters {identifier} = {val}")
                return []
            else:
                warn(
                    f"Cannot resolve INITIAL() for '{elem.name}' — "
                    "falling back to auxiliary variable (may not be constant)."
                )
                rhs = visitor.visit(ast.initial)
                self.aux_decls.append(f"@variables {identifier}(t)")
                return [f"{identifier} ~ {rhs}"]

        # ---- Stock (INTEG) ---------------------------------------------
        if isinstance(ast, IntegStructure):
            flow_expr = visitor.visit(ast.flow)
            initial_expr = visitor.visit(ast.initial)
            if ndim == 0:
                self.stock_decls.append(f"@variables {identifier}(t)")
                self.u0_entries.append(f"{identifier} => {initial_expr}")
                return [f"D({identifier}) ~ {flow_expr}"]
            elif ndim == 1:
                self.stock_decls.append(
                    f"@variables {identifier}(t)[{self._range_str(dims)}]"
                )
                self._nd_u0_entries(identifier, dims, initial_expr)
                return [f"Symbolics.scalarize(D.({identifier}) .~ {flow_expr})..."]
            else:
                # N≥2 dims: comprehension with N index variables
                idx_vars = self._idx_vars(ndim)
                vnd = self._nd_visitor(dims, idx_vars)
                flow_nd = vnd.visit(ast.flow)
                idx_str = ", ".join(idx_vars)
                self.stock_decls.append(
                    f"@variables {identifier}(t)[{self._range_str(dims)}]"
                )
                self._nd_u0_entries(identifier, dims, initial_expr)
                return [
                    f"[D({identifier}[{idx_str}]) ~ {flow_nd} "
                    f"for {self._for_clause(dims, idx_vars)}]..."
                ]

        # ---- First-order Smooth ----------------------------------------
        if isinstance(ast, SmoothStructure) and ast.order == 1:
            return self._expand_smooth(identifier, ast, visitor, order=1)

        # ---- Higher-order Smooth / SmoothN -----------------------------
        if isinstance(ast, (SmoothStructure, SmoothNStructure)):
            try:
                order = int(ast.order)
            except (TypeError, ValueError):
                warn(
                    f"SMOOTH with non-integer order for '{elem.name}'; defaulting to 3."
                )
                order = 3
            return self._expand_smooth(identifier, ast, visitor, order=order)

        # ---- Delay (integer order) -------------------------------------
        if isinstance(ast, (DelayStructure, DelayNStructure)):
            try:
                order = int(ast.order)
            except (TypeError, ValueError):
                warn(
                    f"DELAY with non-integer order for '{elem.name}'; defaulting to 3."
                )
                order = 3
            return self._expand_delay(identifier, ast, visitor, order=order)

        # ---- DELAY FIXED ------------------------------------------------
        # Approximate DELAY FIXED as a first-order ODE delay (same formula
        # as DELAY1 with order=1).  True fixed delays need a DDE solver
        # that ModelingToolkit/OrdinaryDiffEq does not support, so this is
        # the best we can do in the MTK ODE framework.
        if isinstance(ast, DelayFixedStructure):
            return self._expand_delay_fixed(identifier, ast, visitor)

        # ---- External constant (GET XLS/DIRECT CONSTANTS) ----------------
        if all(isinstance(c.ast, GetConstantsStructure) for c in elem.components):
            julia_val = self._read_get_constants(elem, identifier)
            if julia_val is not None:
                if is_control:
                    if identifier in self.control_vals:
                        self.control_vals[identifier] = julia_val
                    return []
                if julia_val.startswith("["):
                    self.ext_const_decls.append(f"const {identifier} = {julia_val}")
                else:
                    self.param_decls.append(f"@parameters {identifier} = {julia_val}")
                return []
            # fall through to unsupported handler if reading failed

        # ---- GET XLS/DIRECT LOOKUPS -------------------------------------
        if all(isinstance(c.ast, GetLookupsStructure) for c in elem.components):
            return self._process_get_lookups(elem, identifier)

        # ---- GET XLS/DIRECT DATA ----------------------------------------
        if isinstance(ast, GetDataStructure) or isinstance(comp, AbstractData):
            return self._process_get_data(elem, identifier, comp)

        # ---- TREND ------------------------------------------------------
        if isinstance(ast, TrendStructure):
            return self._expand_trend(identifier, ast, visitor)

        # ---- FORECAST ---------------------------------------------------
        if isinstance(ast, ForecastStructure):
            return self._expand_forecast(identifier, ast, visitor)

        # ---- SAMPLE IF TRUE ---------------------------------------------
        if isinstance(ast, SampleIfTrueStructure):
            return self._expand_sample_if_true(identifier, ast, visitor)

        # ---- ALLOCATE AVAILABLE / ALLOCATE BY PRIORITY ------------------
        if isinstance(ast, (AllocateAvailableStructure, AllocateByPriorityStructure)):
            return self._expand_allocate(identifier, ast, visitor)

        # ---- Remaining unsupported structures ---------------------------
        if isinstance(ast, _UNSUPPORTED_STRUCTURES):
            warn(
                f"'{type(ast).__name__}' for '{elem.name}' is not supported in the "
                "Julia builder — emitting placeholder equation."
            )
            self.aux_decls.append(f"@variables {identifier}(t)")
            return [f"# UNSUPPORTED({type(ast).__name__}): {identifier} ~ 0.0"]

        # ---- Constant / unchangeable constant --------------------------
        if isinstance(comp, AbstractUnchangeableConstant) or comp.type == "Constant":
            value_expr = visitor.visit(ast)
            if is_control:
                if identifier in self.control_vals:
                    self.control_vals[identifier] = value_expr
                return []
            if ndim == 0:
                self.param_decls.append(f"@parameters {identifier} = {value_expr}")
            else:
                self.param_decls.append(
                    f"@parameters {identifier}[{self._range_str(dims)}] = {value_expr}"
                )
            return []

        # ---- Data component (external time-series) — fallback -----------
        if isinstance(comp, AbstractData):
            warn(
                f"Data component '{elem.name}' references external data, which is "
                "not supported in the Julia builder — emitting 0.0 placeholder."
            )
            self.aux_decls.append(f"@variables {identifier}(t)")
            return [f"# DATA: {identifier} ~ 0.0"]

        # ---- Auxiliary variable (algebraic) ----------------------------
        if ndim == 0:
            rhs_expr = visitor.visit(ast)
            if is_control:
                if identifier in self.control_vals:
                    self.control_vals[identifier] = rhs_expr
                return []
            self.aux_decls.append(f"@variables {identifier}(t)")
            return [f"{identifier} ~ {rhs_expr}"]
        elif ndim == 1:
            rhs_expr = visitor.visit(ast)
            if is_control:
                if identifier in self.control_vals:
                    self.control_vals[identifier] = rhs_expr
                return []
            self.aux_decls.append(
                f"@variables {identifier}(t)[{self._range_str(dims)}]"
            )
            return [f"Symbolics.scalarize({identifier} .~ {rhs_expr})..."]
        else:
            # N≥2 dims: comprehension with N index variables
            idx_vars = self._idx_vars(ndim)
            vnd = self._nd_visitor(dims, idx_vars)
            rhs_nd = vnd.visit(ast)
            if is_control:
                if identifier in self.control_vals:
                    self.control_vals[identifier] = rhs_nd
                return []
            idx_str = ", ".join(idx_vars)
            self.aux_decls.append(
                f"@variables {identifier}(t)[{self._range_str(dims)}]"
            )
            return [
                f"[{identifier}[{idx_str}] ~ {rhs_nd} "
                f"for {self._for_clause(dims, idx_vars)}]..."
            ]

    # ------------------------------------------------------------------
    # Smooth expansion
    # ------------------------------------------------------------------

    def _expand_smooth(
        self,
        identifier: str,
        ast,
        visitor: JuliaASTVisitor,
        order: int,
    ) -> List[str]:
        """Expand a SMOOTH(N) into *order* chained first-order ODE levels.

        The output variable ``identifier`` is declared as an auxiliary equal
        to the final level.
        """
        input_expr = visitor.visit(ast.input)
        smooth_time_expr = visitor.visit(ast.smooth_time)
        initial_expr = visitor.visit(ast.initial)

        eqs: List[str] = []
        prev_expr = input_expr
        for i in range(1, order + 1):
            lv_name = f"_lv{i}_{identifier}"
            # Register in namespace so other expressions can reference it
            self.namespace.namespace[f"__internal_lv{i}_{identifier}"] = lv_name
            self.stock_decls.append(f"@variables {lv_name}(t)")
            self.u0_entries.append(f"{lv_name} => {initial_expr}")
            eqs.append(
                f"D({lv_name}) ~ ({prev_expr} - {lv_name}) / ({smooth_time_expr} / {order})"
            )
            prev_expr = lv_name

        self.aux_decls.append(f"@variables {identifier}(t)")
        eqs.append(f"{identifier} ~ {prev_expr}")
        return eqs

    # ------------------------------------------------------------------
    # Delay expansion
    # ------------------------------------------------------------------

    def _expand_delay(
        self,
        identifier: str,
        ast,
        visitor: JuliaASTVisitor,
        order: int,
    ) -> List[str]:
        """Expand a DELAY(N) into *order* chained first-order pipeline levels.

        Each level ``L_i`` satisfies::

            dL_i/dt = (inflow_i - L_i * rate)
            rate    = order / delay_time
            inflow_1 = input;  inflow_i = L_{i-1} * rate  for i > 1
        """
        input_expr = visitor.visit(ast.input)
        delay_time_expr = visitor.visit(ast.delay_time)
        initial_expr = visitor.visit(ast.initial)

        rate_expr = f"({order} / {delay_time_expr})"
        eqs: List[str] = []
        prev_outflow = input_expr
        for i in range(1, order + 1):
            lv_name = f"_dl{i}_{identifier}"
            self.namespace.namespace[f"__internal_dl{i}_{identifier}"] = lv_name
            self.stock_decls.append(f"@variables {lv_name}(t)")
            # Initial level = initial_value * delay_time / order
            self.u0_entries.append(
                f"{lv_name} => {initial_expr} * {delay_time_expr} / {order}"
            )
            eqs.append(
                f"D({lv_name}) ~ ({prev_outflow} - {lv_name} * {rate_expr})"
            )
            prev_outflow = f"{lv_name} * {rate_expr}"

        self.aux_decls.append(f"@variables {identifier}(t)")
        eqs.append(f"{identifier} ~ {prev_outflow}")
        return eqs

    # ------------------------------------------------------------------
    # DELAY FIXED expansion
    # ------------------------------------------------------------------

    def _expand_delay_fixed(
        self,
        identifier: str,
        ast,
        visitor: "JuliaASTVisitor",
    ) -> List[str]:
        """Approximate DELAY FIXED as a first-order ODE delay.

        The true DELAY FIXED is a pure transport delay (DDE), which
        ModelingToolkit/OrdinaryDiffEq cannot solve.  We approximate it
        with a first-order exponential delay (DELAY1):

            D(output) ~ (input - output) / delay_time

        with initial condition ``output(0) = initial``.
        """
        input_expr = visitor.visit(ast.input)
        delay_time_expr = visitor.visit(ast.delay_time)
        initial_expr = visitor.visit(ast.initial)

        lv_name = f"_df_{identifier}"
        self.namespace.namespace[f"__internal_df_{identifier}"] = lv_name
        self.stock_decls.append(f"@variables {lv_name}(t)")
        self.u0_entries.append(f"{lv_name} => {initial_expr}")

        self.aux_decls.append(f"@variables {identifier}(t)")
        return [
            f"D({lv_name}) ~ ({input_expr} - {lv_name}) / {delay_time_expr}",
            f"{identifier} ~ {lv_name}",
        ]

    # ------------------------------------------------------------------
    # Trend expansion
    # ------------------------------------------------------------------

    def _expand_trend(
        self,
        identifier: str,
        ast,
        visitor: "JuliaASTVisitor",
    ) -> List[str]:
        """Expand TREND(input, average_time, initial_trend) into an ODE.

        Introduces a smooth level ``_sm_{identifier}`` that tracks the
        exponential moving average of the input:

            D(_sm) ~ (input - _sm) / average_time

        Then the trend (fractional growth rate) is:

            output ~ (input - _sm) / (average_time * _sm)

        The smooth level is initialised so that at t=0 the output equals
        ``initial_trend``:

            _sm(0) = input(0) / (1 + initial_trend * average_time)

        We use the simpler ``input(0)`` approximation (same as PySD's
        Trend stateful initialisation) and rely on the model's initial
        conditions to provide a consistent starting point.
        """
        input_expr = visitor.visit(ast.input)
        avg_time_expr = visitor.visit(ast.average_time)
        initial_trend_expr = visitor.visit(ast.initial_trend)

        sm_name = f"_sm_{identifier}"
        self.namespace.namespace[f"__internal_sm_{identifier}"] = sm_name
        self.stock_decls.append(f"@variables {sm_name}(t)")
        # u0: _sm = input / (1 + initial_trend * average_time)
        # We approximate the initial input as the initial_trend expression;
        # a better approximation requires evaluating the input at t0.
        # Use the same formula as PySD: sm0 = input0 (the Trend stateful
        # initialises its smooth to input/1 when initial_trend is given).
        # We store the initial as a formula that Julia will evaluate at t=0.
        self.u0_entries.append(
            f"{sm_name} => {input_expr} / (1.0 + ({initial_trend_expr}) * ({avg_time_expr}))"
        )

        self.aux_decls.append(f"@variables {identifier}(t)")
        return [
            f"D({sm_name}) ~ ({input_expr} - {sm_name}) / ({avg_time_expr})",
            (
                f"{identifier} ~ ifelse(iszero({sm_name}), {initial_trend_expr}, "
                f"({input_expr} - {sm_name}) / (({avg_time_expr}) * {sm_name}))"
            ),
        ]

    # ------------------------------------------------------------------
    # Forecast expansion
    # ------------------------------------------------------------------

    def _expand_forecast(
        self,
        identifier: str,
        ast,
        visitor: "JuliaASTVisitor",
    ) -> List[str]:
        """Expand FORECAST(input, average_time, horizon) = input*(1 + TREND*horizon).

        FORECAST internally computes a TREND and projects it forward by
        *horizon*.  We expand it inline, introducing the same internal
        smooth level as ``_expand_trend``.
        """
        input_expr = visitor.visit(ast.input)
        avg_time_expr = visitor.visit(ast.average_time)
        horizon_expr = visitor.visit(ast.horizon)
        initial_trend_expr = visitor.visit(ast.initial_trend)

        sm_name = f"_sm_{identifier}"
        self.namespace.namespace[f"__internal_sm_{identifier}"] = sm_name
        self.stock_decls.append(f"@variables {sm_name}(t)")
        self.u0_entries.append(
            f"{sm_name} => {input_expr} / (1.0 + ({initial_trend_expr}) * ({avg_time_expr}))"
        )

        # trend = (input - sm) / (avg_time * sm)
        # forecast = input * (1 + trend * horizon)
        self.aux_decls.append(f"@variables {identifier}(t)")
        return [
            f"D({sm_name}) ~ ({input_expr} - {sm_name}) / ({avg_time_expr})",
            (
                f"{identifier} ~ {input_expr} * (1.0 + "
                f"ifelse(iszero({sm_name}), {initial_trend_expr}, "
                f"({input_expr} - {sm_name}) / (({avg_time_expr}) * {sm_name})) "
                f"* ({horizon_expr}))"
            ),
        ]

    # ------------------------------------------------------------------
    # SAMPLE IF TRUE expansion
    # ------------------------------------------------------------------

    def _expand_sample_if_true(
        self,
        identifier: str,
        ast,
        visitor: "JuliaASTVisitor",
    ) -> List[str]:
        """Expand SAMPLE IF TRUE(condition, input, initial).

        SAMPLE IF TRUE is a discrete sample-and-hold: whenever the condition
        is true the output is updated to the input; otherwise the output holds
        its previous value.

        We approximate this as an INTEG with a conditional flow whose rate is
        tied to the simulation time step so that the Euler solver updates the
        state to ``input`` within one time step when the condition is true:

            D(output) ~ ifelse(condition > 0.5,
                               (input - output) / time_step,
                               0.0)

        Here ``time_step`` refers to the Julia variable defined in the
        generated file.  With Euler integration the next step will be:

            output_new = output + dt * (input - output) / dt = input

        which is exact (one-step snap to input).
        """
        condition_expr = visitor.visit(ast.condition)
        input_expr = visitor.visit(ast.input)
        initial_expr = visitor.visit(ast.initial)

        st_name = f"_sit_{identifier}"
        self.namespace.namespace[f"__internal_sit_{identifier}"] = st_name
        self.stock_decls.append(f"@variables {st_name}(t)")
        self.u0_entries.append(f"{st_name} => {initial_expr}")

        # Use the simulation time_step as the relaxation divisor.
        # With Euler integration: output_new = output + dt*(input-output)/dt = input.
        # We look up time_step from control_vals; fall back to a symbolic reference.
        ts_val = self.control_vals.get("time_step")
        ts_expr = ts_val if ts_val is not None else "time_step"

        self.aux_decls.append(f"@variables {identifier}(t)")
        return [
            f"D({st_name}) ~ ifelse({condition_expr} > 0.5, "
            f"({input_expr} - {st_name}) / ({ts_expr}), 0.0)",
            f"{identifier} ~ {st_name}",
        ]

    # ------------------------------------------------------------------
    # ALLOCATE AVAILABLE / ALLOCATE BY PRIORITY
    # ------------------------------------------------------------------

    def _expand_allocate(
        self,
        identifier: str,
        ast,
        visitor: "JuliaASTVisitor",
    ) -> List[str]:
        """Emit a simple proportional allocation approximation.

        Full Vensim priority allocation requires complex logic that is
        difficult to express as a MTK algebraic equation.  We emit a
        proportional-share approximation:

            allocate_available  →  request / sum(request) * avail
            allocate_by_priority →  request / sum(request) * supply

        This is a structural approximation only.  A comment is included
        in the generated file to flag the limitation.
        """
        warn(
            f"AllocateStructure for '{identifier}' is approximated as proportional "
            "allocation — results may differ from the Vensim priority-based algorithm."
        )
        if isinstance(ast, AllocateAvailableStructure):
            request_expr = visitor.visit(ast.request)
            avail_expr = visitor.visit(ast.avail)
            rhs = (
                f"ifelse(iszero(sum({request_expr})), 0.0, "
                f"{request_expr} ./ sum({request_expr}) .* ({avail_expr}))"
            )
        else:
            # AllocateByPriorityStructure
            request_expr = visitor.visit(ast.request)
            supply_expr = visitor.visit(ast.supply)
            rhs = (
                f"ifelse(iszero(sum({request_expr})), 0.0, "
                f"{request_expr} ./ sum({request_expr}) .* ({supply_expr}))"
            )

        self.aux_decls.append(f"@variables {identifier}(t)")
        return [
            f"# ALLOCATE (proportional approximation): {identifier}",
            f"{identifier} ~ {rhs}",
        ]

    # ------------------------------------------------------------------
    # GET LOOKUPS processing
    # ------------------------------------------------------------------

    def _process_get_lookups(
        self, elem: "AbstractElement", identifier: str
    ) -> List[str]:
        """Read external lookup data and emit a named interpolation function.

        Uses ``ExtLookup`` to load the table at translation time, then
        emits the same ``LinearInterpolation`` pattern as inline lookups.
        """
        try:
            from pysd.py_backend.external import ExtLookup

            subs_map: Dict[str, list] = {}
            for sr in self._abstract_subscripts:
                if isinstance(sr.subscripts, list):
                    subs_map[sr.name] = sr.subscripts

            def _coords(comp) -> dict:
                def_subs = comp.subscripts[0] if comp.subscripts else []
                return {s: subs_map.get(s, []) for s in def_subs} if def_subs else {}

            comp0 = elem.components[0]
            ast0 = comp0.ast
            coords0 = _coords(comp0)

            if len(elem.components) > 1:
                final_coords: Dict[str, list] = {}
                for comp in elem.components:
                    for s, v in _coords(comp).items():
                        if s not in final_coords:
                            final_coords[s] = v
            else:
                final_coords = coords0

            ext = ExtLookup(
                file_name=ast0.file,
                tab=ast0.tab,
                x_row_or_col=ast0.x_row_or_col,
                cell=ast0.cell,
                coords=coords0,
                root=self.root,
                final_coords=final_coords,
                py_name=identifier,
            )

            for comp in elem.components[1:]:
                ast_i = comp.ast
                ext.add(ast_i.file, ast_i.tab, ast_i.x_row_or_col, ast_i.cell, _coords(comp))

            ext.initialize()

            # ext.data is an xarray DataArray with dim "lookup_dim"
            import numpy as np
            data = ext.data
            if hasattr(data, "values"):
                arr = data.values
            else:
                arr = np.asarray(data)

            xs = tuple(float(x) for x in data.coords["lookup_dim"].values)

            # For scalar lookups, data has shape (n_points,)
            if arr.ndim == 1:
                ys = tuple(float(y) for y in arr)
                const_decl, func_decl, reg_decl = lookup_interpolation_code(
                    identifier, xs, ys, "interpolate"
                )
                self.lookup_const_decls.append(const_decl)
                self.lookup_func_decls.append(func_decl)
                self.lookup_register_decls.append(reg_decl)
                return []
            elif arr.ndim == 2:
                # 2D: shape (n_points, n_subs).
                # Emit one lookup function per subscript element:
                #   identifier_1(x), identifier_2(x), ...
                # and a dispatch function identifier(i, x) that selects by index.
                n_subs = arr.shape[1]
                sub_func_names = []
                for k in range(n_subs):
                    col_ys = tuple(float(y) for y in arr[:, k])
                    sub_name = f"{identifier}_{k + 1}"
                    const_decl, func_decl, reg_decl = lookup_interpolation_code(
                        sub_name, xs, col_ys, "interpolate"
                    )
                    self.lookup_const_decls.append(const_decl)
                    self.lookup_func_decls.append(func_decl)
                    self.lookup_register_decls.append(reg_decl)
                    sub_func_names.append(sub_name)

                # Build a dispatch array and wrapper:
                # const identifier_fns = [identifier_1, identifier_2, ...]
                # identifier(i, x) = identifier_fns[i](x)
                fn_list = ", ".join(sub_func_names)
                self.lookup_const_decls.append(
                    f"const {identifier}_fns = [{fn_list}]"
                )
                self.lookup_func_decls.append(
                    f"{identifier}(i, x) = {identifier}_fns[i](x)"
                )
                self.lookup_register_decls.append(
                    f"@register_symbolic {identifier}(i::Integer, x::Real)"
                )
                return []
            else:
                warn(
                    f"Subscripted GET LOOKUPS '{elem.name}' has {arr.ndim - 1} "
                    "subscript dimensions (> 1D subs) — only 1D subscripted lookups "
                    "are supported. Emitting flattened first-column lookup as approximation."
                )
                ys = tuple(float(y) for y in arr.reshape(arr.shape[0], -1)[:, 0])
                const_decl, func_decl, reg_decl = lookup_interpolation_code(
                    identifier, xs, ys, "interpolate"
                )
                self.lookup_const_decls.append(const_decl)
                self.lookup_func_decls.append(func_decl)
                self.lookup_register_decls.append(reg_decl)
                return []

        except Exception as exc:
            warn(
                f"Could not read GET LOOKUPS for '{elem.name}': {exc} "
                "— emitting placeholder auxiliary."
            )
            self.aux_decls.append(f"@variables {identifier}(t)")
            return [f"# GET_LOOKUPS_FAILED: {identifier} ~ 0.0"]

    # ------------------------------------------------------------------
    # GET DATA processing
    # ------------------------------------------------------------------

    def _process_get_data(
        self,
        elem: "AbstractElement",
        identifier: str,
        comp: "AbstractComponent",
    ) -> List[str]:
        """Read external time-series data and emit a time-indexed interpolation.

        Uses ``ExtData`` to load the series at translation time, then
        emits a ``LinearInterpolation`` over (time, value) pairs just
        like a lookup, but with ``t`` as the argument.
        """
        try:
            from pysd.py_backend.external import ExtData

            subs_map: Dict[str, list] = {}
            for sr in self._abstract_subscripts:
                if isinstance(sr.subscripts, list):
                    subs_map[sr.name] = sr.subscripts

            def _coords(c) -> dict:
                def_subs = c.subscripts[0] if c.subscripts else []
                return {s: subs_map.get(s, []) for s in def_subs} if def_subs else {}

            # Collect AST from first component that has a GetDataStructure
            comp0 = None
            for c in elem.components:
                if isinstance(c.ast, GetDataStructure):
                    comp0 = c
                    break
            if comp0 is None:
                raise ValueError("No GetDataStructure component found")

            ast0 = comp0.ast
            coords0 = _coords(comp0)

            if len(elem.components) > 1:
                final_coords: Dict[str, list] = {}
                for c in elem.components:
                    for s, v in _coords(c).items():
                        if s not in final_coords:
                            final_coords[s] = v
            else:
                final_coords = coords0

            ext = ExtData(
                file_name=ast0.file,
                tab=ast0.tab,
                time_row_or_col=ast0.time_row_or_col,
                cell=ast0.cell,
                interp="interpolate",
                coords=coords0,
                root=self.root,
                final_coords=final_coords,
                py_name=identifier,
            )

            for c in elem.components[1:]:
                if isinstance(c.ast, GetDataStructure):
                    ai = c.ast
                    ext.add(ai.file, ai.tab, ai.time_row_or_col, ai.cell,
                            "interpolate", _coords(c))

            ext.initialize()

            import numpy as np
            data = ext.data
            if hasattr(data, "values"):
                arr = data.values
                time_vals = data.coords["time"].values
            else:
                arr = np.asarray(data)
                time_vals = None

            if time_vals is None:
                raise ValueError(f"No time dimension in data (shape={arr.shape})")

            xs = tuple(float(t) for t in time_vals)

            if arr.ndim == 1:
                ys = tuple(float(y) for y in arr)
                const_decl, func_decl, reg_decl = lookup_interpolation_code(
                    identifier, xs, ys, "interpolate"
                )
                self.lookup_const_decls.append(const_decl)
                self.lookup_func_decls.append(func_decl)
                self.lookup_register_decls.append(reg_decl)
                return []
            elif arr.ndim == 2:
                # Subscripted time-series: shape (n_time, n_subs)
                n_subs = arr.shape[1]
                sub_func_names = []
                for k in range(n_subs):
                    col_ys = tuple(float(y) for y in arr[:, k])
                    sub_name = f"{identifier}_{k + 1}"
                    const_decl, func_decl, reg_decl = lookup_interpolation_code(
                        sub_name, xs, col_ys, "interpolate"
                    )
                    self.lookup_const_decls.append(const_decl)
                    self.lookup_func_decls.append(func_decl)
                    self.lookup_register_decls.append(reg_decl)
                    sub_func_names.append(sub_name)

                fn_list = ", ".join(sub_func_names)
                self.lookup_const_decls.append(
                    f"const {identifier}_fns = [{fn_list}]"
                )
                self.lookup_func_decls.append(
                    f"{identifier}(i, x) = {identifier}_fns[i](x)"
                )
                self.lookup_register_decls.append(
                    f"@register_symbolic {identifier}(i::Integer, x::Real)"
                )
                return []
            else:
                raise ValueError(f"Unexpected data dimensions: {arr.ndim} (shape={arr.shape})")

        except Exception as exc:
            warn(
                f"Could not read GET DATA for '{elem.name}': {exc} "
                "— emitting placeholder auxiliary."
            )
            self.aux_decls.append(f"@variables {identifier}(t)")
            return [f"# GET_DATA_FAILED: {identifier} ~ 0.0"]

    def _resolve_initial_value(self, inner_ast) -> Optional[str]:
        """Return the t=0 value of *inner_ast* as a Julia literal, or None.

        Handles:
        * Numeric literals
        * References to stocks (in ``u0_entries``)
        * References to parameters/constants (including GetConstantsStructure)
        * References to auxiliaries whose own equation chains back to a stock
          (one level of indirection, e.g. ``INITIAL(InflowA)`` where
          ``InflowA ~ StockA`` and StockA has a known initial condition)
        * GetConstantsStructure directly embedded in the INITIAL() argument
        """
        from pysd.builders.julia.julia_expressions_builder import format_number
        if isinstance(inner_ast, (int, float)):
            return format_number(inner_ast)
        if isinstance(inner_ast, ReferenceStructure):
            return self._resolve_ref_initial(inner_ast.reference, depth=3)
        if isinstance(inner_ast, GetConstantsStructure):
            # Try to read the constant directly
            try:
                from pysd.py_backend.external import ExtConstant
                ext = ExtConstant(
                    file_name=inner_ast.file,
                    tab=inner_ast.tab,
                    cell=inner_ast.cell,
                    coords={},
                    root=self.root,
                    final_coords={},
                    py_name="_initial_resolve",
                )
                ext.initialize()
                return _format_julia_value(ext.data)
            except Exception:
                pass
        return None

    def _resolve_ref_initial(self, ref: str, depth: int) -> Optional[str]:
        """Recursively resolve the t=0 value of a variable reference."""
        if depth < 0:
            return None
        julia_id = self.namespace.get(ref)
        if julia_id is None:
            return None
        # Check u0_entries (stocks)
        for entry in self.u0_entries:
            parts = entry.split("=>", 1)
            if len(parts) == 2 and parts[0].strip() == julia_id:
                return parts[1].strip()
        # Check param_decls (constants)
        for decl in self.param_decls:
            prefix = f"@parameters {julia_id} = "
            if decl.startswith(prefix):
                return decl[len(prefix):]
        # Follow an auxiliary equation one level deeper
        if depth > 0 and julia_id in self.built_elements:
            eqs, _ = self.built_elements[julia_id]
            for eq in eqs:
                if "~" in eq:
                    rhs = eq.split("~", 1)[1].strip()
                    # Plain number
                    try:
                        float(rhs)
                        return rhs
                    except ValueError:
                        pass
                    # Plain identifier → recurse
                    import re as _re
                    if _re.match(r"^[a-z_][a-z0-9_]*$", rhs):
                        result = self._resolve_ref_initial(rhs, depth - 1)
                        if result:
                            return result
        return None

    # ------------------------------------------------------------------
    # External constants reader
    # ------------------------------------------------------------------

    def _read_get_constants(
        self, elem: AbstractElement, identifier: str
    ) -> Optional[str]:
        """Read all GetConstantsStructure components for *elem* using ExtConstant.

        Returns a Julia literal string (scalar or array) on success, or None
        if the file cannot be read, in which case the caller falls through to
        the unsupported-structure handler.
        """
        try:
            from pysd.py_backend.external import ExtConstant

            # Build a map from subscript range name → list of elements
            subs_map: Dict[str, list] = {}
            for sr in self._abstract_subscripts:
                if isinstance(sr.subscripts, list):
                    subs_map[sr.name] = sr.subscripts

            def _coords(comp) -> dict:
                def_subs = comp.subscripts[0] if comp.subscripts else []
                return {s: subs_map.get(s, []) for s in def_subs} if def_subs else {}

            comp0 = elem.components[0]
            coords0 = _coords(comp0)
            ast0 = comp0.ast

            # For multi-component elements, final_coords covers all dims
            if len(elem.components) > 1:
                final_coords: Dict[str, list] = {}
                for comp in elem.components:
                    for s, v in _coords(comp).items():
                        if s not in final_coords:
                            final_coords[s] = v
            else:
                final_coords = coords0

            ext = ExtConstant(
                file_name=ast0.file,
                tab=ast0.tab,
                cell=ast0.cell,
                coords=coords0,
                root=self.root,
                final_coords=final_coords,
                py_name=identifier,
            )

            for comp in elem.components[1:]:
                ast_i = comp.ast
                ext.add(ast_i.file, ast_i.tab, ast_i.cell, _coords(comp))

            ext.initialize()
            return _format_julia_value(ext.data)

        except Exception as exc:
            warn(
                f"Could not read external constant for '{elem.name}': {exc} "
                "— emitting placeholder."
            )
            return None

    # ------------------------------------------------------------------
    # Single-file build
    # ------------------------------------------------------------------

    def _build(self) -> None:
        """Write the whole model as one ``.jl`` file."""
        all_eqs: List[str] = []
        for eqs, _is_ctrl in self.built_elements.values():
            all_eqs.extend(eqs)
        text = self._full_file_content(all_eqs)
        self.path.write_text(text, encoding="UTF-8")

    # ------------------------------------------------------------------
    # Modular build
    # ------------------------------------------------------------------

    def _build_modular(self) -> None:
        """Write main ``.jl`` + one file per Vensim view."""
        modules_dir = self.root / f"modules_{self.model_name}"
        modules_dir.mkdir(exist_ok=True)

        assigned_ids: Set[str] = set()
        include_lines: List[str] = []
        eq_var_names: List[str] = []

        base = Path(f"modules_{self.model_name}")
        self._process_views_tree(
            self.views_dict,
            base,
            self.root,
            assigned_ids,
            include_lines,
            eq_var_names,
        )

        # Variables not assigned to any view go into the main file
        leftover_eqs: List[str] = []
        for identifier, (eqs, is_ctrl) in self.built_elements.items():
            if identifier not in assigned_ids and not is_ctrl:
                leftover_eqs.extend(eqs)
                if leftover_eqs:
                    warn(
                        f"Variable '{identifier}' is not declared in any view — "
                        "added to the main module."
                    )

        text = self._modular_main_content(include_lines, eq_var_names, leftover_eqs)
        self.path.write_text(text, encoding="UTF-8")

    def _process_views_tree(
        self,
        tree: dict,
        current_path: Path,
        wdir: Path,
        assigned_ids: Set[str],
        include_lines: List[str],
        eq_var_names: List[str],
    ) -> None:
        """Recursively walk *tree* and write one module file per leaf view."""
        for view_name, content in tree.items():
            view_path = current_path / view_name
            if isinstance(content, set):
                # Leaf node — collect identifiers for this view
                view_ids = self._resolve_view_ids(content)
                non_ctrl_ids = [
                    vid for vid in view_ids
                    if not self.built_elements.get(vid, ([], True))[1]
                ]
                if not non_ctrl_ids:
                    continue

                module_file = wdir / view_path.with_suffix(".jl")
                module_file.parent.mkdir(parents=True, exist_ok=True)
                eq_var = _path_to_eq_var(view_path)

                module_eqs: List[str] = []
                for vid in sorted(non_ctrl_ids):
                    eqs, _ = self.built_elements.get(vid, ([], False))
                    module_eqs.extend(eqs)
                    assigned_ids.add(vid)

                self._write_module_file(module_file, eq_var, module_eqs, view_path)
                rel = module_file.relative_to(wdir)
                include_lines.append(f'include("{rel}")')
                eq_var_names.append(eq_var)
            else:
                # Intermediate node — recurse
                (wdir / view_path).mkdir(parents=True, exist_ok=True)
                self._process_views_tree(
                    content, view_path, wdir, assigned_ids, include_lines, eq_var_names
                )

    def _resolve_view_ids(self, vensim_names: set) -> List[str]:
        """Map a set of Vensim variable names to Julia identifiers."""
        result = []
        for name in vensim_names:
            julia_id = self.namespace.get(name)
            if julia_id and julia_id in self.built_elements:
                result.append(julia_id)
        return result

    def _write_module_file(
        self,
        path: Path,
        eq_var: str,
        equations: List[str],
        module_path: Path,
    ) -> None:
        # Drop the modules_<name> prefix for the display name
        display = ".".join(list(module_path.parts)[1:])
        eq_lines = ",\n    ".join(equations) if equations else ""
        text = textwrap.dedent(f"""\
            # Module {display}
            # Translated using PySD version {__version__}

            {eq_var} = Equation[
                {eq_lines}
            ]
            """)
        path.write_text(text, encoding="UTF-8")

    # ------------------------------------------------------------------
    # Content assembly helpers
    # ------------------------------------------------------------------

    def _file_header(self, extra_packages: bool = False) -> str:
        # OrdinaryDiffEq v7 split Euler into OrdinaryDiffEqLowOrderRK
        uses = ["ModelingToolkit", "Symbolics", "OrdinaryDiffEq", "OrdinaryDiffEqLowOrderRK"]
        if self.lookup_const_decls or extra_packages:
            uses.append("DataInterpolations")
        return (
            # Use # comments, not a Julia docstring: a triple-quoted string
            # immediately before `using` is parsed as "document the using
            # statement" which is a syntax error.
            f"# Model {self.model_name}\n"
            f"# Translated using PySD version {__version__}\n\n"
            f"using {', '.join(uses)}\n\n"
            # MTK v9+ requires @independent_variables for the time variable
            "@independent_variables t\n"
            "D = Differential(t)\n\n"
        )

    def _helpers_block(self) -> str:
        if not self.needed_helpers:
            return ""
        lines = ["# Helper functions"]
        for name in sorted(self.needed_helpers):
            if name in HELPER_IMPLEMENTATIONS:
                lines.append(HELPER_IMPLEMENTATIONS[name])
        return "\n".join(lines) + "\n\n"

    def _lookup_block(self) -> str:
        if not self.lookup_const_decls:
            return ""
        lines = ["# Lookup tables"]
        for const_decl, func_decl, reg_decl in zip(
            self.lookup_const_decls, self.lookup_func_decls, self.lookup_register_decls
        ):
            lines.append(const_decl)
            lines.append(func_decl)
            # @register_symbolic must come after the function definition and
            # after `using ModelingToolkit` so MTK treats it as a symbolic
            # primitive (called each timestep rather than constant-folded).
            lines.append(reg_decl)
        return "\n".join(lines) + "\n\n"

    def _declarations_block(self) -> str:
        lines: List[str] = []
        if self.subs_const_decls:
            lines.append("# Subscript dimension sizes")
            lines.extend(self.subs_const_decls)
            lines.append("")
        if self.stock_decls:
            lines.append("# Stocks (state variables)")
            lines.extend(self.stock_decls)
        if self.aux_decls:
            lines.append("\n# Auxiliary variables")
            lines.extend(self.aux_decls)
        if self.param_decls:
            lines.append("\n# Parameters")
            lines.extend(self.param_decls)
        if self.ext_const_decls:
            lines.append("\n# External constants")
            lines.extend(self.ext_const_decls)
        return "\n".join(lines) + "\n"

    def _equations_block(self, equations: List[str]) -> str:
        if not equations:
            return "eqs = Equation[]\n"
        lines = ",\n    ".join(equations)
        return f"eqs = [\n    {lines},\n]\n"

    def _u0_block(self) -> str:
        if not self.u0_entries:
            return "u0 = []\n"
        lines = ",\n    ".join(self.u0_entries)
        return f"u0 = [\n    {lines},\n]\n"

    def _control_block(self) -> str:
        it = self.control_vals.get("initial_time") or "0.0"
        ft = self.control_vals.get("final_time") or "100.0"
        ts = self.control_vals.get("time_step") or "1.0"
        return (
            "# Simulation control\n"
            f"initial_time = {it}\n"
            f"final_time   = {ft}\n"
            f"time_step    = {ts}\n"
            "tspan = (initial_time, final_time)\n"
        )

    def _run_function(self) -> str:
        ts = self.control_vals.get("time_step") or "time_step"
        return textwrap.dedent(f"""\
            function run_model(; u0=u0, tspan=tspan, dt={ts}, solver=Euler())
                prob = ODEProblem(sys, u0, tspan)
                # saveat ensures solution is stored at every dt step,
                # which is required for correct output of observed (auxiliary) variables.
                solve(prob, solver; dt=dt, saveat=tspan[1]:dt:tspan[2])
            end
            """)

    def _system_block(self) -> str:
        sym = re.sub(r"[^a-zA-Z0-9_]", "_", self.model_name)
        return (
            f"@named sys = ODESystem(eqs, t; name=:{sym})\n"
            "sys = structural_simplify(sys)\n"
        )

    def _full_file_content(self, equations: List[str]) -> str:
        needs_di = bool(self.lookup_const_decls)
        return "".join([
            self._file_header(extra_packages=needs_di),
            self._helpers_block(),
            self._lookup_block(),
            self._declarations_block(),
            "\n",
            self._equations_block(equations),
            "\n",
            self._u0_block(),
            "\n",
            self._control_block(),
            "\n",
            self._system_block(),
            "\n",
            self._run_function(),
        ])

    def _modular_main_content(
        self,
        include_lines: List[str],
        eq_var_names: List[str],
        leftover_eqs: List[str],
    ) -> str:
        needs_di = bool(self.lookup_const_decls)
        include_block = "\n# Module includes\n" + "\n".join(include_lines) + "\n"

        leftover_block = ""
        if leftover_eqs:
            lines = ",\n    ".join(leftover_eqs)
            leftover_block = f"\n_main_eqs = Equation[\n    {lines},\n]\n"
            eq_var_names = list(eq_var_names) + ["_main_eqs"]

        if eq_var_names:
            concat = "; ".join(f"{v}..." for v in eq_var_names)
            combined = f"eqs = [{concat}]\n"
        else:
            combined = "eqs = Equation[]\n"

        return "".join([
            self._file_header(extra_packages=needs_di),
            self._helpers_block(),
            self._lookup_block(),
            self._declarations_block(),
            include_block,
            leftover_block,
            "\n",
            combined,
            "\n",
            self._u0_block(),
            "\n",
            self._control_block(),
            "\n",
            self._system_block(),
            "\n",
            self._run_function(),
        ])


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _path_to_eq_var(path: Path) -> str:
    """Convert a module path like ``modules_model/Sector A/Sub1`` to ``sector_a_sub1_eqs``."""
    # Drop the first path component (the modules_<name> directory)
    parts = list(path.parts)[1:] if len(path.parts) > 1 else list(path.parts)
    name = "_".join(parts)
    name = re.sub(r"[^a-z0-9_]", "_", name.lower())
    name = re.sub(r"_+", "_", name).strip("_")
    return f"{name}_eqs"


def _format_julia_value(data) -> str:
    """Format a Python/numpy/xarray value as a Julia literal.

    Scalars become plain number strings.
    1-D arrays become ``[v1, v2, ...]``.
    2-D arrays become ``[r1c1 r1c2; r2c1 r2c2]`` (Julia matrix literal).
    Higher-dimensional arrays are flattened to 1-D.
    """
    import numpy as np

    # xarray DataArray → plain numpy array
    if hasattr(data, "values"):
        data = data.values

    if isinstance(data, (int, float)):
        return format_number(float(data))

    arr = np.asarray(data, dtype=float)

    if arr.ndim == 0:
        return format_number(float(arr))

    if arr.ndim == 1:
        vals = ", ".join(format_number(float(v)) for v in arr)
        return f"[{vals}]"

    if arr.ndim == 2:
        rows = "; ".join(
            " ".join(format_number(float(v)) for v in row) for row in arr
        )
        return f"[{rows}]"

    # Higher dims: flatten
    vals = ", ".join(format_number(float(v)) for v in arr.flat)
    return f"[{vals}]"
