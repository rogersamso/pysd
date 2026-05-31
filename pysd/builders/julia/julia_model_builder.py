"""
Translates a PySD AbstractModel into a standalone Julia file that uses
ModelingToolkit.jl.  The generated file requires no PySD or Python at runtime.

Entry point::

    from pysd.builders.julia.julia_model_builder import JuliaModelBuilder
    path = JuliaModelBuilder(abstract_model).build_model()
"""
from __future__ import annotations

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
    TrendStructure,
    ForecastStructure,
    SampleIfTrueStructure,
    GetConstantsStructure,
    GetDataStructure,
    GetLookupsStructure,
    AllocateAvailableStructure,
    AllocateByPriorityStructure,
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

        self.namespace = JuliaNamespaceManager()
        self.inline_registry = InlineLookupRegistry()
        self.needed_helpers: Set[str] = set()

        # Accumulated declarations
        self.stock_decls: List[str] = []
        self.aux_decls: List[str] = []
        self.param_decls: List[str] = []
        self.lookup_const_decls: List[str] = []
        self.lookup_func_decls: List[str] = []
        self.lookup_register_decls: List[str] = []
        self.u0_entries: List[str] = []
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

        # Second pass: process non-INITIAL elements so u0_entries is populated
        # before INITIAL() elements are resolved (they look up stock initial values).
        initial_elems = []
        for elem in self.abstract_elements:
            identifier = self.namespace.namespace[elem.name]
            is_control = isinstance(elem, AbstractControlElement)
            comp = elem.components[0] if elem.components else None
            if comp is not None and isinstance(comp.ast, InitialStructure):
                initial_elems.append((elem, identifier, is_control))
                continue
            eqs = self._process_element(elem, identifier, is_control)
            self.built_elements[identifier] = (eqs, is_control)

        # Third pass: INITIAL elements (u0_entries now complete)
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

        visitor = JuliaASTVisitor(self.namespace, self.inline_registry, self.needed_helpers)

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
            self.stock_decls.append(f"@variables {identifier}(t)")
            self.u0_entries.append(f"{identifier} => {initial_expr}")
            return [f"D({identifier}) ~ {flow_expr}"]

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

        # ---- DELAY FIXED (not supported) --------------------------------
        if isinstance(ast, DelayFixedStructure):
            warn(
                f"DELAY FIXED for '{elem.name}' is not supported in the Julia builder. "
                "Falling back to identity (output = input)."
            )
            self.aux_decls.append(f"@variables {identifier}(t)")
            return [f"{identifier} ~ {visitor.visit(ast.input)}"]

        # ---- Unsupported structures ------------------------------------
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
            self.param_decls.append(f"@parameters {identifier} = {value_expr}")
            return []

        # ---- Data component (external time-series) ---------------------
        if isinstance(comp, AbstractData):
            warn(
                f"Data component '{elem.name}' references external data, which is "
                "not supported in the Julia builder — emitting 0.0 placeholder."
            )
            self.aux_decls.append(f"@variables {identifier}(t)")
            return [f"# DATA: {identifier} ~ 0.0"]

        # ---- Auxiliary variable (algebraic) ----------------------------
        rhs_expr = visitor.visit(ast)
        if is_control:
            if identifier in self.control_vals:
                self.control_vals[identifier] = rhs_expr
            return []
        self.aux_decls.append(f"@variables {identifier}(t)")
        return [f"{identifier} ~ {rhs_expr}"]

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

    def _resolve_initial_value(self, inner_ast) -> Optional[str]:
        """Return the t=0 value of *inner_ast* as a Julia literal, or None.

        Handles:
        * Numeric literals
        * References to stocks (in ``u0_entries``)
        * References to parameters/constants
        * References to auxiliaries whose own equation chains back to a stock
          (one level of indirection, e.g. ``INITIAL(InflowA)`` where
          ``InflowA ~ StockA`` and StockA has a known initial condition)
        """
        from pysd.builders.julia.julia_expressions_builder import format_number
        if isinstance(inner_ast, (int, float)):
            return format_number(inner_ast)
        if isinstance(inner_ast, ReferenceStructure):
            return self._resolve_ref_initial(inner_ast.reference, depth=2)
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
        uses = ["ModelingToolkit", "OrdinaryDiffEq", "OrdinaryDiffEqLowOrderRK"]
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
        if self.stock_decls:
            lines.append("# Stocks (state variables)")
            lines.extend(self.stock_decls)
        if self.aux_decls:
            lines.append("\n# Auxiliary variables")
            lines.extend(self.aux_decls)
        if self.param_decls:
            lines.append("\n# Parameters")
            lines.extend(self.param_decls)
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
