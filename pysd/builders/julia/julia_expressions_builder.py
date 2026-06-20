"""
Converts AMR Abstract Syntax Tree nodes to Julia expression strings.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple
from warnings import warn

from pysd.translators.structures.abstract_expressions import (
    AbstractSyntax,
    AllocateAvailableStructure,
    AllocateByPriorityStructure,
    ArithmeticStructure,
    CallStructure,
    DataStructure,
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
    SubscriptsReferenceStructure,
    TrendStructure,
)

# ---------------------------------------------------------------------------
# Operator tables
# ---------------------------------------------------------------------------

# Vensim arithmetic operator  →  Julia operator
ARITHMETIC_OPS: dict = {
    "+": "+",
    "-": "-",
    "negative": "-",   # Vensim unary negation AST operator
    "*": "*",
    "/": "/",
    "^": "^",
    "**": "^",
    "mod": "mod",
}

# Vensim logic operator  →  Julia operator
LOGIC_OPS: dict = {
    "=": "==",
    "<>": "!=",
    "<": "<",
    ">": ">",
    "<=": "<=",
    ">=": ">=",
    ":AND:": "&&",
    ":OR:": "||",
    ":NOT:": "!",
    "AND": "&&",
    "OR": "||",
    "NOT": "!",
}

# Vensim built-in function name  →  Julia function name
# Keys are matched after .upper(), so include both "SPACE FORM" and "UNDERSCORE_FORM"
# because the Vensim parser may store names either way.
BUILTIN_FUNCTIONS: dict = {
    # Basic math
    "ABS": "abs",
    "EXP": "exp",
    "LN": "log",
    "SQRT": "sqrt",
    "SIN": "sin",
    "COS": "cos",
    "TAN": "tan",
    "ARCSIN": "asin",
    "ARCCOS": "acos",
    "ARCTAN": "atan",
    "SINH": "sinh",
    "COSH": "cosh",
    "TANH": "tanh",
    "INTEGER": "_trunc",
    "INT": "_trunc",
    "POWER": "_power",
    "MIN": "min",
    "MAX": "max",
    "MODULO": "mod",
    "QUANTUM": "_quantum",
    "PI": "_pi",
    # Control flow — parser stores as "if_then_else" (underscores)
    "IF THEN ELSE": "ifelse",
    "IF_THEN_ELSE": "ifelse",
    # Array operations
    "SUM": "sum",
    "PROD": "prod",
    "VMAX": "maximum",
    "VMIN": "minimum",
    "ELMCOUNT": "_elmcount",   # resolved to literal size by caller
    "INVERT MATRIX": "inv",
    "INVERT_MATRIX": "inv",
    "TRANSPOSE": "transpose",
    # ACTIVE INITIAL(expr, initial) — for ODE simulation just return expr
    "ACTIVE INITIAL": "_active_initial",
    "ACTIVE_INITIAL": "_active_initial",
    # SD helpers emitted into the generated file
    "LOG": "_log_base",
    "XIDZ": "_xidz",
    "ZIDZ": "_zidz",
    "PULSE": "_pulse",
    "PULSE TRAIN": "_pulse_train",
    "PULSE_TRAIN": "_pulse_train",
    "RAMP": "_ramp",
    "STEP": "_step",
    "WITH LOOKUP": "_with_lookup",
    "WITH_LOOKUP": "_with_lookup",
    # XMILE pulse/ramp variants
    "XPULSE": "_xpulse",
    "XPULSE_TRAIN": "_xpulse_train",
    "XRAMP": "_xramp",
    # Random functions
    "RANDOM 0 1": "_random_0_1",
    "RANDOM_0_1": "_random_0_1",
    "RANDOM UNIFORM": "_random_uniform",
    "RANDOM_UNIFORM": "_random_uniform",
    "RANDOM NORMAL": "_random_normal",
    "RANDOM_NORMAL": "_random_normal",
    "RANDOM EXPONENTIAL": "_random_exponential",
    "RANDOM_EXPONENTIAL": "_random_exponential",
    # Vector operations
    "VECTOR SELECT": "_vector_select",
    "VECTOR_SELECT": "_vector_select",
    "VECTOR SORT ORDER": "_vector_sort_order",
    "VECTOR_SORT_ORDER": "_vector_sort_order",
    "VECTOR REORDER": "_vector_reorder",
    "VECTOR_REORDER": "_vector_reorder",
    "VECTOR RANK": "_vector_rank",
    "VECTOR_RANK": "_vector_rank",
    # Time value
    "GET TIME VALUE": "_get_time_value",
    "GET_TIME_VALUE": "_get_time_value",
}

# One-line Julia implementations for helper functions.
# All conditions use `ifelse` + `&`/`|` instead of `?:` / `&&` / `||` so
# they remain valid when called with symbolic (Num) arguments inside MTK equations.
HELPER_IMPLEMENTATIONS: dict = {
    # Base.trunc is not available as a symbolic primitive in MTK.
    # Register a thin wrapper so INTEGER(x) works inside equations.
    "_trunc": "_trunc(x::Real) = Base.trunc(x)\n@register_symbolic _trunc(x::Real)",
    "_log_base": "_log_base(x, base) = log(base, x)",
    "_xidz": "_xidz(x, y, z) = ifelse(iszero(y), z, x / y)",
    "_zidz": "_zidz(x, y) = ifelse(iszero(y), 0.0, x / y)",
    "_pulse": (
        "_pulse(t_now, start, width) = "
        "ifelse((t_now >= start) & (t_now < start + width), 1.0, 0.0)"
    ),
    # NOTE: the Vensim parser reorders PULSE TRAIN(start, width, interval, end)
    # to CallStructure arguments (start, interval, width, end).
    "_pulse_train": (
        "_pulse_train(t_now, start, interval, width, end_time) = "
        "ifelse((t_now >= start) & (t_now <= end_time) & "
        "(mod(t_now - start, interval) < width), 1.0, 0.0)"
    ),
    "_ramp": (
        "_ramp(t_now, slope, start_time, end_time=Inf) = "
        "slope * max(0.0, min(t_now - start_time, end_time - start_time))"
    ),
    "_step": (
        "_step(t_now, height, step_time) = "
        "ifelse(t_now >= step_time, float(height), 0.0)"
    ),
    # Vensim logical operators — values are always 0.0 (false) or 1.0 (true).
    # Return Symbolic{Bool} via comparisons so the result can be used as the
    # condition of a symbolic `ifelse` in MTK equations.
    "_logical_and": "_logical_and(a, b) = (a > 0.5) & (b > 0.5)",
    "_logical_or": "_logical_or(a, b) = (a > 0.5) | (b > 0.5)",
    "_logical_not": "_logical_not(a) = !(a > 0.5)",
    # ACTIVE INITIAL(expr, initial) — in ODE mode expr is always live;
    # we just return expr (the first argument).
    "_active_initial": "_active_initial(expr, initial) = expr",
    # INVERT_MATRIX helpers — registered as symbolic black boxes so Symbolics
    # does not attempt symbolic matrix algebra (which hangs for large matrices).
    # At solve time the concrete array is passed and inv is computed numerically.
    "_inv_mat2d_elem": (
        "function _inv_mat2d_elem(mat::AbstractMatrix, i::Int, j::Int)\n"
        "    return inv(mat)[i, j]\n"
        "end\n"
        "@register_symbolic _inv_mat2d_elem(mat::AbstractMatrix, i::Int, j::Int)"
    ),
    "_inv_mat3d_elem": (
        "function _inv_mat3d_elem(mat::AbstractArray, b::Int, i::Int, j::Int)\n"
        "    return inv(mat[b, :, :])[i, j]\n"
        "end\n"
        "@register_symbolic _inv_mat3d_elem(mat::AbstractArray, b::Int, i::Int, j::Int)"
    ),
    "_power": "_power(x, y) = x ^ y\n@register_symbolic _power(x::Real, y::Real)",
    "_quantum": (
        "_quantum(a, b) = ifelse(b < 1e-6, float(a), b * _trunc(a / b))\n"
        "@register_symbolic _quantum(a::Real, b::Real)"
    ),
    "_pi": "_pi() = Base.MathConstants.pi",
    # XMILE variants: Xpulse has (start, magnitude), Xramp has (slope, start)
    "_xpulse": (
        "_xpulse(t_now, start, magnitude) = "
        "ifelse((t_now >= start) & (t_now < start + magnitude), magnitude, 0.0)"
    ),
    "_xpulse_train": (
        "_xpulse_train(t_now, start, interval, magnitude) = "
        "ifelse((t_now >= start) & "
        "(mod(t_now - start, interval) < magnitude), magnitude, 0.0)"
    ),
    "_xramp": (
        "_xramp(t_now, slope, start_time) = "
        "slope * max(0.0, t_now - start_time)"
    ),
    # Random functions — opaque wrappers so MTK calls them at every timestep
    "_random_0_1": (
        "_random_0_1() = Base.rand()\n"
        "@register_symbolic _random_0_1()"
    ),
    "_random_uniform": (
        "_random_uniform(lo, hi, _seed) = lo + (hi - lo) * Base.rand()\n"
        "@register_symbolic _random_uniform(lo::Real, hi::Real, _seed::Real)"
    ),
    "_random_normal": (
        "function _random_normal(lo, hi, mean, std, _seed)\n"
        "    x = mean + std * Base.randn()\n"
        "    return clamp(x, lo, hi)\n"
        "end\n"
        "@register_symbolic _random_normal(lo::Real, hi::Real, mean::Real, std::Real, _seed::Real)"
    ),
    "_random_exponential": (
        "function _random_exponential(lo, hi, mean, _seed)\n"
        "    x = lo + mean * Base.randexp()\n"
        "    return clamp(x, lo, hi)\n"
        "end\n"
        "@register_symbolic _random_exponential(lo::Real, hi::Real, mean::Real, _seed::Real)"
    ),
    # Vector operations
    "_vector_select": (
        "function _vector_select(sel_vec, expr_vec, miss_val, action)\n"
        "    selected = [expr_vec[i] for i in eachindex(sel_vec) if sel_vec[i] != 0]\n"
        "    isempty(selected) && return miss_val\n"
        "    action == 0 && return selected[1]\n"
        "    action == 1 && return sum(selected)\n"
        "    action == 2 && return maximum(selected)\n"
        "    action == 3 && return minimum(selected)\n"
        "    action == 4 && return sum(selected) / length(selected)\n"
        "    return miss_val\n"
        "end"
    ),
    "_vector_sort_order": (
        "_vector_sort_order(vec, dir) = "
        "Float64.(ifelse(dir > 0, sortperm(vec), sortperm(vec, rev=true)))"
    ),
    "_vector_reorder": (
        "_vector_reorder(vec, order) = vec[Int.(order)]"
    ),
    "_vector_rank": (
        "_vector_rank(vec, dir) = "
        "Float64.(invperm(ifelse(dir > 0, sortperm(vec), sortperm(vec, rev=true))))"
    ),
    "_get_time_value": (
        "_get_time_value(t_now, lookup_fn, lo, hi) = "
        "lookup_fn(clamp(t_now, lo, hi))"
    ),
}

# Helper functions that receive the current time *t* as their first argument
_TIME_HELPERS: frozenset = frozenset({
    "_pulse", "_pulse_train", "_ramp", "_step",
    "_xpulse", "_xpulse_train", "_xramp", "_get_time_value",
})


# ---------------------------------------------------------------------------
# Lookup-table utilities
# ---------------------------------------------------------------------------

class InlineLookupRegistry:
    """Collects inline lookup tables encountered during AST traversal.

    Each inline lookup is given a unique name so the generated file can
    declare a named interpolant constant and a one-argument wrapper.
    """

    def __init__(self) -> None:
        self._entries: List[Tuple[str, tuple, tuple, str]] = []
        self._counter: int = 0

    def register(self, xs: tuple, ys: tuple, itp_type: str) -> str:
        """Register an inline lookup table and return its function name."""
        self._counter += 1
        name = f"_inline_lookup_{self._counter}"
        self._entries.append((name, xs, ys, itp_type))
        return name

    @property
    def entries(self) -> List[Tuple[str, tuple, tuple, str]]:
        return list(self._entries)


def format_number(value: Any) -> str:
    """Format a Python numeric value as a Julia floating-point literal."""
    if isinstance(value, float):
        if value == float("inf"):
            return "Inf"
        if value == float("-inf"):
            return "-Inf"
        if value != value:  # NaN
            return "NaN"
    return repr(float(value))


def format_vector(values: tuple) -> str:
    """Format a tuple of numbers as a Julia Float64 vector literal."""
    return "[" + ", ".join(format_number(v) for v in values) + "]"


def lookup_interpolation_code(
    name: str, xs: tuple, ys: tuple, itp_type: str
) -> Tuple[str, str, str]:
    """Return ``(const_decl, func_decl, register_decl)`` for a named lookup table.

    ``itp_type`` controls the DataInterpolations constructor:

    * ``"interpolate"`` / ``"extrapolate"`` → ``LinearInterpolation`` (default)
    * ``"hold_forward"``  → ``ConstantInterpolation`` (previous-value hold)
    * ``"hold_backward"`` → ``ConstantInterpolation(...; dir=:right)`` (next-value hold)

    ``@register_symbolic`` tells ModelingToolkit that this is an opaque
    external function so it is called at every timestep rather than being
    constant-folded during structural_simplify.
    """
    xs_vec = format_vector(xs)
    ys_vec = format_vector(ys)
    itp_name = f"{name}_itp"

    _extrap = "ExtrapolationType.Constant"
    if itp_type == "hold_forward":
        const_decl = (
            f"const {itp_name} = ConstantInterpolation({ys_vec}, {xs_vec};"
            f" extrapolation_left = {_extrap}, extrapolation_right = {_extrap})"
        )
    elif itp_type == "hold_backward":
        const_decl = (
            f"const {itp_name} = ConstantInterpolation({ys_vec}, {xs_vec};"
            f" dir=:right, extrapolation_left = {_extrap}, extrapolation_right = {_extrap})"
        )
    else:
        # "interpolate", "extrapolate", or any unrecognised type → linear
        const_decl = (
            f"const {itp_name} = LinearInterpolation({ys_vec}, {xs_vec};"
            f" extrapolation_left = {_extrap}, extrapolation_right = {_extrap})"
        )

    func_decl = f"{name}(x) = {itp_name}(x)"
    register_decl = f"@register_symbolic {name}(x::Real)"
    return const_decl, func_decl, register_decl


# ---------------------------------------------------------------------------
# AST visitor
# ---------------------------------------------------------------------------

class JuliaASTVisitor:
    """Recursively converts an AMR AST node to a Julia expression string.

    Parameters
    ----------
    namespace:
        A :class:`~pysd.builders.julia.namespace.JuliaNamespaceManager`.
    inline_registry:
        Accumulator for inline lookup tables found during traversal.
    needed_helpers:
        Mutable set; the visitor adds the names of any helper functions
        (``_pulse``, ``_xidz``, …) it emits, so the builder can include
        their implementations in the generated file.
    """

    def __init__(
        self,
        namespace,
        inline_registry: InlineLookupRegistry,
        needed_helpers: Set[str],
        active_subs: Optional[Dict[str, str]] = None,
        var_dims: Optional[Dict[str, List[str]]] = None,
        subs_sizes: Optional[Dict[str, int]] = None,
        subs_elems: Optional[Dict[str, List[str]]] = None,
        lookup_names: Optional[Set[str]] = None,
        root=None,
    ) -> None:
        self.namespace = namespace
        self.registry = inline_registry
        self.needed_helpers = needed_helpers
        # active_subs: dim_name -> julia index variable (e.g. {"sector": "_i"})
        self.active_subs = active_subs or {}
        # _clean_active_subs: normalised-dim-name -> julia index variable, for
        # case-insensitive lookup when subscript names appear as bare references.
        self._clean_active_subs = {
            re.sub(r"[^a-z0-9_]", "_", k.lower()): v
            for k, v in self.active_subs.items()
        }
        # var_dims: julia identifier -> list of dim names it is subscripted over
        self.var_dims = var_dims or {}
        # subs_sizes: subscript range name -> integer size (for ELMCOUNT)
        self.subs_sizes = subs_sizes or {}
        # _clean_subs_sizes: normalised name -> size, for case-insensitive ELMCOUNT lookup
        self._clean_subs_sizes = {
            re.sub(r"[^a-z0-9_]", "_", k.lower()): v
            for k, v in self.subs_sizes.items()
        }
        # lookup_names: identifiers that are GET DATA / GET LOOKUPS functions
        # — bare references to these should be auto-called as f(t) or f(i, t)
        self.lookup_names = lookup_names or set()
        # subs_elems: range_name -> ordered list of element labels
        self.subs_elems = subs_elems or {}
        # Pre-compute element_label -> {range_name: 1-based-index} for fast lookups
        self._elem_index: Dict[str, Dict[str, int]] = {}
        for rng, elems in self.subs_elems.items():
            for i, lbl in enumerate(elems):
                if lbl not in self._elem_index:
                    self._elem_index[lbl] = {}
                self._elem_index[lbl][rng] = i + 1
        # root: Path to the model directory (for reading external files)
        self._root = root

    def _jl_n(self, dim_name: str) -> str:
        """Julia constant name for the size of *dim_name* (``N_DIMNAME``)."""
        return "N_" + re.sub(r"[^a-z0-9]", "_", dim_name.lower()).upper()

    def _collect_bang_subs(self, node) -> List[str]:
        """Return unique '!'-subscript strings found anywhere in *node*'s subtree."""
        result: List[str] = []
        seen: set = set()

        def _scan(n: Any) -> None:
            if isinstance(n, ReferenceStructure):
                subs = (
                    n.subscripts.subscripts
                    if n.subscripts is not None and hasattr(n.subscripts, "subscripts")
                    else []
                )
                for s in subs:
                    if s.endswith("!") and s not in seen:
                        seen.add(s)
                        result.append(s)
            elif isinstance(n, CallStructure):
                fsubs = (
                    n.function.subscripts.subscripts
                    if n.function.subscripts is not None
                    and hasattr(n.function.subscripts, "subscripts")
                    else []
                )
                for s in fsubs:
                    if s.endswith("!") and s not in seen:
                        seen.add(s)
                        result.append(s)
                for arg in n.arguments:
                    _scan(arg)
            elif isinstance(n, (ArithmeticStructure, LogicStructure)):
                for arg in n.arguments:
                    _scan(arg)

        _scan(node)
        return result

    def _with_extra_subs(self, extra: Dict[str, str]) -> "JuliaASTVisitor":
        """Return a child visitor with *extra* entries added to active_subs."""
        return JuliaASTVisitor(
            self.namespace,
            self.registry,
            self.needed_helpers,
            active_subs={**self.active_subs, **extra},
            var_dims=self.var_dims,
            subs_sizes=self.subs_sizes,
            subs_elems=self.subs_elems,
            lookup_names=self.lookup_names,
            root=self._root,
        )

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def visit(self, node: Any) -> str:
        """Return the Julia expression string for *node*."""
        if node is None:
            return "0.0"

        if isinstance(node, bool):
            return "true" if node else "false"

        if isinstance(node, (int, float)):
            return format_number(node)

        if isinstance(node, str):
            # Bare strings occasionally appear as numeric literals in the AMR
            try:
                return format_number(float(node))
            except ValueError:
                return repr(node)

        # numpy arrays (e.g. from GetConstantsStructure values embedded inline)
        try:
            import numpy as np
            if isinstance(node, np.ndarray):
                if node.ndim == 0:
                    return format_number(float(node))
                if node.ndim == 1:
                    vals = ", ".join(format_number(float(v)) for v in node)
                    return f"[{vals}]"
                # Higher dims: flatten
                vals = ", ".join(format_number(float(v)) for v in node.flat)
                return f"[{vals}]"
        except ImportError:
            pass

        if isinstance(node, ArithmeticStructure):
            return self._arithmetic(node)

        if isinstance(node, LogicStructure):
            return self._logic(node)

        if isinstance(node, ReferenceStructure):
            return self._reference(node)

        if isinstance(node, CallStructure):
            return self._call(node)

        if isinstance(node, InlineLookupsStructure):
            return self._inline_lookup(node)

        if isinstance(node, InitialStructure):
            # INITIAL(x) — in an ODE context we just use the expression value
            return self.visit(node.initial)

        if isinstance(node, GameStructure):
            # GAME passes through in simulation (non-interactive) mode
            return self.visit(node.expression)

        if isinstance(node, GetConstantsStructure):
            # GetConstantsStructure nested inside an expression — read the
            # external value at translation time and emit it as a Julia literal.
            try:
                from pysd.py_backend.external import ExtConstant
                from pysd.builders.julia.julia_model_builder import _format_julia_value
                import pathlib as _pathlib
                root = self._root or _pathlib.Path(".")
                ext = ExtConstant(
                    file_name=node.file,
                    tab=node.tab,
                    cell=node.cell,
                    coords={},
                    root=root,
                    final_coords={},
                    py_name="_inline_const",
                )
                ext.initialize()
                return _format_julia_value(ext.data)
            except Exception as exc:
                warn(
                    f"GetConstantsStructure inside expression could not be read "
                    f"({exc}); emitting placeholder 0.0."
                )
                return "0.0"

        if isinstance(node, SubscriptsReferenceStructure):
            # A subscript reference used as a value — emit the first subscript name.
            # This handles cases like ELMCOUNT(SECTORS) where the parser produces
            # a bare SubscriptsReferenceStructure for the subscript range name.
            if node.subscripts:
                ref = node.subscripts[0]
                return self.namespace.get(ref) or repr(ref)
            return "0.0"

        # Structures that are handled at the element level should not appear
        # inside other expressions; warn and emit a placeholder.
        warn(
            f"Unsupported AST node type '{type(node).__name__}' inside expression "
            "— emitting placeholder 0.0."
        )
        return "0.0"

    # ------------------------------------------------------------------
    # Node handlers
    # ------------------------------------------------------------------

    def _arithmetic(self, node: ArithmeticStructure) -> str:
        args = [self.visit(a) for a in node.arguments]
        ops = node.operators

        if len(args) == 1:
            # Unary operator (negation)
            op = ARITHMETIC_OPS.get(ops[0], ops[0])
            return f"({op}{args[0]})"

        parts = [args[0]]
        for op, arg in zip(ops, args[1:]):
            parts.append(ARITHMETIC_OPS.get(op, op))
            parts.append(arg)
        return "(" + " ".join(parts) + ")"

    def _logic(self, node: LogicStructure) -> str:
        args = [self.visit(a) for a in node.arguments]
        ops = node.operators

        # AND / OR / NOT: use helper functions so the expression remains valid
        # when called with symbolic (Num) arguments inside MTK equations.
        # Julia's &&/|| require a concrete Bool; the helpers use ifelse instead.
        if len(args) == 1:
            op_key = ops[0].upper().strip(":")
            if op_key in ("NOT", ":NOT:"):
                self.needed_helpers.add("_logical_not")
                return f"_logical_not({args[0]})"
            op = LOGIC_OPS.get(ops[0], ops[0])
            return f"({op}{args[0]})"

        result = args[0]
        for op, arg in zip(ops, args[1:]):
            op_key = op.upper().strip(":")
            if op_key in ("AND", ":AND:"):
                self.needed_helpers.add("_logical_and")
                result = f"_logical_and({result}, {arg})"
            elif op_key in ("OR", ":OR:"):
                self.needed_helpers.add("_logical_or")
                result = f"_logical_or({result}, {arg})"
            else:
                julia_op = LOGIC_OPS.get(op, op)
                result = f"({result} {julia_op} {arg})"
        return result

    def _reference(self, node: ReferenceStructure) -> str:
        # Subscript dimension names appear as bare references in equations like
        # I_Matrix[s, s1] = IF_THEN_ELSE(s = s1, 1, 0).  When inside an active
        # subscript loop, emit the corresponding loop-index variable directly.
        if self._clean_active_subs:
            clean_ref = re.sub(r"[^a-z0-9_]", "_", node.reference.lower())
            idx_var = self._clean_active_subs.get(clean_ref)
            if idx_var is not None:
                return idx_var

        julia_name = self.namespace.get(node.reference)
        if julia_name is None:
            warn(
                f"Variable '{node.reference}' not found in namespace; "
                "using a sanitised fallback identifier."
            )
            julia_name = re.sub(r"[^a-z0-9_]", "_", node.reference.lower())
        # Apply subscript indices.  Two sources:
        #
        # (A) Explicit subscripts in the AST node  (e.g. share_FEH[solids])
        #     Each entry is either a range name (→ use active loop variable) or
        #     a specific element label (→ resolve to 1-based numeric index).
        # (B) Active loop variables from the enclosing comprehension context
        #     (only when the AST carries no explicit subscripts).
        node_subs = (
            node.subscripts.subscripts
            if node.subscripts is not None and hasattr(node.subscripts, "subscripts")
            else []
        )

        # If this identifier is a GET DATA/LOOKUPS function referenced bare (no
        # call syntax), auto-call it with the active subscript indices + t.
        if julia_name in self.lookup_names and not node_subs:
            dims = self.var_dims.get(julia_name, [])
            if self.active_subs:
                indices = [self.active_subs[d] for d in dims if d in self.active_subs]
                return f"{julia_name}({', '.join(indices + ['t'])})"
            elif dims:
                # Scalar context, subscripted lookup: broadcast over all dims
                idx_vars = [f"_ii{k}" for k in range(len(dims))]
                ranges = ", ".join(
                    f"{iv} in 1:{self._jl_n(d)}" for iv, d in zip(idx_vars, dims)
                )
                return f"[{julia_name}({', '.join(idx_vars + ['t'])}) for {ranges}]"
            else:
                return f"{julia_name}(t)"

        if node_subs:
            # (A) Explicit: resolve each subscript to a Julia index expression.
            var_dims_list = self.var_dims.get(julia_name, [])

            # Aggregation subscripts (ending with '!') are handled here.
            # Normally the outer _call for sum/prod/vmax/vmin pre-populates
            # active_subs for ! dims so that all references sharing the same !
            # subscript are inside ONE comprehension (not separate comprehensions
            # multiplied together).  If an ! dim is already in active_subs we
            # reuse that loop variable; otherwise we generate a new comprehension.
            if any(sub.endswith("!") for sub in node_subs):
                bang_ranges: List[str] = []
                ii_count = 0
                dim_to_idx: Dict[str, str] = {}

                for sub in node_subs:
                    clean_sub = re.sub(r"[^a-z0-9_]", "_", sub.lower())
                    if sub.endswith("!"):
                        bare = sub[:-1]
                        clean_bare = re.sub(r"[^a-z0-9_]", "_", bare.lower())
                        if clean_bare in self._clean_active_subs:
                            # Already being iterated by an outer loop (added by sum
                            # handler) — reuse the existing loop variable.
                            iv = self._clean_active_subs[clean_bare]
                            dim_to_idx[clean_bare] = iv
                        else:
                            # Find the matching dim in var_dims_list (by normalised name)
                            dim_name = next(
                                (d for d in var_dims_list
                                 if re.sub(r"[^a-z0-9_]", "_", d.lower()) == clean_bare),
                                bare,
                            )
                            iv = f"_ii{ii_count}"
                            ii_count += 1
                            dim_to_idx[re.sub(r"[^a-z0-9_]", "_", dim_name.lower())] = iv
                            bang_ranges.append(f"{iv} in 1:{self._jl_n(dim_name)}")
                    elif sub in self.active_subs:
                        dim_to_idx[clean_sub] = self.active_subs[sub]
                    elif sub in self.subs_elems:
                        idx_var = self.active_subs.get(sub)
                        if idx_var:
                            dim_to_idx[clean_sub] = idx_var
                    else:
                        if sub in self._elem_index:
                            # Key dim_to_idx by the variable's DIMENSION NAME (not
                            # the element label) so the index assembly over
                            # var_dims_list can find it.  Also prefer the variable's
                            # own declared dim to avoid picking a larger parent range.
                            target_dim = None
                            for d in var_dims_list:
                                if d in self._elem_index[sub]:
                                    target_dim = d
                                    break
                            if target_dim is None:
                                target_dim = next(iter(self._elem_index[sub]))
                            clean_dim = re.sub(r"[^a-z0-9_]", "_", target_dim.lower())
                            dim_to_idx[clean_dim] = str(self._elem_index[sub][target_dim])

                # Assemble indices in var_dims_list (declaration) order.
                # When a dim name doesn't match any key in dim_to_idx (common when
                # Vensim aliases differ, e.g. 'sectors' decl vs 'sectors1' in ref),
                # fall back to size-matching then positional assignment.
                bang_iv_pool = [
                    dim_to_idx[re.sub(r"[^a-z0-9_]", "_", sub[:-1].lower())]
                    for sub in node_subs
                    if sub.endswith("!")
                    and re.sub(r"[^a-z0-9_]", "_", sub[:-1].lower()) in dim_to_idx
                ]
                used_ivars: set = set()
                if var_dims_list:
                    indices = []
                    for d in var_dims_list:
                        clean_d = re.sub(r"[^a-z0-9_]", "_", d.lower())
                        if clean_d in dim_to_idx:
                            iv = dim_to_idx[clean_d]
                            indices.append(iv)
                            used_ivars.add(iv)
                        else:
                            # Size-based fallback first
                            d_size = (self.subs_sizes.get(d, 0) or
                                      self._clean_subs_sizes.get(clean_d, 0))
                            matched = None
                            for sub in node_subs:
                                if not sub.endswith("!"):
                                    continue
                                bare = sub[:-1]
                                cb = re.sub(r"[^a-z0-9_]", "_", bare.lower())
                                iv = dim_to_idx.get(cb)
                                if iv is None or iv in used_ivars:
                                    continue
                                bare_size = (self.subs_sizes.get(bare, 0) or
                                             self._clean_subs_sizes.get(cb, 0))
                                if d_size > 0 and bare_size == d_size:
                                    matched = iv
                                    used_ivars.add(iv)
                                    break
                            if matched is None:
                                # Positional fallback
                                for iv in bang_iv_pool:
                                    if iv not in used_ivars:
                                        matched = iv
                                        used_ivars.add(iv)
                                        break
                            if matched is not None:
                                indices.append(matched)
                else:
                    # No var_dims info: use node_subs order as fallback
                    indices = []
                    for sub in node_subs:
                        key = re.sub(
                            r"[^a-z0-9_]", "_",
                            (sub[:-1] if sub.endswith("!") else sub).lower(),
                        )
                        if key in dim_to_idx:
                            indices.append(dim_to_idx[key])

                inner = f"{julia_name}[{', '.join(indices)}]"
                if bang_ranges:
                    for_clause = ", ".join(bang_ranges)
                    return f"[{inner} for {for_clause}]"
                else:
                    # All ! dims were already active — no new comprehension
                    return inner

            indices = []
            # Track which active loop vars have been consumed by alignment so
            # two different range names (e.g. sectors_a_matrix and
            # sectors_a_matrix1) that both map to the same element-set don't
            # both resolve to the same variable.
            used_align_vars: set = set()
            for pos, sub in enumerate(node_subs):
                if sub in self.active_subs:
                    # Range name matching an active loop variable
                    lv = self.active_subs[sub]
                    indices.append(lv)
                    used_align_vars.add(lv)
                elif sub in self.subs_elems:
                    # Range name with all elements — use active loop var if available.
                    # First try direct name match, then fall back to element-set
                    # alignment (handles aliases like sectors_a_matrix ↔ sectors).
                    idx_var = self.active_subs.get(sub)
                    if idx_var and idx_var not in used_align_vars:
                        indices.append(idx_var)
                        used_align_vars.add(idx_var)
                    elif not idx_var:
                        # Aligned range: find an active range with the same elements
                        sub_elems = self.subs_elems.get(sub, [])
                        aligned = None
                        # Element-set match (exact) — prefer first unused
                        for ar, lv in self.active_subs.items():
                            if lv in used_align_vars:
                                continue
                            if sub_elems and self.subs_elems.get(ar, []) == sub_elems:
                                aligned = lv
                                break
                        # Size match fallback
                        if aligned is None and sub_elems:
                            sub_size = len(sub_elems)
                            for ar, lv in self.active_subs.items():
                                if lv in used_align_vars:
                                    continue
                                if len(self.subs_elems.get(ar, [])) == sub_size:
                                    aligned = lv
                                    break
                        if aligned:
                            indices.append(aligned)
                            used_align_vars.add(aligned)
                        # else: genuinely unresolvable — skip (rare)
                else:
                    # Specific element label → numeric index in the variable's own dim.
                    # Prefer the variable's declared dim at this position so that
                    # sub-ranges (e.g. matter_final_sources) yield a local index,
                    # not the index from a larger parent range (e.g. final_sources).
                    parent_range = None
                    if pos < len(var_dims_list):
                        candidate = var_dims_list[pos]
                        if sub in self._elem_index and candidate in self._elem_index[sub]:
                            parent_range = candidate
                    if parent_range is None:
                        # Fallback: any of the variable's declared dims that contain sub
                        for rng in var_dims_list:
                            if sub in self._elem_index and rng in self._elem_index[sub]:
                                parent_range = rng
                                break
                    if parent_range is None and sub in self._elem_index:
                        # Last resort: first known range (may be wrong for sub-ranges)
                        parent_range = next(iter(self._elem_index[sub]))
                    if parent_range is not None and sub in self._elem_index:
                        indices.append(str(self._elem_index[sub][parent_range]))
                    elif sub in self._elem_index:
                        idx_val = next(iter(self._elem_index[sub].values()))
                        indices.append(str(idx_val))
            if indices:
                # GET DATA / LOOKUPS functions must use call syntax f(i, t),
                # not array-index syntax f[i].
                if julia_name in self.lookup_names:
                    return f"{julia_name}({', '.join(indices + ['t'])})"
                julia_name = julia_name + "[" + ", ".join(indices) + "]"
        elif self.active_subs and self.var_dims:
            # (B) No explicit subscripts: apply active loop variables.
            dims = self.var_dims.get(julia_name, [])
            indices = [self.active_subs[d] for d in dims if d in self.active_subs]
            if indices:
                julia_name = julia_name + "[" + ", ".join(indices) + "]"

        return julia_name

    def _call(self, node: CallStructure) -> str:
        func_upper = node.function.reference.upper()
        julia_func = BUILTIN_FUNCTIONS.get(func_upper)

        if julia_func is None:
            # Check whether the function name is a model variable (lookup table).
            # Vensim allows calling a lookup variable as a function:
            #   result = my_lookup_table(input_value)
            # We check the namespace and emit the variable name directly
            # (which will be a Julia interpolation function if loaded correctly).
            julia_id = self.namespace.get(node.function.reference)
            if julia_id is not None:
                args = [self.visit(a) for a in node.arguments]

                # Handle aggregation subscripts on the function reference itself,
                # e.g. f[dim1!, dim2](t) → [f(_ii0, _i0, t) for _ii0 in 1:N_DIM1]
                func_node_subs = (
                    node.function.subscripts.subscripts
                    if node.function.subscripts is not None
                    and hasattr(node.function.subscripts, "subscripts")
                    else []
                )
                if func_node_subs and any(s.endswith("!") for s in func_node_subs):
                    var_dims_list = self.var_dims.get(julia_id, [])
                    bang_ranges_c: List[str] = []
                    ii_count_c = 0
                    dim_to_idx_c: Dict[str, str] = {}
                    for sub in func_node_subs:
                        clean_sub = re.sub(r"[^a-z0-9_]", "_", sub.lower())
                        if sub.endswith("!"):
                            bare = sub[:-1]
                            clean_bare = re.sub(r"[^a-z0-9_]", "_", bare.lower())
                            if clean_bare in self._clean_active_subs:
                                # Already iterated by an outer SUM comprehension —
                                # reuse the existing loop variable, don't add a new range.
                                iv = self._clean_active_subs[clean_bare]
                                dim_to_idx_c[clean_bare] = iv
                            else:
                                dim_name = next(
                                    (d for d in var_dims_list
                                     if re.sub(r"[^a-z0-9_]", "_", d.lower()) == clean_bare),
                                    bare,
                                )
                                iv = f"_ii{ii_count_c}"
                                ii_count_c += 1
                                dim_to_idx_c[re.sub(r"[^a-z0-9_]", "_", dim_name.lower())] = iv
                                bang_ranges_c.append(f"{iv} in 1:{self._jl_n(dim_name)}")
                        elif sub in self.active_subs:
                            dim_to_idx_c[clean_sub] = self.active_subs[sub]
                        elif sub in self.subs_elems:
                            idx_var = self.active_subs.get(sub)
                            if idx_var:
                                dim_to_idx_c[clean_sub] = idx_var
                        else:
                            if sub in self._elem_index:
                                idx_val = next(iter(self._elem_index[sub].values()))
                                dim_to_idx_c[clean_sub] = str(idx_val)
                    # Assemble call indices in var_dims_list order with fallback
                    bang_iv_pool_c = [
                        dim_to_idx_c[re.sub(r"[^a-z0-9_]", "_", sub[:-1].lower())]
                        for sub in func_node_subs
                        if sub.endswith("!")
                        and re.sub(r"[^a-z0-9_]", "_", sub[:-1].lower()) in dim_to_idx_c
                    ]
                    used_ivars_c: set = set()
                    if var_dims_list:
                        call_indices = []
                        for d in var_dims_list:
                            clean_d = re.sub(r"[^a-z0-9_]", "_", d.lower())
                            if clean_d in dim_to_idx_c:
                                iv = dim_to_idx_c[clean_d]
                                call_indices.append(iv)
                                used_ivars_c.add(iv)
                            else:
                                d_size = (self.subs_sizes.get(d, 0) or
                                          self._clean_subs_sizes.get(clean_d, 0))
                                matched = None
                                for sub in func_node_subs:
                                    if not sub.endswith("!"):
                                        continue
                                    bare = sub[:-1]
                                    cb = re.sub(r"[^a-z0-9_]", "_", bare.lower())
                                    iv = dim_to_idx_c.get(cb)
                                    if iv is None or iv in used_ivars_c:
                                        continue
                                    bare_size = (self.subs_sizes.get(bare, 0) or
                                                 self._clean_subs_sizes.get(cb, 0))
                                    if d_size > 0 and bare_size == d_size:
                                        matched = iv
                                        used_ivars_c.add(iv)
                                        break
                                if matched is None:
                                    for iv in bang_iv_pool_c:
                                        if iv not in used_ivars_c:
                                            matched = iv
                                            used_ivars_c.add(iv)
                                            break
                                if matched is not None:
                                    call_indices.append(matched)
                    else:
                        call_indices = []
                        for sub in func_node_subs:
                            key = re.sub(
                                r"[^a-z0-9_]", "_",
                                (sub[:-1] if sub.endswith("!") else sub).lower(),
                            )
                            if key in dim_to_idx_c:
                                call_indices.append(dim_to_idx_c[key])
                    inner_call = f"{julia_id}({', '.join(call_indices + args)})"
                    if bang_ranges_c:
                        for_clause_c = ", ".join(bang_ranges_c)
                        return f"[{inner_call} for {for_clause_c}]"
                    else:
                        # All ! dims already active via outer comprehension.
                        return inner_call

                # Explicit function subscripts without '!': resolve each subscript
                # positionally (range name → active loop var; element label →
                # literal index), matching the logic in _reference for explicit
                # node_subs.  This handles e.g.
                #   Historic_water_use[sectors, water](Time)
                # where var_dims uses the parent dim 'sectors_and_households'
                # which doesn't appear in active_subs, but 'sectors' does.
                if func_node_subs:
                    var_dims_list = self.var_dims.get(julia_id, [])
                    call_indices: List[str] = []
                    used_align_vars_c2: set = set()
                    for pos, sub in enumerate(func_node_subs):
                        if sub in self.active_subs:
                            lv = self.active_subs[sub]
                            call_indices.append(lv)
                            used_align_vars_c2.add(lv)
                        elif sub in self.subs_elems:
                            idx_var = self.active_subs.get(sub)
                            if idx_var and idx_var not in used_align_vars_c2:
                                call_indices.append(idx_var)
                                used_align_vars_c2.add(idx_var)
                            elif not idx_var:
                                sub_elems = self.subs_elems.get(sub, [])
                                aligned: Optional[str] = None
                                for ar, lv in self.active_subs.items():
                                    if lv in used_align_vars_c2:
                                        continue
                                    if sub_elems and self.subs_elems.get(ar, []) == sub_elems:
                                        aligned = lv
                                        break
                                if aligned is None and sub_elems:
                                    sub_size = len(sub_elems)
                                    for ar, lv in self.active_subs.items():
                                        if lv in used_align_vars_c2:
                                            continue
                                        if len(self.subs_elems.get(ar, [])) == sub_size:
                                            aligned = lv
                                            break
                                if aligned:
                                    call_indices.append(aligned)
                                    used_align_vars_c2.add(aligned)
                        else:
                            parent_range: Optional[str] = None
                            if pos < len(var_dims_list):
                                candidate = var_dims_list[pos]
                                if sub in self._elem_index and candidate in self._elem_index[sub]:
                                    parent_range = candidate
                            if parent_range is None:
                                for rng in var_dims_list:
                                    if sub in self._elem_index and rng in self._elem_index[sub]:
                                        parent_range = rng
                                        break
                            if parent_range is None and sub in self._elem_index:
                                parent_range = next(iter(self._elem_index[sub]))
                            if parent_range is not None and sub in self._elem_index:
                                call_indices.append(str(self._elem_index[sub][parent_range]))
                            elif sub in self._elem_index:
                                call_indices.append(str(next(iter(self._elem_index[sub].values()))))
                    if call_indices:
                        return f"{julia_id}({', '.join(call_indices + args)})"

                if self.var_dims:
                    dims = self.var_dims.get(julia_id, [])
                    if dims:
                        if self.active_subs:
                            # Subscript comprehension context: prepend active indices.
                            # historic_gfcf(t) → historic_gfcf(_i0, t)
                            indices = [
                                self.active_subs[d] for d in dims if d in self.active_subs
                            ]
                            if indices:
                                args = indices + args
                        else:
                            # Scalar context: broadcast over all dim indices.
                            # sum(historic_labour_compensation(t))
                            # → sum([historic_labour_compensation(_ii0, t) for _ii0 in 1:N_SECTORS])
                            idx_vars = [f"_ii{k}" for k in range(len(dims))]
                            full_args = idx_vars + args
                            ranges = ", ".join(
                                f"{iv} in 1:{self._jl_n(d)}"
                                for iv, d in zip(idx_vars, dims)
                            )
                            return f"[{julia_id}({', '.join(full_args)}) for {ranges}]"
                return f"{julia_id}({', '.join(args)})"
            warn(f"Unknown Vensim function '{node.function.reference}'; using lowercase name.")
            julia_func = re.sub(r"[^a-z0-9_]", "_", node.function.reference.lower())

        # ELMCOUNT(SubscriptRange) → emit the integer literal size
        if julia_func == "_elmcount":
            if node.arguments:
                arg = node.arguments[0]
                if isinstance(arg, ReferenceStructure):
                    size = self.subs_sizes.get(arg.reference)
                    if size is None:
                        # Case-insensitive fallback (abstract model may use different
                        # casing from the expression parser)
                        clean = re.sub(r"[^a-z0-9_]", "_", arg.reference.lower())
                        size = self._clean_subs_sizes.get(clean)
                    if size is not None:
                        return str(size)
                # Fall back: try to visit the argument and return it
                return self.visit(arg)
            return "0"

        if julia_func in HELPER_IMPLEMENTATIONS:
            self.needed_helpers.add(julia_func)
            if julia_func == "_quantum":
                self.needed_helpers.add("_trunc")

        # sum/prod/vmax/vmin with ! subscripts: generate ONE comprehension that
        # covers ALL references sharing the same ! dim, rather than separate
        # per-reference comprehensions that would be multiplied/added as arrays.
        if julia_func in ("sum", "prod", "maximum", "minimum") and len(node.arguments) == 1:
            bang_subs = self._collect_bang_subs(node.arguments[0])
            new_bang_subs = [
                s for s in bang_subs
                if re.sub(r"[^a-z0-9_]", "_", s[:-1].lower())
                not in self._clean_active_subs
            ]
            if new_bang_subs:
                extra_subs: Dict[str, str] = {}
                agg_ranges: List[str] = []
                ii_cnt = 0
                for sub in new_bang_subs:
                    bare = sub[:-1]
                    clean_bare = re.sub(r"[^a-z0-9_]", "_", bare.lower())
                    iv = f"_ii{ii_cnt}"
                    ii_cnt += 1
                    extra_subs[bare] = iv
                    extra_subs[clean_bare] = iv
                    agg_ranges.append(f"{iv} in 1:{self._jl_n(bare)}")
                child = self._with_extra_subs(extra_subs)
                arg_expr = child.visit(node.arguments[0])
                for_clause_agg = ", ".join(agg_ranges)
                return f"{julia_func}([{arg_expr} for {for_clause_agg}])"

        args = [self.visit(a) for a in node.arguments]

        # Symbolics.jl ifelse requires a Bool condition.  Vensim IF THEN ELSE
        # accepts any numeric condition (nonzero = true), so a bare variable or
        # arithmetic expression must be wrapped with `!= 0`.  Only LogicStructure
        # arguments (comparisons like `<`, `>`, `==`, and logical operators) are
        # already Bool — leave them untouched.
        if julia_func == "ifelse" and args:
            if not isinstance(node.arguments[0], LogicStructure):
                args[0] = f"({args[0]} != 0)"

        # Time-dependent helpers receive the symbolic *t* as their first arg
        if julia_func in _TIME_HELPERS:
            return f"{julia_func}(t, {', '.join(args)})"

        return f"{julia_func}({', '.join(args)})"

    def _inline_lookup(self, node: InlineLookupsStructure) -> str:
        arg_expr = self.visit(node.argument)
        name = self.registry.register(node.lookups.x, node.lookups.y, node.lookups.type)
        return f"{name}({arg_expr})"
