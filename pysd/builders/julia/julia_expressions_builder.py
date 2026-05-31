"""
Converts AMR Abstract Syntax Tree nodes to Julia expression strings.
"""
from __future__ import annotations

import re
from typing import Any, List, Set, Tuple
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
    TrendStructure,
)

# ---------------------------------------------------------------------------
# Operator tables
# ---------------------------------------------------------------------------

# Vensim arithmetic operator  →  Julia operator
ARITHMETIC_OPS: dict = {
    "+": "+",
    "-": "-",
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
    "INTEGER": "trunc",
    "INT": "trunc",
    "MIN": "min",
    "MAX": "max",
    "MODULO": "mod",
    # Control flow
    "IF THEN ELSE": "ifelse",
    # SD helpers emitted into the generated file
    "LOG": "_log_base",
    "XIDZ": "_xidz",
    "ZIDZ": "_zidz",
    "PULSE": "_pulse",
    "PULSE TRAIN": "_pulse_train",
    "RAMP": "_ramp",
    "STEP": "_step",
}

# One-line Julia implementations for helper functions
HELPER_IMPLEMENTATIONS: dict = {
    "_log_base": "_log_base(x, base) = log(base, x)",
    "_xidz": "_xidz(x, y, z) = iszero(y) ? z : x / y",
    "_zidz": "_zidz(x, y) = iszero(y) ? 0.0 : x / y",
    "_pulse": (
        "_pulse(t_now, start, width) = "
        "(t_now >= start && t_now < start + width) ? 1.0 : 0.0"
    ),
    "_pulse_train": (
        "_pulse_train(t_now, start, width, interval, end_time) = "
        "(t_now >= start && t_now <= end_time && "
        "mod(t_now - start, interval) < width) ? 1.0 : 0.0"
    ),
    "_ramp": (
        "_ramp(t_now, slope, start_time, end_time=Inf) = "
        "slope * max(0.0, min(t_now - start_time, end_time - start_time))"
    ),
    "_step": (
        "_step(t_now, height, step_time) = "
        "t_now >= step_time ? float(height) : 0.0"
    ),
}

# Helper functions that receive the current time *t* as their first argument
_TIME_HELPERS: frozenset = frozenset({"_pulse", "_pulse_train", "_ramp", "_step"})


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
    name: str, xs: tuple, ys: tuple, _itp_type: str
) -> Tuple[str, str]:
    """Return ``(const_decl, func_decl)`` for a named lookup table.

    Uses ``DataInterpolations.LinearInterpolation(u, t)`` where ``u`` are the
    y-values and ``t`` the x-values (DataInterpolations convention).
    """
    xs_vec = format_vector(xs)
    ys_vec = format_vector(ys)
    itp_name = f"{name}_itp"
    const_decl = f"const {itp_name} = LinearInterpolation({ys_vec}, {xs_vec})"
    func_decl = f"{name}(x) = {itp_name}(x)"
    return const_decl, func_decl


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
    ) -> None:
        self.namespace = namespace
        self.registry = inline_registry
        self.needed_helpers = needed_helpers

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

        if len(args) == 1:
            op = LOGIC_OPS.get(ops[0], ops[0])
            return f"({op}{args[0]})"

        parts = [args[0]]
        for op, arg in zip(ops, args[1:]):
            parts.append(LOGIC_OPS.get(op, op))
            parts.append(arg)
        return "(" + " ".join(parts) + ")"

    def _reference(self, node: ReferenceStructure) -> str:
        julia_name = self.namespace.get(node.reference)
        if julia_name is None:
            warn(
                f"Variable '{node.reference}' not found in namespace; "
                "using a sanitised fallback identifier."
            )
            julia_name = re.sub(r"[^a-z0-9_]", "_", node.reference.lower())
        return julia_name

    def _call(self, node: CallStructure) -> str:
        func_upper = node.function.reference.upper()
        julia_func = BUILTIN_FUNCTIONS.get(func_upper)

        if julia_func is None:
            warn(f"Unknown Vensim function '{node.function.reference}'; using lowercase name.")
            julia_func = re.sub(r"[^a-z0-9_]", "_", node.function.reference.lower())

        if julia_func in HELPER_IMPLEMENTATIONS:
            self.needed_helpers.add(julia_func)

        args = [self.visit(a) for a in node.arguments]

        # Time-dependent helpers receive the symbolic *t* as their first arg
        if julia_func in _TIME_HELPERS:
            return f"{julia_func}(t, {', '.join(args)})"

        return f"{julia_func}({', '.join(args)})"

    def _inline_lookup(self, node: InlineLookupsStructure) -> str:
        arg_expr = self.visit(node.argument)
        name = self.registry.register(node.lookups.x, node.lookups.y, node.lookups.type)
        return f"{name}({arg_expr})"
