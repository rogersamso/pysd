"""
Julia DSL builder: writes a model in PySD.jl's DSL (``@equations`` and
``Model``), one equation set per view.

Each Vensim element becomes a declaration (``@parameters`` for constants,
lookups and Excel inputs, with their value; ``@variables`` for the rest,
``x(t)`` when it changes with time) with its unit and description, and its
components become equations (``~``) with the part they define on the left.
References with subscripts become the whole array, a selection
(``x[d = At(:e)]``), a renamed copy (``set(x, :d => :d1)``) or, in a SUM,
``sum_over``.  The model is built with ``numerics = :python`` so it keeps
Python PySD's last digits.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from pysd.translators.structures import abstract_expressions as ae
from pysd.translators.structures.abstract_model import (
    AbstractControlElement, AbstractData, AbstractLookup, AbstractModel,
    AbstractSection)

# names the DSL uses for its own functions (and Julia keywords): elements
# with these names get a suffix
RESERVED = frozenset("""
    t t₀ Δt begin end for if else elseif while function return let local global
    const struct module using import export true false nothing in isa where do try
    catch finally quote macro abstract mutable primitive type baremodule
    smooth delay trend sample_if_true sum_over maximum_over minimum_over prod_over
    previous piecewise_linear zidz xidz constant lookup data excel set At Not I inv
    Model ifelse min max abs exp log sqrt sin cos tan asin acos atan sinh cosh tanh
    trunc floor ceil round mod length sum prod maximum minimum pi NoUnits
    """.split())

# Vensim's unit words, in Unitful's notation (or PySD.jl's units)
UNIT_WORDS = {
    "dmnl": None, "dimensionless": None, "1": None, "percent": None, "%": None,
    "unitless": None, "fraction": None,
    "year": "yr", "years": "yr", "yr": "yr", "month": "month", "months": "month",
    "week": "wk", "weeks": "wk", "day": "d", "days": "d", "hour": "hr",
    "hours": "hr", "minute": "minute", "minutes": "minute", "second": "s",
    "seconds": "s", "s": "s",
    "people": "people", "person": "people", "persons": "people",
    "mpeople": "Mpeople", "dollar": "dollar", "dollars": "dollar", "$": "dollar",
    "mdollar": "Mdollar", "mdollars": "Mdollar", "m$": "Mdollar",
    "tdollar": "Tdollar", "tdollars": "Tdollar", "t$": "Tdollar",
    "degrees fahrenheit": "°F", "degreesfahrenheit": "°F",
    "degrees celsius": "°C", "degreesc": "°C", "ºc": "°C",
    "meter": "m", "meters": "m", "tonne": "t", "tonnes": "t", "ton": "t",
}

_TOKEN = re.compile(r"\s*([A-Za-z$€%º°][A-Za-z0-9$€%_º° ]*|\d+(?:\.\d+)?|[*/^()])")


def clean(name: str) -> str:
    """The name PySD gives references: lowercase, non-alphanumerics as _."""
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]", "_", name.lower())).strip("_")


def identifier(name: str, taken: set) -> str:
    ident = clean(name) or "_var"
    if ident[0].isdigit():
        ident = "_" + ident
    if ident in RESERVED:
        ident += "_var"
    base, i = ident, 1
    while ident in taken:
        ident = f"{base}_{i}"
        i += 1
    taken.add(ident)
    return ident


def label(element: str) -> str:
    """A subscript element as a Julia label (a Symbol)."""
    lab = clean(element)
    if lab and not lab[0].isdigit():
        return ":" + lab
    return f'Symbol("{lab}")'


def number(value) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "true" if value else "false"
    value = float(value)
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return repr(value)


def jl_string(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$") + '"'


# --- units ---------------------------------------------------------------------

def translate_units(text: str) -> Tuple[Optional[str], List[str]]:
    """Vensim's unit text in Unitful's notation (None: NoUnits), and the words
    Unitful and PySD.jl don't know, to declare in units.jl."""
    text = (text or "").split("[")[0].strip()
    if not text:
        return None, []
    pos, out, unknown = 0, [], []
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if not m:
            return None, []
        tok = m.group(1).strip()
        pos = m.end()
        if tok in "*/^()" or re.fullmatch(r"\d+(\.\d+)?", tok):
            out.append(tok)
            continue
        key = tok.lower()
        if key in UNIT_WORDS:
            word = UNIT_WORDS[key]
            out.append("1" if word is None else word)
        else:
            word = re.sub(r"[^A-Za-z0-9_]", "_", tok)
            out.append(word)
            unknown.append(word)
    unit = "".join(out)
    if re.fullmatch(r"[1*/()^ ]*", unit):
        return None, []
    return unit, unknown


# --- the expression writer --------------------------------------------------------

# precedence: higher binds tighter
P_ATOM, P_POW, P_UNARY, P_MUL, P_ADD, P_CMP, P_AND, P_OR = 100, 90, 80, 70, 60, 50, 40, 30


@dataclass
class Code:
    text: str
    dims: Tuple[str, ...] = ()       # the dimensions of its value (Julia dim names)
    prec: int = P_ATOM

    def wrap(self, prec: int) -> str:
        return f"({self.text})" if self.prec < prec else self.text


class DSLError(Exception):
    pass


@dataclass
class Element:
    name: str                         # Vensim's
    ident: str                        # Julia's
    kind: str                         # parameter, stock, variable
    dims: Tuple[str, ...]             # Julia dim names, declared order
    abstract: object
    timevarying: bool = True
    view: str = ""
    lines: List[str] = field(default_factory=list)
    declaration: str = ""


class DSLBuilder:
    def __init__(self, abstract_model: AbstractModel):
        self.model = abstract_model
        self.section: AbstractSection = abstract_model.sections[0]
        self.path = Path(getattr(abstract_model, "original_path", None) or self.section.path)
        self.taken: set = set()
        self.unknown_units: set = set()
        self.sheets: Dict[Tuple[str, str], str] = {}
        self.missing_units: List[str] = []
        self._ranges()
        self._elements()

    # --- subscripts ------------------------------------------------------------------

    def _ranges(self):
        self.ranges: Dict[str, List[str]] = {}       # Vensim range → its elements
        self.copies: Dict[str, str] = {}
        for r in self.section.subscripts:
            if isinstance(r.subscripts, str):
                self.copies[r.name] = r.subscripts
            elif isinstance(r.subscripts, list):
                self.ranges[r.name] = list(r.subscripts)
            else:
                raise DSLError(f"subscript range {r.name}: {r.subscripts!r} is not supported yet")
        for c, of in self.copies.items():
            self.ranges[c] = self.ranges[of]
        names = list(self.ranges)
        # a subrange: its elements are part of a larger range's (copies are dimensions of their own)
        self.main: Dict[str, str] = {}
        for r in names:
            parents = [o for o in names if o != r and o not in self.copies
                       and set(self.ranges[r]) < set(self.ranges[o])]
            self.main[r] = max(parents, key=lambda o: len(self.ranges[o])) if parents else r
        self.dimname = {r: identifier(r, self.taken) for r in names if self.main[r] == r}
        self.subname = {r: identifier(r, self.taken) for r in names if self.main[r] != r}
        self.elem_range: Dict[str, str] = {}
        for r in names:
            if self.main[r] == r and r not in self.copies:
                for e in self.ranges[r]:
                    cur = self.elem_range.get(e)
                    if cur is None or len(self.ranges[r]) > len(self.ranges[cur]):
                        self.elem_range[e] = r
        self.lower_range = {r.lower(): r for r in names}
        self.lower_elem = {e.lower(): e for e in self.elem_range}

    def _range_of(self, sub: str) -> Optional[str]:
        return self.lower_range.get(sub.lower().rstrip("!").strip())

    def _elem(self, sub: str) -> Optional[str]:
        return self.lower_elem.get(sub.lower().strip())

    def _main_dim(self, sub: str) -> str:
        r = self._range_of(sub)
        if r is not None:
            return self.main[r]
        e = self._elem(sub)
        if e is None:
            raise DSLError(f"unknown subscript {sub!r}")
        return self.elem_range[e]

    # --- elements ----------------------------------------------------------------------

    def _elements(self):
        self.controls: Dict[str, object] = {}
        self.elements: Dict[str, Element] = {}      # clean name → element
        for el in self.section.elements:
            if isinstance(el, AbstractControlElement):
                self.controls[clean(el.name)] = el
                continue
            dims = self._declared_dims(el)
            kind = self._kind(el)
            e = Element(el.name, identifier(el.name, self.taken), kind,
                        tuple(self.dimname[d] for d in dims), el)
            e.vensim_dims = dims
            self.elements[clean(el.name)] = e
        views = self.section.views_dict or {}
        where = {}
        for view, members in views.items():
            for m in (members if isinstance(members, (list, tuple, set)) else members.keys()):
                where[clean(m)] = view
        for k, e in self.elements.items():
            e.view = where.get(k, "")
        self._time_dependence()

    def _declared_dims(self, el) -> List[str]:
        dims = None
        for c in el.components:
            subs = c.subscripts[0]
            d = [self._main_dim(s) for s in subs]
            if dims is None:
                dims = d
            elif d != dims:
                raise DSLError(f"{el.name}: its parts have different dimensions")
        return dims or []

    def _kind(self, el) -> str:
        asts = [c.ast for c in el.components]
        if any(isinstance(a, ae.IntegStructure) for a in asts):
            return "stock"
        if isinstance(el.components[0], (AbstractLookup, AbstractData)) or all(
                isinstance(a, (int, float, np.ndarray, ae.GetConstantsStructure,
                               ae.LookupsStructure, ae.GetLookupsStructure, ae.GetDataStructure))
                for a in asts):
            return "parameter"
        return "variable"

    def _time_dependence(self):
        """A variable changes with time if its equations use time, a stateful
        function or anything that does."""
        deps: Dict[str, set] = {}
        direct: Dict[str, bool] = {}
        for k, e in self.elements.items():
            refs, timed = set(), False
            for c in e.abstract.components:
                r, tm = self._scan(c.ast)
                refs |= r
                timed |= tm
            deps[k] = refs
            direct[k] = timed or e.kind == "stock"
        memo: Dict[str, bool] = {}

        def timed(k, seen=()):
            if k in memo:
                return memo[k]
            if k not in self.elements or self.elements[k].kind == "parameter":
                return False
            if direct[k]:
                memo[k] = True
                return True
            if k in seen:
                return False
            memo[k] = any(timed(d, seen + (k,)) for d in deps[k])
            return memo[k]

        for k, e in self.elements.items():
            e.timevarying = e.kind != "parameter" and timed(k)

    STATEFUL = (ae.IntegStructure, ae.DelayStructure, ae.DelayNStructure, ae.DelayFixedStructure,
                ae.SmoothStructure, ae.SmoothNStructure, ae.TrendStructure, ae.SampleIfTrueStructure,
                ae.InitialStructure)

    def _scan(self, node) -> Tuple[set, bool]:
        refs, timed = set(), False
        if isinstance(node, self.STATEFUL):
            timed = True
        if isinstance(node, ae.ReferenceStructure):
            if clean(node.reference) in ("time",):
                timed = True
            refs.add(clean(node.reference))
            return refs, timed
        if isinstance(node, ae.CallStructure):
            name = clean(node.function.reference)
            if name in ("step", "pulse", "pulse_train", "ramp", "random_uniform", "random_0_1",
                        "random_normal", "random_exponential"):
                timed = True
            refs.add(name)
        for child in self._children(node):
            r, tm = self._scan(child)
            refs |= r
            timed |= tm
        return refs, timed

    @staticmethod
    def _children(node):
        if isinstance(node, (int, float, str, np.ndarray)) or node is None:
            return []
        if isinstance(node, (list, tuple)):
            return list(node)
        if hasattr(node, "__dataclass_fields__"):
            return [getattr(node, f) for f in node.__dataclass_fields__]
        return []

    # --- writing ---------------------------------------------------------------------

    def build_model(self, out_dir: Optional[Path] = None) -> Path:
        out = Path(out_dir) if out_dir else self.path.parent / (self.path.stem + "_dsl")
        out.mkdir(parents=True, exist_ok=True)
        self.out = out
        decls_by_view: Dict[str, List[Element]] = {}
        for e in self.elements.values():
            self._write_element(e)
            decls_by_view.setdefault(e.view, []).append(e)
        files = []
        for view, els in decls_by_view.items():
            name = identifier(view or "equations", set()) if view else "equations"
            fname = f"{name}.jl"
            files.append((name + "_view", fname))
            (out / fname).write_text(self._view_text(name + "_view", view, els), encoding="utf-8")
        (out / "dims.jl").write_text(self._dims_text(), encoding="utf-8")
        (out / "inputs.jl").write_text(self._inputs_text(), encoding="utf-8")
        if self.unknown_units:
            (out / "units.jl").write_text(self._units_text(), encoding="utf-8")
        names = ["# Vensim's names of the model's variables, for results and comparisons", "vensim\tjulia"]
        names += [f"{e.name}\t{e.ident}" for e in self.elements.values()]
        (out / "names.tsv").write_text("\n".join(names) + "\n", encoding="utf-8")
        main = out / (identifier(self.path.stem, set()) + ".jl")
        main.write_text(self._main_text(files), encoding="utf-8")
        return main

    def _view_text(self, value, view, els):
        params = [e for e in els if e.kind == "parameter"]
        variables = [e for e in els if e.kind != "parameter"]
        out = [f"# {'View ' + repr(view) if view else 'The equations'}: {len(els)} variables.", "",
               f"{value} = @equations begin"]
        if params:
            out.append("    @parameters begin")
            out += ["        " + e.declaration.replace("\n", "\n        ") for e in params]
            out.append("    end")
        if variables:
            out.append("    @variables begin")
            out += ["        " + e.declaration for e in variables]
            out.append("    end")
        out.append("")
        for e in variables:
            out += ["    " + line.replace("\n", "\n    ") for line in e.lines]
        out.append("end")
        return "\n".join(out) + "\n"

    def _dims_text(self):
        out = ["# Dimensions (Vensim subscript ranges) and subsets: plain Julia values.",
               "using DimensionalData: Dim", ""]
        for r, d in self.dimname.items():
            labels = ", ".join(label(e) for e in self.ranges[r])
            out.append(f"const {d} = Dim{{:{d}}}([{labels}])")
        for r, s in self.subname.items():
            labels = ", ".join(label(e) for e in self.ranges[r])
            out.append(f"const {s} = [{labels}]{' ' * 2}# a subset of {self.dimname[self.main[r]]}")
        return "\n".join(out) + "\n"

    def _inputs_text(self):
        out = ["# Input sheets, given to Model by name; a different scenario is a different",
               "# sheet given to the run: run_model(model; inputs = (; name = excel(…)))."]
        import os
        entries = [f"{ident} = excel(joinpath(@__DIR__, {jl_string(os.path.relpath(self.path.parent / file, self.out))}), "
                   f"{jl_string(tab)})" for (file, tab), ident in self.sheets.items()]
        out.append("inputs = (; " + ",\n            ".join(entries) + ")" if entries else "inputs = (;)")
        return "\n".join(out) + "\n"

    def _units_text(self):
        out = ["# Units of the model that Unitful.jl and PySD.jl don't have: each its own dimension.",
               "import Unitful", "module ModelUnits", "using Unitful", "using Unitful: @dimension, @refunit", ""]
        for i, u in enumerate(sorted(self.unknown_units)):
            dim = f"𝐔{i}"
            out.append(f'@dimension {dim} "{u}" Dim{u}')
            out.append(f'@refunit {u} "{u}" Unit{u} {dim} false')
        out += ["end", "Unitful.register(ModelUnits)"]
        return "\n".join(out) + "\n"

    def _control(self, name, default=None):
        el = self.controls.get(name)
        if el is None:
            return default
        ast = el.components[0].ast
        if isinstance(ast, (int, float)):
            return float(ast)
        if isinstance(ast, ae.ReferenceStructure):
            return self._control(clean(ast.reference), default)
        if isinstance(ast, ae.ArithmeticStructure) and all(isinstance(a, (int, float)) for a in ast.arguments):
            v = float(ast.arguments[0])
            for op, a in zip(ast.operators, ast.arguments[1:]):
                v = {"+": v + a, "-": v - a, "*": v * a, "/": v / a}[op]
            return v
        raise DSLError(f"{name}: only numbers are supported for the control variables")

    def _main_text(self, files):
        t0, tf = self._control("initial_time", 0), self._control("final_time")
        dt = self._control("time_step", 1)
        save = self._control("saveper", dt)
        tunit, _ = translate_units(self.controls["time_step"].units if "time_step" in self.controls else "")
        tunit = tunit or "s"
        out = [f"# {self.path.name}, translated to PySD.jl's DSL by PySD.", "using PySD, DimensionalData"]
        if self.unknown_units:
            out.append('include("units.jl")')
        out += ['include("dims.jl")', 'include("inputs.jl")']
        out += [f'include("{f}")' for _, f in files]
        sets = ", ".join(v for v, _ in files)
        out += ["", f"{identifier(self.path.stem, set())} = Model({sets};",
                f"    inputs, time = {number(t0)}:{number(dt)}:{number(tf)}, saveat = {number(save)},",
                f'    time_unit = u"{tunit}", numerics = :python)']
        return "\n".join(out) + "\n"

    # --- declarations and equations ----------------------------------------------------

    def _meta(self, e: Element) -> str:
        el = e.abstract
        unit, unknown = translate_units(el.units)
        self.unknown_units.update(unknown)
        if not (el.units or "").strip():
            self.missing_units.append(el.name)
        items = [f'unit = {"NoUnits" if unit is None else "u" + jl_string(unit)}']
        doc = " ".join((el.documentation or "").split())
        if el.limits and any(x is not None for x in el.limits):
            lo, hi = el.limits[:2]
            rng = f"[{'?' if lo is None else number(lo)}, {'?' if hi is None else number(hi)}]"
            doc = (doc + " " + rng).strip()
        if doc:
            items.append(f"description = {jl_string(doc)}")
        if any(isinstance(c.ast, ae.IntegStructure) and c.ast.non_negative for c in el.components):
            items.append("nonnegative = true")
        return "[" + ", ".join(items) + "]"

    def _decl_target(self, e: Element) -> str:
        t = e.ident + ("(t)" if e.timevarying else "")
        if e.dims:
            t += "[" + ", ".join(e.dims) + "]"
        return t

    def _write_element(self, e: Element):
        meta = self._meta(e)
        if e.kind == "parameter":
            e.declaration = f"{self._decl_target(e)} = {self._parameter_value(e)},\n    {meta}"
            return
        e.declaration = f"{self._decl_target(e)}, {meta}"
        inits = []
        for c in e.abstract.components:
            lhs = e.ident + self._lhs_selection(e, c)
            ctx = Context(self, e, c)
            ast = c.ast
            if isinstance(ast, ae.IntegStructure):
                e.lines.append(f"∂t({lhs}) ~ {ctx.visit(ast.flow).text}")
                init = Context(self, e, c, initial=True).visit(ast.initial).text
                if init not in inits:
                    inits.append(init)
            elif isinstance(ast, ae.CallStructure) and clean(ast.function.reference) == "active_initial":
                e.lines.append(f"{lhs} ~ {ctx.visit(ast.arguments[0]).text}")
                e.lines.append(f"{e.ident}(t₀) ~ {ctx.visit(ast.arguments[1]).text}")
            else:
                e.lines.append(f"{lhs} ~ {ctx.visit(ast).text}")
        if len(inits) > 1:
            raise DSLError(f"{e.name}: a stock whose parts have different initial values is not supported yet")
        if inits:
            e.lines.append(f"{e.ident}(t₀) ~ {inits[0]}")

    def _lhs_selection(self, e: Element, c) -> str:
        subs, exceptions = c.subscripts
        sels = []
        for p, s in enumerate(subs):
            dim_v = e.vensim_dims[p]
            dim = self.dimname[dim_v]
            full = self.ranges[dim_v]
            sel = self._selected(s, dim_v)
            excluded = set()
            for exc in exceptions or []:
                if p < len(exc):
                    excluded |= set(self._selected(exc[p], dim_v))
            chosen = [x for x in sel if x not in excluded]
            if len(chosen) == len(full):
                continue
            if excluded and len(sel) == len(full) and len(excluded) < len(chosen):
                ex = sorted(excluded, key=full.index)
                sels.append(f"{dim} = Not(At({self._labels(ex)}))")
            else:
                sels.append(f"{dim} = At({self._labels(chosen, s)})")
        return "[" + ", ".join(sels) + "]" if sels else ""

    def _selected(self, s, dim_v) -> List[str]:
        r = self._range_of(s)
        if r is not None:
            return list(self.ranges[r])
        e = self._elem(s)
        if e is None:
            raise DSLError(f"unknown subscript {s!r}")
        return [e]

    def _labels(self, elems, sub=None) -> str:
        if len(elems) == 1 and (sub is None or self._range_of(sub) is None):
            return label(elems[0])
        if sub is not None and self._range_of(sub) in self.subname:
            return self.subname[self._range_of(sub)]
        return "[" + ", ".join(label(x) for x in elems) + "]"

    def _sheet(self, file, tab) -> str:
        key = (file, tab)
        if key not in self.sheets:
            base = identifier(Path(file).stem + "_" + tab, self.taken)
            self.sheets[key] = base
        return self.sheets[key]

    def _parameter_value(self, e: Element) -> str:
        comps = e.abstract.components
        vals = [self._one_value(e, c) for c in comps]
        if len(vals) == 1:
            return vals[0]
        # parts along the first dimension: one value per element of it
        subs0 = [c.subscripts[0][0] if c.subscripts[0] else None for c in comps]
        firsts = [self._elem(s) for s in subs0]
        if None in firsts or len(set(firsts)) != len(firsts) or \
                set(firsts) != set(self.ranges[e.vensim_dims[0]]):
            raise DSLError(f"{e.name}: parameters defined in parts are supported along the first dimension only")
        order = self.ranges[e.vensim_dims[0]]
        by = dict(zip(firsts, vals))
        return "[" + ", ".join(by[x] for x in order) + "]"

    def _one_value(self, e: Element, c) -> str:
        ast = c.ast
        if isinstance(ast, (int, float)):
            return number(ast)
        if isinstance(ast, np.ndarray):
            if ast.ndim == 1:
                return "[" + ", ".join(number(x) for x in ast) + "]"
            if ast.ndim == 2:
                return "[" + "; ".join(" ".join(number(x) for x in row) for row in ast) + "]"
            raise DSLError(f"{e.name}: arrays of {ast.ndim} dimensions are not supported yet")
        if isinstance(ast, ae.GetConstantsStructure):
            cell, tr = ast.cell, ast.cell.endswith("*")
            cell = cell.rstrip("*")
            return f"constant({self._sheet(ast.file, ast.tab)}, {jl_string(cell)}" + \
                   ("; transpose = true)" if tr else ")")
        if isinstance(ast, ae.GetLookupsStructure):
            return f"lookup({self._sheet(ast.file, ast.tab)}, {jl_string(str(ast.x_row_or_col))}, {jl_string(ast.cell)})"
        if isinstance(ast, ae.GetDataStructure):
            kw = getattr(c, "keyword", None)
            method = {"interpolate": ":interpolate", "hold_backward": ":hold_backward",
                      "look_forward": ":look_forward", "raw": ":raw"}.get((kw or "interpolate").lower().strip(":"))
            m = "" if method in (None, ":interpolate") else f"; method = {method}"
            return f"data({self._sheet(ast.file, ast.tab)}, {jl_string(str(ast.time_row_or_col))}, {jl_string(ast.cell)}{m})"
        if isinstance(ast, ae.LookupsStructure):
            pts = ", ".join(f"{number(x)} => {number(y)}" for x, y in zip(ast.x, ast.y))
            return f"piecewise_linear({pts})"
        raise DSLError(f"{e.name}: {type(ast).__name__} as a parameter is not supported yet")


ARITH = {"+": "+", "-": "-", "*": "*", "/": "/", "^": "^", "**": "^"}
LOGIC = {"=": "==", "<>": "!=", "<": "<", ">": ">", "<=": "≤", ">=": "≥",
         ":and:": "&", ":or:": "|", "and": "&", "or": "|"}
FUNCTIONS = {"abs": "abs", "exp": "exp", "ln": "log", "sqrt": "sqrt", "sin": "sin", "cos": "cos",
             "tan": "tan", "arcsin": "asin", "arccos": "acos", "arctan": "atan", "sinh": "sinh",
             "cosh": "cosh", "tanh": "tanh", "integer": "trunc", "int": "trunc", "min": "min",
             "max": "max", "modulo": "mod", "zidz": "zidz", "xidz": "xidz"}
REDUCTIONS = {"sum": "sum_over", "prod": "prod_over", "vmax": "maximum_over", "vmin": "minimum_over"}


class Context:
    """Translating one component's expression: references line up with the
    dimensions of the part being defined."""

    def __init__(self, builder: DSLBuilder, element: Element, component, initial=False):
        self.b = builder
        self.e = element
        self.c = component
        self.initial = initial

    def visit(self, node) -> Code:
        if isinstance(node, (bool, np.bool_)):
            return Code("true" if node else "false")
        if isinstance(node, (int, float)):
            text = number(node)
            return Code(text, (), P_ATOM if not text.startswith("-") else P_UNARY)
        if isinstance(node, ae.ArithmeticStructure):
            return self._arithmetic(node)
        if isinstance(node, ae.LogicStructure):
            return self._logic(node)
        if isinstance(node, ae.ReferenceStructure):
            return self._reference(node)
        if isinstance(node, ae.CallStructure):
            return self._call(node)
        for cls, kind in ((ae.SmoothStructure, "smooth"), (ae.SmoothNStructure, "smooth"),
                          (ae.DelayStructure, "delay"), (ae.DelayNStructure, "delay")):
            if isinstance(node, cls):
                return self._stateful(kind, node.input, getattr(node, "smooth_time", None) or
                                      getattr(node, "delay_time", None), node.order, node.initial)
        if isinstance(node, ae.TrendStructure):
            x, tau, init = self.visit(node.input), self.visit(node.average_time), self.visit(node.initial_trend)
            return Code(f"trend({x.text}, {tau.text}; init = {init.text})", x.dims)
        if isinstance(node, ae.SampleIfTrueStructure):
            cnd, x, init = self.visit(node.condition), self.visit(node.input), self.visit(node.initial)
            return Code(f"sample_if_true({cnd.text}, {x.text}; init = {init.text})",
                        self._union(cnd.dims, x.dims, init.dims))
        if isinstance(node, ae.DelayFixedStructure):
            if not isinstance(node.input, ae.ReferenceStructure):
                raise DSLError(f"{self.e.name}: DELAY FIXED of an expression is not supported yet")
            x = self._reference(node.input)
            tau, init = self.visit(node.delay_time), self.visit(node.initial)
            name, sel = x.text.split("[", 1) if "[" in x.text and not x.text.startswith("set(") else (x.text, None)
            if sel is not None or x.text.startswith("set("):
                raise DSLError(f"{self.e.name}: DELAY FIXED of a part of an array is not supported yet")
            return Code(f"{name}(t - {tau.wrap(P_ADD + 1)}; init = {init.text})", x.dims)
        if isinstance(node, ae.InitialStructure):
            return self._initial(node.initial)
        raise DSLError(f"{self.e.name}: {type(node).__name__} is not supported yet")

    @staticmethod
    def _union(*dims_list) -> Tuple[str, ...]:
        out: List[str] = []
        for ds in dims_list:
            for d in ds:
                if d not in out:
                    out.append(d)
        return tuple(out)

    def _stateful(self, kind, x, tau, order, init) -> Code:
        xc, tc = self.visit(x), self.visit(tau)
        args = [xc.text, tc.text]
        kws = []
        if isinstance(order, (int, float)):
            if order != 1:
                kws.append(f"order = {number(order)}")
        else:
            kws.append(f"order = {self.visit(order).text}")
        if not (isinstance(init, ae.ReferenceStructure) and init is x) and init is not None and \
                not self._same_as(init, x):
            kws.append(f"init = {self.visit(init).text}")
        text = f"{kind}({', '.join(args)}" + (f"; {', '.join(kws)})" if kws else ")")
        return Code(text, self._union(xc.dims, tc.dims))

    @staticmethod
    def _same_as(a, b) -> bool:
        return isinstance(a, ae.ReferenceStructure) and isinstance(b, ae.ReferenceStructure) and \
            clean(a.reference) == clean(b.reference) and a.subscripts == b.subscripts

    def _initial(self, node) -> Code:
        if isinstance(node, ae.ReferenceStructure):
            r = self._reference(node)
            el = self.b.elements.get(clean(node.reference))
            if el is None or not el.timevarying:
                return r
            if "[" in r.text or r.text.startswith("set("):
                raise DSLError(f"{self.e.name}: INITIAL of a part of an array is not supported yet")
            return Code(f"{r.text}(t₀)", r.dims)
        raise DSLError(f"{self.e.name}: INITIAL of an expression is not supported yet")

    # dotted when an operand is an array
    @staticmethod
    def _op(op, *codes):
        return ("." + op) if any(c.dims for c in codes) else op

    def _arithmetic(self, node) -> Code:
        ops, args = node.operators, [self.visit(a) for a in node.arguments]
        if len(args) == 1:
            if ops and ops[0] in ("negative", "-"):
                a = args[0]
                return Code(("." if a.dims else "") + "-" + a.wrap(P_UNARY), a.dims, P_UNARY)
            return args[0]
        if all(op in ("^", "**") for op in ops):        # right-associative
            out = args[-1]
            for a in reversed(args[:-1]):
                op = self._op("^", a, out)
                out = Code(f"{a.wrap(P_POW + 1)} {op} {out.wrap(P_POW)}", self._union(a.dims, out.dims), P_POW)
            return out
        out = args[0]
        for op, a in zip(ops, args[1:]):
            if op == "mod":
                out = Code(f"mod{'.' if out.dims or a.dims else ''}({out.text}, {a.text})",
                           self._union(out.dims, a.dims))
                continue
            jop = ARITH[op]
            prec = P_MUL if jop in ("*", "/") else P_POW if jop == "^" else P_ADD
            right = a.wrap(prec + 1)
            out = Code(f"{out.wrap(prec)} {self._op(jop, out, a)} {right}", self._union(out.dims, a.dims), prec)
        return out

    def _logic(self, node) -> Code:
        ops, args = [o.lower() for o in node.operators], [self.visit(a) for a in node.arguments]
        if len(args) == 1 and ops and ops[0] in (":not:", "not"):
            a = args[0]
            return Code(("." if a.dims else "") + "!" + a.wrap(P_UNARY), a.dims, P_UNARY)
        out = args[0]
        for op, a in zip(ops, args[1:]):
            jop = LOGIC[op]
            prec = P_AND if jop == "&" else P_OR if jop == "|" else P_CMP
            out = Code(f"{out.wrap(prec + 1)} {self._op(jop, out, a)} {a.wrap(prec + 1)}",
                       self._union(out.dims, a.dims), prec)
        return out

    def _lhs_ranges(self) -> List[str]:
        return [self.b._range_of(s) for s in self.c.subscripts[0]]

    def _reference(self, node: ae.ReferenceStructure, bang_ok=False) -> Code:
        name = clean(node.reference)
        if name == "time":
            return Code("t")
        if name == "time_step":
            return Code("Δt")
        if name == "initial_time":
            return Code("t₀")
        el = self.b.elements.get(name)
        if el is None:
            if name in self.b.controls:
                raise DSLError(f"{self.e.name}: {node.reference} in an equation is not supported yet")
            raise DSLError(f"{self.e.name}: unknown name {node.reference!r}")
        subs = node.subscripts.subscripts if node.subscripts else []
        base = el.ident
        if isinstance(el.abstract.components[0], AbstractData):
            base = f"{el.ident}(t)"                       # a data series: its value now
        if not subs:
            return Code(base, el.dims)
        sels, renames, dims = [], [], []
        lhs = self._lhs_ranges()
        for p, s in enumerate(subs):
            dv = el.vensim_dims[p]
            d = self.b.dimname[dv]
            r = self.b._range_of(s)
            if r is None:                                   # an element
                e = self.b._elem(s)
                sels.append(f"{d} = At({label(e)})")
                continue
            if r == dv:
                dims.append(d)
            elif self.b.main[r] == dv:                      # a subset
                sels.append(f"{d} = At({self.b.subname[r]})")
                dims.append(d)
            elif self.b.main[r] == r and len(self.b.ranges[r]) == len(self.b.ranges[dv]):
                renames.append(f":{d} => :{self.b.dimname[r]}")   # a copy of the dimension
                dims.append(self.b.dimname[r])
            else:
                raise DSLError(f"{self.e.name}: {node.reference}[{s}]: subscript mappings are not supported yet")
        text = base
        if sels:
            text += "[" + ", ".join(sels) + "]"
        if renames:
            text = f"set({text}, {', '.join(renames)})"
        return Code(text, tuple(dims))

    def _call(self, node: ae.CallStructure) -> Code:
        fname = clean(node.function.reference)
        args = node.arguments
        if fname in self.b.elements:                         # a lookup
            f = self.b.elements[fname]
            fref = self._reference(node.function) if node.function.subscripts else Code(f.ident, f.dims)
            x = self.visit(args[0])
            if f.dims:
                return Code(f"{fref.text}({x.text})", self._union(fref.dims, x.dims))
            dot = "." if x.dims else ""
            return Code(f"{fref.text}{dot}({x.text})", x.dims)
        if fname in ("if_then_else",):
            c, a, b = (self.visit(x) for x in args)
            dot = "." if (c.dims or a.dims or b.dims) else ""
            return Code(f"ifelse{dot}({c.text}, {a.text}, {b.text})", self._union(c.dims, a.dims, b.dims))
        if fname in FUNCTIONS:
            cs = [self.visit(x) for x in args]
            dims = self._union(*(c.dims for c in cs))
            dot = "." if dims else ""
            return Code(f"{FUNCTIONS[fname]}{dot}({', '.join(c.text for c in cs)})", dims)
        if fname == "power":
            a, b = self.visit(args[0]), self.visit(args[1])
            return Code(f"{a.wrap(P_POW + 1)} {self._op('^', a, b)} {b.wrap(P_POW)}", self._union(a.dims, b.dims), P_POW)
        if fname == "log":
            a, b = self.visit(args[0]), self.visit(args[1])
            dot = "." if (a.dims or b.dims) else ""
            return Code(f"log{dot}({a.text}) {'./' if dot else '/'} log{dot}({b.text})", self._union(a.dims, b.dims), P_MUL)
        if fname == "step":
            h, t0 = self.visit(args[0]), self.visit(args[1])
            return Code(f"{h.wrap(P_MUL)} {self._op('*', h)} (t ≥ {t0.text})", h.dims, P_MUL)
        if fname == "ramp":
            s, t0 = self.visit(args[0]), self.visit(args[1])
            t1 = self.visit(args[2]) if len(args) > 2 else None
            hi = f"{t1.wrap(P_ADD + 1)} - {t0.wrap(P_ADD + 1)}" if t1 else "Inf"
            return Code(f"{s.wrap(P_MUL)} {self._op('*', s)} clamp(t - {t0.wrap(P_ADD + 1)}, 0, {hi})", s.dims, P_MUL)
        if fname == "pulse":
            t0, w = self.visit(args[0]), self.visit(args[1])
            return Code(f"({t0.text} ≤ t < {t0.wrap(P_ADD)} + {w.wrap(P_ADD + 1)})", ())
        if fname in REDUCTIONS:
            x = self.visit(args[0])
            summed = [self.b.dimname.get(self.b.main[self.b._range_of(s)], None) or self.b.dimname[self.b._range_of(s)]
                      for s in self._bangs(args[0])]
            if not summed:
                raise DSLError(f"{self.e.name}: {fname.upper()} without a subscript with !")
            text, dims = x.text, list(x.dims)
            for d in summed:
                if d not in dims:
                    raise DSLError(f"{self.e.name}: summing over {d}, which {x.text} hasn't")
                text = f"{REDUCTIONS[fname]}({text}, {d})"
                dims.remove(d)
            return Code(text, tuple(dims))
        if fname == "elmcount":
            r = self.b._range_of(args[0].reference if isinstance(args[0], ae.ReferenceStructure) else str(args[0]))
            if r is None:
                raise DSLError(f"{self.e.name}: ELMCOUNT of {args[0]!r}")
            return Code(f"length({self.b.dimname.get(r) or self.b.subname[r]})")
        if fname == "invert_matrix":
            m = self.visit(args[0])
            return Code(f"inv({m.text})", tuple(reversed(m.dims)))
        if fname == "active_initial":
            raise DSLError(f"{self.e.name}: ACTIVE INITIAL inside an expression is not supported yet")
        raise DSLError(f"{self.e.name}: the function {node.function.reference.upper()} is not supported yet")

    def _bangs(self, node) -> List[str]:
        """The subscripts marked with ! in an expression, once each (SUM's)."""
        out: List[str] = []
        if isinstance(node, ae.ReferenceStructure) and node.subscripts:
            for s in node.subscripts.subscripts:
                if s.endswith("!") and s not in out:
                    out.append(s)
        for child in DSLBuilder._children(node):
            for s in self._bangs(child):
                if s not in out:
                    out.append(s)
        return out
