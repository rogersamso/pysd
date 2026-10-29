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

from .julia_model_builder import PYSD_JL_VERSION

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

# units Unitful.jl (and PySD.jl) know, which also take SI prefixes: E + J is EJ
_KNOWN_UNITS = {"m", "g", "s", "A", "K", "mol", "cd", "J", "W", "N", "Pa", "Hz", "L", "eV",
                "Wh", "t", "people", "dollar", "V", "C", "Ω", "S", "F", "T", "Wb", "H", "Bq", "Gy",
                "Sv", "kat", "bar", "atm"}
_UNPREFIXED = {"yr", "d", "hr", "minute", "wk", "month", "ha", "ppm", "ppb", "ppt", "percent",
               "rad", "°", "°C", "°F", "Ra", "inch", "ft", "mi", "lb", "kg"}
_PREFIXES = ("Y", "Z", "E", "P", "T", "G", "M", "k", "h", "da", "d", "c", "m", "μ", "n", "p", "f", "a")


def known_unit(word: str) -> bool:
    if word in _UNPREFIXED or word in _KNOWN_UNITS:
        return True
    return any(word.startswith(pre) and word[len(pre):] in _KNOWN_UNITS for pre in _PREFIXES)


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
            word = re.sub(r"[^A-Za-z0-9_°]", "_", tok)
            out.append(word)
            if not known_unit(word):
                unknown.append(word)
    unit = "".join(out)
    if re.fullmatch(r"[1*/()^ ]*", unit):
        return None, []
    return unit, unknown


# --- the expression writer --------------------------------------------------------

# precedence: higher binds tighter
# Julia's: & binds as *, | as +, both tighter than comparisons
P_ATOM, P_POW, P_UNARY, P_MUL, P_ADD, P_CMP = 100, 90, 80, 70, 60, 50
P_AND, P_OR = P_MUL, P_ADD


@dataclass
class Code:
    text: str
    dims: Tuple[str, ...] = ()       # the dimensions of its value (Julia dim names)
    prec: int = P_ATOM

    def wrap(self, prec: int) -> str:
        return f"({self.text})" if self.prec < prec else self.text


def invert_matrix(text: str, dims) -> str:
    """Vensim's INVERT MATRIX: the inverse under the same dimensions.  Julia's
    inv of a DimArray swaps them (its rows are the columns of the matrix), so
    the names and labels are swapped back, position by position."""
    r, c = dims
    return f"set(inv({text}), :{c} => {r}, :{r} => {c})"


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
        # subscript mappings (lower -> upper): position by position, both ways
        self.mapped = set()
        for r in self.section.subscripts:
            for m in r.mapping or []:
                if isinstance(m, str):
                    self.mapped.add((r.name, m))
                    self.mapped.add((m, r.name))
        names = list(self.ranges)
        # every range is a dimension (as in Python PySD), with its elements as labels;
        # a range whose elements are part of another's is a selection of it there
        self.dimname = {r: identifier(r, self.taken) for r in names}
        self.elements_name = {r: identifier(r + "_elements", self.taken)
                              for r in names if r not in self.copies}
        for c, of in self.copies.items():
            self.elements_name[c] = self.elements_name[of]
        self.lower_range = {r.lower(): r for r in names}
        self.lower_elem = {e.lower(): e for r in names for e in self.ranges[r]}

    def _range_of(self, sub: str) -> Optional[str]:
        return self.lower_range.get(sub.lower().rstrip("!").strip())

    def _elem(self, sub: str) -> Optional[str]:
        return self.lower_elem.get(sub.lower().strip())


    # --- elements ----------------------------------------------------------------------

    def _elements(self):
        self.controls: Dict[str, object] = {}
        self.kind_errors: List[str] = []
        self.hidden: Dict[str, List[str]] = {}       # view → declarations the translator adds
        self.positions: Dict[str, str] = {}          # range → its positions parameter
        self.elements: Dict[str, Element] = {}      # clean name → element
        for el in self.section.elements:
            if isinstance(el, AbstractControlElement):
                self.controls[clean(el.name)] = el
                continue
            dims = self._declared_dims(el)
            try:
                kind = self._kind(el)
            except DSLError as err:
                self.kind_errors.append(str(err))
                kind = "parameter"
            e = Element(el.name, identifier(el.name, self.taken), kind,
                        tuple(self.dimname[d] for d in dims), el)
            e.vensim_dims = dims
            split = self._split_position(el) if kind == "parameter" and len(el.components) > 1 else None
            e.split = split
            if split:                                    # the dimension it is split along, first
                order = [split] + [i for i in range(len(dims)) if i != split]
                e.dims = tuple(e.dims[i] for i in order)
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
        """At each position, the range the equations cover: the one written on the
        left when there is one equation, else the range of all their elements."""
        n = len(el.components[0].subscripts[0])
        dims = []
        for p in range(n):
            written = [c.subscripts[0][p] for c in el.components if len(c.subscripts[0]) > p]
            covered: List[str] = []
            for sub in written:
                for x in self._selected(sub, None):
                    if x not in covered:
                        covered.append(x)
            named = [self._range_of(w) for w in written if self._range_of(w) is not None]
            exact = [r for r in named if set(self.ranges[r]) == set(covered)] or \
                    [r for r in self.ranges if set(self.ranges[r]) == set(covered)]
            if exact:
                dims.append(exact[0])
                continue
            # no range has exactly these elements: a dimension of its own, of the elements defined
            # (references to it name its elements one at a time)
            r = f"{el.name} elements"
            self.ranges[r] = covered
            self.dimname[r] = identifier(r if p == 0 else f"{el.name} elements {p + 1}", self.taken)
            self.elements_name[r] = identifier(self.dimname[r] + "_labels", self.taken)
            self.lower_range[r.lower()] = r
            dims.append(r)
        # a range used at two positions: a copy of it for the second (one name per dimension)
        for p, r in enumerate(dims):
            if r in dims[:p]:
                k = dims[:p].count(r) + 1
                copy = f"{r} {k}"
                if copy not in self.ranges:
                    self.ranges[copy] = self.ranges[r]
                    self.copies[copy] = r
                    self.dimname[copy] = identifier(copy, self.taken)
                    self.elements_name[copy] = self.elements_name[r]
                dims[p] = copy
        return dims

    def _kind(self, el) -> str:
        asts = [c.ast for c in el.components]
        if any(isinstance(a, ae.IntegStructure) for a in asts):
            return "stock"
        inputs = (ae.LookupsStructure, ae.GetLookupsStructure, ae.GetDataStructure, ae.DataStructure)
        if all(isinstance(a, (int, float, np.ndarray, ae.GetConstantsStructure) + inputs) for a in asts):
            if len(asts) > 1 and self._split_position(el) is None:
                if any(isinstance(a, inputs) for a in asts):
                    raise DSLError(f"{el.name}: lookups or data in parts other than one element each are not supported yet")
                return "constant_parts"          # a constant variable with one equation per part
            return "parameter"
        return "variable"

    def _split_position(self, el) -> Optional[int]:
        """The position where each part is one element (all of them, once) and the
        others are whole ranges: a parameter in parts is then a vector of values."""
        comps = el.components
        n = len(comps[0].subscripts[0])
        if any(c.subscripts[1] for c in comps):
            return None
        for p in range(n):
            elems = [self._elem(c.subscripts[0][p]) if self._range_of(c.subscripts[0][p]) is None else None
                     for c in comps]
            if None in elems or len(set(elems)) != len(elems):
                continue
            others = {tuple(c.subscripts[0][q] for q in range(n) if q != p) for c in comps}
            if len(others) == 1 and all(self._range_of(x) is not None for x in next(iter(others))):
                return p
        return None

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
            if k not in self.elements:
                return False
            el = self.elements[k]
            if el.kind == "parameter":                    # a data series changes with time
                return isinstance(el.abstract.components[0], AbstractData) and \
                    isinstance(el.abstract.components[0].ast, (ae.GetDataStructure, ae.DataStructure))
            if direct[k]:
                memo[k] = True
                return True
            if k in seen:
                return False
            memo[k] = any(timed(d, seen + (k,)) for d in deps[k])
            return memo[k]

        for k, e in self.elements.items():
            e.timevarying = e.kind not in ("parameter", "constant_parts") and timed(k)

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
        problems = list(self.kind_errors)
        for e in self.elements.values():
            try:
                self._write_element(e)
            except DSLError as err:
                problems.append(str(err))
            decls_by_view.setdefault(e.view, []).append(e)
        if problems:
            raise DSLError(f"{len(problems)} elements can't be written in the DSL yet:\n  " + "\n  ".join(problems))
        files = []
        for view, els in decls_by_view.items():
            name = identifier(view or "equations", set()) if view else "equations"
            fname = f"{name}.jl"
            files.append((name + "_view", fname))
            (out / fname).write_text(self._view_text(name + "_view", view, els), encoding="utf-8")
        (out / "dims.jl").write_text(self._dims_text(), encoding="utf-8")
        (out / "inputs.jl").write_text(self._inputs_text(), encoding="utf-8")
        # the time unit too (XMILE's "Time", for instance)
        _, unknown = translate_units(self.controls["time_step"].units if "time_step" in self.controls else "")
        self.unknown_units.update(unknown)
        if self.unknown_units:
            (out / "units.jl").write_text(self._units_text(), encoding="utf-8")
        names = ["# Vensim's names of the model's variables, for results and comparisons", "vensim\tjulia"]
        names += [f"{e.name}\t{e.ident}" for e in self.elements.values()]
        (out / "names.tsv").write_text("\n".join(names) + "\n", encoding="utf-8")
        main = out / (identifier(self.path.stem, set()) + ".jl")
        main.write_text(self._main_text(files), encoding="utf-8")
        self._write_package(out, main)
        return main

    # The folder is also a Julia package, so that Julia can save the model built
    # and compiled: `using Pkg; Pkg.develop(path = …)` once, then `using Name`
    # loads it in seconds, and builds it again only when a file changes.
    _PACKAGES = {"PySD": "d7e3e0f0-7a2b-4e3a-9c1d-5a6b8c9d0e1f",
                 "DimensionalData": "0703355e-b756-11e9-17c0-8b28908087d0",
                 "Unitful": "1986cc42-f94f-5a68-af5c-568840ba703d"}

    def _package_name(self):
        name = "".join(w[:1].upper() + w[1:] for w in re.split(r"[^0-9A-Za-z]+", self.path.stem) if w)
        return name if name and name[0].isalpha() else "Model" + name

    def _write_package(self, out, main):
        import uuid
        name = self._package_name()
        deps = "\n".join(f'{k} = "{v}"' for k, v in self._PACKAGES.items())
        (out / "Project.toml").write_text(
            f'name = "{name}"\nuuid = "{uuid.uuid5(uuid.NAMESPACE_URL, "pysd-dsl-model:" + name)}"\n'
            f'version = "0.1.0"\n\n[deps]\n{deps}\n\n'
            f'[compat]\nPySD = "{PYSD_JL_VERSION}"\n', encoding="utf-8")
        (out / "src").mkdir(exist_ok=True)
        (out / "src" / f"{name}.jl").write_text("\n".join([
            f"# {self.path.name} as a package: `using {name}` loads the model built and",
            "# compiled, as Julia saved it the first time (and again when a file changes).",
            f"module {name}",
            "using PySD",
            f'@precompile_model model = include("../{main.name}")',
            "end"]) + "\n", encoding="utf-8")

    def _view_text(self, value, view, els):
        params = [e for e in els if e.kind == "parameter"]
        variables = [e for e in els if e.kind != "parameter"]
        hidden_params = [d for d in self.hidden.get(view, []) if " = " in d.split(",")[0]]
        hidden_vars = [d for d in self.hidden.get(view, []) if d not in hidden_params]
        out = [f"# {'View ' + repr(view) if view else 'The equations'}: {len(els)} variables.", "",
               f"{value} = @equations begin"]
        if params or hidden_params:
            out.append("    @parameters begin")
            out += ["        " + e.declaration.replace("\n", "\n        ") for e in params]
            out += ["        " + d.replace("\n", "\n        ") for d in hidden_params]
            out.append("    end")
        if variables or hidden_vars:
            out.append("    @variables begin")
            out += ["        " + e.declaration for e in variables]
            out += ["        " + d for d in hidden_vars]
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
            if r not in self.copies:
                labels = ", ".join(label(e) for e in self.ranges[r])
                out.append(f"const {self.elements_name[r]} = [{labels}]")
            note = f"  # a copy of {self.dimname[self.copies[r]]}" if r in self.copies else ""
            out.append(f"const {d} = Dim{{:{d}}}({self.elements_name[r]}){note}")
        return "\n".join(out) + "\n"

    def _inputs_text(self):
        out = ["# Input sheets, given to Model by name; a different scenario is a different",
               "# sheet given to the run: run_model(model; inputs = (; name = excel(…)))."]
        import os
        entries = [f"{ident} = excel(joinpath(@__DIR__, {jl_string(os.path.relpath(self.path.parent / file, self.out))}), "
                   f"{jl_string(tab)})" for (file, tab), ident in self.sheets.items()]
        if getattr(self, "uses_parents", False):
            out += ["# The data variables without an equation come from outside the model, by name:",
                    "# runs of other models or files of series, run_model(model; inputs = (; parents =",
                    "# (run_model(world),))) or (; parents = (series_file(\"world.tab\"),))."]
            entries.append("parents = ()")
        out.append("inputs = (; " + ",\n            ".join(entries) + ")" if entries else "inputs = (;)")
        return "\n".join(out) + "\n"

    def _units_text(self):
        out = ["# Units of the model that Unitful.jl and PySD.jl don't have: each its own dimension.",
               "import Unitful", "module ModelUnits", "using Unitful", "using Unitful: @dimension, @refunit", ""]
        for i, u in enumerate(sorted(self.unknown_units)):
            dim = f"𝐔{i}"
            out.append(f'@dimension {dim} "{u}" Dim{u}')
            out.append(f'@refunit {u} "{u}" Unit{u} {dim} false')
        # registered now, and again when a precompiled package loads it (__init__)
        out += ["", "const localfactors = copy(Unitful.basefactors)",
                "function __init__()", "    merge!(Unitful.basefactors, localfactors)",
                "    Unitful.register(ModelUnits)", "end", "end",
                "Unitful.register(ModelUnits)"]
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

    def _hide(self, view: str, declaration: str):
        self.hidden.setdefault(view, []).append(declaration)

    def _write_element(self, e: Element):
        meta = self._meta(e)
        if e.kind == "parameter":
            e.declaration = f"{self._decl_target(e)} = {self._parameter_value(e)},\n    {meta}"
            return
        e.declaration = f"{self._decl_target(e)}, {meta}"
        inits, init_parts = [], []
        for c in e.abstract.components:
            lhs = e.ident + self._lhs_selection(e, c)
            ctx = Context(self, e, c)
            ast = c.ast
            if isinstance(ast, ae.IntegStructure):
                e.lines.append(f"∂t({lhs}) ~ {ctx.visit(ast.flow).text}")
                init = Context(self, e, c, initial=True).visit(ast.initial).text
                if init not in inits:
                    inits.append(init)
                init_parts.append((self._lhs_selection(e, c), init, c))
            elif isinstance(ast, ae.CallStructure) and clean(ast.function.reference) == "invert_matrix" \
                    and len(e.dims) > 2:
                # one inverse per element of the leading dimensions: a loop, as in the design
                m = ctx.visit(ast.arguments[0])
                lead = [d for d in e.dims[:-2]]
                if len(lead) != 1 or m.dims[:1] != (lead[0],):
                    raise DSLError(f"{e.name}: INVERT MATRIX over {len(e.dims)} dimensions is not supported yet")
                d = lead[0]
                r = next(r for r, n in self.dimname.items() if n == d)
                var = f"each_{d}"
                sel = f"[{d} = At({var})]"
                target = lhs + sel if not lhs.endswith("]") else lhs[:-1] + f", {d} = At({var})]"
                e.lines.append(f"for {var} in {self.elements_name[r]}\n"
                               f"    {target} ~ {invert_matrix(m.wrap(P_ATOM) + sel, m.dims[1:])}\nend")
            elif isinstance(ast, ae.CallStructure) and clean(ast.function.reference) == "active_initial":
                e.lines.append(f"{lhs} ~ {ctx.visit(ast.arguments[0]).text}")
                e.lines.append(f"{e.ident}(t₀) ~ {ctx.visit(ast.arguments[1]).text}")
            else:
                e.lines.append(f"{lhs} ~ {ctx.visit(ast).text}")
        if len(inits) > 1:
            # the initial value has parts too: a variable of its own
            init = identifier(e.name + " initial", self.taken)
            timed = any(self._scan(c.ast.initial)[1] or any(
                self.elements[r].timevarying for r in self._scan(c.ast.initial)[0] if r in self.elements)
                for _, _, c in init_parts)
            dims = "[" + ", ".join(e.dims) + "]" if e.dims else ""
            unit = meta.split(",")[0].lstrip("[")
            self._hide(e.view, f"{init}{'(t)' if timed else ''}{dims}, [{unit}, description = "
                               f"\"The initial value of {e.ident}, by parts.\"]")
            e.lines += [f"{init}{sel} ~ {txt}" for sel, txt, _ in init_parts]
            e.lines.append(f"{e.ident}(t₀) ~ {init}")
        elif inits:
            e.lines.append(f"{e.ident}(t₀) ~ {inits[0]}")

    def _lhs_selection(self, e: Element, c) -> str:
        sels = [sel for _, _, _, sel in self._lhs_parts(e, c) if sel]
        return "[" + ", ".join(sels) + "]" if sels else ""

    def _lhs_parts(self, e: Element, c):
        """For each position of the part defined: its range (None for an element),
        the declared dimension, the elements, and the selector (None: all)."""
        subs, exceptions = c.subscripts
        out = []
        for p, s in enumerate(subs):
            dim_v = e.vensim_dims[p]
            dim = self.dimname[dim_v]
            full = self.ranges[dim_v]
            sel = self._selected(s, dim_v)
            excluded = set()
            for exc in exceptions or []:
                # an exception removes its elements at the positions where it is narrower
                # than the part: [dac_final_sources, sectors] removes two sources, every sector
                narrower = [q for q in range(len(exc))
                            if not set(self._selected(subs[q])) <= set(self._selected(exc[q]))]
                if len(narrower) > 1:
                    raise DSLError(f"{e.name}: an :EXCEPT: narrower at several positions is not supported yet")
                if narrower == [p]:
                    excluded |= set(self._selected(exc[p], dim_v))
            chosen = [x for x in sel if x not in excluded]
            r = self._range_of(s)
            if chosen == list(full):
                out.append((r, dim, chosen, None))
            elif excluded and len(sel) == len(full) and len(excluded) < len(chosen):
                ex = sorted(excluded, key=full.index)
                out.append((r, dim, chosen, f"{dim} = Not(At({self._labels(ex)}))"))
            else:
                out.append((r, dim, chosen, f"{dim} = At({self._labels(chosen, s)})"))
        return out

    def _selected(self, s, dim_v=None) -> List[str]:
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
        r = self._range_of(sub) if sub is not None else None
        if r is not None and list(self.ranges[r]) == list(elems):
            return self.elements_name[r]
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
        # one part per element of a dimension (declared first): one value per element
        p = e.split
        firsts = [self._elem(c.subscripts[0][p]) for c in comps]
        order = self.ranges[e.vensim_dims[p]]
        if set(firsts) != set(order):
            raise DSLError(f"{e.name}: its parts don't define every element of {self.dimname[e.vensim_dims[p]]}")
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
        if isinstance(ast, ae.DataStructure):
            # no equation: a parent model's results give it (by its name there)
            self.uses_parents = True
            return f"data(parents, :{e.ident})"
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


@dataclass
class _Literal:
    """An expression to write as a parameter's value (what _one_value takes)."""
    ast: object
    keyword: Optional[str] = None


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
        # the ranges the part defined iterates over: its dimension, elements, selector
        self.part = {r: (dim, elems, sel) for r, dim, elems, sel in builder._lhs_parts(element, component)
                     if r is not None}

    def visit(self, node) -> Code:
        if isinstance(node, (bool, np.bool_)):
            return Code("true" if node else "false")
        if isinstance(node, (int, float)):
            text = number(node)
            return Code(text, (), P_ATOM if not text.startswith("-") else P_UNARY)
        if isinstance(node, np.ndarray):                    # values for a part
            return Code(self.b._one_value(self.e, _Literal(node)))
        if isinstance(node, ae.GetConstantsStructure):
            text = self.b._one_value(self.e, _Literal(node))
            if not self.e.timevarying:
                return Code(text)
            # read once: a parameter of its own (a part that is one element)
            sel = self.b._lhs_selection(self.e, self.c)
            if any(ch in sel for ch in "[,") and "At(:" not in sel or sel.count("=") > 1:
                raise DSLError(f"{self.e.name}: Excel constants in a part of several elements of a "
                               "variable that changes with time are not supported yet")
            name = identifier(self.e.name + " " + (sel.split("At(:")[-1].rstrip(")]") if sel else "input"), self.b.taken)
            unit = self.b._meta(self.e).split(",")[0].strip("[]")
            self.b._hide(self.e.view, f"{name} = {text},\n    [{unit}, description = \"Read from Excel for {self.e.ident}.\"]")
            return Code(name)
        if isinstance(node, ae.InlineLookupsStructure):
            # WITH LOOKUP: the lookup a parameter of its own, called with the argument
            arg = self.visit(node.argument)
            name = identifier(self.e.name + " lookup", self.b.taken)
            unit = self.b._meta(self.e).split(",")[0].strip("[]")
            text = self.b._one_value(self.e, _Literal(node.lookups))
            self.b._hide(self.e.view, f"{name} = {text},\n    [{unit}, description = \"The lookup of {self.e.ident}.\"]")
            return Code(f"{name}({arg.text})", arg.dims)
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
            tau, init = self.visit(node.delay_time), self.visit(node.initial)
            # the initial value as the delayed value: of the whole array, as it, when
            # it's the same reference (Vensim's usual DELAY FIXED(x, τ, x))
            same = isinstance(node.initial, ae.ReferenceStructure) and \
                node.initial.reference == node.input.reference and \
                node.initial.subscripts == node.input.subscripts
            init_text = self.b.elements[clean(node.input.reference)].ident if same else init.text
            # the whole array delayed, then the part: x(t - τ; init)[d = At(…)]
            return self._of_whole(node.input, f"(t - {tau.wrap(P_ADD + 1)}; init = {init_text})")
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
            el = self.b.elements.get(clean(node.reference))
            if el is None or not (el.timevarying or self._is_data(el)):
                return self._reference(node)
            return self._of_whole(node, "(t₀)")
        raise DSLError(f"{self.e.name}: INITIAL of an expression is not supported yet")

    @staticmethod
    def _is_data(el) -> bool:
        return el.kind == "parameter" and isinstance(el.abstract.components[0], AbstractData)

    def _of_whole(self, node: ae.ReferenceStructure, call: str) -> Code:
        """`x(t₀)` or `x(t - τ; init)` of a reference: of the whole array, then the
        selection and the renames of the reference."""
        el = self.b.elements[clean(node.reference)]
        ref = self._reference(node)
        whole = el.ident + call
        base = f"{el.ident}(t)" if self._is_data(el) else el.ident
        text = ref.text.replace(base, whole, 1)
        return Code(text, ref.dims)

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

    def _condition(self, node) -> Code:
        """A condition: true or false.  Vensim also takes a number (nonzero: true)."""
        c = self.visit(node)
        if isinstance(node, ae.LogicStructure) or (isinstance(node, (bool, np.bool_))):
            return c
        return Code(f"{c.wrap(P_CMP + 1)} {self._op('!=', c)} 0", c.dims, P_CMP)

    def _logic(self, node) -> Code:
        """Comparisons, then :AND:, then :OR: (Vensim gives them as one flat list)."""
        ops = [o.lower() for o in node.operators]
        if len(node.arguments) == 1 and ops and ops[0] in (":not:", "not"):
            a = self._condition(node.arguments[0])
            return Code(("." if a.dims else "") + "!" + a.wrap(P_UNARY), a.dims, P_UNARY)
        items = [self.visit(node.arguments[0])]
        raw = [node.arguments[0]]
        for op, a in zip(ops, node.arguments[1:]):
            items += [op, self.visit(a)]
            raw.append(a)

        def bool_of(code, ast):
            if isinstance(ast, (ae.LogicStructure, bool, np.bool_)) or code.prec == P_CMP:
                return code
            return Code(f"{code.wrap(P_CMP + 1)} {self._op('!=', code)} 0", code.dims, P_CMP)

        # comparisons
        vals, opers, asts = [items[0]], [], [raw[0]]
        for i in range(1, len(items), 2):
            op, b = items[i], items[i + 1]
            if op in (":and:", ":or:", "and", "or"):
                opers.append(op)
                vals.append(b)
                asts.append(raw[(i + 1) // 2])
            else:
                a = vals[-1]
                jop = LOGIC[op]
                vals[-1] = Code(f"{a.wrap(P_CMP + 1)} {self._op(jop, a, b)} {b.wrap(P_CMP + 1)}",
                                self._union(a.dims, b.dims), P_CMP)
                asts[-1] = None
        vals = [bool_of(v, a) if a is not None else v for v, a in zip(vals, asts)]
        # :AND: before :OR:
        for group in (("and", ":and:"), ("or", ":or:")):
            i = 0
            while i < len(opers):
                if opers[i] in group:
                    a, b = vals[i], vals[i + 1]
                    jop = LOGIC[opers[i]]
                    prec = P_AND if jop == "&" else P_OR
                    vals[i:i + 2] = [Code(f"{a.wrap(prec)} {self._op(jop, a, b)} {b.wrap(prec + 1)}",
                                          self._union(a.dims, b.dims), prec)]
                    del opers[i]
                else:
                    i += 1
        return vals[0]

    def _lhs_declared(self, r: str) -> str:
        for p, s in enumerate(self.c.subscripts[0]):
            if self.b._range_of(s) == r:
                return self.e.vensim_dims[p]
        return r

    def _mapped_to(self, r: str):
        """The range of the part being defined that `r` maps to, and its dimension."""
        for p, s in enumerate(self.c.subscripts[0]):
            lr = self.b._range_of(s)
            if lr is not None and lr != r and (r, lr) in self.b.mapped:
                return lr, self.b.dimname[self.e.vensim_dims[p]]
        return None

    def _target(self, r: str) -> str:
        """The dimension a range in a reference lines up with: the declared one at
        the position where the part being defined iterates over it, else its own."""
        for p, s in enumerate(self.c.subscripts[0]):
            if self.b._range_of(s) == r:
                return self.b.dimname[self.e.vensim_dims[p]]
        return self.b.dimname[r]

    def _reference(self, node: ae.ReferenceStructure, bang_ok=False) -> Code:
        name = clean(node.reference)
        if name == "time":
            return Code("t")
        if name == "time_step":
            return Code("Δt")
        if name == "initial_time":
            return Code("t₀")
        el = self.b.elements.get(name)
        rng = next((r for r in self.b.ranges if clean(r) == name), None)
        if el is None and rng is not None:
            return self._positions(rng)
        if el is None:
            if name in self.b.controls:                 # FINAL TIME, SAVEPER: their value
                return Code(number(self.b._control(name)))
            raise DSLError(f"{self.e.name}: unknown name {node.reference!r}")
        subs = node.subscripts.subscripts if node.subscripts else []
        base = el.ident
        if self._is_data(el):
            base = f"{el.ident}(t)"                       # a data series: its value now
        if not subs:
            return Code(base, el.dims)
        sels, renames, dims = [], [], []
        for p, s in enumerate(subs):
            dv = el.vensim_dims[p]
            d = self.b.dimname[dv]
            r = self.b._range_of(s)
            if r is None:                                   # an element
                e = self.b._elem(s)
                sels.append(f"{d} = At({label(e)})")
                continue
            target = self._target(r)
            want, have = self.b.ranges[r], self.b.ranges[dv]
            if r in self.part:                              # the part's elements only
                pdim, want, psel = self.part[r]
                if list(want) != list(have):
                    if not set(want) <= set(have):
                        raise DSLError(f"{self.e.name}: {node.reference}[{s}]: elements it hasn't")
                    same_dim = list(have) == list(self.b.ranges[self._lhs_declared(r)])
                    sels.append(psel.replace(f"{pdim} =", f"{d} =", 1) if psel and same_dim
                                else f"{d} = At({self.b._labels(want, s if list(want) == list(self.b.ranges[r]) else None)})")
                if pdim != d:
                    renames.append(f":{d} => :{pdim}")
                dims.append(pdim)
                continue
            partner = self._mapped_to(r)
            if partner is not None:
                # a mapping: these elements, with the labels of the range they map to
                lhs_range, lhs_dim = partner
                if list(want) != list(have):
                    if not set(want) <= set(have):
                        raise DSLError(f"{self.e.name}: {node.reference}[{s}]: this mapping is not supported yet")
                    sels.append(f"{d} = At({self.b.elements_name[r]})")
                if len(self.b.ranges[lhs_range]) != len(want):
                    raise DSLError(f"{self.e.name}: {node.reference}[{s}]: a mapping between ranges of different sizes")
                new = lhs_dim if self.b.ranges[lhs_range] == self.b.ranges[self._lhs_declared(lhs_range)] and \
                    lhs_dim == self.b.dimname[self._lhs_declared(lhs_range)] else \
                    f"Dim{{:{lhs_dim}}}({self.b.elements_name[lhs_range]})"
                renames.append(f":{d} => {new}")
                dims.append(lhs_dim)
                continue
            if not set(want) <= set(have):
                raise DSLError(f"{self.e.name}: {node.reference}[{s}]: subscript mappings are not supported yet")
            if list(want) != list(have):                    # some of its elements, or in another order
                sels.append(f"{d} = At({self.b.elements_name[r]})")
            if target != d:                                 # under the name it lines up with
                renames.append(f":{d} => :{target}")
            dims.append(target)
        text = base
        if sels:
            text += "[" + ", ".join(sels) + "]"
        if renames:
            text = f"set({text}, {', '.join(renames)})"
        return Code(text, tuple(dims))

    def _positions(self, r: str) -> Code:
        """A subscript used as a value: the positions of its elements (1, 2, …)."""
        if r not in self.b.positions:
            name = identifier(r + " position", self.b.taken)
            self.b.positions[r] = name
            d = self.b.dimname[r]
            self.b._hide(self.e.view, f"{name}[{d}] = 1:length({d}),\n    [unit = NoUnits, "
                                      f"description = \"The position of each element of {d}.\"]")
        name, d, target = self.b.positions[r], self.b.dimname[r], self._target(r)
        text = name if target == d else f"set({name}, :{d} => :{target})"
        return Code(text, (target,))

    def _call(self, node: ae.CallStructure) -> Code:
        fname = clean(node.function.reference)
        args = node.arguments
        if fname in self.b.elements:                         # a lookup
            f = self.b.elements[fname]
            fref = self._reference(node.function) if node.function.subscripts else Code(f.ident, f.dims)
            x = self.visit(args[0])
            if f.dims:
                # called first, then the part: set(f(x)[d = At(…)], …)
                text = fref.text.replace(f.ident, f"{f.ident}({x.text})", 1)
                return Code(text, self._union(fref.dims, x.dims))
            dot = "." if x.dims else ""
            return Code(f"{fref.text}{dot}({x.text})", x.dims)
        if fname in ("if_then_else",):
            c, a, b = self._condition(args[0]), self.visit(args[1]), self.visit(args[2])
            dot = "." if (c.dims or a.dims or b.dims) else ""
            return Code(f"ifelse{dot}({c.text}, {a.text}, {b.text})", self._union(c.dims, a.dims, b.dims))
        if fname == "pi":
            return Code("π")
        if fname in ("pulse_train", "quantum"):
            # PySD.jl's, as Vensim (the parser gives PULSE TRAIN's arguments
            # as start, interval, width, end)
            cs = [self.visit(x) for x in args]
            dims = self._union(*(c.dims for c in cs))
            dot = "." if dims else ""
            first = ["t"] if fname == "pulse_train" else []
            return Code(f"pysd_{fname}{dot}({', '.join(first + [c.text for c in cs])})", dims)
        if fname in ("vmin_xmile", "vmax_xmile"):
            # XMILE's MIN and MAX: of all the elements of one array, or element by element
            cs = [self.visit(x) for x in args]
            f = "min" if fname == "vmin_xmile" else "max"
            if len(cs) == 1:
                if not cs[0].dims:
                    return cs[0]
                over = cs[0].dims[0] if len(cs[0].dims) == 1 else "(" + ", ".join(cs[0].dims) + ")"
                return Code(f"{'minimum' if f == 'min' else 'maximum'}_over({cs[0].text}, {over})", ())
            dims = self._union(*(c.dims for c in cs))
            return Code(f"{f}{'.' if dims else ''}({', '.join(c.text for c in cs)})", dims)
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
            dims = self._union(h.dims, t0.dims)
            ge = ".≥" if t0.dims else "≥"
            return Code(f"{h.wrap(P_MUL)} {'.*' if dims else '*'} (t {ge} {t0.text})", dims, P_MUL)
        if fname == "ramp":
            s, t0 = self.visit(args[0]), self.visit(args[1])
            t1 = self.visit(args[2]) if len(args) > 2 else None
            times = self._union(t0.dims, t1.dims if t1 else ())
            sub = ".-" if times else "-"
            hi = f"{t1.wrap(P_ADD + 1)} {sub} {t0.wrap(P_ADD + 1)}" if t1 else "Inf"
            dims = self._union(s.dims, times)
            clamp = f"clamp{'.' if times else ''}(t {sub} {t0.wrap(P_ADD + 1)}, 0, {hi})"
            return Code(f"{s.wrap(P_MUL)} {'.*' if dims else '*'} {clamp}", dims, P_MUL)
        if fname == "pulse":
            t0, w = self.visit(args[0]), self.visit(args[1])
            dims = self._union(t0.dims, w.dims)
            if dims:
                return Code(f"({t0.text} .≤ t .< {t0.wrap(P_ADD)} .+ {w.wrap(P_ADD + 1)})", dims)
            return Code(f"({t0.text} ≤ t < {t0.wrap(P_ADD)} + {w.wrap(P_ADD + 1)})", ())
        if fname in REDUCTIONS:
            x = self.visit(args[0])
            summed = [self._target(self.b._range_of(s)) for s in self._bangs(args[0])]
            if not summed:
                raise DSLError(f"{self.e.name}: {fname.upper()} without a subscript with !")
            dims = list(x.dims)
            for d in summed:
                if d not in dims:
                    raise DSLError(f"{self.e.name}: summing over {d}, which {x.text} hasn't")
                dims.remove(d)
            # over several dimensions at once, as Vensim and Python (the order of
            # the additions, and so the last digits, are numpy's)
            over = summed[0] if len(summed) == 1 else "(" + ", ".join(summed) + ")"
            return Code(f"{REDUCTIONS[fname]}({x.text}, {over})", tuple(dims))
        if fname == "elmcount":
            r = self.b._range_of(args[0].reference if isinstance(args[0], ae.ReferenceStructure) else str(args[0]))
            if r is None:
                raise DSLError(f"{self.e.name}: ELMCOUNT of {args[0]!r}")
            return Code(f"length({self.b.dimname[r]})")
        if fname == "invert_matrix":
            m = self.visit(args[0])
            return Code(invert_matrix(m.text, m.dims), m.dims)
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
