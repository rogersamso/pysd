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
    data_format : str, optional
        How to store external numeric data.  ``"hardcoded"`` (default) inlines
        all values as Julia literals in the generated ``.jl`` file.
        ``"json"`` writes a companion ``<model>_data.json`` file and generates
        Julia code that reads it at startup via ``JSON3.jl``.
    """

    def __init__(
        self,
        abstract_model: AbstractModel,
        data_format: str = "hardcoded",
    ) -> None:
        if data_format not in ("hardcoded", "json"):
            raise ValueError(
                f"data_format must be 'hardcoded' or 'json', got {data_format!r}"
            )
        self.original_path = abstract_model.original_path
        self.sections = [
            JuliaSectionBuilder(section, data_format=data_format)
            for section in abstract_model.sections
        ]

    def build_model(self) -> Path:
        """Translate all sections and return the path to the main ``.jl`` file.

        The first section is always the main model.  Any additional sections
        are Vensim macros; each gets its own ``<macro_name>.jl`` companion file.
        """
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
    data_format : str, optional
        ``"hardcoded"`` (default) or ``"json"``.  When ``"json"``, numeric data
        is written to a companion ``<model>_data.json`` file and the generated
        Julia code reads it at startup via ``JSON3.jl``.
    """

    def __init__(
        self,
        abstract_section: AbstractSection,
        data_format: str = "hardcoded",
    ) -> None:
        self.name: str = abstract_section.name
        self.path: Path = abstract_section.path.with_suffix(".jl")
        self.root: Path = self.path.parent
        self.model_name: str = self.path.stem
        self.split: bool = abstract_section.split
        self.views_dict: Optional[dict] = abstract_section.views_dict
        self.abstract_elements: List[AbstractElement] = list(abstract_section.elements)
        self._abstract_subscripts = abstract_section.subscripts
        self.data_format: str = data_format
        # JSON data accumulator — populated when data_format == "json"
        self._json_data: Dict[str, dict] = {
            "constants": {},
            "lookups": {},
            "data": {},
        }

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
        # Names of identifiers that are lookup/data functions (need `(t)` when referenced bare)
        self._lookup_func_names: Set[str] = set()

        # Reverse map: element label → parent range name (for per-element component coords)
        self._elem_to_range: Dict[str, str] = {}
        for sr in self._abstract_subscripts:
            if isinstance(sr.subscripts, list):
                for elem_label in sr.subscripts:
                    if elem_label not in self._elem_to_range:
                        self._elem_to_range[elem_label] = sr.name

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

    def _build_macro_section(self) -> None:
        """Generate a companion ``.jl`` file for a Vensim macro section.

        The file declares the macro's variables, builds its equations in a
        vector ``{macro_name}_eqs``, and writes the file to
        ``{model_stem}_{macro_name}.jl`` next to the main model.
        """
        # Populate namespace
        for elem in self.abstract_elements:
            self.namespace.add_to_namespace(elem.name)

        # Process all elements (macros have no control elements)
        for elem in self.abstract_elements:
            identifier = self.namespace.namespace[elem.name]
            eqs = self._process_element(elem, identifier, is_control=False)
            self.built_elements[identifier] = (eqs, False)

        # Register inline lookups
        for lut_name, xs, ys, itp_type in self.inline_registry.entries:
            const_decl, func_decl, reg_decl = lookup_interpolation_code(
                lut_name, xs, ys, itp_type
            )
            self.lookup_const_decls.append(const_decl)
            self.lookup_func_decls.append(func_decl)
            self.lookup_register_decls.append(reg_decl)

        all_eqs: List[str] = []
        for eqs, _ in self.built_elements.values():
            all_eqs.extend(eqs)

        macro_jl_name = re.sub(r"[^a-z0-9_]", "_", self.name.lower())
        eq_var = f"{macro_jl_name}_eqs"
        eq_lines = ",\n    ".join(all_eqs) if all_eqs else ""
        uses = ["ModelingToolkit", "Symbolics"]
        if self.lookup_const_decls:
            uses.append("DataInterpolations")
        if self.data_format == "json":
            uses.append("JSON3")
        using_line = f"using {', '.join(uses)}"

        text = textwrap.dedent(f"""\
            # Macro {self.name}
            # Translated using PySD version {__version__}

            {using_line}

            {self._helpers_block()}
            {self._lookup_block()}
            {self._declarations_block()}
            {eq_var} = Equation[
                {eq_lines}
            ]
            """)

        # Write to {main_stem}_{macro_name}.jl next to the main model.
        # Update self.path BEFORE _write_data_json so the companion .json
        # file lands next to the macro .jl, not the main model.
        self.path = self.path.with_name(
            f"{self.path.stem}_{macro_jl_name}.jl"
        )
        if self.data_format == "json":
            self._write_data_json()
        self.path.write_text(text, encoding="UTF-8")

    def build_section(self) -> None:
        """Build the section, writing one or more ``.jl`` files.

        For macro sections (``type == 'macro'``) a standalone companion
        ``.jl`` file is generated containing the macro's equations as a
        Julia ``Equation`` vector named ``{macro_name}_eqs``.  The file
        is written next to the main model file.
        """
        is_macro = (self.name != "__main__")

        if is_macro:
            self._build_macro_section()
            return

        # First pass: populate the namespace with all element names
        for elem in self.abstract_elements:
            self.namespace.add_to_namespace(elem.name)

        # Pre-populate _var_dims for every subscripted element so that the
        # subscript-indexed visitor can correctly index forward-referenced
        # variables even when they haven't been processed yet.
        for elem in self.abstract_elements:
            identifier = self.namespace.namespace.get(elem.name)
            if identifier:
                dims = self._element_dims(elem)
                if dims:
                    self._var_dims[identifier] = [d for d, _ in dims]

        # Pre-populate _lookup_func_names for GET DATA / GET LOOKUPS elements
        # so that bare references to them in equations auto-call f(t).
        for elem in self.abstract_elements:
            identifier = self.namespace.namespace.get(elem.name)
            if identifier and elem.components:
                if all(isinstance(c.ast, GetLookupsStructure) for c in elem.components) or \
                   any(isinstance(c.ast, GetDataStructure) for c in elem.components):
                    self._lookup_func_names.add(identifier)

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
            if self.data_format == "json":
                self._json_data["lookups"][lut_name] = {
                    "x": list(xs), "y": list(ys),
                    "interp_type": itp_type, "subscripts": [],
                }

        if self.split and self.views_dict:
            self._build_modular()
        else:
            self._build()

    # ------------------------------------------------------------------
    # Subscript helpers
    # ------------------------------------------------------------------

    def _comp_coords(self, comp: "AbstractComponent") -> Dict[str, list]:
        """Build an ``{range_name: [element_labels]}`` coords dict for *comp*.

        Each item in ``comp.subscripts[0]`` may be either a subscript-range
        name (→ use all its elements) or a specific element name (→ resolve to
        its parent range with a single-element list).  This mirrors what the
        Python builder passes to ``ExtLookup``/``ExtData``/``ExtConstant``.
        """
        def_subs = comp.subscripts[0] if comp.subscripts else []
        if not def_subs:
            return {}
        result: Dict[str, list] = {}
        for s in def_subs:
            if s in self._subs_elems:
                # Range name → full element list
                result[s] = self._subs_elems[s]
            elif s in self._elem_to_range:
                # Specific element → map to parent range with single-element list
                parent = self._elem_to_range[s]
                result[parent] = [s]
            else:
                result[s] = []
        return result

    def _infer_parent_range(self, elements: List[str]) -> Optional[str]:
        """Return the smallest subscript range that contains *all* given elements.

        Used to resolve ambiguity when the same element name appears in
        multiple ranges (e.g. ``Agriculture`` is in both ``sectors`` and
        ``SECTORS_and_HOUSEHOLDS``).
        """
        candidates = []
        for sr in self._abstract_subscripts:
            if isinstance(sr.subscripts, list) and all(
                e in sr.subscripts for e in elements
            ):
                candidates.append((len(sr.subscripts), sr.name))
        return min(candidates, key=lambda x: x[0])[1] if candidates else None

    def _infer_parent_range_not_in(
        self, elements: List[str], exclude: Optional[set] = None
    ) -> Optional[str]:
        """Like :meth:`_infer_parent_range` but skips ranges in *exclude*.

        Used when a candidate range is already occupied by a non-split
        subscript position (e.g. ``final_sources`` used for pos 1 should
        not also be the parent for the split pos 2 — use ``final_sources1``
        instead).
        """
        exclude = exclude or set()
        candidates = []
        for sr in self._abstract_subscripts:
            if isinstance(sr.subscripts, list) and sr.name not in exclude:
                if all(e in sr.subscripts for e in elements):
                    candidates.append((len(sr.subscripts), sr.name))
        return min(candidates, key=lambda x: x[0])[1] if candidates else None

    def _detect_split_ranges(
        self, components: List["AbstractComponent"]
    ) -> Dict[int, str]:
        """Return ``{position: parent_range}`` for subscript positions that
        carry different element names across *components*.

        Non-split positions' ranges are recorded first so that the split
        position is assigned a *different* range when the naive best-fit would
        collide (e.g. ``efficiency_rate_of_substitution`` has ``final_sources``
        at pos 1 and the split at pos 2 also maps to ``final_sources`` — we
        instead assign ``final_sources1`` to avoid the collision).
        """
        all_subs = [
            c.subscripts[0]
            for c in components
            if c.subscripts and c.subscripts[0]
        ]
        if not all_subs:
            return {}
        n_pos = len(all_subs[0])

        # Collect ranges committed by non-split (constant) positions
        committed: set = set()
        for pos in range(n_pos):
            vals = list({s[pos] for s in all_subs if len(s) > pos})
            if len(vals) == 1:
                s = vals[0]
                if s in self._subs_elems:
                    committed.add(s)
                elif s in self._elem_to_range:
                    committed.add(self._elem_to_range[s])

        result: Dict[int, str] = {}
        for pos in range(n_pos):
            vals = list({s[pos] for s in all_subs if len(s) > pos})
            if len(vals) > 1:
                parent = self._infer_parent_range_not_in(vals, exclude=committed)
                if parent is None:
                    parent = self._infer_parent_range(vals)
                if parent:
                    result[pos] = parent
        return result

    def _comp_coords_split(
        self,
        comp: "AbstractComponent",
        split_ranges: Dict[int, str],
    ) -> Dict[str, list]:
        """Like :meth:`_comp_coords` but uses *split_ranges* to override the
        parent-range lookup for positions that vary across components.
        """
        subs = comp.subscripts[0] if comp.subscripts else []
        result: Dict[str, list] = {}
        for pos, s in enumerate(subs):
            if pos in split_ranges:
                result[split_ranges[pos]] = [s]
            elif s in self._subs_elems:
                result[s] = self._subs_elems[s]
            elif s in self._elem_to_range:
                result[self._elem_to_range[s]] = [s]
            else:
                result[s] = []
        return result

    def _element_dims(self, elem: "AbstractElement") -> List[Tuple[str, int]]:
        """Return ``[(dim_name, dim_size), ...]`` for *elem*'s defining subscripts.

        Uses the first component's first subscript list.  Dims with size == 0
        (unresolved aliases) are filtered out.

        When a multi-component element has per-element subscripts (e.g.
        ``['electricity']``, ``['heat']``, …) rather than a range name, the
        parent range is inferred from all components' element labels so the
        variable is correctly declared as an array.
        """
        if not elem.components:
            return []
        comp = elem.components[0]
        if not comp.subscripts or not comp.subscripts[0]:
            return []
        dims = []
        for pos, dim_name in enumerate(comp.subscripts[0]):
            size = self._subs_sizes.get(dim_name, 0)
            if size > 0:
                # Check whether other components reference elements that fall
                # outside this range (the "range + sibling-element" pattern,
                # e.g. C_in_Deep_Ocean[upper] + C_in_Deep_Ocean[Layer4]).
                if len(elem.components) > 1:
                    range_elems = set(self._subs_elems.get(dim_name, []))
                    all_labels: set = set()
                    for c in elem.components:
                        if c.subscripts and len(c.subscripts[0]) > pos:
                            s = c.subscripts[0][pos]
                            if s in self._subs_elems:
                                all_labels.update(self._subs_elems[s])
                            elif s in self._elem_to_range:
                                all_labels.add(s)
                    if all_labels and not all_labels <= range_elems:
                        parent = self._infer_parent_range(list(all_labels))
                        if parent and self._subs_sizes.get(parent, 0) > size:
                            dim_name = parent
                            size = self._subs_sizes[parent]
                dims.append((dim_name, size))
            elif dim_name in self._elem_to_range:
                # dim_name is a specific element — infer the parent range from
                # all components' element at this position, or fall back to the
                # first known parent.
                if len(elem.components) > 1:
                    all_elems_at_pos = list({
                        c.subscripts[0][pos]
                        for c in elem.components
                        if c.subscripts and len(c.subscripts[0]) > pos
                    })
                    parent = self._infer_parent_range(all_elems_at_pos) or self._elem_to_range[dim_name]
                else:
                    parent = self._elem_to_range[dim_name]
                parent_size = self._subs_sizes.get(parent, 0)
                if parent_size > 0:
                    dims.append((parent, parent_size))
        return dims

    def _jl_n(self, dim_name: str) -> str:
        """Julia constant name for the size of a subscript dimension."""
        return "N_" + re.sub(r"[^a-z0-9]", "_", dim_name.lower()).upper()

    def _range_str(self, dims: List[Tuple[str, int]]) -> str:
        """Build ``'1:N_D0, 1:N_D1, ...'`` for array declarations."""
        return ", ".join(f"1:{self._jl_n(d)}" for d, _ in dims)

    def _per_index_subs(
        self,
        dim_name: str,
        dim_elems: List[str],
        abs_idx: int,
        def_range_name: Optional[str],
    ) -> Dict[str, str]:
        """Build ``active_subs`` for a per-index visitor in EXCEPT expansion.

        When iterating over a sub-range (*def_range_name*), Vensim aligns
        same-size ranges positionally: if we are at position *p* within the
        defining range, a reference ``[other_range]`` of the same size refers
        to element *other_range[p]*.  We pre-compute the absolute Julia array
        index for each such range so the expression visitor resolves them
        correctly without needing to understand range aliasing.
        """
        subs: Dict[str, str] = {dim_name: str(abs_idx)}
        if def_range_name is None or def_range_name == dim_name:
            return subs

        def_elems = self._subs_elems.get(def_range_name, [])
        if not def_elems:
            return subs

        element_label = dim_elems[abs_idx - 1]
        if element_label not in def_elems:
            return subs

        pos = def_elems.index(element_label)   # 0-based position within def_range
        def_size = len(def_elems)
        dim_idx_map = {e: i + 1 for i, e in enumerate(dim_elems)}

        # Add the defining range mapped to the absolute parent-dimension index.
        subs[def_range_name] = str(abs_idx)

        # For every range of the same size, map it to the absolute index of its
        # p-th element in the parent dimension (positional alignment).
        for sr in self._abstract_subscripts:
            if (
                isinstance(sr.subscripts, list)
                and len(sr.subscripts) == def_size
                and sr.name != def_range_name
                and sr.name != dim_name
            ):
                aligned_elem = sr.subscripts[pos]
                if aligned_elem in dim_idx_map:
                    subs[sr.name] = str(dim_idx_map[aligned_elem])
                else:
                    subs[sr.name] = str(pos + 1)   # fallback: position

        return subs

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
            subs_sizes=self._subs_sizes, subs_elems=self._subs_elems,
            lookup_names=self._lookup_func_names, root=self.root,
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
    # Limits helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _limits_comment(elem: "AbstractElement") -> str:
        """Return a ``# limits: [min, max]`` comment if *elem* has non-trivial limits,
        otherwise return an empty string."""
        lims = getattr(elem, "limits", (None, None))
        if not lims or (lims[0] is None and lims[1] is None):
            return ""
        lo = "-Inf" if lims[0] is None else format_number(float(lims[0]))
        hi = "Inf" if lims[1] is None else format_number(float(lims[1]))
        return f"  # limits: [{lo}, {hi}]"

    def _json_add_limits(self, elem: "AbstractElement", identifier: str) -> None:
        """Store limits metadata into ``_json_data["constants"]`` when in json mode."""
        lims = getattr(elem, "limits", (None, None))
        if not lims or (lims[0] is None and lims[1] is None):
            return
        entry = self._json_data["constants"].get(identifier)
        if entry is not None:
            entry["limits"] = [
                None if lims[0] is None else float(lims[0]),
                None if lims[1] is None else float(lims[1]),
            ]

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

        # ---- EXCEPT subscript exclusion / per-element multi-component ----
        # Delegate when:
        # (a) at least one component has an :EXCEPT: clause, OR
        # (b) multiple components each cover a specific element (not a full range)
        #     of the same subscript dimension — this is the Vensim pattern for
        #     piecewise-defined auxiliaries (e.g. hist_share[elec]=0, [heat]=0,
        #     [liquids]=f(...)).
        if len(elem.components) > 1:
            _has_except = any(comp.subscripts[1] for comp in elem.components)
            # Only apply per-element detection to plain auxiliary/constant
            # components — skip when the element uses external structures
            # (GET LOOKUPS, GET DATA, GET CONSTANTS) which have their own
            # dedicated handlers.
            _is_external = any(
                isinstance(c.ast, (GetLookupsStructure, GetDataStructure, GetConstantsStructure))
                for c in elem.components
            )
            _has_per_elem = not _is_external and any(
                c.subscripts and c.subscripts[0]
                and c.subscripts[0][0] not in self._subs_elems
                and c.subscripts[0][0] in self._elem_to_range
                for c in elem.components
            )
            if _has_except or _has_per_elem:
                return self._process_except_element(elem, identifier, is_control)

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
            var_dims=self._var_dims, subs_sizes=self._subs_sizes,
            subs_elems=self._subs_elems, lookup_names=self._lookup_func_names,
            root=self.root,
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
        # Vensim's INITIAL(x) returns the value of x at t=0.  Two strategies:
        #
        # (a) If the inner value can be resolved at translation time → @parameters.
        # (b) Otherwise → implement as a zero-derivative stock so MTK evaluates
        #     the initial condition at t0 and holds it constant:
        #       D(initial_var) ~ 0.0 ; initial_var(t0) = inner_expr
        if isinstance(ast, InitialStructure):
            val = self._resolve_initial_value(ast.initial)
            if val is not None:
                if not is_control:
                    self.param_decls.append(f"@parameters {identifier} = {val}")
                return []
            # Fall back: frozen stock — D = 0, initial value = inner expression.
            return self._expand_initial_frozen_stock(
                identifier, ast.initial, dims, ndim
            )

        # ---- Stock (INTEG) ---------------------------------------------
        if isinstance(ast, IntegStructure):
            flow_expr = visitor.visit(ast.flow)
            initial_expr = visitor.visit(ast.initial)
            if ndim == 0:
                self.stock_decls.append(f"@variables {identifier}(t)")
                self.u0_entries.append(f"{identifier} => {initial_expr}")
                return [f"D({identifier}) ~ {flow_expr}"]
            elif ndim == 1:
                (d0, n0) = dims[0]
                vnd1 = self._nd_visitor(dims, ["_i0"])
                flow_nd1 = vnd1.visit(ast.flow)
                init_nd1 = vnd1.visit(ast.initial)
                self.stock_decls.append(
                    f"@variables {identifier}(t)[{self._range_str(dims)}]"
                )
                for i in range(1, n0 + 1):
                    self.u0_entries.append(
                        f"{identifier}[{i}] => {init_nd1.replace('_i0', str(i))}"
                    )
                return [
                    f"[D({identifier}[_i0]) ~ {flow_nd1} "
                    f"for _i0 in 1:{self._jl_n(d0)}]..."
                ]
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
            return self._expand_smooth(identifier, ast, visitor, order=1, dims=dims)

        # ---- Higher-order Smooth / SmoothN -----------------------------
        if isinstance(ast, (SmoothStructure, SmoothNStructure)):
            try:
                order = int(ast.order)
            except (TypeError, ValueError):
                warn(
                    f"SMOOTH with non-integer order for '{elem.name}'; defaulting to 3."
                )
                order = 3
            return self._expand_smooth(identifier, ast, visitor, order=order, dims=dims)

        # ---- Delay (integer order) -------------------------------------
        if isinstance(ast, (DelayStructure, DelayNStructure)):
            try:
                order = int(ast.order)
            except (TypeError, ValueError):
                warn(
                    f"DELAY with non-integer order for '{elem.name}'; defaulting to 3."
                )
                order = 3
            return self._expand_delay(identifier, ast, visitor, order=order, dims=dims)

        # ---- DELAY FIXED ------------------------------------------------
        # Approximate DELAY FIXED as a first-order ODE delay (same formula
        # as DELAY1 with order=1).  True fixed delays need a DDE solver
        # that ModelingToolkit/OrdinaryDiffEq does not support, so this is
        # the best we can do in the MTK ODE framework.
        if isinstance(ast, DelayFixedStructure):
            return self._expand_delay_fixed(identifier, ast, visitor, dims=dims)

        # ---- External constant (GET XLS/DIRECT CONSTANTS) ----------------
        # Also catches piecewise-constant elements where some components are
        # GCS and others are plain numeric literals (e.g. var[fuel1]=GCS,
        # var[electricity]=0, var[heat]=0).
        # Require at least one GCS so pure-literal or stock elements are not
        # accidentally routed here.
        _const_like = any(
            isinstance(c.ast, GetConstantsStructure) for c in elem.components
        ) and all(
            isinstance(c.ast, GetConstantsStructure) or isinstance(c.ast, (int, float))
            for c in elem.components
        )
        if _const_like:
            julia_val = self._read_get_constants(elem, identifier)
            if julia_val is not None:
                if is_control:
                    if identifier in self.control_vals:
                        self.control_vals[identifier] = julia_val
                    return []
                if self.data_format == "json":
                    self._json_accumulate_constant(elem, identifier, julia_val)
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
        # Guard: only route to GET DATA when at least one component actually
        # carries a GetDataStructure (avoids misrouting variables that are typed
        # as AbstractData but whose equation is a plain CallStructure).
        _has_get_data_ast = any(
            isinstance(c.ast, GetDataStructure) for c in elem.components
        )
        if isinstance(comp, AbstractData) and not _has_get_data_ast:
            warn(
                f"'{elem.name}' is a DATA variable but its equation is not "
                "GET DATA — data-override mechanism not supported in the Julia "
                "builder; emitting as a regular auxiliary."
            )
        if (isinstance(ast, GetDataStructure) or isinstance(comp, AbstractData)) and _has_get_data_ast:
            return self._process_get_data(elem, identifier, comp)

        # ---- TREND ------------------------------------------------------
        if isinstance(ast, TrendStructure):
            return self._expand_trend(identifier, ast, visitor)

        # ---- FORECAST ---------------------------------------------------
        if isinstance(ast, ForecastStructure):
            return self._expand_forecast(identifier, ast, visitor)

        # ---- SAMPLE IF TRUE ---------------------------------------------
        if isinstance(ast, SampleIfTrueStructure):
            return self._expand_sample_if_true(identifier, ast, visitor, dims=dims, ndim=ndim)

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
            lim_comment = self._limits_comment(elem)
            if ndim == 0:
                self.param_decls.append(
                    f"@parameters {identifier} = {value_expr}{lim_comment}"
                )
                if self.data_format == "json":
                    try:
                        self._json_data["constants"][identifier] = {
                            "dims": [], "coords": {},
                            "values": float(value_expr),
                            "units": elem.units or "",
                        }
                        self._json_add_limits(elem, identifier)
                    except (ValueError, TypeError):
                        pass
            else:
                self.param_decls.append(
                    f"@parameters {identifier}[{self._range_str(dims)}] = {value_expr}{lim_comment}"
                )
            return []

        # ---- Auxiliary variable (algebraic) ----------------------------
        if ndim == 0:
            rhs_expr = visitor.visit(ast)
            if is_control:
                if identifier in self.control_vals:
                    self.control_vals[identifier] = rhs_expr
                return []
            lim_comment = self._limits_comment(elem)
            self.aux_decls.append(f"@variables {identifier}(t)")
            return [f"{identifier} ~ {rhs_expr}{lim_comment}"]
        elif ndim == 1:
            (d0, n0) = dims[0]
            vnd1 = self._nd_visitor(dims, ["_i0"])
            rhs_nd1 = vnd1.visit(ast)
            if is_control:
                if identifier in self.control_vals:
                    self.control_vals[identifier] = rhs_nd1
                return []
            self.aux_decls.append(
                f"@variables {identifier}(t)[{self._range_str(dims)}]"
            )
            return [
                f"[{identifier}[_i0] ~ {rhs_nd1} "
                f"for _i0 in 1:{self._jl_n(d0)}]..."
            ]
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
    # EXCEPT subscript exclusion
    # ------------------------------------------------------------------

    def _process_except_element(
        self,
        elem: AbstractElement,
        identifier: str,
        is_control: bool,
    ) -> List[str]:
        """Handle multi-component elements that use ``:EXCEPT:`` subscript exclusion.

        For each component we determine which integer indices it covers (the
        component's defined range minus the EXCEPT-excluded elements) and
        emit one equation per covered index.  The variable declaration
        (``@variables`` or ``@parameters``) is still emitted once for the
        full range.

        Limitations:
        - Only 1-D subscripted auxiliaries and constants are handled.
        - Components covering more than one dimension are not yet supported
          and fall back to a ``UserWarning`` + plain broadcast equation.
        """
        dims = self._element_dims(elem)
        ndim = len(dims)

        if ndim != 1:
            if ndim == 2:
                return self._process_except_element_2d(
                    elem, identifier, dims, is_control
                )
            warn(
                f"EXCEPT subscript exclusion for '{elem.name}' with {ndim}D "
                "subscripts is not yet supported — emitting plain broadcast equation."
            )
            # Fallback: use first component, ignore EXCEPT
            comp = elem.components[0]
            visitor = JuliaASTVisitor(
                self.namespace, self.inline_registry, self.needed_helpers,
                subs_sizes=self._subs_sizes, root=self.root,
            )
            rhs = visitor.visit(comp.ast)
            if not is_control:
                self.aux_decls.append(
                    f"@variables {identifier}(t)[{self._range_str(dims)}]"
                )
            return [f"Symbolics.scalarize({identifier} .~ {rhs})..."]

        dim_name, dim_size = dims[0]
        dim_elems = self._subs_elems.get(dim_name, [])

        # Build a map: element_label → 1-based index
        label_to_idx: Dict[str, int] = {
            label: i + 1 for i, label in enumerate(dim_elems)
        }

        # Pre-scan: detect stock and delay-fixed components so we can choose
        # the right declaration type and pre-allocate internal state arrays.
        has_integ = any(isinstance(c.ast, IntegStructure) for c in elem.components)
        has_delay_fixed = any(
            isinstance(c.ast, DelayFixedStructure) for c in elem.components
        )
        df_name: Optional[str] = None
        if has_delay_fixed:
            df_name = f"_df_{identifier}"
            self.namespace.namespace[f"__internal_df_{identifier}"] = df_name
            self.stock_decls.append(
                f"@variables {df_name}(t)[{self._range_str(dims)}]"
            )

        equations: List[str] = []

        for comp in elem.components:
            # Collect the excluded element labels for this component
            excluded_labels: set = set()
            for except_list in comp.subscripts[1]:
                for label in except_list:
                    excluded_labels.add(label)

            # Determine which indices this component covers.
            # The defining subscript (comp.subscripts[0]) may be:
            #   (a) the full dimension range name  → all elements
            #   (b) a sub-range name               → elements of that sub-range
            #   (c) specific element label(s)      → those elements only
            def_subs = comp.subscripts[0] if comp.subscripts else []
            def_range_name: Optional[str] = None  # non-None only for sub-ranges (b)

            if def_subs and def_subs[0] == dim_name:
                # (a) Full range
                candidate_labels = dim_elems
            elif def_subs and def_subs[0] in self._subs_sizes:
                # (b) A named sub-range — expand to its elements within dim_elems
                def_range_name = def_subs[0]
                range_elems = self._subs_elems.get(def_range_name, [])
                candidate_labels = [e for e in range_elems if e in label_to_idx]
            else:
                # (c) Specific element label(s)
                candidate_labels = [s for s in def_subs if s in label_to_idx]

            covered_indices = [
                label_to_idx[label]
                for label in candidate_labels
                if label not in excluded_labels
            ]

            if comp.type in ("Constant", ) or isinstance(comp, AbstractUnchangeableConstant):
                # Constant component — emit as parameter entries
                visitor = JuliaASTVisitor(
                    self.namespace, self.inline_registry, self.needed_helpers,
                    var_dims=self._var_dims, subs_sizes=self._subs_sizes,
                    subs_elems=self._subs_elems, lookup_names=self._lookup_func_names,
                    root=self.root,
                )
                value_expr = visitor.visit(comp.ast)
                for idx in covered_indices:
                    if not is_control:
                        equations.append(
                            f"# EXCEPT: {identifier}[{idx}] = {value_expr}"
                        )

            elif isinstance(comp.ast, IntegStructure):
                # Stock component — emit per-index ODE + initial condition.
                for idx in covered_indices:
                    vis_idx = JuliaASTVisitor(
                        self.namespace, self.inline_registry, self.needed_helpers,
                        active_subs=self._per_index_subs(
                            dim_name, dim_elems, idx, def_range_name
                        ),
                        var_dims=self._var_dims, subs_sizes=self._subs_sizes,
                        subs_elems=self._subs_elems, lookup_names=self._lookup_func_names,
                        root=self.root,
                    )
                    flow_expr = vis_idx.visit(comp.ast.flow)
                    init_expr = vis_idx.visit(comp.ast.initial)
                    self.u0_entries.append(f"{identifier}[{idx}] => {init_expr}")
                    equations.append(f"D({identifier}[{idx}]) ~ {flow_expr}")

            elif isinstance(comp.ast, DelayFixedStructure):
                # DELAY FIXED — approximate as first-order ODE (same as
                # _expand_delay_fixed) but per-index with separate initials.
                for idx in covered_indices:
                    vis_idx = JuliaASTVisitor(
                        self.namespace, self.inline_registry, self.needed_helpers,
                        active_subs=self._per_index_subs(
                            dim_name, dim_elems, idx, def_range_name
                        ),
                        var_dims=self._var_dims, subs_sizes=self._subs_sizes,
                        subs_elems=self._subs_elems, lookup_names=self._lookup_func_names,
                        root=self.root,
                    )
                    input_expr = vis_idx.visit(comp.ast.input)
                    delay_expr = vis_idx.visit(comp.ast.delay_time)
                    init_expr = vis_idx.visit(comp.ast.initial)
                    self.u0_entries.append(f"{df_name}[{idx}] => {init_expr}")
                    equations.append(
                        f"D({df_name}[{idx}]) ~ "
                        f"({input_expr} - {df_name}[{idx}]) / {delay_expr}"
                    )
                    equations.append(f"{identifier}[{idx}] ~ {df_name}[{idx}]")

            else:
                # Auxiliary component — use a per-index visitor with aligned
                # subscripts so cross-range references resolve correctly.
                for idx in covered_indices:
                    vis_idx = JuliaASTVisitor(
                        self.namespace, self.inline_registry, self.needed_helpers,
                        active_subs=self._per_index_subs(
                            dim_name, dim_elems, idx, def_range_name
                        ),
                        var_dims=self._var_dims, subs_sizes=self._subs_sizes,
                        subs_elems=self._subs_elems, lookup_names=self._lookup_func_names,
                        root=self.root,
                    )
                    rhs_expr = vis_idx.visit(comp.ast)
                    equations.append(f"{identifier}[{idx}] ~ {rhs_expr}")

        if not is_control:
            if has_integ:
                self.stock_decls.append(
                    f"@variables {identifier}(t)[{self._range_str(dims)}]"
                )
            else:
                self.aux_decls.append(
                    f"@variables {identifier}(t)[{self._range_str(dims)}]"
                )
        return equations

    def _process_except_element_2d(
        self,
        elem: "AbstractElement",
        identifier: str,
        dims: List[Tuple[str, int]],
        is_control: bool,
    ) -> List[str]:
        """Handle 2-D EXCEPT subscript exclusion.

        For each component, resolve its subscript specification (which may name a
        full subscript range or a specific element) plus any EXCEPT exclusions to a
        concrete set of 1-based (row, col) index pairs, then emit one comprehension
        equation per component covering exactly those pairs.
        """
        dim0_name, _ = dims[0]
        dim1_name, _ = dims[1]
        dim0_elems = self._subs_elems.get(dim0_name, [])
        dim1_elems = self._subs_elems.get(dim1_name, [])

        def _resolve_spec(spec: str, dim_elems: List[str]) -> List[int]:
            """Return 1-based indices in *dim_elems* for *spec*.

            *spec* is either a subscript-range name (all its elements that appear
            in dim_elems are included) or a bare element name (only that element).
            """
            if spec in self._subs_sizes:
                range_elems = set(self._subs_elems.get(spec, []))
                return [i + 1 for i, e in enumerate(dim_elems) if e in range_elems]
            # Bare element name
            return [i + 1 for i, e in enumerate(dim_elems) if e == spec]

        equations: List[str] = []

        for comp in elem.components:
            sub0_spec = comp.subscripts[0][0] if comp.subscripts[0] else dim0_name
            sub1_spec = comp.subscripts[0][1] if len(comp.subscripts[0]) > 1 else dim1_name

            covered0 = _resolve_spec(sub0_spec, dim0_elems)
            covered1 = _resolve_spec(sub1_spec, dim1_elems)

            # Build set of excluded (i0, i1) pairs from EXCEPT clauses
            excluded: set = set()
            for exc_clause in comp.subscripts[1]:
                exc0_spec = exc_clause[0] if len(exc_clause) > 0 else None
                exc1_spec = exc_clause[1] if len(exc_clause) > 1 else None
                exc0_idx = _resolve_spec(exc0_spec, dim0_elems) if exc0_spec else list(range(1, len(dim0_elems) + 1))
                exc1_idx = _resolve_spec(exc1_spec, dim1_elems) if exc1_spec else list(range(1, len(dim1_elems) + 1))
                for i in exc0_idx:
                    for j in exc1_idx:
                        excluded.add((i, j))

            final0 = [i for i in covered0 if all((i, j) not in excluded for j in covered1)]
            final1 = covered1  # column coverage doesn't change

            # Check if all remaining rows still cover the full column range
            # (so we can use a range expression rather than an explicit list)
            full_col_range = list(range(1, len(dim1_elems) + 1))
            use_full_cols = final1 == full_col_range

            if not final0 or not final1:
                continue

            vnd = self._nd_visitor(dims, ["_i0", "_i1"])
            rhs_expr = vnd.visit(comp.ast)

            row_str = (
                f"1:{self._jl_n(dim0_name)}"
                if final0 == list(range(1, len(dim0_elems) + 1))
                else "[" + ", ".join(str(i) for i in final0) + "]"
            )
            col_str = (
                f"1:{self._jl_n(dim1_name)}"
                if use_full_cols
                else "[" + ", ".join(str(j) for j in final1) + "]"
            )

            equations.append(
                f"[{identifier}[_i0, _i1] ~ {rhs_expr} "
                f"for _i0 in {row_str}, _i1 in {col_str}]..."
            )

        if not is_control:
            self.aux_decls.append(
                f"@variables {identifier}(t)[{self._range_str(dims)}]"
            )
        return equations

    # ------------------------------------------------------------------
    # Smooth expansion
    # ------------------------------------------------------------------

    def _expand_smooth(
        self,
        identifier: str,
        ast,
        visitor: JuliaASTVisitor,
        order: int,
        dims: Optional[List[Tuple[str, int]]] = None,
    ) -> List[str]:
        """Expand a SMOOTH(N) into *order* chained first-order ODE levels.

        The output variable ``identifier`` is declared as an auxiliary equal
        to the final level.  When *dims* is provided the internal levels are
        subscripted arrays and the equations are emitted as comprehensions.
        """
        dims = dims or []
        eqs: List[str] = []

        if dims:
            # Subscripted SMOOTH — each internal level is an array.
            (d0, n0) = dims[0]
            vnd = self._nd_visitor(dims, ["_i0"])
            input_nd = vnd.visit(ast.input)
            st_nd = vnd.visit(ast.smooth_time)
            init_nd = vnd.visit(ast.initial)

            prev_nd = input_nd
            for i in range(1, order + 1):
                lv_name = f"_lv{i}_{identifier}"
                self.namespace.namespace[f"__internal_lv{i}_{identifier}"] = lv_name
                self.stock_decls.append(
                    f"@variables {lv_name}(t)[{self._range_str(dims)}]"
                )
                for idx in range(1, n0 + 1):
                    init_i = init_nd.replace("_i0", str(idx))
                    self.u0_entries.append(f"{lv_name}[{idx}] => {init_i}")
                lv_ref = f"{lv_name}[_i0]"
                eqs.append(
                    f"[D({lv_ref}) ~ ({prev_nd} - {lv_ref}) / "
                    f"({st_nd} / {order}) for _i0 in 1:{self._jl_n(d0)}]..."
                )
                prev_nd = lv_ref
            self.aux_decls.append(
                f"@variables {identifier}(t)[{self._range_str(dims)}]"
            )
            eqs.append(
                f"[{identifier}[_i0] ~ {prev_nd} for _i0 in 1:{self._jl_n(d0)}]..."
            )
        else:
            input_expr = visitor.visit(ast.input)
            smooth_time_expr = visitor.visit(ast.smooth_time)
            initial_expr = visitor.visit(ast.initial)
            prev_expr = input_expr
            for i in range(1, order + 1):
                lv_name = f"_lv{i}_{identifier}"
                self.namespace.namespace[f"__internal_lv{i}_{identifier}"] = lv_name
                self.stock_decls.append(f"@variables {lv_name}(t)")
                self.u0_entries.append(f"{lv_name} => {initial_expr}")
                eqs.append(
                    f"D({lv_name}) ~ ({prev_expr} - {lv_name}) / "
                    f"({smooth_time_expr} / {order})"
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
        dims: Optional[List[Tuple[str, int]]] = None,
    ) -> List[str]:
        """Expand a DELAY(N) into *order* chained first-order pipeline levels.

        Each level ``L_i`` satisfies::

            dL_i/dt = (inflow_i - L_i * rate)
            rate    = order / delay_time
            inflow_1 = input;  inflow_i = L_{i-1} * rate  for i > 1
        """
        dims = dims or []
        eqs: List[str] = []

        if dims:
            ndim = len(dims)
            idx_vars = self._idx_vars(ndim)
            vnd = self._nd_visitor(dims, idx_vars)
            input_nd = vnd.visit(ast.input)
            delay_time_nd = vnd.visit(ast.delay_time)
            initial_nd = vnd.visit(ast.initial)
            idx_str_t = ", ".join(idx_vars)
            for_clause = self._for_clause(dims, idx_vars)
            ranges_list = [range(1, size + 1) for _, size in dims]

            rate_nd = f"({order} / ({delay_time_nd}))"
            prev_nd = input_nd
            for stage in range(1, order + 1):
                lv_name = f"_dl{stage}_{identifier}"
                self.namespace.namespace[f"__internal_dl{stage}_{identifier}"] = lv_name
                self.stock_decls.append(
                    f"@variables {lv_name}(t)[{self._range_str(dims)}]"
                )
                for idx_combo in itertools.product(*ranges_list):
                    expr_i = initial_nd
                    dt_i = delay_time_nd
                    for iv, idx in zip(idx_vars, idx_combo):
                        expr_i = expr_i.replace(iv, str(idx))
                        dt_i = dt_i.replace(iv, str(idx))
                    idx_s = ", ".join(str(v) for v in idx_combo)
                    self.u0_entries.append(
                        f"{lv_name}[{idx_s}] => {expr_i} * ({dt_i}) / {order}"
                    )
                lv_ref = f"{lv_name}[{idx_str_t}]"
                eqs.append(
                    f"[D({lv_ref}) ~ ({prev_nd} - {lv_ref} * {rate_nd}) "
                    f"for {for_clause}]..."
                )
                prev_nd = f"{lv_ref} .* {rate_nd}"
            self.aux_decls.append(
                f"@variables {identifier}(t)[{self._range_str(dims)}]"
            )
            eqs.append(f"[{identifier}[{idx_str_t}] ~ {prev_nd} for {for_clause}]...")
            return eqs

        input_expr = visitor.visit(ast.input)
        delay_time_expr = visitor.visit(ast.delay_time)
        initial_expr = visitor.visit(ast.initial)
        rate_expr = f"({order} / {delay_time_expr})"
        prev_outflow = input_expr
        for i in range(1, order + 1):
            lv_name = f"_dl{i}_{identifier}"
            self.namespace.namespace[f"__internal_dl{i}_{identifier}"] = lv_name
            self.stock_decls.append(f"@variables {lv_name}(t)")
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
        dims: Optional[List[Tuple[str, int]]] = None,
    ) -> List[str]:
        """Approximate DELAY FIXED as a first-order ODE delay.

        The true DELAY FIXED is a pure transport delay (DDE), which
        ModelingToolkit/OrdinaryDiffEq cannot solve.  We approximate it
        with a first-order exponential delay (DELAY1):

            D(output) ~ (input - output) / delay_time

        with initial condition ``output(0) = initial``.
        """
        dims = dims or []
        lv_name = f"_df_{identifier}"
        self.namespace.namespace[f"__internal_df_{identifier}"] = lv_name

        if dims:
            ndim = len(dims)
            idx_vars = self._idx_vars(ndim)
            vnd = self._nd_visitor(dims, idx_vars)
            input_nd = vnd.visit(ast.input)
            delay_time_nd = vnd.visit(ast.delay_time)
            initial_nd = vnd.visit(ast.initial)
            self.stock_decls.append(
                f"@variables {lv_name}(t)[{self._range_str(dims)}]"
            )
            ranges_list = [range(1, size + 1) for _, size in dims]
            for idx_combo in itertools.product(*ranges_list):
                expr_i = initial_nd
                for iv, idx in zip(idx_vars, idx_combo):
                    expr_i = expr_i.replace(iv, str(idx))
                idx_str = ", ".join(str(v) for v in idx_combo)
                self.u0_entries.append(f"{lv_name}[{idx_str}] => {expr_i}")
            self.aux_decls.append(
                f"@variables {identifier}(t)[{self._range_str(dims)}]"
            )
            idx_str_t = ", ".join(idx_vars)
            for_clause = self._for_clause(dims, idx_vars)
            return [
                f"[D({lv_name}[{idx_str_t}]) ~ ({input_nd} - {lv_name}[{idx_str_t}]) / ({delay_time_nd}) "
                f"for {for_clause}]...",
                f"[{identifier}[{idx_str_t}] ~ {lv_name}[{idx_str_t}] for {for_clause}]...",
            ]

        input_expr = visitor.visit(ast.input)
        delay_time_expr = visitor.visit(ast.delay_time)
        initial_expr = visitor.visit(ast.initial)
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
        dims: Optional[List[Tuple[str, int]]] = None,
        ndim: int = 0,
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
        dims = dims or []
        st_name = f"_sit_{identifier}"
        self.namespace.namespace[f"__internal_sit_{identifier}"] = st_name

        ts_val = self.control_vals.get("time_step")
        ts_expr = ts_val if ts_val is not None else "time_step"

        if dims:
            idx_vars = self._idx_vars(len(dims))
            vnd = self._nd_visitor(dims, idx_vars)
            condition_nd = vnd.visit(ast.condition)
            input_nd = vnd.visit(ast.input)
            initial_nd = vnd.visit(ast.initial)
            idx_str_t = ", ".join(idx_vars)
            for_clause = self._for_clause(dims, idx_vars)
            ranges_list = [range(1, size + 1) for _, size in dims]

            self.stock_decls.append(
                f"@variables {st_name}(t)[{self._range_str(dims)}]"
            )
            for idx_combo in itertools.product(*ranges_list):
                expr_i = initial_nd
                for iv, idx in zip(idx_vars, idx_combo):
                    expr_i = expr_i.replace(iv, str(idx))
                idx_s = ", ".join(str(v) for v in idx_combo)
                self.u0_entries.append(f"{st_name}[{idx_s}] => {expr_i}")
            self.aux_decls.append(
                f"@variables {identifier}(t)[{self._range_str(dims)}]"
            )
            return [
                f"[D({st_name}[{idx_str_t}]) ~ ifelse({condition_nd} > 0.5, "
                f"({input_nd} - {st_name}[{idx_str_t}]) / ({ts_expr}), 0.0) "
                f"for {for_clause}]...",
                f"[{identifier}[{idx_str_t}] ~ {st_name}[{idx_str_t}] for {for_clause}]...",
            ]

        condition_expr = visitor.visit(ast.condition)
        input_expr = visitor.visit(ast.input)
        initial_expr = visitor.visit(ast.initial)
        self.stock_decls.append(f"@variables {st_name}(t)")
        self.u0_entries.append(f"{st_name} => {initial_expr}")
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
        Supports scalar (1D), 1-subscript (2D), and 2-subscript (3D) lookup
        arrays.  For multi-component elements where each component covers one
        element of a subscript range, split-range detection is used to resolve
        parent-range ambiguity before delegating to ExtLookup.
        """
        try:
            from pysd.py_backend.external import ExtLookup

            comp0 = elem.components[0]
            ast0 = comp0.ast

            if len(elem.components) > 1:
                # Detect which subscript positions vary across components and
                # find the unique containing range for each such position.
                split_ranges = self._detect_split_ranges(elem.components)
                coords0 = self._comp_coords_split(comp0, split_ranges)
                final_coords: Dict[str, list] = {}
                for comp in elem.components:
                    for range_key, elem_val in self._comp_coords_split(comp, split_ranges).items():
                        if range_key not in final_coords:
                            final_coords[range_key] = self._subs_elems.get(range_key, elem_val)
            else:
                split_ranges = {}
                coords0 = self._comp_coords(comp0)
                final_coords = {k: self._subs_elems.get(k, v) for k, v in coords0.items()}

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
                comp_coords = self._comp_coords_split(comp, split_ranges)
                ext.add(ast_i.file, ast_i.tab, ast_i.x_row_or_col, ast_i.cell, comp_coords)

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
                if self.data_format == "json":
                    self._json_data["lookups"][identifier] = {
                        "x": list(xs), "y": list(ys),
                        "interp_type": "interpolate", "subscripts": [],
                    }
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
                if self.data_format == "json":
                    for k in range(n_subs):
                        sub_name = f"{identifier}_{k + 1}"
                        col_ys = tuple(float(y) for y in arr[:, k])
                        self._json_data["lookups"][sub_name] = {
                            "x": list(xs), "y": list(col_ys),
                            "interp_type": "interpolate", "subscripts": [],
                        }
                return []
            elif arr.ndim == 3:
                # 3D: shape (n_points, n_dim1, n_dim2).
                # Emit one lookup per (i, j) pair and a 2-index dispatch.
                n_dim1, n_dim2 = arr.shape[1], arr.shape[2]
                rows: List[List[str]] = []
                for i in range(n_dim1):
                    row: List[str] = []
                    for j in range(n_dim2):
                        col_ys = tuple(float(y) for y in arr[:, i, j])
                        sub_name = f"{identifier}_{i + 1}_{j + 1}"
                        const_decl, func_decl, reg_decl = lookup_interpolation_code(
                            sub_name, xs, col_ys, "interpolate"
                        )
                        self.lookup_const_decls.append(const_decl)
                        self.lookup_func_decls.append(func_decl)
                        self.lookup_register_decls.append(reg_decl)
                        if self.data_format == "json":
                            self._json_data["lookups"][sub_name] = {
                                "x": list(xs), "y": list(col_ys),
                                "interp_type": "interpolate", "subscripts": [],
                            }
                        row.append(sub_name)
                    rows.append(row)
                inner = ", ".join("[" + ", ".join(r) + "]" for r in rows)
                self.lookup_const_decls.append(
                    f"const {identifier}_fns = [{inner}]"
                )
                self.lookup_func_decls.append(
                    f"{identifier}(i, j, x) = {identifier}_fns[i][j](x)"
                )
                self.lookup_register_decls.append(
                    f"@register_symbolic {identifier}(i::Integer, j::Integer, x::Real)"
                )
                return []
            else:
                warn(
                    f"Subscripted GET LOOKUPS '{elem.name}' has {arr.ndim - 1} "
                    "subscript dimensions (> 2D) — only up to 2D subscripted lookups "
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

            # Collect only components that carry a GetDataStructure
            data_comps = [c for c in elem.components if isinstance(c.ast, GetDataStructure)]
            if not data_comps:
                raise ValueError("No GetDataStructure component found")

            comp0 = data_comps[0]
            ast0 = comp0.ast

            if len(data_comps) > 1:
                split_ranges = self._detect_split_ranges(data_comps)
                coords0 = self._comp_coords_split(comp0, split_ranges)
                final_coords: Dict[str, list] = {}
                for c in data_comps:
                    for range_key, elem_val in self._comp_coords_split(c, split_ranges).items():
                        if range_key not in final_coords:
                            final_coords[range_key] = self._subs_elems.get(range_key, elem_val)
            else:
                split_ranges = {}
                coords0 = self._comp_coords(comp0)
                final_coords = {k: self._subs_elems.get(k, v) for k, v in coords0.items()}

            # Determine interpolation type from AbstractData keyword
            julia_itp = _vensim_keyword_to_itp_type(
                getattr(comp, "keyword", None)
            )

            ext = ExtData(
                file_name=ast0.file,
                tab=ast0.tab,
                time_row_or_col=ast0.time_row_or_col,
                cell=ast0.cell,
                interp="interpolate",  # always interpolate when reading at translate time
                coords=coords0,
                root=self.root,
                final_coords=final_coords,
                py_name=identifier,
            )

            for c in data_comps[1:]:
                ai = c.ast
                comp_coords = self._comp_coords_split(c, split_ranges)
                ext.add(ai.file, ai.tab, ai.time_row_or_col, ai.cell, "interpolate", comp_coords)

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
                    identifier, xs, ys, julia_itp
                )
                self.lookup_const_decls.append(const_decl)
                self.lookup_func_decls.append(func_decl)
                self.lookup_register_decls.append(reg_decl)
                if self.data_format == "json":
                    self._json_data["data"][identifier] = {
                        "time": list(xs), "values": list(ys),
                        "interp_type": julia_itp, "subscripts": [],
                    }
                return []
            elif arr.ndim == 2:
                # Subscripted time-series: shape (n_time, n_subs)
                n_subs = arr.shape[1]
                sub_func_names = []
                for k in range(n_subs):
                    col_ys = tuple(float(y) for y in arr[:, k])
                    sub_name = f"{identifier}_{k + 1}"
                    const_decl, func_decl, reg_decl = lookup_interpolation_code(
                        sub_name, xs, col_ys, julia_itp
                    )
                    self.lookup_const_decls.append(const_decl)
                    self.lookup_func_decls.append(func_decl)
                    self.lookup_register_decls.append(reg_decl)
                    if self.data_format == "json":
                        self._json_data["data"][sub_name] = {
                            "time": list(xs), "values": list(col_ys),
                            "interp_type": julia_itp, "subscripts": [],
                        }
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
            elif arr.ndim == 3:
                # 3D time-series: shape (n_time, n_dim1, n_dim2)
                n_dim1, n_dim2 = arr.shape[1], arr.shape[2]
                rows: List[List[str]] = []
                for i in range(n_dim1):
                    row: List[str] = []
                    for j in range(n_dim2):
                        col_ys = tuple(float(y) for y in arr[:, i, j])
                        sub_name = f"{identifier}_{i + 1}_{j + 1}"
                        const_decl, func_decl, reg_decl = lookup_interpolation_code(
                            sub_name, xs, col_ys, julia_itp
                        )
                        self.lookup_const_decls.append(const_decl)
                        self.lookup_func_decls.append(func_decl)
                        self.lookup_register_decls.append(reg_decl)
                        if self.data_format == "json":
                            self._json_data["data"][sub_name] = {
                                "time": list(xs), "values": list(col_ys),
                                "interp_type": julia_itp, "subscripts": [],
                            }
                        row.append(sub_name)
                    rows.append(row)
                inner = ", ".join("[" + ", ".join(r) + "]" for r in rows)
                self.lookup_const_decls.append(
                    f"const {identifier}_fns = [{inner}]"
                )
                self.lookup_func_decls.append(
                    f"{identifier}(i, j, x) = {identifier}_fns[i][j](x)"
                )
                self.lookup_register_decls.append(
                    f"@register_symbolic {identifier}(i::Integer, j::Integer, x::Real)"
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

    def _expand_initial_frozen_stock(
        self,
        identifier: str,
        inner_ast,
        dims: List[Tuple[str, int]],
        ndim: int,
    ) -> List[str]:
        """Emit ``INITIAL(expr)`` as a zero-derivative stock.

        MTK evaluates the initial-condition expression at t=t0, which gives
        the correct Vensim semantics (value frozen at the initial time).

        Scalar::

            @variables x(t)
            D(x) ~ 0.0
            u0: x => expr

        1D subscripted::

            @variables x(t)[1:N]
            Symbolics.scalarize(D.(x) .~ 0.0)...
            u0: x[i] => expr_at_i   (for i in 1..N)

        2D subscripted::

            @variables x(t)[1:N0, 1:N1]
            [D(x[_i0, _i1]) ~ 0.0 for _i0 in 1:N0, _i1 in 1:N1]...
            u0: x[i, j] => expr_at_ij
        """
        if ndim == 0:
            v = JuliaASTVisitor(
                self.namespace, self.inline_registry, self.needed_helpers,
                subs_sizes=self._subs_sizes, root=self.root,
            )
            init_expr = v.visit(inner_ast)
            self.stock_decls.append(f"@variables {identifier}(t)")
            self.u0_entries.append(f"{identifier} => {init_expr}")
            return [f"D({identifier}) ~ 0.0"]

        if ndim == 1:
            (d0, n0) = dims[0]
            idx_vars = ["_i0"]
            vnd = self._nd_visitor(dims, idx_vars)
            raw_expr = vnd.visit(inner_ast)
            self.stock_decls.append(
                f"@variables {identifier}(t)[{self._range_str(dims)}]"
            )
            for i in range(1, n0 + 1):
                expr_i = raw_expr.replace("_i0", str(i))
                self.u0_entries.append(f"{identifier}[{i}] => {expr_i}")
            return [f"Symbolics.scalarize(D.({identifier}) .~ 0.0)..."]

        # ndim >= 2
        idx_vars = self._idx_vars(ndim)
        vnd = self._nd_visitor(dims, idx_vars)
        raw_expr = vnd.visit(inner_ast)
        ranges_list = [range(1, size + 1) for _, size in dims]
        self.stock_decls.append(
            f"@variables {identifier}(t)[{self._range_str(dims)}]"
        )
        for idx_combo in itertools.product(*ranges_list):
            expr_ij = raw_expr
            for iv, idx in zip(idx_vars, idx_combo):
                expr_ij = expr_ij.replace(iv, str(idx))
            idx_str = ", ".join(str(i) for i in idx_combo)
            self.u0_entries.append(f"{identifier}[{idx_str}] => {expr_ij}")
        for_clause = self._for_clause(dims, idx_vars)
        idx_str_template = ", ".join(idx_vars)
        return [
            f"[D({identifier}[{idx_str_template}]) ~ 0.0 "
            f"for {for_clause}]..."
        ]

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

        Handles three layouts:

        * All-GCS: one ExtConstant handles all components via .add().
        * Mixed GCS + numeric literal: piecewise assembly — each component is
          read/valued independently and the results are combined into one array
          ordered by the parent subscript range.
        * Single scalar: trivial ExtConstant read.

        Returns a Julia literal string (scalar or array) on success, or None
        if the file cannot be read, in which case the caller falls through to
        the unsupported-structure handler.
        """
        import numpy as np
        try:
            from pysd.py_backend.external import ExtConstant

            gcs_comps = [c for c in elem.components if isinstance(c.ast, GetConstantsStructure)]
            lit_comps = [c for c in elem.components if not isinstance(c.ast, GetConstantsStructure)]

            # ----- Piecewise: mix of GCS + numeric literals -----
            if gcs_comps and lit_comps:
                return self._read_get_constants_piecewise(
                    elem, identifier, gcs_comps, lit_comps
                )

            # ----- All GCS (the common case) -----
            comp0 = elem.components[0]
            ast0 = comp0.ast

            if len(elem.components) > 1:
                split_ranges = self._detect_split_ranges(elem.components)
                coords0 = self._comp_coords_split(comp0, split_ranges)
                final_coords: Dict[str, list] = {}
                for comp in elem.components:
                    for range_key, elem_val in self._comp_coords_split(comp, split_ranges).items():
                        if range_key not in final_coords:
                            final_coords[range_key] = self._subs_elems.get(range_key, elem_val)
            else:
                split_ranges = {}
                coords0 = self._comp_coords(comp0)
                final_coords = {k: self._subs_elems.get(k, v) for k, v in coords0.items()}

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
                comp_coords = self._comp_coords_split(comp, split_ranges)
                ext.add(ast_i.file, ast_i.tab, ast_i.cell, comp_coords)

            ext.initialize()
            return _format_julia_value(ext.data)

        except Exception as exc:
            warn(
                f"Could not read external constant for '{elem.name}': {exc} "
                "— emitting placeholder."
            )
            return None

    def _read_get_constants_piecewise(
        self,
        elem: "AbstractElement",
        identifier: str,
        gcs_comps: List["AbstractComponent"],
        lit_comps: List["AbstractComponent"],
    ) -> Optional[str]:
        """Build a constant array from a mix of GCS and numeric-literal components.

        Vensim allows piecewise definitions such as::

            var[fuel1, fuel2, fuel3] = GET DIRECT CONSTANTS(...)
            var[electricity] = 0
            var[heat] = 0

        Here we read each GCS component with its own subscript coords, collect
        the literal values, and assemble the full array in the order given by
        the parent subscript range.
        """
        import numpy as np
        from pysd.py_backend.external import ExtConstant
        from pysd.builders.julia.julia_expressions_builder import format_number

        all_comps = elem.components
        split_ranges = self._detect_split_ranges(all_comps)

        # Build a map: element_label → float value
        elem_values: Dict[str, float] = {}

        for comp in lit_comps:
            val = float(comp.ast) if isinstance(comp.ast, (int, float)) else 0.0
            # Each literal component covers exactly the elements in its subscripts
            subs = comp.subscripts[0] if comp.subscripts else []
            for s in subs:
                if s in self._subs_elems:
                    for e in self._subs_elems[s]:
                        elem_values[e] = val
                else:
                    elem_values[s] = val

        for comp in gcs_comps:
            ast = comp.ast
            coords = self._comp_coords_split(comp, split_ranges)
            final_c = {k: self._subs_elems.get(k, v) for k, v in coords.items()}
            ext = ExtConstant(
                file_name=ast.file, tab=ast.tab, cell=ast.cell,
                coords=coords, root=self.root, final_coords=final_c,
                py_name=identifier,
            )
            ext.initialize()
            data = ext.data
            arr = data.values if hasattr(data, "values") else np.asarray(data)
            arr = np.asarray(arr, dtype=float)
            # Map each axis label to its value
            if arr.ndim == 0:
                subs = comp.subscripts[0] if comp.subscripts else []
                if subs:
                    elem_values[subs[0]] = float(arr)
            else:
                for dim_name, coord_vals in data.coords.items():
                    labels = [str(v) for v in coord_vals.values]
                    # For each label, slice the array along this dim
                    for idx, label in enumerate(labels):
                        sliced = arr.take(idx, axis=list(data.dims).index(dim_name))
                        if sliced.ndim == 0:
                            elem_values[label] = float(sliced)
                        # Multi-element slices need further handling; skip for now

        # Find the parent range that covers all collected element labels
        all_elems = list(elem_values.keys())
        parent_range = self._infer_parent_range(all_elems)
        if parent_range is None:
            # Can't determine order; just return values in encountered order
            vals = list(elem_values.values())
        else:
            ordered_elems = self._subs_elems.get(parent_range, all_elems)
            vals = [elem_values.get(e, 0.0) for e in ordered_elems]

        if len(vals) == 1:
            return format_number(vals[0])
        return "[" + ", ".join(format_number(v) for v in vals) + "]"

    # ------------------------------------------------------------------
    # JSON helpers
    # ------------------------------------------------------------------

    def _json_accumulate_constant(
        self, elem: "AbstractElement", identifier: str, julia_val: str
    ) -> None:
        """Store an external constant's value in ``_json_data["constants"]``."""
        import numpy as np
        try:
            from pysd.py_backend.external import ExtConstant
            comp0 = elem.components[0]
            coords0 = self._comp_coords(comp0)
            ext = ExtConstant(
                file_name=comp0.ast.file,
                tab=comp0.ast.tab,
                cell=comp0.ast.cell,
                coords=coords0,
                root=self.root,
                final_coords={k: self._subs_elems.get(k, v) for k, v in coords0.items()},
                py_name=identifier,
            )
            ext.initialize()
            raw = ext.data
            if hasattr(raw, "values"):
                raw = raw.values
            arr = np.asarray(raw, dtype=float)
            if arr.ndim == 0:
                values: object = float(arr)
                dims: list = []
            else:
                values = arr.tolist()
                dims = [f"dim{i}" for i in range(arr.ndim)]
            self._json_data["constants"][identifier] = {
                "dims": dims,
                "coords": {},
                "values": values,
                "units": elem.units or "",
            }
        except Exception:
            # Best-effort; fall back to the Julia literal string
            self._json_data["constants"][identifier] = {
                "dims": [], "coords": {},
                "values": julia_val,
                "units": elem.units or "",
            }

    # ------------------------------------------------------------------
    # JSON data file
    # ------------------------------------------------------------------

    def _write_data_json(self) -> Path:
        """Write accumulated external data to ``<model>_data.json``.

        Returns the path of the written file.

        Schema::

            {
              "constants": {
                "<jl_id>": {
                  "dims": [...],
                  "coords": {...},
                  "values": <scalar|list>,
                  "units": ""
                }
              },
              "lookups": {
                "<jl_id>": {
                  "x": [...],
                  "y": [...],
                  "interp_type": "interpolate",
                  "subscripts": []
                }
              },
              "data": {
                "<jl_id>": {
                  "time": [...],
                  "values": [...],
                  "interp_type": "interpolate",
                  "subscripts": []
                }
              }
            }
        """
        import json

        # Use self.path.stem (not self.model_name) so macro sections write
        # their data file next to their own .jl file.
        json_path = self.path.with_name(f"{self.path.stem}_data.json")
        json_path.write_text(
            json.dumps(self._json_data, indent=2), encoding="UTF-8"
        )
        return json_path

    # ------------------------------------------------------------------
    # Single-file build
    # ------------------------------------------------------------------

    def _build(self) -> None:
        """Write the whole model as one ``.jl`` file."""
        all_eqs: List[str] = []
        for eqs, _is_ctrl in self.built_elements.values():
            all_eqs.extend(eqs)
        if self.data_format == "json":
            self._write_data_json()
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

        if self.data_format == "json":
            self._write_data_json()
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
        has_lookups = bool(self.lookup_const_decls)
        if has_lookups or extra_packages:
            uses.append("DataInterpolations")
        if self.data_format == "json":
            uses.append("JSON3")
        uses.append("NCDatasets")
        header = (
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
        if self.data_format == "json":
            json_fname = f"{self.path.stem}_data.json"
            header += (
                f'const _model_data = JSON3.read(read(joinpath(@__DIR__, "{json_fname}"), String))\n\n'
            )
        return header

    def _helpers_block(self) -> str:
        if not self.needed_helpers:
            return ""
        lines = ["# Helper functions"]
        for name in sorted(self.needed_helpers):
            if name in HELPER_IMPLEMENTATIONS:
                lines.append(HELPER_IMPLEMENTATIONS[name])
        return "\n".join(lines) + "\n\n"

    def _lookup_block(self) -> str:
        if not self.lookup_const_decls and not self._json_data.get("lookups") \
                and not self._json_data.get("data"):
            return ""
        lines = ["# Lookup tables"]
        if self.data_format == "json":
            # JSON mode: build LinearInterpolation from _model_data at startup
            for key in list(self._json_data.get("lookups", {})):
                itp_name = f"{key}_itp"
                lines.append(
                    f'const {itp_name} = LinearInterpolation('
                    f'Float64.(_model_data["lookups"]["{key}"]["y"]), '
                    f'Float64.(_model_data["lookups"]["{key}"]["x"]))'
                )
                lines.append(f"{key}(x) = {itp_name}(x)")
                lines.append(f"@register_symbolic {key}(x::Real)")
            for key in list(self._json_data.get("data", {})):
                itp_name = f"{key}_itp"
                lines.append(
                    f'const {itp_name} = LinearInterpolation('
                    f'Float64.(_model_data["data"]["{key}"]["values"]), '
                    f'Float64.(_model_data["data"]["{key}"]["time"]))'
                )
                lines.append(f"{key}(x) = {itp_name}(x)")
                lines.append(f"@register_symbolic {key}(x::Real)")
        else:
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
            if self.data_format == "json":
                # JSON mode: replace hardcoded defaults with _model_data reads.
                # All param_decls entries match "@parameters <name> = <val>" by
                # construction, so no else branch is needed.
                for decl in self.param_decls:
                    name_part = decl.split(" = ", 1)[0][len("@parameters "):]
                    base_name = name_part.split("[")[0]
                    lines.append(
                        f'@parameters {name_part} = '
                        f'_model_data["constants"]["{base_name}"]["values"]'
                    )
            else:
                lines.extend(self.param_decls)
        if self.ext_const_decls:
            lines.append("\n# External constants")
            if self.data_format == "json":
                # All ext_const_decls entries match "const <name> = <val>" by
                # construction, so no else branch is needed.
                for decl in self.ext_const_decls:
                    name = decl.split(" = ", 1)[0][len("const "):]
                    lines.append(
                        f'const {name} = '
                        f'_model_data["constants"]["{name}"]["values"]'
                    )
            else:
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

    def _save_results_function(self) -> str:
        """Generate a save_results(sol, path) function that writes model output to NetCDF4."""
        decl_pat = re.compile(r"@variables\s+(\w+)\(t\)(?:\[([^\]]+)\])?")

        # Build reverse map: N_CONST_STR → (nc_dim_name, [element_labels])
        n_const_to_dim: Dict[str, Tuple[str, List[str]]] = {}
        for dim_name, size in self._subs_sizes.items():
            if size <= 0:
                continue
            nc = self._jl_n(dim_name)
            labels = self._subs_elems.get(dim_name, [str(i + 1) for i in range(size)])
            nc_dim = re.sub(r"[^a-z0-9]+", "_", dim_name.lower()).strip("_")
            n_const_to_dim[nc] = (nc_dim, labels)

        # Parse @variables declarations → [(var_name, [N_CONST, ...])]
        var_list: List[Tuple[str, List[str]]] = []
        seen: set = set()
        for decl in self.stock_decls + self.aux_decls:
            m = decl_pat.search(decl)
            if not m:
                continue
            vname = m.group(1)
            if vname in seen or vname.startswith("_"):
                continue
            seen.add(vname)
            dims_str = m.group(2)
            if dims_str:
                n_consts = [
                    part.strip().split(":")[-1].strip()
                    for part in dims_str.split(",")
                ]
            else:
                n_consts = []
            var_list.append((vname, n_consts))

        if not var_list:
            return ""

        # Collect used N_CONST names in order of first appearance
        used_n_consts: List[str] = []
        for _, n_consts in var_list:
            for nc in n_consts:
                if nc not in used_n_consts:
                    used_n_consts.append(nc)

        lines: List[str] = []
        lines.append("function save_results(sol, path::String)")
        lines.append("    ds = NCDataset(path, \"c\")")
        lines.append("    defDim(ds, \"time\", length(sol.t))")
        lines.append("    let v = defVar(ds, \"time\", Float64, (\"time\",)); v[:] = sol.t; end")

        # Subscript dimension declarations + label coordinates
        for nc in used_n_consts:
            if nc in n_const_to_dim:
                nc_dim, labels = n_const_to_dim[nc]
                labels_jl = ", ".join(f'"{lbl}"' for lbl in labels)
                lines.append(f"    defDim(ds, \"{nc_dim}\", {nc})")
                lines.append(
                    f"    let v = defVar(ds, \"{nc_dim}_labels\", String, (\"{nc_dim}\",));"
                    f" v[:] = [{labels_jl}]; end"
                )
            else:
                nc_dim = re.sub(r"[^a-z0-9]+", "_", nc.lower()).strip("_")
                nc_dim = nc_dim[2:] if nc_dim.startswith("n_") else nc_dim
                lines.append(f"    defDim(ds, \"{nc_dim}\", {nc})")

        lines.append("")
        lines.append("    # --- model variables ---")

        for vname, n_consts in var_list:
            if not n_consts:
                lines.append(
                    f"    try; let v = defVar(ds, \"{vname}\", Float64, (\"time\",));"
                    f" v[:] = sol[sys.{vname}, :]; end; catch; end"
                )
            elif len(n_consts) == 1:
                nc = n_consts[0]
                nc_dim = n_const_to_dim[nc][0] if nc in n_const_to_dim else (
                    nc[2:].lower() if nc.upper().startswith("N_") else nc.lower()
                )
                lines.append(f"    try")
                lines.append(
                    f"        let v = defVar(ds, \"{vname}\", Float64, (\"{nc_dim}\", \"time\"))"
                )
                lines.append(f"            for _i in 1:{nc}")
                lines.append(f"                v[_i, :] = sol[sys.{vname}[_i], :]")
                lines.append(f"            end")
                lines.append(f"        end")
                lines.append(f"    catch; end")
            elif len(n_consts) == 2:
                nc1, nc2 = n_consts
                d1 = n_const_to_dim[nc1][0] if nc1 in n_const_to_dim else nc1.lower()
                d2 = n_const_to_dim[nc2][0] if nc2 in n_const_to_dim else nc2.lower()
                lines.append(f"    try")
                lines.append(
                    f"        let v = defVar(ds, \"{vname}\", Float64, (\"{d1}\", \"{d2}\", \"time\"))"
                )
                lines.append(f"            for _i in 1:{nc1}, _j in 1:{nc2}")
                lines.append(f"                v[_i, _j, :] = sol[sys.{vname}[_i, _j], :]")
                lines.append(f"            end")
                lines.append(f"        end")
                lines.append(f"    catch; end")
            else:
                # ≥3 dimensions
                dim_names_jl = ", ".join(
                    f'"{n_const_to_dim[nc][0] if nc in n_const_to_dim else nc.lower()}"'
                    for nc in n_consts
                )
                size_tuple = "(" + ", ".join(nc for nc in n_consts) + ",)"
                idx_parts = ", ".join(f"_idx[{i + 1}]" for i in range(len(n_consts)))
                lines.append(f"    try")
                lines.append(
                    f"        let v = defVar(ds, \"{vname}\", Float64, ({dim_names_jl}, \"time\"))"
                )
                lines.append(f"            for _idx in CartesianIndices{size_tuple}")
                lines.append(f"                v[Tuple(_idx)..., :] = sol[sys.{vname}[{idx_parts}], :]")
                lines.append(f"            end")
                lines.append(f"        end")
                lines.append(f"    catch; end")

        lines.append("")
        lines.append("    close(ds)")
        lines.append("end")
        lines.append("")
        return "\n".join(lines) + "\n"

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
            self._save_results_function(),
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
            # Control variables (time_step, initial_time, …) must be defined
            # before the module includes so equations can reference them.
            self._control_block(),
            "\n",
            include_block,
            leftover_block,
            "\n",
            combined,
            "\n",
            self._u0_block(),
            "\n",
            self._system_block(),
            "\n",
            self._run_function(),
            self._save_results_function(),
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


def _vensim_keyword_to_itp_type(keyword: Optional[str]) -> str:
    """Map a Vensim DATA keyword to the ``itp_type`` used by
    :func:`lookup_interpolation_code`.

    Vensim keywords and their meanings:

    * ``None`` / ``"interpolate"`` — linear interpolation (default)
    * ``"hold_backward"``          — step function, hold previous value
      → ``ConstantInterpolation``
    * ``"look_forward"``           — step function, hold next value
      → ``ConstantInterpolation(dir=:right)``
    * ``"raw"``                    — no interpolation; approximated as linear
    """
    if keyword == "hold_backward":
        return "hold_forward"   # ConstantInterpolation (left/previous)
    if keyword == "look_forward":
        return "hold_backward"  # ConstantInterpolation(dir=:right) (right/next)
    return "interpolate"
