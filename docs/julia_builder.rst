Julia Builder
=============

PySD can translate Vensim (``.mdl``) and Stella (``.xmile`` / ``.stmx``) models
into Julia, to run without Python or PySD.  Two backends are available:

.. list-table::
   :header-rows: 1
   :widths: 15 45 40

   * - Backend
     - How it works
     - When to use
   * - ``"dsl"`` *(default)*
     - Writes the model in PySD.jl's language for models (``@equations``,
       one file per view), to read and edit as Julia code; PySD.jl builds and
       runs it (see `The DSL backend`_)
     - General use, large models, models you want to keep working on in
       Julia; changing parameters and inputs between runs; derivatives and
       uncertainties through a run
   * - ``"mtk"``
     - Writes a ``ModelingToolkit.ODESystem``; ModelingToolkit simplifies it
       symbolically before solving (see `The MTK backend`_)
     - Small-to-medium models where you want symbolic analysis, LaTeX
       export of the simplified system, or automatic sparsity detection

.. note::
   The MTK backend runs ``structural_simplify`` before solving, which can
   take minutes to hours for large subscripted models.  For large models use
   the DSL backend.


Prerequisites
-------------

Julia 1.10 or later. Install via `juliaup <https://github.com/JuliaLang/juliaup>`_::

   curl -fsSL https://install.julialang.org | sh

Then install the ``PySD.jl`` companion library, which the translated models
use (https://github.com/rogersamso/PySD.jl).  ``pip install pysd`` doesn't
include it, and it isn't in Julia's General registry, so clone it from
GitHub at the version this PySD's translated models need (0.11) and add the
clone (``Pkg.add(url=...)`` can't, as PySD.jl's test models are a git
submodule)::

   git clone --branch v0.11.0 https://github.com/rogersamso/PySD.jl
   julia -e 'using Pkg; Pkg.develop(path="PySD.jl")'

Working on PySD from a git checkout, use its submodule instead, which is at
that version::

   git submodule update --init pysd/builders/julia/PySD.jl
   julia -e 'using Pkg; Pkg.develop(path="pysd/builders/julia/PySD.jl")'

That is all the DSL backend needs.  The MTK backend also needs::

   julia -e 'using Pkg; Pkg.add([
       "OrdinaryDiffEq",
       "OrdinaryDiffEqLowOrderRK",
       "ModelingToolkit",
       "Symbolics",
       "DataInterpolations",
       "NCDatasets",
       "JSON3",
   ])'


Translating a model
--------------------

From Python
^^^^^^^^^^^

.. code-block:: python

   import pysd

   # DSL backend (default): a folder model_dsl/
   path = pysd.translate_to_julia("model.mdl")

   # Split views: one file per Vensim view
   path = pysd.translate_to_julia("model.mdl", split_views=True)

   # MTK backend: one model.jl
   path = pysd.translate_to_julia("model.mdl", backend="mtk")

   # MTK backend, JSON data format: a companion _data.json instead of
   # Excel reads at load time
   path = pysd.translate_to_julia("model.mdl", backend="mtk", data_format="json")

**Parameters**

``backend``
   ``"dsl"`` (default) or ``"mtk"``.  (The former ``"ode"`` backend, a
   plain ODE function, was removed: the DSL backend runs the model with the
   same solvers, faster.)

``split_views``
   When ``True`` and the model has multiple Vensim views, the output is
   split into one file per view (with the MTK backend, a main ``.jl`` file
   and one module file per view under a ``modules_<name>/`` directory).

``data_format``
   MTK backend only.  ``"hardcoded"`` (default) reads Excel files at Julia
   startup via ``PySD.jl`` helpers.  ``"json"`` reads the external data
   (GET CONSTANTS, GET LOOKUPS, GET DATA) when the model is translated and
   writes it to a companion ``<model>_data.json``, which the model reads
   via ``JSON3.jl``: the model no longer needs the Excel files.  Arrays
   are stored flattened in Julia's (column-major) order with their
   ``shape``; missing values are ``null``.  Constants written in the model
   itself stay in the ``.jl`` file.

``encoding``
   Source file encoding (Vensim only). If ``None`` the encoding is
   detected from the model file header.

From the command line
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   python -c "import pysd; pysd.translate_to_julia('model.mdl')"


The DSL backend
---------------

``backend="dsl"`` writes the model as Julia code in PySD.jl's language for
models, meant to be read and edited: each Vensim variable is a declaration
with its unit and description, and an equation (see PySD.jl's README,
"Writing models in Julia").  Vensim and XMILE models translate this way;
see below for what it translates and what not yet.

.. code-block:: python

   path = pysd.translate_to_julia("model.mdl", backend="dsl", split_views=True)

It writes a folder, ``model_dsl/``:

``model.jl``
   Builds the model: includes the other files and calls ``Model(…)`` with
   the time settings.
``<view>.jl``
   The equations of each Vensim view (with ``split_views=True``; otherwise
   one ``equations.jl``).
``dims.jl``, ``inputs.jl``, ``units.jl``
   The subscript ranges as dimensions; the Excel sheets the model reads;
   the units Unitful.jl doesn't know (each its own dimension).
``names.tsv``
   Each Vensim name and its Julia name.
``Project.toml``, ``src/Model.jl``
   The folder as a Julia package (see below).

To run it:

.. code-block:: julia

   using PySD
   m = include("model_dsl/model.jl")      # builds the model
   r = run_model(m)
   r[:population]                         # a variable, at each saved time

   # again, with other values: no rebuild
   r2 = run_model(m; params = (; growth_rate = 0.02))
   r3 = run_model(m; inputs = (; scenario = excel("scen.xlsx", "High")))

Data variables without an equation (in Vensim, filled from a dataset, often
another model's results) are read with ``data(parents, :name)``: ``parents``,
in ``inputs.jl``, is given their sources, runs of other models in memory or
files of series (``.csv`` or ``.tab``, as ``data_files`` in Python):

.. code-block:: julia

   w = run_model(world)
   r = run_model(region; inputs = (; parents = (w,)))
   r = run_model(region; inputs = (; parents = (series_file("world.tab"),)))

Building and compiling a large model takes minutes, in every new Julia
session.  To do it only once, use the folder as a package: Julia then saves
the model built and compiled the first time, and later sessions load it in
seconds.

.. code-block:: julia

   using Pkg; Pkg.develop(path = "model_dsl")   # once, where PySD.jl is installed

   using Model                                  # the first time: as long as a build
   r = run_model(Model.model)                   # later sessions: seconds

For a model of 1,867 names, the first ``using`` takes about 8 minutes;
later sessions load it in about 8 s, and its first run takes about 6 s.
``src/Model.jl`` builds the model with PySD.jl's ``@precompile_model``, which
runs it once while Julia precompiles the package so that all it compiles is
saved.  The package's name is the model's file name in CamelCase.  Julia builds the
model again, once, when one of its files or Excel sheets changes; changing
parameters or inputs at run time never needs it.

What a translated model gives you, besides running:

- **Units**: every name has its unit (Vensim's, in Unitful.jl's notation; the
  units Unitful.jl doesn't know are declared in ``units.jl``, each its own
  dimension).  ``check_units(m)`` checks that every equation's units add up;
  ``run_model(m; check_units = true)`` checks them before running (``:warn``
  to only print what doesn't).
- **Other values without rebuilding**: ``run_model(m; params = (; name =
  value))`` and ``run_model(m; inputs = (; sheet = excel(…)))`` run the model
  again, in seconds.  A parameter that fixes the structure of an operator
  (the time of a DELAY FIXED, the order of a SMOOTH N) needs the model built
  again, and ``run_model`` says so.
- **Python's digits**: models are built with ``numerics = :python``, so sums,
  powers, logarithms, exponentials and inverses round as in Python PySD
  (numpy's order of additions, the C library's functions, LAPACK's solver).
- **Names**: ``names.tsv`` maps each Vensim name to its Julia name, to compare
  results with Vensim's or Python's.
- **Derivatives and uncertainties**: a run computes in the number type of
  its parameters.  With ForwardDiff.jl,
  ``ForwardDiff.derivative(k -> run_model(m; params = (; k))[:x][end], 0.1)``
  is the exact derivative of a result in a parameter, for calibration or to
  see what a result is most sensitive to; with Measurements.jl,
  ``params = (; k = 0.1 ± 0.01)`` carries the uncertainty to every result, in
  one run.
- **Other solvers**: with a SciML solver package loaded,
  ``run_model(m, Euler())`` gives the same values with SciML's Euler, and
  ``run_model(m, BS3(); reltol = 1e-8)`` solves the ODE with adaptive steps,
  to check that ``TIME STEP`` is small enough; ``ODEProblem(m)`` is the model
  as a SciML problem, for callbacks and ensembles of runs.  ``DELAY FIXED``
  and ``SAMPLE IF TRUE`` move at the model's ticks with any solver (see
  "Choosing a solver"); only Euler with steps of ``TIME STEP`` gives
  Vensim's values.
- **Results as tables**: ``DataFrame(r)`` and ``CSV.write(path, r)`` give a
  column per variable and element, with the Julia names in Python PySD's
  form (``stock[north,a]``); ``DimStack(r)`` is a dataset, as xarray's, a
  layer per variable over the time and its dimensions.

These need PySD.jl 0.10.0, which the model's package requires.

What it translates: stocks, auxiliaries and constants; subscripts (ranges,
subranges, ranges defined with other ranges, mappings, arrays of up to 3
dimensions, ``:EXCEPT:`` (also narrower at several positions), subscripts and
elements as values, variables defined in parts or element by element); lookups (also inline, ``WITH
LOOKUP``); ``GET DIRECT CONSTANTS``, ``LOOKUPS`` and ``DATA`` (also in parts,
one per element of one subscript or of several, or mixed with numbers);
``GET DIRECT SUBSCRIPT`` (read as it translates); macros (written in place, a
copy of their equations per call); data variables without an
equation (from other models' runs or files, above); ``SMOOTH``, ``SMOOTHI``,
``SMOOTH3``, ``SMOOTH N``, ``DELAY1``, ``DELAY3``, ``DELAY N``, ``DELAY
FIXED`` (also of a constant or of an expression), ``TREND``, ``FORECAST``, ``SAMPLE IF TRUE``,
``INITIAL`` (also of an expression), ``ACTIVE INITIAL``; the random functions (a draw per tick, the
same with any solver); ``SUM``, ``PROD``, ``VMIN``, ``VMAX`` (over one or
several dimensions); ``VECTOR SELECT``, ``VECTOR SORT ORDER``, ``VECTOR
RANK``, ``VECTOR REORDER``; ``ALLOCATE BY PRIORITY``, ``ALLOCATE
AVAILABLE``; ``INVERT MATRIX``; ``GAME`` (its value);
``GET TIME VALUE`` of the time or the initial time plus an offset
(``(0, offset, 0)``, ``(1, offset, 0)``) and of the computer's clock
(``(2, 0, measure)``, with Julia's Dates); ``IF THEN ELSE``, ``STEP``, ``RAMP``,
``PULSE``, ``PULSE TRAIN``, ``QUANTUM``, ``ZIDZ``, ``XIDZ``, ``MIN``, ``MAX``
(also XMILE's of one argument), ``MODULO``, ``ELMCOUNT``, ``LN``, ``LOG``,
``EXP``, ``SQRT``, the trigonometric functions, ``PI``; ``TIME``, ``TIME
STEP``, ``INITIAL TIME``, ``FINAL TIME`` and ``SAVEPER`` in equations, and
as equations: time settings that change during the run (a ``FINAL TIME``
that depends on a variable, a ``TIME STEP`` that depends on ``Time``) are
variables of the model that ``Model`` reads at every step, as the Python
builder (``time = (0, :final_time), dt = :time_step``).

What it doesn't translate yet.  A function it does not know (as ``LOOKUP
FORWARD``): as the Python builder, a warning, and a call that stops the
model with an error naming the function when it is evaluated (also
``GET TIME VALUE`` of a calendar measure of the model's time, as the Python
builder).  The translation stops, listing the elements, for: macros with
several outputs (PySD's Vensim parser doesn't read their calls yet); constants written as a table of numbers of more than 2 dimensions; ``INVERT
MATRIX`` over more than 3 dimensions.

The reference tests run every test model with the DSL backend
(``PYSD_JULIA_BACKENDS=dsl``): 186 match Vensim's output (139 Vensim models,
47 XMILE); the others are those the Python pathway marks as failing.
Three large models (a global model and two regional models that take data
from it, 1,900 to 2,000 names each) match Python PySD to 1e-12 in every saved
value.

Limits to know:

- A model whose data variables come from other models can't be saved as a
  package yet (the package would need its sources while it's precompiled).
- A model that compares values that should be exactly equal (``IF x < 0``,
  where ``x`` is a difference of two equal sums) can still take another
  branch than Python at a step where the last digit differs, and diverge from
  there.


The MTK backend
---------------

``backend="mtk"`` writes one ``model.jl``: the variables and parameters, the
equations as a ``ModelingToolkit.ODESystem``, its structural simplification,
and ``run_model``.  It can be run directly::

   julia --project=/path/to/PySD.jl model.jl

It prints progress to stdout, runs the simulation, and writes a NetCDF
results file (``<model>_results.nc``) next to the ``.jl`` file.

You can also ``include`` the file interactively:

.. code-block:: julia

   include("model.jl")       # defines sys, run_model, u0, tspan, …

   sol = run_model()         # run with defaults

The ``run_model`` function accepts keyword arguments:

.. code-block:: julia

   # Higher-order solver
   sol = run_model(solver=Tsit5())

   # Finer time step
   sol = run_model(dt=0.01)

   # Custom time span
   sol = run_model(tspan=(2000.0, 2100.0))

   # Custom initial values of the stocks (pairs, as u0)
   sol = run_model(u0=my_u0)


Choosing a solver
^^^^^^^^^^^^^^^^^

The default is ``Euler()`` with steps of ``TIME STEP``, which matches
Vensim's integration method.  All solvers from `OrdinaryDiffEq.jl
<https://docs.sciml.ai/DiffEqDocs/stable/solvers/ode_solve/>`_ work, with
both backends: ``run_model(m, RK4())`` or ``run_model(m, BS3(); reltol =
1e-8)`` with the DSL backend, ``run_model(solver = RK4())`` or
``run_model(dt = time_step / 4)`` with the MTK backend.  Other solvers give
more accurate stocks (and so other numbers than Vensim's), which shows
whether ``TIME STEP`` is small enough.

``DELAY FIXED`` and ``SAMPLE IF TRUE`` are defined by steps of ``TIME STEP``,
the model's ticks: a fixed delay gives its input N ticks before, a sample
takes its input at the ticks where its condition holds.  The translated
model records them at the ticks (PySD.jl's ``PySDTicks``): the solver stops
at every tick, where each delay's input and each sample's value are recorded
from the state there, and they are held in between, so the solver's own
steps only integrate the stocks.  With Euler steps of ``TIME STEP`` the
values are Vensim's; with any other solver the model is the same.  (A
``DELAY N`` whose delay time changes keeps its past delay times the same
way.)  Stopping at every tick also means that an adaptive solver never takes
steps longer than ``TIME STEP``.

A ``DELAY FIXED`` whose delay time can't be evaluated when the model is
translated is approximated by a first-order delay, exact only with Euler
steps of ``TIME STEP``; ``run_model`` then refuses other solvers, with an
error that names it.

Switches on the time (``STEP``, ``PULSE``, ``PULSE TRAIN``, comparisons with
``Time``) also make the model stop at every tick, and its equations see the
time through PySD.jl's ``pysd_tick_time``: a solver's step that ends at a
tick sees the time just before it, so a switch at a tick changes the rates
from that tick on, with any solver.

The ``RANDOM`` functions draw once per tick, as Vensim: each call is a
variable of its own, a standard draw (PySD.jl's ``pysd_tick_uniform``,
``_normal``, ``_exponential``) seeded from the call, its seed, the element and
the tick, which the equation scales and cuts.  A draw holds between ticks,
and every run draws the same, with any solver.

Some equations are written for Euler steps of ``TIME STEP`` and may mean
something else with another solver: ``TIME STEP`` used as a value
(``(target - x) / TIME STEP`` sets a stock in one step), ``Time`` tested for
equality, non-negative stocks.  With any other solver than Euler with steps
of ``TIME STEP``, ``run_model`` names them in a warning, once.

.. list-table::
   :header-rows: 1

   * - Solver
     - Notes
   * - ``Euler()``
     - Default; matches Vensim output exactly
   * - ``Tsit5()``
     - Fast, accurate explicit solver; good general replacement
   * - ``Rodas5P()``
     - Stiff systems (widely different time scales)
   * - ``RK4()``
     - Classic 4th-order Runge-Kutta


Accessing results
^^^^^^^^^^^^^^^^^

``sol`` is a standard
`DiffEq solution object <https://docs.sciml.ai/DiffEqDocs/stable/basics/solution/>`_;
read any variable symbolically via ``sys``:

.. code-block:: julia

   sol.t                      # the saved times
   sol[sys.population]        # time series for a scalar variable
   sol[sys.stock_a[2]]        # an element of a subscripted variable
   sol(2025.0)[sys.gdp]       # interpolate at a specific time


Saving results
^^^^^^^^^^^^^^

The generated file calls ``save_results`` automatically, writing a
`NetCDF <https://www.unidata.ucar.edu/software/netcdf/>`_ file.  You can
also call it manually:

.. code-block:: julia

   save_results(sol, sys, _dim_labels, "output.nc")

The file holds the unknowns of the simplified system, a series over time
each.  (With the DSL backend, ``DataFrame(r)``, ``CSV.write(path, r)`` and
``DimStack(r)`` give the results; see `The DSL backend`_.)

Read the results with any NetCDF library, e.g. in Python:

.. code-block:: python

   import xarray as xr
   ds = xr.open_dataset("model_results.nc")
   print(ds["population"])


External data (Excel and text files)
-------------------------------------

Models that use ``GET XLS`` / ``GET DIRECT`` ``CONSTANTS``, ``LOOKUPS`` or
``DATA`` are fully supported.  The translated Julia model reads from the
**same files at runtime** — no intermediate conversion is needed.  The same
file formats as in the Python builder are read:

- ``.xlsx`` / ``.xlsm`` (with XLSX.jl);
- ``.xls``, the Excel 97-2003 format (with LibXLS.jl);
- any other file as delimited text (``.csv``, ``.tab``, ...).  The sheet
  argument is the separator when it is not alphanumeric (e.g. ``','`` or
  ``';'``); otherwise the separator is ``,`` for ``.csv`` files and a tab
  for any other extension.

Excel file paths in the generated code are relative to the ``.jl`` file
(via ``@__DIR__``), so Excel files must remain at their original locations
relative to the model.

All Vensim cell reference modes are supported:

- **Named ranges** (``.xlsx`` only) — ``GET DIRECT CONSTANTS('data.xlsx', 'Sheet1', 'param_name')``
- **Cell references** — ``GET DIRECT CONSTANTS('data.xlsx', 'Sheet1', 'B2')``
- **Row/column mode** — ``GET DIRECT LOOKUPS('data.xlsx', 'Sheet1', '4', 'C5')``

A named range takes precedence over a cell or column reference that looks
the same (e.g. a range named ``xs``).  In row/column mode, a subscripted
variable reads one row (or column) per element, starting at the given cell.

Files are read once, when the model is loaded, and cached in memory.


PySD.jl companion library
--------------------------

``PySD.jl`` (located at ``pysd/builders/julia/PySD.jl/``) provides the
runtime helper functions used by generated models, imported via
``using PySD``.

**Vensim built-in functions**

- ``pysd_xidz(x, y, z)`` — safe division; returns ``z`` when ``y == 0``
- ``pysd_zidz(x, y)`` — safe division; returns ``0`` when ``y == 0``
- ``pysd_pulse(t, start, width)``
- ``pysd_pulse_train(t, start, interval, width, end_time)``
- ``pysd_ramp(t, slope, start, end)``
- ``pysd_step(t, height, step_time)``
- ``pysd_log_base(x, base)``
- ``pysd_logical_and(a, b)``, ``pysd_logical_or(a, b)``, ``pysd_logical_not(a)``
- ``pysd_safe(x)`` — replaces ``NaN``/``Inf`` with ``0.0`` (guards array
  allocations against uninitialised reads in subscripted equations)
- ``pysd_allocate_by_priority(request, priority, width, supply)`` — Vensim priority allocation
- ``pysd_allocate_available(request, pp, avail)`` — Vensim profile-based demand allocation

**Excel data readers**

- ``pysd_xlsx_read_constant(path, sheet, name)``
- ``pysd_xlsx_read_series(path, sheet, x_ref, y_ref)``
- ``pysd_xlsx_build_lookup_dispatch(path, sheet, x_ref, y_ref; n=1, method=:interpolate)``
  — ``n`` is the number of series to read in row/column mode; ``method`` is
  the DATA keyword (``:interpolate``, ``:hold_backward``, ``:look_forward``)

These read ``.xlsx``, ``.xlsm``, ``.xls`` and delimited text files.

**Result writer**

- ``save_results(sol, sys, dim_labels, path)`` — writes a NetCDF file of
  an MTK model's results

**LaTeX export** *(MTK backend only)*

- ``pysd_export_latex(sys; filename=nothing)`` — returns a LaTeX string of
  the simplified ODE system; writes a standalone ``.tex`` file when
  ``filename`` is given


Exporting equations to LaTeX (MTK and DSL)
-------------------------------------------

.. code-block:: julia

   include("model.jl")           # MTK backend

   # LaTeX string
   tex = export_latex()

   # Write a standalone compilable .tex file
   export_latex(filename="equations.tex")

The exported equations correspond to the **structurally simplified** system
after ModelingToolkit's index reduction — redundant auxiliaries are
substituted away.

A model of the DSL backend exports its equations as they are written, a
section per view with a table of its names, units and descriptions (needs
Latexify.jl):

.. code-block:: julia

   using PySD, Latexify
   m = include("model_dsl/model.jl")
   pysd_export_latex(m; filename = "model.tex")


Supported Vensim features (MTK backend)
---------------------------------------

For the DSL backend, see `The DSL backend`_ (what it translates and what
not yet).

.. list-table::
   :header-rows: 1

   * - Feature
     - MTK backend
   * - Stocks (``INTEG``)
     - Supported
   * - Auxiliaries
     - Supported
   * - Constants / parameters
     - Supported
   * - Subscripts / arrays (1D, 2D, 3D; numerically validated against Vensim)
     - Supported
   * - Lookup tables (inline)
     - Supported
   * - ``GET DIRECT CONSTANTS``
     - Supported
   * - ``GET DIRECT LOOKUPS``
     - Supported
   * - ``GET DIRECT DATA``
     - Supported
   * - ``GET DIRECT SUBSCRIPT`` (subscript ranges from Excel)
     - Supported (read at translation time; correct array shapes)
   * - ``SMOOTH`` / ``SMOOTH3`` / ``SMOOTHN``
     - Supported (also over several subscript dimensions)
   * - ``DELAY1`` / ``DELAY3`` / ``DELAYN``
     - Supported (as the Python builder: the order is read at the initial
       time and truncated, and lowered while order × TIME STEP exceeds the
       delay time; with a changing delay time each stage keeps the delay time
       it started with)
   * - ``DELAY FIXED``
     - Supported (the input recorded at every tick, read N ticks later, with
       any solver; N comes from the delay time at the initial time, as in the
       Python builder)
   * - ``TREND``, ``FORECAST``
     - Supported (as the Python builder, also subscripted)
   * - ``SAMPLE IF TRUE``
     - Supported (its value recorded at every tick and held in between, with
       any solver)
   * - ``INITIAL``
     - Supported
   * - ``ACTIVE INITIAL``
     - Supported (its initial value while the initial values are computed)
   * - ``IF THEN ELSE``
     - Supported (``ifelse``)
   * - ``EXP``, ``LN``, ``LOG``, ``SQRT``, the trigonometric functions
     - Supported (PySD.jl's ``PythonMath.exp``, ``PythonMath.log``, ...:
       Python PySD's digits, and NaN out of the domain, as numpy)
   * - ``INTEGER``
     - Supported, written ``ifelse(x < 0, ceil(x), floor(x))``: Symbolics has
       no ``trunc`` for symbolic variables (only ``floor`` and ``ceil``).
       The values are the same; the generated equations read differently from
       the DSL backend's (``trunc``)
   * - ``PULSE``, ``STEP``, ``RAMP``, ``PULSE TRAIN``
     - Supported
   * - ``XIDZ``, ``ZIDZ``
     - Supported
   * - Multiple views (``split_views=True``)
     - Supported
   * - ``GAME``
     - Supported (passes through; interactive play ignored)
   * - ``ELMCOUNT``
     - Supported (resolved to integer literal at translation time)
   * - ``DATA`` variables (tab-delimited ``.tab`` files)
     - Supported (runtime ``_tab_val`` interpolation via ``tab_data_files=``
       parameter)
   * - ``DATA`` variables fed from another model's NetCDF output
     - Not supported
   * - Subscripted ``GET DIRECT LOOKUPS`` > 2D
     - Partial (flattened to first column with warning)
   * - ``SMOOTH N`` with non-integer order
     - Partial (order rounded to the nearest integer)
   * - ``EXCEPT`` exclusion on 3-D subscripts
     - Supported (per-index comprehension equations)
   * - Macros (``MACRO`` … ``END OF MACRO``, including stocks and macros
       calling other macros)
     - Supported (expanded in place before building: each call gets its own
       copies of the macro's variables, named
       ``<variable>_<macro>_<element>``)
   * - Subranges, subscript mapping, ranges defined with other ranges,
       subscripts used as values
     - Supported (as the Python builder: a subscript used as a value is its
       position in the largest range containing it)
   * - XMILE ``MIN``/``MAX`` aggregation (``vmin_xmile``, ``vmax_xmile``)
     - Supported
   * - XMILE ``DELAY`` embedded in expression
     - Supported (lifted to an auxiliary, a fixed delay recorded at the
       ticks)
   * - ``ALLOCATE AVAILABLE`` / ``ALLOCATE BY PRIORITY``
     - Supported (exact Vensim algorithm via PySD.jl helpers; subscripted
       variables allocate along their last dimension, as the Python builder)
   * - ``VECTOR SORT ORDER``, ``VECTOR RANK``, ``VECTOR REORDER``
     - Supported (with comparisons, which ModelingToolkit can build equations
       from)
   * - ``VECTOR SELECT``
     - Partial (the error actions are not checked; with selections that are
       variables, building the system can take very long)
   * - ``GET TIME VALUE``
     - Supported for the current time plus an offset (``GET TIME VALUE(0,
       offset, 0)``), as the Python builder
   * - Control variables that change during the run (``FINAL TIME``,
       ``TIME STEP``, ``SAVEPER`` depending on ``Time`` or on model variables)
     - Supported (as the Python builder: evaluated at every step;
       ``run_model`` takes Euler steps; not with fixed delays, samples,
       random draws or switches on the time, which move by steps of ``TIME
       STEP``)

.. note::
   When the builder encounters an unsupported or partially-supported construct
   it emits a Python ``UserWarning`` during translation and writes a
   placeholder (``0.0``) in the generated file.  Always review warnings after
   translation to identify gaps.

Comparison with the Python builder
-----------------------------------

The Python builder supports every Vensim/Stella construct that PySD can
parse.  The Julia builder does not yet cover:

.. list-table::
   :header-rows: 1

   * - Feature
     - Python builder
     - Julia builder
   * - ``ALLOCATE AVAILABLE`` / ``ALLOCATE BY PRIORITY``
     - Full
     - Supported via ``pysd_allocate_available`` / ``pysd_allocate_by_priority`` in PySD.jl
   * - ``DATA`` variables (tab-delimited ``.tab`` file source)
     - Full
     - Supported — pass ``tab_data_files=["data.tab"]`` to ``run_model()``
   * - ``DATA`` variables fed from another model's NetCDF output
     - Full
     - Supported — pass ``nc_data_files=["other_model_results.nc"]`` to ``run_model()``
   * - Step-by-step execution (``model.step()``)
     - Full — essential for ABM coupling (e.g. Mesa)
     - Not supported — ``run_model()`` always runs the full simulation in one call
   * - Changing constants, lookups and data (``model.run(params=...)``)
     - Full — also in the middle of a run, and any variable's equation
     - Between runs, with the DSL backend: ``run_model(m; params = (; name =
       value))`` and ``run_model(m; inputs = ...)`` (see `The DSL backend`_)
   * - State export/import (``model.export()`` / ``model.import_()``)
     - Full — snapshot and restore model state for warm restarts or ensemble branching
     - Not supported
   * - Submodel selection (``model.select_submodel()``)
     - Full — prune to a variable subset for faster targeted simulation
     - Not supported — the full model is always simulated
   * - Subscripted lookups with > 2 subscript dimensions
     - Full
     - Flattened to first column (with warning)
   * - ``SMOOTH N`` with non-integer order
     - Full (arbitrary real order)
     - Order rounded to the nearest integer
   * - ``EXCEPT`` exclusion on 3-D subscripts
     - Full
     - Supported (per-index comprehension equations, same as 1-D and 2-D)
   * - Macros (``MACRO`` … ``END OF MACRO``)
     - Full
     - Supported; expanded in place before building, stocks included
   * - ``GAME`` interactive input
     - Full
     - Passes through; interactive value ignored in batch simulation
   * - ``DELAY FIXED`` exact semantics
     - Full (discrete transport delay)
     - Supported (the input recorded at every tick, as Vensim's ring buffer,
       with any solver; N comes from the delay time at the initial time)
   * - ``SAMPLE IF TRUE`` exact semantics
     - Full (holds last-true value)
     - Supported (recorded at every tick and held in between, with any solver)

With the same integration method (Euler with the model's time step), the
results match Python PySD's to the last digits in most variables: ``SUM``
adds the elements in numpy's order (``pysd_sum``: pairwise, the last
subscript varying fastest, NaN counted as 0), ``INVERT MATRIX`` uses the
same LAPACK routine as ``numpy.linalg.inv``, and the maths functions
(``EXP``, ``LN``, ``SIN``, ..., ``^``) call the C maths library, as numpy
does.  The elements a model does not define are NaN, as in Python.  This
matters where a model compares two values that are equal up to rounding
(``x < 0`` with ``x`` about ``1e-16``): both versions then take the same
branch.


Limitations
-----------

- **MTK structural analysis** scales poorly with model size.  For models
  with hundreds of subscripted equations (which expand into thousands of
  scalar equations) ``structural_simplify`` can take hours.  Use the DSL
  backend for large models.

- **EXCEPT subscript exclusion** on 4-D or higher subscripts emits a
  plain broadcast equation (the exclusion is ignored) with a warning.
  1-D, 2-D, and 3-D EXCEPT are fully supported.

- **DELAY FIXED** (MTK backend): as in Vensim and the Python builder, the
  number of ticks N is fixed by the delay time at the initial time, which
  may be an expression (e.g. ``2.5 * TIME STEP`` or ``4 + STEP(2, 15)``).
  Only if the delay time cannot be evaluated at translation time (e.g. it
  depends on external data) does the builder fall back to a first-order
  delay, exact only with Euler steps of ``TIME STEP``, with a warning.

- **Last digits.**  A few constructs can still differ from Python PySD in
  the last digit: ``SUM`` over an array that Python stores transposed adds
  the elements in another order, and ``TANH`` uses Julia's function (numpy
  has its own).  Such differences only change a result where the model compares
  two values that are equal up to rounding (``x < 0`` with ``x`` about
  ``1e-16``).

- **Non-integer SMOOTH N order** is rounded to the nearest integer; the
  Python builder supports arbitrary real-valued orders.  A SMOOTH N or
  DELAY N order that cannot be evaluated at translation time defaults to 3,
  with a warning.  DELAY N orders are truncated, as in the Python builder.

- **Tab-delimited DATA variables** (``.tab`` file sources), MTK backend:
  pass the file paths as ``run_model(tab_data_files=["data.tab"])``; the
  model reads and interpolates the time series at runtime.  (The DSL backend
  reads them with ``data(parents, :name)``; see `The DSL backend`_.)

- **Step-by-step execution** is not available.  The Python builder exposes
  ``model.set_stepper()`` / ``model.step()`` for advancing the simulation one
  time step at a time, which is the standard pattern for coupling with
  agent-based frameworks (e.g. Mesa, Agents.jl).  The Julia builder has no
  equivalent — ``run_model()`` always executes the full simulation in a single
  call.

- **Changing values in the middle of a run** is not available.  The Python
  builder's ``model.set_components()`` can replace any variable's equation
  with a new function or constant value at any point during a run.  In the
  Julia builder, constants, lookups and data are changed between runs with
  the DSL backend's ``run_model(m; params = ..., inputs = ...)`` (see `The
  DSL backend`_); variables that change during the run cannot be replaced.

- **State export/import** is not available.  The Python builder's
  ``model.export()`` / ``model.import_()`` snapshot and restore the full model
  state — stock values, stateful caches, and current time — enabling warm
  restarts and ensemble branching from a common saved point.  The Julia builder
  writes results to NetCDF via ``save_results()`` but cannot restore mid-run
  state.

- **Submodel selection** is not available.  The Python builder's
  ``model.select_submodel()`` prunes the model to a requested subset of
  variables, which can substantially reduce simulation time when only part of
  the model is needed.  The Julia builder always simulates the full model.

- The Euler solver (default) produces output that matches Vensim's built-in
  integration.  Higher-order solvers (e.g. ``Tsit5()``) are generally more
  accurate but may produce slightly different results.
