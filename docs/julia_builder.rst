Julia Builder
=============

PySD can translate Vensim (``.mdl``) and Stella (``.xmile`` / ``.stmx``) models
into standalone Julia files that run without Python or PySD at runtime.
Two backends are available:

.. list-table::
   :header-rows: 1
   :widths: 15 45 40

   * - Backend
     - How it works
     - When to use
   * - ``"ode"`` *(default)*
     - Emits a plain ``rhs!(du, u, p, t)`` function solved by
       ``OrdinaryDiffEq.jl``
     - General use; fast startup, full subscript support
   * - ``"mtk"``
     - Emits a ``ModelingToolkit.ODESystem``; ModelingToolkit performs
       symbolic simplification before solving
     - Small-to-medium models where you want symbolic analysis, LaTeX
       export, or automatic sparsity detection

.. note::
   The MTK backend runs ``structural_simplify`` before solving, which can
   take minutes to hours for large subscripted models.  For production runs
   the ODE backend is recommended.


Prerequisites
-------------

Julia 1.10 or later. Install via `juliaup <https://github.com/JuliaLang/juliaup>`_::

   curl -fsSL https://install.julialang.org | sh

Install the required Julia packages once::

   julia -e 'using Pkg; Pkg.add([
       "OrdinaryDiffEq",
       "DataInterpolations",
       "NCDatasets",
       "XLSX",
       "ModelingToolkit",   # only needed for the mtk backend
   ])'

Then install the ``PySD.jl`` companion library from your PySD checkout::

   julia -e 'using Pkg; Pkg.develop(path="pysd/builders/julia/PySD.jl")'


Translating a model
--------------------

From Python
^^^^^^^^^^^

.. code-block:: python

   import pysd

   # ODE backend (default)
   path = pysd.translate_to_julia("model.mdl")

   # MTK backend
   path = pysd.translate_to_julia("model.mdl", backend="mtk")

   # Split views — one module file per Vensim view
   path = pysd.translate_to_julia("model.mdl", split_views=True)

   # JSON data format — companion _data.json instead of inline Excel reads
   path = pysd.translate_to_julia("model.mdl", data_format="json")

**Parameters**

``backend``
   ``"ode"`` (default) or ``"mtk"``.

``split_views``
   When ``True`` and the model has multiple Vensim views, the output is
   split into a main ``.jl`` file and one module file per view under a
   ``modules_<name>/`` directory.

``data_format``
   ``"hardcoded"`` (default) reads Excel files at Julia startup via
   ``PySD.jl`` helpers.  ``"json"`` writes a companion
   ``<model>_data.json`` file and reads it via ``JSON3.jl``.

``encoding``
   Source file encoding (Vensim only). If ``None`` the encoding is
   detected from the model file header.

From the command line
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   python -c "import pysd; pysd.translate_to_julia('model.mdl')"


Running the model
-----------------

The generated ``.jl`` file is self-contained and can be run directly::

   julia --project=/path/to/PySD.jl model.jl

It prints progress to stdout, runs the simulation, and writes a NetCDF
results file (``<model>_results.nc``) next to the ``.jl`` file.

You can also ``include`` the file interactively:

.. code-block:: julia

   include("model.jl")       # defines run_model, u0, tspan, …

   sol = run_model()         # run with defaults

The ``run_model`` function accepts keyword arguments:

.. code-block:: julia

   # Higher-order solver
   sol = run_model(solver=Tsit5())

   # Finer time step
   sol = run_model(dt=0.01)

   # Custom time span
   sol = run_model(tspan=(2000.0, 2100.0))

   # Custom initial conditions
   sol = run_model(u0=my_u0)


Choosing a solver
^^^^^^^^^^^^^^^^^

The default is ``Euler()``, which matches Vensim's integration method.
All solvers from `OrdinaryDiffEq.jl
<https://docs.sciml.ai/DiffEqDocs/stable/solvers/ode_solve/>`_ work.

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
`DiffEq solution object <https://docs.sciml.ai/DiffEqDocs/stable/basics/solution/>`_.

**ODE backend** — use ``observe(u, t)`` to read any variable at any saved
time step:

.. code-block:: julia

   # All saved time points
   sol.t

   # Read a scalar variable at every time step
   obs = [mod.observe(sol.u[i], sol.t[i]) for i in eachindex(sol.t)]
   population = [o["population"] for o in obs]

   # Read a subscripted variable (returns a Vector)
   stock_a = [o["stock_a"] for o in obs]

   # Individual subscript element
   stock_a_entry1 = [o["stock_a"][1] for o in obs]

**MTK backend** — access variables symbolically via ``sys``:

.. code-block:: julia

   sol[sys.population]        # time series for a scalar variable
   sol(2025.0)[sys.gdp]       # interpolate at a specific time


Saving results
^^^^^^^^^^^^^^

The generated file calls ``save_results`` automatically, writing a
`NetCDF <https://www.unidata.ucar.edu/software/netcdf/>`_ file.  You can
also call it manually:

.. code-block:: julia

   # ODE backend
   save_results(sol, _state_map, _dim_labels, "output.nc")

   # MTK backend
   save_results(sol, sys, _dim_labels, "output.nc")

Read the results with any NetCDF library, e.g. in Python:

.. code-block:: python

   import xarray as xr
   ds = xr.open_dataset("model_results.nc")
   print(ds["population"])


External data (Excel files)
----------------------------

Models that use ``GET DIRECT CONSTANTS``, ``GET DIRECT LOOKUPS``, or
``GET DIRECT DATA`` are fully supported.  The translated Julia model reads
from the **same Excel files at runtime** — no intermediate conversion is
needed.

Excel file paths in the generated code are relative to the ``.jl`` file
(via ``@__DIR__``), so Excel files must remain at their original locations
relative to the model.

All Vensim cell reference modes are supported:

- **Named ranges** — ``GET DIRECT CONSTANTS('data.xlsx', 'Sheet1', 'param_name')``
- **Cell references** — ``GET DIRECT CONSTANTS('data.xlsx', 'Sheet1', 'B2')``
- **Row/column mode** — ``GET DIRECT LOOKUPS('data.xlsx', 'Sheet1', '4', 'C5')``

Excel files are cached in memory after the first read.


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

**Excel data readers**

- ``pysd_xlsx_read_constant(path, sheet, name)``
- ``pysd_xlsx_read_series(path, sheet, x_ref, y_ref)``
- ``pysd_xlsx_build_lookup_dispatch(path, sheet, x_ref, y_ref)``

**Result writer**

- ``save_results(sol, state_map_or_sys, dim_labels, path)`` — writes a
  NetCDF file; dispatches on ODE ``state_map`` (``AbstractVector``) or MTK
  ``sys`` (``AbstractSystem``)

**LaTeX export** *(MTK backend only)*

- ``pysd_export_latex(sys; filename=nothing)`` — returns a LaTeX string of
  the simplified ODE system; writes a standalone ``.tex`` file when
  ``filename`` is given


Exporting equations to LaTeX (MTK only)
----------------------------------------

.. code-block:: julia

   include("model.jl")           # MTK backend

   # LaTeX string
   tex = export_latex()

   # Write a standalone compilable .tex file
   export_latex(filename="equations.tex")

The exported equations correspond to the **structurally simplified** system
after ModelingToolkit's index reduction — redundant auxiliaries are
substituted away.


Supported Vensim features
--------------------------

.. list-table::
   :header-rows: 1

   * - Feature
     - ODE backend
     - MTK backend
   * - Stocks (``INTEG``)
     - Supported
     - Supported
   * - Auxiliaries
     - Supported
     - Supported
   * - Constants / parameters
     - Supported
     - Supported
   * - Subscripts / arrays (1D, 2D)
     - Supported
     - Supported
   * - Lookup tables (inline)
     - Supported
     - Supported
   * - ``GET DIRECT CONSTANTS``
     - Supported
     - Supported
   * - ``GET DIRECT LOOKUPS``
     - Supported
     - Supported
   * - ``GET DIRECT DATA``
     - Supported
     - Supported
   * - ``SMOOTH`` / ``SMOOTH3`` / ``SMOOTHN``
     - Supported
     - Supported
   * - ``DELAY1`` / ``DELAY3`` / ``DELAYN``
     - Supported
     - Supported
   * - ``DELAY FIXED``
     - Supported (first-order ODE approximation)
     - Supported
   * - ``TREND``, ``FORECAST``
     - Supported
     - Supported
   * - ``SAMPLE IF TRUE``
     - Supported (conditional ODE stock)
     - Supported
   * - ``INITIAL``
     - Supported
     - Supported
   * - ``IF THEN ELSE``
     - Supported (``ifelse``)
     - Supported (``ifelse``)
   * - ``PULSE``, ``STEP``, ``RAMP``, ``PULSE TRAIN``
     - Supported
     - Supported
   * - ``XIDZ``, ``ZIDZ``
     - Supported
     - Supported
   * - Multiple views (``split_views=True``)
     - Supported
     - Supported
   * - Macros
     - Partial (companion ``.jl`` file per macro)
     - Partial
   * - ``ALLOCATE AVAILABLE`` / ``ALLOCATE BY PRIORITY``
     - Not supported (placeholder ``0.0``)
     - Not supported

.. note::
   When the builder encounters an unsupported construct it emits a Python
   ``UserWarning`` during translation and writes a placeholder ``0.0`` in
   the generated file.  Review warnings after translation to identify any
   gaps.


Limitations
-----------

- **MTK structural analysis** scales poorly with model size.  For models
  with hundreds of subscripted equations (which expand into thousands of
  scalar equations) ``structural_simplify`` can take hours.  Use the ODE
  backend for large models.

- **EXCEPT subscript exclusion** (e.g. ``var[A,B] :EXCEPT: [A1,B1]``)
  with 3-D subscripts is not yet supported; a plain broadcast equation is
  emitted with a warning.

- **SAMPLE IF TRUE** uses a conditional ODE stock to approximate the
  hold-until-true behaviour.  Results match Vensim for typical use but may
  diverge for very large time steps.

- **DELAY FIXED** is approximated as a first-order ODE with the same delay
  constant; the discrete transport-delay semantics are not exact.

- The Euler solver (default) produces output that matches Vensim's built-in
  integration.  Higher-order solvers (e.g. ``Tsit5()``) are generally more
  accurate but may produce slightly different results.
