Julia / ModelingToolkit Builder
===============================

PySD can translate Vensim (``.mdl``) and Stella (``.xmile`` / ``.stmx``) models
into standalone Julia files that use
`ModelingToolkit.jl <https://docs.sciml.ai/ModelingToolkit/stable/>`_ for
symbolic ODE construction and
`OrdinaryDiffEq.jl <https://docs.sciml.ai/OrdinaryDiffEq/stable/>`_ for
numerical integration.

The generated Julia code requires **no Python or PySD at runtime** — only the
Julia packages listed below and the small companion library ``PySD.jl`` shipped
with PySD.


Prerequisites
-------------

Julia 1.10 or later is required. Install it from https://julialang.org or via
`juliaup <https://github.com/JuliaLang/juliaup>`_::

   curl -fsSL https://install.julialang.org | sh
   juliaup update

Install the required Julia packages once::

   julia -e 'using Pkg; Pkg.add([
       "ModelingToolkit",
       "OrdinaryDiffEq",
       "OrdinaryDiffEqLowOrderRK",
       "DataInterpolations",
       "XLSX",
   ])'

Then install the ``PySD.jl`` companion library that ships with PySD. From the
root of your PySD checkout::

   julia -e 'using Pkg; Pkg.develop(path="pysd/builders/julia/PySD.jl")'


Translating a model
--------------------

From Python
^^^^^^^^^^^

Use :func:`pysd.translate_to_julia`::

   >>> import pysd
   >>> path = pysd.translate_to_julia("path/to/model.mdl")
   >>> print(path)
   path/to/model.jl

The function returns the path to the generated ``.jl`` file, which is placed
next to the original model file.

Options:

``split_views``
   When ``True`` and the model has multiple Vensim views, the output is split
   into a main ``.jl`` file and one module file per view under a
   ``modules_<name>/`` directory.  Default is ``False``.

``encoding``
   Source file encoding (Vensim only). If ``None`` the encoding is read from
   the model file header; defaults to ``'UTF-8'``.

Example with split views::

   >>> path = pysd.translate_to_julia("model.mdl", split_views=True)


Running the translated model
-----------------------------

Basic usage
^^^^^^^^^^^

.. code-block:: julia

   include("model.jl")

   # Run with default settings (Euler solver, model time step)
   sol = run_model()

The ``run_model`` function accepts keyword arguments to override defaults:

.. code-block:: julia

   # Override the solver
   sol = run_model(solver=Tsit5())

   # Override the time step
   sol = run_model(dt=0.01)

   # Override the time span
   sol = run_model(tspan=(2000.0, 2030.0))

   # Override initial conditions
   sol = run_model(u0=u0)


Choosing a solver
^^^^^^^^^^^^^^^^^

The default solver is ``Euler()``, which matches Vensim's integration method.
All solvers from `OrdinaryDiffEq.jl
<https://docs.sciml.ai/DiffEqDocs/stable/solvers/ode_solve/>`_ are available.
Common alternatives:

.. list-table::
   :header-rows: 1

   * - Solver
     - Use case
   * - ``Euler()``
     - Default; matches Vensim output exactly
   * - ``Tsit5()``
     - Good general-purpose explicit solver; faster and more accurate
   * - ``Rodas5P()``
     - Stiff systems (e.g. models with very different time scales)
   * - ``RK4()``
     - Classic 4th-order Runge-Kutta

Example::

   using OrdinaryDiffEq

   sol = run_model(solver=Tsit5(), dt=0.1)


Accessing results
^^^^^^^^^^^^^^^^^

The return value ``sol`` is a standard
`DiffEq solution object <https://docs.sciml.ai/DiffEqDocs/stable/basics/solution/>`_:

.. code-block:: julia

   # Time points
   sol.t

   # All state variables at all time points
   sol.u

   # Access a specific variable by its symbolic name
   sol[population]

   # Interpolate at a specific time
   sol(2025.0)


External data (Excel files)
----------------------------

Vensim models that use ``GET DIRECT CONSTANTS``, ``GET DIRECT LOOKUPS``, or
``GET DIRECT DATA`` to read from Excel files are fully supported. The translated
Julia model reads from the **same Excel files at runtime** using
`XLSX.jl <https://github.com/felipenoris/XLSX.jl>`_ — no intermediate data
conversion is needed.

The Excel file paths in the generated code are relative to the ``.jl`` file
(using Julia's ``@__DIR__``), so the Excel files must remain at their original
locations relative to the model. For example, if the Vensim model references
``../data.xlsx``, the Excel file must be one directory up from the ``.jl`` file.

All three Vensim cell reference modes are supported:

- **Named ranges** — e.g. ``GET DIRECT CONSTANTS('data.xlsx', 'Sheet1', 'my_param')``
- **Cell references** — e.g. ``GET DIRECT CONSTANTS('data.xlsx', 'Sheet1', 'B2')``
- **Row/column mode** — e.g. ``GET DIRECT LOOKUPS('data.xlsx', 'Sheet1', '4', 'C5')``

Excel files are cached in memory so each file is read only once, regardless of
how many variables reference it.


PySD.jl companion library
--------------------------

``PySD.jl`` is a small Julia package (located at
``pysd/builders/julia/PySD.jl/``) that provides the runtime helper functions
used by generated models. It is imported via ``using PySD`` in each generated
file.

The library provides:

**Vensim built-in functions** — symbolic-safe implementations that work inside
ModelingToolkit equations:

- ``pysd_xidz(x, y, z)`` — safe division (returns ``z`` when ``y == 0``)
- ``pysd_zidz(x, y)`` — safe division (returns ``0`` when ``y == 0``)
- ``pysd_pulse(t, start, width)`` — pulse function
- ``pysd_pulse_train(t, start, interval, width, end_time)`` — repeating pulse
- ``pysd_ramp(t, slope, start, end)`` — ramp function
- ``pysd_step(t, height, step_time)`` — step function
- ``pysd_log_base(x, base)`` — logarithm with arbitrary base
- ``pysd_logical_and(a, b)``, ``pysd_logical_or(a, b)``,
  ``pysd_logical_not(a)`` — symbolic-safe logical operators

**Excel data readers** — functions for reading Vensim external data:

- ``pysd_xlsx_read_constant(path, sheet, name; transpose=false)``
- ``pysd_xlsx_read_series(path, sheet, x_ref, y_ref)``

**LaTeX export** — render the simplified ODE system as LaTeX equations:

- ``pysd_export_latex(sys; filename=nothing)`` — returns the LaTeX string;
  writes a standalone ``.tex`` file when ``filename`` is given


Exporting equations to LaTeX
-----------------------------

Translated models include a convenience function to export the simplified ODE
system as LaTeX equations, using ModelingToolkit's integration with
`Latexify.jl <https://github.com/korsbo/Latexify.jl>`_.

.. code-block:: julia

   include("model.jl")

   # Get the LaTeX string
   tex = export_latex()

   # Write a standalone .tex file (compilable with pdflatex)
   export_latex(filename="equations.tex")

The exported equations correspond to the **structurally simplified** system —
the actual ODEs that are solved, not the raw Vensim definitions. This means
redundant auxiliary variables are substituted away, giving a compact
representation.

You can also call ``pysd_export_latex`` directly from the ``PySD`` module on
any ``ODESystem``:

.. code-block:: julia

   using PySD
   tex = pysd_export_latex(sys)
   pysd_export_latex(sys; filename="equations.tex")

When ``filename`` is given, the output is wrapped in a minimal LaTeX document
preamble (``\documentclass{article}``, ``amsmath``, ``breqn``) so it can be
compiled standalone with ``pdflatex``.


Supported Vensim features
--------------------------

.. list-table::
   :header-rows: 1

   * - Feature
     - Status
   * - Stocks (``INTEG``)
     - Supported
   * - Auxiliaries (algebraic equations)
     - Supported
   * - Constants
     - Supported
   * - Lookup tables (inline)
     - Supported
   * - ``SMOOTH`` / ``SMOOTH3`` / ``SMOOTHN``
     - Supported (expanded to chained first-order ODEs)
   * - ``DELAY1`` / ``DELAY3`` / ``DELAYN``
     - Supported (expanded to pipeline levels)
   * - ``DELAY FIXED``
     - Partial (falls back to identity: output = input)
   * - ``INITIAL``
     - Supported (resolved to parameter constant when possible)
   * - ``IF THEN ELSE``
     - Supported (``ifelse``)
   * - ``PULSE``, ``STEP``, ``RAMP``
     - Supported
   * - ``PULSE TRAIN``
     - Supported
   * - ``XIDZ``, ``ZIDZ``
     - Supported
   * - ``GET DIRECT CONSTANTS``
     - Supported (reads from Excel at runtime)
   * - ``GET DIRECT LOOKUPS``
     - Supported (reads from Excel at runtime)
   * - ``GET DIRECT DATA``
     - Supported (reads from Excel at runtime)
   * - ``SAMPLE IF TRUE``
     - Partial (simplified to ``ifelse``; does not hold last-true value)
   * - ``TREND``, ``FORECAST``
     - Not yet supported (placeholder emitted)
   * - ``ALLOCATE AVAILABLE``, ``ALLOCATE BY PRIORITY``
     - Not yet supported (placeholder emitted)
   * - Subscripts / arrays
     - Not yet supported
   * - Macros
     - Not yet supported
   * - Multiple views (``split_views=True``)
     - Supported (separate module files per view)

.. note::
   When the builder encounters an unsupported feature, it emits a Python
   warning during translation and writes a placeholder equation (``0.0``)
   in the generated file. Review warnings after translation to identify
   any unsupported constructs in your model.


Limitations and notes
----------------------

- **Subscripts/arrays** are not yet supported. Subscripted variables from
  external data (e.g. multi-row lookups) are reduced to their first element.

- **SAMPLE IF TRUE** uses a simplified approximation
  (``ifelse(condition, input, initial_value)``) that does not preserve the
  "hold last true value" behaviour of Vensim's implementation.

- **DELAY FIXED** falls back to an identity function (output equals input)
  because fixed transport delays require discrete-event callbacks not yet
  implemented.

- The generated code uses ``structural_simplify`` from ModelingToolkit to
  reduce the system before solving. For very large models this step can take
  a few minutes.

- The Euler solver (default) produces output that matches Vensim's built-in
  integration. Switching to a higher-order solver (e.g. ``Tsit5()``) may
  produce slightly different results due to the different integration scheme,
  but is generally more accurate.
