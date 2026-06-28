# LaTeX export for translated ModelingToolkit systems.
#
# Latexify.jl is a transitive dependency of ModelingToolkit and does not need
# to be listed in PySD.jl's own Project.toml.

"""
    pysd_export_latex(sys; filename=nothing)

Render a ModelingToolkit `ODESystem` as LaTeX equations.

If `filename` is given the LaTeX string is written to that file (wrapped in a
minimal document preamble so it can be compiled standalone).  The raw LaTeX
string is always returned.

# Example
```julia
include("my_model.jl")          # defines `sys`
pysd_export_latex(sys)                         # returns LaTeX string
pysd_export_latex(sys; filename="eqs.tex")     # also writes to file
```
"""
function pysd_export_latex(sys; filename::Union{Nothing,AbstractString}=nothing)
    lat = try
        using_latexify = Base.require(Base.PkgId(
            Base.UUID("23fbe1c1-3f47-55db-b15f-69d7ec21a316"), "Latexify"))
        using_latexify.latexify(sys)
    catch e
        error("Latexify.jl is required for LaTeX export. " *
              "Install it with: using Pkg; Pkg.add(\"Latexify\")")
    end

    tex_str = string(lat)

    if filename !== nothing
        doc = """
        \\documentclass{article}
        \\usepackage{amsmath}
        \\usepackage{breqn}
        \\begin{document}
        $tex_str
        \\end{document}
        """
        write(filename, doc)
    end

    return tex_str
end
