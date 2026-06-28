module PySDMTKExt

using PySD
using ModelingToolkit
using NCDatasets

"""
    save_results(sol, sys, dim_labels, path)

MTK-backend variant.  `sys` is the `ODESystem`; variable names and ordering
are introspected from `unknowns(sys)`.
"""
function PySD.save_results(
    sol,
    sys::ModelingToolkit.AbstractSystem,
    dim_labels::Dict,
    path::AbstractString,
)
    ts = sol.t
    vars = ModelingToolkit.unknowns(sys)
    NCDatasets.Dataset(path, "c") do ds
        ds.attrib["Conventions"] = "CF-1.8"
        NCDatasets.defDim(ds, "time", length(ts))
        vt = NCDatasets.defVar(ds, "time", Float64, ("time",))
        vt[:] = ts
        vt.attrib["units"] = "1"

        for var in vars
            name = string(ModelingToolkit.getname(var))
            v = NCDatasets.defVar(ds, name, Float64, ("time",))
            v[:] = sol[var]
        end
    end
    return path
end

end # module
