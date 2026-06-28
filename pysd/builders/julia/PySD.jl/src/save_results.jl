# Save ODE simulation results to a NetCDF file.

using NCDatasets

"""
    save_results(sol, state_map, dim_labels, path)

ODE-backend variant.  `state_map` is a vector of `(name, index, subscript_labels)`.
Scalar state variables are saved with dimension `(time,)`.
Subscripted state variables add their element index as an extra dimension.
"""
function save_results(
    sol,
    state_map::AbstractVector,
    dim_labels::Dict,
    path::AbstractString,
)
    ts = sol.t
    NCDatasets.Dataset(path, "c") do ds
        ds.attrib["Conventions"] = "CF-1.8"
        NCDatasets.defDim(ds, "time", length(ts))
        vt = NCDatasets.defVar(ds, "time", Float64, ("time",))
        vt[:] = ts
        vt.attrib["units"] = "1"

        for (name, idx, subs) in state_map
            if isempty(subs)
                v = NCDatasets.defVar(ds, name, Float64, ("time",))
                v[:] = sol[idx, :]
            else
                n = length(subs)
                dim_name = name * "_dim"
                NCDatasets.defDim(ds, dim_name, n)
                v = NCDatasets.defVar(ds, name, Float64, ("time", dim_name))
                for (k, _label) in enumerate(subs)
                    v[:, k] = [sol.u[i][idx + k - 1] for i in eachindex(sol.t)]
                end
            end
        end
    end
    return path
end
