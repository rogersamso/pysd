module PySD

using DataInterpolations
using NCDatasets
using Symbolics
using XLSX

const PYSD_JL_VERSION = let
    proj = joinpath(@__DIR__, "..", "Project.toml")
    m = match(r"version\s*=\s*\"([^\"]+)\"", read(proj, String))
    VersionNumber(m[1])
end

"""
    check_compat(built_with::VersionNumber)

Verify that the installed PySD.jl is compatible with the version the model
was translated against.  Raises an error when the major version differs.
"""
function check_compat(built_with::VersionNumber)
    if PYSD_JL_VERSION.major != built_with.major
        error(
            "This model was translated with PySD.jl v", built_with,
            " but the installed version is v", PYSD_JL_VERSION,
            ". Major-version mismatch — please update PySD.jl or re-translate the model."
        )
    end
    if PYSD_JL_VERSION < built_with
        @warn(
            "This model was translated with PySD.jl v$built_with " *
            "but the installed version is v$PYSD_JL_VERSION (older). " *
            "Some features may be missing."
        )
    end
end

export pysd_trunc, pysd_log_base, pysd_xidz, pysd_zidz,
       pysd_pulse, pysd_pulse_train, pysd_ramp, pysd_step,
       pysd_active_initial, pysd_ifelse,
       pysd_inv_mat2d_elem, pysd_inv_mat3d_elem,
       pysd_invert_matrix, pysd_elmcount,
       pysd_power, pysd_quantum, pysd_pi,
       pysd_xpulse, pysd_xpulse_train, pysd_xramp,
       pysd_random_0_1, pysd_random_uniform,
       pysd_random_normal, pysd_random_exponential,
       pysd_vector_select, pysd_vector_sort_order,
       pysd_vector_reorder, pysd_vector_rank,
       pysd_get_time_value,
       pysd_logical_and, pysd_logical_or, pysd_logical_not,
       pysd_safe, SafeArray,
       pysd_xlsx_read_constant, pysd_xlsx_read_series,
       pysd_xlsx_build_lookup_dispatch,
       pysd_export_latex,
       save_results,
       check_compat, PYSD_JL_VERSION

include("helpers.jl")
include("xlsx.jl")
include("latex.jl")
include("save_results.jl")

end
