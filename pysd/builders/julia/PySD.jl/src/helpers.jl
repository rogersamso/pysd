# Vensim built-in function implementations for ModelingToolkit.
#
# All conditions use `ifelse` + `&`/`|` instead of `?:` / `&&` / `||` so
# they remain valid when called with symbolic (Num) arguments inside MTK
# equations.

# Base.trunc is not available as a symbolic primitive in MTK.
# Register a thin wrapper so INTEGER(x) / INT(x) works inside equations.
pysd_trunc(x::Real) = Base.trunc(x)
@register_symbolic pysd_trunc(x::Real)

pysd_log_base(x, base) = log(base, x)

pysd_xidz(x, y, z) = ifelse(iszero(y), z, x / y)

pysd_zidz(x, y) = ifelse(iszero(y), 0.0, x / y)

pysd_pulse(t_now, start, width) =
    ifelse((t_now >= start) & (t_now < start + width), 1.0, 0.0)

# NOTE: the Vensim parser reorders PULSE TRAIN(start, width, interval, end)
# to CallStructure arguments (start, interval, width, end).
pysd_pulse_train(t_now, start, interval, width, end_time) =
    ifelse((t_now >= start) & (t_now <= end_time) &
           (mod(t_now - start, interval) < width), 1.0, 0.0)

pysd_ramp(t_now, slope, start_time, end_time=Inf) =
    slope * max(0.0, min(t_now - start_time, end_time - start_time))

pysd_step(t_now, height, step_time) =
    ifelse(t_now >= step_time, float(height), 0.0)

# Vensim logical operators — values are always 0.0 (false) or 1.0 (true).
# Return Symbolic{Bool} via comparisons so the result can be used as the
# condition of a symbolic `ifelse` in MTK equations.
pysd_logical_and(a, b) = (a > 0.5) & (b > 0.5)
pysd_logical_or(a, b)  = (a > 0.5) | (b > 0.5)
pysd_logical_not(a)    = !(a > 0.5)

# ACTIVE INITIAL(expr, initial) — in ODE mode expr is always live;
# we just return expr (the first argument).
pysd_active_initial(expr, initial) = expr

# Symbolics' `ifelse` has type issues with `SymReal` conditions, so PySD
# emits `pysd_ifelse` instead.  A concrete Bool dispatches to the ternary;
# a symbolic / numeric condition compares against 0.5 then defers to `ifelse`.
pysd_ifelse(cond::Bool, a, b) = cond ? a : b
pysd_ifelse(cond, a, b) = ifelse(cond > 0.5, a, b)

# INVERT_MATRIX helpers — registered as symbolic black boxes so Symbolics
# does not attempt symbolic matrix algebra (which hangs for large matrices).
# At solve time the concrete array is passed and inv is computed numerically.
function pysd_inv_mat2d_elem(mat::AbstractMatrix, i::Int, j::Int)
    return inv(mat)[i, j]
end
@register_symbolic pysd_inv_mat2d_elem(mat::AbstractMatrix, i::Int, j::Int)

function pysd_inv_mat3d_elem(mat::AbstractArray, b::Int, i::Int, j::Int)
    return inv(mat[b, :, :])[i, j]
end
@register_symbolic pysd_inv_mat3d_elem(mat::AbstractArray, b::Int, i::Int, j::Int)

pysd_invert_matrix(m::AbstractArray, n) = vec(inv(reshape(m, Int(n), Int(n))))
pysd_invert_matrix(m, n) = m

pysd_elmcount(n) = Float64(n)

pysd_power(x, y) = abs(x) ^ y * sign(x)
@register_symbolic pysd_power(x::Real, y::Real)

struct SafeArray{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    data::A
end
Base.size(s::SafeArray) = size(s.data)
Base.getindex(s::SafeArray{T,1}, i::Integer) where T =
    checkbounds(Bool, s.data, i) ? s.data[i] : zero(T)
Base.getindex(s::SafeArray{T,2}, i::Integer, j::Integer) where T =
    checkbounds(Bool, s.data, i, j) ? s.data[i, j] : zero(T)
Base.getindex(s::SafeArray{T}, idx::Integer...) where T =
    checkbounds(Bool, s.data, idx...) ? s.data[idx...] : zero(T)
function Base.setindex!(s::SafeArray, v, idx...)
    checkbounds(Bool, s.data, idx...) && (s.data[idx...] = v)
    return v
end
pysd_safe(x::AbstractArray) = SafeArray(x)
pysd_safe(x) = x

pysd_quantum(a, b) = ifelse(b < 1e-6, float(a), b * pysd_trunc(a / b))
@register_symbolic pysd_quantum(a::Real, b::Real)

pysd_pi() = Base.MathConstants.pi

# XMILE variants: Xpulse has (start, magnitude), Xramp has (slope, start)
pysd_xpulse(t_now, start, magnitude) =
    ifelse((t_now >= start) & (t_now < start + magnitude), magnitude, 0.0)

pysd_xpulse_train(t_now, start, interval, magnitude) =
    ifelse((t_now >= start) &
           (mod(t_now - start, interval) < magnitude), magnitude, 0.0)

pysd_xramp(t_now, slope, start_time) =
    slope * max(0.0, t_now - start_time)

# Random functions — opaque wrappers so MTK calls them at every timestep
pysd_random_0_1() = Base.rand()
@register_symbolic pysd_random_0_1()

pysd_random_uniform(lo, hi, _seed) = lo + (hi - lo) * Base.rand()
@register_symbolic pysd_random_uniform(lo::Real, hi::Real, _seed::Real)

function pysd_random_normal(lo, hi, mean, std, _seed)
    x = mean + std * Base.randn()
    return clamp(x, lo, hi)
end
@register_symbolic pysd_random_normal(lo::Real, hi::Real, mean::Real, std::Real, _seed::Real)

function pysd_random_exponential(lo, hi, mean, _seed)
    x = lo + mean * Base.randexp()
    return clamp(x, lo, hi)
end
@register_symbolic pysd_random_exponential(lo::Real, hi::Real, mean::Real, _seed::Real)

# Vector operations
function pysd_vector_select(sel_vec, expr_vec, miss_val, action)
    selected = [expr_vec[i] for i in eachindex(sel_vec) if sel_vec[i] != 0]
    isempty(selected) && return miss_val
    action == 0 && return selected[1]
    action == 1 && return sum(selected)
    action == 2 && return maximum(selected)
    action == 3 && return minimum(selected)
    action == 4 && return sum(selected) / length(selected)
    return miss_val
end

pysd_vector_sort_order(vec, dir) =
    Float64.(ifelse(dir > 0, sortperm(vec), sortperm(vec, rev=true)))

pysd_vector_reorder(vec, order) = vec[Int.(order)]

pysd_vector_rank(vec, dir) =
    Float64.(invperm(ifelse(dir > 0, sortperm(vec), sortperm(vec, rev=true))))

pysd_get_time_value(t_now, lookup_fn, lo, hi) =
    lookup_fn(clamp(t_now, lo, hi))
