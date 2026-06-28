# Excel data reader helpers for Vensim GET DIRECT CONSTANTS / LOOKUPS / DATA.
#
# Supports three reference modes matching the Vensim conventions:
#   - Named ranges  (e.g. cell = "my_param", x = "year_range")
#   - Cell refs     (e.g. cell = "B2")
#   - Row/column    (e.g. x = "4" for row 4, x = "A" for column A)

const _XLSX_CACHE = Dict{String, XLSX.XLSXFile}()

function _xlsx_open(path::String)
    p = abspath(path)
    get!(() -> XLSX.readxlsx(p), _XLSX_CACHE, p)
end

function _xlsx_is_cell_ref(s::String)
    return match(r"^[A-Za-z]{1,3}[0-9]+$", s) !== nothing
end

function _xlsx_col_to_num(col::AbstractString)
    n = 0
    for c in uppercase(col)
        n = n * 26 + (Int(c) - Int('A') + 1)
    end
    return n
end

function _xlsx_num_to_col(n::Int)
    s = ""
    while n > 0
        n, r = divrem(n - 1, 26)
        s = Char('A' + r) * s
    end
    return s
end

function _xlsx_resolve_range(xf::XLSX.XLSXFile, sheet::String, name::String)
    if _xlsx_is_cell_ref(name)
        return sheet * "!" * name
    end
    wb = xf.workbook
    if haskey(wb.workbook_names, name)
        return string(wb.workbook_names[name].value)
    end
    sheet_lower = lowercase(sheet)
    sheet_idx = nothing
    for (i, s) in enumerate(XLSX.sheetnames(xf))
        if lowercase(s) == sheet_lower
            sheet_idx = i
            break
        end
    end
    fallback = nothing
    for ((idx, n), dn) in wb.worksheet_names
        if n == name
            ref = string(dn.value)
            occursin('!', ref) || (ref = sheet * "!" * ref)
            if sheet_idx !== nothing && idx == sheet_idx
                return ref
            end
            fallback === nothing && (fallback = ref)
        end
    end
    fallback !== nothing && return fallback
    error("Named range '" * name * "' not found in " * string(xf.source))
end

function _xlsx_split_ref(ref::String)
    parts = split(ref, '!')
    return String(parts[1]), String(parts[2])
end

function _to_float(x)
    x === nothing && return NaN
    ismissing(x) && return NaN
    x isa Number && return Float64(x)
    return NaN
end

function _to_float64_vec(data)
    v = vec(data isa Matrix ? data : reshape([data], 1, 1))
    return Float64[_to_float(x) for x in v]
end

"""
    pysd_xlsx_read_constant(path, sheet, name; transpose=false)

Read a scalar or array constant from an Excel named range or cell reference.
The `name` may end with `*` to indicate transposition (Vensim convention).
"""
function pysd_xlsx_read_constant(path::String, sheet::String, name::String;
                                 transpose::Bool=false, scalar::Bool=false)
    xf = _xlsx_open(path)
    ref = _xlsx_resolve_range(xf, sheet, name)
    sname, cells = _xlsx_split_ref(ref)
    data = xf[sname][cells]
    if data isa Matrix
        transpose && (data = permutedims(data))
        if scalar || length(data) == 1
            return _to_float(data[1])
        end
        nr, nc = size(data)
        if nc == 1
            return Float64[_to_float(data[i, 1]) for i in 1:nr]
        elseif nr == 1
            return Float64[_to_float(data[1, j]) for j in 1:nc]
        end
        return Float64[_to_float(data[i, j]) for i in 1:nr, j in 1:nc]
    end
    return _to_float(data)
end

"""
    pysd_xlsx_read_constant(path, sheet, names::Vector; transpose=false)

Read multiple named ranges from Excel and concatenate them into a single array.
Each element of `names` is either a `String` (range name to read from Excel)
or a `Vector{Float64}` (literal values to insert directly).
"""
function pysd_xlsx_read_constant(path::String, sheet::String, names::Vector;
                                 transpose::Bool=false, dims::Tuple=())
    parts = Float64[]
    for spec in names
        if spec isa String
            v = pysd_xlsx_read_constant(path, sheet, spec; transpose=transpose)
            if v isa AbstractArray
                append!(parts, vec(v))
            else
                push!(parts, Float64(v))
            end
        elseif spec isa AbstractVector
            append!(parts, Float64.(spec))
        elseif spec isa Number
            push!(parts, Float64(spec))
        end
    end
    if !isempty(dims) && length(dims) >= 2
        return reshape(parts, dims...)
    end
    return parts
end

"""
    pysd_xlsx_read_series(path, sheet, x_name, y_names::Vector{String})

Read multiple y-series from Excel sharing the same x-axis.
Returns `(xs, ys_list)` where `ys_list` is a `Vector{Vector{Float64}}`.
"""
function pysd_xlsx_read_series(path::String, sheet::String,
                               x_name::String, y_names::Vector{String})
    xs = nothing
    ys_list = Vector{Float64}[]
    for y_name in y_names
        xi, yi = pysd_xlsx_read_series(path, sheet, x_name, y_name)
        xs === nothing && (xs = xi)
        push!(ys_list, yi)
    end
    return xs, ys_list
end

"""
    pysd_xlsx_build_lookup_dispatch(path, sheet, x_name, y_name)

Read a (possibly 2D) lookup from Excel and return a vector of interpolation
functions, one per column of the y data.  For 1D data returns a single-element
vector.
"""
function pysd_xlsx_build_lookup_dispatch(path::String, sheet::String,
                                         x_name::String, y_name::String)
    xf = _xlsx_open(path)
    x_ref = _xlsx_resolve_range(xf, sheet, x_name)
    y_ref = _xlsx_resolve_range(xf, sheet, y_name)
    x_sname, x_cells = _xlsx_split_ref(x_ref)
    y_sname, y_cells = _xlsx_split_ref(y_ref)
    x_data = xf[x_sname][x_cells]
    y_data = xf[y_sname][y_cells]
    xs = _to_float64_vec(x_data)
    if y_data isa Matrix
        nr, nc = size(y_data)
        if nr == length(xs)
            return [LinearInterpolation(
                        Float64[_to_float(y_data[i, j]) for i in 1:nr], xs;
                        extrapolation_left=ExtrapolationType.Constant,
                        extrapolation_right=ExtrapolationType.Constant)
                    for j in 1:nc]
        elseif nc == length(xs)
            return [LinearInterpolation(
                        Float64[_to_float(y_data[i, j]) for j in 1:nc], xs;
                        extrapolation_left=ExtrapolationType.Constant,
                        extrapolation_right=ExtrapolationType.Constant)
                    for i in 1:nr]
        end
    end
    ys = _to_float64_vec(y_data)
    return [LinearInterpolation(ys, xs;
                extrapolation_left=ExtrapolationType.Constant,
                extrapolation_right=ExtrapolationType.Constant)]
end

"""
    pysd_xlsx_read_series(path, sheet, x_row_or_col, y_cell)

Read x/y series data from Excel for lookups or time-series data.
Handles three Vensim reference modes:
- **Row mode**: `x_row_or_col` is a number (row), `y_cell` is a cell ref
- **Column mode**: `x_row_or_col` is a column letter, `y_cell` is a cell ref
- **Name mode**: both are named ranges
Returns `(xs::Vector{Float64}, ys::Vector{Float64})`.
"""
function pysd_xlsx_read_series(path::String, sheet::String,
                               x_row_or_col::String, y_cell::String)
    xf = _xlsx_open(path)
    ws = xf[sheet]
    x_is_row = all(isdigit, x_row_or_col)
    y_is_cell = _xlsx_is_cell_ref(y_cell)

    if x_is_row && y_is_cell
        row_num = parse(Int, x_row_or_col)
        m = match(r"^([A-Za-z]+)([0-9]+)$", y_cell)
        y_col = _xlsx_col_to_num(m[1])
        y_row = parse(Int, m[2])
        nr = XLSX.get_dimension(ws).stop.row_number
        nc = XLSX.get_dimension(ws).stop.column_number
        x_data = ws[XLSX.CellRef(row_num, y_col):XLSX.CellRef(row_num, nc)]
        xs_raw = vec(x_data)
        last_valid = findlast(v -> v !== nothing && !ismissing(v), xs_raw)
        last_valid === nothing && error("No x data found in row " * x_row_or_col)
        ncols = last_valid
        xs = Float64[_to_float(xs_raw[i]) for i in 1:ncols]
        y_data = ws[XLSX.CellRef(y_row, y_col):XLSX.CellRef(y_row, y_col + ncols - 1)]
        ys = _to_float64_vec(y_data)
        return xs, ys
    elseif !x_is_row && y_is_cell && !_xlsx_is_cell_ref(x_row_or_col)
        x_col = _xlsx_col_to_num(x_row_or_col)
        m = match(r"^([A-Za-z]+)([0-9]+)$", y_cell)
        y_col = _xlsx_col_to_num(m[1])
        y_row = parse(Int, m[2])
        nr = XLSX.get_dimension(ws).stop.row_number
        x_data = ws[XLSX.CellRef(y_row, x_col):XLSX.CellRef(nr, x_col)]
        xs_raw = vec(x_data)
        last_valid = findlast(v -> v !== nothing && !ismissing(v), xs_raw)
        last_valid === nothing && error("No x data found in column " * x_row_or_col)
        nrows = last_valid
        xs = Float64[_to_float(xs_raw[i]) for i in 1:nrows]
        y_data = ws[XLSX.CellRef(y_row, y_col):XLSX.CellRef(y_row + nrows - 1, y_col)]
        ys = _to_float64_vec(y_data)
        return xs, ys
    else
        x_ref = _xlsx_resolve_range(xf, sheet, x_row_or_col)
        y_ref = _xlsx_resolve_range(xf, sheet, y_cell)
        x_sname, x_cells = _xlsx_split_ref(x_ref)
        y_sname, y_cells = _xlsx_split_ref(y_ref)
        x_data = xf[x_sname][x_cells]
        y_data = xf[y_sname][y_cells]
        xs = _to_float64_vec(x_data)
        if y_data isa Matrix
            nr, nc = size(y_data)
            n = length(xs)
            if nr == n
                ys = Float64[_to_float(y_data[i, 1]) for i in 1:nr]
            elseif nc == n
                ys = Float64[_to_float(y_data[1, i]) for i in 1:nc]
            else
                ys = _to_float64_vec(y_data)
            end
        else
            ys = Float64[_to_float(y_data)]
        end
        return xs, ys
    end
end
