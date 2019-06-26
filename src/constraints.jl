using SparseArrays: sparse

export constant_constraint
"""
    constant_constraint(comp::CellComplex{N},
        cells::AbstractVector{Cell{N}}, k::Int) where N

Find a sparse representation of the constraint that a `k-1` form is constant on given
`k-1` dimensional cells.
"""
function constant_constraint(comp::CellComplex{N},
    cells::AbstractVector{Cell{N}}, k::Int) where N
    row_inds, col_inds, vals = Int[], Int[], Float64[]
    for (row_ind, (c1, c2)) in enumerate(zip(cells[1:end-1], cells[2:end]))
        col1 = findfirst(isequal(c1), comp.cells[k])
        col2 = findfirst(isequal(c2), comp.cells[k])
        push!(row_inds, row_ind); push!(col_inds, col1); push!(vals, 1)
        push!(row_inds, row_ind); push!(col_inds, col2); push!(vals, -1)
    end
    return sparse(row_inds, col_inds, vals, length(cells)-1, length(comp.cells[k]))
end

export zero_constraint
"""
    zero_constraint(comp::CellComplex{N}, cells::AbstractVector{Cell{N}}, k::Int) where N

Find a sparse representation of the constraint that a `k-1` form is zero on given `k-1`
dimensional cells.
"""
function zero_constraint(comp::CellComplex{N}, cells::AbstractVector{Cell{N}},
    k::Int) where N
    col_inds = [findfirst(isequal(c), comp.cells[k]) for c in cells]
    num_rows = length(col_inds)
    return sparse(1:num_rows, col_inds, ones(num_rows), num_rows, length(comp.cells[k]))
end
