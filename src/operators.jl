using SparseArrays: spdiagm, sparse, spzeros, SparseMatrixCSC
using LinearAlgebra: diag

# discrete circumcentric hodge star on (k-1)-forms
function circumcenter_hodge(m::Metric{N}, mesh::Mesh{N, K}, k::Int,
    primal::Bool) where {N, K}
    @assert 1 <= k <= K+1
    if k == K+1
        return spzeros(0,0)
    end
    if primal
        ratios = Float64[]
        for (p_cell, d_cell) in zip(mesh.primal.complex.cells[k],
                mesh.dual.complex.cells[K-k+1])
            p_vol = volume(m, mesh.primal, p_cell)
            d_vol = volume(m, mesh.dual, d_cell)
            @assert p_vol != 0
            push!(ratios, d_vol/p_vol)
        end
    else
        pm = hodge_square_sign(m, K, k)
        v = diag(circumcenter_hodge(m, mesh, K-k+1, true))
        @assert !any(v .== 0)
        ratios = pm ./ v
    end
    return spdiagm(0 => ratios)
end

# ★★ = sign(det(metric)) * (-1)^(k(K-k)) * I for k forms in K dimensions
export hodge_square_sign
function hodge_square_sign(m::Metric, K::Int, k::Int)
    return sign(det(collect(m.mat))) * (mod((k-1) * (K-k), 2) == 0 ? 1 : -1)
end

# exterior derivative taking (k-1)-forms to k-forms
function exterior_derivative(comp::CellComplex{N, K}, k::Int) where {N, K}
    @assert 0 <= k <= K
    if k == 0
        return spzeros(length(comp.cells[k+1]), 0)
    end
    row_inds, col_inds, vals = Int[], Int[], Int[]
    for (col_ind, cell) in enumerate(comp.cells[k])
        for p in keys(cell.parents)
            o = cell.parents[p]
            row_ind = findfirst(isequal(p), comp.cells[k+1])
            push!(row_inds, row_ind); push!(col_inds, col_ind); push!(vals, 2 * o - 1)
        end
    end
    num_rows = k+1 <= K ? length(comp.cells[k+1]) : 0
    num_cols = length(comp.cells[k])
    return sparse(row_inds, col_inds, vals, num_rows, num_cols)
end

# compute the differential operators indicated by the expression
# expr must be a string consisting of the characters 'd' and '★' (note ★ is
# \bigstar). k-1 indicates the dimension of the form the operator will act on
# while primal indicates whether the form is a primal or dual form
export differential_operator_sequence
function differential_operator_sequence(m::Metric{N}, mesh::Mesh{N, K},
    expr::String, k::Int, primal::Bool) where {N, K}
    ops = SparseMatrixCSC{Float64,Int64}[]
    for char in reverse(expr)
        @assert char in ['d', '★', 'δ', 'Δ']
        if char == 'd'
            comp = primal ? mesh.primal.complex : mesh.dual.complex
            push!(ops, exterior_derivative(comp, k))
            k += 1
        elseif char == '★'
            push!(ops, circumcenter_hodge(m, mesh, k, primal))
            primal = !primal
            k = K-k+1
        elseif char == 'δ'
            push!(ops, codifferential(m, mesh, k, primal))
            k -= 1
        elseif char == 'Δ'
            push!(ops, laplacian(m, mesh, k, primal))
        end
    end
    return reverse(ops)
end

export differential_operator
function differential_operator(m::Metric{N}, mesh::Mesh{N}, expr::String,
    k::Int, primal::Bool) where N
    ops = differential_operator_sequence(m, mesh, expr, k, primal)
    return reduce(*, ops)
end

function differential_operator(m::Metric{N}, mesh::Mesh{N}, expr::String,
    k::Int, primal::Bool, v::AbstractVector{<:Real}) where N
    ops = differential_operator_sequence(m, mesh, expr, k, primal)
    return foldr(*, ops, init=v)
end

function codifferential(m::Metric{N}, mesh::Mesh{N, K}, k::Int,
    primal::Bool) where {N, K}
    ★d★ = differential_operator(m, mesh, "★d★", k, primal)
    s = sign(det(collect(m.mat)))
    return s * (mod((K-1) * (k-2) + 1, 2) == 0 ? 1 : -1) * ★d★
end

function laplacian(m::Metric{N}, mesh::Mesh{N, K}, k::Int,
    primal::Bool) where {N, K}
    dδ = differential_operator(m, mesh, "dδ", k, primal)
    δd = differential_operator(m, mesh, "δd", k, primal)
    return dδ + δd
end

# Approximate a vector field from a 1-form using least squares
export sharp
function sharp(m::Metric{N}, comp::CellComplex{N}, form::AbstractVector{<:Real}) where N
    field = Vector{Float64}[]
    for c in comp.cells[1]
        mat = zeros(length(c.parents), N)
        w = zeros(length(c.parents))
        for (row_ind, e) in enumerate(collect(keys(c.parents)))
            mat[row_ind, :] = sum([x.points[1].coords * (2 * x.parents[e] - 1)
                for x in e.children])
            w[row_ind] = form[findfirst(isequal(e), comp.cells[2])]
        end
        push!(field, (mat * m.mat) \ w)
    end
    return field
end
