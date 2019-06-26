using SparseArrays: spdiagm, sparse, spzeros, SparseMatrixCSC
using LinearAlgebra: diag, I

export differential_operator_sequence
"""
    differential_operator_sequence(m::Metric{N}, mesh::Mesh{N, K}, expr::String,
        k::Int, primal::Bool) where {N, K}

Compute the differential operators defined by the string `expr`. This string must consist
of the characters `d`, `★`, `δ`, and `Δ` indicating the exterior derivative, hodge dual,
codifferential, and Laplace-de Rham operators, respectively. If there are two `★` operators
in a row, a single `★★` operator is computed since this avoids unnecessary calculation.
The dimension of the form on which the operator acts is `k-1` and its primality or duality
is indicated by `primal`.
"""
function differential_operator_sequence(m::Metric{N}, mesh::Mesh{N, K}, expr::String,
    k::Int, primal::Bool) where {N, K}
    ops = SparseMatrixCSC{Float64,Int64}[]
    chars = collect(expr)
    i = length(chars)
    while i > 0
        char = chars[i]
        @assert char in ['d', '★', 'δ', 'Δ']
        if char == 'd'
            comp = primal ? mesh.primal.complex : mesh.dual.complex
            push!(ops, exterior_derivative(comp, k))
            k += 1
        elseif char == '★'
            if (i > 1) && chars[i-1] == '★'
                push!(ops, hodge_square(m, mesh, k, primal))
                i -= 1
            else
                push!(ops, circumcenter_hodge(m, mesh, k, primal))
                primal = !primal
                k = K-k+1
            end
        elseif char == 'δ'
            push!(ops, codifferential(m, mesh, k, primal))
            k -= 1
        elseif char == 'Δ'
            push!(ops, laplace_de_Rham(m, mesh, k, primal))
        end
        i -= 1
    end
    return reverse(ops)
end

export differential_operator
"""
    differential_operator(m::Metric{N}, mesh::Mesh{N}, expr::String, k::Int,
        primal::Bool) where N

Compute the differential operator defined by the string `expr`. This string must consist
of the characters `d`, `★`, `δ`, and `Δ` indicating the exterior derivative, hodge dual,
codifferential, and Laplace-de Rham operators, respectively. The dimension of the form on
which the operator acts is `k-1` and its primality or duality is indicated by `primal`.
"""
function differential_operator(m::Metric{N}, mesh::Mesh{N}, expr::String, k::Int,
    primal::Bool) where N
    ops = differential_operator_sequence(m, mesh, expr, k, primal)
    return reduce(*, ops)
end

"""
    differential_operator(m::Metric{N}, mesh::Mesh{N}, expr::String, k::Int,
        primal::Bool, v::AbstractVector{<:Real}) where N

Apply the differential operator defined by the string `expr` to the vector `v`.
"""
function differential_operator(m::Metric{N}, mesh::Mesh{N}, expr::String, k::Int,
    primal::Bool, v::AbstractVector{<:Real}) where N
    ops = differential_operator_sequence(m, mesh, expr, k, primal)
    return foldr(*, ops, init=v)
end

"""
    circumcenter_hodge(m::Metric{N}, mesh::Mesh{N, K}, k::Int, primal::Bool) where {N, K}

Find the discrete hodge star operator using the circumcentric hodge star construction. If
`primal == true`, this operator takes primal `k-1` forms to dual `K-k+1` forms. Otherwise
it takes dual `k-1` forms to priaml `K-k+1` forms.
"""
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

"""
    hodge_square_sign(m::Metric, K::Int, k::Int)

There is an identity `★★ = sign(det(metric)) * (-1)^((k-1) * (K-k)) * I` for `k-1` forms in
`K-1` dimensions. Compute the coefficient of `I` in this expression.
"""
hodge_square_sign(m::Metric, K::Int, k::Int) = sign(det(collect(m.mat))) *
    (mod((k-1) * (K-k), 2) == 0 ? 1 : -1)

"""
    hodge_square(m::Metric{N}, mesh::Mesh{N, K}, k::Int, primal::Bool) where {N, K}

Compute `★★` without computing the `★` operators by using the identity
`★★ = sign(det(metric)) * (-1)^((k-1) * (K-k)) * I` for `k-1` forms in `K-1` dimensions.
"""
function hodge_square(m::Metric{N}, mesh::Mesh{N, K}, k::Int, primal::Bool) where {N, K}
    comp = primal ? mesh.primal.complex : mesh.dual.complex
    n = k < K+1 ? length(comp.cells[k]) : 0
    return hodge_square_sign(m, K, k) * sparse(I, n, n)
end

"""
    exterior_derivative(comp::CellComplex{N, K}, k::Int) where {N, K}

Find the discrete exterior derivative operator.
"""
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

"""
    codifferential(m::Metric{N}, mesh::Mesh{N, K}, k::Int, primal::Bool) where {N, K}

Compute the codifferential defined by
`δ = sign(det(collect(m.mat))) * (-1)^((K-1) * (k-2) + 1) * ★d★` for `k-1` forms in `K-1`
dimensions.
"""
function codifferential(m::Metric{N}, mesh::Mesh{N, K}, k::Int, primal::Bool) where {N, K}
    ★d★ = differential_operator(m, mesh, "★d★", k, primal)
    s = sign(det(collect(m.mat)))
    return s * (mod((K-1) * (k-2) + 1, 2) == 0 ? 1 : -1) * ★d★
end


"""
    laplace_de_Rham(m::Metric{N}, mesh::Mesh{N, K}, k::Int, primal::Bool) where {N, K}

Compute the Laplace-de Rham operator defined by `Δ = dδ + δd`.
"""
function laplace_de_Rham(m::Metric{N}, mesh::Mesh{N, K}, k::Int, primal::Bool) where {N, K}
    dδ = differential_operator(m, mesh, "dδ", k, primal)
    δd = differential_operator(m, mesh, "δd", k, primal)
    return dδ + δd
end

export sharp
"""
    sharp(m::Metric{N}, comp::CellComplex{N}, form::AbstractVector{<:Real}) where N

Given a 1-form on a cell complex, approximate a vector of length `N` at each vertex using
least squares.
"""
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
