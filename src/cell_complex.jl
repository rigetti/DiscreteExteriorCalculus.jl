using StaticArrays: SVector
using UniqueVectors: UniqueVector
using Combinatorics: combinations

Cell(points::AbstractVector{Point{N}}, K::Int) where N = Cell{N}(Cell{N}[],
    Dict{Cell{N}, Bool}(), Vector(points), K)

Cell(s::Simplex{N, K}) where {N, K} = Cell(s.points, K)

import Base: show
show(io::IO, c::Cell{N}) where N = print(io,
    "Cell{$N,$(c.K)}(children: $(length(c.children)), " *
    "parents: $(length(c.parents)), points: $(c.points))")

CellComplex{N,K}() where {N, K} = CellComplex{N,K}(
    SVector{K}([UniqueVector(Cell{N}[]) for _ in 1:K]))

CellComplex{N,K}(cells::AbstractVector{Cell{N}}) where {N, K} =
    append!(CellComplex{N,K}(), cells)

CellComplex(simplices::AbstractVector{Simplex{N, K}}) where {N, K} =
    TriangulatedComplex(simplices).complex

"""
    CellComplex(cells::AbstractVector{Cell{N}}) where N

Create a cell complex consisting of `cells` and all descendants of `cells`.
"""
function CellComplex(cells::AbstractVector{Cell{N}}) where N
    K = maximum(c.K for c in cells)
    comp = CellComplex{N, K}(cells)
    for k in reverse(1:K)
        for c in comp.cells[k]
            append!(comp, c.children)
        end
    end
    return comp
end

show(io::IO, comp::CellComplex{N,K}) where {N, K} = print(
     io, "CellComplex{$N,$K}$(tuple(map(length, comp.cells)...))")

export parent!
"""
    parent!(c::Cell{N}, p::Cell{N}, o::Bool) where N

Make `p` a parent of `c` with relative orientation `o` and make `c` a child of `p`.
Return c.
"""
function parent!(c::Cell{N}, p::Cell{N}, o::Bool) where N
    @assert p.K == c.K+1
    c.parents[p] = o
    push!(p.children, c)
    return c
end

import Base: push!
"""
    push!(comp::CellComplex{N,K}, c::Cell{N}) where {N, K}

Add a Cell to a CellComplex unless it is already in the CellComplex. Return comp.
"""
function push!(comp::CellComplex{N,K}, c::Cell{N}) where {N, K}
    @assert c.K <= K
    if !(c in comp.cells[c.K])
        push!(comp.cells[c.K], c)
    end
    return comp
end

import Base: append!
"""
    append!(comp::CellComplex{N}, cells::AbstractVector{Cell{N}}) where N
    append!(comp1::CellComplex{N, K}, comp2::CellComplex{N, J}) where {N, K, J}

`push!` all cells in an array or a cell complex into another cell complex. Return the first
cell complex.
"""
function append!(comp::CellComplex{N}, cells::AbstractVector{Cell{N}}) where N
    for c in cells
        push!(comp, c)
    end
    return comp
end

function append!(comp1::CellComplex{N, K}, comp2::CellComplex{N, J}) where {N, K, J}
    @assert 1 <= J <= K
    for cells in comp2.cells
        append!(comp1, cells)
    end
    return comp1
end

export subcomplex
"""
    subcomplex(comp::CellComplex{N}, simplices::AbstractVector{Simplex{N, K}}) where {N, K}

Find the subcomplex of the given cell complex that includes the given simplices.
"""
function subcomplex(comp::CellComplex{N},
    simplices::AbstractVector{Simplex{N, K}}) where {N, K}
    simplex_sets = [Set(s.points) for s in simplices]
    return CellComplex(filter(c -> Set(c.points) in simplex_sets, comp.cells[K]))
end

export simplicial
"""
    simplicial(comp::CellComplex{N, K}) where {N, K}

Return `true` if every cell of `comp` is a simplex, else `false`.
"""
function simplicial(comp::CellComplex{N, K}) where {N, K}
    for k in 1:K
        for cell in comp.cells[k]
            if !(length(cell.points) == cell.K == k)
                return false
            end
        end
    end
    return true
end

export pairwise_delaunay
"""
    pairwise_delaunay(m::Metric{N}, comp::CellComplex{N, K}, ϵ::Real=0.0) where {N, K}

Return `true` if every pair of cells sharing a face are within `ϵ` of pairwise delaunay,
else `false`.
"""
function pairwise_delaunay(m::Metric{N}, comp::CellComplex{N, K},
                           ϵ::Real=0.0) where {N, K}
    for k in 1:K
        for cell in comp.cells[k]
            for (c1, c2) in combinations(collect(keys(cell.parents)), 2)
                s1, s2 = Simplex(c1), Simplex(c2)
                if !(pairwise_delaunay(m, s1, s2) > ϵ &&
                     pairwise_delaunay(m, s2, s1) > ϵ)
                    return false
                end
            end
        end
    end
    return true
end

export one_sided
"""
    one_sided(m::Metric{N}, comp::CellComplex{N, K}, ϵ::Real=0.0) where {N, K}

Return `true` if every boundary cell has its circumcenter inside the cell complex up to `ϵ`.
"""
function one_sided(m::Metric{N}, comp::CellComplex{N, K},
    ϵ::Real=0.0) where {N, K}
    for k in 1:K
        for cell in comp.cells[k]
            if length(cell.parents) == 1 # boundary parent
                s = Simplex(collect(keys(cell.parents))[1])
                center, _ = circumsphere_barycentric(m, s)
                p = first_setdiff(s.points, cell.points)
                i = findfirst(isequal(p), s.points)
                if center.coords[i] <= ϵ
                    return false
                end
            end
        end
    end
    return true
end

export pairwise_noncocyclic
"""
    pairwise_noncocyclic(m::Metric{N}, comp::CellComplex{N, K}, ϵ::Real=0.0) where {N, K}

Return `true` if every pair of cells sharing a face are more than `ϵ` away from sharing
a circumsphere, else `false`.
"""
function pairwise_noncocyclic(m::Metric{N}, comp::CellComplex{N, K},
    ϵ::Real=0.0) where {N, K}
    for k in 1:K
        for cell in comp.cells[k]
            for (c1, c2) in combinations(collect(keys(cell.parents)), 2)
                s1, s2 = Simplex(c1), Simplex(c2)
                if (ϵ >= pairwise_delaunay(m, s1, s2) >= -ϵ) ||
                    (ϵ >= pairwise_delaunay(m, s2, s1) >= -ϵ)
                    return c1, c2
                end
            end
        end
    end
    return true
end

export well_centered
"""
    well_centered(m::Metric{N}, comp::CellComplex{N, K}, ϵ::Real=0.0) where {N, K}

Return `true` if every cell contains is circumcenter, within `ϵ`, else `false`.
"""
function well_centered(m::Metric{N}, comp::CellComplex{N, K},
    ϵ::Real=0.0) where {N, K}
    for k in 1:K
        for cell in comp.cells[k]
            s = Simplex(cell)
            center, _ = circumsphere_barycentric(m, s)
            if any(center.coords .<= ϵ)
                return false
            end
        end
    end
    return true
end
