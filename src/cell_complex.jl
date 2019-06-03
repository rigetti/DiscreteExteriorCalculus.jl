import Base: push!, append!, show
using StaticArrays: SVector
using UniqueVectors: UniqueVector
using Combinatorics: combinations
using LinearAlgebra: det
using SparseArrays: sparse
using MatrixNetworks: scomponents, bfs

export Cell, CellComplex

################################################################################
# Cell
################################################################################

# Cells represent polytopes of arbitrary dimension e.g. points, line segments,
# triangles, tetrahedra. A Cell does not necessarily refer to a simplex.

# NOTE: Cell has an inner constructor that enforces the heirarchy of dimension.
# If the parents field is only modified thereafter using parent! the heirarchy
# will be maintained.

struct Cell{N} # (K-1) dimensional cell in ℝᴺ
    children::Vector{Cell{N}} # (K-2) dimensional cells
    parents::Dict{Cell{N}, Bool} # K dimensional cells => relative orientations
    points::Vector{Point{N}} # all vertices of the cell
    K::Int # dimension + 1
    function Cell{N}(children::Vector{Cell{N}}, parents::Dict{Cell{N}, Bool},
                     points::Vector{Point{N}}, K::Int) where N
        @assert 1 <= K <= N+1
        for c in children
            @assert c.K == K-1
        end
        for c in keys(parents)
            @assert c.K == K+1
        end
        return new{N}(children, parents, points, K)
    end
end
show(io::IO, c::Cell{N}) where N = print(io,
    "Cell{$N,$(c.K)}(children: $(length(c.children)), " *
    "parents: $(length(c.parents)), points: $(c.points))")

Cell(points::AbstractVector{Point{N}}, K::Int) where N = Cell{N}(Cell{N}[],
    Dict{Cell{N}, Bool}(), Vector(points), K)
Cell(s::Simplex{N, K}) where {N, K} = Cell(s.points, K)

# construct a Simplex from a Cell in the case that the cell
# has the right number of points
function Simplex(c::Cell)
    @assert length(c.points) == c.K
    return Simplex(c.points)
end

function parent!(c::Cell{N}, p::Cell{N}, o::Bool) where N
    @assert p.K == c.K+1
    c.parents[p] = o
    push!(p.children, c)
    return c
end

################################################################################
# CellComplex
################################################################################

# A CellComplex is a collection of cells organized by dimension and ordered
# within a dimension.

# NOTE: the cells of a CellComplex are graded so that cells c in cells[K]
# have c.K == K. This is enforced by the inner constructor of CellComplex.
# If cells are only added to a CellComplex thereafter with the push! and
# append! methods defined below this property will be maintained.

struct CellComplex{N, K}
    cells::SVector{K, UniqueVector{Cell{N}}}
    function CellComplex{N,K}(cells::SVector{K, UniqueVector{Cell{N}}}) where {N, K}
        @assert 1 <= K <= N+1
        for k in 1:K
            for c in cells[k]
                @assert c.K == k
            end
        end
        return new{N,K}(cells)
    end
end
show(io::IO, comp::CellComplex{N,K}) where {N, K} = print(
     io, "CellComplex{$N,$K}$(tuple(map(length, comp.cells)...))")

CellComplex{N,K}() where {N, K} = CellComplex{N,K}(
    SVector{K}([UniqueVector(Cell{N}[]) for _ in 1:K]))

CellComplex{N,K}(cells::AbstractVector{Cell{N}}) where {N, K} =
    append!(CellComplex{N,K}(), cells)

function push!(comp::CellComplex{N,K}, c::Cell{N}) where {N, K}
    @assert c.K <= K
    if !(c in comp.cells[c.K])
        push!(comp.cells[c.K], c)
    end
    return comp
end

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

# Create a cell complex with the given cells and all of their descendants
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

# check if a CellComplex is a simplicial complex
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

# check if all pairs of cells sharing a face are pairwise delaunay.
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

# check if all pairs of cells sharing a face do not share a circumsphere
# after being rotated onto the same affine space
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

# check if all boundary cells have their circumcenter on the same side of
# the boundary as their apex
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

# check if all simplices in a simplicial complex contain their circumcenters
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

################################################################################
# Components and boundary
################################################################################

# find adjacency matrix for cells where two cells are considered adjacent if
# they share a child. Additionally return the mapping from pairs of cells
# to common faces.
function adjacency(cells::AbstractVector{Cell{N}}) where N
    function common_face_index(c1, c2)
        for (i, c) in enumerate(c1.children)
            if c2 in keys(c.parents)
                return i
            end
        end
        return 0
    end
    faces = Dict{Tuple{Cell{N}, Cell{N}}, Cell{N}}()
    row_inds, col_inds, vals = Int[], Int[], Int[]
    for (i1, i2) in combinations(1:length(cells), 2)
        c1, c2 = cells[i1], cells[i2]
        i = common_face_index(c1, c2)
        if i > 0
            push!(row_inds, i1); push!(col_inds, i2); push!(vals, 1)
            push!(row_inds, i2); push!(col_inds, i1); push!(vals, 1)
            faces[(c1, c2)] = faces[(c2, c1)] = c1.children[i]
        end
    end
    return sparse(row_inds, col_inds, vals, length(cells), length(cells)), faces
end

function connected_components(cells::AbstractVector{Cell{N}}) where N
    adj, _ = adjacency(cells)
    cc = scomponents(adj)
    components = [Cell{N}[] for i in 1:cc.number]
    for (i, c) in enumerate(cells)
        push!(components[cc.map[i]], c)
    end
    return components
end

connected_components(comp::CellComplex) = connected_components(comp.cells[end])

function boundary(cells::AbstractVector{Cell{N}}) where N
    children = unique(vcat([c.children for c in cells]...))
    return filter(c -> length(c.parents) < 2, children)
end

boundary(comp::CellComplex) = CellComplex(boundary(comp.cells[end]))

# find the interior boundaries and the exterior boundary assuming that the cells
# form a connected manifold
function boundary_components_connected(cells::AbstractVector{Cell{N}}) where N
    boundary_comps = connected_components(boundary(cells))
    # identify the exterior boundary by finding which component contains
    # a point with maximal first coordinate
    exterior_ind = argmax([maximum([maximum(p.coords[1] for p in f.points)
        for f in component]) for component in boundary_comps])
    interior_inds = collect(1:length(boundary_comps))
    deleteat!(interior_inds, exterior_ind)
    return boundary_comps[interior_inds], boundary_comps[exterior_ind]
end

function boundary_components_connected(comp::CellComplex)
    interiors, exterior = boundary_components_connected(comp.cells[end])
    return map(CellComplex, interiors), CellComplex(exterior)
end

# find the interior boundaries and the exterior boundary without assuming that
# the cells form a connected manifold
function boundary_components(cells::AbstractVector{Cell{N}}) where N
    interior_boundaries, exterior_boundaries = Vector{Cell{N}}[], Vector{Cell{N}}[]
    for component in connected_components(cells)
        ibs, e = boundary_components_connected(component)
        append!(interior_boundaries, ibs)
        push!(exterior_boundaries, e)
    end
    return interior_boundaries, exterior_boundaries
end

function boundary_components(comp::CellComplex)
    interiors, exteriors = boundary_components(comp.cells[end])
    return map(CellComplex, interiors), map(CellComplex, exteriors)
end

################################################################################
# Orienting manifolds
################################################################################

# reverse the orientation of a cell
function change_orientation!(c::Cell)
    if length(c.points) >= 2
        a = c.points[1]
        c.points[1] = c.points[2]
        c.points[2] = a
    end
    for parent in keys(c.parents)
        c.parents[parent] = !c.parents[parent]
    end
    for child in c.children
        child.parents[c] = !child.parents[c]
    end
    return c
end

# Produce a consistent orientation on a list of cells, in the sense that any
# two cells that share a face have opposite relative orientations to that face.
# If the cells form a non-orientable manifold (e.g. a mobius strip) the function
# will nonetheless complete, but the manifold will not be oriented.
# For an orientable manifold, all cells of dimension N will become positively
# oriented, and all boundary cells of dimension N-1 will become oriented
# consistently with their parent.
function orient!(cells::AbstractVector{Cell{N}}) where N
    adj, faces = adjacency(cells)
    cc = scomponents(adj)
    roots = [findfirst(isequal(i), cc.map) for i in 1:cc.number]
    # start with higest dimensional component
    for i in reverse(sortperm([cells[r].K for r in roots]))
        orient_component!(cells, adj, faces, roots[i])
    end
    return cells
end

function orient_component!(cells::AbstractVector{Cell{N}},
    adj::AbstractMatrix{<:Real},
    faces::Dict{Tuple{Cell{N}, Cell{N}}, Cell{N}}, root::Int) where N
    orient!(cells[root])
    dists, _, predecessors = bfs(adj, root)
    for j in sortperm(dists)
        if dists[j] > 0 # exclude unreachables and the root
            c1, c2 = cells[predecessors[j]], cells[j]
            f = faces[(c1, c2)]
            if f.parents[c1] == f.parents[c2]
                change_orientation!(c2)
            end
        end
    end
    return cells
end

# If the cell is of maximum dimension and is simplicial, orient it positively.
# If the cell is a boundary cell, orient it accoring to its parent.
# Otherwise leave its orientation as it is.
function orient!(cell::Cell{N}) where N
    num_parents = length(cell.parents)
    if (num_parents == 0) && (cell.K == length(cell.points) == N + 1)
        if orientation(Simplex(cell)) < 0
            change_orientation!(cell)
        end
    elseif (num_parents == 1) && (!(collect(values(cell.parents))[1]))
        change_orientation!(cell)
    end
    return cell
end

# to orient a CellComplex with K == N + 1 it is much faster to simply
# positively orient each cell than to perform the tree algorithm.
function orient!(comp::CellComplex{N, K}) where {N, K}
    if K == N + 1
        map(orient!, comp.cells[end])
    else
        orient!(comp.cells[end])
    end
    return comp
end

# find the submanifold that includes the given simplices
function submanifold(comp::CellComplex{N}, simplices::AbstractVector{Simplex{N, K}}) where {N, K}
    simplex_sets = [Set(s.points) for s in simplices]
    return CellComplex(filter(c -> Set(c.points) in simplex_sets, comp.cells[K]))
end
