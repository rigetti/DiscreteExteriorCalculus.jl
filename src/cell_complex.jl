import Base: push!, append!, show
using StaticArrays: SVector
using UniqueVectors: UniqueVector
using Combinatorics: combinations
using SparseArrays: sparse
using MatrixNetworks: scomponents, bfs

################################################################################
# Cell
################################################################################

Cell(points::AbstractVector{Point{N}}, K::Int) where N = Cell{N}(Cell{N}[],
    Dict{Cell{N}, Bool}(), Vector(points), K)

Cell(s::Simplex{N, K}) where {N, K} = Cell(s.points, K)

show(io::IO, c::Cell{N}) where N = print(io,
    "Cell{$N,$(c.K)}(children: $(length(c.children)), " *
    "parents: $(length(c.parents)), points: $(c.points))")

function parent!(c::Cell{N}, p::Cell{N}, o::Bool) where N
    @assert p.K == c.K+1
    c.parents[p] = o
    push!(p.children, c)
    return c
end

################################################################################
# CellComplex
################################################################################

CellComplex{N,K}() where {N, K} = CellComplex{N,K}(
    SVector{K}([UniqueVector(Cell{N}[]) for _ in 1:K]))

CellComplex{N,K}(cells::AbstractVector{Cell{N}}) where {N, K} =
    append!(CellComplex{N,K}(), cells)

CellComplex(simplices::AbstractVector{Simplex{N, K}}) where {N, K} =
    TriangulatedComplex(simplices).complex

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

# check if a CellComplex is a simplicial complex
export simplicial
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
export pairwise_delaunay
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
export pairwise_noncocyclic
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
export one_sided
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
export well_centered
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

export connected_components
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

export boundary
function boundary(cells::AbstractVector{Cell{N}}) where N
    children = unique(vcat([c.children for c in cells]...))
    return filter(c -> length(c.parents) < 2, children)
end

boundary(comp::CellComplex) = CellComplex(boundary(comp.cells[end]))

# find the interior boundaries and the exterior boundary assuming that the cells
# form a connected manifold
export boundary_components_connected
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
export boundary_components
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
export orient!
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

# find the subcomplex that includes the given simplices
export subcomplex
function subcomplex(comp::CellComplex{N}, simplices::AbstractVector{Simplex{N, K}}) where {N, K}
    simplex_sets = [Set(s.points) for s in simplices]
    return CellComplex(filter(c -> Set(c.points) in simplex_sets, comp.cells[K]))
end
