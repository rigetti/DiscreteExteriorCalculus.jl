using SparseArrays: sparse
using MatrixNetworks: scomponents

# find adjacency matrix for cells where two cells are considered adjacent if
# they share a child. Additionally return the mapping from pairs of cells
# to common faces.
export adjacency
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
