using MatrixNetworks: scomponents, bfs
using LinearAlgebra: det

export orientation
"""
    orientation(s::Simplex{N, K}) where {N, K}

If `K == N+1`, compute the sign of the orientation of the simplex with respect to the
embedding space.
"""
function orientation(s::Simplex{N, K}) where {N, K}
    @assert K == N+1
    return sign(det(hcat(Wedge(s).vectors...)))
end

# reverse the orientation of a cell
export change_orientation!
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

export orient_component!
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
