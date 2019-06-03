import Base: show

export TriangulatedComplex, Mesh

################################################################################
# TriangulatedComplex
################################################################################

# A CellComplex along with a decomposition of each cell into simplices.
# Each simplex comes with a boolean indicating if it should be considered to
# have positive or negative volume.
SignedSimplex{N} = Tuple{Vector{Point{N}}, Bool}
struct TriangulatedComplex{N, K}
    complex::CellComplex{N, K}
    simplices::Dict{Cell{N}, Vector{SignedSimplex{N}}}
end
show(io::IO, tcomp::TriangulatedComplex{N,K}) where {N, K} = print(
     io, "TriangulatedComplex{$N,$K}$(tuple(map(length, tcomp.complex.cells)...))")

TriangulatedComplex{N,K}() where {N, K} = TriangulatedComplex{N,K}(
    CellComplex{N,K}(), Dict{Cell{N}, Vector{SignedSimplex{N}}}())

# Create a simplicial TriangulatedComplex from simplices. Creates relative
# orientations for the cells from the simplices.
function TriangulatedComplex(simplices::AbstractVector{Simplex{N, K}}) where {N, K}
    # cache to make sure the same simplex isn't turned into a cell twice
    cells = Dict{Set{Point{N}}, Cell{N}}()
    simplex_mapping = Dict{Cell{N}, Vector{SignedSimplex{N}}}()
    for k in 1:K
        for simplex in simplices
            for s in subsimplices(simplex, k)
                key = Set(s.points)
                if !(key in keys(cells))
                    s_cell = Cell(s)
                    cells[key] = s_cell
                    simplex_mapping[s_cell] = [(s.points, true)]
                    for face in subsimplices(s, k-1)
                        face_cell = cells[Set(face.points)]
                        i = findfirst(p -> !(p in face.points), s.points)
                        o = orientation(face.points, face_cell.points, i-1)
                        parent!(face_cell, s_cell, o)
                    end
                end
            end
        end
    end
    complex = CellComplex{N,K}(collect(values(cells)))
    return TriangulatedComplex{N,K}(complex, simplex_mapping)
end

CellComplex(simplices::AbstractVector{Simplex{N, K}}) where {N, K} =
    TriangulatedComplex(simplices).complex

# if i is even keep the orientation given by the sign_of_permutation, otherwise
# switch it. see [3] page 12.
function orientation(ps1::AbstractVector{T}, ps2::AbstractVector{T}, i::Int) where T
    return sign_of_permutation(ps1, ps2) * (1 - 2 * mod(i, 2)) > 0
end

signed_volume(m::Metric{N}, tcomp::TriangulatedComplex{N}, c::Cell{N}) where N =
    sum([volume(m, Simplex(ps)) * (2 * b - 1) for (ps, b) in tcomp.simplices[c]])

volume(m::Metric{N}, tcomp::TriangulatedComplex{N}, c::Cell{N}) where N =
    abs(signed_volume(m, tcomp, c))

# simplices is a mapping from primal cells to dual elementary cells
# specified as Barycentrics along with signs.
# center takes a Simplex{N, K} and returns a Barycentric{N, K}.
SignedBarySimplex{N} = Vector{Tuple{Vector{Barycentric{N, K} where K}, Bool}}
function elementary_duals!(simplices::Dict{Cell{N}, SignedBarySimplex{N}},
    center::Function, c::Cell{N}) where N
    if !(c in keys(simplices))
        c_center = center(Simplex(c))
        simplices[c] = SignedBarySimplex{N}[]
        if isempty(c.parents)
            push!(simplices[c], ([c_center], true))
        end
        for p in keys(c.parents)
            for (ps, sign) in elementary_duals!(simplices, center, p)
                opposite_index = first_setdiff_index(ps[1].simplex.points, c.points)
                new_sign = (ps[1].coords[opposite_index] >= 0) == sign
                push!(simplices[c], ([c_center, ps...], new_sign))
            end
        end
    end
    return simplices[c]
end

# compute the dual TriangulatedComplex for a simplicial CellComplex primal.
# center takes a Simplex{N} and returns a Point{N}.
function dual(primal::CellComplex{N, K}, center::Function) where {N, K}
    dual_tcomp = TriangulatedComplex{N, K}()
    primal_to_elementary_duals = Dict{Cell{N}, SignedBarySimplex{N}}()
    primal_to_duals = Dict{Cell{N}, Cell{N}}()
    # iterate from high to low dimension so the cells of
    # the dual are constructed in order of dimension
    for k in reverse(1:K)
        for cell in primal.cells[k]
            signed_elementary_duals = [(map(Point, p[1]), p[2]) for p in
                elementary_duals!(primal_to_elementary_duals, center, cell)]
            points = unique(vcat([p[1] for p in signed_elementary_duals]...))
            dual_cell = Cell(points, K-k+1)
            push!(dual_tcomp.complex, dual_cell)
            dual_tcomp.simplices[dual_cell] = signed_elementary_duals
            primal_to_duals[cell] = dual_cell
            for parent in keys(cell.parents)
                o = cell.parents[parent]
                dual_child = primal_to_duals[parent]
                # for k > 1, use o but for k == 1 use !o. This guarantees
                # that d² = 0 and that the dual mesh is positively oriented.
                parent!(dual_child, dual_cell, k == 1 ? !o : o)
            end
        end
    end
    return dual_tcomp
end

################################################################################
# Mesh
################################################################################

# A mesh refers to both the primal and dual meshes.

struct Mesh{N,K}
    primal::TriangulatedComplex{N,K}
    dual::TriangulatedComplex{N,K}
end
show(io::IO, m::Mesh{N,K}) where {N, K} = print(io, "Mesh{$N,$K}($(m.primal), $(m.dual))")

Mesh(tcomp::TriangulatedComplex{N}, center::Function) where N =
    Mesh(tcomp, dual(tcomp.complex, center))

# get the dual of a cell
function dual(mesh::Mesh{N, K}, c::Cell{N}) where {N, K}
    primal = mesh.primal.complex
    dual = mesh.dual.complex
    comp1, comp2 = (c in primal.cells[c.K]) ? (primal, dual) : (dual, primal)
    i = findfirst(isequal(c), comp1.cells[c.K])
    return comp2.cells[K-c.K+1][i]
end
