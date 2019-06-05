TriangulatedComplex{N,K}() where {N, K} = TriangulatedComplex{N,K}(
    CellComplex{N,K}(), Dict{Cell{N}, Vector{SignedSimpleSimplex{N}}}())

# Create a simplicial TriangulatedComplex from simplices. Creates relative
# orientations for the cells from the simplices.
function TriangulatedComplex(simplices::AbstractVector{Simplex{N, K}}) where {N, K}
    # cache to make sure the same simplex isn't turned into a cell twice
    cells = Dict{Set{Point{N}}, Cell{N}}()
    simplex_mapping = Dict{Cell{N}, Vector{SignedSimpleSimplex{N}}}()
    for k in 1:K
        for simplex in simplices
            for s in subsimplices(simplex, k)
                key = Set(s.points)
                if !(key in keys(cells))
                    s_cell = Cell(s)
                    cells[key] = s_cell
                    simplex_mapping[s_cell] = [(SimpleSimplex(s), true)]
                    for face in subsimplices(s, k-1)
                        face_cell = cells[Set(face.points)]
                        i = findfirst(p -> !(p in face.points), s.points)
                        pm = sign_of_permutation(face.points, face_cell.points)
                        parent!(face_cell, s_cell, pm * (1 - 2 * mod(i-1, 2)) > 0)
                    end
                end
            end
        end
    end
    complex = CellComplex{N,K}(collect(values(cells)))
    return TriangulatedComplex{N,K}(complex, simplex_mapping)
end

import Base: show
show(io::IO, tcomp::TriangulatedComplex{N,K}) where {N, K} = print(
     io, "TriangulatedComplex{$N,$K}$(tuple(map(length, tcomp.complex.cells)...))")

signed_volume(m::Metric{N}, tcomp::TriangulatedComplex{N}, c::Cell{N}) where N =
    sum([volume(m, Simplex(s)) * (2 * b - 1) for (s, b) in tcomp.simplices[c]])

volume(m::Metric{N}, tcomp::TriangulatedComplex{N}, c::Cell{N}) where N =
    abs(signed_volume(m, tcomp, c))

# simplices is a mapping from primal cells to dual elementary cells
# specified as Barycentrics along with signs.
# center takes a Simplex{N, K} and returns a Barycentric{N, K}.
SimpleBarySimplex{N} = Vector{SimpleBarycentric{N}}
function elementary_duals!(simplices::Dict{Cell{N}, Vector{Tuple{SimpleBarySimplex{N}, Bool}}},
    center::Function, c::Cell{N}) where N
    if !(c in keys(simplices))
        c_center = SimpleBarycentric(center(Simplex(c)))
        simplices[c] = Tuple{SimpleBarySimplex{N}, Bool}[]
        if isempty(c.parents)
            push!(simplices[c], ([c_center], true))
        end
        for p in keys(c.parents)
            for (ps, sign) in elementary_duals!(simplices, center, p)
                opposite_index = first_setdiff_index(ps[1].simplex.points, c.points)
                new_sign = (ps[1].coords[opposite_index] >= 0) == sign
                new_barycentrics = [c_center, ps...]
                push!(simplices[c], (new_barycentrics, new_sign))
            end
        end
    end
    return simplices[c]
end

# compute the dual TriangulatedComplex for a simplicial CellComplex primal.
# center takes a Simplex{N} and returns a Point{N}.
function dual(primal::CellComplex{N, K}, center::Function) where {N, K}
    dual_tcomp = TriangulatedComplex{N, K}()
    primal_to_elementary_duals = Dict{Cell{N}, Vector{Tuple{SimpleBarySimplex{N}, Bool}}}()
    primal_to_duals = Dict{Cell{N}, Cell{N}}()
    # iterate from high to low dimension so the cells of
    # the dual are constructed in order of dimension
    for k in reverse(1:K)
        for cell in primal.cells[k]
            elementary_duals = elementary_duals!(primal_to_elementary_duals, center, cell)
            signed_elementary_duals = [(SimpleSimplex(p[1]), p[2])
                for p in elementary_duals]
            points = unique(vcat([p[1].points for p in signed_elementary_duals]...))
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

############################################################################################
# Mesh
############################################################################################

Mesh(tcomp::TriangulatedComplex{N}, center::Function) where N =
    Mesh(tcomp, dual(tcomp.complex, center))

show(io::IO, m::Mesh{N,K}) where {N, K} = print(io, "Mesh{$N,$K}($(m.primal), $(m.dual))")

# get the dual of a cell
function dual(mesh::Mesh{N, K}, c::Cell{N}) where {N, K}
    primal = mesh.primal.complex
    dual = mesh.dual.complex
    comp1, comp2 = (c in primal.cells[c.K]) ? (primal, dual) : (dual, primal)
    i = findfirst(isequal(c), comp1.cells[c.K])
    return comp2.cells[K-c.K+1][i]
end
