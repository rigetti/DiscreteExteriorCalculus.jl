import VoronoiDelaunay; const vd = VoronoiDelaunay
using LinearAlgebra: norm

# This file uses VoronoiDelaunay.jl to create 2 dimensional triangulations from a list of
# points. The mesh is guaranteed to be (weakly) delaunay, however it is not
# necessarily one-sided.

vd_scale(x::Real, min_val::Real, max_val::Real) = (x - min_val) *
    (vd.max_coord - vd.min_coord)/(max_val - min_val) + vd.min_coord
vd_scale(p::Point{2}, min_val::Real, max_val::Real) = vd.Point(
    map(x -> vd_scale(x, min_val, max_val), p.coords)...)
vd_unscale(x::Real, min_val::Real, max_val::Real) = (x - vd.min_coord) *
    (max_val - min_val)/(vd.max_coord - vd.min_coord) + min_val
vd_unscale(p::vd.Point2D, min_val::Real, max_val::Real) = Point(
    map(x -> vd_unscale(x, min_val, max_val), [vd.getx(p), vd.gety(p)])...)

export triangulate
"""
    triangulate(points::AbstractVector{Point{2}})

Produce a triangulation that is weakly delaunay (i.e. no circumcircle contains any vertex
in its interior but it can contain have more than 3 vertices on its boundary).
"""
function triangulate(points::AbstractVector{Point{2}})
    @assert length(points) >= 3
    all_coords = vcat([p.coords for p in points]...)
    min_val, max_val = minimum(all_coords), maximum(all_coords)
    @assert min_val < max_val
    tess = vd.DelaunayTessellation(length(points))
    push!(tess, map(p -> vd_scale(p, min_val, max_val), points))
    triangles = Simplex{2,3}[]
    for t in tess
        t_points = [vd_unscale(p, min_val, max_val)
                    for p in [vd.geta(t), vd.getb(t), vd.getc(t)]]
        s = Simplex([points[argmin([norm(p.coords - q.coords) for q in points])]
            for p in t_points])
        push!(triangles, s)
    end
    return TriangulatedComplex(triangles)
end
