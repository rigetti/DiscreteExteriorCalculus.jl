import VoronoiDelaunay; const vd = VoronoiDelaunay # there are name overlaps
using LinearAlgebra: norm

# This file uses VoronoiDelaunay.jl to create 2 dimensional Delaunay
# triangulation from a list of points. The mesh is guaranteed to be
# (weakly) delaunay, however it is not necessarily one-sided.

vd_scale(x::Real, min_val::Real, max_val::Real) = (x - min_val) *
    (vd.max_coord - vd.min_coord)/(max_val - min_val) + vd.min_coord
vd_scale(p::Point{2}, min_val::Real, max_val::Real) = vd.Point(
    map(x -> vd_scale(x, min_val, max_val), p.coords)...)
vd_unscale(x::Real, min_val::Real, max_val::Real) = (x - vd.min_coord) *
    (max_val - min_val)/(vd.max_coord - vd.min_coord) + min_val
vd_unscale(p::vd.Point2D, min_val::Real, max_val::Real) = Point(
    map(x -> vd_unscale(x, min_val, max_val), [vd.getx(p), vd.gety(p)])...)

# produce a triangulation that is weakly delaunay (i.e. no circumcircle
# contains any vertex on its interior but it can contain have than 3
# vertices on its boundary).
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
        s = Simplex([points[argmin([norm(p.coords - q.coords) for q in points])] for p in t_points])
        push!(triangles, s)
    end
    return TriangulatedComplex(triangles)
end

# To plot a mesh run
# using Gadly: plot, Geom
# xs, ys = plot_cells(comp)
# plot(x=xs, y=ys, Geom.path)
function plot_cells(comp::CellComplex{2,3})
    xs, ys = Float64[], Float64[]
    for e in comp.cells[2]
        for p in e.points
            push!(xs, p.coords[1]); push!(ys, p.coords[2])
        end
        push!(xs, NaN); push!(ys, NaN)
    end
    return xs, ys
end
