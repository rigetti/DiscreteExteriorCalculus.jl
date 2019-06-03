using Test, DiscreteExteriorCalculus
const DEC = DiscreteExteriorCalculus

@testset "triangle and pairwise_delaunay" begin
    m = Metric(2)
    points = [Point(0,0), Point(0,1), Point(1,0), Point(1,1)]
    comp = DEC.triangulate(points).complex
    @test map(length, comp.cells) == [4, 5, 2]
    @test Set([c.points[1] for c in comp.cells[1]]) == Set(points)
    @test !DEC.pairwise_delaunay(m, comp)
    @test DEC.pairwise_delaunay(m, comp, -sqrt(eps(Float64)))
    @test DEC.one_sided(m, comp)
    xs, ys = DEC.plot_cells(comp)
    # plot(x=xs, y=ys, Geom.path)

    points = [Point(0,0), Point(0,1), Point(1,0), Point(1.5,2)]
    comp = DEC.triangulate(points).complex
    @test map(length, comp.cells) == [4, 5, 2]
    @test Set([c.points[1] for c in comp.cells[1]]) == Set(points)
    @test DEC.simplicial(comp)
    @test DEC.one_sided(m, comp)
    @test DEC.pairwise_delaunay(m, comp)
    xs, ys = DEC.plot_cells(comp)
end
