using Test, DiscreteExteriorCalculus

@testset "triangle and pairwise_delaunay" begin
    m = Metric(2)
    points = [Point(0,0), Point(0,1), Point(1,0), Point(1,1)]
    comp = triangulate(points).complex
    @test map(length, comp.cells) == [4, 5, 2]
    @test Set([c.points[1] for c in comp.cells[1]]) == Set(points)
    @test !pairwise_delaunay(m, comp)
    @test pairwise_delaunay(m, comp, -sqrt(eps(Float64)))
    @test one_sided(m, comp)

    points = [Point(0,0), Point(0,1), Point(1,0), Point(1.5,2)]
    comp = triangulate(points).complex
    @test map(length, comp.cells) == [4, 5, 2]
    @test Set([c.points[1] for c in comp.cells[1]]) == Set(points)
    @test simplicial(comp)
    @test one_sided(m, comp)
    @test pairwise_delaunay(m, comp)
end
