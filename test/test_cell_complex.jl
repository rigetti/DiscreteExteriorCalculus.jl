using Test, DiscreteExteriorCalculus
const DEC = DiscreteExteriorCalculus
using Combinatorics: combinations

@testset "relative orientation" begin
    s = Simplex(Point(0,0,0), Point(1,0,0))
    comp = CellComplex([s])
    for v in comp.cells[1]
        @test length(v.parents) == 1
        p = collect(keys(v.parents))[1]
        o = v.parents[p]
        @test p == comp.cells[2][1]
        if v.points[1] == s.points[1]
            @test !o
        else
            @test o
        end
    end
    @test comp.cells[1][1].points[1] != comp.cells[1][2].points[1]
end

@testset "CellComplex" begin
    points = [Point(0,0,0), Point(1,0,0), Point(0,1,0), Point(0,0,1), Point(1,1,1)]
    s1, s2 = Simplex(points[1:4]), Simplex(points[2:5])
    comp1, comp2 = CellComplex([s1]), CellComplex([s2])
    @test Set([c.points[1] for c in comp1.cells[1]]) == Set(s1.points)
    @test Set(comp1.cells[end][1].points) == Set(s1.points)
    @test Set([c.points[1] for c in comp2.cells[1]]) == Set(s2.points)
    @test Set(comp2.cells[end][1].points) == Set(s2.points)
    for k in 1:4
        @test length(comp1.cells[k]) == length(combinations(1:4, k))
        @test length(comp2.cells[k]) == length(combinations(1:4, k))
    end
    comp = CellComplex([s1, s2])
    @test [length(a) for a in comp.cells] == [4 * 2 - 3, 6 * 2 - 3, 4 * 2 - 1, 1 * 2 - 0]
    @test Set([c.points[1] for c in comp.cells[1]]) == Set(points)
    ps = [Set(c.points) for c in comp.cells[end]]
    @test (ps == [Set(s1.points), Set(s2.points)]) || (ps == [Set(s2.points), Set(s1.points)])
end


@testset "boundary and absolute orientation" begin
    points = [Point(0,0), Point(1,0), Point(0,1), Point(1,1),
        Point(2,0), Point(2,1), Point(3,0)]
    s1, s2, s3 = Simplex(points[1:3]), Simplex(points[2:4]), Simplex(points[5:7])
    comp = CellComplex([s1, s2, s3])
    @test Set(map(Simplex, comp.cells[end])) == Set([s1, s2, s3])
    orient!(comp)
    s2_positive = Simplex(points[3], points[2], points[4])
    s3_positive = Simplex(points[6], points[5], points[7])
    @test Set(map(Simplex, comp.cells[end])) == Set([s1, s2_positive, s3_positive])
    # boundaries
    interior, exterior = boundary_components(comp)
    @test isempty(interior)
    @test length(exterior) == 2
    @test Set([length(e.cells[end]) for e in exterior]) == Set([3,4])
    exterior_union = vcat([e.cells[end] for e in exterior]...)
    relative_orientations = Bool[]
    for c in exterior_union
        @test length(c.parents) == 1
        push!(relative_orientations, collect(values(c.parents))[1])
    end
    # boundary not positively oriented
    @test !(all(relative_orientations))
    orient!(exterior_union)
    # now boundary is positively oriented
    for c in exterior_union
        @test length(c.parents) == 1
        @test collect(values(c.parents))[1]
    end
end
