using Test, DiscreteExteriorCalculus
const DEC = DiscreteExteriorCalculus

@testset "Mesh obtuse triangle" begin
    m = Metric(2)
    s = Simplex(Point(0,0), Point(2,0), Point(1,.5))
    for center in [DEC.centroid, DEC.circumcenter(m)]
        primal = TriangulatedComplex([s])
        mesh = Mesh(primal, center)
        @test map(length, mesh.primal.complex.cells) ==
            reverse(map(length, mesh.dual.complex.cells))
        cell_1 = mesh.dual.complex.cells[1][1]
        @test cell_1.K == 1
        @test cell_1.points[1] == Point(center(s))
        @test length(cell_1.parents) == 3
        @test length(cell_1.children) == 0
        cells_2 = mesh.dual.complex.cells[2]
        @test all([length(c.parents) == 2 for c in cells_2])
        @test all([length(c.children) == 1 for c in cells_2])
        cells_3 = mesh.dual.complex.cells[3]
        @test all([length(c.parents) == 0 for c in cells_3])
        @test all([length(c.children) == 2 for c in cells_3])
    end
end

@testset "signed_volume" begin
    points = [Point(0, 0), Point(2, 0), Point(1, .5), Point(1, -5)]
    obtuse = Simplex(points[1:3])
    acute = Simplex(points[1], points[2], points[4])
    m = Metric(2)
    b, _ = DEC.circumsphere_barycentric(m, obtuse)
    @test any(b.coords .< 0) # obtuse
    b, _ = DEC.circumsphere_barycentric(m, acute)
    @test !any(b.coords .< 0) # acute
    tcomp = TriangulatedComplex([acute, obtuse])
    center = DEC.circumcenter(m)
    mesh = Mesh(tcomp, center)
    # check that the length of the edge between the cells is correct
    edge = DEC.subcomplex(mesh.primal.complex, [Simplex(points[1:2])]).cells[2][1]
    dual_edge = DEC.dual(mesh, edge)
    ccs = [Point(center(x)) for x in [Simplex(edge), obtuse, acute]]
    d1 = DEC.norm(m, ccs[1].coords - ccs[2].coords)
    d2 = DEC.norm(m, ccs[1].coords - ccs[3].coords)
    @test abs(DEC.volume(m, mesh.dual, DEC.dual(mesh, edge))) == abs(d1 - d2)
    # check that the choice of sign was correct
    simplices, bools = zip(mesh.dual.simplices[dual_edge]...)
    @test bools[1] != bools[2]
    Set(simplices[bools[1] ? 1 : 2]) == Set([ccs[1], ccs[3]])
    Set(simplices[bools[1] ? 2 : 1]) == Set([ccs[1], ccs[2]])
end

@testset "honeycomb mesh" begin
    basis = [[1,0], [.5, .5 * sqrt(3)]]
    n = 5
    points = [Point((basis[1] * i + basis[2] * j)...) for i in 1:n for j in 1:n]
    tcomp = DEC.triangulate(points)
    m = Metric(2)
    mesh = Mesh(tcomp, DEC.centroid)
    for c in mesh.primal.complex.cells[1]
        @test DEC.volume(m, mesh.primal, c) == 1
    end
    for c in mesh.primal.complex.cells[2]
        @test DEC.volume(m, mesh.primal, c) ≈ 1
    end
    for c in mesh.primal.complex.cells[3]
        @test DEC.volume(m, mesh.primal, c) ≈ sqrt(3)/4
    end
    for c in mesh.dual.complex.cells[1]
        @test DEC.volume(m, mesh.dual, c) == 1
    end
    for c in mesh.dual.complex.cells[2]
        @test DEC.volume(m, mesh.dual, c) ≈ sqrt(3)/6 * length(c.children)
    end
    for c in mesh.dual.complex.cells[3]
        k = length(c.children)
        @test DEC.volume(m, mesh.dual, c) ≈ (sqrt(3)/12) * (k == 6 ? k : k-1)
    end
end
