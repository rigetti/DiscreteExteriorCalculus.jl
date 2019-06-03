using Test, DiscreteExteriorCalculus
const DEC = DiscreteExteriorCalculus
using LinearAlgebra: norm, det
using Statistics: mean
using StaticArrays: SVector

@testset "Point and Simplex" begin
    p = Point(1,2,3)
    @test typeof(p) == Point{3}
    s = Simplex(Point(1,2,3), Point(4,5,6), Point(1,4,7))
    @test typeof(s) == Simplex{3, 3}
end

@testset "norm, wedge, and volume" begin
    m = Metric(3)
    @test norm(m, SVector{3, Float64}([1,2,3])) == norm([1,2,3])
    a1, a2 = 3.4, 2.8
    b1, b2 = -2.4, 20.3
    s = Simplex(Point(0, 0, 0), Point(a1, a2, 0), Point(b1, b2, 0))
    @test DEC.volume(m, s) ≈ det([[a1, a2] [b1, b2]])/2
    a1, a2, a3 = 3.4, 2.8, 3.9
    b1, b2, b3 = -2.4, 20.3, -3.2
    c1, c2, c3 = .1, -49.2, 12.0
    s = Simplex(Point(0, 0, 0), Point(a1, a2, a3), Point(b1, b2, b3), Point(c1, c2, c3))
    @test DEC.volume(m, s) ≈ det([[a1, a2, a3] [b1, b2, b3] [c1, c2, c3]])/6
    m = Metric(2)
    s = Simplex(Point(1,2))
    DEC.volume(m, s) == 1.0
end

@testset "Barycentric and barycentric_projection" begin
    m = Metric(3)
    s = Simplex(Point(0,0,0), Point(2,0,0), Point(1,.5,0))
    for p in [s.points; Point(3,5,7); Point(-12,25,-100); Point(24.3,-3.5,-2.2)]
        b = Barycentric(m, s, p)
        @test norm(Point(b).coords - [p.coords[1:end-1]; 0]) < 1e-14
    end
end

@testset "distance" begin
    m = Metric(3)
    s = Simplex(Point(0,0,0), Point(2,0,0), Point(1.3,.7,0), Point(0,0,1))
    ps = [s.points; Point(3,5,7); Point(-12,25,-100); Point(24.3,-3.5,-2.2)]
    for p1 in ps
        for p2 in ps
            d1 = sqrt(abs(DEC.distance_square(m, Barycentric(m, s, p1), p2)))
            d2 = norm(m, p1.coords - p2.coords)
            d3 = norm(p1.coords - p2.coords) # using Euclidean metric
            @test d2 ≈ d3
            @test abs(d1 - d2) < 1e-6
        end
    end
end

@testset "circumsphere" begin
    function is_circumcenter(m::Metric{N}, s::Simplex{N, K}, p::Point{N}) where {N, K}
        if K == 1
            return s.points[1] == p
        else
            dists = [norm(m, p.coords - q.coords) for q in s.points]
            return all([d ≈ dists[1] for d in dists])
        end
    end
    function circumradius_square(m::Metric{N}, s::Simplex{N}) where N
        center, _ = DEC.circumsphere(m, s)
        return mean([DEC.norm_square(m, center.coords - v.coords) for v in s.points])
    end
    m = Metric(3)
    s = Simplex(Point(1,2,3), Point(-12,3,5), Point(8,3,4), Point(9,25,-2))
    for k in 1:4
        for sub_s in DEC.subsimplices(s, k)
            center, radius_square = DEC.circumsphere(m, sub_s)
            @test is_circumcenter(m, sub_s, center)
            @test radius_square ≈ circumradius_square(m, sub_s)
        end
    end
end

@testset "barycentric_subspace" begin
    m = Metric(2)
    s = Simplex(Point(0,0), Point(1,0), Point(0,1))
    f = Simplex(s.points[1:2]...)
    p = Point(.5, 3)
    M, v = DEC.barycentric_subspace(m, s, f, p)
    @test M * Barycentric(m, s, p).coords == v
    @test M * [.5, .5, 0] == v
    @test M * Barycentric(m, s, Point(.5, -2)).coords == v
end

@testset "rotate_about_face" begin
    m = Metric(3)
    s = Simplex(Point(-1,0,0), Point(0,0,0), Point(0,1,0))
    r, θ = 3, π/6
    p = Point(r * cos(θ), 0, r * sin(θ))
    b = DEC.rotate_about_face(m, s, p, 1)
    @test norm(Point(b).coords - [r, 0, 0]) < 1e-15

    m = Metric(2)
    s = Simplex(Point(0,0),Point(1,0))
    p = Point(1,1)
    b = DEC.rotate_about_face(m, s, p, 2)
    @test Point(b).coords ≈ [-sqrt(2), 0]
    b = DEC.rotate_about_face(m, s, p, 1)
    @test Point(b).coords ≈ [2, 0]
end

@testset "pairwise_delaunay" begin
    m = Metric(3)
    s1 = Simplex(Point(-.1,.5,0), Point(0,0,0), Point(0,1,0))
    center, _ = DEC.circumsphere(m, s1)
    θ = π/6
    for r in [.1, 1, 5, 10]
        p = Point(r * cos(θ), 0, r * sin(θ))
        proj_p = Point(r, 0, 0)
        # using Euclidean metric
        delaunay = norm(center.coords - proj_p.coords)^2 - norm(center.coords - s1.points[1].coords)^2
        s2 = Simplex(s1.points[2:end]..., p)
        @test DEC.pairwise_delaunay(m, s1, s2) ≈ delaunay
    end
end

# TODO test with other metrics
