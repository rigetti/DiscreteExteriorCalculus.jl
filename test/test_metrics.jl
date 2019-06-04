using Test, DiscreteExteriorCalculus
const DEC = DiscreteExteriorCalculus
using LinearAlgebra: norm, det
using StaticArrays: SVector

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
