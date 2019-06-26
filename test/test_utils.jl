using Test, DiscreteExteriorCalculus
const DEC = DiscreteExteriorCalculus

@testset "parametrize_subspace" begin
    M, v = [[0,1] [-1,1] [0,1]], [-.5,1]
    a, b = DEC.parametrize_subspace(M, v)
    @test M * a ≈ v
    @test M * (a + b * [1]) ≈ v
    @test M * (a + b * [-2]) ≈ v

    M, v = [[.5, 1] [-.5, 1] [-.5, 1]], [0.0, 1]
    a, b = DEC.parametrize_subspace(M, v)
    @test M * a ≈ v
    @test M * (a + b * [1]) ≈ v
    @test M * (a + b * [-2]) ≈ v

    M, v = [[.5, 1] [-.5, 1]], [0.0, 1]
    a, b = DEC.parametrize_subspace(M, v)
    @test size(b) == (2,0)
    @test M * a ≈ v
end

@testset "sign_of_permutation" begin
    @test DEC.sign_of_permutation([1],[1]) == 1
    @test DEC.sign_of_permutation([1,2],[2,1]) == -1
    @test DEC.sign_of_permutation([1,2],[1,2]) == 1
end
