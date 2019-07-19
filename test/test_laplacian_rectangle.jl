using Test, DiscreteExteriorCalculus
const DEC = DiscreteExteriorCalculus
using LinearAlgebra: eigen
using AdmittanceModels: sparse_nullbasis
using Base.Iterators: product
using UniqueVectors: UniqueVector

r1, r2 = .5, .4
num = 40
points, tcomp = DEC.triangulated_lattice([r1, 0], [0, r2], num, num)

N, K = 2, 3
@test typeof(tcomp) <: TriangulatedComplex{N, K}

orient!(tcomp.complex)
m = Metric(N)
mesh = Mesh(tcomp, circumcenter(m))

laplacian = differential_operator(m, mesh, "Δ", 1, true)
comp = tcomp.complex
_, exterior = boundary_components_connected(comp)
constraint = zero_constraint(comp, exterior.cells[1], 1)
nullbasis = sparse_nullbasis(constraint)

vals, vects = eigen(collect(transpose(nullbasis) * laplacian * nullbasis))
inds = sortperm(vals)
vals, vects = vals[inds], vects[:, inds]

# solution is ψ(x,y) = sin(m*π*x/R1)*sin(n*π*y/R2) for integers m and n
function rect_vecs(m, n)
    v = zeros(num+1, num+1)
    for i in 1:(num+1)
        for j in 1:(num+1)
            x, y = (i-1)/num, (j-1)/num
            v[i, j] = sin(m*π*x)*sin(n*π*y)
        end
    end
    return v
end
rect_vals(m, n, R1, R2) = (m * π/R1)^2 + (n * π/R2)^2
l = 16
correct_vals = [rect_vals(i,j,r1,r2) for (i,j) in vcat(product(1:l, 1:l)...)]
correct_inds = sortperm(correct_vals)[1:l]

correct_vs = [rect_vecs(i,j)
    for (i,j) in vcat(product(1:l, 1:l)...)[correct_inds]]
comp_points = UniqueVector([c.points[1] for c in comp.cells[1]])
ordering = [findfirst(isequal(p), comp_points) for p in points]
vs = [collect(transpose(reshape((nullbasis * vects[:,i])[ordering],
    num+1, num+1))) for i in 1:l]
for i in 1:l
    vs[i] /= maximum(abs.(vs[i]))
    j = argmax(abs.(correct_vs[i]))
    if vs[i][j] * correct_vs[i][j] < 0
        vs[i] *= -1
    end
end

@testset "laplacian on a rectangle" begin
    @test isapprox(vals[1:l], correct_vals[correct_inds], rtol=1e-2)
    @test isapprox(vs, correct_vs, rtol=1e-2)
end

if false # set to true to make heatmap plots of modes
    using PlotlyJS: plot, heatmap
    plot(heatmap(z=vs[1]))
end
