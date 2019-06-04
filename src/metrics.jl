import LinearAlgebra: norm

################################################################################
# Metric
################################################################################

Metric(mat::T) where {N, T<:SMatrix{N, N, Float64}} = Metric{N, T}(Symmetric(mat))

function Metric(mat::AbstractMatrix{<:Real})
    N, M = size(mat)
    @assert N == M
    return Metric(SMatrix{N, N, Float64}(mat))
end

Metric(N::Int) = Metric(SMatrix{N,N}(1.0I))

inner_product(m::Metric{N}, v1::SVector{N, Float64}, v2::SVector{N, Float64}) where N = transpose(v1) * m.mat * v2

norm_square(m::Metric{N}, v::SVector{N, Float64}) where N = inner_product(m, v, v)

norm(m::Metric{N}, v::SVector{N, Float64}) where N = sqrt(abs(norm_square(m, v)))

################################################################################
# Wedge
################################################################################

function Wedge(vectors::Vector{SVector{N, Float64}}) where N
    K = length(vectors)
    Wedge{N, K}(SVector{K, SVector{N, Float64}}(vectors))
end

function Wedge(s::Simplex{N, K}) where {N, K}
    vectors = SVector{N, Float64}[p.coords - s.points[1].coords for p in s.points[2:end]]
    return Wedge(vectors)
end

orientation(s::Simplex) = sign(det(hcat(Wedge(s).vectors...)))

function inner_product(m::Metric{N}, w1::Wedge{N, K}, w2::Wedge{N, K}) where {N, K}
    if K == 0
        return 1.0
    else
        inner_products = hcat([[inner_product(m, v1, v2) for v1 in w1.vectors] for v2 in w2.vectors]...)
        return det(inner_products)
    end
end

norm_square(m::Metric{N}, w::Wedge{N}) where N = inner_product(m, w, w)

volume_square(m::Metric{N}, s::Simplex{N, K}) where {N, K} = norm_square(m, Wedge(s))/factorial(K-1)^2

volume(m::Metric{N}, s::Simplex{N, K}) where {N, K} = sqrt(abs(norm_square(m, Wedge(s))))/factorial(K-1)
