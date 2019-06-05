Metric(mat::T) where {N, T<:SMatrix{N, N, Float64}} = Metric{N, T}(Symmetric(mat))

function Metric(mat::AbstractMatrix{<:Real})
    N, M = size(mat)
    @assert N == M
    return Metric(SMatrix{N, N, Float64}(mat))
end

Metric(N::Int) = Metric(SMatrix{N,N}(1.0I))

function Wedge(vectors::Vector{SVector{N, Float64}}) where N
    K = length(vectors)
    Wedge{N, K}(SVector{K, SVector{N, Float64}}(vectors))
end

function Wedge(s::Simplex{N, K}) where {N, K}
    vectors = SVector{N, Float64}[p.coords - s.points[1].coords for p in s.points[2:end]]
    return Wedge(vectors)
end

export inner_product
"""
    inner_product(m::Metric{N}, v1::SVector{N, Float64}, v2::SVector{N, Float64}) where N
    inner_product(m::Metric{N}, w1::Wedge{N, K}, w2::Wedge{N, K}) where {N, K}

Compute the inner product of two vectors with respect to the given metric.
"""
inner_product(m::Metric{N}, v1::SVector{N, Float64}, v2::SVector{N, Float64}) where N =
    transpose(v1) * m.mat * v2

function inner_product(m::Metric{N}, w1::Wedge{N, K}, w2::Wedge{N, K}) where {N, K}
    if K == 0
        return 1.0
    else
        inner_products = hcat([[inner_product(m, v1, v2) for v1 in w1.vectors]
            for v2 in w2.vectors]...)
        return det(inner_products)
    end
end

export norm_square
"""
    norm_square(m::Metric{N}, v::SVector{N, Float64}) where N
    norm_square(m::Metric{N}, w::Wedge{N}) where N

Compute the inner product of a vector with itself with respect to the given metric. Note
that this value can be negative if the metric is not positive semi-definite.
"""
norm_square(m::Metric{N}, v::SVector{N, Float64}) where N = inner_product(m, v, v)

norm_square(m::Metric{N}, w::Wedge{N}) where N = inner_product(m, w, w)

import LinearAlgebra: norm
"""
    norm(m::Metric{N}, v::SVector{N, Float64}) where N
    norm(m::Metric{N}, w::Wedge{N}) where N

Compute the square root of the magnitude of the inner product of a vector `v` with
itself with respect to the metric `m`.
"""
norm(m::Metric{N}, v::SVector{N, Float64}) where N = sqrt(abs(norm_square(m, v)))

norm(m::Metric{N}, w::Wedge{N}) where N = sqrt(abs(norm_square(m, w)))

export volume_square
"""
    volume_square(m::Metric{N}, s::Simplex{N, K}) where {N, K}

Compute the squared volume of a simplex with respect to the given metric. Note that this
value can be negative if the metric is not positive semi-definite.
"""
volume_square(m::Metric{N}, s::Simplex{N, K}) where {N, K} =
    norm_square(m, Wedge(s))/factorial(K-1)^2

export volume
"""
    volume(m::Metric{N}, s::Simplex{N, K}) where {N, K}

Compute the volume of a simplex with respect to the given metric.
"""
volume(m::Metric{N}, s::Simplex{N, K}) where {N, K} =
    sqrt(abs(norm_square(m, Wedge(s))))/factorial(K-1)
