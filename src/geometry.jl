using StaticArrays: SVector, SMatrix
using LinearAlgebra: I, Symmetric, det
import LinearAlgebra: norm
using Combinatorics: combinations

################################################################################
# Point
################################################################################

function Point(coords::AbstractVector{<:Real})
    N = length(coords)
    return Point{N}(SVector{N, Float64}(coords))
end

Point(coords::Vararg{<:Real}) = Point(collect(coords))

Point(b::Barycentric) = Point(barycentric_matrix(b.simplex) * b.coords)

################################################################################
# Simplex
################################################################################

function Simplex(points::AbstractVector{Point{N}}) where N
    K = length(points)
    return Simplex{N, K}(SVector{K, Point{N}}(points))
end

Simplex(points::Vararg{Point{N}}) where N = Simplex(collect(points))

function Simplex(c::Cell)
    @assert length(c.points) == c.K
    return Simplex(c.points)
end

function subsimplices(s::Simplex{N, K}, k::Int) where {N, K}
    @assert k >= 0
    if (k == 0) || (k > K)
        return Simplex{N, k}[]
    else
        return [Simplex(ps...) for ps in combinations(s.points, k)]
    end
end

################################################################################
# Metric
################################################################################

Metric(mat::T) where {N, T<:SquareSMatrix{N}} = Metric{N, T}(Symmetric(mat))

function Metric(mat::AbstractMatrix{<:Real})
    N, M = size(mat)
    @assert N == M
    return Metric(SquareSMatrix{N}(mat))
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

################################################################################
# Barycentric
################################################################################

function Barycentric(s::Simplex{N, K}, coords::AbstractVector{<:Real}) where {N, K}
    @assert length(coords) == K
    return Barycentric{N, K}(s, SVector{K, Float64}(coords))
end

Barycentric(s::Simplex, coords::Vararg{<:Real}) = Barycentric(s, collect(coords))

function Barycentric(m::Metric{N}, s::Simplex{N}, p::Point{N}) where N
    A, B = barycentric_projection_matrices(m, s)
    coords = A * p.coords + B
    return Barycentric(s, coords)
end

barycentric_matrix(s::Simplex) = hcat([p.coords for p in s.points]...)

# Extremize norm_square(m, p - b * x) over x constrained by sum(x) == 1 where
# b = barycentric_matrix(s) to find the barycentric coordinates of the
# projection of p onto the affine subspace of s.
# The answer can be written in the form x = A * p + B. Return A, B.
function barycentric_projection_matrices(m::Metric{N}, s::Simplex{N, K}) where {N, K}
    b = barycentric_matrix(s)
    # solve sum(x) == 1 by saying x = c * y + d
    c = transpose(hcat(Matrix{Float64}(I, K-1, K-1), -ones(K-1)))
    d = zeros(K); d[end] = 1
    bc = b * c; bcT = transpose(bc)
    A = c * ((bcT * m.mat * bc) \ (bcT * m.mat))
    B = d - A * b * d
    return A, B
end

# The squared distance from a point with barycentric coordinates x to a point p
# can be written in the form xᵀAx + Bᵀx where B depends on p but A does not.
# See Theorem 2 in [1].
distance_quadratic(m::Metric{N}, s::Simplex{N}) where N = -.5 * circumsphere_matrix(m, s)

distance_linear(m::Metric{N}, s::Simplex{N}, p::Point{N}) where N = [norm_square(
    m, p.coords - q.coords) for q in s.points]

function distance_square(m::Metric{N}, b::Barycentric{N}, p::Point{N}) where N
    A = distance_quadratic(m, b.simplex)
    B = distance_linear(m, b.simplex, p)
    return transpose(b.coords) * A * b.coords + transpose(B) * b.coords
end

function circumsphere_matrix(m::Metric{N}, s::Simplex{N}) where N
    norm_squares = [[norm_square(m, p.coords - q.coords) for p in s.points] for q in s.points]
    return transpose(hcat(norm_squares...))
end

# The circumcenter satisfies circumsphere_matrix(s) * x = 2 * R^2 * ones(K)
# where R^2 is the squared circumradius. Return x, R^2.
function circumsphere_barycentric(m::Metric{N}, s::Simplex{N, K}) where {N, K}
    if K == 1
        return Barycentric(s, [1.0]), 0
    else
        coords = circumsphere_matrix(m, s) \ ones(K)
        sum_coords = sum(coords)
        return Barycentric(s, coords / sum_coords), .5/sum_coords
    end
end

function circumsphere(m::Metric{N}, s::Simplex{N}) where N
    b, R² = circumsphere_barycentric(m, s)
    return Point(b), R²
end

# find the linear equation of the affine subspace of the affine space of s
# specified by requiring that the projection onto f is the same as that for p
function barycentric_subspace(m::Metric{N}, s::Simplex{N, K}, f::Simplex{N, J},
                              p::Point{N}) where {N, K, J}
    @assert J <= K
    # require that the projection of x onto f is the same as for p
    A, _ = barycentric_projection_matrices(m, f)
    M = A * barycentric_matrix(s)
    v = A * p.coords
    # since the sum of barycentric coordinates is 1, one equation is redundant
    # and one variable is redundant. Replace the last equation with sum(x) == 1
    M[end, :] = ones(K)
    v[end] = 1
    return M, v
end

# let s be a simplex, q = s.points[i], and f the face of s opposite q.
# Rotate the point p about f into the affine subspace of s but on the halfspace
# of f not containing q.
function rotate_about_face(m::Metric{N}, s::Simplex{N, K}, p::Point{N}, i::Int) where {N, K}
    f = Simplex(s.points[filter(j -> j != i, 1:K)])
    a, b = parametrize_subspace(barycentric_subspace(m, s, f, p)...)
    q = f.points[1]
    A, B = distance_quadratic(m, s), distance_linear(m, s, q)
    C = -norm_square(m, p.coords - q.coords)
    x1, x2 = intersect_linear_quadratic(A, B, C, a, b[:]) # b[:] reshapes it to a vector
    x = x1[i] > 0 ? x2 : x1
    return Barycentric(s, x)
end

# given two simplices s1 and s2 that share a face, rotate s2 into the
# affine subspace of s1 and then check where its point not in s1 is mapped
# to in relation to the circumsphere of s1. > 0 indicates outside, 0 indicates
# boundary, < 0 indicates inside. See [2].
function pairwise_delaunay(m::Metric{N}, s1::Simplex{N, K}, s2::Simplex{N, K}) where {N, K}
    face_points = Point{N}[]
    i = 0
    for j in 1:K
        if s1.points[j] in s2.points
            push!(face_points, s1.points[j])
        else
            i = j
        end
    end
    @assert length(face_points) == K-1
    @assert i > 0
    p = first_setdiff(s2.points, face_points)
    b = rotate_about_face(m, s1, p, i)
    return transpose(b.coords) * distance_quadratic(m, s1) * b.coords
end

# functions for computing simplex centers
circumcenter(m::Metric{N}) where N = s::Simplex{N} -> circumsphere_barycentric(m, s)[1]

centroid(s::Simplex{N, K}) where {N, K} = Barycentric(s, SVector{K, Float64}(ones(K)/K))
