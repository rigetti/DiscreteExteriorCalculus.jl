using StaticArrays: SVector
using LinearAlgebra: I
using Combinatorics: combinations

function Point(coords::AbstractVector{<:Real})
    N = length(coords)
    return Point{N}(SVector{N, Float64}(coords))
end

Point(coords::Vararg{<:Real}) = Point(collect(coords))

Point(b::Barycentric) = Point(barycentric_matrix(b.simplex) * b.coords)

function Simplex(points::AbstractVector{Point{N}}) where N
    K = length(points)
    return Simplex{N, K}(SVector{K, Point{N}}(points))
end

Simplex(points::Vararg{Point{N}}) where N = Simplex(collect(points))

function Simplex(c::Cell)
    @assert length(c.points) == c.K
    return Simplex(c.points)
end

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

export subsimplices
"""
    subsimplices(s::Simplex{N, K}, k::Int) where {N, K}

Find all subsimplices of `s` with dimension `k-1`. If `k == 0` or `k > K`, return an
empty vector.
"""
function subsimplices(s::Simplex{N, K}, k::Int) where {N, K}
    @assert k >= 0
    if (k == 0) || (k > K)
        return Simplex{N, k}[]
    else
        return [Simplex(ps...) for ps in combinations(s.points, k)]
    end
end

export barycentric_matrix
"""
    barycentric_matrix(s::Simplex)

Find the matrix `M` so that `Mx` is the point corresponding to the barycentric
coordinates `x`.
"""
barycentric_matrix(s::Simplex) = hcat([p.coords for p in s.points]...)

export barycentric_projection_matrices
"""
    barycentric_projection_matrices(m::Metric{N}, s::Simplex{N, K}) where {N, K}

For a vector `p`, extremize `norm_square(m, p - barycentric_matrix(s) * x)` over `x`
with the constraint `sum(x) == 1`. The result is the barycentric coordinates of the
projection of `p` onto the affine subspace spanned by `s` and it can be written in the form
`x = A * p + B`. Return `A`, `B`.
"""
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

export distance_quadratic
"""
    distance_quadratic(m::Metric{N}, s::Simplex{N})

The squared distance from a point with barycentric coordinates `x` to a point `p`
can be written in the form `xᵀAx + Bᵀx` where `B` depends on `p` but `A` does not. Return A.
"""
distance_quadratic(m::Metric{N}, s::Simplex{N}) where N = -.5 * circumsphere_matrix(m, s)

export distance_linear
"""
    distance_linear(m::Metric{N}, s::Simplex{N}, p::Point{N})

The squared distance from a point with barycentric coordinates `x` to a point `p`
can be written in the form `xᵀAx + Bᵀx` where `B` depends on `p` but `A` does not. Return B.
"""
distance_linear(m::Metric{N}, s::Simplex{N}, p::Point{N}) where N = [norm_square(
    m, p.coords - q.coords) for q in s.points]

export distance_square
"""
    distance_square(m::Metric{N}, b::Barycentric{N}, p::Point{N}) where N

Find the squared distance between the points `Point(b)` and `p`. Note that this
value can be negative if the metric is not positive semi-definite.
"""
function distance_square(m::Metric{N}, b::Barycentric{N}, p::Point{N}) where N
    A = distance_quadratic(m, b.simplex)
    B = distance_linear(m, b.simplex, p)
    return transpose(b.coords) * A * b.coords + transpose(B) * b.coords
end

export circumsphere_matrix
"""
    circumsphere_matrix(m::Metric{N}, s::Simplex{N}) where N

If `x` are the barycentric coordinates of the circumcenter of `s` then there is a matrix M
such that `M * x = 2 * R² * ones(K)` where `R²` is the square circumradius. Return M.
"""
function circumsphere_matrix(m::Metric{N}, s::Simplex{N}) where N
    norm_squares = [[norm_square(m, p.coords - q.coords) for p in s.points] for q in s.points]
    return transpose(hcat(norm_squares...))
end

export circumsphere_barycentric
"""
    circumsphere_barycentric(m::Metric{N}, s::Simplex{N, K}) where {N, K}

Find the Barycentric of the circumcenter and the squared circumradius.
"""
function circumsphere_barycentric(m::Metric{N}, s::Simplex{N, K}) where {N, K}
    if K == 1
        return Barycentric(s, [1.0]), 0
    else
        coords = circumsphere_matrix(m, s) \ ones(K)
        sum_coords = sum(coords)
        return Barycentric(s, coords / sum_coords), .5/sum_coords
    end
end

export circumsphere
"""
    circumsphere(m::Metric{N}, s::Simplex{N}) where N

Find the circumcenter and the squared circumradius.
"""
function circumsphere(m::Metric{N}, s::Simplex{N}) where N
    b, R² = circumsphere_barycentric(m, s)
    return Point(b), R²
end

export circumcenter
"""
    circumcenter(m::Metric{N}) where N

Return a function taking a `Simplex{N}` to the Barycentric of its circumcenter.
"""
circumcenter(m::Metric{N}) where N = s::Simplex{N} -> circumsphere_barycentric(m, s)[1]

export centroid
"""
    centroid(s::Simplex{N, K}) where {N, K}

Compute the Barycentric of the centroid of a simplex.
"""
centroid(s::Simplex{N, K}) where {N, K} = Barycentric(s, SVector{K, Float64}(ones(K)/K))

"""
    barycentric_subspace(m::Metric{N}, s::Simplex{N, K}, f::Simplex{N, J},
        p::Point{N}) where {N, K, J}

Let `s` and `f` be simplices where `f` lies in the affine subspace spanned by `s`. For a
point `p`, the subset of points with the same projection onto `f` as that of `p` is a line.
This line can be represented in the form `Mx = v` where `x` are barycentric coordinates
with respect to `s`. Return M and v.
"""
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

"""
    rotate_about_face(m::Metric{N}, s::Simplex{N, K}, p::Point{N}, i::Int) where {N, K}

Let `s` be a simplex, `q = s.points[i]`, and `f` the face of `s` opposite `q`. Rotate the
point `p` about `f` into the affine subspace spanned by `s` and on the halfspace bounded
by `f` that does not contain `q`.
"""
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

"""
    pairwise_delaunay(m::Metric{N}, s1::Simplex{N, K}, s2::Simplex{N, K}) where {N, K}

Given two simplices `s1` and `s2` that share a face, rotate `s2` into the affine subspace
spanned by `s1`. Let `p` be the image of the the point of `s2` that is not in `s1` under
this rotation and let `c` and `R²` be the circumcenter and squared circumradius of `s1`.
Return the squared distance from `p` to `c` minus `R²`.
"""
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
