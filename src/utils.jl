using LinearAlgebra: I, nullspace
using StaticArrays: MMatrix

# M is a K-n × K full row rank matrix and v is a K-n length vector. The equation
# Mx = v describes an n-dimensional affine subspace. Find a and b so that this
# space is parametrized as x = a + b * x[1:n]. The columns of b are a basis
# for the null space of M and a is any particular solution of Mx=v.
# TODO implement this without the assumption that M has full row rank.
function parametrize_subspace(M::AbstractMatrix{<:Real}, v::AbstractVector{<:Real})
    n = size(M, 2) - size(M, 1)
    @assert n >= 0
    b = nullspace(M)
    N = inv([transpose(b); M]) # this is square because M has full row rank
    return N[:, (n+1):end] * v, b
end

# Intersect the line x = a + b * t with the quadratic
# xᵀAx + Bᵀx + C = 0 for its two solutions
function intersect_linear_quadratic(A::AbstractMatrix{<:Real}, B::AbstractVector{<:Real},
                                    C::Real, a::AbstractVector{<:Real},
                                    b::AbstractVector{<:Real})
    quadratic = transpose(b) * A * b
    linear = transpose(a) * A * b + transpose(b) * A * a + transpose(B) * b
    constant = transpose(a) * A * a + transpose(B) * a + C
    t1, t2 = solve_quadratic(quadratic, linear, constant)
    return a + b * t1, a + b * t2
end

# find the roots of ax² + bx + c = 0
function solve_quadratic(a::Real, b::Real, c::Real)
    t1 = -b/(2 * a)
    t2 = sqrt(b^2 - 4 * a * c)/(2 * a)
    return t1 + t2, t1 - t2
end

# given two arrays which are permutations of each other, find the sign of the permutation
function sign_of_permutation(a1::AbstractVector{T}, a2::AbstractVector{T}) where T
    n = length(a1)
    @assert length(a2) == n
    inds = [findfirst(isequal(x), a2) for x in a1]
    m = MMatrix{n,n,Int}(zeros(n,n)) # det is faster for small matrices using StaticArrays
    for i in 1:n
        m[i, inds[i]] = 1
    end
    return sign(det(m))
end

# Given two collections l1 and l2 find the index of the first element of l1
# not in l2. If no such element exists return 0.
function first_setdiff_index(l1, l2)
    for (i, e) in enumerate(l1)
        if !(e in l2)
            return i
        end
    end
    return 0
end

# Given two collections l1 and l2 find the first element of l1 not in l2.
# If no such element exists return nothing.
function first_setdiff(l1, l2)
    i = first_setdiff_index(l1, l2)
    return i > 0 ? l1[i] : nothing
end