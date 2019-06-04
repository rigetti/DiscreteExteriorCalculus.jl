module DiscreteExteriorCalculus
using StaticArrays: SVector, SMatrix
using LinearAlgebra: Symmetric
using UniqueVectors: UniqueVector

export Point
"""
    Point{N}(coords::SVector{N, Float64}) where N
    Point(coords::AbstractVector{<:Real})
    Point(coords::Vararg{<:Real})
    Point(b::Barycentric)

A point in `N` dimensional space.
"""
struct Point{N}
    coords::SVector{N, Float64}
end

export Simplex
"""
    Simplex{N, K}(points::SVector{K, Point{N}}) where {N, K}
    Simplex(points::AbstractVector{Point{N}}) where N
    Simplex(points::Vararg{Point{N}}) where N
    Simplex(c::Cell)

A simplex of dimension `K-1` embedded in `N` dimensional space.
"""
struct Simplex{N, K}
    points::SVector{K, Point{N}}
    function Simplex{N, K}(points::SVector{K, Point{N}}) where {N, K}
        @assert 1 <= K <= N+1
        return new{N, K}(points)
    end
end

SquareSMatrix{N} = SMatrix{N, N, Float64}

export Metric
"""
    Metric{N, T<:SMatrix{N, N, Float64}}(mat::Symmetric{Float64, T})
    Metric(mat::T) where {N, T<:SMatrix{N, N, Float64}}
    Metric(mat::AbstractMatrix{<:Real})
    Metric(N::Int)

A symmetric but not necessarily positive semi-definite metric. The
last method creates the Euclidean metric.
"""
struct Metric{N, T<:SquareSMatrix{N}}
    mat::Symmetric{Float64, T}
end

export Wedge
"""
    Wedge{N, K}(vectors::SVector{K, SVector{N, Float64}})
    Wedge(vectors::Vector{SVector{N, Float64}}) where N
    Wedge(s::Simplex{N, K}) where {N, K}

The wedge product of `K` vectors in `N` dimensional space. The last
method creates the wedge product of all vectors emanating from the first
vertex of a simplex.
"""
struct Wedge{N, K}
    vectors::SVector{K, SVector{N, Float64}}
end

export Barycentric
"""
    Barycentric{N, K}(s::Simplex{N, K}, coords::SVector{K, Float64}) where {N, K}
    Barycentric(s::Simplex{N, K}, coords::AbstractVector{<:Real}) where {N, K}
    Barycentric(s::Simplex, coords::Vararg{<:Real})
    Barycentric(m::Metric{N}, s::Simplex{N}, p::Point{N}) where N

A representation of a point with respect to a simplex using barycentric
coordinates. The last method produces the Barycentric for the projection of
a point onto the affine subspace spanned by a simplex according to a given
metric.
"""
struct Barycentric{N, K}
    simplex::Simplex{N, K}
    coords::SVector{K, Float64}
    function Barycentric{N, K}(s::Simplex{N, K}, coords::SVector{K, Float64}) where {N, K}
        @assert sum(coords) ≈ 1
        if sum(coords) != 1
            coords = SVector{K, Float64}([coords[1:end-1]..., 1 - sum(coords[1:end-1])])
        end
        return new{N, K}(s, coords)
    end
end


export Cell
"""
    Cell{N}(children::Vector{Cell{N}}, parents::Dict{Cell{N}, Bool},
        points::Vector{Point{N}}, K::Int) where N
    Cell(points::AbstractVector{Point{N}}, K::Int) where N
    Cell(s::Simplex{N, K}) where {N, K}

A polytope of dimension `K-1` embedded in `N` dimensional space.
"""
struct Cell{N}
    children::Vector{Cell{N}} # (K-2) dimensional cells
    parents::Dict{Cell{N}, Bool} # K dimensional cells => relative orientations
    points::Vector{Point{N}} # all vertices of the cell
    K::Int # dimension + 1
    function Cell{N}(children::Vector{Cell{N}}, parents::Dict{Cell{N}, Bool},
                     points::Vector{Point{N}}, K::Int) where N
        @assert 1 <= K <= N+1
        for c in children
            @assert c.K == K-1
        end
        for c in keys(parents)
            @assert c.K == K+1
        end
        return new{N}(children, parents, points, K)
    end
end

export CellComplex
"""
    CellComplex{N,K}(cells::SVector{K, UniqueVector{Cell{N}}}) where {N, K}
    CellComplex{N,K}() where {N, K}
    CellComplex{N,K}(cells::AbstractVector{Cell{N}}) where {N, K}
    CellComplex(simplices::AbstractVector{Simplex{N, K}})
    CellComplex(cells::AbstractVector{Cell{N}}) where N

A cell complex with cells of dimension `0` through `K-1` embedded in `N`
dimensional space. The last method creates a cell complex consisting of all
descendants of `cells`.
"""
struct CellComplex{N, K}
    cells::SVector{K, UniqueVector{Cell{N}}}
    function CellComplex{N,K}(cells::SVector{K, UniqueVector{Cell{N}}}) where {N, K}
        @assert 1 <= K <= N+1
        for k in 1:K
            for c in cells[k]
                @assert c.K == k
            end
        end
        return new{N,K}(cells)
    end
end

SignedSimplex{N} = Tuple{Vector{Point{N}}, Bool}

export TriangulatedComplex
"""
    TriangulatedComplex{N, K}(complex::CellComplex{N, K},
        simplices::Dict{Cell{N}, Vector{Tuple{Vector{Point{N}}, Bool}}})
    TriangulatedComplex{N,K}() where {N, K}
    TriangulatedComplex(simplices::AbstractVector{Simplex{N, K}}) where {N, K}

A cell complex along with a decomposition of each cell into simplices. The
decomposition comes with signs which are used to compute a signed volume for
each cell.
"""
struct TriangulatedComplex{N, K}
    complex::CellComplex{N, K}
    simplices::Dict{Cell{N}, Vector{SignedSimplex{N}}}
end

export Mesh
"""
    Mesh{N,K}(primal::TriangulatedComplex{N,K}, dual::TriangulatedComplex{N,K})
    Mesh(tcomp::TriangulatedComplex{N}, center::Function) where N

A pair of TriangulatedComplexes representing a primal and dual pair. The second
constructor takes a primal TriangulatedComplex and a function `center` that
takes a simplex to a point (e.g. its circumcenter or centroid) and constructs
the corresponding dual TriangulatedComplex.
"""
struct Mesh{N,K}
    primal::TriangulatedComplex{N,K}
    dual::TriangulatedComplex{N,K}
end

include("utils.jl")
include("geometry.jl")
include("cell_complex.jl")
include("mesh.jl")
include("operators.jl")
include("triangulation.jl")
end
