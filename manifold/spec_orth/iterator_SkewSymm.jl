include("../../inc/global_path.jl")

using LoopVectorization

import Base.Threads.@threads



@inline mat2vec_ind(leading_dim::Int, first_ind::Int, second_ind::Int) = (second_ind - 1) * leading_dim + first_ind
@inline function vec2mat_ind(leading_dim::Int, vec_ind::Int)
    d = div(vec_ind, leading_dim)
    r = vec_ind - d * leading_dim
    return r == 0 ? (d - 1, leading_dim) : (d, r);
end

mutable struct STRICT_LOWER_ITERATOR
    leading_dim::Int
    offset::Int
    mat_dim::Int
    vec_dim::Int
    mat2vec::Ref{Matrix{Int}}
    vec2mat::Ref{Matrix{Int}}
    vec2lower::Ref{Vector{Int}}
    vec2upper::Ref{Vector{Int}}

    STRICT_LOWER_ITERATOR(n::Int) = new(n, 0, n, div(n * (n - 1), 2), Matrix{Int}(undef, n, n), Matrix{Int}(undef, div(n * (n - 1), 2), 2), Matrix{Int}(undef, div(n * (n - 1), 2)), Matrix{Int}(undef, div(n * (n - 1), 2)))
    STRICT_LOWER_ITERATOR(n::Int, alg::Function; os::Int = 0, ed::Int = n) = new(n, os, ed - os, div((ed - os) * (ed - os - 1), 2), alg(n; os = os, ed = ed)...)
    STRICT_LOWER_ITERATOR(n::Int, os::Int, ed::Int, alg::Function) = new(n, os, ed - os, div((ed - os) * (ed - os - 1), 2), alg(n; os = os, ed = ed)...)
end


function lower_col_traversal(leading_dim::Int; os::Int = 0, ed::Int = leading_dim)
    mat_dim::Int = ed - os;
    mat2vec = zeros(Int, leading_dim, leading_dim);
    vec2mat = Matrix{Int}(undef, div(mat_dim * (mat_dim - 1), 2), 2)
    vec2lower = Vector{Int}(undef, div(mat_dim * (mat_dim - 1), 2))
    vec2upper = Vector{Int}(undef, div(mat_dim * (mat_dim - 1), 2))



    v_ind::Int = 1;

    for c_ind in 1:mat_dim
        for r_ind in (c_ind + 1):mat_dim
            @inbounds mat2vec[r_ind + os, c_ind + os] = v_ind;
            @inbounds mat2vec[c_ind + os, r_ind + os] = v_ind;

            @inbounds vec2mat[v_ind, 1] = r_ind + os;
            @inbounds vec2mat[v_ind, 2] = c_ind + os;

            @inbounds vec2lower[v_ind] = mat2vec_ind(leading_dim, r_ind + os, c_ind + os);
            @inbounds vec2upper[v_ind] = mat2vec_ind(leading_dim, c_ind + os, r_ind + os);

            v_ind += 1;
        end
    end
    return mat2vec, vec2mat, vec2lower, vec2upper                                          
end

function lower_blk_traversal(leading_dim::Int; os::Int = 0, ed::Int = leading_dim)

    mat_dim = ed - os;
    blk_dim = div(mat_dim, 2);

    mat2vec = zeros(Int, leading_dim, leading_dim);
    vec2mat = Matrix{Int}(undef, div(mat_dim * (mat_dim - 1), 2), 2)
    vec2lower = Vector{Int}(undef, div(mat_dim * (mat_dim - 1), 2))
    vec2upper = Vector{Int}(undef, div(mat_dim * (mat_dim - 1), 2))



    v_ind::Int = 1;
    
    # 2 × 2 lower triangular blocks
    for c_ind = 1:blk_dim
        for r_ind = (c_ind + 1):blk_dim
            @inbounds mat2vec[2 * r_ind - 1 + os, 2 * c_ind - 1 + os] = v_ind;
            @inbounds mat2vec[2 * c_ind - 1 + os, 2 * r_ind - 1 + os] = v_ind;

            @inbounds mat2vec[2 * r_ind + os, 2 * c_ind - 1 + os] = v_ind + 1;
            @inbounds mat2vec[2 * c_ind - 1 + os, 2 * r_ind + os] = v_ind + 1;

            @inbounds mat2vec[2 * r_ind - 1 + os, 2 * c_ind + os] = v_ind + 2;
            @inbounds mat2vec[2 * c_ind + os, 2 * r_ind - 1 + os] = v_ind + 2;

            @inbounds mat2vec[2 * r_ind + os, 2 * c_ind + os] = v_ind + 3;
            @inbounds mat2vec[2 * c_ind + os, 2 * r_ind + os] = v_ind + 3;

            @inbounds vec2mat[v_ind, 1] = 2 * r_ind - 1 + os;
            @inbounds vec2mat[v_ind, 2] = 2 * c_ind - 1 + os;

            @inbounds vec2mat[v_ind + 1, 1] = 2 * r_ind + os;
            @inbounds vec2mat[v_ind + 1, 2] = 2 * c_ind - 1 + os;

            @inbounds vec2mat[v_ind + 2, 1] = 2 * r_ind - 1 + os;
            @inbounds vec2mat[v_ind + 2, 2] = 2 * c_ind + os;

            @inbounds vec2mat[v_ind + 3, 1] = 2 * r_ind + os;
            @inbounds vec2mat[v_ind + 3, 2] = 2 * c_ind + os;


            @inbounds vec2lower[v_ind] = mat2vec_ind(leading_dim, 2 * r_ind - 1 + os, 2 * c_ind - 1 + os);
            @inbounds vec2lower[v_ind + 1] = mat2vec_ind(leading_dim, 2 * r_ind + os, 2 * c_ind - 1 + os);
            @inbounds vec2lower[v_ind + 2] = mat2vec_ind(leading_dim, 2 * r_ind - 1 + os, 2 * c_ind + os);
            @inbounds vec2lower[v_ind + 3] = mat2vec_ind(leading_dim, 2 * r_ind + os, 2 * c_ind + os);

            @inbounds vec2upper[v_ind] = mat2vec_ind(leading_dim, 2 * c_ind - 1 + os, 2 * r_ind - 1 + os);
            @inbounds vec2upper[v_ind + 1] = mat2vec_ind(leading_dim, 2 * c_ind - 1 + os, 2 * r_ind + os);
            @inbounds vec2upper[v_ind + 2] = mat2vec_ind(leading_dim, 2 * c_ind + os, 2 * r_ind - 1 + os);
            @inbounds vec2upper[v_ind + 3] = mat2vec_ind(leading_dim, 2 * c_ind + os, 2 * r_ind + os);

            v_ind += 4;
        end
    end

    # 1 × 2 leftover blocks

    if isodd(mat_dim)
        for c_ind = 1:blk_dim
            @inbounds mat2vec[ed, 2 * c_ind - 1 + os] = v_ind;
            @inbounds mat2vec[2 * c_ind - 1 + os, ed] = v_ind;

            @inbounds mat2vec[ed, 2 * c_ind + os] = v_ind + 1;
            @inbounds mat2vec[2 * c_ind + os, ed] = v_ind + 1;

            @inbounds vec2mat[v_ind, 1] = ed;
            @inbounds vec2mat[v_ind, 2] = 2 * c_ind - 1 + os;

            @inbounds vec2mat[v_ind + 1, 1] = ed;
            @inbounds vec2mat[v_ind + 1, 2] = 2 * c_ind + os;

            @inbounds vec2lower[v_ind] = mat2vec_ind(leading_dim, ed, 2 * c_ind - 1 + os);
            @inbounds vec2lower[v_ind + 1] = mat2vec_ind(leading_dim, ed, 2 * c_ind + os);


            @inbounds vec2upper[v_ind] = mat2vec_ind(leading_dim, 2 * c_ind - 1 + os, ed);
            @inbounds vec2upper[v_ind + 1] = mat2vec_ind(leading_dim, 2 * c_ind + os, ed);

            v_ind += 2;
        end
    end

    # 2 × 2 diagonal blocks

    for d_ind = 1:blk_dim

        @inbounds mat2vec[2 * d_ind + os, 2 * d_ind - 1 + os] = v_ind;
        @inbounds mat2vec[2 * d_ind - 1 + os, 2 * d_ind + os] = v_ind;

        @inbounds vec2mat[v_ind, 1] = 2 * d_ind + os;
        @inbounds vec2mat[v_ind, 2] = 2 * d_ind - 1 + os;

        @inbounds vec2lower[v_ind] = mat2vec_ind(leading_dim, 2 * d_ind + os, 2 * d_ind - 1 + os);

        @inbounds vec2upper[v_ind] = mat2vec_ind(leading_dim, 2 * d_ind - 1 + os, 2 * d_ind + os);
        v_ind += 1;
    end

    return mat2vec, vec2mat, vec2lower, vec2upper
end

@inline function _SkewSymm_mat2vec_by_iterator!(V::Ref{Vector{elty}}, vos::Int, M::Ref{Matrix{elty}}, it::STRICT_LOWER_ITERATOR; lower::Bool = true) where elty
    Vec = V[];
    Mat = M[];

    if lower
        Vec2LowerMat = it.vec2lower[]
    
        @tturbo for ind in eachindex(Vec2LowerMat)
            @inbounds Vec[ind + vos] = Mat[Vec2LowerMat[ind]];
        end
    else
        Vec2UpperMat = it.vec2upper[]
    
        @tturbo for ind in eachindex(Vec2UpperMat)
            @inbounds Vec[ind + vos] = - Mat[Vec2UpperMat[ind]];
        end
    end
end

@inline function _SkewSymm_mat2vec_by_iterator!(V::Ref{Vector{elty}}, M::Ref{Matrix{elty}}, it::STRICT_LOWER_ITERATOR; lower::Bool = true) where elty
    Vec = V[];
    Mat = M[];

    if lower
        Vec2LowerMat = it.vec2lower[]
    
        @tturbo for ind in eachindex(Vec2LowerMat)
            @inbounds Vec[ind] = Mat[Vec2LowerMat[ind]];
        end
    else
        Vec2UpperMat = it.vec2upper[]
    
        @tturbo for ind in eachindex(Vec2UpperMat)
            @inbounds Vec[ind] = - Mat[Vec2UpperMat[ind]];
        end
    end
end

@inline function _SkewSymm_mat2vec_by_iterator!(V::AbstractVector{elty}, M::AbstractMatrix{elty}, it::STRICT_LOWER_ITERATOR; lower::Bool = true) where elty
    if lower
        Vec2LowerMat = it.vec2lower[]
    
        @tturbo for ind in eachindex(Vec2LowerMat)
            @inbounds V[ind] = M[Vec2LowerMat[ind]];
        end
    else
        Vec2UpperMat = it.vec2upper[]
    
        @tturbo for ind in eachindex(Vec2UpperMat)
            @inbounds V[ind] = - M[Vec2UpperMat[ind]];
        end
    end
end

@inline function _SkewSymm_vec2mat_by_iterator!(M::Ref{Matrix{elty}}, it::STRICT_LOWER_ITERATOR, V::Ref{Vector{elty}}; fil::Bool = false) where elty
    Vec = V[];
    Mat = M[];

    if fil
        Vec2LowerMat = it.vec2lower[]
        Vec2UpperMat = it.vec2upper[]
        
        @tturbo for ind in eachindex(Vec2LowerMat)
            @inbounds Mat[Vec2LowerMat[ind]] = Vec[ind];
            @inbounds Mat[Vec2UpperMat[ind]] = - Vec[ind];
        end
    else
        Vec2LowerMat = it.vec2lower[]
        
        @tturbo for ind in eachindex(Vec2LowerMat)
            @inbounds Mat[Vec2LowerMat[ind]] = Vec[ind];
        end
    end 
end


@inline function _SkewSymm_vec2mat_by_iterator!(M::Ref{Matrix{elty}}, it::STRICT_LOWER_ITERATOR, V::Ref{Vector{elty}}, vos::Int; fil::Bool = false) where elty
    Vec = V[];
    Mat = M[];

    if fil
        Vec2LowerMat = it.vec2lower[]
        Vec2UpperMat = it.vec2upper[]
        
        @tturbo for ind in eachindex(Vec2LowerMat)
            @inbounds Mat[Vec2LowerMat[ind]] = Vec[ind + vos];
            @inbounds Mat[Vec2UpperMat[ind]] = - Vec[ind + vos];
        end
    else
        Vec2LowerMat = it.vec2lower[]
        
        @tturbo for ind in eachindex(Vec2LowerMat)
            @inbounds Mat[Vec2LowerMat[ind]] = Vec[ind + vos];
        end
    end 
end



@inline function _SkewSymm_vec2mat_by_iterator!(M::AbstractMatrix{elty}, it::STRICT_LOWER_ITERATOR, V::AbstractVector{elty}; fil::Bool = false) where elty
    if fil
        Vec2LowerMat = it.vec2lower[]
        Vec2UpperMat = it.vec2upper[]
        
        @tturbo for ind in eachindex(Vec2LowerMat)
            @inbounds M[Vec2LowerMat[ind]] = V[ind];
            @inbounds M[Vec2UpperMat[ind]] = - V[ind];
        end
    else
        Vec2LowerMat = it.vec2lower[]
        
        @tturbo for ind in eachindex(Vec2LowerMat)
            @inbounds M[Vec2LowerMat[ind]] = V[ind];
        end
    end 
end



"""
    vec_SkewSymm_col!(v, S, [it::STRICT_LOWER_ITERATOR]; os = 0, ed = size(S, 1), lower = true) -> v::Ref{Vector{Float64}}

Vectorize the lower triangular part of skew symmetric `M` in column-major order, write them at V. When `lower = false`, the negation of the upper triangular part of M in row-major order is used.
#Example
```julia-repl

```
"""
function vec_SkewSymm_col!(v::Ref{Vector{elty}}, vos::Int, S::Ref{Matrix{elty}}, mos::Int, med::Int; lower::Bool = true) where elty

    leading_dim::Int = size(S[], 1)
    elsize = sizeof(elty)

    if lower
        ptrV = pointer(v[]) + elsize * vos
        ptrS = pointer(S[])
        for c_ind = (mos + 1):med
            unsafe_copyto!(ptrV, ptrS + elsize * (mat2vec_ind(leading_dim, c_ind + 1, c_ind)- 1), med - c_ind)
            ptrV = ptrV + elsize * (med - c_ind);
        end
    else
        Vec = v[];
        Mat = S[];
        v_ind::Int = vos;
        for r_ind in (os + 1):ed
            for c_ind in (r_ind + 1):ed
                Vec[v_ind] = - Mat[r_ind, c_ind];
                v_ind = v_ind + 1;
            end
        end
    end

    return v
end

vec_SkewSymm_col!(v::Ref{Vector{elty}}, S::Ref{Matrix{elty}}, mos::Int, med::Int; lower::Bool = true) where elty =
    vec_SkewSymm_col!(v, 0, S, mos, med; lower = lower)

vec_SkewSymm_col!(v::Ref{Vector{elty}}, S::Ref{Matrix{elty}}; lower::Bool = true) where elty =
    vec_SkewSymm_col!(v, 0, S, 0, size(S[], 1); lower = lower)

vec_SkewSymm_col!(v::Ref{Vector{elty}}, vos::Int, S::Ref{Matrix{elty}}, it::STRICT_LOWER_ITERATOR; lower::Bool = true) where elty =
    _SkewSymm_mat2vec_by_iterator!(v, vos, S, it; lower = lower)

vec_SkewSymm_col!(v::Ref{Vector{elty}}, S::Ref{Matrix{elty}}, it::STRICT_LOWER_ITERATOR; lower::Bool = true) where elty =
    _SkewSymm_mat2vec_by_iterator!(v, S, it; lower = lower)

"""
    mat_SkewSymm_col!(v, S, [it::STRICT_LOWER_ITERATOR]; os = 0, ed = size(S, 1)) -> v::Ref{Vector{Float64}}


Matricize the vector `V` holding the lower triangular part of skew symmetric `M` in column major, overwrites the lower triangular of `M` of fill the entire `M` if `fil = true`.
#Example
```julia-repl

```
"""
function mat_SkewSymm_col!(S::Ref{Matrix{elty}}, mos::Int, med::Int, v::Ref{Vector{elty}}, vos::Int; fil::Bool = true) where elty

    elsize = sizeof(elty)

    ptrV = pointer(v[]) + sizeof(elty) * vos;
    ptrS = pointer(S[])
    leading_dim::Int = size(S[], 1)

    for c_ind = (mos + 1):med
        unsafe_copyto!(ptrS + elsize * (mat2vec_ind(leading_dim, c_ind + 1, c_ind)- 1), ptrV, med - c_ind)
        ptrV = ptrV + elsize * (med - c_ind);
    end

    if fil
        fill_upper_SkewSymm!(S);
    end

    return S;
end

mat_SkewSymm_col!(S::Ref{Matrix{elty}}, mos::Int, med::Int, v::Ref{Vector{elty}}; fil::Bool = true) where elty =
    mat_SkewSymm_col!(S, mos, med, v, 0; fil = fil)

mat_SkewSymm_col!(S::Ref{Matrix{elty}}, v::Ref{Vector{elty}}, vos::Int; fil::Bool = true) where elty =
    mat_SkewSymm_col!(S, 0, size(S[], 1), v, vos; fil = fil)

mat_SkewSymm_col!(S::Ref{Matrix{elty}}, v::Ref{Vector{elty}}; fil::Bool = true) where elty =
    mat_SkewSymm_col!(S, 0, size(S[], 1), v, 0; fil = fil)

mat_SkewSymm_col!(S::Ref{Matrix{elty}}, it::STRICT_LOWER_ITERATOR, v::Ref{Vector{elty}}, vos::Int; fil::Bool = true) where elty = 
    _SkewSymm_vec2mat_by_iterator!(S, it, v, vos; fil = fil);

"""
    vec_SkewSymm_blk!(v, S; os = 0, ed = size(S, 1)) -> v::Ref{Vector{Float64}}

Convert the block vectorized skew symmetric submatrix stored in v to submatrix in S[os+1:ed, os+1:ed]. See also ``vec_SkewSymm_blk!`` for the reverse conversion. 

For S[os+1:ed, os+1:ed] partited by `2 × 2` blocks from top left, the first `4m(m - 1)/2` entries, where `m = div(ed - os, 2)`, sequentially stores the `2 × 2` lower triangular blocks, `m(m - 1)/2` in total, in column major order. Then the following `m` entries sequentially stores the bottom left entries from diagonal blocks. If ed - os is odd, the remaining `ed - os - 1` entries stores the leftover row.

#Example
```julia-repl
julia> S = zeros(7, 7)
julia> v = [1., 2., 3., 4., 5., 6., 7., 8., 9., 10.];
julia> mat_SkewSymm_blk!(Ref(S), Ref(v); os = 2, fil = false);
julia> v
10-element Vector{Float64}:
  1.0
  2.0
  3.0
  4.0
  5.0
  6.0
  7.0
  8.0
  9.0
 10.0
julia> S
7×7 Matrix{Float64}:
  0.0  0.0  0.0  0.0  0.0   0.0  0.0
  0.0  0.0  0.0  0.0  0.0   0.0  0.0
  0.0  0.0  0.0  0.0  0.0   0.0  0.0
  0.0  0.0  5.0  0.0  0.0   0.0  0.0
  0.0  0.0  1.0  3.0  0.0   0.0  0.0
  0.0  0.0  2.0  4.0  6.0   0.0  0.0
  0.0  0.0  7.0  8.0  9.0  10.0  0.0
```
"""
function vec_SkewSymm_blk!(v::Ref{Vector{elty}}, vos::Int, S::Ref{Matrix{elty}}, mos::Int, med::Int; lower::Bool = true) where elty
    Mat = S[];
    Vec = v[];

    mat_dim::Int = med - mos;
    blk_dim::Int = div(mat_dim, 2);

    v_ind::Int = vos + 1;

    if lower
        # 2 × 2 lower triangular blocks
        for c_ind = 1:blk_dim
            for r_ind = (c_ind + 1):blk_dim
                @inbounds Vec[v_ind] = Mat[2 * r_ind - 1 + mos, 2 * c_ind - 1 + mos];
                @inbounds Vec[v_ind + 1] = Mat[2 * r_ind + mos, 2 * c_ind - 1 + mos];
                @inbounds Vec[v_ind + 2] = Mat[2 * r_ind - 1 + mos, 2 * c_ind + mos];
                @inbounds Vec[v_ind + 3] = Mat[2 * r_ind + mos, 2 * c_ind + mos];
                v_ind += 4;
            end
        end

        # 1 × 2 leftover blocks

        if isodd(mat_dim)
            for c_ind = 1:blk_dim
                @inbounds Vec[v_ind] = Mat[med, 2 * c_ind - 1 + mos];
                @inbounds Vec[v_ind + 1] = Mat[med, 2 * c_ind + mos];
                v_ind += 2;
            end
        end

        # 2 × 2 diagonal blocks

        for d_ind = 1:blk_dim
            @inbounds Vec[v_ind] = Mat[2 * d_ind + mos, 2 * d_ind - 1 + mos];
            v_ind += 1;
        end
    else
        for c_ind = 1:blk_dim
            for r_ind = (c_ind + 1):blk_dim
                @inbounds Vec[v_ind] = - Mat[2 * c_ind - 1 + mos, 2 * r_ind - 1 + mos];
                @inbounds Vec[v_ind + 1] = - Mat[2 * c_ind - 1 + mos, 2 * r_ind + mos];
                @inbounds Vec[v_ind + 2] = - Mat[2 * c_ind + mos, 2 * r_ind - 1 + mos];
                @inbounds Vec[v_ind + 3] = - Mat[2 * c_ind + mos, 2 * r_ind + mos];
                v_ind += 4;
            end
        end

        # 1 × 2 leftover blocks

        if isodd(mat_dim)
            for c_ind = 1:blk_dim
                @inbounds Vec[v_ind] = - Mat[2 * c_ind - 1 + mos, med];
                @inbounds Vec[v_ind + 1] = - Mat[2 * c_ind + mos, med];
                v_ind += 2;
            end
        end

        # 2 × 2 diagonal blocks

        for d_ind = 1:blk_dim
            @inbounds Vec[v_ind] = - Mat[2 * d_ind - 1 + mos, 2 * d_ind + mos];
            v_ind += 1;
        end
    end

    return v;
end

vec_SkewSymm_blk!(v::Ref{Vector{elty}}, S::Ref{Matrix{elty}}, mos::Int, med::Int; lower::Bool = true) where elty = vec_SkewSymm_blk!(v, 0, S, mos, med; lower = lower)

function vec_SkewSymm_blk!(v::Ref{Vector{elty}}, S::Ref{Matrix{elty}}; lower::Bool = true) where elty
    Mat = S[];
    Vec = v[];

    mat_dim::Int = size(Mat, 1)
    blk_dim::Int = div(mat_dim, 2);

    v_ind::Int = 1;

    if lower
        # 2 × 2 lower triangular blocks
        for c_ind = 1:blk_dim
            for r_ind = (c_ind + 1):blk_dim
                @inbounds Vec[v_ind] = Mat[2 * r_ind - 1, 2 * c_ind - 1];
                @inbounds Vec[v_ind + 1] = Mat[2 * r_ind, 2 * c_ind - 1];
                @inbounds Vec[v_ind + 2] = Mat[2 * r_ind - 1, 2 * c_ind];
                @inbounds Vec[v_ind + 3] = Mat[2 * r_ind, 2 * c_ind];
                v_ind += 4;
            end
        end

        # 1 × 2 leftover blocks

        if isodd(mat_dim)
            for c_ind = 1:blk_dim
                @inbounds Vec[v_ind] = Mat[mat_dim, 2 * c_ind - 1];
                @inbounds Vec[v_ind + 1] = Mat[mat_dim, 2 * c_ind];
                v_ind += 2;
            end
        end

        # 2 × 2 diagonal blocks

        for d_ind = 1:blk_dim
            @inbounds Vec[v_ind] = Mat[2 * d_ind, 2 * d_ind - 1];
            v_ind += 1;
        end
    else
        for c_ind = 1:blk_dim
            for r_ind = (c_ind + 1):blk_dim
                @inbounds Vec[v_ind] = - Mat[2 * c_ind - 1, 2 * r_ind - 1];
                @inbounds Vec[v_ind + 1] = - Mat[2 * c_ind - 1, 2 * r_ind];
                @inbounds Vec[v_ind + 2] = - Mat[2 * c_ind, 2 * r_ind - 1];
                @inbounds Vec[v_ind + 3] = - Mat[2 * c_ind, 2 * r_ind];
                v_ind += 4;
            end
        end

        # 1 × 2 leftover blocks

        if isodd(mat_dim)
            for c_ind = 1:blk_dim
                @inbounds Vec[v_ind] = - Mat[2 * c_ind - 1, ed];
                @inbounds Vec[v_ind + 1] = - Mat[2 * c_ind, ed];
                v_ind += 2;
            end
        end

        # 2 × 2 diagonal blocks

        for d_ind = 1:blk_dim
            @inbounds Vec[v_ind] = - Mat[2 * d_ind - 1, 2 * d_ind];
            v_ind += 1;
        end
    end

    return v;
end

vec_SkewSymm_blk!(v::Ref{Vector{elty}}, vos::Int, S::Ref{Matrix{elty}}, it::STRICT_LOWER_ITERATOR; lower::Bool = true) where elty = 
    it.mat_dim < 180 ? vec_SkewSymm_blk!(v, vos, S, it.offset, it.offset + it.mat_dim; lower = lower) : _SkewSymm_mat2vec_by_iterator!(v, vos, S, it; lower = lower)

vec_SkewSymm_blk!(v::Ref{Vector{elty}}, S::Ref{Matrix{elty}}, it::STRICT_LOWER_ITERATOR; lower::Bool = true) where elty = 
    vec_SkewSymm_blk!(v, 0, S, it; lower = lower)

function vec_SkewSymm_blk!(v::AbstractVector{elty}, S::AbstractMatrix{elty}, os::Int, ed::Int; lower::Bool = true) where elty
    mat_dim::Int = ed - os;
    blk_dim::Int = div(mat_dim, 2);

    v_ind::Int = 1;

    if lower
        # 2 × 2 lower triangular blocks
        for c_ind = 1:blk_dim
            for r_ind = (c_ind + 1):blk_dim
                @inbounds v[v_ind] = S[2 * r_ind - 1 + os, 2 * c_ind - 1 + os];
                @inbounds v[v_ind + 1] = S[2 * r_ind + os, 2 * c_ind - 1 + os];
                @inbounds v[v_ind + 2] = S[2 * r_ind - 1 + os, 2 * c_ind + os];
                @inbounds v[v_ind + 3] = S[2 * r_ind + os, 2 * c_ind + os];
                v_ind += 4;
            end
        end

        # 1 × 2 leftover blocks

        if isodd(mat_dim)
            for c_ind = 1:blk_dim
                @inbounds v[v_ind] = S[ed, 2 * c_ind - 1 + os];
                @inbounds v[v_ind + 1] = S[ed, 2 * c_ind + os];
                v_ind += 2;
            end
        end

        # 2 × 2 diagonal blocks

        for d_ind = 1:blk_dim
            @inbounds v[v_ind] = S[2 * d_ind + os, 2 * d_ind - 1 + os];
            v_ind += 1;
        end
    else
        for c_ind = 1:blk_dim
            for r_ind = (c_ind + 1):blk_dim
                @inbounds v[v_ind] = - S[2 * c_ind - 1 + os, 2 * r_ind - 1 + os];
                @inbounds v[v_ind + 1] = - S[2 * c_ind - 1 + os, 2 * r_ind + os];
                @inbounds v[v_ind + 2] = - S[2 * c_ind + os, 2 * r_ind - 1 + os];
                @inbounds v[v_ind + 3] = - S[2 * c_ind + os, 2 * r_ind + os];
                v_ind += 4;
            end
        end

        # 1 × 2 leftover blocks

        if isodd(mat_dim)
            for c_ind = 1:blk_dim
                @inbounds v[v_ind] = - S[2 * c_ind - 1 + os, ed];
                @inbounds v[v_ind + 1] = - S[2 * c_ind + os, ed];
                v_ind += 2;
            end
        end

        # 2 × 2 diagonal blocks

        for d_ind = 1:blk_dim
            @inbounds v[v_ind] = - S[2 * d_ind - 1 + os, 2 * d_ind + os];
            v_ind += 1;
        end
    end

    return v;
end

vec_SkewSymm_blk!(v::AbstractVector{elty}, S::AbstractMatrix{elty}, it::STRICT_LOWER_ITERATOR; lower::Bool = true) where elty = it.mat_dim < 180 ? vec_SkewSymm_blk!(v, S, it.offset, it.offset + it.mat_dim; lower = lower) : _SkewSymm_mat2vec_by_iterator!(v, S, it; lower = lower)


"""
Threading is always better
"""
function mat_SkewSymm_blk!(S::Ref{Matrix{elty}}, mos::Int, med::Int, v::Ref{Vector{elty}}, vos::Int; fil::Bool = true) where elty
    Mat = S[];
    Vec = v[];

    mat_dim::Int = med - mos;
    blk_dim::Int = div(mat_dim, 2);

    ## !!!! Bound check required here. !!!!

    v_ind::Int = 1 + vos;
    
    # 2 × 2 lower triangular blocks
    for c_ind = 1:blk_dim
        for r_ind = (c_ind + 1):blk_dim
            @inbounds Mat[2 * r_ind - 1 + mos, 2 * c_ind - 1 + mos] = Vec[v_ind];
            @inbounds Mat[2 * r_ind + mos, 2 * c_ind - 1 + mos] = Vec[v_ind + 1];
            @inbounds Mat[2 * r_ind - 1 + mos, 2 * c_ind + mos] = Vec[v_ind + 2];
            @inbounds Mat[2 * r_ind + mos, 2 * c_ind + mos] = Vec[v_ind + 3];
            v_ind += 4;
        end
    end

    # 1 × 2 leftover blocks

    if isodd(mat_dim)
        for c_ind = 1:blk_dim
            @inbounds Mat[med, 2 * c_ind - 1 + mos] = Vec[v_ind];
            @inbounds Mat[med, 2 * c_ind + mos] = Vec[v_ind + 1];
            v_ind += 2;
        end
    end

    # 2 × 2 diagonal blocks

    for d_ind = 1:blk_dim
        @inbounds Mat[2 * d_ind + mos, 2 * d_ind - 1 + mos] = Vec[v_ind];
        v_ind += 1;
    end

    if fil
        fill_upper_SkewSymm!(S);
    end

    return S;
end

mat_SkewSymm_blk!(S::Ref{Matrix{elty}}, mos::Int, med::Int, v::Ref{Vector{elty}}; fil::Bool = true) where elty = 
    mat_SkewSymm_blk!(S, mos, med, v, 0; fil = fil)

function mat_SkewSymm_blk!(S::Ref{Matrix{elty}}, v::Ref{Vector{elty}}; fil::Bool = true) where elty
    Mat = S[];
    Vec = v[];

    mat_dim::Int = size(Mat, 1);
    blk_dim::Int = div(mat_dim, 2);

    ## !!!! Bound check required here. !!!!

    v_ind::Int = 1;
    
    # 2 × 2 lower triangular blocks
    for c_ind = 1:blk_dim
        for r_ind = (c_ind + 1):blk_dim
            @inbounds Mat[2 * r_ind - 1, 2 * c_ind - 1] = Vec[v_ind];
            @inbounds Mat[2 * r_ind, 2 * c_ind - 1] = Vec[v_ind + 1];
            @inbounds Mat[2 * r_ind - 1, 2 * c_ind] = Vec[v_ind + 2];
            @inbounds Mat[2 * r_ind, 2 * c_ind] = Vec[v_ind + 3];
            v_ind += 4;
        end
    end

    # 1 × 2 leftover blocks

    if isodd(mat_dim)
        for c_ind = 1:blk_dim
            @inbounds Mat[ed, 2 * c_ind - 1] = Vec[v_ind];
            @inbounds Mat[ed, 2 * c_ind] = Vec[v_ind + 1];
            v_ind += 2;
        end
    end

    # 2 × 2 diagonal blocks

    for d_ind = 1:blk_dim
        @inbounds Mat[2 * d_ind, 2 * d_ind - 1] = Vec[v_ind];
        v_ind += 1;
    end


    if fil
        fill_upper_SkewSymm!(S);
    end

    return S;
end

mat_SkewSymm_blk!(S::Ref{Matrix{elty}}, it::STRICT_LOWER_ITERATOR, v::Ref{Vector{elty}}, vos::Int; fil::Bool = true) where elty = 
    _SkewSymm_vec2mat_by_iterator!(S, it, v, vos; fil = fil);



mat_SkewSymm_blk!(S::Ref{Matrix{elty}}, it::STRICT_LOWER_ITERATOR, v::Ref{Vector{elty}}; fil::Bool = true) where elty = 
    _SkewSymm_vec2mat_by_iterator!(S, it, v; fil = fil);


function mat_SkewSymm_blk!(S::AbstractMatrix{elty}, it::STRICT_LOWER_ITERATOR, v::AbstractVector{elty}; fil::Bool = true) where elty

    _SkewSymm_vec2mat_by_iterator!(S, it, v; fil = fil)

    return S;
end



"""
    fill_upper_SkewSymm!(S, [os::Int, ed::Int], [lower_ind_map::STRICT_LOWER_ITERATOR]) -> S::Ref{Matrix{Float64}}
    
Fill the upper triangular part of `S` with the negation of its tranposed lower triangular part so that `S` is skew symmetric. The extra `lower_ind_map` specified a traversal order in the lower triangular part of S, that is used for parallel computing. For 128 threads, multi-threading `fill_upper_SkewSymm` does not worth the effort for matrix with size `n < 180`.

See more about the acceleration gains from `STRICT_LOWER_ITERATOR`.

"""
function fill_upper_SkewSymm!(S::Ref{Matrix{elty}}) where elty
    Mat = S[];

    for c_ind in axes(Mat, 2)
        for r_ind in 1:(c_ind - 1)
            @inbounds Mat[r_ind, c_ind] = - Mat[c_ind, r_ind];
        end
        @inbounds Mat[c_ind, c_ind] = 0.0;
    end
    return S;
end

function fill_upper_SkewSymm!(S::Ref{Matrix{elty}}, os::Int, ed::Int) where elty
    Mat = S[];

    for c_ind in (os + 1):ed
        for r_ind in (os + 1):(c_ind - 1)
            @inbounds Mat[r_ind, c_ind] = - Mat[c_ind, r_ind];
        end
        @inbounds Mat[c_ind, c_ind] = 0.0;
    end
    return S;
end

function fill_upper_SkewSymm!(S::Ref{Matrix{elty}}, lower_ind_map::STRICT_LOWER_ITERATOR) where elty

    if lower_ind_map.mat_dim < 200
        return fill_upper_SkewSymm!(S);
    end

    Mat = S[];

    Vec2LowerMat = lower_ind_map.vec2lower[];
    Vec2UpperMat = lower_ind_map.vec2upper[];

    @tturbo for ind in eachindex(Vec2LowerMat)
        @inbounds Mat[Vec2UpperMat[ind]] = - Mat[Vec2LowerMat[ind]];
    end

    @tturbo for ind in axes(Mat, 1)
        @inbounds Mat[ind, ind] = 0.0;
    end

    return S;
end

"""
    fill_lower_SkewSymm!(S) -> S::Ref{Matrix{Float64}}
    
Fill the lower triangular part of `S` with the negation of its tranposed lower triangular part so that `S` is skew symmetric.
"""
function fill_lower_SkewSymm!(S::Ref{Matrix{elty}}) where elty
    Mat = S[];
    mat_dim::Int = size(Mat, 1);

    for c_ind = 1:mat_dim
        for r_ind = (c_ind + 1):mat_dim
            @inbounds Mat[r_ind, c_ind] = - Mat[c_ind, r_ind];
        end
        @inbounds Mat[c_ind, c_ind] = 0.0;
    end
    return S;
end

function fill_lower_SkewSymm!(S::Ref{Matrix{elty}}, os::Int, ed::Int) where elty
    Mat = S[];

    for c_ind = (os + 1):ed
        for r_ind = (c_ind + 1):ed
            @inbounds Mat[r_ind, c_ind] = - Mat[c_ind, r_ind];
        end
        @inbounds Mat[c_ind, c_ind] = 0.0;
    end
    return S;
end

function fill_lower_SkewSymm!(S::Ref{Matrix{elty}}, lower_ind_map::STRICT_LOWER_ITERATOR) where elty
    if lower_ind_map.mat_dim < 200
        return fill_lower_SkewSymm!(S);
    end

    Mat = S[];

    Vec2LowerMat = lower_ind_map.vec2lower[];
    Vec2UpperMat = lower_ind_map.vec2upper[];

    @tturbo for ind in eachindex(Vec2LowerMat)
        @inbounds Mat[Vec2LowerMat[ind]] = - Mat[Vec2UpperMat[ind]];
    end

    @tturbo for ind in axes(Mat, 1)
        @inbounds Mat[ind, ind] = 0.0;
    end

    return S;
end



"""
    getSkewSymm!(S, [lower_ind_map::STRICT_LOWER_ITERATOR]) -> S::Ref{Matrix{Float64}}
    
Compute the skew symmetric portion of S by the formula (S - S')/ 2, overwrite it to S. The extra `lower_ind_map` specified a traversal order in the lower triangular part of S, that is used for parallel computing. For 128 threads, multi-threading `getSkewSymm!` does not worth the effort for matrix with size `n < 250`.

See more about the acceleration gains from `STRICT_LOWER_ITERATOR`.
"""
function getSkewSymm!(S::Ref{Matrix{elty}}) where elty
    Mat = S[];

    for c_ind in axes(Mat, 2)
        for r_ind in 1:(c_ind - 1)
            @inbounds Mat[r_ind, c_ind] = (Mat[r_ind, c_ind] - Mat[c_ind, r_ind]) / 2.0;
            @inbounds Mat[c_ind, r_ind] = - Mat[r_ind, c_ind];
        end
        @inbounds Mat[c_ind, c_ind] = 0.0;
    end
    return S;
end

function getSkewSymm!(S::Ref{Matrix{elty}}, os::Int, ed::Int) where elty
    Mat = S[];

    for c_ind in (os + 1):ed
        for r_ind in (os + 1):(c_ind - 1)
            @inbounds Mat[r_ind, c_ind] = (Mat[r_ind, c_ind] - Mat[c_ind, r_ind]) / 2.0;
            @inbounds Mat[c_ind, r_ind] = - Mat[r_ind, c_ind];
        end
        @inbounds Mat[c_ind, c_ind] = 0.0;
    end
    return S;
end

function getSkewSymm!(S::Ref{Matrix{elty}}, lower_ind_map::STRICT_LOWER_ITERATOR) where elty
    if lower_ind_map.mat_dim < 200
        return getSkewSymm!(S);
    end

    Mat = S[];

    Vec2LowerMat = lower_ind_map.vec2lower[];
    Vec2UpperMat = lower_ind_map.vec2upper[];


    @tturbo for ind in eachindex(Vec2LowerMat)
        @inbounds Mat[Vec2LowerMat[ind]] = (Mat[Vec2LowerMat[ind]] - Mat[Vec2UpperMat[ind]]) / 2.0;
        @inbounds Mat[Vec2UpperMat[ind]] = - Mat[Vec2LowerMat[ind]];
    end

    @tturbo for ind in axes(Mat, 1)
        @inbounds Mat[ind, ind] = 0.0;
    end
    return S;
end






#######################################Test functions#######################################

using BenchmarkTools

function test_SkewSymm_iterator_threading_speed(n = 200)
    col_iter = STRICT_LOWER_ITERATOR(n, lower_col_traversal);

    MatM1 = rand(n, n);
    MatM2 = copy(MatM1);


    M1 = Ref(MatM1)
    M2 = Ref(MatM2)

    println("Fill skew symmetric matrice, fill_upper_SkewSymm!, with raw loops")
    @btime fill_upper_SkewSymm!($M1)

    println("Fill skew symmetric matrice, fill_upper_SkewSymm!, with iterator plus $(Threads.nthreads()) threads.")
    @btime fill_upper_SkewSymm!($M2, $col_iter)

    println("Same result? \t", MatM1 ≈ MatM2)

    if !(MatM1 ≈ MatM2)
        display(MatM1)
        display(MatM2)
    end


    println("Compute skew symmetric part, getSkewSymm!, with raw loops")
    @btime getSkewSymm!($M1)

    println("Fill skew symmetric matrice, getSkewSymm!, with iterator plus $(Threads.nthreads()) threads.")
    @btime getSkewSymm!($M2, $col_iter)
    println("Same result? \t", MatM1 ≈ MatM2)

    if !(MatM1 ≈ MatM2)
        display(MatM1)
        display(MatM2)
    end

end

function test_SkewSymm_MatVec_threading_speed(n = 10)
    Mat1 = rand(n, n);
    Mat1 .-= Mat1;
    Mat2 = copy(Mat1);

    Mat3 = similar(Mat1);
    Mat4 = similar(Mat1);


    Vec1 = zeros(div(n * (n - 1), 2))
    Vec2 = zeros(div(n * (n - 1), 2))


    M1 = Ref(Mat1)
    M2 = Ref(Mat2)
    M3 = Ref(Mat3)
    M4 = Ref(Mat4)

    V1 = Ref(Vec1)
    V2 = Ref(Vec2)

    col_it = STRICT_LOWER_ITERATOR(n, lower_col_traversal)
    blk_it = STRICT_LOWER_ITERATOR(n, lower_blk_traversal)


    println("Vectorization in lower-column-major order, vec_SkewSymm_col!, using lower triangular part.")
    @btime vec_SkewSymm_col!($V1, $M1, $col_it; lower = true)

    println("Vectorization in lower-column-major order, vec_SkewSymm_col!, using upper triangular part.")
    @btime vec_SkewSymm_col!($V2, $M1, $col_it; lower = false)

    println("Same result? \t", Vec1 ≈ Vec2, "\n")

    println("Matricization in lower-column-major order, mat_SkewSymm_col!.")
    @btime mat_SkewSymm_col!($M3, $col_it, $V1; fil = true)

    println("Same result? \t", Mat1 ≈ Mat3, "\n")

    println("Vectorization in lower-block-major order, vec_SkewSymm_blk!, with raw loops.")
    @btime vec_SkewSymm_blk!($V1, $M1; lower = true)

    println("Vectorization in lower-block-major order, vec_SkewSymm_blk!, with iterator plus $(Threads.nthreads()) threads.")
    @btime vec_SkewSymm_blk!($V2, $M1, $blk_it; lower = true)

    println("Same result? \t", Vec1 ≈ Vec2, "\n")

    println("Matricization in lower-block-major order, mat_SkewSymm_blk!, with raw loops.")
    @btime mat_SkewSymm_blk!($M1, $V1; fil = true)

    println("Matricization in lower-block-major order, mat_SkewSymm_blk!, with iterator plus $(Threads.nthreads()) threads.")
    @btime mat_SkewSymm_blk!($M3, $blk_it, $V1; fil = true)

    println("Same result? \t", Mat1 ≈ Mat3, "\n")

    println("Vectorization in lower-block-major order, vec_SkewSymm_blk!, with raw loops.")
    @btime vec_SkewSymm_blk!($V1, $M1; lower = true)

    println("Vectorization in lower-block-major order, vec_SkewSymm_blk!, with iterator plus $(Threads.nthreads()) threads.")
    @btime vec_SkewSymm_blk!($V2, $M1, $blk_it; lower = false)

    println("Same result? \t", Vec1 ≈ Vec2, "\n")
end