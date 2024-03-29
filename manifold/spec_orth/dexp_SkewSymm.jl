include("../../inc/global_path.jl")
include(joinpath(JULIA_INCLUDE_PATH, "workspace.jl"))

include("so_explog.jl")
include("iterator_SkewSymm.jl")

using LoopVectorization


"""
    dexp_sxdx(x) -> sin(x)/x

This function is well defined everywhere.
"""
@inline dexp_sxdx(x) =
    if (abs(x) < 1e-15)
        return 1.0
    else
        return sin(x) / x
    end
"""
    dexp_cxdx(x) -> (cos(x) - 1)/x

This function is well defined everywhere.
"""
@inline dexp_cxdx(x) =
    if (abs(x) < 1e-15)
        return 0.0
    else
        return (cos(x) - 1.0) / x
    end

"""
    dexp_invx(x) -> (-0.5x ⋅ sin(x)) / (cos(x)-1)

This is only well defined on domains ``(-2π + 4kπ, 2π + 4kπ)``, ∀ integer ``k`` and not defined on ``± 2π + 4kπ``, ∀ integer ``k``.
"""
@inline dexp_invx(x) = abs(x) < 1e-15 ? 1.0 : -0.5 * x * sin(x) / (cos(x) - 1.0)


@inline lower_triangle_cnt(n::Int) = div(n * (n - 1), 2)

"""

# Summary

    mutable struct dexp_SkewSymm_system

This is an object that stores the linear system and its inverse requried for the directional derivative ``dexp_{M}[S]`` and its inverse ``dexp_{M}^{-1}[S]`` at skew-symmetric foot `M` along skew symmetric perturbation `S`. It is fully charaterized by ``compute_dexp_SkewSymm_both_system!`` with the angles of `M` obtained from the real Schur angular decomposisiton ``schurAngular_SkewSymm``. This object can be initialized as an empty object provided dimension only or fully determined provided `M`, its Schur angular decomposition `saf::SAFactor` or just a set of angles. 
    
For matrices with size of `2n` and `2n + 1`, they both have `n` angles. When `n` angles are passed to the initializer, system with size of `2n + 1` will be generated, as the `2n` linear system can be obtained from the `2n + 1` system.

# Fields

    mat_system :: Ref{Matrix{Float64}}
    vec_system :: Ref{Vector{Float64}}
    mat_dim    :: Int64
    blk_dim    :: Int64

# Initializers

    dexp_SkewSymm_system(n) -> sys::dexp_SkewSymm_system

Allocate the object for the linear systems requried for the directional derivative ``dexp_{M}[S]`` or its inverse ``dexp_{M}^{-1}[S]`` with size `n × n`

-----------------------------------------------------------------------------------

    dexp_SkewSymm_system(M::Ref{Matrix{Float64}}) -> sys::dexp_SkewSymm_system

Given a skew symmetric matrix `M`, real schur angular factorization `schurAngular_SkewSymm` is performed to find the associated angles and then the angles are used to form the linear systems requried for the directional derivative ``dexp_{M}`` or its inverse ``dexp_{M}^{-1}``.

-----------------------------------------------------------------------------------

    dexp_SkewSymm_system(saf::SAFactor) -> sys::dexp_SkewSymm_system

Given a real schur angular factorization `saf` of some skew symmetric matrix `M`, the angles are used to form the linear systems requried for the directional derivative ``dexp_{M}`` or its inverse ``dexp_{M}^{-1}``.

-----------------------------------------------------------------------------------

    dexp_SkewSymm_system(V::Ref{Vector{Float64}}) -> sys::dexp_SkewSymm_system

Given a real vector `V` with `n` angles, form the linear systems requried for the directional derivative ``dexp_{M}`` or its inverse ``dexp_{M}^{-1}``, where `M` has dimension of `2n × 2n` or `(2n + 1) × (2n + 1)` and angles `V`.
"""
mutable struct dexp_SkewSymm_system
    # Lower blocks for the system and upper blocks for the inverse system
    mat_system::Ref{Matrix{Float64}}
    mat_trasys::Ref{Matrix{Float64}}
    mat_dim::Int
    blk_dim::Int
    trans::Bool

    dexp_SkewSymm_system(n::Int) = new(Ref(Matrix{Float64}(undef, n, n)), Ref(Matrix{Float64}(undef, n, n)), n, div(n, 2), false)


    function dexp_SkewSymm_system(M::Ref{Matrix{Float64}}; trans::Bool=false)
        sys = dexp_SkewSymm_system(size(M[], 1))
        M_saf = schurAngular_SkewSymm(M; order=false)
        compute_dexp_SkewSymm_both_system!(sys, M_saf.angle; trans=trans)
        return sys
    end

    function dexp_SkewSymm_system(saf::SAFactor; trans::Bool=false)
        sys = dexp_SkewSymm_system(size(saf.vector[], 1))
        compute_dexp_SkewSymm_both_system!(sys, saf.angle; trans=trans)
    end

    function dexp_SkewSymm_system(angle::Ref{Vector{Float64}}; trans::Bool=false)
        sys = dexp_SkewSymm_system(2 * length(angle[]) + 1)
        compute_dexp_SkewSymm_both_system!(sys, angle; trans=trans)
    end
end

function compute_dexp_SkewSymm_both_system!(M_sys::dexp_SkewSymm_system, A::Ref{Vector{Float64}}; trans::Bool=true)
    MatM = M_sys.mat_system[]
    VecA = A[]
    mat_dim = M_sys.mat_dim
    blk_dim = M_sys.blk_dim
    for c_ind = 1:blk_dim
        for r_ind = (c_ind+1):blk_dim
            @inbounds di = VecA[r_ind] - VecA[c_ind]
            @inbounds su = VecA[r_ind] + VecA[c_ind]
            a = dexp_sxdx(di)
            b = dexp_sxdx(su)
            c = dexp_cxdx(di)
            d = dexp_cxdx(su)
            ap = dexp_invx(di)
            bp = dexp_invx(su)

            # Forward system
            # Storing blocks as |a+b a-b|
            #                   |c+d c-d|
            @inbounds MatM[2*r_ind-1, 2*c_ind-1] = a + b
            @inbounds MatM[2*r_ind-1, 2*c_ind] = a - b
            @inbounds MatM[2*r_ind, 2*c_ind-1] = c + d
            @inbounds MatM[2*r_ind, 2*c_ind] = c - d
            # Backward system (inversion)
            # Storing blocks as |a'+b' c'+d'| 
            #                   |a'-b' c'-d'|
            # Note that cp = -0.5 di and dp = -0.5 su
            @inbounds MatM[2*c_ind-1, 2*r_ind-1] = ap + bp
            @inbounds MatM[2*c_ind-1, 2*r_ind] = -0.5 * (di + su)
            @inbounds MatM[2*c_ind, 2*r_ind-1] = ap - bp
            @inbounds MatM[2*c_ind, 2*r_ind] = -0.5 * (di - su)
        end
        @inbounds MatM[2*c_ind-1, 2*c_ind-1] = 0.0
        @inbounds MatM[2*c_ind-1, 2*c_ind] = 1.0
        @inbounds MatM[2*c_ind, 2*c_ind-1] = 1.0
        @inbounds MatM[2*c_ind, 2*c_ind] = 0.0
    end

    if isodd(mat_dim)
        for c_ind = 1:blk_dim
            @inbounds a = dexp_sxdx(VecA[c_ind])
            @inbounds b = dexp_cxdx(VecA[c_ind])
            @inbounds ap = dexp_invx(VecA[c_ind])
            # Note that bp = -0.5 VecA[c_ind]

            @inbounds MatM[mat_dim, 2*c_ind-1] = a
            @inbounds MatM[mat_dim, 2*c_ind] = b

            @inbounds MatM[2*c_ind-1, mat_dim] = ap
            @inbounds MatM[2*c_ind, mat_dim] = -0.5 * VecA[c_ind]
        end
        @inbounds MatM[mat_dim, mat_dim] = 0.0
    end

    if trans
        MatMtrans = M_sys.mat_trasys[]
        for c_ind = axes(MatM, 1)
            for r_ind = axes(MatM, 2)
                MatMtrans[r_ind, c_ind] = MatM[c_ind, r_ind]
            end
        end
    end

    M_sys.trans = trans
end

function compute_dexp_SkewSymm_forward_system!(M_sys::dexp_SkewSymm_system, A::Ref{Vector{Float64}})
    MatM = M_sys.mat_system[]
    VecA = A[]
    mat_dim = M_sys.mat_dim
    blk_dim = M_sys.blk_dim
    for c_ind = 1:blk_dim
        for r_ind = (c_ind+1):blk_dim
            @inbounds di = VecA[r_ind] - VecA[c_ind]
            @inbounds su = VecA[r_ind] + VecA[c_ind]
            a = dexp_sxdx(di)
            b = dexp_sxdx(su)
            c = dexp_cxdx(di)
            d = dexp_cxdx(su)
            # ap = dexp_invx(di);
            # bp = dexp_invx(su);

            # Forward system
            # Storing blocks as |a+b a-b|
            #                   |c+d c-d|
            @inbounds MatM[2*r_ind-1, 2*c_ind-1] = a + b
            @inbounds MatM[2*r_ind-1, 2*c_ind] = a - b
            @inbounds MatM[2*r_ind, 2*c_ind-1] = c + d
            @inbounds MatM[2*r_ind, 2*c_ind] = c - d
            # Backward system (inversion)
            # Storing blocks as |a'+b' c'+d'| 
            #                   |a'-b' c'-d'|
            # Note that cp = -0.5 di and dp = -0.5 su
            # @inbounds MatM[2 * c_ind - 1, 2 * r_ind - 1] = ap + bp;
            # @inbounds MatM[2 * c_ind - 1, 2 * r_ind] = - 0.5 * (di + su);
            # @inbounds MatM[2 * c_ind, 2 * r_ind - 1] = ap - bp;
            # @inbounds MatM[2 * c_ind, 2 * r_ind] = - 0.5 * (di - su);
        end
        # @inbounds MatM[2 * c_ind - 1, 2 * c_ind - 1] = 0.0;
        # @inbounds MatM[2 * c_ind - 1, 2 * c_ind] = 1.0;
        # @inbounds MatM[2 * c_ind, 2 * c_ind - 1] = 1.0;
        # @inbounds MatM[2 * c_ind, 2 * c_ind] = 0.0;
    end

    if isodd(mat_dim)
        for c_ind = 1:blk_dim
            @inbounds a = dexp_sxdx(VecA[c_ind])
            @inbounds b = dexp_cxdx(VecA[c_ind])
            # @inbounds ap = dexp_invx(VecA[c_ind])
            # Note that bp = -0.5 VecA[c_ind]

            @inbounds MatM[mat_dim, 2*c_ind-1] = a
            @inbounds MatM[mat_dim, 2*c_ind] = b

            # @inbounds MatM[2 * c_ind - 1, mat_dim] = ap
            # @inbounds MatM[2 * c_ind, mat_dim] = - 0.5 * VecA[c_ind]
        end
        @inbounds MatM[mat_dim, mat_dim] = 0.0
    end

    M_sys.trans = false
end


"""
    _dexp_SkewSymm_core!(Δ, S, M_sys; inv) -> Δ::Ref{Matrix{Float64}}

When `inv == false`, the default, computes the directional derivative ``dexp_{M}[S] = Δ``, otherwise, computes the inverse ``dexp_{M}^{-1}[S] = Δ``.
The skew symmetric foot `M` should have the block diagonal structure with `2 × 2` blocks 

| 0 \t-θ|\\
| θ \t0\t|

which can be obtained by the upper triangular matrix from the real Schur decomposisiton on any skew-symmetric matrix `S`, see ``schurAngular_SkewSymm`` and the built-in ``schur``. The `θ`'s are referred as the angles of `M`. The parameter ``M_sys::dexp_SkewSymm_system`` of such ``M`` should be computed by the ``compute_dexp_SkewSymm_both_system!``, provied the angles.

The skew symmetric `Δ` and `S` are in matrix form, see also ``_dexp_SkewSymm_vec_core!`` for Δ, S in
block vector form.

"""
_dexp_SkewSymm_core!(Δ::Ref{Matrix{Float64}}, S::Ref{Matrix{Float64}}, M_sys::dexp_SkewSymm_system; inv::Bool=false) = inv ? _dexp_SkewSymm_backward!(Δ, S, M_sys) : _dexp_SkewSymm_forward!(Δ, S, M_sys);

_dexp_SkewSymm_core!(Δ::Ref{Matrix{Float64}}, S::Ref{Matrix{Float64}}, M_sys::dexp_SkewSymm_system, blk_it::STRICT_LOWER_ITERATOR; inv::Bool=false) = inv ? _dexp_SkewSymm_backward!(Δ, S, M_sys, blk_it) : _dexp_SkewSymm_forward!(Δ, S, M_sys, blk_it);

_dexp_SkewSymm_core_compact!(Δ::Ref{Matrix{Float64}}, S::Ref{Matrix{Float64}}, M_sys::dexp_SkewSymm_system, blk_it::STRICT_LOWER_ITERATOR; inv::Bool=false) = inv ? _dexp_SkewSymm_backward_compact!(Δ, S, M_sys, blk_it) : _dexp_SkewSymm_forward!(Δ, S, M_sys, blk_it);


"""
    _dexp_SkewSymm_forward!(Δ, S, M_sys; inv) -> Δ::Ref{Matrix{Float64}}

Subroutine that computes the directional derivative ``dexp_{M}[S] = Δ`` with `2 × 2` block diagonal skew symmetric foot `M`, where the diagonal blocks takes the form of

| 0 \t-θ|\\
| θ \t0\t|

which can be obtained by the upper triangular matrix from the real Schur decomposisiton on any skew-symmetric matrix `S`, see ``schurAngular_SkewSymm`` and the built-in ``schur``. The `θ`'s are referred as the angles of `M`. The parameter ``M_sys::dexp_SkewSymm_system`` of such ``M`` should be computed by the ``compute_dexp_SkewSymm_both_system!``, provied the angles.

The skew symmetric `Δ` and `S` are in matrix form, only lower triangular parts are accessed.
"""
function _dexp_SkewSymm_forward!(Δ::Ref{Matrix{Float64}}, S::Ref{Matrix{Float64}}, M_sys::dexp_SkewSymm_system)
    MatΔ = Δ[]
    MatS = S[]
    Sys = M_sys.mat_system[]

    mat_dim::Int = size(MatS, 1)
    blk_dim::Int = div(mat_dim, 2)

    for d_ind = 1:blk_dim
        @inbounds MatΔ[2*d_ind-1, 2*d_ind-1] = 0.0
        @inbounds MatΔ[2*d_ind, 2*d_ind] = 0.0

        @inbounds MatΔ[2*d_ind, 2*d_ind-1] = MatS[2*d_ind, 2*d_ind-1]
    end


    for c_ind = 1:blk_dim
        @fastmath for r_ind = (c_ind+1):blk_dim

            @inbounds vS1 = MatS[2*r_ind-1, 2*c_ind-1]
            @inbounds vS2 = MatS[2*r_ind, 2*c_ind-1]
            @inbounds vS3 = MatS[2*r_ind-1, 2*c_ind]
            @inbounds vS4 = MatS[2*r_ind, 2*c_ind]

            @inbounds apb = Sys[2*r_ind-1, 2*c_ind-1]
            @inbounds cpd = Sys[2*r_ind, 2*c_ind-1]
            @inbounds amb = Sys[2*r_ind-1, 2*c_ind]
            @inbounds cmd = Sys[2*r_ind, 2*c_ind]

            @inbounds MatΔ[2*r_ind-1, 2*c_ind-1] = 0.5 * (apb * vS1 - cpd * vS2 + cmd * vS3 + amb * vS4)
            @inbounds MatΔ[2*r_ind, 2*c_ind-1] = 0.5 * (cpd * vS1 + apb * vS2 - amb * vS3 + cmd * vS4)
            @inbounds MatΔ[2*r_ind-1, 2*c_ind] = 0.5 * (-cmd * vS1 - amb * vS2 + apb * vS3 - cpd * vS4)
            @inbounds MatΔ[2*r_ind, 2*c_ind] = 0.5 * (amb * vS1 - cmd * vS2 + cpd * vS3 + apb * vS4)
        end
    end



    if isodd(mat_dim)
        @fastmath for c_ind = 1:blk_dim
            @inbounds MatΔ[mat_dim, 2*c_ind-1] = Sys[mat_dim, 2*c_ind-1] * MatS[mat_dim, 2*c_ind-1] - Sys[mat_dim, 2*c_ind] * MatS[mat_dim, 2*c_ind]
            @inbounds MatΔ[mat_dim, 2*c_ind] = Sys[mat_dim, 2*c_ind] * MatS[mat_dim, 2*c_ind-1] + Sys[mat_dim, 2*c_ind-1] * MatS[mat_dim, 2*c_ind]
        end
    end

    return Δ
end

function _dexp_SkewSymm_forward!(Δ::Ref{Matrix{Float64}}, S::Ref{Matrix{Float64}}, M_sys::dexp_SkewSymm_system, blk_it::STRICT_LOWER_ITERATOR)
    MatΔ = Δ[]
    MatS = S[]
    Sys = M_sys.mat_system[]

    mat_dim::Int = size(MatS, 1)
    blk_dim::Int = div(mat_dim, 2)
    blk_num::Int = div(blk_dim * (blk_dim - 1), 2)

    lower_ind1::Int = 0
    lower_ind2::Int = 0
    lower_ind3::Int = 0
    lower_ind4::Int = 0
    ind_shift::Int = 0

    vS1::Float64 = 0
    vS2::Float64 = 0
    vS3::Float64 = 0
    vS4::Float64 = 0

    apb::Float64 = 0
    amb::Float64 = 0
    cpd::Float64 = 0
    cmd::Float64 = 0


    Vec2Lower = blk_it.vec2lower[]
    # ptrInd = pointer(Vec2Lower)

    # @turbo for blk_ind in 1:blk_num
    # @inbounds lower_ind1 = Vec2Lower[4*(blk_ind-1)+1]
    # @inbounds lower_ind2 = Vec2Lower[4*(blk_ind-1)+2]
    # @inbounds lower_ind3 = Vec2Lower[4*(blk_ind-1)+3]
    # @inbounds lower_ind4 = Vec2Lower[4*(blk_ind-1)+4]
    # end

    # @turbo for blk_ind in range(1, 4 * blk_num; step=4)

    #     @inbounds lower_ind1 = Vec2Lower[blk_ind]
    #     @inbounds lower_ind2 = Vec2Lower[blk_ind+1]
    #     @inbounds lower_ind3 = Vec2Lower[blk_ind+2]
    #     @inbounds lower_ind4 = Vec2Lower[blk_ind+3]

    #     @inbounds vS1 = MatS[lower_ind1]
    #     @inbounds vS2 = MatS[lower_ind2]
    #     @inbounds vS3 = MatS[lower_ind3]
    #     @inbounds vS4 = MatS[lower_ind4]

    #     @inbounds apb = Sys[lower_ind1]
    #     @inbounds cpd = Sys[lower_ind2]
    #     @inbounds amb = Sys[lower_ind3]
    #     @inbounds cmd = Sys[lower_ind4]

    #     @inbounds MatΔ[lower_ind1] = 0.5 * (apb * vS1 - cpd * vS2 + cmd * vS3 + amb * vS4)
    #     @inbounds MatΔ[lower_ind2] = 0.5 * (cpd * vS1 + apb * vS2 - amb * vS3 + cmd * vS4)
    #     @inbounds MatΔ[lower_ind3] = 0.5 * (-cmd * vS1 - amb * vS2 + apb * vS3 - cpd * vS4)
    #     @inbounds MatΔ[lower_ind4] = 0.5 * (amb * vS1 - cmd * vS2 + cpd * vS3 + apb * vS4)

    #     # @inbounds MatΔ[lower_ind1] = apb * vS1 - cpd * vS2 + cmd * vS3 + amb * vS4
    #     # @inbounds MatΔ[lower_ind2] = cpd * vS1 + apb * vS2 - amb * vS3 + cmd * vS4
    #     # @inbounds MatΔ[lower_ind3] = -cmd * vS1 - amb * vS2 + apb * vS3 - cpd * vS4
    #     # @inbounds MatΔ[lower_ind4] = amb * vS1 - cmd * vS2 + cpd * vS3 + apb * vS4
    # end

    # println("Enter Forward Action Dexp_M[S] = Δ. Matrix S:")
    # display(S[])

    @turbo for blk_ind in range(1, 4 * blk_num; step=4)

        @inbounds lower_ind1 = Vec2Lower[blk_ind]
        @inbounds lower_ind2 = Vec2Lower[blk_ind+1]
        @inbounds lower_ind3 = Vec2Lower[blk_ind+2]
        @inbounds lower_ind4 = Vec2Lower[blk_ind+3]

        @inbounds vS1 = MatS[lower_ind1]
        @inbounds vS2 = MatS[lower_ind2]
        @inbounds vS3 = MatS[lower_ind3]
        @inbounds vS4 = MatS[lower_ind4]

        @inbounds apb = Sys[lower_ind1]
        @inbounds cpd = Sys[lower_ind2]
        @inbounds amb = Sys[lower_ind3]
        @inbounds cmd = Sys[lower_ind4]

        @inbounds MatΔ[lower_ind1] = 0.5 * (apb * vS1 - cpd * vS2 + cmd * vS3 + amb * vS4)
        @inbounds MatΔ[lower_ind2] = 0.5 * (cpd * vS1 + apb * vS2 - amb * vS3 + cmd * vS4)
        @inbounds MatΔ[lower_ind3] = 0.5 * (-cmd * vS1 - amb * vS2 + apb * vS3 - cpd * vS4)
        @inbounds MatΔ[lower_ind4] = 0.5 * (amb * vS1 - cmd * vS2 + cpd * vS3 + apb * vS4)

        # @inbounds MatΔ[lower_ind1] = apb * vS1 - cpd * vS2 + cmd * vS3 + amb * vS4
        # @inbounds MatΔ[lower_ind2] = cpd * vS1 + apb * vS2 - amb * vS3 + cmd * vS4
        # @inbounds MatΔ[lower_ind3] = -cmd * vS1 - amb * vS2 + apb * vS3 - cpd * vS4
        # @inbounds MatΔ[lower_ind4] = amb * vS1 - cmd * vS2 + cpd * vS3 + apb * vS4
    end

    if isodd(mat_dim)
        # ptrInd = ptrInd + (4 * blk_num) * sizeof(eltype(Vec2Lower))
        ind_shift = 4 * blk_num
        @turbo for blk_ind in range(ind_shift, ind_shift + 2 * blk_dim; step=2)
            # @inbounds lower_ind1 = unsafe_load(ptrInd + (2 * blk_ind - 2) * sizeof(eltype(Vec2Lower)))
            # @inbounds lower_ind2 = unsafe_load(ptrInd + (2 * blk_ind - 1) * sizeof(eltype(Vec2Lower)))
            # @inbounds lower_ind1 = Vec2Lower[ind_shift+2*blk_ind-1]
            # @inbounds lower_ind2 = Vec2Lower[ind_shift+2*blk_ind]

            @inbounds lower_ind1 = Vec2Lower[blk_ind+1]
            @inbounds lower_ind2 = Vec2Lower[blk_ind+2]

            @inbounds MatΔ[lower_ind1] = Sys[lower_ind1] * MatS[lower_ind1] - Sys[lower_ind2] * MatS[lower_ind2]
            @inbounds MatΔ[lower_ind2] = Sys[lower_ind2] * MatS[lower_ind1] + Sys[lower_ind1] * MatS[lower_ind2]
        end
    end

    for d_ind = 1:2:(mat_dim-1)
        @inbounds MatΔ[d_ind, d_ind] = 0.0
        @inbounds MatΔ[d_ind+1, d_ind+1] = 0.0

        @inbounds MatΔ[d_ind+1, d_ind] = MatS[d_ind+1, d_ind]
    end

    # println("Leave Forward Action Dexp_M[S] = Δ. Matrix Δ:")
    # display(MatΔ)

    return Δ
end

function _dexp_SkewSymm_forward_thread!(Δ::Ref{Matrix{Float64}}, S::Ref{Matrix{Float64}}, M_sys::dexp_SkewSymm_system, blk_it::STRICT_LOWER_ITERATOR)
    MatΔ = Δ[]
    MatS = S[]
    Sys = M_sys.mat_system[]

    mat_dim::Int = size(MatS, 1)
    blk_dim::Int = div(mat_dim, 2)
    blk_num::Int = div(blk_dim * (blk_dim - 1), 2)

    lower_ind1::Int = 0
    lower_ind2::Int = 0
    lower_ind3::Int = 0
    lower_ind4::Int = 0
    ind_shift::Int = 0

    vS1::Float64 = 0
    vS2::Float64 = 0
    vS3::Float64 = 0
    vS4::Float64 = 0

    apb::Float64 = 0
    amb::Float64 = 0
    cpd::Float64 = 0
    cmd::Float64 = 0

    Vec2Lower = blk_it.vec2lower[]
    # ptrInd = pointer(Vec2Lower)

    @tturbo for blk_ind in 1:blk_num
        @inbounds lower_ind1 = Vec2Lower[4*(blk_ind-1)+1]
        @inbounds lower_ind2 = Vec2Lower[4*(blk_ind-1)+2]
        @inbounds lower_ind3 = Vec2Lower[4*(blk_ind-1)+3]
        @inbounds lower_ind4 = Vec2Lower[4*(blk_ind-1)+4]

        @inbounds vS1 = MatS[lower_ind1]
        @inbounds vS2 = MatS[lower_ind2]
        @inbounds vS3 = MatS[lower_ind3]
        @inbounds vS4 = MatS[lower_ind4]

        @inbounds apb = Sys[lower_ind1]
        @inbounds cpd = Sys[lower_ind2]
        @inbounds amb = Sys[lower_ind3]
        @inbounds cmd = Sys[lower_ind4]

        @inbounds MatΔ[lower_ind1] = 0.5 * (apb * vS1 - cpd * vS2 + cmd * vS3 + amb * vS4)
        @inbounds MatΔ[lower_ind2] = 0.5 * (cpd * vS1 + apb * vS2 - amb * vS3 + cmd * vS4)
        @inbounds MatΔ[lower_ind3] = 0.5 * (-cmd * vS1 - amb * vS2 + apb * vS3 - cpd * vS4)
        @inbounds MatΔ[lower_ind4] = 0.5 * (amb * vS1 - cmd * vS2 + cpd * vS3 + apb * vS4)
    end

    if isodd(mat_dim)
        # ptrInd = ptrInd + (4 * blk_num) * sizeof(eltype(Vec2Lower))
        ind_shift = 4 * blk_num
        @tturbo for blk_ind in 1:blk_dim
            # @inbounds lower_ind1 = unsafe_load(ptrInd + (2 * blk_ind - 2) * sizeof(eltype(Vec2Lower)))
            # @inbounds lower_ind2 = unsafe_load(ptrInd + (2 * blk_ind - 1) * sizeof(eltype(Vec2Lower)))
            @inbounds lower_ind1 = Vec2Lower[ind_shift+2*blk_ind-1]
            @inbounds lower_ind2 = Vec2Lower[ind_shift+2*blk_ind]

            @inbounds MatΔ[lower_ind1] = Sys[lower_ind1] * MatS[lower_ind1] - Sys[lower_ind2] * MatS[lower_ind2]
            @inbounds MatΔ[lower_ind2] = Sys[lower_ind2] * MatS[lower_ind1] + Sys[lower_ind1] * MatS[lower_ind2]
        end
    end

    for d_ind = 1:2:(mat_dim-1)
        @inbounds MatΔ[d_ind, d_ind] = 0.0
        @inbounds MatΔ[d_ind+1, d_ind+1] = 0.0

        @inbounds MatΔ[d_ind+1, d_ind] = MatS[d_ind+1, d_ind]
    end

    return Δ
end



"""
    _dexp_SkewSymm_backward!(Δ, S, M_sys; inv) -> Δ::Ref{Matrix{Float64}}

Subroutine that computes the directional derivative ``dexp_{M}^{-1}[S] = Δ`` with `2 × 2` block diagonal skew symmetric foot `M`, where the diagonal blocks takes the form of

| 0 \t-θ|\\
| θ \t0\t|

which can be obtained by the upper triangular matrix from the real Schur decomposisiton on any skew-symmetric matrix `S`, see ``schurAngular_SkewSymm`` and the built-in ``schur``. The `θ`'s are referred as the angles of `M`. The parameter ``M_sys::dexp_SkewSymm_system`` of such ``M`` should be computed by the ``compute_dexp_SkewSymm_both_system!``, provied the angles.

The skew symmetric `Δ` and `S` are in matrix form, only lower triangular parts are accessed.
"""
function _dexp_SkewSymm_backward!(Δ::Ref{Matrix{Float64}}, S::Ref{Matrix{Float64}}, M_sys::dexp_SkewSymm_system)
    MatΔ = Δ[]
    MatS = S[]
    Sys = M_sys.mat_system[]

    mat_dim::Int = size(MatS, 1)
    blk_dim::Int = div(mat_dim, 2)

    for d_ind = 1:blk_dim
        @inbounds MatΔ[2*d_ind-1, 2*d_ind-1] = 0.0
        @inbounds MatΔ[2*d_ind, 2*d_ind] = 0.0

        @inbounds MatΔ[2*d_ind, 2*d_ind-1] = MatS[2*d_ind, 2*d_ind-1]
    end

    for c_ind = 1:blk_dim
        for r_ind = (c_ind+1):blk_dim
            # invSys is stored in row_major order.
            @inbounds apb = Sys[2*c_ind-1, 2*r_ind-1]
            @inbounds cpd = Sys[2*c_ind-1, 2*r_ind]
            @inbounds amb = Sys[2*c_ind, 2*r_ind-1]
            @inbounds cmd = Sys[2*c_ind, 2*r_ind]

            @inbounds vS1 = MatS[2*r_ind-1, 2*c_ind-1]
            @inbounds vS2 = MatS[2*r_ind, 2*c_ind-1]
            @inbounds vS3 = MatS[2*r_ind-1, 2*c_ind]
            @inbounds vS4 = MatS[2*r_ind, 2*c_ind]

            # ap+bp cp+dp -cp+dp -bp+ap
            # -cp-dp ap+bp bp-ap -cp+dp
            # cp-dp bp-ap ap+bp cp+dp
            # -bp+ap cp-dp -cp-dp ap+bp


            @inbounds MatΔ[2*r_ind-1, 2*c_ind-1] = 0.5 * (apb * vS1 + cpd * vS2 - cmd * vS3 + amb * vS4)
            @inbounds MatΔ[2*r_ind, 2*c_ind-1] = 0.5 * (-cpd * vS1 + apb * vS2 - amb * vS3 - cmd * vS4)
            @inbounds MatΔ[2*r_ind-1, 2*c_ind] = 0.5 * (cmd * vS1 - amb * vS2 + apb * vS3 + cpd * vS4)
            @inbounds MatΔ[2*r_ind, 2*c_ind] = 0.5 * (amb * vS1 + cmd * vS2 - cpd * vS3 + apb * vS4)
        end
    end

    if isodd(mat_dim)
        for c_ind = 1:blk_dim
            @inbounds MatΔ[mat_dim, 2*c_ind-1] = Sys[2*c_ind-1, mat_dim] * MatS[mat_dim, 2*c_ind-1] + Sys[2*c_ind, mat_dim] * MatS[mat_dim, 2*c_ind]
            @inbounds MatΔ[mat_dim, 2*c_ind] = -Sys[2*c_ind, mat_dim] * MatS[mat_dim, 2*c_ind-1] + Sys[2*c_ind-1, mat_dim] * MatS[mat_dim, 2*c_ind]
        end
    end

    return Δ
end

function _dexp_SkewSymm_backward!(Δ::Ref{Matrix{Float64}}, S::Ref{Matrix{Float64}}, M_sys::dexp_SkewSymm_system, blk_it::STRICT_LOWER_ITERATOR)
    MatΔ = Δ[]
    MatS = S[]

    if !M_sys.trans
        # The inverse system can also be accessed in the lower triangular part of M_sys.mat_trasys, which is the transpose of M_sys.mat_system
        # The consisent lower triangular indices saves half of the dereferencing work and leads to roughly 2 times faster inverse actions.
        # Due to the minor work of getting M_sys.mat_trasys and the fact that M_sys.mat_trasys is preallocated, M_sys.mat_trasys is formed
        # as long as this routine is called. To stay with compact system M_sys.mat_system, use _dexp_SkewSymm_backward_compact! instead.

        MatMtrans = M_sys.mat_trasys[]
        MatM = M_sys.mat_system[]

        copy!(MatMtrans, MatM')
        M_sys.trans = true
    end

    Sys = M_sys.mat_trasys[]

    mat_dim::Int = size(MatS, 1)
    blk_dim::Int = div(mat_dim, 2)
    blk_num::Int = div(blk_dim * (blk_dim - 1), 2)

    lower_ind1::Int = 0
    lower_ind2::Int = 0
    lower_ind3::Int = 0
    lower_ind4::Int = 0
    blk_ind_offset::Int = 0

    vS1::Float64 = 0
    vS2::Float64 = 0
    vS3::Float64 = 0
    vS4::Float64 = 0

    apb::Float64 = 0
    amb::Float64 = 0
    cpd::Float64 = 0
    cmd::Float64 = 0


    Vec2Lower = blk_it.vec2lower[]
    # ptrInd = pointer(Vec2Lower)

    @turbo for blk_ind in 1:blk_num
        @inbounds lower_ind1 = Vec2Lower[4*(blk_ind-1)+1]
        @inbounds lower_ind2 = Vec2Lower[4*(blk_ind-1)+2]
        @inbounds lower_ind3 = Vec2Lower[4*(blk_ind-1)+3]
        @inbounds lower_ind4 = Vec2Lower[4*(blk_ind-1)+4]

        @inbounds vS1 = MatS[lower_ind1]
        @inbounds vS2 = MatS[lower_ind2]
        @inbounds vS3 = MatS[lower_ind3]
        @inbounds vS4 = MatS[lower_ind4]

        # invSys is stored in row_major order.
        @inbounds apb = Sys[lower_ind1]
        @inbounds cpd = Sys[lower_ind2]
        @inbounds amb = Sys[lower_ind3]
        @inbounds cmd = Sys[lower_ind4]

        @inbounds MatΔ[lower_ind1] = 0.5 * (apb * vS1 + cpd * vS2 - cmd * vS3 + amb * vS4)
        @inbounds MatΔ[lower_ind2] = 0.5 * (-cpd * vS1 + apb * vS2 - amb * vS3 - cmd * vS4)
        @inbounds MatΔ[lower_ind3] = 0.5 * (cmd * vS1 - amb * vS2 + apb * vS3 + cpd * vS4)
        @inbounds MatΔ[lower_ind4] = 0.5 * (amb * vS1 + cmd * vS2 - cpd * vS3 + apb * vS4)
    end

    if isodd(mat_dim)
        # ptrInd = ptrInd + (4 * blk_num) * sizeof(eltype(Vec2Lower))
        blk_ind_offset = 4 * blk_num
        @turbo for blk_ind in 1:blk_dim
            @inbounds lower_ind1 = Vec2Lower[blk_ind_offset+2*blk_ind-1]
            @inbounds lower_ind2 = Vec2Lower[blk_ind_offset+2*blk_ind]

            # @inbounds lower_ind1 = unsafe_load(ptrInd + (2 * blk_ind - 2) * sizeof(eltype(Vec2Lower)))
            # @inbounds lower_ind2 = unsafe_load(ptrInd + (2 * blk_ind - 1) * sizeof(eltype(Vec2Lower)))

            @inbounds MatΔ[lower_ind1] = Sys[lower_ind1] * MatS[lower_ind1] + Sys[lower_ind2] * MatS[lower_ind2]
            @inbounds MatΔ[lower_ind2] = -Sys[lower_ind2] * MatS[lower_ind1] + Sys[lower_ind1] * MatS[lower_ind2]
        end
    end

    for d_ind = 1:2:(mat_dim-1)
        @inbounds MatΔ[d_ind, d_ind] = 0.0
        @inbounds MatΔ[d_ind+1, d_ind+1] = 0.0

        @inbounds MatΔ[d_ind+1, d_ind] = MatS[d_ind+1, d_ind]
    end

    return Δ
end

function _dexp_SkewSymm_backward_compact!(Δ::Ref{Matrix{Float64}}, S::Ref{Matrix{Float64}}, M_sys::dexp_SkewSymm_system, blk_it::STRICT_LOWER_ITERATOR)
    MatΔ = Δ[]
    MatS = S[]
    Sys = M_sys.mat_system[]

    mat_dim::Int = size(MatS, 1)
    blk_dim::Int = div(mat_dim, 2)

    blk_num::Int = div(blk_dim * (blk_dim - 1), 2)

    lower_ind1::Int = 0
    lower_ind2::Int = 0
    lower_ind3::Int = 0
    lower_ind4::Int = 0
    upper_ind1::Int = 0
    upper_ind2::Int = 0
    upper_ind3::Int = 0
    upper_ind4::Int = 0
    # blk_ind_offset::Int = 0

    vS1::Float64 = 0
    vS2::Float64 = 0
    vS3::Float64 = 0
    vS4::Float64 = 0

    apb::Float64 = 0
    amb::Float64 = 0
    cpd::Float64 = 0
    cmd::Float64 = 0

    Vec2Lower = blk_it.vec2lower[]
    Vec2Upper = blk_it.vec2upper[]

    ptrLowerInd = pointer(Vec2Lower)
    ptrUpperInd = pointer(Vec2Upper)



    eltysize = sizeof(eltype(Vec2Lower))

    @turbo for blk_ind in 1:blk_num
        shift = 4 * (blk_ind - 1)

        @inbounds lower_ind1 = Vec2Lower[shift+1]
        @inbounds lower_ind2 = Vec2Lower[shift+2]
        @inbounds lower_ind3 = Vec2Lower[shift+3]
        @inbounds lower_ind4 = Vec2Lower[shift+4]

        @inbounds upper_ind1 = Vec2Upper[shift+1]
        @inbounds upper_ind2 = Vec2Upper[shift+2]
        @inbounds upper_ind3 = Vec2Upper[shift+3]
        @inbounds upper_ind4 = Vec2Upper[shift+4]

        @inbounds vS1 = MatS[lower_ind1]
        @inbounds vS2 = MatS[lower_ind2]
        @inbounds vS3 = MatS[lower_ind3]
        @inbounds vS4 = MatS[lower_ind4]

        # invSys is stored in row_major order.
        @inbounds apb = Sys[upper_ind1]
        @inbounds cpd = Sys[upper_ind2]
        @inbounds amb = Sys[upper_ind3]
        @inbounds cmd = Sys[upper_ind4]

        @inbounds MatΔ[lower_ind1] = 0.5 * (apb * vS1 + cpd * vS2 - cmd * vS3 + amb * vS4)
        @inbounds MatΔ[lower_ind2] = 0.5 * (-cpd * vS1 + apb * vS2 - amb * vS3 - cmd * vS4)
        @inbounds MatΔ[lower_ind3] = 0.5 * (cmd * vS1 - amb * vS2 + apb * vS3 + cpd * vS4)
        @inbounds MatΔ[lower_ind4] = 0.5 * (amb * vS1 + cmd * vS2 - cpd * vS3 + apb * vS4)
    end

    if isodd(mat_dim)
        ptrLowerInd = ptrLowerInd + (4 * blk_num) * eltysize
        ptrUpperInd = ptrUpperInd + (4 * blk_num) * eltysize

        @turbo for blk_ind in 1:blk_dim
            shift = (2 * blk_ind - 2) * eltysize
            lower_ind1 = unsafe_load(ptrLowerInd + shift)
            lower_ind2 = unsafe_load(ptrLowerInd + shift + eltysize)

            upper_ind1 = unsafe_load(ptrUpperInd + shift)
            upper_ind2 = unsafe_load(ptrUpperInd + shift + eltysize)

            @inbounds MatΔ[lower_ind1] = Sys[upper_ind1] * MatS[lower_ind1] + Sys[upper_ind2] * MatS[lower_ind2]
            @inbounds MatΔ[lower_ind2] = -Sys[upper_ind2] * MatS[lower_ind1] + Sys[upper_ind1] * MatS[lower_ind2]
        end
    end

    for d_ind = 1:2:(mat_dim-1)
        @inbounds MatΔ[d_ind, d_ind] = 0.0
        @inbounds MatΔ[d_ind+1, d_ind+1] = 0.0

        @inbounds MatΔ[d_ind+1, d_ind] = MatS[d_ind+1, d_ind]
    end

    return Δ
end

function _dexp_SkewSymm_backward_thread!(Δ::Ref{Matrix{Float64}}, S::Ref{Matrix{Float64}}, M_sys::dexp_SkewSymm_system, blk_it::STRICT_LOWER_ITERATOR)
    MatΔ = Δ[]
    MatS = S[]

    if !M_sys.trans
        # The inverse system can also be accessed in the lower triangular part of M_sys.mat_trasys, which is the transpose of M_sys.mat_system
        # The consisent lower triangular indices saves half of the dereferencing work and leads to roughly 2 times faster inverse actions.
        # Due to the minor work of getting M_sys.mat_trasys and the fact that M_sys.mat_trasys is preallocated, M_sys.mat_trasys is formed
        # as long as this routine is called. To stay with compact system M_sys.mat_system, use _dexp_SkewSymm_backward_compact! instead.

        MatMtrans = M_sys.mat_trasys[]
        MatM = M_sys.mat_system[]

        copy!(MatMtrans, MatM')
        M_sys.trans = true
    end

    Sys = M_sys.mat_trasys[]

    mat_dim::Int = size(MatS, 1)
    blk_dim::Int = div(mat_dim, 2)
    blk_num::Int = div(blk_dim * (blk_dim - 1), 2)

    lower_ind1::Int = 0
    lower_ind2::Int = 0
    lower_ind3::Int = 0
    lower_ind4::Int = 0
    # blk_ind_offset::Int = 0

    vS1::Float64 = 0
    vS2::Float64 = 0
    vS3::Float64 = 0
    vS4::Float64 = 0

    apb::Float64 = 0
    amb::Float64 = 0
    cpd::Float64 = 0
    cmd::Float64 = 0


    Vec2Lower = blk_it.vec2lower[]
    ptrInd = pointer(Vec2Lower)

    @tturbo for blk_ind in 1:blk_num
        @inbounds lower_ind1 = Vec2Lower[4*(blk_ind-1)+1]
        @inbounds lower_ind2 = Vec2Lower[4*(blk_ind-1)+2]
        @inbounds lower_ind3 = Vec2Lower[4*(blk_ind-1)+3]
        @inbounds lower_ind4 = Vec2Lower[4*(blk_ind-1)+4]

        @inbounds vS1 = MatS[lower_ind1]
        @inbounds vS2 = MatS[lower_ind2]
        @inbounds vS3 = MatS[lower_ind3]
        @inbounds vS4 = MatS[lower_ind4]

        # invSys is stored in row_major order.
        @inbounds apb = Sys[lower_ind1]
        @inbounds cpd = Sys[lower_ind2]
        @inbounds amb = Sys[lower_ind3]
        @inbounds cmd = Sys[lower_ind4]

        @inbounds MatΔ[lower_ind1] = 0.5 * (apb * vS1 + cpd * vS2 - cmd * vS3 + amb * vS4)
        @inbounds MatΔ[lower_ind2] = 0.5 * (-cpd * vS1 + apb * vS2 - amb * vS3 - cmd * vS4)
        @inbounds MatΔ[lower_ind3] = 0.5 * (cmd * vS1 - amb * vS2 + apb * vS3 + cpd * vS4)
        @inbounds MatΔ[lower_ind4] = 0.5 * (amb * vS1 + cmd * vS2 - cpd * vS3 + apb * vS4)
    end

    if isodd(mat_dim)
        ptrInd = ptrInd + (4 * blk_num) * sizeof(eltype(Vec2Lower))
        @tturbo for blk_ind in 1:blk_dim
            @inbounds lower_ind1 = unsafe_load(ptrInd + (2 * blk_ind - 2) * sizeof(eltype(Vec2Lower)))
            @inbounds lower_ind2 = unsafe_load(ptrInd + (2 * blk_ind - 1) * sizeof(eltype(Vec2Lower)))

            @inbounds MatΔ[lower_ind1] = Sys[lower_ind1] * MatS[lower_ind1] + Sys[lower_ind2] * MatS[lower_ind2]
            @inbounds MatΔ[lower_ind2] = -Sys[lower_ind2] * MatS[lower_ind1] + Sys[lower_ind1] * MatS[lower_ind2]
        end
    end

    for d_ind = 1:2:(mat_dim-1)
        @inbounds MatΔ[d_ind, d_ind] = 0.0
        @inbounds MatΔ[d_ind+1, d_ind+1] = 0.0

        @inbounds MatΔ[d_ind+1, d_ind] = MatS[d_ind+1, d_ind]
    end

    return Δ
end

function _dexp_SkewSymm_backward_compact_thread!(Δ::Ref{Matrix{Float64}}, S::Ref{Matrix{Float64}}, M_sys::dexp_SkewSymm_system, blk_it::STRICT_LOWER_ITERATOR)
    MatΔ = Δ[]
    MatS = S[]
    Sys = M_sys.mat_system[]

    mat_dim::Int = size(MatS, 1)
    blk_dim::Int = div(mat_dim, 2)

    blk_num::Int = div(blk_dim * (blk_dim - 1), 2)


    lower_ind1::Int = 0
    lower_ind2::Int = 0
    lower_ind3::Int = 0
    lower_ind4::Int = 0
    upper_ind1::Int = 0
    upper_ind2::Int = 0
    upper_ind3::Int = 0
    upper_ind4::Int = 0
    # blk_ind_offset::Int = 0

    vS1::Float64 = 0
    vS2::Float64 = 0
    vS3::Float64 = 0
    vS4::Float64 = 0

    apb::Float64 = 0
    amb::Float64 = 0
    cpd::Float64 = 0
    cmd::Float64 = 0

    Vec2Lower = blk_it.vec2lower[]
    Vec2Upper = blk_it.vec2upper[]

    ptrLowerInd = pointer(Vec2Lower)
    ptrUpperInd = pointer(Vec2Upper)



    eltysize = sizeof(eltype(Vec2Lower))

    @tturbo for blk_ind in 1:blk_num
        shift = 4 * (blk_ind - 1)

        @inbounds lower_ind1 = Vec2Lower[shift+1]
        @inbounds lower_ind2 = Vec2Lower[shift+2]
        @inbounds lower_ind3 = Vec2Lower[shift+3]
        @inbounds lower_ind4 = Vec2Lower[shift+4]

        @inbounds upper_ind1 = Vec2Upper[shift+1]
        @inbounds upper_ind2 = Vec2Upper[shift+2]
        @inbounds upper_ind3 = Vec2Upper[shift+3]
        @inbounds upper_ind4 = Vec2Upper[shift+4]

        @inbounds vS1 = MatS[lower_ind1]
        @inbounds vS2 = MatS[lower_ind2]
        @inbounds vS3 = MatS[lower_ind3]
        @inbounds vS4 = MatS[lower_ind4]

        # invSys is stored in row_major order.
        @inbounds apb = Sys[upper_ind1]
        @inbounds cpd = Sys[upper_ind2]
        @inbounds amb = Sys[upper_ind3]
        @inbounds cmd = Sys[upper_ind4]

        @inbounds MatΔ[lower_ind1] = 0.5 * (apb * vS1 + cpd * vS2 - cmd * vS3 + amb * vS4)
        @inbounds MatΔ[lower_ind2] = 0.5 * (-cpd * vS1 + apb * vS2 - amb * vS3 - cmd * vS4)
        @inbounds MatΔ[lower_ind3] = 0.5 * (cmd * vS1 - amb * vS2 + apb * vS3 + cpd * vS4)
        @inbounds MatΔ[lower_ind4] = 0.5 * (amb * vS1 + cmd * vS2 - cpd * vS3 + apb * vS4)
    end

    if isodd(mat_dim)
        ptrLowerInd = ptrLowerInd + (4 * blk_num) * eltysize
        ptrUpperInd = ptrUpperInd + (4 * blk_num) * eltysize

        @tturbo for blk_ind in 1:blk_dim
            shift = (2 * blk_ind - 2) * eltysize
            lower_ind1 = unsafe_load(ptrLowerInd + shift)
            lower_ind2 = unsafe_load(ptrLowerInd + shift + eltysize)

            upper_ind1 = unsafe_load(ptrUpperInd + shift)
            upper_ind2 = unsafe_load(ptrUpperInd + shift + eltysize)

            @inbounds MatΔ[lower_ind1] = Sys[upper_ind1] * MatS[lower_ind1] + Sys[upper_ind2] * MatS[lower_ind2]
            @inbounds MatΔ[lower_ind2] = -Sys[upper_ind2] * MatS[lower_ind1] + Sys[upper_ind1] * MatS[lower_ind2]
        end
    end

    for d_ind = 1:2:(mat_dim-1)
        @inbounds MatΔ[d_ind, d_ind] = 0.0
        @inbounds MatΔ[d_ind+1, d_ind+1] = 0.0

        @inbounds MatΔ[d_ind+1, d_ind] = MatS[d_ind+1, d_ind]
    end

    return Δ
end

# @inline get_wsp_cong(n) = WSP(Matrix{Float64}(undef, n, n), Matrix{Float64}(undef, n, n))

function dexp_SkewSymm!(Δ::Ref{Matrix{Float64}}, S::Ref{Matrix{Float64}}, M_sys::dexp_SkewSymm_system, M_saf::SAFactor, wsp_cong::WSP=get_wsp_cong(size(S[], 1)); inv::Bool=false, cong::Bool=true)
    MatS = S[]
    MatΔ = Δ[]
    MatP = getVector(M_saf)

    MatTmp = wsp_cong[1]
    Tmp = wsp_cong(1)
    # MatTemp2 = wsp_cong[2];



    if cong
        # if inv
        #     println("\nEnter Backward Action Dexp_M^{-1}[Δ] = S. Input Matrix Δ:\n")
        #     display(S[])
        # else
        #     println("\nEnter Forward Action Dexp_M[S] = Δ. Input Matrix S:\n")
        #     display(S[])
        # end
        cong_dense!(Δ, M_saf.vector, S, wsp_cong; trans=true)
        fill_upper_SkewSymm!(Δ)

        # if inv
        #     println("\nConverted Input Matrix P^TΔP:\n")
        #     display(Δ[])
        # else
        #     println("\nConvert Input Matrix P^TSP:\n")
        #     display(Δ[])
        # end

        # @inbounds mul!(MatΔ, MatP', MatS)
        # @inbounds mul!(MatTmp, MatΔ, MatP)
        _dexp_SkewSymm_core!(Tmp, Δ, M_sys; inv=inv)
        fill_upper_SkewSymm!(Tmp)

        # if inv
        #     println("\nConverted OutPut Matrix P^TSP:\n")
        #     display(Tmp[])
        # else
        #     println("\nConvert OutPut Matrix P^TΔP:\n")
        #     display(Tmp[])
        # end

        unsafe_copyto!(pointer(MatΔ), pointer(MatTmp), length(MatΔ))
        cong_dense!(Δ, M_saf.vector, Δ; trans=false)
        fill_upper_SkewSymm!(Δ)

        # if inv
        #     println("\nLeave Backward Action Dexp_M^{-1}[Δ] = S. Output Matrix S:\n")
        #     display(Δ[])
        # else
        #     println("\nLeave Forward Action Dexp_M[S] = Δ. Output Matrix Δ:\n")
        #     display(Δ[])
        # end
    else
        _dexp_SkewSymm_core!(Δ, S, M_sys; inv=inv)
        fill_upper_SkewSymm!(Δ)
    end
end



function dexp_SkewSymm!(Δ::Ref{Matrix{Float64}}, S::Ref{Matrix{Float64}}, M_sys::dexp_SkewSymm_system, M_saf::SAFactor, blk_it::STRICT_LOWER_ITERATOR, wsp_cong::WSP; inv::Bool=false, cong::Bool=true, compact::Bool=false)
    MatS = S[]
    MatΔ = Δ[]
    MatP = getVector(M_saf)

    MatTmp = wsp_cong[1]
    Tmp = wsp_cong(1)
    # MatTemp2 = wsp_cong[2];


    if cong
        # if inv
        #     println("\nEnter Backward Action Dexp_M^{-1}[Δ] = S. Input Matrix Δ:\n")
        #     display(S[])
        # else
        #     println("\nEnter Forward Action Dexp_M[S] = Δ. Input Matrix S:\n")
        #     display(S[])
        # end

        cong_dense!(Δ, M_saf.vector, S, wsp_cong; trans=true)
        # fill_upper_SkewSymm!(Δ, blk_it)
        fill_upper_SkewSymm!(Δ)
        # @inbounds mul!(MatΔ, MatP', MatS)
        # @inbounds mul!(MatTmp, MatΔ, MatP)

        # if inv
        #     println("\nConverted Input Matrix P^TΔP:\n")
        #     display(Δ[])
        # else
        #     println("\nConvert Input Matrix P^TSP:\n")
        #     display(Δ[])
        # end

        if compact
            _dexp_SkewSymm_core_compact!(Tmp, Δ, M_sys, blk_it; inv=inv)
        else
            _dexp_SkewSymm_core!(Tmp, Δ, M_sys, blk_it; inv=inv)
        end
        # fill_upper_SkewSymm!(Tmp, blk_it)
        fill_upper_SkewSymm!(Tmp)


        # if inv
        #     println("\nConverted OutPut Matrix P^TSP:\n")
        #     display(Tmp[])
        # else
        #     println("\nConvert OutPut Matrix P^TΔP:\n")
        #     display(Tmp[])
        # end

        unsafe_copyto!(pointer(MatΔ), pointer(MatTmp), length(MatΔ))
        cong_dense!(Δ, M_saf.vector, Δ, wsp_cong; trans=false)
        # fill_upper_SkewSymm!(Δ, blk_it)
        fill_upper_SkewSymm!(Δ)


        # if inv
        #     println("\nLeave Backward Action Dexp_M^{-1}[Δ] = S. Output Matrix S:\n")
        #     display(Δ[])
        # else
        #     println("\nLeave Forward Action Dexp_M[S] = Δ. Output Matrix Δ:\n")
        #     display(Δ[])
        # end

        # @inbounds mul!(MatTemp2, MatP, MatΔ)
        # @inbounds mul!(MatΔ, MatTemp2, MatP')
    else
        if compact
            _dexp_SkewSymm_core_compact!(Δ, S, M_sys, blk_it; inv=inv)
        else
            _dexp_SkewSymm_core!(Δ, S, M_sys, blk_it; inv=inv)
        end
        # fill_upper_SkewSymm!(Δ, blk_it)
        fill_upper_SkewSymm!(Δ)


        # unsafe_copyto!(pointer(MatΔ), pointer(MatTmp), length(MatΔ))

    end
end


dexp_SkewSymm!(S::Ref{Matrix{Float64}}, M_sys::dexp_SkewSymm_system, M_saf::SAFactor, wsp_cong::WSP=get_wsp_cong(size(S[], 1)); inv::Bool=false, cong::Bool=true) =
    dexp_SkewSymm!(S, S, M_sys, M_saf, wsp_cong; inv=inv, cong=cong)

dexp_SkewSymm!(S::Ref{Matrix{Float64}}, M_sys::dexp_SkewSymm_system, M_saf::SAFactor, blk_it::STRICT_LOWER_ITERATOR, wsp_cong::WSP=get_wsp_cong(size(S[], 1)); inv::Bool=false, cong::Bool=true, compact::Bool=false) =
    dexp_SkewSymm!(S, S, M_sys, M_saf, wsp_cong; inv=inv, cong=cong, compact=compact)


function dexp_SkewSymm_thread!(Δ::Ref{Matrix{Float64}}, S::Ref{Matrix{Float64}}, M_sys::dexp_SkewSymm_system, M_saf::SAFactor, blk_it::STRICT_LOWER_ITERATOR, wsp_cong::WSP; inv::Bool=false, cong::Bool=true, compact::Bool=false)
    MatS = S[]
    MatΔ = Δ[]
    MatP = getVector(M_saf)

    MatTmp = wsp_cong[1]
    Tmp = wsp_cong(1)
    # MatTemp2 = wsp_cong[2];


    if cong
        cong_dense!(Δ, M_saf.vector, S, wsp_cong; trans=true)
        fill_upper_SkewSymm!(Δ, blk_it)
        # @inbounds mul!(MatΔ, MatP', MatS)
        # @inbounds mul!(MatTmp, MatΔ, MatP)
        if compact
            _dexp_SkewSymm_core_compact_thread!(Tmp, Δ, M_sys, blk_it; inv=inv)
        else
            _dexp_SkewSymm_core_thread!(Tmp, Δ, M_sys, blk_it; inv=inv)
        end
        fill_upper_SkewSymm!(Tmp, blk_it)

        unsafe_copyto!(pointer(MatΔ), pointer(MatTmp), length(MatΔ))
        cong_dense!(Δ, M_saf.vector, Δ; trans=false)
        fill_upper_SkewSymm!(Δ, blk_it)

        # @inbounds mul!(MatTemp2, MatP, MatΔ)
        # @inbounds mul!(MatΔ, MatTemp2, MatP')
    else
        if compact
            _dexp_SkewSymm_core_compact_thread!(Tmp, Δ, M_sys, blk_it; inv=inv)
        else
            _dexp_SkewSymm_core_thread!(Tmp, Δ, M_sys, blk_it; inv=inv)
        end
        fill_upper_SkewSymm!(Δ, blk_it)
    end
end



#######################################Test functions#######################################

using Plots, Printf

include("real_dexp_st.jl")
include(joinpath(JULIA_ROUTINE_PATH, "dexp_SemiSimple.jl"))
include(joinpath(JULIA_ROUTINE_PATH, "dexp_ScaleSquare.jl"))





function test_dexp_SkewSymm_Normal(n; seed=9527)
    eng = MersenneTwister(seed)

    S = rand(eng, n, n)
    S .-= S'

    D_real = rand(eng, n, n)
    D_real .-= D_real'

    D_complex = zeros(ComplexF64, n, n)
    D_complex .= D_real

    blk_it = STRICT_LOWER_ITERATOR(n, lower_blk_traversal)
    S_saf = SAFactor(n)
    schurAngular_SkewSymm!(S_saf, Ref(S), get_wsp_saf(n))
    sys_SkewSymm = dexp_SkewSymm_system(n)
    compute_dexp_SkewSymm_both_system!(sys_SkewSymm, S_saf.angle)
    wsp_SkewSymm = get_wsp_cong(n)


    S_eig = eigen(S)
    sys_SemiSimple = dexp_SemiSimple_system(n)
    compute_dexp_Normal_system!(sys_SemiSimple, S_eig)
    wsp_SemiSimple = WSP(Matrix{ComplexF64}(undef, n, n))

    Δ_real = zeros(n, n)
    Δ_complex = zeros(ComplexF64, n, n)
    Δ_complex_skewsymm = zeros(n, n)

    Q = exp(S)


    sample_num = 100
    record = zeros(sample_num, 4)

    M = sys_SemiSimple.mat_inv_vect[] * D_complex * sys_SemiSimple.mat_eig_vect[]
    # display(M)
    # Phi = [S_eig.values[r_ind] ≈ S_eig.values[c_ind] ? exp(S_eig.values[r_ind]) : (exp(S_eig.values[r_ind]) - exp(S_eig.values[c_ind])) / (S_eig.values[r_ind] - S_eig.values[c_ind]) for r_ind = 1:n, c_ind = 1:n]

    # display(M .* Phi)
    # display(sys_SemiSimple.mat_eig_vect[] * (M .* Phi) * sys_SemiSimple.mat_inv_vect[])

    # display(sys_SemiSimple.mat_psi_comp[])


    for s_ind in 1:sample_num


        stat = @timed dexp_SkewSymm!(Ref(Δ_real), Ref(D_real), sys_SkewSymm, S_saf, blk_it, wsp_SkewSymm; cong=false)
        record[s_ind, 1] = (stat.time - stat.gctime) * 1000

        stat = @timed dexp_SkewSymm!(Ref(Δ_real), Ref(D_real), sys_SkewSymm, S_saf, blk_it, wsp_SkewSymm; cong=true)
        record[s_ind, 2] = (stat.time - stat.gctime) * 1000

        stat = @timed dexp_SemiSimple!(Ref(Δ_complex), Ref(D_complex), sys_SemiSimple, wsp_SemiSimple; simi=false)
        record[s_ind, 3] = (stat.time - stat.gctime) * 1000

        stat = @timed begin
            dexp_SemiSimple!(Ref(Δ_complex), Ref(D_complex), sys_SemiSimple, wsp_SemiSimple; simi=true)
            # mul!(Δ_complex_skewsymm, Q', real.(Δ_complex))
        end
        record[s_ind, 4] = (stat.time - stat.gctime) * 1000
    end

    println("Same result?\t", exp(S) * Δ_real ≈ real.(Δ_complex))

    # display(Δ_real)
    # display(Δ_complex)


    @printf "Methods\t\t\t|\t Min time \t|\t Avg Time \t|\t Max Time \t|\n"
    methods = ["dexp_SkewSymm, Core", "dexp_SkewSymm, Full", "dexp_SemiSimple, Core", "dexp_SemiSimple, Full"]

    for ind = 1:4
        @printf "%s\t|\t%.8f\t|\t%.8f\t|\t%.8f\t|\n" methods[ind] minimum(record[:, ind]) mean(record[:, ind]) maximum(record[:, ind])
    end

end

function test_dexp_SkewSymm(n::Int; seed=9527, loop=1000, cong=true, print=false)
    eng = MersenneTwister(seed)

    MatM = rand(eng, n, n)
    MatM .-= MatM'


    MatS_r = rand(eng, n, n)
    MatS_r .-= MatS_r'
    MatS_c = zeros(ComplexF64, n, n)
    MatS_c .= MatS_r
    MatS_s = copy(MatS_r)

    MatΔ_r = zeros(n, n)
    MatΔ_c = zeros(ComplexF64, n, n)
    MatΔ_s = zeros(n, n)


    blk_it = STRICT_LOWER_ITERATOR(n, lower_blk_traversal)
    M_saf = SAFactor(n)
    schurAngular_SkewSymm!(M_saf, Ref(MatM), get_wsp_saf(n))
    M_sys = dexp_SkewSymm_system(n)
    compute_dexp_SkewSymm_both_system!(M_sys, M_saf.angle)
    wsp_cong_n = get_wsp_cong(n)

    M_eig = eigen(MatM)
    M_sys_SemiSimple = dexp_SemiSimple_system(n)
    compute_dexp_Normal_system!(M_sys_SemiSimple, M_eig)
    wsp_SemiSimple_n = WSP(Matrix{ComplexF64}(undef, n, n))

    M_sys_ScaleSquare = dexp_ScaleSquare_system(n)
    compute_dexp_ScaleSquare_system!(M_sys_ScaleSquare, Ref(MatM))
    wsp_ScaleSquare_n = get_wsp_dexp_ScaleSquare(n)

    # MatQ = exp(MatM)

    record = zeros(4, loop)
    # MatM_c = M_sys_SemiSimple.mat_inv_vect[] * MatS_c * M_sys_SemiSimple.mat_eig_vect[]

    S_r = Ref(MatS_r)
    S_c = Ref(MatS_c)
    S_s = Ref(MatS_s)


    Δ_r = Ref(MatΔ_r)
    Δ_c = Ref(MatΔ_c)
    Δ_s = Ref(MatΔ_s)


    MateM = zeros(n, n)
    M = Ref(MatM)
    eM = Ref(MateM)


    for s_ind in 1:loop

        # stat = @timed dexp_SemiSimple!(Δ_c, S_c, M_sys_SemiSimple, wsp_SemiSimple_n; simi=cong)

        stat = @timed dexp_ScaleSquare!(eM, Δ_s, M, M_sys_ScaleSquare, S_s, wsp_ScaleSquare_n)
        record[1, s_ind] = (stat.time - stat.gctime) * 1e3

        stat = @timed dexp_SemiSimple!(Δ_c, S_c, M_sys_SemiSimple, wsp_SemiSimple_n; simi=cong)
        record[2, s_ind] = (stat.time - stat.gctime) * 1e3

        stat = @timed dexp_SkewSymm!(Δ_r, S_r, M_sys, M_saf, wsp_cong_n; cong=cong)
        record[3, s_ind] = (stat.time - stat.gctime) * 1e3

        stat = @timed dexp_SkewSymm!(Δ_r, S_r, M_sys, M_saf, blk_it, wsp_cong_n; cong=cong)
        record[4, s_ind] = (stat.time - stat.gctime) * 1e3
    end

    # println("Same result?\t", exp(S) * Δ_real ≈ real.(Δ_complex))

    method_str = (b) -> b ? "(Full)" : "(Core)"

    if print
        @printf "+-----------------------------------------------------------------------------------------------+\n"
        @printf "|Dexp_S[Δ]\t\t|Min. Time\t|Avg. Time\t|Max. Time\t| (Over %i attempt)\t|\n" loop
        @printf "+-----------------------------------------------------------------------------------------------+\n"
        @printf "|ScaleSquare %s\t|%.8f\t|%.8f\t|%.8f\t| (micro second)\t|\n" method_str(cong) minimum(record[1, :]) mean(record[1, :]) maximum(record[1, :])
        @printf "|SemiSimple  %s\t|%.8f\t|%.8f\t|%.8f\t| (micro second)\t|\n" method_str(cong) minimum(record[2, :]) mean(record[2, :]) maximum(record[2, :])
        @printf "|Skew Raw    %s\t|%.8f\t|%.8f\t|%.8f\t| (micro second)\t|\n" method_str(cong) minimum(record[3, :]) mean(record[3, :]) maximum(record[3, :])
        @printf "|Skew Iter.  %s\t|%.8f\t|%.8f\t|%.8f\t| (micro second)\t|\n" method_str(cong) minimum(record[4, :]) mean(record[4, :]) maximum(record[4, :])
        @printf "+-----------------------------------------------------------------------------------------------+\n\n"
    end

    return record

    # @printf "Methods\t\t\t|\t Min time \t|\t Avg Time \t|\t Max Time \t|\n"
    # methods = ["dexp_SkewSymm, Core", "dexp_SkewSymm, Full", "dexp_SemiSimple, Core", "dexp_SemiSimple, Full"]

    # for ind = 1:4
    #     @printf "%s\t|\t%.8f\t|\t%.8f\t|\t%.8f\t|\n" methods[ind] minimum(record[:, ind]) mean(record[:, ind]) maximum(record[:, ind])
    # end

end

function test_dexp_SkewSymm(dims::Vector{Int}; seed=9527, loop=1000, cong=true, print=false, savefigure=false)
    eng = MersenneTwister(seed)

    record = zeros(length(dims), 5, loop)

    method_str = (b) -> b ? "(Full)" : "(Core)"



    for d_ind in eachindex(dims)
        n = dims[d_ind]

        MatM = rand(eng, n, n)
        MatM .-= MatM'

        MatS_r = rand(eng, n, n)
        MatS_r .-= MatS_r'
        MatS_c = zeros(ComplexF64, n, n)
        MatS_c .= MatS_r
        MatS_s = copy(MatS_r)

        MatΔ_r = zeros(n, n)
        MatΔ_c = zeros(ComplexF64, n, n)
        MatΔ_s = copy(MatΔ_r)


        blk_it = STRICT_LOWER_ITERATOR(n, lower_blk_traversal)
        M_saf = SAFactor(n)
        schurAngular_SkewSymm!(M_saf, Ref(MatM), get_wsp_saf(n))
        M_sys = dexp_SkewSymm_system(n)
        compute_dexp_SkewSymm_both_system!(M_sys, M_saf.angle)
        wsp_cong_n = get_wsp_cong(n)

        M_eig = eigen(MatM)
        M_sys_SemiSimple = dexp_SemiSimple_system(n)
        compute_dexp_Normal_system!(M_sys_SemiSimple, M_eig)
        wsp_SemiSimple_n = WSP(Matrix{ComplexF64}(undef, n, n))

        M_sys_ScaleSquare = dexp_ScaleSquare_system(n)
        compute_dexp_ScaleSquare_system!(M_sys_ScaleSquare, Ref(MatM))
        wsp_ScaleSquare_n = get_wsp_dexp_ScaleSquare(n)

        # MatQ = exp(MatM)
        # MatM_c = M_sys_SemiSimple.mat_inv_vect[] * MatS_c * M_sys_SemiSimple.mat_eig_vect[]

        S_r = Ref(MatS_r)
        S_c = Ref(MatS_c)
        S_s = Ref(MatS_s)

        Δ_r = Ref(MatΔ_r)
        Δ_c = Ref(MatΔ_c)
        Δ_s = Ref(MatΔ_s)

        MateM = zeros(n, n)
        eM = Ref(MateM)
        M = Ref(MatM)


        for s_ind in 1:loop

            stat = @timed dexp_ScaleSquare!(eM, Δ_s, M, M_sys_ScaleSquare, S_s, wsp_ScaleSquare_n)
            record[d_ind, 1, s_ind] = (stat.time - stat.gctime) * 1e3

            stat = @timed dexp_SemiSimple!(Δ_c, S_c, M_sys_SemiSimple, wsp_SemiSimple_n; simi=cong)
            record[d_ind, 2, s_ind] = (stat.time - stat.gctime) * 1e3

            # stat = @timed dexp_SkewSymm!(Δ_r, S_r, M_sys, M_saf, wsp_cong_n; cong=cong)
            # record[d_ind, 2, s_ind] = (stat.time - stat.gctime) * 1e3

            stat = @timed dexp_SkewSymm!(Δ_r, S_r, M_sys, M_saf, blk_it, wsp_cong_n; cong=cong)
            record[d_ind, 3, s_ind] = (stat.time - stat.gctime) * 1e3

            stat = @timed dexp_SemiSimple!(Δ_c, S_c, M_sys_SemiSimple, wsp_SemiSimple_n; simi=cong, inv=true)
            record[d_ind, 4, s_ind] = (stat.time - stat.gctime) * 1e3

            stat = @timed dexp_SkewSymm!(Δ_r, S_r, M_sys, M_saf, blk_it, wsp_cong_n; cong=cong, inv=true)
            record[d_ind, 5, s_ind] = (stat.time - stat.gctime) * 1e3
        end

        if print
            @printf "+-----------------------------------------------------------------------------------------------+\n"
            @printf "|Dexp_S[Δ], n = %i \t\t|Min. Time\t|Avg. Time\t|Max. Time\t| (Over %i attempt)\t|\n" n loop
            @printf "+-----------------------------------------------------------------------------------------------+\n"
            @printf "|ScaleSquare %s\t|%.8f\t|%.8f\t|%.8f\t| (micro second)\t|\n" method_str(cong) minimum(record[d_ind, 1, :]) mean(record[d_ind, 1, :]) maximum(record[d_ind, 1, :])
            @printf "|SemiSimple  %s\t|%.8f\t|%.8f\t|%.8f\t| (micro second)\t|\n" method_str(cong) minimum(record[d_ind, 2, :]) mean(record[d_ind, 2, :]) maximum(record[d_ind, 2, :])
            @printf "|Skew Iter.  %s\t|%.8f\t|%.8f\t|%.8f\t| (micro second)\t|\n" method_str(cong) minimum(record[d_ind, 3, :]) mean(record[d_ind, 3, :]) maximum(record[d_ind, 3, :])
            @printf "+-----------------------------------------------------------------------------------------------+\n"
            @printf "|Dexp_S^{-1}[Δ], n = %i \t|Min. Time\t|Avg. Time\t|Max. Time\t| (Over %i attempt)\t|\n" n loop
            @printf "+-----------------------------------------------------------------------------------------------+\n"
            @printf "|SemiSimple  %s\t|%.8f\t|%.8f\t|%.8f\t| (micro second)\t|\n" method_str(cong) minimum(record[d_ind, 4, :]) mean(record[d_ind, 4, :]) maximum(record[d_ind, 2, :])
            @printf "|Skew Iter.  %s\t|%.8f\t|%.8f\t|%.8f\t| (micro second)\t|\n" method_str(cong) minimum(record[d_ind, 5, :]) mean(record[d_ind, 5, :]) maximum(record[d_ind, 3, :])
            @printf "+-----------------------------------------------------------------------------------------------+\n\n"
        end
    end

    # println("Same result?\t", exp(S) * Δ_real ≈ real.(Δ_complex))

    record_maximum = maximum(record, dims=3)
    record_minimum = minimum(record, dims=3)
    record_mean = mean(record, dims=3)

    # figure()
    # p1 = plot(dims, record_mean[:, 1, :], label="ScaleSquare, Avg. time", xlabel="n, dimension", ylabel="Time (ms)")
    # plot!(dims, record_mean[:, 2, :], label="SemiSimple., Avg. time")
    # plot!(dims, record_mean[:, 3, :], label="SkewSymm., Avg. time")


    # p2 = plot(dims, record_mean[:, 1, :], label="ScaleSquare, Avg. time", yscale=:log2, xlabel="n, dimension", ylabel="Time (ms)")
    # plot!(dims, record_mean[:, 2, :], label="SemiSimple., Avg. time", yscale=:log2, xscale=:log2)
    # plot!(dims, record_mean[:, 3, :], label="SkewSymm., Avg. time", yscale=:log2, xscale=:log2)


    p1 = plot(dims, record_minimum[:, 2, :], label="Dexp, SemiSimple", xlabel="n, dimension", ylabel="Time (ms)")
    plot!(dims, record_minimum[:, 3, :], label="Dexp, SkewSymm.")
    plot!(dims, record_minimum[:, 1, :], label="Dexp, General")


    p2 = plot(dims, record_minimum[:, 2, :], label="Dexp, Semisimple", yscale=:log2, xscale=:log2, xlabel="n, dimension", ylabel="Time (ms)")

    plot!(dims, record_minimum[:, 3, :], label="Dexp, SkewSymm.", yscale=:log2, xscale=:log2)
    plot!(dims, record_minimum[:, 1, :], label="Dexp, General", yscale=:log2, xscale=:log2)

    p3 = plot(dims, record_minimum[:, 4, :], label="Dexp^{-1}, SemiSimple", xlabel="n, dimension", ylabel="Time (ms)")
    plot!(dims, record_minimum[:, 5, :], label="Dexp^{-1}, SkewSymm.")


    p4 = plot(dims, record_minimum[:, 1, :], label="Dexp^{-1}, SemiSimple", yscale=:log2, xscale=:log2, xlabel="n, dimension", ylabel="Time (ms)")
    plot!(dims, record_minimum[:, 2, :], label="Dexp^{-1}, SkewSymm.", yscale=:log2, xscale=:log2)

    plt = plot(p1, p2, p3, p4, layout=[2, 2], figsize=(1000, 1000))

    display(plt)

    # return record

    # @printf "Methods\t\t\t|\t Min time \t|\t Avg Time \t|\t Max Time \t|\n"
    # methods = ["dexp_SkewSymm, Core", "dexp_SkewSymm, Full", "dexp_SemiSimple, Core", "dexp_SemiSimple, Full"]

    # for ind = 1:4
    #     @printf "%s\t|\t%.8f\t|\t%.8f\t|\t%.8f\t|\n" methods[ind] minimum(record[:, ind]) mean(record[:, ind]) maximum(record[:, ind])
    # end

    println("Random seed:\t $(seed)")

    if savefigure
        savefig(plt, "figures/dexp_speed_seed_$(seed).pdf")
    end

end

function test_dexp_SkewSymm_Error(n::Int; seed=9527, loop=1000, cong=true, print=false)
    eng = MersenneTwister(seed)

    record = zeros(4, loop)

    method_str = (b) -> b ? "(Full)" : "(Core)"

    MatM = zeros(n, n)
    MatS = zeros(n, n)


    MatS_r = zeros(n, n)
    MatS_c = zeros(ComplexF64, n, n)
    MatΔ_r = zeros(n, n)
    MatΔ_c = zeros(ComplexF64, n, n)

    M = Ref(MatM)
    S = Ref(MatS)
    S_r = Ref(MatS_r)
    S_c = Ref(MatS_c)
    Δ_r = Ref(MatΔ_r)
    Δ_c = Ref(MatΔ_c)

    blk_it = STRICT_LOWER_ITERATOR(n, lower_blk_traversal)
    M_saf = SAFactor(n)
    M_sys = dexp_SkewSymm_system(n)
    M_sys_SemiSimple = dexp_SemiSimple_system(n)

    wsp_cong_n = get_wsp_cong(n)
    wsp_SemiSimple_n = WSP(Matrix{ComplexF64}(undef, n, n))


    for ind = 1:loop

        MatM .= rand(eng, n, n)
        MatM .-= MatM'

        MatS .= rand(eng, n, n)
        MatS .-= MatS'

        MatS_r .= MatS
        MatS_c .= MatS

        MatΔ_r .= 0.0
        MatΔ_c .= 0.0

        schurAngular_SkewSymm!(M_saf, M, get_wsp_saf(n))
        compute_dexp_SkewSymm_both_system!(M_sys, M_saf.angle)

        M_eig = eigen(MatM)
        compute_dexp_Normal_system!(M_sys_SemiSimple, M_eig)


        dexp_SemiSimple!(Δ_c, S_c, M_sys_SemiSimple, wsp_SemiSimple_n; simi=cong)
        dexp_SemiSimple!(S_c, Δ_c, M_sys_SemiSimple, wsp_SemiSimple_n; inv=true, simi=cong)

        record[1, ind] = norm(real.(MatS_c) .- MatS)
        record[2, ind] = record[1, ind] / norm(MatS)


        # stat = @timed dexp_SkewSymm!(Δ_r, S_r, M_sys, M_saf, wsp_cong_n; cong=cong)
        # record[d_ind, 2, s_ind] = (stat.time - stat.gctime) * 1e3

        dexp_SkewSymm!(Δ_r, S_r, M_sys, M_saf, blk_it, wsp_cong_n; cong=cong)
        dexp_SkewSymm!(S_r, Δ_r, M_sys, M_saf, blk_it, wsp_cong_n; inv=true, cong=cong)

        # display(real.(MatS_c) .- MatS_r)


        record[3, ind] = norm(MatS_r .- MatS)
        record[4, ind] = record[3, ind] / norm(MatS)
    end

    if print
        @printf "+-----------------------------------------------------------------------------------------------+\n"
        @printf "|S = Dexp_M^{-1}[Dexp_M[S]]\t|Min. AbsErr\t|Min. RelErr\t|Avg. AbsErr\t|Avg. RelErr\t|Max. AbsErr\t|Max. RelErr\t| (Over %i attempt)\t|\n" loop
        @printf "+-----------------------------------------------------------------------------------------------+\n"
        @printf "|SemiSimple \t\t\t|%.12f\t|%.12f\t|%.12f\t|%.12f\t|%.12f\t|%.12f\t|\n" minimum(record[1, :]) minimum(record[2, :]) mean(record[1, :]) mean(record[2, :]) maximum(record[1, :]) maximum(record[2, :])
        @printf "|SkewSymm \t\t\t|%.12f\t|%.12f\t|%.12f\t|%.12f\t|%.12f\t|%.12f\t|\n" minimum(record[3, :]) minimum(record[4, :]) mean(record[3, :]) mean(record[4, :]) maximum(record[3, :]) maximum(record[4, :])
        @printf "+-----------------------------------------------------------------------------------------------+\n\n"
    end
end

function test_dexp_inverse_formula()
    θ1, θ2 = rand(2)
    a = dexp_sxdx(θ1 - θ2)
    b = dexp_sxdx(θ1 + θ2)
    c = dexp_cxdx(θ1 - θ2)
    d = dexp_cxdx(θ1 + θ2)

    ap = dexp_invx(θ1 - θ2)
    bp = dexp_invx(θ1 + θ2)
    cp = -0.5 * (θ1 - θ2)
    dp = -0.5 * (θ1 + θ2)

    MatM = 0.5 .* [a+b -c-d c-d a-b; c+d a+b -a+b c-d; -c+d -a+b a+b -c-d; a-b -c+d c+d a+b]
    InvM = 0.5 .* [ap+bp cp+dp -cp+dp -bp+ap; -cp-dp ap+bp bp-ap -cp+dp; cp-dp bp-ap ap+bp cp+dp; -bp+ap cp-dp -cp-dp ap+bp]
    display(MatM)
    display(InvM)
    display(MatM * InvM)

    # t = rand();
    # a = dexp_sxdx(t)
    # c = dexp_cxdx(t)
    # ap = a/(a^2+c^2);
    # cp = c/(a^2+c^2);

    a = dexp_sxdx(θ1)
    b = dexp_cxdx(θ1)

    ap = dexp_invx(θ1)
    bp = -0.5 * θ1


    MatM22 = [a b; -b a]
    InvM22 = [ap -bp; bp ap]
    display(MatM22)
    display(InvM22)
    display(MatM22 * InvM22)
end

function test_dexp_para(n=10)
    mat_dim::Int = n
    blk_dim::Int = div(n, 2)
    VecA1 = rand(blk_dim)
    VecA2 = copy(VecA1)
    VecA2 .*= -1

    M_sys = dexp_SkewSymm_system(n)
    compute_dexp_SkewSymm_both_system!(M_sys, Ref(VecA1))

    MatP1 = zeros(div(blk_dim * (blk_dim + 1), 2), 4)
    MatP2 = zeros(blk_dim, 2)
    MatP3 = zeros(div(blk_dim * (blk_dim + 1), 2), 4)
    MatP4 = zeros(blk_dim, 2)

    compute_dexp_para_all!(VecA1, Ref(MatP1), Ref(MatP2), Ref(MatP3), Ref(MatP4))

    MatM1 = zeros(mat_dim, mat_dim)

    b_ind::Int = 1
    for r_ind = 1:blk_dim
        for c_ind = 1:(r_ind-1)
            MatM1[2*r_ind-1, 2*c_ind-1] = MatP1[b_ind, 1] + MatP1[b_ind, 2]
            MatM1[2*r_ind-1, 2*c_ind] = MatP1[b_ind, 1] - MatP1[b_ind, 2]
            MatM1[2*r_ind, 2*c_ind-1] = MatP1[b_ind, 3] + MatP1[b_ind, 4]
            MatM1[2*r_ind, 2*c_ind] = MatP1[b_ind, 3] - MatP1[b_ind, 4]

            MatM1[2*c_ind-1, 2*r_ind-1] = MatP3[b_ind, 1] + MatP3[b_ind, 2]
            MatM1[2*c_ind-1, 2*r_ind] = MatP3[b_ind, 1] - MatP3[b_ind, 2]
            MatM1[2*c_ind, 2*r_ind-1] = -MatP3[b_ind, 3] - MatP3[b_ind, 4]
            MatM1[2*c_ind, 2*r_ind] = -MatP3[b_ind, 3] + MatP3[b_ind, 4]

            b_ind += 1
        end

        MatM1[2*r_ind, 2*r_ind-1] = -MatP1[b_ind, 1]
        MatM1[2*r_ind-1, 2*r_ind] = -MatP1[b_ind, 1]
        MatM1[2*r_ind-1, 2*r_ind-1] = 0.0
        MatM1[2*r_ind, 2*r_ind] = 0.0

        b_ind += 1
    end

    if isodd(mat_dim)
        for d_ind = 1:blk_dim
            MatM1[mat_dim, 2*d_ind-1] = MatP2[d_ind, 1] + MatP2[d_ind, 2]
            MatM1[mat_dim, 2*d_ind] = MatP2[d_ind, 1] - MatP2[d_ind, 2]

            MatM1[2*d_ind-1, mat_dim] = MatP4[d_ind, 1] + MatP4[d_ind, 2]
            MatM1[2*d_ind, mat_dim] = MatP4[d_ind, 1] - MatP4[d_ind, 2]
        end
    end

    println(M_sys.mat_system[] ≈ MatM1)
end

function test_dexp_derivative(n=10; runs=5)
    X = rand(n, n)
    X .-= X'
    X .*= 4π

    Q = exp(X)
    S = log_SpecOrth(Ref(Q))

    println(exp(S) ≈ Q)

    M_sys = dexp_SkewSymm_system(n)

    S_saf = schurAngular_SkewSymm(Ref(S); regular=true)
    S_ang = getAngle(S_saf)
    S_vec = getVector(S_saf)

    compute_dexp_SkewSymm_both_system!(M_sys, Ref(S_ang))

    # display(S_ang);
    # display(M_sys.mat_system[])

    Ω = similar(X)
    Δ = similar(X)

    PTΩP = similar(X)
    PTΔP = similar(X)




    t_grid = range(1e-7, 1e-6, 100)

    plt = plot(title="Verfication of the directional derivative dexp",
        xlabel="scale, t",
        ylabel="|exp(M)+t⋅dexp_M[S] - exp(M+t⋅S)|")

    for ind = 1:runs
        Ω .= rand(n, n)
        Ω .-= Ω'


        dexp_SkewSymm!(Ref(Δ), Ref(Ω), M_sys, S_saf; cong=true)


        val = [norm(Q * exp(t .* Δ) .- exp(S .+ t .* Ω)) for t in t_grid]

        plot!(t_grid, val, xaxis=:log, yaxis=:log, label="Run $(ind), estimated order = $(round(log.(t_grid)\ log.(val), digits = 4))")
    end
    display(plt)
end

function test_dexp_threading_implementation(n=10)
    MatX = rand(n, n)
    MatX .-= MatX'
    MatX .*= 4π

    MatQ = exp(MatX)
    MatS = log_SpecOrth(Ref(MatQ))

    M_sys = dexp_SkewSymm_system(n)

    S_saf = schurAngular_SkewSymm(Ref(MatS); regular=true)
    compute_dexp_SkewSymm_both_system!(M_sys, S_saf.angle)


    MatΔ1 = rand(n, n)
    MatΔ1 .-= MatΔ1'
    MatΔ2 = copy(MatΔ1)

    MatΩ1 = similar(MatS)
    MatΩ2 = similar(MatS)

    blk_it = STRICT_LOWER_ITERATOR(n, lower_blk_traversal)

    wsp_cong = get_wsp_cong(n)

    println("dexp with raw loops")
    @time dexp_SkewSymm!(Ref(MatΩ1), Ref(MatΔ1), M_sys, S_saf, wsp_cong; cong=true, inv=false)

    println("dexp with $(Threads.nthreads()) threads")
    @time dexp_SkewSymm!(Ref(MatΩ2), Ref(MatΔ2), M_sys, S_saf, blk_it, wsp_cong; cong=true, inv=false)

    println("Correct forward action? \t", MatΩ1 ≈ MatΩ2)

    @time dexp_SkewSymm!(Ref(MatΔ2), Ref(MatΩ2), M_sys, S_saf, blk_it, wsp_cong; cong=true, inv=true)

    println("Correct backward action? \t", MatΔ1 ≈ MatΔ2)
end

function test_matmul_speed(dim_grid, runs=10; filename="")
    RecTime = zeros(runs * length(dim_grid), 2)
    record_ind = 1
    record_time::Float64 = 1000000

    for dim_ind in eachindex(dim_grid)
        n = dim_grid[dim_ind]

        for run in 1:runs
            MatA = rand(Float64, n, n)
            MatB = rand(Float64, n, n)
            MatAB = similar(MatA)

            CMatA = rand(ComplexF64, n, n)
            CMatB = rand(ComplexF64, n, n)
            CMatAB = similar(CMatA)

            record_time = 1000000
            for sample = 1:20
                stat = @timed begin
                    mul!(MatAB, MatA, MatB)
                end
                record_time = min(record_time, 1000 * (stat.time - stat.gctime))
            end

            RecTime[record_ind, 2] = record_time

            record_time = 1000000
            for sample = 1:20
                stat = @timed begin
                    mul!(CMatAB, CMatA, CMatB)
                end
                record_time = min(record_time, 1000 * (stat.time - stat.gctime))
            end

            RecTime[record_ind, 1] = record_time
            record_ind += 1
        end
    end
    dim_vec = vcat(ones(runs) * dim_grid'...)
    plt1 = scatter(dim_vec, RecTime,
        label=["Complex Matrix multiplication" "Real matrix multiplication"],
        markershape=[:circle :star5],
        markerstorkecolor=:auto,
        markerstrokewidth=0,
        lw=0,
        ms=1.5,
        markeralpha=0.5
    )

    plt2 = scatter(dim_vec, RecTime[:, 1] ./ RecTime[:, 2:end],
        label="Matrix multiplication speed up, real against complex",
        xlabel="dimension, n",
        ylabel="Speedup",
        markershape=:star5,
        markerstrokecolor=:auto,
        # ylim = (0, 10),
        yscale=:log2,
        markerstrokewidth=0,
        lw=0,
        ms=1.5,
        markeralpha=0.5
    )

    plot!(dim_grid, ones(length(dim_grid)), label=:none, linestyle=:dash)

    display(plt1)

    display(plt2)

    if filename != ""
        pos = findlast('.', filename)
        savefig(plt1, filename[1:(pos-1)] * "_time." * filename[(pos+1):end])
        savefig(plt2, filename[1:(pos-1)] * "_rate." * filename[(pos+1):end])
    end
end

function test_dexp_speed(n=10)
    MatM = rand(n, n)
    MatM .-= MatM'

    MatS = rand(n, n)
    MatS .-= MatS'

    MatM .*= 2π
    MatS .*= 2

    MatH = Matrix{ComplexF64}(undef, n, n)
    MateigVec = Matrix{ComplexF64}(undef, n, n)
    VeceigVal = Vector{ComplexF64}(undef, n)


    MatΔ_real = similar(MatM)
    MatΔ_comp = similar(MatM)

    M = Ref(MatM)
    S = Ref(MatS)
    H = Ref(MatH)
    eigVec = Ref(MateigVec)
    eigVal = Ref(VeceigVal)
    Δ_real = Ref(MatΔ_real)
    Δ_comp = Ref(MatΔ_comp)

    M_sys = dexp_SkewSymm_system(n)
    M_saf = SAFactor(n)

    wsp_cong = get_wsp_cong(n)
    wsp_saf = get_wsp_saf(n)
    wsp_complex_dexp = WSP(Matrix{ComplexF64}(undef, n, n), Matrix{ComplexF64}(undef, n, n))


    MatΔ = similar(MatS)
    MatΩ = similar(MatS)
    MatΔhat = similar(MatS)
    MatΩhat = similar(MatS)

    Δ = Ref(MatΔ)
    Ω = Ref(MatΩ)
    Δhat = Ref(MatΔhat)
    Ωhat = Ref(MatΩhat)

    MatΔ .= rand(n, n)
    MatΔ .-= MatΔ'

    println("---------------------------------------------------------------------------")
    println("Computation workload distribution:\n")

    println("Preprocessing: \tReal Schur angular decomposition:")
    @btime schurAngular_SkewSymm!($M_saf, $M, $wsp_saf; regular=false, order=false)

    println("Preprocessing: \tConstruct core linear map:")
    @btime compute_dexp_SkewSymm_forward_system!($M_sys, $M_saf.angle)

    println("Derivative action: \tCongruence on perturbation by Schur vectors:")
    @btime cong_dense!($Δhat, $M_saf.vector, $Δ, $wsp_cong; trans=true)
    getSkewSymm!(Δhat)        # Ensure skew symmetry.

    println("Derivative action: \tCore linear map:")
    @btime _dexp_SkewSymm_core!($Ωhat, $Δhat, $M_sys; inv=false)
    fill_upper_SkewSymm!(Ωhat) # Ensure skew symmetry.

    println("Derivative action: \tCongruence on perturbation by Schur vectors:")
    @btime cong_dense!($Ω, $M_saf.vector, $Ωhat, $wsp_cong; trans=false)
    getSkewSymm!(Ωhat)        # Ensure skew symmetry.
    println("---------------------------------------------------------------------------")


    println("Complex formula computation time:")
    @btime begin
        ef = eigen($MatM)
        $VeceigVal .= ef.values
        $MateigVec .= ef.vectors
        dexp_complex_para($H, $eigVal)
        dexp_complex_action($Δ_comp, $S, $H, $eigVec, $wsp_complex_dexp)
    end

    println("Real formula computation time:")
    @btime begin
        schurAngular_SkewSymm!($M_saf, $M, $wsp_saf; regular=false, order=false)

        compute_dexp_SkewSymm_forward_system!($M_sys, $M_saf.angle)

        dexp_SkewSymm!($Δ_real, $S, $M_sys, $M_saf, $wsp_cong; cong=true)
    end

    println("Same result?\t", MatΔ_real ≈ MatΔ_comp)
end

function test_dexp_threading_speed(n=10)
    MatX = rand(n, n)
    MatX .-= MatX'
    MatX .*= 4π

    MatQ = exp(MatX)
    MatS = log_SpecOrth(Ref(MatQ))

    M_sys = dexp_SkewSymm_system(n)

    S_saf = schurAngular_SkewSymm(Ref(MatS); regular=true)
    compute_dexp_SkewSymm_both_system!(M_sys, S_saf.angle)


    MatΔ = rand(n, n)
    MatΔ .-= MatΔ'
    MatΩ = similar(MatS)

    Δ = Ref(MatΔ)
    Ω = Ref(MatΩ)

    blk_it = STRICT_LOWER_ITERATOR(n, lower_blk_traversal)

    wsp_cong = get_wsp_cong(n)


    println("dexp forward action, dexp_SkewSymm!, with raw loops.")
    @btime dexp_SkewSymm!($Ω, $Δ, $M_sys, $S_saf, $wsp_cong; cong=true, inv=false)

    println("dexp forward action, dexp_SkewSymm!, with iterator plus $(Threads.nthreads()) threads.")
    @btime dexp_SkewSymm!($Ω, $Δ, $M_sys, $S_saf, $blk_it, $wsp_cong; cong=true, inv=false)

    println("\n")

    println("dexp backward action, dexp_SkewSymm!, with raw loops.")
    @btime dexp_SkewSymm!($Ω, $Δ, $M_sys, $S_saf, $wsp_cong; cong=true, inv=true)

    println("dexp backward action, dexp_SkewSymm!, with iterator plus $(Threads.nthreads()) threads.")
    @btime dexp_SkewSymm!($Ω, $Δ, $M_sys, $S_saf, $blk_it, $wsp_cong; cong=true, inv=true)

    println("\n")

    println("dexp forward core action without congruence, dexp_SkewSymm!, with raw loops.")
    @btime dexp_SkewSymm!($Ω, $Δ, $M_sys, $S_saf, $wsp_cong; cong=false, inv=false)

    println("dexp forward core action without congruence, dexp_SkewSymm!, with iterator plus $(Threads.nthreads()) threads.")
    @btime dexp_SkewSymm!($Ω, $Δ, $M_sys, $S_saf, $blk_it, $wsp_cong; cong=false, inv=false)

    println("\n")

    println("dexp backward core action without congruence, dexp_SkewSymm!, with raw loops.")
    @btime dexp_SkewSymm!($Ω, $Δ, $M_sys, $S_saf, $wsp_cong; cong=false, inv=true)

    println("dexp backward core action without congruence, dexp_SkewSymm!, with iterator plus $(Threads.nthreads()) threads.")
    @btime dexp_SkewSymm!($Ω, $Δ, $M_sys, $S_saf, $blk_it, $wsp_cong; cong=false, inv=true)
end

function test_dexp_inv_trasys_speed(n=10)
    MatX = rand(n, n)
    MatX .-= MatX'
    MatX .*= 4π

    MatQ = exp(MatX)
    MatS = log_SpecOrth(Ref(MatQ))

    M_sys = dexp_SkewSymm_system(n)

    S_saf = schurAngular_SkewSymm(Ref(MatS); regular=true)
    compute_dexp_SkewSymm_both_system!(M_sys, S_saf.angle; trans=true)


    MatΔ = rand(n, n)
    MatΔ .-= MatΔ'
    MatΩ1 = similar(MatS)
    MatΩ2 = similar(MatS)
    MatΩ3 = similar(MatS)



    Δ = Ref(MatΔ)
    Ω1 = Ref(MatΩ1)
    Ω2 = Ref(MatΩ2)
    Ω3 = Ref(MatΩ3)

    blk_it = STRICT_LOWER_ITERATOR(n, lower_blk_traversal)

    wsp_cong = get_wsp_cong(n)


    println("dexp backward action by compact system with raw loops.")
    @btime dexp_SkewSymm!($Ω3, $Δ, $M_sys, $S_saf, $wsp_cong; cong=true, inv=true)

    println("dexp backward action by compact system with iterator plus $(Threads.nthreads()) threads.")
    @btime dexp_SkewSymm!($Ω1, $Δ, $M_sys, $S_saf, $blk_it, $wsp_cong; cong=true, inv=true, compact=true)

    println("dexp backward action by transposed system with iterator plus $(Threads.nthreads()) threads.")
    @btime dexp_SkewSymm!($Ω2, $Δ, $M_sys, $S_saf, $blk_it, $wsp_cong; cong=true, inv=true, compact=false)

    println("Same result? \t", MatΩ1 ≈ MatΩ2, "\n")


    println("dexp backward core action by compact system with raw loops.")
    @btime dexp_SkewSymm!($Ω3, $Δ, $M_sys, $S_saf, $wsp_cong; cong=false, inv=true)

    println("dexp backward core action by compact system with iterator plus $(Threads.nthreads()) threads.")
    @btime dexp_SkewSymm!($Ω1, $Δ, $M_sys, $S_saf, $blk_it, $wsp_cong; cong=false, inv=true, compact=true)

    println("dexp backward core action by transposed system with iterator plus $(Threads.nthreads()) threads.")
    @btime dexp_SkewSymm!($Ω2, $Δ, $M_sys, $S_saf, $blk_it, $wsp_cong; cong=false, inv=true, compact=false)

    println("Same result? \t", MatΩ1 ≈ MatΩ2, "\n")


end

function test_dexp_action_implementation_speed_vesus_dim(dim_grid, runs=10; filename="", seed=9527)

    RecTime = zeros(runs * length(dim_grid), 4)

    rand_eng = MersenneTwister(seed)
    record_ind::Int = 1
    record_time::Float64 = 1000000
    for dim_ind in eachindex(dim_grid)
        n = dim_grid[dim_ind]
        MatM = zeros(n, n)
        MatS = zeros(n, n)
        MatΔ1 = zeros(n, n)
        MatΔ2 = zeros(n, n)
        MatΔ3 = zeros(n, n)
        MatΔ4 = zeros(n, n)

        blk_it = STRICT_LOWER_ITERATOR(n, lower_blk_traversal)



        MatH = Matrix{ComplexF64}(undef, n, n)
        MateigVec = Matrix{ComplexF64}(undef, n, n)
        VeceigVal = Vector{ComplexF64}(undef, n)
        wsp_complex_dexp = WSP(Matrix{ComplexF64}(undef, n, n), Matrix{ComplexF64}(undef, n, n))

        M_saf = SAFactor(n)
        M_sys = dexp_SkewSymm_system(n)
        wsp_saf = get_wsp_saf(n)
        wsp_cong = get_wsp_cong(n)

        M = Ref(MatM)
        S = Ref(MatS)
        Δ1 = Ref(MatΔ1)
        Δ2 = Ref(MatΔ2)
        Δ3 = Ref(MatΔ3)
        Δ4 = Ref(MatΔ4)
        H = Ref(MatH)
        eigVec = Ref(MateigVec)
        eigVal = Ref(VeceigVal)

        for run_ind = 1:runs
            MatM .= rand(rand_eng, n, n)
            MatM .-= MatM'
            MatM .*= 2π


            MatS .= rand(rand_eng, n, n)
            MatS .-= MatS'
            MatS .*= 2π

            ef = eigen(MatM)
            VeceigVal .= ef.values
            MateigVec .= ef.vectors
            dexp_complex_para(H, eigVal)

            schurAngular_SkewSymm!(M_saf, M, wsp_saf; order=false, regular=false)
            compute_dexp_SkewSymm_both_system!(M_sys, M_saf.angle)

            record_time = 1000000
            for sample = 1:20
                stat = @timed begin
                    dexp_complex_action(Δ1, S, H, eigVec, wsp_complex_dexp)
                end
                record_time = min(record_time, 1000 * (stat.time - stat.gctime))
            end
            RecTime[record_ind, 1] = record_time

            record_time = 1000000
            for sample = 1:20
                stat = @timed begin
                    dexp_complex_action_thread(Δ2, S, H, eigVec, wsp_complex_dexp)
                end
                record_time = min(record_time, 1000 * (stat.time - stat.gctime))
            end
            RecTime[record_ind, 2] = record_time

            record_time = 1000000
            for sample = 1:20
                stat = @timed begin
                    dexp_SkewSymm!(Δ3, S, M_sys, M_saf, wsp_cong; inv=false, cong=true)
                end
                record_time = min(record_time, 1000 * (stat.time - stat.gctime))
            end
            RecTime[record_ind, 3] = record_time

            record_time = 1000000
            for sample = 1:20
                stat = @timed begin
                    dexp_SkewSymm!(Δ4, S, M_sys, M_saf, blk_it, wsp_cong; inv=false, cong=true)
                end
                record_time = min(record_time, 1000 * (stat.time - stat.gctime))
            end
            RecTime[record_ind, 4] = record_time
            record_ind += 1
        end
    end

    dim_vec = vcat(ones(runs) * dim_grid'...)

    plt1 = scatter(dim_vec, RecTime,
        label=["Complex formula, raw loop" "Complex formula, threading" "Real formula, raw loop" "Real formula, threading"],
        # xlabel="dimension, n",
        ylabel="Compute time (ms)",
        markershape=[:circle :circle :circle :circle],
        markerstrokecolor=:auto,
        # ylim = (0, 20),
        # yscale=:log2,
        markerstrokewidth=[0 0 0 0],
        lw=[0 0 0 0],
        ms=1.5,
        markeralpha=0.5
    )

    plt2 = scatter(dim_vec, RecTime[:, 1] ./ RecTime,
        label=["Complex formula, raw loop" "Complex formula, threading" "Real formula, raw loop" "Real formula, threading"],
        xlabel="dimension, n",
        ylabel="Ratio",
        markershape=[:circle :circle :circle :circle],
        markerstrokecolor=:auto,
        # ylim = (0, 10),
        # yscale=:log2,
        markerstrokewidth=[0 0],
        lw=[0 0],
        ms=1.5,
        markeralpha=0.5
    )

    plt = plot(layout=(2, 1), size=(600, 800), ylim=[(0, 10) (0, 10)],
        plt1, plt2
    )

    plt_log = plot(layout=(2, 1), size=(600, 800), yscale=:log2,
        plt1, plt2
    )

    # display(plt)

    # display(plt_log)


    if filename != ""
        pos = findlast('.', filename)
        savefig(plot(lw=0, markerstrokewidth=0, plt1), filename[1:(pos-1)] * "_time." * filename[(pos+1):end])
        savefig(plot(lw=0, markerstrokewidth=0, plt2), filename[1:(pos-1)] * "_rate." * filename[(pos+1):end])
        savefig(plot(yscale=:log2, lw=0, markerstrokewidth=0, plt1), filename[1:(pos-1)] * "_time_logscale." * filename[(pos+1):end])
        savefig(plot(yscale=:log2, lw=0, markerstrokewidth=0, plt2), filename[1:(pos-1)] * "_rate_logscale." * filename[(pos+1):end])
    end
end

function test_dexp_speed_vesus_dim(dim_grid, runs=10; filename="", seed=9527)

    RecTime = zeros(runs * length(dim_grid), 5)

    rand_eng = MersenneTwister(seed)
    record_ind::Int = 1
    record_time::Float64 = 1000000
    for dim_ind in eachindex(dim_grid)
        n = dim_grid[dim_ind]
        M = zeros(n, n)
        S = zeros(n, n)
        Δ = zeros(n, n)

        C1 = zeros(ComplexF64, n, n)
        C2 = zeros(ComplexF64, n, n)

        blk_it = STRICT_LOWER_ITERATOR(n, lower_blk_traversal)



        H = Matrix{ComplexF64}(undef, n, n)
        eigVec = Matrix{ComplexF64}(undef, n, n)
        eigVal = Vector{ComplexF64}(undef, n)
        wsp_complex_dexp = WSP(Matrix{ComplexF64}(undef, n, n), Matrix{ComplexF64}(undef, n, n), Matrix{Float64}(undef, n, n), Matrix{Float64}(undef, n, n))

        M_saf = SAFactor(n)
        M_sys = dexp_SkewSymm_system(n)
        wsp_saf = get_wsp_saf(n)
        wsp_cong = get_wsp_cong(n)


        for run_ind = 1:runs
            M .= rand(rand_eng, n, n)
            M .-= M'
            M .*= 2π


            S .= rand(rand_eng, n, n)
            S .-= S'
            S .*= 2π

            record_time = 1000000
            for sample = 1:10
                stat = @timed begin
                    ef = eigen(M)
                    eigVal .= ef.values
                    eigVec = ef.vectors
                    dexp_complex_para(Ref(H), Ref(eigVal))
                end
                record_time = min(record_time, 1000 * (stat.time - stat.gctime))
            end
            RecTime[record_ind, 1] = record_time

            record_time = 1000000
            for sample = 1:10
                stat = @timed dexp_complex_action(Ref(Δ), Ref(S), Ref(H), Ref(eigVec), wsp_complex_dexp)
                record_time = min(record_time, 1000 * (stat.time - stat.gctime))
            end
            RecTime[record_ind, 2] = record_time

            record_time = 1000000
            for sample = 1:10
                stat = @timed begin
                    schurAngular_SkewSymm!(M_saf, Ref(M), wsp_saf; order=false, regular=false)
                    compute_dexp_SkewSymm_both_system!(M_sys, M_saf.angle)
                end
                record_time = min(record_time, 1000 * (stat.time - stat.gctime))
            end
            RecTime[record_ind, 3] = record_time

            record_time = 1000000
            for sample = 1:10
                stat = @timed dexp_SkewSymm!(Ref(Δ), Ref(S), M_sys, M_saf, blk_it, wsp_cong; inv=false, cong=true)
                record_time = min(record_time, 1000 * (stat.time - stat.gctime))
            end
            RecTime[record_ind, 4] = record_time

            record_time = 1000000
            for sample = 1:10
                C1 = eigVec' * S * eigVec
                stat = @timed _dexp_complex_action_core(Ref(C2), Ref(C1), Ref(H))
                record_time = min(record_time, 1000 * (stat.time - stat.gctime))
            end
            RecTime[record_ind, 5] = record_time

            record_ind += 1
        end
    end

    dim_vec = vcat(ones(runs) * dim_grid'...)

    total_time_complex = RecTime[:, 1] .+ RecTime[:, 2]
    total_time_real = RecTime[:, 3] .+ RecTime[:, 4]


    plt1 = scatter(dim_vec, [total_time_complex RecTime[:, 1] total_time_real RecTime[:, 3]],
        label=["Complex formula" "Complex preprocessing(eigen)" "Real formula" "Real preprocessing(schur)"],
        # xlabel="dimension, n",
        ylabel="Compute time (ms)",
        markershape=[:circle :circle :circle :circle],
        markerstrokecolor=:auto,
        # ylim = (0, 20),
        # yscale=:log2,
        markerstrokewidth=[0 0 0 0],
        lw=[0 0 0 0],
        ms=1.5,
        markeralpha=0.5
    )

    plt2 = scatter(dim_vec, [RecTime[:, 2] RecTime[:, 4]],
        label=["Complex formula, action only" "Real formula, action only"],
        xlabel="dimension, n",
        ylabel="Compute time (ms)",
        markershape=[:circle :circle],
        markerstrokecolor=:auto,
        # ylim = (0, 10),
        # yscale=:log2,
        markerstrokewidth=[0 0],
        lw=[0 0],
        ms=1.5,
        markeralpha=0.5
    )

    # plt2 = scatter(dim_vec, [RecTime[:, 2] RecTime[:, 5] RecTime[:, 4]],
    #     label=["Complex formula, action only" "Complex formula, Hadamard action" "Real formula, action only"],
    #     xlabel="dimension, n",
    #     ylabel="Compute time (ms)",
    #     markershape = [:circle :circle :circle],
    #     markerstrokecolor = :auto,
    #     # ylim = (0, 10),
    #     # yscale=:log2,
    #     markerstrokewidth=[0 0 0],
    #     lw=[0 0 0],
    #     ms=1.5,
    #     markeralpha=0.5
    # )

    plt = plot(layout=(2, 1), size=(600, 800), ylim=[(0, 10) (0, 10)],
        plt1, plt2
    )

    plt_log = plot(layout=(2, 1), size=(600, 800), yscale=:log2,
        plt1, plt2
    )

    # display(plt)

    # display(plt_log)


    if filename != ""
        pos = findlast('.', filename)
        savefig(plot(ylim=[(0, 10) (0, 10)], lw=0, markerstrokewidth=0, plt1), filename)
        savefig(plot(ylim=[(0, 10) (0, 10)], lw=0, markerstrokewidth=0, plt2), filename[1:(pos-1)] * "_action." * filename[(pos+1):end])
        savefig(plot(yscale=:log2, lw=0, markerstrokewidth=0, plt1), filename[1:(pos-1)] * "_logscale." * filename[(pos+1):end])
        savefig(plot(yscale=:log2, lw=0, markerstrokewidth=0, plt2), filename[1:(pos-1)] * "_action_logscale." * filename[(pos+1):end])
    end
end


