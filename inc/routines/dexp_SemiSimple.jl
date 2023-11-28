include("../global_path.jl")
include(joinpath(JULIA_INCLUDE_PATH, "workspace.jl"))
include("hadamard.jl")

using LinearAlgebra, LoopVectorization

mutable struct dexp_SemiSimple_system
    # Lower blocks for the system and upper blocks for the inverse system
    mat_psi_real::Ref{Matrix{Float64}}
    mat_psi_imag::Ref{Matrix{Float64}}
    mat_psi_comp::Ref{Matrix{ComplexF64}}
    mat_eig_vect::Ref{Matrix{ComplexF64}}
    mat_inv_vect::Ref{Matrix{ComplexF64}}

    mat_dim::Int

    dexp_SemiSimple_system(n::Int) = new(Ref(Matrix{Float64}(undef, n, n)), Ref(Matrix{Float64}(undef, n, n)), Ref(Matrix{ComplexF64}(undef, n, n)), Ref(Matrix{ComplexF64}(undef, n, n)), Ref(Matrix{ComplexF64}(undef, n, n)), n)


    # function dexp_SkewSymm_system(M::Ref{Matrix{Float64}}; trans::Bool=false)
    #     sys = dexp_SkewSymm_system(size(M[], 1))
    #     M_saf = schurAngular_SkewSymm(M; order=false)
    #     compute_dexp_SkewSymm_both_system!(sys, M_saf.angle; trans=trans)
    #     return sys
    # end

    # function dexp_SkewSymm_system(saf::SAFactor; trans::Bool=false)
    #     sys = dexp_SkewSymm_system(size(saf.vector[], 1))
    #     compute_dexp_SkewSymm_both_system!(sys, saf.angle; trans=trans)
    # end

    # function dexp_SkewSymm_system(angle::Ref{Vector{Float64}}; trans::Bool=false)
    #     sys = dexp_SkewSymm_system(2 * length(angle[]) + 1)
    #     compute_dexp_SkewSymm_both_system!(sys, angle; trans=trans)
    # end
end

function compute_dexp_SemiSimple_system!(sys::dexp_SemiSimple_system, eig::Eigen{ComplexF64,ComplexF64,Matrix{ComplexF64},Vector{ComplexF64}})
    MatReal = sys.mat_psi_real[]
    MatImag = sys.mat_psi_imag[]
    MatComp = sys.mat_psi_comp[]

    n = sys.mat_dim

    eigvals = eig.values
    phi::ComplexF64 = 0.0 + 0.0im
    dif::ComplexF64 = 0.0 + 0.0im

    @inbounds for r_ind in 1:n, c_ind in 1:n
        if (norm(eigvals[c_ind] - eigvals[r_ind]) < 1e-14)
            MatComp[r_ind, c_ind] = 1.0
            MatReal[r_ind, c_ind] = 1.0
            MatImag[r_ind, c_ind] = 0.0
        else
            MatComp[r_ind, c_ind] = (1.0 - exp(eigvals[c_ind] - eigvals[r_ind])) / (eigvals[r_ind] - eigvals[c_ind])
            MatReal[c_ind, r_ind] = MatComp[r_ind, c_ind].re
            MatImag[c_ind, r_ind] = MatComp[r_ind, c_ind].im
        end
    end

    # for c_ind in 1:n
    #     @inbounds phi = exp(eigvals[c_ind])
    #     @inbounds MatComp[c_ind, c_ind] = phi
    #     @inbounds MatReal[c_ind, c_ind] = phi.re
    #     @inbounds MatImag[c_ind, c_ind] = phi.im
    #     for r_ind = (c_ind+1):n
    #         @inbounds dif = eigvals[c_ind] - eigvals[r_ind]
    #         @inbounds if (norm(dif) < 1e-14)
    #             @inbounds phi = exp(eigvals[r_ind])
    #             @inbounds phi = 1.0
    #             @inbounds MatComp[r_ind, c_ind] = phi
    #             @inbounds MatReal[r_ind, c_ind] = phi.re
    #             @inbounds MatImag[r_ind, c_ind] = phi.im
    #             @inbounds MatComp[c_ind, r_ind] = phi
    #             @inbounds MatReal[c_ind, r_ind] = phi.re
    #             @inbounds MatImag[c_ind, r_ind] = phi.im
    #         else
    #             @inbounds phi = (exp(eigvals[r_ind]) - exp(eigvals[c_ind]))
    #             @inbounds phi /= (eigvals[r_ind] - eigvals[c_ind])
    #             @inbounds phi /= exp(eigvals[r_ind])
    #             # @inbounds phi = (exp(dif) - 1.0) / dif

    #             @inbounds MatComp[r_ind, c_ind] = phi
    #             @inbounds MatReal[r_ind, c_ind] = phi.re
    #             @inbounds MatImag[r_ind, c_ind] = phi.im
    #             @inbounds MatComp[c_ind, r_ind] = phi
    #             @inbounds MatReal[c_ind, r_ind] = phi.re
    #             @inbounds MatImag[c_ind, r_ind] = phi.im
    #         end
    #     end
    # end

    # for c_ind in 1:n
    #     @inbounds phi = 1.0
    #     @inbounds MatComp[c_ind, c_ind] = phi
    #     @inbounds MatReal[c_ind, c_ind] = phi.re
    #     @inbounds MatImag[c_ind, c_ind] = phi.im
    #     for r_ind = (c_ind+1):n
    #         dif = eigvals[r_ind] - eigvals[c_ind]
    #         @inbounds if (norm(dif) < 1e-14)
    #             # @inbounds phi = exp(eigvals[r_ind])
    #             @inbounds phi = 1.0
    #             @inbounds MatComp[r_ind, c_ind] = phi
    #             @inbounds MatReal[r_ind, c_ind] = phi.re
    #             @inbounds MatImag[r_ind, c_ind] = phi.im
    #             @inbounds MatComp[c_ind, r_ind] = -phi
    #             @inbounds MatReal[c_ind, r_ind] = -phi.re
    #             @inbounds MatImag[c_ind, r_ind] = phi.im
    #         else
    #             # @inbounds phi = (1.0 - exp(eigvals[c_ind] - eigvals[r_ind]))
    #             # @inbounds phi /= (eigvals[r_ind] - eigvals[c_ind])
    #             @inbounds phi = (exp(dif) - 1.0) / dif

    #             @inbounds MatComp[r_ind, c_ind] = phi
    #             @inbounds MatReal[r_ind, c_ind] = phi.re
    #             @inbounds MatImag[r_ind, c_ind] = phi.im
    #             @inbounds MatComp[c_ind, r_ind] = -phi'
    #             @inbounds MatReal[c_ind, r_ind] = -phi.re
    #             @inbounds MatImag[c_ind, r_ind] = phi.im
    #         end
    #     end
    # end

    MatEigVec = sys.mat_eig_vect[]
    MatInvVec = sys.mat_inv_vect[]

    copy!(MatEigVec, eig.vectors)
    MatInvVec .= inv(MatEigVec)
end

function compute_dexp_Normal_system!(sys::dexp_SemiSimple_system, eig::Eigen{ComplexF64,ComplexF64,Matrix{ComplexF64},Vector{ComplexF64}})
    MatReal = sys.mat_psi_real[]
    MatImag = sys.mat_psi_imag[]
    MatComp = sys.mat_psi_comp[]

    n = sys.mat_dim

    eigvals = eig.values
    phi::ComplexF64 = 0.0 + 0.0im
    dif::ComplexF64 = 0.0 + 0.0im

    @inbounds for r_ind in 1:n, c_ind in 1:n
        if (norm(eigvals[c_ind] - eigvals[r_ind]) < 1e-14)
            MatComp[r_ind, c_ind] = 1.0
            MatReal[r_ind, c_ind] = 1.0
            MatImag[r_ind, c_ind] = 0.0
        else
            MatComp[r_ind, c_ind] = (1.0 - exp(eigvals[c_ind] - eigvals[r_ind])) / (eigvals[r_ind] - eigvals[c_ind])
            MatReal[c_ind, r_ind] = MatComp[r_ind, c_ind].re
            MatImag[c_ind, r_ind] = MatComp[r_ind, c_ind].im
        end
    end

    # for c_ind in 1:n
    #     for r_ind in 1:n
    #         @inbounds dif = eigvals[c_ind] - eigvals[r_ind]
    #         if (norm(dif) < 1e-14)
    #             @inbounds phi = exp(eigvals[r_ind])
    #             @inbounds MatComp[r_ind, c_ind] = phi
    #             @inbounds MatReal[r_ind, c_ind] = phi.re
    #             @inbounds MatImag[r_ind, c_ind] = phi.im
    #         else
    #             @inbounds phi = (exp(eigvals[r_ind]) - exp(eigvals[c_ind]))
    #             @inbounds phi /= (eigvals[r_ind] - eigvals[c_ind])
    #             # @inbounds phi = (exp(dif) - 1.0) / dif

    #             @inbounds MatComp[r_ind, c_ind] = phi
    #             @inbounds MatReal[r_ind, c_ind] = phi.re
    #             @inbounds MatImag[r_ind, c_ind] = phi.im
    #         end
    #     end
    # end

    # for c_ind in 1:n
    #     @inbounds phi = exp(eigvals[c_ind])
    #     @inbounds MatComp[c_ind, c_ind] = phi
    #     @inbounds MatReal[c_ind, c_ind] = phi.re
    #     @inbounds MatImag[c_ind, c_ind] = phi.im
    #     @turbo for r_ind = (c_ind+1):n
    #         @inbounds dif = eigvals[c_ind] - eigvals[r_ind]
    #         @inbounds if (norm(dif) < 1e-14)
    #             @inbounds phi = exp(eigvals[r_ind])
    #             @inbounds phi = 1.0
    #             @inbounds MatComp[r_ind, c_ind] = phi
    #             @inbounds MatReal[r_ind, c_ind] = phi.re
    #             @inbounds MatImag[r_ind, c_ind] = phi.im
    #             @inbounds MatComp[c_ind, r_ind] = phi
    #             @inbounds MatReal[c_ind, r_ind] = phi.re
    #             @inbounds MatImag[c_ind, r_ind] = phi.im
    #         else
    #             @inbounds phi = (exp(eigvals[r_ind]) - exp(eigvals[c_ind]))
    #             @inbounds phi /= (eigvals[r_ind] - eigvals[c_ind])
    #             # @inbounds phi = (exp(dif) - 1.0) / dif

    #             @inbounds MatComp[r_ind, c_ind] = phi
    #             @inbounds MatReal[r_ind, c_ind] = phi.re
    #             @inbounds MatImag[r_ind, c_ind] = phi.im
    #             @inbounds MatComp[c_ind, r_ind] = phi
    #             @inbounds MatReal[c_ind, r_ind] = phi.re
    #             @inbounds MatImag[c_ind, r_ind] = phi.im
    #         end
    #     end
    # end

    # for c_ind in 1:n
    #     @inbounds phi = 1.0
    #     @inbounds MatComp[c_ind, c_ind] = phi
    #     @inbounds MatReal[c_ind, c_ind] = phi.re
    #     @inbounds MatImag[c_ind, c_ind] = phi.im
    #     for r_ind = (c_ind+1):n
    #         dif = eigvals[c_ind] - eigvals[r_ind]
    #         @inbounds if (norm(dif) < 1e-14)
    #             # @inbounds phi = exp(eigvals[r_ind])
    #             @inbounds phi = 1.0
    #             @inbounds MatComp[r_ind, c_ind] = phi
    #             @inbounds MatReal[r_ind, c_ind] = phi.re
    #             @inbounds MatImag[r_ind, c_ind] = phi.im
    #             @inbounds MatComp[c_ind, r_ind] = -phi'
    #             @inbounds MatReal[c_ind, r_ind] = -phi.re
    #             @inbounds MatImag[c_ind, r_ind] = phi.im
    #         else
    #             # @inbounds phi = (exp(eigvals[r_ind]) - exp(eigvals[c_ind]))
    #             # @inbounds phi /= (eigvals[r_ind] - eigvals[c_ind])
    #             @inbounds phi = (exp(dif) - 1.0) / dif

    #             @inbounds MatComp[r_ind, c_ind] = phi
    #             @inbounds MatReal[r_ind, c_ind] = phi.re
    #             @inbounds MatImag[r_ind, c_ind] = phi.im
    #             @inbounds MatComp[c_ind, r_ind] = -phi'
    #             @inbounds MatReal[c_ind, r_ind] = -phi.re
    #             @inbounds MatImag[c_ind, r_ind] = phi.im
    #         end
    #     end
    # end

    MatEigVec = sys.mat_eig_vect[]
    MatInvVec = sys.mat_inv_vect[]

    copy!(MatEigVec, eig.vectors)
    copy!(MatInvVec, eig.vectors')

end

function get_wsp_dexp_SemiSimple(n)
    return WSP(Matrix{Float64}(undef, n, n), Matrix{Float64}(undef, n, n), Matrix{ComplexF64}(undef, n, n))
end

function dexp_SemiSimple!(Δ::Ref{Matrix{ComplexF64}}, S::Ref{Matrix{ComplexF64}}, sys::dexp_SemiSimple_system, wsp_dexp_SemiSimple::WSP=WSP(Matrix{ComplexF64}(undef, size(Δ[])...)); inv::Bool=false, simi::Bool=true)
    if simi
        MatTemp = wsp_dexp_SemiSimple[1]

        mul!(MatTemp, sys.mat_inv_vect[], S[])
        mul!(Δ[], MatTemp, sys.mat_eig_vect[])

        hadamard!(Δ, sys.mat_psi_comp; inv=inv, Hadamard_Mode='R')

        mul!(MatTemp, sys.mat_eig_vect[], Δ[])
        mul!(Δ[], MatTemp, sys.mat_inv_vect[])
    else
        hadamard!(Δ, S, sys.mat_psi_comp; inv=inv, Hadamard_Mode='R')

        # MatS = S[]
        # MatPSI = sys.mat_psi_comp[]

        # MatΔ = Δ[]
        # n::Int = size(MatS, 1)

        # @inbounds for c_ind in 1:n
        #     @inbounds for r_ind in (c_ind+1):n
        #         MatΔ[r_ind, c_ind] = MatPSI[r_ind, c_ind] * MatS[r_ind, c_ind]
        #         MatΔ[c_ind, r_ind] = -MatΔ[r_ind, c_ind]
        #     end
        # end
    end
end

function dexp_SemiSimple!(Δ::Ref{Matrix{ComplexF64}}, S::Ref{Matrix{ComplexF64}}, blk_it::STRICT_LOWER_ITERATOR, sys::dexp_SemiSimple_system, wsp_dexp_SemiSimple::WSP=WSP(Matrix{ComplexF64}(undef, size(Δ[])...)); inv::Bool=false, simi::Bool=true)
    if simi
        MatTemp = wsp_dexp_SemiSimple[1]

        mul!(MatTemp, sys.mat_inv_vect[], S[])
        mul!(Δ[], MatTemp, sys.mat_eig_vect[])

        hadamard!(Δ, sys.mat_psi_comp, blk_it)

        mul!(MatTemp, sys.mat_eig_vect[], Δ[])
        mul!(Δ[], MatTemp, sys.mat_inv_vect[])
    else
        hadamard!(Δ, S, sys.mat_psi_comp, blk_it)
    end
end