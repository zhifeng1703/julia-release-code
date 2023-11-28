include("../global_path.jl")
include(joinpath(JULIA_INCLUDE_PATH, "workspace.jl"))

using LinearAlgebra

# No rescale is included

@inline get_wsp_dexp_ScaleSquare(n::Int) = WSP(Matrix{Float64}(undef, n, n), Matrix{Float64}(undef, n, n), Matrix{Float64}(undef, n, n), Matrix{Float64}(undef, n, n), Matrix{Float64}(undef, n, n),
    Matrix{Float64}(undef, n, n), Matrix{Float64}(undef, n, n), Matrix{Float64}(undef, n, n), Matrix{Float64}(undef, n, n), Matrix{Float64}(undef, n, n))

mutable struct dexp_ScaleSquare_system
    A2::Ref{Matrix{Float64}}
    A4::Ref{Matrix{Float64}}
    A6::Ref{Matrix{Float64}}
    W::Ref{Matrix{Float64}}
    W1::Ref{Matrix{Float64}}
    W2::Ref{Matrix{Float64}}
    Z1::Ref{Matrix{Float64}}
    Z2::Ref{Matrix{Float64}}
    U::Ref{Matrix{Float64}}
    V::Ref{Matrix{Float64}}
    b::Ref{Vector{Int64}}
    lu::LU{Float64,Matrix{Float64},Vector{Int64}}

    dexp_ScaleSquare_system(n::Int) = new(Ref(zeros(n, n)), Ref(zeros(n, n)), Ref(zeros(n, n)), Ref(zeros(n, n)), Ref(zeros(n, n)),
        Ref(zeros(n, n)), Ref(zeros(n, n)), Ref(zeros(n, n)), Ref(zeros(n, n)), Ref(zeros(n, n)), Ref(zeros(Int64, 14)), lu(diagm(ones(n))))
end

function compute_dexp_ScaleSquare_system!(A_sys::dexp_ScaleSquare_system, A::Ref{Matrix{Float64}})
    MatA = A[]

    MatA2 = A_sys.A2[]
    MatA4 = A_sys.A4[]
    MatA6 = A_sys.A6[]
    MatW = A_sys.W[]
    MatW1 = A_sys.W1[]
    MatW2 = A_sys.W2[]
    MatZ1 = A_sys.Z1[]
    MatZ2 = A_sys.Z2[]
    MatU = A_sys.U[]
    MatV = A_sys.V[]
    VecB = A_sys.b[]

    VecB .= [64764752532480000, 32382376266240000, 7771770303897600, 1187353796428800, 129060195264000, 10559470521600,
        670442572800, 33522128640, 1323241920, 40840800, 960960, 16380, 182, 1]

    mul!(MatA2, MatA, MatA)
    mul!(MatA4, MatA2, MatA2)
    mul!(MatA6, MatA2, MatA4)

    fill!(MatW1, 0.0)
    axpy!(VecB[14], MatA6, MatW1)
    axpy!(VecB[12], MatA4, MatW1)
    axpy!(VecB[10], MatA2, MatW1)

    fill!(MatW2, 0.0)
    axpy!(VecB[8], MatA6, MatW2)
    axpy!(VecB[6], MatA4, MatW2)
    axpy!(VecB[4], MatA2, MatW2)
    @inbounds for ind in axes(MatW2, 1)
        MatW2[ind, ind] += VecB[2]
    end

    fill!(MatZ1, 0.0)
    axpy!(VecB[13], MatA6, MatZ1)
    axpy!(VecB[11], MatA4, MatZ1)
    axpy!(VecB[9], MatA2, MatZ1)

    fill!(MatZ2, 0.0)
    axpy!(VecB[7], MatA6, MatZ2)
    axpy!(VecB[5], MatA4, MatZ2)
    axpy!(VecB[3], MatA2, MatZ2)
    @inbounds for ind in axes(MatZ2, 1)
        MatZ2[ind, ind] += VecB[1]
    end

    copy!(MatW, MatW2)
    mul!(MatW, MatA6, MatW1, 1, 1)

    mul!(MatU, MatA, MatW1)

    copy!(MatV, MatZ2)
    mul!(MatV, MatA6, MatZ1, 1, 1)

    A_sys.lu = lu(MatV .- MatU)
end

function _dexp_ScaleSquare_m13_no_scale!(R::Ref{Matrix{Float64}}, Δ::Ref{Matrix{Float64}}, A::Ref{Matrix{Float64}}, A_sys::dexp_ScaleSquare_system, E::Ref{Matrix{Float64}},
    wsp_dexp_NonNegative::WSP=get_wsp_dexp_nonnegative(size(R[], 1)))

    MatR = R[]
    MatΔ = Δ[]
    MatA = A[]
    MatE = E[]

    MatA2 = A_sys.A2[]
    MatA4 = A_sys.A4[]
    MatA6 = A_sys.A6[]
    MatW = A_sys.W[]
    MatW1 = A_sys.W1[]
    MatW2 = A_sys.W2[]
    MatZ1 = A_sys.Z1[]
    MatZ2 = A_sys.Z2[]
    MatU = A_sys.U[]
    MatV = A_sys.V[]
    VecB = A_sys.b[]

    MatM2 = wsp_dexp_NonNegative[1]
    MatM4 = wsp_dexp_NonNegative[2]
    MatM6 = wsp_dexp_NonNegative[3]
    MatLW1 = wsp_dexp_NonNegative[4]
    MatLW2 = wsp_dexp_NonNegative[5]
    MatLZ1 = wsp_dexp_NonNegative[6]
    MatLZ2 = wsp_dexp_NonNegative[7]
    MatLW = wsp_dexp_NonNegative[8]
    MatLU = wsp_dexp_NonNegative[9]
    MatLV = wsp_dexp_NonNegative[10]



    mul!(MatM2, MatA, MatE)
    mul!(MatM2, MatE, MatA, 1, 1)

    mul!(MatM4, MatA2, MatM2)
    mul!(MatM4, MatM2, MatA2, 1, 1)

    mul!(MatM6, MatA4, MatM2)
    mul!(MatM6, MatM2, MatA4, 1, 1)

    fill!(MatLW1, 0.0)
    axpy!(VecB[14], MatM6, MatLW1)
    axpy!(VecB[12], MatM4, MatLW1)
    axpy!(VecB[10], MatM2, MatLW1)

    fill!(MatLW2, 0.0)
    axpy!(VecB[8], MatM6, MatLW2)
    axpy!(VecB[6], MatM4, MatLW2)
    axpy!(VecB[4], MatM2, MatLW2)

    fill!(MatLZ1, 0.0)
    axpy!(VecB[13], MatM6, MatLZ1)
    axpy!(VecB[11], MatM4, MatLZ1)
    axpy!(VecB[9], MatM2, MatLZ1)

    fill!(MatLZ2, 0.0)
    axpy!(VecB[7], MatM6, MatLZ2)
    axpy!(VecB[5], MatM4, MatLZ2)
    axpy!(VecB[3], MatM2, MatLZ2)

    mul!(MatLW, MatA6, MatLW1)
    mul!(MatLW, MatM6, MatW1, 1, 1)
    axpy!(1, MatLW2, MatLW)

    mul!(MatLU, MatA, MatLW)
    mul!(MatLU, MatE, MatW, 1, 1)

    mul!(MatLV, MatA6, MatLZ1)
    mul!(MatLV, MatM6, MatZ1, 1, 1)
    axpy!(1, MatLZ2, MatLV)

    copy!(MatR, MatU)
    axpy!(1, MatV, MatR)
    ldiv!(A_sys.lu, MatR)

    # LZ1 = LU - LV, reuse LZ1
    copy!(MatLZ1, MatLU)
    axpy!(-1, MatLV, MatLZ1)

    mul!(MatΔ, MatLZ1, MatR)
    axpy!(1, MatLU, MatΔ)
    axpy!(1, MatLV, MatΔ)
    ldiv!(A_sys.lu, MatΔ)
end

function dexp_ScaleSquare!(R::Ref{Matrix{Float64}}, Δ::Ref{Matrix{Float64}}, A::Ref{Matrix{Float64}}, A_sys::dexp_ScaleSquare_system, E::Ref{Matrix{Float64}},
    wsp_dexp_NonNegative::WSP=get_wsp_dexp_nonnegative(size(R[], 1)))
    lm_13::Float64 = 4.74


    MatR = R[]
    MatΔ = Δ[]
    MatA = A[]
    MatE = E[]

    s::Int = ceil(log2(opnorm(MatA, 1) / lm_13))

    lmul!(2.0^(-s), MatA)
    lmul!(2.0^(-s), MatE)

    _dexp_ScaleSquare_m13_no_scale!(R, Δ, A, A_sys, E, wsp_dexp_NonNegative)

    MatTmp1 = wsp_dexp_NonNegative[1]
    MatTmp2 = wsp_dexp_NonNegative[2]

    for k = 1:s
        mul!(MatTmp1, MatR, MatΔ)
        mul!(MatTmp2, MatΔ, MatR)
        copy!(MatΔ, MatTmp1)
        axpy!(1, MatTmp2, MatΔ)

        mul!(MatTmp1, MatR, MatR)
        copy!(MatR, MatTmp1)
    end
end



# function _dlog_NonNegative_m13_no_scale(R::Ref{Matrix{Float64}}, Δ::Ref{Matrix{Float64}}, A::Ref{Matrix{Float64}}, A_sys::dexp_NonNegative_system, E::Ref{Matrix{Float64}},
#     wsp_dexp_NonNegative::WSP=get_wsp_dexp_nonnegative(size(R[], 1)))

# end

# function _dexp_NonNegative_backward_m13()

# end