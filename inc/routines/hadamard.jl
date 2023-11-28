include("../global_path.jl")

include(joinpath(JULIA_INCLUDE_PATH, "workspace.jl"))

include(joinpath(JULIA_MANIFOLD_PATH, "spec_orth/iterator_SkewSymm.jl"))

using LoopVectorization


get_wsp_hada_complex(n, type) = WSP(Vector{real(type)}(undef, n), Vector{real(type)}(undef, n));

get_wsp_hada_allreal(n, type) = WSP(Vector{real(type)}(undef, n), Vector{real(type)}(undef, n), Vector{real(type)}(undef, n), Vector{real(type)}(undef, n), Vector{real(type)}(undef, n), Vector{real(type)}(undef, n));

get_wsp_hada(n, type) = get_wsp_hada_allreal(n, type);

function hadamard!(C, A, B; inv=false, Hadamard_Mode='V', wsp=nothing)
    if Hadamard_Mode == 'V'
        hada_vecloop!(C, A, B; inv=inv)
    elseif Hadamard_Mode == 'R'
        hada_rawloop!(C, A, B; inv=inv)
    elseif Hadamard_Mode == 'C'
        if isnothing(wsp)
            hada_vecloop_complex!(C, A, B; inv=inv)
        else
            hada_vecloop_complex!(C, A, B, wsp; inv=inv)
        end
    elseif Hadamard_Mode == 'A'
        if isnothing(wsp)
            hada_vecloop_allreal!(C, A, B; inv=inv)
        else
            hada_vecloop_allreal!(C, A, B, wsp; inv=inv)
        end
    else
        throw(1)
    end
end

function hadamard!(C::Ref{Matrix{T}}, A::Ref{Matrix{T}}, B::Ref{Matrix{T}}, iter::STRICT_LOWER_ITERATOR; inv=false) where {T}
    MatC = C[]
    MatA = A[]
    MatB = B[]

    n = size(MatC, 1)

    LowerInd = iter.vec2lower[]
    UpperInd = iter.vec2lower[]

    if inv
        @inbounds for ind in LowerInd
            MatC[ind] = MatA[ind] / MatB[ind]
        end

        @inbounds for ind in UpperInd
            MatC[ind] = MatA[ind] / MatB[ind]
        end

        @inbounds for ind in 1:n
            MatC[ind, ind] = MatA[ind, ind] / MatB[ind, ind]
        end
    else

        @inbounds for ind in LowerInd
            MatC[ind] = MatA[ind] * MatB[ind]
        end

        @inbounds for ind in UpperInd
            MatC[ind] = MatA[ind] * MatB[ind]
        end

        @inbounds for ind in 1:n
            MatC[ind, ind] = MatA[ind, ind] * MatB[ind, ind]
        end
    end
end

hadamard!(A, B; inv=false, Hadamard_Mode='V', wsp=nothing) = hadamard!(A, A, B; inv=inv, Hadamard_Mode=Hadamard_Mode, wsp=wsp)

function hada_rawloop!(C::Ref{Matrix{T}}, A::Ref{Matrix{T}}, B::Ref{Matrix{T}}; inv::Bool=false) where {T}
    MatC = C[]
    MatA = A[]
    MatB = B[]
    if inv
        @inbounds @simd for ind in eachindex(MatC)
            MatC[ind] = MatA[ind] / MatB[ind]
        end
    else
        @inbounds @simd for ind in eachindex(MatC)
            MatC[ind] = MatA[ind] * MatB[ind]
        end
    end
end

function hada_rawloop!(C::Ref{Vector{T}}, A::Ref{Vector{T}}, B::Ref{Vector{T}}; inv::Bool=false) where {T}
    MatC = C[]
    MatA = A[]
    MatB = B[]
    if inv
        @inbounds @simd for ind in eachindex(MatC)
            MatC[ind] = MatA[ind] / MatB[ind]
        end
    else
        @inbounds @simd for ind in eachindex(MatC)
            MatC[ind] = MatA[ind] * MatB[ind]
        end
    end
end

function hada_vecloop!(C::Ref{Matrix{T}}, A::Ref{Matrix{T}}, B::Ref{Matrix{T}}; inv::Bool=false) where {T}
    MatC = C[]
    MatA = A[]
    MatB = B[]
    if inv
        @turbo warn_check_args = false for ind in eachindex(MatC)
            MatC[ind] = MatA[ind] / MatB[ind]
        end
    else

        @turbo warn_check_args = false for ind in eachindex(MatC)
            MatC[ind] = MatA[ind] * MatB[ind]
        end
    end
end

function hada_vecloop!(C::Ref{Vector{T}}, A::Ref{Vector{T}}, B::Ref{Vector{T}}; inv::Bool=false) where {T}
    MatC = C[]
    MatA = A[]
    MatB = B[]
    if inv
        @turbo warn_check_args = false for ind in eachindex(MatC)
            MatC[ind] = MatA[ind] / MatB[ind]
        end
    else
        @turbo warn_check_args = false for ind in eachindex(MatC)
            MatC[ind] = MatA[ind] * MatB[ind]
        end
    end
end





function split_real_imag(AR, AI, A)
    MatA = A[]
    MatAR = AR[]
    MatAI = AI[]

    @inbounds for ind in eachindex(MatA)
        @inbounds MatAR[ind] = MatA[ind].re
        @inbounds MatAI[ind] = MatA[ind].im
    end
end

merge_real_imag(A, AR, AI) = merge_real_imag_loop(A, AR, AI);

function merge_real_imag_vmap(A, AR, AI)
    MatA = A[]
    MatAR = AR[]
    MatAI = AI[]

    vmap!(complex, MatA, MatAR, MatAI)
end

function merge_real_imag_loop(A, AR, AI)
    MatA = A[]
    MatAR = AR[]
    MatAI = AI[]

    @inbounds for ind in eachindex(MatA)
        @inbounds MatA[ind] = complex(MatAR[ind], MatAI[ind])
    end
end

function hada_vecloop_complex!(C::Ref{Matrix{T}}, A::Ref{Matrix{T}}, B::Ref{Matrix{T}}, wsp_hada_complex=get_wsp_hada_complex(sizeof(C[]), eltype(C[]))) where {T}
    MatC = C[]
    MatA = A[]
    MatB = B[]
    MatCR = wsp_hada_complex[1]
    MatCI = wsp_hada_complex[2]

    ar = zero(real(T))
    ai = zero(real(T))
    br = zero(real(T))
    bi = zero(real(T))

    @turbo for ind in eachindex(MatC)
        @inbounds ar = MatA[ind].re
        @inbounds ai = MatA[ind].im
        @inbounds br = MatB[ind].re
        @inbounds bi = MatB[ind].im

        @inbounds MatCR[ind] = ar * br - ai * bi
        @inbounds MatCI[ind] = ar * bi + ar * bi
    end
    merge_real_imag(C, wsp_hada_complex(1), wsp_hada_complex(2))
end


function hada_vecloop_complex!(C::Ref{Vector{T}}, A::Ref{Vector{T}}, B::Ref{Vector{T}}, wsp_hada_complex=get_wsp_hada_complex(sizeof(C[]), eltype(C[]))) where {T}
    MatC = C[]
    MatA = A[]
    MatB = B[]
    MatCR = wsp_hada_complex[1]
    MatCI = wsp_hada_complex[2]

    ar = zero(real(T))
    ai = zero(real(T))
    br = zero(real(T))
    bi = zero(real(T))

    a = zero(T)
    b = zero(T)


    @turbo for ind in eachindex(MatC)
        @inbounds a = MatA[ind]
        @inbounds b = MatB[ind]

        @inbounds MatCR[ind] = a.re * b.re - a.im * b.im
        @inbounds MatCI[ind] = a.re * b.im + a.re * b.im


        # @inbounds ar = (MatA[ind]).re
        # @inbounds ai = (MatA[ind]).im
        # @inbounds br = (MatB[ind]).re
        # @inbounds bi = (MatB[ind]).im

        # @inbounds ar = real(MatA[ind])
        # @inbounds ai = imag(MatA[ind])
        # @inbounds br = real(MatB[ind])
        # @inbounds bi = imag(MatB[ind])

        # @inbounds MatCR[ind] = ar * br - ai * bi
        # @inbounds MatCI[ind] = ar * bi + ar * bi
    end
    merge_real_imag(C, wsp_hada_complex(1), wsp_hada_complex(2))
end


function hada_vecloop_allreal!(C::Ref{Matrix{T}}, A::Ref{Matrix{T}}, B::Ref{Matrix{T}}, wsp_hada_allreal=get_wsp_hada_allreal(sizeof(C[]), eltype(C[]))) where {T}
    MatC = C[]

    MatAR = wsp_hada_allreal[1]
    MatAI = wsp_hada_allreal[2]
    MatBR = wsp_hada_allreal[3]
    MatBI = wsp_hada_allreal[4]
    MatCR = wsp_hada_allreal[5]
    MatCI = wsp_hada_allreal[6]

    split_real_imag(wsp_hada_allreal(1), wsp_hada_allreal(2), A)
    split_real_imag(wsp_hada_allreal(3), wsp_hada_allreal(4), B)


    @turbo for ind in eachindex(MatC)
        @inbounds ar = MatAR[ind]
        @inbounds ai = MatAI[ind]
        @inbounds br = MatBR[ind]
        @inbounds bi = MatBI[ind]

        @inbounds MatCR[ind] = ar * br - ai * bi
        @inbounds MatCI[ind] = ar * bi + ar * bi
    end
    merge_real_imag(C, wsp_hada_allreal(5), wsp_hada_allreal(6))
end

function hada_vecloop_allreal!(C::Ref{Vector{T}}, A::Ref{Vector{T}}, B::Ref{Vector{T}}, wsp_hada_allreal=get_wsp_hada_allreal(sizeof(C[]), eltype(C[]))) where {T}
    MatC = C[]

    MatAR = wsp_hada_allreal[1]
    MatAI = wsp_hada_allreal[2]
    MatBR = wsp_hada_allreal[3]
    MatBI = wsp_hada_allreal[4]
    MatCR = wsp_hada_allreal[5]
    MatCI = wsp_hada_allreal[6]

    split_real_imag(wsp_hada_allreal(1), wsp_hada_allreal(2), A)
    split_real_imag(wsp_hada_allreal(3), wsp_hada_allreal(4), B)


    @turbo for ind in eachindex(MatC)
        @inbounds ar = MatAR[ind]
        @inbounds ai = MatAI[ind]
        @inbounds br = MatBR[ind]
        @inbounds bi = MatBI[ind]

        @inbounds MatCR[ind] = ar * br - ai * bi
        @inbounds MatCI[ind] = ar * bi + ar * bi
    end
    merge_real_imag(C, wsp_hada_allreal(5), wsp_hada_allreal(6))
end