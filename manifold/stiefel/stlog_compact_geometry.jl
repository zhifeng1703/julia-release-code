

include("../../inc/global_path.jl")
include(joinpath(JULIA_MANIFOLD_PATH, "spec_orth/so_explog.jl"))
include(joinpath(JULIA_MANIFOLD_PATH, "spec_orth/so_nearlog_newton.jl"))


@inline inner_skew!(S1::Ref{Matrix{Float64}}, S2::Ref{Matrix{Float64}}) = dot(S1[], S2[]) / 2.0;
@inline inner_skew!(S1::Ref{Vector{Float64}}, S2::Ref{Vector{Float64}}; lower=true) =
    if lower
        return dot(S1[], S2[])
    else
        return dot(S1[], S2[]) / 2.0
    end

@inline function stlog_cost(S::Ref{Matrix{Float64}}, k::Int)
    MatS = S[]
    n = size(MatS, 1)
    fval::Float64 = 0.0
    @inbounds for c_ind = (k+1):n
        @inbounds for r_ind = (c_ind+1):n
            fval += MatS[r_ind, c_ind] * MatS[r_ind, c_ind]
        end
    end
    return fval / 2.0
end

@inline function stlog_dcost(S::Ref{Matrix{Float64}}, M::Ref{Matrix{Float64}}, k::Int)
    MatS = S[]
    MatM = M[]
    n = size(MatS, 1)
    fval::Float64 = 0.0
    @inbounds for c_ind = (k+1):n
        @inbounds for r_ind = (c_ind+1):n
            fval += MatS[r_ind, c_ind] * MatM[r_ind, c_ind]
        end
    end
    return fval
end

@inline get_wsp_stlog_ret(n, k) = WSP(Matrix{Float64}(undef, n, n), Matrix{Float64}(undef, n, n), Matrix{Float64}(undef, n, n), Matrix{Float64}(undef, n - k, n - k), get_wsp_saf(n), get_wsp_nearlog(n))


function _stlog_ret_exp!(Q::Ref{Matrix{Float64}}, Z::Ref{Matrix{Float64}})
    MatQ = Q[]
    MatZ = Z[]

    copyto!(MatQ, MatZ)
    copyto!(MatQ, LinearAlgebra.exp!(MatQ))
end

_stlog_ret!(M::Ref{Matrix{Float64}}, M_saf::SAFactor, Up::Ref{Matrix{Float64}}, U::Ref{Matrix{Float64}}, Z::Ref{Matrix{Float64}}, M_sys::dexp_SkewSymm_system, n, k, nearby, wsp_stlog_ret::WSP) =
    nearby ? _stlog_ret_nearby!(M, M_saf, Up, U, Z, M_sys, n, k, wsp_stlog_ret) : _stlog_ret_principal!(M, M_saf, Up, U, Z, n, k, wsp_stlog_ret);


function _stlog_ret_principal!(M::Ref{Matrix{Float64}}, M_saf::SAFactor, Up::Ref{Matrix{Float64}}, Uk::Ref{Matrix{Float64}}, Z::Ref{Matrix{Float64}}, n, k, wsp_stlog_ret::WSP)
    MatUk = Uk[]
    MatUp = Up[]
    MatM = M[]

    MateZ = wsp_stlog_ret[4]
    wsp_saf_n = wsp_stlog_ret[5]

    eZ = wsp_stlog_ret(4)


    _stlog_ret_exp!(eZ, Z)

    copyto!(view(MatM, :, 1:k), MatUk)
    mul!(view(MatM, :, (k+1):n), MatUp, MateZ)
    copyto!(MatUp, view(MatM, :, (k+1):n))

    log_SpecOrth!(M, M_saf, M, wsp_saf_n; order=false, regular=false)


    # schurAngular_SpecOrth!(M_saf, M, wsp_saf_n, order=false, regular=false)
    # computeSkewSymm!(M, M_saf)
end

function _stlog_ret_nearby!(M::Ref{Matrix{Float64}}, M_saf::SAFactor, Up::Ref{Matrix{Float64}}, U::Ref{Matrix{Float64}}, Z::Ref{Matrix{Float64}}, M_sys::dexp_SkewSymm_system, n, k, wsp_stlog_ret::WSP)
    MatU = U[]
    MatUp = Up[]
    MatM = M[]

    MateM = wsp_stlog_ret[1]
    MateS = wsp_stlog_ret[2]
    MatS = wsp_stlog_ret[3]
    MateZ = wsp_stlog_ret[4]

    wsp_saf_n = wsp_stlog_ret[5]
    wsp_nearlog_n = wsp_stlog_ret[6]

    eM = wsp_stlog_ret(1)
    eS = wsp_stlog_ret(2)
    S = wsp_stlog_ret(3)
    eZ = wsp_stlog_ret(4)


    copyto!(MatS, MatM)
    copyto!(view(MateS, :, 1:k), MatU)
    copyto!(view(MateS, :, (k+1):n), MatUp)

    _stlog_ret_exp!(eZ, Z)

    copyto!(view(MateM, :, 1:k), MatU)
    mul!(view(MateM, :, (k+1):n), MatUp, MateZ)
    copyto!(MatUp, view(MatM, :, (k+1):n))

    nearlog_SpecOrth!(M, eM, S, eS, M_saf, M_sys, wsp_nearlog_n)
end

@inline get_wsp_stlog_UpZ_ret(n::Int, k::Int) = WSP(Matrix{Float64}(undef, n - k, n - k), Matrix{Float64}(undef, n, n - k), get_wsp_saf(n - k), get_wsp_saf(n))
@inline get_wsp_stlog_UpZ_ret(n::Int, k::Int, wsp_saf_m::WSP, wsp_saf_n::WSP) = WSP(Matrix{Float64}(undef, n - k, n - k), Matrix{Float64}(undef, n, n - k), wsp_saf_m, wsp_saf_n)

function ret_UpZ_builtin_exp!(U::Ref{Matrix{Float64}}, Up::Ref{Matrix{Float64}}, M::Ref{Matrix{Float64}}, M_saf::SAFactor, Z::Ref{Matrix{Float64}}, wsp_stlog_UpZ_ret::WSP; nearlog::Bool=false)
    # wsp_ret carrys real n-k x n-k matrix Mm, n x m matrix Mnm, workspaces wsp_exp and wsp_log
    MatU = U[]
    MatUp = Up[]
    MatTmpm = wsp_stlog_UpZ_ret[1]
    MatTmpnm = wsp_stlog_UpZ_ret[2]
    wsp_saf_m = wsp_stlog_UpZ_ret[3]
    wsp_saf_n = wsp_stlog_UpZ_ret[4]

    Tmpm = wsp_stlog_UpZ_ret(1)


    n::Int, m::Int = size(MatUp)
    k::Int = n - m

    # Q <- exp(Z), using scaling and squaring algorithm
    # Z needs to ensure skew symmetry to guarantee special orthogonality in this exp(Z).
    MatZ = Z[]
    for r_ind = 1:m
        for c_ind = 1:m
            @inbounds MatZ[r_ind, c_ind] = (MatZ[r_ind, c_ind] - MatZ[c_ind, r_ind]) / 2.0
            @inbounds MatZ[c_ind, r_ind] = -MatZ[r_ind, c_ind]
        end
    end
    MatTmpm .= exp(MatZ)
    copyto!(MatTmpm, exp(MatZ))
    # exp_SkewSymm!(Tmpm, Z_saf, Z, wsp_saf_m)                                   


    copyto!(MatTmpnm, MatUp)

    mul!(MatUp, MatTmpnm, MatTmpm)                                              # Up <- Up * Q

    copyto!(view(MatU, :, (k+1):n), MatUp)                                      # Write Up to U

    if nearlog
        nearlog_SpecOrth!(M, M_saf, U, M, wsp_saf_n; order=true, regular=true)        # Get the nearest log of U from M and overwrite it to M
    else
        log_SpecOrth!(M, M_saf, U, wsp_saf_n; order=false, regular=false)
        # schurAngular_SpecOrth!(M_saf, M, wsp_saf_n, order=false, regular=false)
        # computeSkewSymm!(M, M_saf)
    end
end

function ret_UpZ_builtin_exp2!(U::Ref{Matrix{Float64}}, Up::Ref{Matrix{Float64}}, M::Ref{Matrix{Float64}}, M_saf::SAFactor, Z::Ref{Matrix{Float64}}, wsp_stlog_UpZ_ret::WSP; nearlog::Bool=false)
    # wsp_ret carrys real n-k x n-k matrix Mm, n x m matrix Mnm, workspaces wsp_exp and wsp_log
    MatU = U[]
    MatUp = Up[]
    MatTmpm = wsp_stlog_UpZ_ret[1]
    MatTmpnm = wsp_stlog_UpZ_ret[2]
    wsp_saf_m = wsp_stlog_UpZ_ret[3]
    wsp_saf_n = wsp_stlog_UpZ_ret[4]

    Tmpm = wsp_stlog_UpZ_ret(1)


    n::Int, m::Int = size(MatUp)
    k::Int = n - m

    # Q <- exp(Z), using scaling and squaring algorithm
    # Z needs to ensure skew symmetry to guarantee special orthogonality in this exp(Z).
    MatZ = Z[]
    for r_ind = 1:m
        for c_ind = 1:m
            @inbounds MatZ[r_ind, c_ind] = (MatZ[r_ind, c_ind] - MatZ[c_ind, r_ind]) / 2.0
            @inbounds MatZ[c_ind, r_ind] = -MatZ[r_ind, c_ind]
        end
    end
    copyto!(MatTmpm, exp(MatZ))
    # exp_SkewSymm!(Tmpm, Z_saf, Z, wsp_saf_m)                                   

    _stlog_ret_exp!(Tmpm, Z)

    mul!(view(MatU, :, (k+1):n), MatUp, MatTmpm)
    copyto!(MatUp, view(MatU, :, (k+1):n))

    log_SpecOrth!(M, M_saf, U, wsp_saf_n; order=false, regular=false)
end

#######################################Test functions#######################################


using Printf, Statistics

function test_stlog_ret_speed(n, k; loops=1000)

    M_saf = SAFactor(n)
    M_sys = dexp_SkewSymm_system(n)

    wsp_stlog_ret = get_wsp_stlog_ret(n, k)
    wsp_saf_n = get_wsp_saf(n)

    Record = zeros(Int, 2, loops)

    for ind in 1:loops
        MatM = rand(n, n)
        MatM .-= MatM'
        MateM = exp(MatM)
        MatUk = copy(view(MateM, :, 1:k))
        MatUp = copy(view(MateM, :, (k+1):n))
        MatZ = rand(n - k, n - k)
        MatZ .-= MatZ'
        MatZ .*= 0.5 / opnorm(MatZ, 2)

        M = Ref(MatM)
        Up = Ref(MatUp)
        Uk = Ref(MatUk)
        Z = Ref(MatZ)

        schurAngular_SkewSymm!(M_saf, M, wsp_saf_n; order=true, regular=true)
        compute_dexp_SkewSymm_both_system!(M_sys, M_saf.angle)



        stats = @timed begin
            _stlog_ret_principal!(M, M_saf, Up, U, Z, n, k, wsp_stlog_ret)
        end
        Record[1, ind] = Int(round((stats.time - stats.gctime) * 1e9))

        stats = @timed begin
            _stlog_ret_nearby!(M, M_saf, Up, U, Z, M_sys, n, k, wsp_stlog_ret)
        end
        Record[2, ind] = Int(round((stats.time - stats.gctime) * 1e9))
    end

    @printf "+-----------------------+---------------+---------------+---------------+\n"
    @printf "|Methods\t\t|Min. time\t|Avg. time\t|Max. time\t|\n"
    @printf "+-----------------------+---------------+---------------+---------------+\n"
    @printf "|Principal logarithm\t|%i  \t|%.1f \t|%i \t|\n" minimum(Record[1, :]) mean(Record[1, :]) maximum(Record[1, :])
    @printf "|Nearby logarithm\t|%i  \t|%.1f \t|%i \t|\n" minimum(Record[2, :]) mean(Record[2, :]) maximum(Record[2, :])
    @printf "+-----------------------+---------------+---------------+---------------+\n\n"

end


