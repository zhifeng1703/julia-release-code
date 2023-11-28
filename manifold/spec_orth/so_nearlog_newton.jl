include("../../inc/global_path.jl")
include(joinpath(JULIA_INCLUDE_PATH, "workspace.jl"))

include("so_safactor.jl")
include("dexp_SkewSymm.jl")

@inline get_wsp_nearlog(n::Int) = WSP(Matrix{Float64}(undef, n, n), Matrix{Float64}(undef, n, n), Matrix{Float64}(undef, n, n), Matrix{Float64}(undef, n, n),
    SAFactor(n), dexp_SkewSymm_system(n), STRICT_LOWER_ITERATOR(n, lower_blk_traversal), get_wsp_cong(n), get_wsp_saf(n))

function nearlog_SpecOrth_newton(eX::Matrix{Float64}, S::Matrix{Float64}, eS::Matrix{Float64})
    # solve X from exp(X) = eX near exp(S) = eS.
    # eX_iter = exp(S);

    n = size(eX, 1)

    X_iter = copy(S)
    eX_iter = copy(eS)

    ΔQ_iter = eX_iter' * (eX .- eX_iter)
    ΔQ_iter .-= ΔQ_iter'
    ΔQ_iter .*= 0.5

    ΔX_iter = zeros(n, n)

    X_saf_iter = schurAngular_SkewSymm(Ref(X_iter); order=true, regular=true)
    X_sys_iter = dexp_SkewSymm_system(n)
    compute_dexp_SkewSymm_both_system!(X_sys_iter, X_saf_iter.angle)

    iter = 1
    blk_it_n = STRICT_LOWER_ITERATOR(n, lower_blk_traversal)
    wsp_cong_n = get_wsp_cong(n)
    wsp_saf = get_wsp_saf(n)

    while norm(ΔQ_iter) > 1e-8
        if iter > 100
            println("Near log by Newton's method failed.")
            return false, iter, X_iter
        end
        dexp_SkewSymm!(Ref(ΔX_iter), Ref(ΔQ_iter), X_sys_iter, X_saf_iter, blk_it_n, wsp_cong_n; inv=true, cong=true, compact=true)

        X_iter .+= ΔX_iter
        eX_iter .= exp(X_iter)

        schurAngular_SkewSymm!(X_saf_iter, Ref(X_iter), wsp_saf; order=true, regular=true)
        compute_dexp_SkewSymm_both_system!(X_sys_iter, X_saf_iter.angle)


        ΔQ_iter = eX_iter' * (eX .- eX_iter)
        ΔQ_iter .-= ΔQ_iter'
        ΔQ_iter .*= 0.5

        iter += 1
    end

    return true, iter, X_iter
end



function nearlog_SpecOrth!(X::Ref{Matrix{Float64}}, eX::Ref{Matrix{Float64}}, S::Ref{Matrix{Float64}}, eS::Ref{Matrix{Float64}}, wsp_nearlog::WSP=get_wsp_nearlog(size(X[], 1)))
    # solve X from exp(X) = eX near exp(S) = eS.
    # eX_iter = exp(S);


    MatX_iter = wsp_nearlog[1]
    MateX_iter = wsp_nearlog[2]
    MatΔQ_iter = wsp_nearlog[3]
    MatΔX_iter = wsp_nearlog[4]
    X_saf_iter = wsp_nearlog[5]
    X_sys_iter = wsp_nearlog[6]
    blk_it_n = wsp_nearlog[7]
    wsp_cong_n = wsp_nearlog[8]
    wsp_saf_n = wsp_nearlog[9]

    X_iter = wsp_nearlog(1)

    MatX = X[]
    MatS = S[]
    MateX = eX[]
    MateS = eS[]

    copy!(MatX_iter, MatS)
    copy!(MateX_iter, MateS)



    # fill!(MatΔX_iter, 0.0)
    schurAngular_SkewSymm!(X_saf_iter, X_iter, wsp_saf_n; order=true, regular=true)
    compute_dexp_SkewSymm_both_system!(X_sys_iter, X_saf_iter.angle)

    @inbounds @simd for ind in eachindex(MatΔQ_iter)
        MatΔX_iter[ind] = MateX[ind] - MateX_iter[ind]
    end
    mul!(MatΔQ_iter, MateX_iter', MatΔX_iter)
    getSkewSymm!(MatΔQ_iter)





    # X_iter = copy(S)
    # eX_iter = copy(eS)

    # ΔQ_iter = eX_iter' * (eX .- eX_iter)
    # ΔQ_iter .-= ΔQ_iter'
    # ΔQ_iter .*= 0.5

    # ΔX_iter = zeros(n, n)

    # X_saf_iter = schurAngular_SkewSymm(Ref(X_iter); order=true, regular=true)
    # X_sys_iter = dexp_SkewSymm_system(n)
    # compute_dexp_SkewSymm_both_system!(X_sys_iter, X_saf_iter.angle)

    iter = 1
    # blk_it_n = STRICT_LOWER_ITERATOR(n, lower_blk_traversal)
    # wsp_cong_n = get_wsp_cong(n)
    # wsp_saf = get_wsp_saf(n)

    while norm(ΔQ_iter) > 1e-14
        if iter > 100
            println("Near log by Newton's method failed.")
            fill!(MatX, 0.0)
            return false
        end

        dexp_SkewSymm!(ΔX_iter, ΔQ_iter, X_sys_iter, X_saf_iter, blk_it_n, wsp_cong_n; inv=true, cong=true)

        @inbounds @simd for ind in eachindex(MatX_iter)
            MatX_iter[ind] += MatΔX_iter[ind]
        end

        copy!(MateX_iter, MatX_iter)
        copy!(MateX_iter, LinearAlgebra.exp!(MateX_iter))

        # X_iter .+= ΔX_iter
        # eX_iter .= exp(X_iter)

        schurAngular_SkewSymm!(X_saf_iter, X_iter, wsp_saf; order=true, regular=true)
        compute_dexp_SkewSymm_both_system!(X_sys_iter, X_saf_iter.angle)


        @inbounds @simd for ind in eachindex(MatΔQ_iter)
            MatΔX_iter[ind] = MateX[ind] - MateX_iter[ind]
        end
        mul!(MatΔQ_iter, MateX_iter', MatΔX_iter)
        getSkewSymm!(MatΔQ_iter)

        # ΔQ_iter = eX_iter' * (eX .- eX_iter)
        # ΔQ_iter .-= ΔQ_iter'
        # ΔQ_iter .*= 0.5

        iter += 1
    end

    copy!(MatX, MatX_iter)

    return true
end

function nearlog_SpecOrth!(X::Ref{Matrix{Float64}}, eX::Ref{Matrix{Float64}}, S::Ref{Matrix{Float64}}, eS::Ref{Matrix{Float64}}, S_saf::SAFactor, S_sys::dexp_SkewSymm_system,
    wsp_nearlog::WSP=get_wsp_nearlog(size(X[], 1)))
    # solve X from exp(X) = eX near exp(S) = eS.
    # eX_iter = exp(S);


    MatX_iter = wsp_nearlog[1]
    MateX_iter = wsp_nearlog[2]
    MatΔQ_iter = wsp_nearlog[3]
    MatΔX_iter = wsp_nearlog[4]
    # X_saf_iter = wsp_nearlog[5]
    # X_sys_iter = wsp_nearlog[6]
    blk_it_n = wsp_nearlog[7]
    wsp_cong_n = wsp_nearlog[8]
    wsp_saf_n = wsp_nearlog[9]

    ΔQ_iter = wsp_nearlog(3)
    ΔX_iter = wsp_nearlog(4)





    X_iter = wsp_nearlog(1)

    MatX = X[]
    MatS = S[]
    MateX = eX[]
    MateS = eS[]

    copy!(MatX_iter, MatS)
    copy!(MateX_iter, MateS)

    X_saf_iter = S_saf
    X_sys_iter = S_sys


    # fill!(MatΔX_iter, 0.0)
    # schurAngular_SkewSymm!(X_saf_iter, X_iter, wsp_saf_n; order=true, regular=true)
    # compute_dexp_SkewSymm_both_system!(X_sys_iter, X_saf_iter.angle)

    @inbounds @simd for ind in eachindex(MatΔQ_iter)
        MatΔX_iter[ind] = MateX[ind] - MateX_iter[ind]
    end
    mul!(MatΔQ_iter, MateX_iter', MatΔX_iter)
    getSkewSymm!(ΔQ_iter)





    # X_iter = copy(S)
    # eX_iter = copy(eS)

    # ΔQ_iter = eX_iter' * (eX .- eX_iter)
    # ΔQ_iter .-= ΔQ_iter'
    # ΔQ_iter .*= 0.5

    # ΔX_iter = zeros(n, n)

    # X_saf_iter = schurAngular_SkewSymm(Ref(X_iter); order=true, regular=true)
    # X_sys_iter = dexp_SkewSymm_system(n)
    # compute_dexp_SkewSymm_both_system!(X_sys_iter, X_saf_iter.angle)

    iter = 1
    # blk_it_n = STRICT_LOWER_ITERATOR(n, lower_blk_traversal)
    # wsp_cong_n = get_wsp_cong(n)
    # wsp_saf = get_wsp_saf(n)

    while norm(ΔQ_iter) > 1e-14
        if iter > 100
            println("Near log by Newton's method failed.")
            fill!(MatX, 0.0)
            return false
        end

        dexp_SkewSymm!(ΔX_iter, ΔQ_iter, X_sys_iter, X_saf_iter, blk_it_n, wsp_cong_n; inv=true, cong=true)

        @inbounds @simd for ind in eachindex(MatX_iter)
            MatX_iter[ind] += MatΔX_iter[ind]
        end

        copy!(MateX_iter, MatX_iter)
        copy!(MateX_iter, LinearAlgebra.exp!(MateX_iter))

        # X_iter .+= ΔX_iter
        # eX_iter .= exp(X_iter)

        schurAngular_SkewSymm!(X_saf_iter, X_iter, wsp_saf_n; order=true, regular=true)
        compute_dexp_SkewSymm_both_system!(X_sys_iter, X_saf_iter.angle)


        @inbounds @simd for ind in eachindex(MatΔQ_iter)
            MatΔX_iter[ind] = MateX[ind] - MateX_iter[ind]
        end
        mul!(MatΔQ_iter, MateX_iter', MatΔX_iter)
        getSkewSymm!(ΔQ_iter)

        # ΔQ_iter = eX_iter' * (eX .- eX_iter)
        # ΔQ_iter .-= ΔQ_iter'
        # ΔQ_iter .*= 0.5

        iter += 1
    end

    copy!(MatX, MatX_iter)

    return true
end

#######################################Test functions#######################################


function test_nearlog_SpecOrth_newton(n, s, e)
    S = rand(n, n)
    S .-= S'
    S .*= s / opnorm(S)

    ΔS = rand(n, n)
    ΔS .-= ΔS'
    ΔS .*= e / opnorm(ΔS)

    eX = exp(S .+ ΔS)
    eS = exp(S)

    flag, iter, X = nearlog_SpecOrth_newton(eX, S, eS)
    println("Number of iterations used: $(iter)")
    display(X .- (S .+ ΔS))
end