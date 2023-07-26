# Plots, MKL, LoopVectorization, IterativeSolvers, BenchmarkTools

include("../../inc/global_path.jl")

include(joinpath(JULIA_INCLUDE_PATH, "algorithm_port.jl"))
include(joinpath(JULIA_MANIFOLD_PATH, "spec_orth/so_nearlog_newton.jl"))

include("grhor_init_guess.jl")
include("grhor_descent.jl")


include("stlog_solver.jl")

BCH_MAX_ITER = 8
BCH_ABSTOL = 1e-3
BCH_SHUTDOWN = -max(20, BCH_MAX_ITER)

NMLS_SET = NMLS_Paras(0.1, 20.0, 0.9, 0.3, 0)

SOLVER_STOP = terminator(500, 5, 1e-8, 1e-4)

NEARLOG_THRESHOLD = π
RESTART_THRESHOLD = 2.0
DIRECTION_THRESHOLD = 2.0


ENABLE_NEARLOG = true
ENABLE_RESTART_BCH = true


ENABLE_NEARLOG = false
ENABLE_RESTART_BCH = false

LINESEARCH_CHECK = x -> (x != 3)



function grhor_newton_core(V, Vp=nothing; Stop=terminator(100, 10000, 1e-7, 1e-6), RandEng=nothing)

    # Fundamental equation :: exp(S_{A, B, C}) = | Uk | Up exp(C) | = | V * EX | Up * EC | = | W | Wp |;

    n, k = size(V)
    m = n - k

    if isnothing(Vp)
        MatS, MatQ = grhor_init_guess_random(V; RandEng=RandEng)
    else
        MatQ = hcat(V, Vp)
        MatS = log(MatQ)
        MatS .-= MatS'
        MatS .*= 0.5
    end
    # MatS, MatQ = grhor_init_guess_grassmann(V);

    MatA = copy(MatS[1:k, 1:k])
    MatB = copy(MatS[(k+1):n, 1:k])
    MatC = copy(MatS[(k+1):n, (k+1):n])

    MatUk = copy(MatQ[:, 1:k])
    MatUp = copy(MatQ[:, (k+1):n]) * exp(MatC)'

    MatX = log(V' * MatUk)
    MatX .-= MatX'
    MatX .*= 0.5
    MatEX = exp(MatX)

    MatΔS = zeros(n, n)
    MatΔX = zeros(k, k)

    MatΔV = zeros(k, k)
    MatΔVp = zeros(m, m)
    MatΔQ = zeros(n, n)

    MatS_new = similar(MatS)
    MatX_new = similar(MatX)
    MatQ_new = similar(MatQ)
    MatEX_new = similar(MatEX)


    S = Ref(MatS)
    X = Ref(MatX)
    S_new = Ref(MatS_new)
    X_new = Ref(MatX_new)
    Q_new = Ref(MatQ_new)
    EX_new = Ref(MatEX_new)

    Uk = Ref(MatUk)
    Up = Ref(MatUp)
    ΔS = Ref(MatΔS)
    ΔX = Ref(MatΔX)
    ΔV = Ref(MatΔV)
    ΔVp = Ref(MatΔVp)
    ΔQ = Ref(MatΔQ)


    blk_it_n = STRICT_LOWER_ITERATOR(n, lower_blk_traversal)
    blk_it_k = STRICT_LOWER_ITERATOR(k, lower_blk_traversal)
    col_it_k = STRICT_LOWER_ITERATOR(k, lower_col_traversal)

    S_saf = SAFactor(n)
    S_sys = dexp_SkewSymm_system(n)

    X_sys = dexp_SkewSymm_system(k)
    X_saf = SAFactor(k)




    MaxIter = Stop.MaxIter
    AbsTol = Stop.AbsTol

    time_record::Vector{Float64} = zeros(MaxIter)
    cost_record::Vector{Float64} = Vector{Float64}(undef, MaxIter)
    mats_record::Vector{Matrix{Float64}} = Vector{Matrix{Float64}}(undef, MaxIter)
    matx_record::Vector{Matrix{Float64}} = Vector{Matrix{Float64}}(undef, MaxIter)
    matΔuk_record::Vector{Matrix{Float64}} = Vector{Matrix{Float64}}(undef, MaxIter)
    matΔup_record::Vector{Matrix{Float64}} = Vector{Matrix{Float64}}(undef, MaxIter)

    vect_record = nothing
    step_record = nothing

    iter::Int = 1

    fval_nm = norm(MatX .- MatA)
    fval_sq = fval_nm^2 / 2.0

    cost_record[iter] = fval_nm
    mats_record[iter] = copy(MatS)
    matx_record[iter] = copy(MatX)
    matΔuk_record[iter] = zeros(k, k)
    matΔup_record[iter] = zeros(m, m)



    stepsize_shrink = 0.5

    stepsize = 1.0
    minstep = 1e-3
    descent = 100.0
    γ = 0.0
    slope = -2.0 * fval_sq


    wsp_grhor_sys = get_wsp_grhor_sys(n, k)
    wsp_saf_n = get_wsp_saf(n)
    wsp_saf_k = get_wsp_saf(k)
    wsp_cong_n = get_wsp_cong(n)
    wsp_cong_k = get_wsp_cong(k)


    result_flag = 0


    while result_flag == 0
        stats = @timed begin

            schurAngular_SkewSymm!(S_saf, S, wsp_saf_n; regular=true)
            compute_dexp_SkewSymm_both_system!(S_sys, S_saf.angle)

            schurAngular_SkewSymm!(X_saf, X, wsp_saf_k; regular=true)
            compute_dexp_SkewSymm_both_system!(X_sys, X_saf.angle)

            grhor_action = _GRHOR_ACTION(Ref(similar(MatX)), Ref(similar(MatΔS)), Ref(similar(MatΔX)), Ref(similar(MatΔV)), Ref(similar(MatΔVp)),
                S_sys, S_saf, X_sys, X_saf, blk_it_n, blk_it_k, col_it_k, wsp_grhor_sys, grhor_dim(n, k), div(k * (k - 1), 2))

            grhor_gmres_newton_descent_itsol(ΔV, ΔVp, S, X, grhor_action, size(grhor_action, 2))

            MatΔQ[1:k, 1:k] .= MatΔV
            MatΔQ[(k+1):n, (k+1):n] .= MatΔVp
            dexp_SkewSymm!(ΔS, ΔQ, S_sys, S_saf, blk_it_n, wsp_cong_n; inv=true, cong=true, compact=true)
            dexp_SkewSymm!(ΔX, ΔV, X_sys, X_saf, blk_it_k, wsp_cong_k; inv=true, cong=true, compact=true)



            if DEBUG
                check_grhor_descent(S, X, ΔV, ΔVp, 1e-5)
            end

            stepsize = min(1.0, 0.5 / opnorm(MatΔV))
            # stepsize = 1.0
            MatQ_new[:, 1:k] .= MatQ[:, 1:k] * exp(stepsize .* MatΔV)
            MatQ_new[:, (k+1):n] .= MatQ[:, (k+1):n] * exp(stepsize .* MatΔVp)
            MatEX_new .= MatEX * exp(stepsize .* MatΔV)

            # nearlog_SpecOrth!(S_new, S_saf, Q_new, S, wsp_saf_n; order=true, regular=true)
            # nearlog_SpecOrth!(X_new, X_saf, EX_new, X, wsp_saf_k; order=true, regular=true)
            # log_SpecOrth!(X_new, X_saf, EX_new, wsp_saf_k; order=true, regular=true)

            logS_flag, logS_iter, tempS = nearlog_SpecOrth_newton(MatQ_new, (MatS .+ stepsize .* MatΔS), exp(MatS .+ stepsize .* MatΔS))
            logX_flag, logX_iter, tempX = nearlog_SpecOrth_newton(MatEX_new, (MatX .+ stepsize .* MatΔX), exp(MatX .+ stepsize .* MatΔX))
            if logS_flag
                MatS_new .= tempS
            else
                # throw(1)
                println("Warning! Near-by log on S_{k+1} = log_{S_k}(Q_{k+1}) by Newton's method failed. Retreat to near-by log in the extended manifold.")
                nearlog_SpecOrth!(S_new, S_saf, Q_new, S, wsp_saf_n; order=true, regular=true)
            end
            if logX_flag
                MatX_new .= tempX
            else
                # throw(1)
                println("Warning! Near-by log on X_{k+1} = log_{X_k}(eX_{k+1}) by Newton's method failed. Retreat to near-by log in the extended manifold.")
                nearlog_SpecOrth!(X_new, X_saf, EX_new, X, wsp_saf_k; order=true, regular=true)
            end

            fval_nm_new = norm(MatX_new .- MatS_new[1:k, 1:k])
            fval_sq_new = fval_nm_new^2 / 2.0
            descent = fval_sq_new - fval_sq

            if descent > max(γ * slope * stepsize, -0.9 * fval_sq)
                d_msgln("\n==============Armijo Line Search==============\n")
                d_msgln("Current objective value:\t$(fval_sq)")

            end

            while descent > max(γ * slope * stepsize, -0.9 * fval_sq)

                stepsize = stepsize_shrink * stepsize

                MatQ_new[:, 1:k] .= MatQ[:, 1:k] * exp(stepsize .* MatΔV)
                MatQ_new[:, (k+1):n] .= MatQ[:, (k+1):n] * exp(stepsize .* MatΔVp)
                MatEX_new .= MatEX * exp(stepsize .* MatΔV)


                # nearlog_SpecOrth!(S_new, S_saf, Q_new, S, wsp_saf_n; order=true, regular=true)
                # nearlog_SpecOrth!(X_new, X_saf, EX_new, X, wsp_saf_k; order=true, regular=true)
                # log_SpecOrth!(X_new, X_saf, EX_new, wsp_saf_k; order=true, regular=true)

                logS_flag, logS_iter, tempS = nearlog_SpecOrth_newton(MatQ_new, (MatS .+ stepsize .* MatΔS), exp(MatS .+ stepsize .* MatΔS))
                logX_flag, logX_iter, tempX = nearlog_SpecOrth_newton(MatEX_new, (MatX .+ stepsize .* MatΔX), exp(MatX .+ stepsize .* MatΔX))
                if logS_flag
                    MatS_new .= tempS
                else
                    # throw(1)
                    println("Warning! Near-by log on S_{k+1} = log_{S_k}(Q_{k+1}) by Newton's method failed. Retreat to near-by log in the extended manifold.")
                    nearlog_SpecOrth!(S_new, S_saf, Q_new, S, wsp_saf_n; order=true, regular=true)
                end
                if logX_flag
                    MatX_new .= tempX
                else
                    # throw(1)
                    println("Warning! Near-by log on X_{k+1} = log_{X_k}(eX_{k+1}) by Newton's method failed. Retreat to near-by log in the extended manifold.")
                    nearlog_SpecOrth!(X_new, X_saf, EX_new, X, wsp_saf_k; order=true, regular=true)
                end

                fval_nm_new = norm(MatX_new .- MatS_new[1:k, 1:k])
                fval_sq_new = fval_nm_new^2 / 2.0
                descent = fval_sq_new - fval_sq

                d_msgln("Stepsize:\t $(stepsize)\tDescent:\t$(descent)\tRequirement:\t$(max(γ * slope * stepsize, -0.9 * fval_sq))")


                if stepsize < minstep
                    msgln("Fail to perform Armijo line search, using minimal step.")
                    break
                end
            end

            fval_nm = fval_nm_new
            fval_sq = fval_sq_new

            MatQ .= MatQ_new
            MatS .= MatS_new
            MatX .= MatX_new
            MatEX .= MatEX_new

            MatA .= MatS[1:k, 1:k]
            MatB .= MatS[(k+1):n, 1:k]
            MatC .= MatS[(k+1):n, (k+1):n]

            MatUk .= MatQ[:, 1:k]
            MatUp .= MatQ[:, (k+1):n] * exp(MatC)'

        end

        iter += 1
        cost_record[iter] = fval_nm
        time_record[iter] = time_record[iter-1] + 1000 * (stats.time - stats.gctime)
        mats_record[iter] = copy(MatS)
        matx_record[iter] = copy(MatX)
        matΔuk_record[iter] = copy(MatΔV)
        matΔup_record[iter] = copy(MatΔVp)

        result_flag = check_termination_vec(cost_record, nothing, vect_record, nothing, step_record, iter, Stop)
        # result_flag = check_termination_vec(cost_record, nothing, vect_record, time_record, step_record, iter, Stop)
    end

    return (iter, mats_record[iter], matx_record[iter]), (cost_record[1:iter], time_record[1:iter], mats_record[1:iter], matx_record[1:iter], matΔuk_record[1:iter], matΔup_record[1:iter])
end


function grhor_newton_full_core(V, Vp=nothing; Stop=terminator(100, 10000, 1e-7, 1e-6), RandEng=nothing)

    # Fundamental equation :: exp(S_{A, B, C}) = | Uk | Up exp(C) | = | V * EX | Up * EC | = | W | Wp |;

    n, k = size(V)
    m = n - k

    if isnothing(Vp)
        MatS, MatQ = grhor_init_guess_random(V; RandEng=RandEng)
    else
        MatQ = hcat(V, Vp)
        MatS = log(MatQ)
        MatS .-= MatS'
        MatS .*= 0.5
    end
    # MatS, MatQ = grhor_init_guess_grassmann(V);

    MatA = copy(MatS[1:k, 1:k])
    MatB = copy(MatS[(k+1):n, 1:k])
    MatC = copy(MatS[(k+1):n, (k+1):n])

    MatUk = copy(MatQ[:, 1:k])
    MatUp = copy(MatQ[:, (k+1):n])

    MatVk = copy(V)
    MatVp = copy(MatUp)

    MatX = log(MatVk' * MatUk)
    MatX .-= MatX'
    MatX .*= 0.5
    MatEX = exp(MatX)

    MatZ = zeros(m, m)
    MatEZ = exp(MatZ)

    MatΔS = zeros(n, n)
    MatΔX = zeros(k, k)
    MatΔZ = zeros(m, m)

    MatΔQ = zeros(n, n)
    MatΔUk = zeros(k, k)
    MatΔUp = zeros(m, m)

    MatS_new = similar(MatS)
    MatX_new = similar(MatX)
    MatZ_new = similar(MatZ)
    MatQ_new = similar(MatQ)
    MatEX_new = similar(MatEX)
    MatEZ_new = similar(MatEZ)


    S = Ref(MatS)
    X = Ref(MatX)
    Z = Ref(MatZ)
    S_new = Ref(MatS_new)
    X_new = Ref(MatX_new)
    Z_new = Ref(MatZ_new)
    Q_new = Ref(MatQ_new)
    EX_new = Ref(MatEX_new)
    EZ_new = Ref(MatEZ_new)

    Uk = Ref(MatUk)
    Up = Ref(MatUp)
    ΔS = Ref(MatΔS)
    ΔX = Ref(MatΔX)
    ΔZ = Ref(MatΔZ)
    ΔQ = Ref(MatΔQ)
    ΔUk = Ref(MatΔUk)
    ΔUp = Ref(MatΔUp)


    blk_it_n = STRICT_LOWER_ITERATOR(n, lower_blk_traversal)
    blk_it_k = STRICT_LOWER_ITERATOR(k, lower_blk_traversal)
    blk_it_m = STRICT_LOWER_ITERATOR(m, lower_blk_traversal)

    S_saf = SAFactor(n)
    S_sys = dexp_SkewSymm_system(n)

    X_sys = dexp_SkewSymm_system(k)
    X_saf = SAFactor(k)

    Z_sys = dexp_SkewSymm_system(m)
    Z_saf = SAFactor(m)




    MaxIter = Stop.MaxIter
    AbsTol = Stop.AbsTol

    time_record::Vector{Float64} = zeros(MaxIter)
    cost_record::Vector{Float64} = Vector{Float64}(undef, MaxIter)
    mats_record::Vector{Matrix{Float64}} = Vector{Matrix{Float64}}(undef, MaxIter)
    matx_record::Vector{Matrix{Float64}} = Vector{Matrix{Float64}}(undef, MaxIter)
    matz_record::Vector{Matrix{Float64}} = Vector{Matrix{Float64}}(undef, MaxIter)
    matΔuk_record::Vector{Matrix{Float64}} = Vector{Matrix{Float64}}(undef, MaxIter)
    matΔup_record::Vector{Matrix{Float64}} = Vector{Matrix{Float64}}(undef, MaxIter)

    vect_record = nothing
    step_record = nothing

    iter::Int = 1

    fval_sq = (sum((MatX .- MatA) .^ 2) + sum((MatZ .- MatC) .^ 2)) / 2.0
    fval_nm = sqrt(2.0 * fval_sq)

    cost_record[iter] = fval_nm
    mats_record[iter] = copy(MatS)
    matx_record[iter] = copy(MatX)
    matz_record[iter] = copy(MatZ)
    matΔuk_record[iter] = zeros(k, k)
    matΔup_record[iter] = zeros(m, m)



    stepsize_shrink = 0.5

    stepsize = 1.0
    minstep = 1e-3
    descent = 100.0
    γ = 0.0
    slope = -2.0 * fval_sq


    wsp_grhor_sys = get_wsp_grhor_sys(n, k)
    wsp_saf_n = get_wsp_saf(n)
    wsp_saf_k = get_wsp_saf(k)
    wsp_saf_m = get_wsp_saf(m)

    wsp_cong_n = get_wsp_cong(n)
    wsp_cong_k = get_wsp_cong(k)
    wsp_cong_m = get_wsp_cong(m)



    result_flag = 0


    while result_flag == 0
        stats = @timed begin

            schurAngular_SkewSymm!(S_saf, S, wsp_saf_n; regular=true)
            compute_dexp_SkewSymm_both_system!(S_sys, S_saf.angle)

            schurAngular_SkewSymm!(X_saf, X, wsp_saf_k; regular=true)
            compute_dexp_SkewSymm_both_system!(X_sys, X_saf.angle)

            schurAngular_SkewSymm!(Z_saf, Z, wsp_saf_m; regular=true)
            compute_dexp_SkewSymm_both_system!(Z_sys, Z_saf.angle)

            grhor_full_action = _GRHOR_FULL_ACTION(Ref(similar(MatΔS)), Ref(similar(MatΔX)), Ref(similar(MatΔZ)), Ref(similar(MatΔUk)), Ref(similar(MatΔUp)),
                S_sys, S_saf, X_sys, X_saf, Z_sys, Z_saf, blk_it_n, blk_it_k, blk_it_m, wsp_grhor_sys, grhor_dim(n, k))

            grhor_gmres_newton_descent_full_itsol(ΔUk, ΔUp, S, X, Z, grhor_full_action, size(grhor_full_action, 2))

            if DEBUG
                check_grhor_descent(S, X, Z, ΔUk, ΔUp, 1e-5)
            end

            MatΔQ[1:k, 1:k] .= MatΔUk
            MatΔQ[(k+1):n, (k+1):n] .= MatΔUp
            dexp_SkewSymm!(ΔS, ΔQ, S_sys, S_saf, blk_it_n, wsp_cong_n; inv=true, cong=true, compact=true)
            dexp_SkewSymm!(ΔX, ΔUk, X_sys, X_saf, blk_it_k, wsp_cong_k; inv=true, cong=true, compact=true)
            dexp_SkewSymm!(ΔZ, ΔUp, Z_sys, Z_saf, blk_it_m, wsp_cong_m; inv=true, cong=true, compact=true)


            stepsize = min(1.0, 0.5 / opnorm(MatΔUk))
            # stepsize = 1.0
            MatQ_new[:, 1:k] .= MatQ[:, 1:k] * exp(stepsize .* MatΔUk)
            MatQ_new[:, (k+1):n] .= MatQ[:, (k+1):n] * exp(stepsize .* MatΔUp)
            MatEX_new .= MatEX * exp(stepsize .* MatΔUk)
            MatEZ_new .= MatEZ * exp(stepsize .* MatΔUp)


            # nearlog_SpecOrth!(S_new, S_saf, Q_new, S, wsp_saf_n; order=true, regular=true)
            # nearlog_SpecOrth!(X_new, X_saf, EX_new, X, wsp_saf_k; order=true, regular=true)
            # nearlog_SpecOrth!(Z_new, Z_saf, EZ_new, Z, wsp_saf_m; order=true, regular=true)
            logS_flag, logS_iter, tempS = nearlog_SpecOrth_newton(MatQ_new, (MatS .+ stepsize .* MatΔS), exp(MatS .+ stepsize .* MatΔS))
            logX_flag, logX_iter, tempX = nearlog_SpecOrth_newton(MatEX_new, (MatX .+ stepsize .* MatΔX), exp(MatX .+ stepsize .* MatΔX))
            logZ_flag, logZ_iter, tempZ = nearlog_SpecOrth_newton(MatEZ_new, (MatZ .+ stepsize .* MatΔZ), exp(MatZ .+ stepsize .* MatΔZ))

            if logS_flag
                MatS_new .= tempS
            else
                # throw(1)
                println("Warning! Near-by log on S_{k+1} = log_{S_k}(Q_{k+1}) by Newton's method failed. Retreat to near-by log in the extended manifold.")
                nearlog_SpecOrth!(S_new, S_saf, Q_new, S, wsp_saf_n; order=true, regular=true)
            end
            if logX_flag
                MatX_new .= tempX
            else
                # throw(1)
                println("Warning! Near-by log on X_{k+1} = log_{X_k}(eX_{k+1}) by Newton's method failed. Retreat to near-by log in the extended manifold.")
                nearlog_SpecOrth!(X_new, X_saf, EX_new, X, wsp_saf_k; order=true, regular=true)
            end
            if logZ_flag
                MatZ_new .= tempZ
            else
                # throw(1)
                println("Warning! Near-by log on Z_{k+1} = log_{Z_k}(eZ_{k+1}) by Newton's method failed. Retreat to near-by log in the extended manifold.")
                nearlog_SpecOrth!(Z_new, Z_saf, EZ_new, Z, wsp_saf_m; order=true, regular=true)
            end


            fval_sq_new = (sum((MatX_new .- MatS_new[1:k, 1:k]) .^ 2) + sum((MatZ_new .- MatS_new[(k+1):n, (k+1):n]) .^ 2)) / 2.0
            fval_nm_new = sqrt(2.0 * fval_sq_new)

            descent = fval_sq_new - fval_sq

            # if descent > γ * (slope + min(stepsize, 1.5) * fval_sq) * stepsize
            if descent > max(γ * slope * stepsize, -0.9 * fval_sq)
                d_msgln("\n==============Armijo Line Search==============\n")
                d_msgln("Current objective value:\t$(fval_sq)")
            end

            while descent > max(γ * slope * stepsize, -0.9 * fval_sq)


                stepsize = stepsize_shrink * stepsize

                MatQ_new[:, 1:k] .= MatQ[:, 1:k] * exp(stepsize .* MatΔUk)
                MatQ_new[:, (k+1):n] .= MatQ[:, (k+1):n] * exp(stepsize .* MatΔUp)
                MatEX_new .= MatEX * exp(stepsize .* MatΔUk)
                MatEZ_new .= MatEZ * exp(stepsize .* MatΔUp)


                # nearlog_SpecOrth!(S_new, S_saf, Q_new, S, wsp_saf_n; order=true, regular=true)
                # nearlog_SpecOrth!(X_new, X_saf, EX_new, X, wsp_saf_k; order=true, regular=true)
                # nearlog_SpecOrth!(Z_new, Z_saf, EZ_new, Z, wsp_saf_m; order=true, regular=true)
                # log_SpecOrth!(X_new, X_saf, EX_new, wsp_saf_k; order=true, regular=true)
                # log_SpecOrth!(Z_new, Z_saf, EZ_new, wsp_saf_m; order=true, regular=true)

                logS_flag, logS_iter, tempS = nearlog_SpecOrth_newton(MatQ_new, (MatS .+ stepsize .* MatΔS), exp(MatS .+ stepsize .* MatΔS))
                logX_flag, logX_iter, tempX = nearlog_SpecOrth_newton(MatEX_new, (MatX .+ stepsize .* MatΔX), exp(MatX .+ stepsize .* MatΔX))
                logZ_flag, logZ_iter, tempZ = nearlog_SpecOrth_newton(MatEZ_new, (MatZ .+ stepsize .* MatΔZ), exp(MatZ .+ stepsize .* MatΔZ))

                if logS_flag
                    MatS_new .= tempS
                else
                    # throw(1)
                    println("Warning! Near-by log on S_{k+1} = log_{S_k}(Q_{k+1}) by Newton's method failed. Retreat to near-by log in the extended manifold.")
                    nearlog_SpecOrth!(S_new, S_saf, Q_new, S, wsp_saf_n; order=true, regular=true)
                end
                if logX_flag
                    MatX_new .= tempX
                else
                    # throw(1)
                    println("Warning! Near-by log on X_{k+1} = log_{X_k}(eX_{k+1}) by Newton's method failed. Retreat to near-by log in the extended manifold.")
                    nearlog_SpecOrth!(X_new, X_saf, EX_new, X, wsp_saf_k; order=true, regular=true)
                end
                if logZ_flag
                    MatZ_new .= tempZ
                else
                    # throw(1)
                    println("Warning! Near-by log on Z_{k+1} = log_{Z_k}(eZ_{k+1}) by Newton's method failed. Retreat to near-by log in the extended manifold.")
                    nearlog_SpecOrth!(Z_new, Z_saf, EZ_new, Z, wsp_saf_m; order=true, regular=true)
                end


                fval_sq_new = (sum((MatX_new .- MatS_new[1:k, 1:k]) .^ 2) + sum((MatZ_new .- MatS_new[(k+1):n, (k+1):n]) .^ 2)) / 2.0
                fval_nm_new = sqrt(2.0 * fval_sq_new)

                descent = fval_sq_new - fval_sq

                d_msgln("Stepsize:\t $(stepsize)\tDescent:\t$(descent)\tRequirement:\t$(max(γ * slope * stepsize, -0.9 * fval_sq))")


                if stepsize < minstep
                    msgln("Fail to perform Armijo line search, restart.")

                    stepsize = min(π / opnorm(MatΔUk), π / opnorm(MatΔUp))

                    MatQ_new[:, 1:k] .= MatQ[:, 1:k] * exp(stepsize .* MatΔUk)
                    MatQ_new[:, (k+1):n] .= MatQ[:, (k+1):n] * exp(stepsize .* MatΔUp)
                    Q_new = Ref(MatQ_new)

                    MatUk .= MatQ_new[:, 1:k]
                    MatUp .= MatQ_new[:, (k+1):n]


                    MatX_new = log(MatVk' * MatUk)
                    MatX_new .-= MatX_new'
                    MatX_new .*= 0.5
                    MatEX_new = exp(MatX_new)

                    MatZ_new = zeros(m, m)
                    MatEZ_new = exp(MatZ_new)


                    log_SpecOrth!(S_new, S_saf, Q_new, wsp_saf_n; order=true, regular=true)
                    log_SpecOrth!(X_new, X_saf, EX_new, wsp_saf_k; order=true, regular=true)
                    log_SpecOrth!(Z_new, Z_saf, EZ_new, wsp_saf_m; order=true, regular=true)
                    break
                end
            end

            fval_nm = fval_nm_new
            fval_sq = fval_sq_new

            MatQ .= MatQ_new
            MatS .= MatS_new
            MatX .= MatX_new
            MatZ .= MatZ_new

            MatEX .= MatEX_new
            MatEZ .= MatEZ_new

            MatUk .= MatQ[:, 1:k]
            MatUp .= MatQ[:, (k+1):n]

            MatA .= MatS[1:k, 1:k]
            MatB .= MatS[(k+1):n, 1:k]
            MatC .= MatS[(k+1):n, (k+1):n]

        end

        iter += 1
        cost_record[iter] = fval_nm
        time_record[iter] = time_record[iter-1] + 1000 * (stats.time - stats.gctime)
        mats_record[iter] = copy(MatS)
        matx_record[iter] = copy(MatX)
        matΔuk_record[iter] = copy(MatΔUk)
        matΔup_record[iter] = copy(MatΔUp)

        result_flag = check_termination_vec(cost_record, nothing, vect_record, nothing, step_record, iter, Stop)
        # result_flag = check_termination_vec(cost_record, nothing, vect_record, time_record, step_record, iter, Stop)
    end

    return (iter, mats_record[iter], matx_record[iter]), (cost_record[1:iter], time_record[1:iter], mats_record[1:iter], matx_record[1:iter], matΔuk_record[1:iter], matΔup_record[1:iter])
end



#######################################Test functions#######################################
using Random, Plots

function grhor_grdist(Vk)
    n, k = size(Vk)
    factor = svd(Vk[(k+1):n, :])
    angles = -asin.(factor.S)
    return sqrt(sum(angles .^ 2))
end

function test_grhor_newton(n, k, scale::T; MaxIter=200, AbsTol=1e-8, InitSeed=1234, FullSolver=false, RandAttempt=10, TestProb="V", MatS=zeros(n, n)) where {T<:Real}
    println("Test on GrHor solver on connecting I_{n,k} to V, with various special orthogonal completion Vp specified on entry.")

    if FullSolver
        println("For a specified Vp, return S_{A,B,C} that satisfies exp(S_{A,B,C}) = | V exp(A) | Vp exp(C) |.")
    else
        println("For a specified Vp, return S_{A,B,C} that satisfies exp(S_{A,B,C}) = | V exp(A) | Vp | = | V exp(A) | Vp exp(-C) exp(C) |.")
    end

    S = nothing
    S_saf = nothing
    Q = nothing
    A = nothing
    B = nothing
    C = nothing

    Vk = nothing
    Vp = nothing
    Wp = nothing
    StVp = nothing

    InitList = nothing
    InitLabel = nothing

    if TestProb == "V"
        println("The Stiefel matrix V is generated by a randomly sampled S_{A,B,C} with θ_1 + θ_2 = $(scale) as V = exp(S_{A,B,C})I_{n,k}exp(-A). Note that the generating S_{A,B,C} is a solution.")
        S = rand(n, n)
        S .-= S'
        S_saf = schurAngular_SkewSymm(Ref(S))
        S .*= scale / (abs(S_saf.angle[][1]) + abs(S_saf.angle[][2]))

        Q = exp(S)

        A = copy(S[1:k, 1:k])
        B = copy(S[(k+1):n, 1:k])
        C = copy(S[(k+1):n, (k+1):n])

        Vk = Q[:, 1:k] * exp(-A)

        Vp = Q[:, (k+1):n] * exp(-C)
        Wp = copy(Q[:, (k+1):n])

        StVp = zeros(n, n - k)
        StSkew, flag, iter, = stlog_hybrid_Newton_armijo(Vk, StVp; Init=init_guess_simple, NMLS_Set=NMLS_SET, Stop=terminator(100, 10000, 1e-8, 1e-6))

        if flag > 2
            println("Stlog failed!")
            InitList = [Vp, Wp]
            InitLabel = ["V_perp Initial", "W_perp Initial"]
        else
            println("Stlog done!")
            StVp .= exp(StSkew)[:, (k+1):n]

            InitList = [Vp, Wp, StVp]
            InitLabel = ["V_perp Initial", "W_perp Initial", "St_log Initial"]
        end
    elseif TestProb == "StV"
        println("The Stiefel matrix V is generated by a randomly sampled S_{A,B,0} with θ_1 + θ_2 = $(scale) as V = exp(S_{A,B,0})I_{n,k}exp(-A). Note that the generating S_{A,B,0} is a solution.")

        S = rand(n, n)
        S .-= S'
        S[(k+1):n, (k+1):n] .= 0.0
        S_saf = schurAngular_SkewSymm(Ref(S))
        S .*= scale / (abs(S_saf.angle[][1]) + abs(S_saf.angle[][2]))

        Q = exp(S)

        A = copy(S[1:k, 1:k])
        B = copy(S[(k+1):n, 1:k])
        C = copy(S[(k+1):n, (k+1):n])

        Vk = Q[:, 1:k] * exp(-A)

        Vp = copy(Q[:, (k+1):n])

        StVp = zeros(n, n - k)
        StSkew, flag, iter, = stlog_hybrid_Newton_armijo(Vk, StVp; Init=init_guess_simple, NMLS_Set=NMLS_SET, Stop=terminator(100, 10000, 1e-8, 1e-6))

        if flag > 2
            println("Stlog failed!")
            InitList = [Vp]
            InitLabel = ["V_perp Initial"]
        else
            println("Stlog done!")
            StVp .= exp(StSkew)[:, (k+1):n]

            InitList = [Vp, StVp]
            InitLabel = ["V_perp Initial", "St_log Initial"]
        end
    elseif TestProb == "W"
        println("The Stiefel matrix V is generated by a randomly sampled S_{A,B,C} with θ_1 + θ_2 = $(scale) as V = exp(S_{A,B,C})I_{n,k}. Note that the generating S_{A,B,C} is not a solution.")
        S = rand(n, n)
        S .-= S'
        S_saf = schurAngular_SkewSymm(Ref(S))
        S .*= scale / (abs(S_saf.angle[][1]) + abs(S_saf.angle[][2]))

        Q = exp(S)

        A = copy(S[1:k, 1:k])
        B = copy(S[(k+1):n, 1:k])
        C = copy(S[(k+1):n, (k+1):n])

        Vk = Q[:, 1:k]

        Vp = Q[:, (k+1):n] * exp(-C)
        Wp = copy(Q[:, (k+1):n])

        StVp = zeros(n, n - k)
        StSkew, flag, iter, = stlog_hybrid_Newton_armijo(Vk, StVp; Init=init_guess_simple, NMLS_Set=NMLS_SET, Stop=terminator(100, 10000, 1e-8, 1e-6))

        if flag > 2
            println("Stlog failed!")
            InitList = [Vp, Wp]
            InitLabel = ["V_perp Initial", "W_perp Initial"]
        else
            println("Stlog done!")
            StVp .= exp(StSkew)[:, (k+1):n]

            InitList = [Vp, Wp, StVp]
            InitLabel = ["V_perp Initial", "W_perp Initial", "St_log Initial"]
        end
    elseif TestProb == "StW"
        println("The Stiefel matrix V is generated by a randomly sampled S_{A,B,0} with θ_1 + θ_2 = $(scale) as V = exp(S_{A,B,0})I_{n,k}. Note that the generating S_{A,B,0} is not a solution.")

        S = rand(n, n)
        S .-= S'
        S[(k+1):n, (k+1):n] .= 0.0
        S_saf = schurAngular_SkewSymm(Ref(S))
        S .*= scale / (abs(S_saf.angle[][1]) + abs(S_saf.angle[][2]))

        Q = exp(S)

        A = copy(S[1:k, 1:k])
        B = copy(S[(k+1):n, 1:k])
        C = copy(S[(k+1):n, (k+1):n])

        Vk = Q[:, 1:k]

        Vp = copy(Q[:, (k+1):n])

        StVp = zeros(n, n - k)
        StSkew, flag, iter, = stlog_hybrid_Newton_armijo(Vk, StVp; Init=init_guess_simple, NMLS_Set=NMLS_SET, Stop=terminator(100, 10000, 1e-8, 1e-6))

        if flag > 2
            println("Stlog failed!")
            InitList = [Vp]
            InitLabel = ["V_perp Initial"]
        else
            println("Stlog done!")
            StVp .= exp(StSkew)[:, (k+1):n]

            InitList = [Vp, StVp]
            InitLabel = ["V_perp Initial", "St_log Initial"]
        end
    elseif TestProb == "MSV"
        println("The Stiefel matrix V is generated by the input matrix S_{A,B,C} as V = exp(S_{A,B,C})I_{n, k} exp(-A). Note that S_{A,B,C} is a solution.")
        if norm(S) == 0
            println("Input S not specified")
            throw(1)
        end
        S = copy(MatS)
        Q = exp(S)

        A = copy(S[1:k, 1:k])
        B = copy(S[(k+1):n, 1:k])
        C = copy(S[(k+1):n, (k+1):n])

        Vk = Q[:, 1:k] * exp(-A)

        Vp = Q[:, (k+1):n] * exp(-C)
        Wp = copy(Q[:, (k+1):n])

        StVp = zeros(n, n - k)
        StSkew, flag, iter, = stlog_hybrid_Newton_armijo(Vk, StVp; Init=init_guess_simple, NMLS_Set=NMLS_SET, Stop=terminator(100, 10000, 1e-8, 1e-6))

        if flag > 2
            println("Stlog failed!")
            InitList = [Vp, Wp]
            InitLabel = ["V_perp Initial", "W_perp Initial"]
        else
            println("Stlog done!")
            StVp .= exp(StSkew)[:, (k+1):n]

            InitList = [Vp, Wp, StVp]
            InitLabel = ["V_perp Initial", "W_perp Initial", "St_log Initial"]
        end
    elseif TestProb == "MSW"
        println("The Stiefel matrix V is generated by the input matrix S_{A,B,C} as V = exp(S_{A,B,C})I_{n, k}. Note that S_{A,B,C} is not a solution.")
        if norm(S) == 0
            println("Input S not specified")
            throw(1)
        end
        S = copy(MatS)
        Q = exp(S)

        A = copy(S[1:k, 1:k])
        B = copy(S[(k+1):n, 1:k])
        C = copy(S[(k+1):n, (k+1):n])

        Vk = Q[:, 1:k]

        Vp = Q[:, (k+1):n] * exp(-C)
        Wp = copy(Q[:, (k+1):n])

        StVp = zeros(n, n - k)
        StSkew, flag, iter, = stlog_hybrid_Newton_armijo(Vk, StVp; Init=init_guess_simple, NMLS_Set=NMLS_SET, Stop=terminator(100, 10000, 1e-8, 1e-6))

        if flag > 2
            println("Stlog failed!")
            InitList = [Vp, Wp]
            InitLabel = ["V_perp Initial", "W_perp Initial"]
        else
            println("Stlog done!")
            StVp .= exp(StSkew)[:, (k+1):n]

            InitList = [Vp, Wp, StVp]
            InitLabel = ["V_perp Initial", "W_perp Initial", "St_log Initial"]
        end
    elseif TestProb == "Col"
        println("The Stiefel matrix V is generated by a randomly sampled A with θ_1 + θ_2 = $(scale) as V = I_{n,k} exp(A). Note that the I_{n,k} and V span a same column space.")
        A = rand(n - k, n - k)
        A .-= A'
        A_saf = schurAngular_SkewSymm(Ref(A))
        A .*= scale / (abs(A_saf.angle[][1]) + abs(A_saf.angle[][2]))

        S = zeros(n, n)
        S[1:k, 1:k] .= A
        S_saf = schurAngular_SkewSymm(Ref(S))

        Q = exp(S)

        A = copy(S[1:k, 1:k])
        B = copy(S[(k+1):n, 1:k])
        C = copy(S[(k+1):n, (k+1):n])

        Vk = Q[:, 1:k]

        Vp = Q[:, (k+1):n]


        InitList = [Vp]
        InitLabel = ["V_perp Initial"]
    end

    if FullSolver
        solver = grhor_newton_full_core
    else
        solver = grhor_newton_core
    end



    println("==========================Properties of the generating S_{A,B,C} and the V==========================")

    saf = schurAngular_SkewSymm(Ref(S))
    # display(saf3)
    println("θ_1 + θ_2 =\t $(abs(saf.angle[][1]) + abs(saf.angle[][2])) =\t $((abs(saf.angle[][1]) + abs(saf.angle[][2]))/π) π")

    println("Gr distance between col(I_{n,k}) and col(V): $(grhor_grdist(Vk))")

    println("==========================Properties of the generating S_{A,B,C} and the V==========================\n")

    sol = Vector{Any}(undef, length(InitList))
    rec = Vector{Any}(undef, length(InitList))
    saf = Vector{Any}(undef, length(InitList))

    rand_eng = MersenneTwister(InitSeed)


    println("Random Initialization ...")

    sol_rand = nothing
    rec_rand = nothing
    saf_rand = nothing

    attempt::Int = 1


    while attempt <= RandAttempt
        println("Attempt $(attempt):\n")
        if TestProb == "Col"
            C = rand(rand_eng, n - k, n - k)
            C .-= C'
            C .*= rand(rand_eng) * π / opnorm(C)
            Vp = zeros(n, n - k)
            Vp[(k+1):n, :] .= exp(C)
            sol_rand, rec_rand = solver(Vk, Vp; Stop=terminator(MaxIter, 100000, AbsTol, 1e-6))
        else
            sol_rand, rec_rand = solver(Vk; Stop=terminator(MaxIter, 100000, AbsTol, 1e-6), RandEng=rand_eng)
        end

        println("A ≈ X?\t", sol_rand[2][1:k, 1:k] ≈ sol_rand[3], "\t Difference: ", maximum(abs.(sol_rand[2][1:k, 1:k] .- sol_rand[3])))

        saf_rand = schurAngular_SkewSymm(Ref(sol_rand[2]))

        if maximum(abs.(sol_rand[2][1:k, 1:k] .- sol_rand[3])) < 1e-6
            println("Solution found with $(sol_rand[1]) iterations.")
            println("θ_1 + θ_2 =\t $(abs(saf_rand.angle[][1]) + abs(saf_rand.angle[][2])) =\t $((abs(saf_rand.angle[][1]) + abs(saf_rand.angle[][2]))/π) π")
            println("Curve length:\t $(norm(sol_rand[2][(k+1):n, 1:k])).")
            break
        else
            attempt += 1
        end
    end
    if attempt > RandAttempt
        println("Solver failed.\n")
    end

    println("Random Initialization: Done!\n\n")

    println("Specified Initialization ...")

    for iter in eachindex(InitList)
        println(InitLabel[iter], " ...")
        sol[iter], rec[iter] = solver(Vk, InitList[iter]; Stop=terminator(MaxIter, 100000, AbsTol, 1e-6))
        println("A ≈ X?\t", sol[iter][2][1:k, 1:k] ≈ sol[iter][3], "\t Difference: ", maximum(abs.(sol[iter][2][1:k, 1:k] .- sol[iter][3])))

        saf[iter] = schurAngular_SkewSymm(Ref(sol[iter][2]))

        if maximum(abs.(sol[iter][2][1:k, 1:k] .- sol[iter][3])) < 1e-6
            println("Solution found with $(sol[iter][1]) iterations..")
            println("θ_1 + θ_2 =\t $(abs(saf[iter].angle[][1]) + abs(saf[iter].angle[][2])) =\t $((abs(saf[iter].angle[][1]) + abs(saf[iter].angle[][2]))/π) π")
            println("Curve length:\t $(norm(sol[iter][2][(k+1):n, 1:k])).")
        else
            println("Solver failed.")
        end

        println(InitLabel[iter], ": Done!\n\n")
    end

    # plt = plot(rec1[1], label="Random initial guess", yscale=:log10, ylabel="Objective value", xlabel="Number of iteration")
    # plot!(rec2[1], label="V_perp initial guess", yscale=:log10)
    # plot!(rec3[1], label="W_perp initial guess", yscale=:log10)
    # plot!(rec4[1], label="St_log initial guess", yscale=:log10)

    # display(plt)

    @printf "Initialization\t\tConvergence\tθ_1+θ_2\t\tLength\n"
    @printf "Random Initial\t\t%e\t%e π\t%e\n" maximum(abs.(sol_rand[2][1:k, 1:k] .- sol_rand[3])) ((abs(saf_rand.angle[][1]) + abs(saf_rand.angle[][2])) / π) norm(sol_rand[2][(k+1):n, 1:k])
    for iter in eachindex(InitList)
        @printf "%s\t\t%e\t%e π\t%e\n" InitLabel[iter] maximum(abs.(sol[iter][2][1:k, 1:k] .- sol[iter][3])) ((abs(saf[iter].angle[][1]) + abs(saf[iter].angle[][2])) / π) norm(sol[iter][2][(k+1):n, 1:k])
    end
    print("\n")
    if TestProb == "V" || TestProb == "StV"
        @printf "Length of the generating horizontal curve:\t\t%e\n" norm(S[(k+1):n, 1:k])
    end
    @printf "Gr distance between col(I_{n,k}) and col(V):\t\t%e\n" grhor_grdist(Vk)

end

function test_grhor_newton_Vproblem(n, k, scale::T; MaxIter=200, AbsTol=1e-8, InitSeed=1234, FullSolver=false, RandAttempt=10) where {T<:Real}
    println("Testing GrHor solver on connecting I_{n,k} and Vk, where Vk is generated by skew-symmetric S_{A,B,C} with exp(S_{A,B,C}) = | Vk exp(A)| Vp exp(C) | and this problem has a known solution S_{A,B,C} and Vp.")
    println("S_{A,B,C} is randomly sampled with its θ_1 + θ_2 specified from the inputs.")

    S = rand(n, n)
    S .-= S'
    S_saf = schurAngular_SkewSymm(Ref(S))
    S .*= scale / (abs(S_saf.angle[][1]) + abs(S_saf.angle[][2]))

    Q = exp(S)

    A = copy(S[1:k, 1:k])
    B = copy(S[(k+1):n, 1:k])
    C = copy(S[(k+1):n, (k+1):n])

    Vk = Q[:, 1:k] * exp(-A)
    Vp = Q[:, (k+1):n] * exp(-C)
    Wp = Q[:, (k+1):n]

    println("==========================Generating Horizontal Curve==========================")

    saf = schurAngular_SkewSymm(Ref(S))
    # display(saf3)
    println("θ_1 + θ_2 =\t $(abs(saf.angle[][1]) + abs(saf.angle[][2])) =\t $((abs(saf.angle[][1]) + abs(saf.angle[][2]))/π) π")

    println("Curve length:\t $(norm(B)).")

    println("==========================Generating Horizontal Curve==========================\n")

    sol1, rec1 = nothing, nothing
    sol2, rec2 = nothing, nothing
    sol3, rec3 = nothing, nothing
    sol4, rec4 = nothing, nothing

    saf1 = nothing
    saf2 = nothing
    saf3 = nothing
    saf4 = nothing

    rand_eng = MersenneTwister(InitSeed)





    println("==========================Random initialization==========================")
    println("Random special orthogonal completion to Vk is used.\n")
    attempt::Int = 1

    flags = Vector{Bool}(undef, 3)
    fill!(flags, false)

    while attempt <= RandAttempt
        println("Attempt $(attempt):\n")
        if FullSolver
            sol1, rec1 = grhor_newton_full_core(Vk; Stop=terminator(MaxIter, 100000, AbsTol, 1e-6), RandEng=rand_eng)
        else
            sol1, rec1 = grhor_newton_core(Vk; Stop=terminator(MaxIter, 100000, AbsTol, 1e-6), RandEng=rand_eng)
        end
        println("A ≈ X?\t", sol1[2][1:k, 1:k] ≈ sol1[3], "\t Difference: ", maximum(abs.(sol1[2][1:k, 1:k] .- sol1[3])))
        if maximum(abs.(sol1[2][1:k, 1:k] .- sol1[3])) < 1e-6
            println("Solution Found")
            saf1 = schurAngular_SkewSymm(Ref(sol1[2]))
            # display(saf1)
            println("θ_1 + θ_2 =\t $(abs(saf1.angle[][1]) + abs(saf1.angle[][2])) =\t $((abs(saf1.angle[][1]) + abs(saf1.angle[][2]))/π) π")
            println("Curve length:\t $(norm(sol1[2][(k+1):n, 1:k])).")
            flags[1] = true
            break
        else
            println("Solver failed.\n")
            attempt += 1
        end
    end

    println("==========================Random initialization==========================\n")


    println("==========================V_perp initialization==========================")
    println("Particular initialization Vp from the solution that satisfies exp(S_{A,B,C}) = | Vk exp(A) | Vp exp(C) | is used.\n")
    if FullSolver
        sol2, rec2 = grhor_newton_full_core(Vk, Vp; Stop=terminator(MaxIter, 100000, AbsTol, 1e-6))
    else
        sol2, rec2 = grhor_newton_core(Vk, Vp; Stop=terminator(MaxIter, 100000, AbsTol, 1e-6))
    end
    println("A ≈ X?\t", sol2[2][1:k, 1:k] ≈ sol2[3], "\t Difference: ", maximum(abs.(sol2[2][1:k, 1:k] .- sol2[3])))
    if maximum(abs.(sol2[2][1:k, 1:k] .- sol2[3])) < 1e-6
        println("Solution found.")
        saf2 = schurAngular_SkewSymm(Ref(sol2[2]))
        # display(saf2)
        println("θ_1 + θ_2 =\t $(abs(saf2.angle[][1]) + abs(saf2.angle[][2])) =\t $((abs(saf2.angle[][1]) + abs(saf2.angle[][2]))/π) π")
        println("Curve length:\t $(norm(sol2[2][(k+1):n, 1:k])).")
        flags[2] = true
    else
        println("Solver failed.")
    end
    println("==========================V_perp initialization==========================\n")


    println("==========================W_perp initialization==========================")

    println("Particular initialization Wp from the solution that satisfies exp(S_{A,B,C}) = | Vk exp(A) | Wp | is used.\n")

    if FullSolver
        sol3, rec3 = grhor_newton_full_core(Vk, Wp; Stop=terminator(MaxIter, 100000, AbsTol, 1e-6))
    else
        sol3, rec3 = grhor_newton_core(Vk, Wp; Stop=terminator(MaxIter, 100000, AbsTol, 1e-6))
    end
    println("A ≈ X?\t", sol3[2][1:k, 1:k] ≈ sol3[3], "\t Difference: ", maximum(abs.(sol3[2][1:k, 1:k] .- sol3[3])))
    if maximum(abs.(sol3[2][1:k, 1:k] .- sol3[3])) < 1e-6
        println("Solution found.")
        saf3 = schurAngular_SkewSymm(Ref(sol3[2]))
        # display(saf3)
        println("θ_1 + θ_2 =\t $(abs(saf3.angle[][1]) + abs(saf3.angle[][2])) =\t $((abs(saf3.angle[][1]) + abs(saf3.angle[][2]))/π) π")
        println("Curve length:\t $(norm(sol3[2][(k+1):n, 1:k])).")
    else
        println("Solver failed.")
    end
    println("==========================W_perp initialization==========================\n")


    println("==========================St_log initialization==========================")

    println("Particular initialization Wp from the Stiefel geodesic that satisfies exp(S_{A,B,0}) = | Vk | Wp | is used.\n")

    StVp = zeros(n, n - k)
    StSkew, flag, iter, = stlog_hybrid_Newton_armijo(Vk, StVp; Init=init_guess_simple, NMLS_Set=NMLS_SET, Stop=terminator(100, 10000, 1e-8, 1e-6))

    if flag > 2
        println("Stlog failed!")
    else
        println("Stlog done!")
        StVp .= exp(StSkew)[:, (k+1):n]
    end

    if FullSolver
        sol4, rec4 = grhor_newton_full_core(Vk, StVp; Stop=terminator(MaxIter, 100000, AbsTol, 1e-6))
    else
        sol4, rec4 = grhor_newton_core(Vk, StVp; Stop=terminator(MaxIter, 100000, AbsTol, 1e-6))
    end
    println("A ≈ X?\t", sol4[2][1:k, 1:k] ≈ sol4[3], "\t Difference: ", maximum(abs.(sol4[2][1:k, 1:k] .- sol4[3])))
    if maximum(abs.(sol4[2][1:k, 1:k] .- sol4[3])) < 1e-6
        println("Solution found.")
        saf4 = schurAngular_SkewSymm(Ref(sol4[2]))
        # display(saf3)
        println("θ_1 + θ_2 =\t $(abs(saf4.angle[][1]) + abs(saf4.angle[][2])) =\t $((abs(saf4.angle[][1]) + abs(saf4.angle[][2]))/π) π")
        println("Curve length:\t $(norm(sol4[2][(k+1):n, 1:k])).")
    else
        println("Solver failed.")
    end
    println("==========================St_log initialization==========================\n")

    plt = plot(rec1[1], label="Random initial guess", yscale=:log10, ylabel="Objective value", xlabel="Number of iteration")
    plot!(rec2[1], label="V_perp initial guess", yscale=:log10)
    plot!(rec3[1], label="W_perp initial guess", yscale=:log10)
    plot!(rec4[1], label="St_log initial guess", yscale=:log10)
    display(plt)

    @printf "Method\t\t\tConvergence\tθ_1+θ_2\t\tLength\n"
    @printf "Generating Curve\tNot-Applied\t%e π\t%e\n" ((abs(saf.angle[][1]) + abs(saf.angle[][2])) / π) norm(S[(k+1):n, 1:k])
    @printf "Random Initial\t\t%e\t%e π\t%e\n" maximum(abs.(sol1[2][1:k, 1:k] .- sol1[3])) ((abs(saf1.angle[][1]) + abs(saf1.angle[][2])) / π) norm(sol1[2][(k+1):n, 1:k])
    @printf "V_perp Initial\t\t%e\t%e π\t%e\n" maximum(abs.(sol2[2][1:k, 1:k] .- sol2[3])) ((abs(saf2.angle[][1]) + abs(saf2.angle[][2])) / π) norm(sol2[2][(k+1):n, 1:k])
    @printf "W_perp Initial\t\t%e\t%e π\t%e\n" maximum(abs.(sol3[2][1:k, 1:k] .- sol3[3])) ((abs(saf3.angle[][1]) + abs(saf3.angle[][2])) / π) norm(sol3[2][(k+1):n, 1:k])
    @printf "St_log Initial\t\t%e\t%e π\t%e\n" maximum(abs.(sol4[2][1:k, 1:k] .- sol4[3])) ((abs(saf4.angle[][1]) + abs(saf4.angle[][2])) / π) norm(sol4[2][(k+1):n, 1:k])


end

function test_grhor_newton_Vproblem(n, k, S::Matrix{Float64}; MaxIter=200, AbsTol=1e-8, InitSeed=1234, FullSolver=false, RandAttempt=10)
    println("Testing GrHor solver on connecting I_{n,k} and Vk, where Vk is generated by skew-symmetric S_{A,B,C} with exp(S_{A,B,C}) = | Vk exp(A)| Vp exp(C) | and this problem has a known solution S_{A,B,C} and Vp.")
    println("S_{A,B,C} is specified from inputs.")

    Q = exp(S)

    A = copy(S[1:k, 1:k])
    B = copy(S[(k+1):n, 1:k])
    C = copy(S[(k+1):n, (k+1):n])

    Vk = Q[:, 1:k] * exp(-A)
    Vp = Q[:, (k+1):n] * exp(-C)
    Wp = Q[:, (k+1):n]

    println("==========================Generating Horizontal Curve==========================")

    saf = schurAngular_SkewSymm(Ref(S))
    # display(saf3)
    println("θ_1 + θ_2 =\t $(abs(saf.angle[][1]) + abs(saf.angle[][2])) =\t $((abs(saf.angle[][1]) + abs(saf.angle[][2]))/π) π")

    println("Curve length:\t $(norm(B)).")

    println("==========================Generating Horizontal Curve==========================\n")

    sol1, rec1 = nothing, nothing
    sol2, rec2 = nothing, nothing
    sol3, rec3 = nothing, nothing
    sol4, rec4 = nothing, nothing

    saf1 = nothing
    saf2 = nothing
    saf3 = nothing
    saf4 = nothing

    rand_eng = MersenneTwister(InitSeed)





    println("==========================Random initialization==========================")
    println("Random special orthogonal completion to Vk is used.\n")
    attempt::Int = 1

    flags = Vector{Bool}(undef, 3)
    fill!(flags, false)

    while attempt <= RandAttempt
        println("Attempt $(attempt):\n")
        if FullSolver
            sol1, rec1 = grhor_newton_full_core(Vk; Stop=terminator(MaxIter, 100000, AbsTol, 1e-6), RandEng=rand_eng)
        else
            sol1, rec1 = grhor_newton_core(Vk; Stop=terminator(MaxIter, 100000, AbsTol, 1e-6), RandEng=rand_eng)
        end
        println("A ≈ X?\t", sol1[2][1:k, 1:k] ≈ sol1[3], "\t Difference: ", maximum(abs.(sol1[2][1:k, 1:k] .- sol1[3])))
        if maximum(abs.(sol1[2][1:k, 1:k] .- sol1[3])) < 1e-6
            println("Solution Found")
            saf1 = schurAngular_SkewSymm(Ref(sol1[2]))
            # display(saf1)
            println("θ_1 + θ_2 =\t $(abs(saf1.angle[][1]) + abs(saf1.angle[][2])) =\t $((abs(saf1.angle[][1]) + abs(saf1.angle[][2]))/π) π")
            println("Curve length:\t $(norm(sol1[2][(k+1):n, 1:k])).")
            flags[1] = true
            break
        else
            println("Solver failed.\n")
            attempt += 1
        end
    end

    println("==========================Random initialization==========================\n")


    println("==========================V_perp initialization==========================")
    println("Particular initialization Vp from the solution that satisfies exp(S_{A,B,C}) = | Vk exp(A) | Vp exp(C) | is used.\n")
    if FullSolver
        sol2, rec2 = grhor_newton_full_core(Vk, Vp; Stop=terminator(MaxIter, 100000, AbsTol, 1e-6))
    else
        sol2, rec2 = grhor_newton_core(Vk, Vp; Stop=terminator(MaxIter, 100000, AbsTol, 1e-6))
    end
    println("A ≈ X?\t", sol2[2][1:k, 1:k] ≈ sol2[3], "\t Difference: ", maximum(abs.(sol2[2][1:k, 1:k] .- sol2[3])))
    if maximum(abs.(sol2[2][1:k, 1:k] .- sol2[3])) < 1e-6
        println("Solution found.")
        saf2 = schurAngular_SkewSymm(Ref(sol2[2]))
        # display(saf2)
        println("θ_1 + θ_2 =\t $(abs(saf2.angle[][1]) + abs(saf2.angle[][2])) =\t $((abs(saf2.angle[][1]) + abs(saf2.angle[][2]))/π) π")
        println("Curve length:\t $(norm(sol2[2][(k+1):n, 1:k])).")
        flags[2] = true
    else
        println("Solver failed.")
    end
    println("==========================V_perp initialization==========================\n")


    println("==========================W_perp initialization==========================")

    println("Particular initialization Wp from the solution that satisfies exp(S_{A,B,C}) = | Vk exp(A) | Wp | is used.\n")

    if FullSolver
        sol3, rec3 = grhor_newton_full_core(Vk, Wp; Stop=terminator(MaxIter, 100000, AbsTol, 1e-6))
    else
        sol3, rec3 = grhor_newton_core(Vk, Wp; Stop=terminator(MaxIter, 100000, AbsTol, 1e-6))
    end
    println("A ≈ X?\t", sol3[2][1:k, 1:k] ≈ sol3[3], "\t Difference: ", maximum(abs.(sol3[2][1:k, 1:k] .- sol3[3])))
    if maximum(abs.(sol3[2][1:k, 1:k] .- sol3[3])) < 1e-6
        println("Solution found.")
        saf3 = schurAngular_SkewSymm(Ref(sol3[2]))
        # display(saf3)
        println("θ_1 + θ_2 =\t $(abs(saf3.angle[][1]) + abs(saf3.angle[][2])) =\t $((abs(saf3.angle[][1]) + abs(saf3.angle[][2]))/π) π")
        println("Curve length:\t $(norm(sol3[2][(k+1):n, 1:k])).")
    else
        println("Solver failed.")
    end
    println("==========================W_perp initialization==========================\n")

    println("==========================St_log initialization==========================")

    println("Particular initialization Wp from the Stiefel geodesic that satisfies exp(S_{A,B,0}) = | Vk | Wp | is used.\n")

    StVp = zeros(n, n - k)
    StSkew, flag, iter, = stlog_hybrid_Newton_armijo(Vk, StVp; Init=init_guess_simple, NMLS_Set=NMLS_SET, Stop=terminator(100, 10000, 1e-8, 1e-6))

    if flag > 2
        println("Stlog failed!")
    else
        println("Stlog done!")
        StVp .= exp(StSkew)[:, (k+1):n]
    end

    if FullSolver
        sol4, rec4 = grhor_newton_full_core(Vk, StVp; Stop=terminator(MaxIter, 100000, AbsTol, 1e-6))
    else
        sol4, rec4 = grhor_newton_core(Vk, StVp; Stop=terminator(MaxIter, 100000, AbsTol, 1e-6))
    end
    println("A ≈ X?\t", sol4[2][1:k, 1:k] ≈ sol4[3], "\t Difference: ", maximum(abs.(sol4[2][1:k, 1:k] .- sol4[3])))
    if maximum(abs.(sol4[2][1:k, 1:k] .- sol4[3])) < 1e-6
        println("Solution found.")
        saf4 = schurAngular_SkewSymm(Ref(sol4[2]))
        # display(saf3)
        println("θ_1 + θ_2 =\t $(abs(saf4.angle[][1]) + abs(saf4.angle[][2])) =\t $((abs(saf4.angle[][1]) + abs(saf4.angle[][2]))/π) π")
        println("Curve length:\t $(norm(sol4[2][(k+1):n, 1:k])).")
    else
        println("Solver failed.")
    end
    println("==========================St_log initialization==========================\n")

    plt = plot(rec1[1], label="Random initial guess", yscale=:log10, ylabel="Objective value", xlabel="Number of iteration")
    plot!(rec2[1], label="V_perp initial guess", yscale=:log10)
    plot!(rec3[1], label="W_perp initial guess", yscale=:log10)
    plot!(rec4[1], label="St_log initial guess", yscale=:log10)
    display(plt)

    @printf "Method\t\t\tConvergence\tθ_1+θ_2\t\tLength\n"
    @printf "Generating Curve\tNot-Applied\t%e π\t%e\n" ((abs(saf.angle[][1]) + abs(saf.angle[][2])) / π) norm(S[(k+1):n, 1:k])
    @printf "Random Initial\t\t%e\t%e π\t%e\n" maximum(abs.(sol1[2][1:k, 1:k] .- sol1[3])) ((abs(saf1.angle[][1]) + abs(saf1.angle[][2])) / π) norm(sol1[2][(k+1):n, 1:k])
    @printf "V_perp Initial\t\t%e\t%e π\t%e\n" maximum(abs.(sol2[2][1:k, 1:k] .- sol2[3])) ((abs(saf2.angle[][1]) + abs(saf2.angle[][2])) / π) norm(sol2[2][(k+1):n, 1:k])
    @printf "W_perp Initial\t\t%e\t%e π\t%e\n" maximum(abs.(sol3[2][1:k, 1:k] .- sol3[3])) ((abs(saf3.angle[][1]) + abs(saf3.angle[][2])) / π) norm(sol3[2][(k+1):n, 1:k])
    @printf "St_log Initial\t\t%e\t%e π\t%e\n" maximum(abs.(sol4[2][1:k, 1:k] .- sol4[3])) ((abs(saf4.angle[][1]) + abs(saf4.angle[][2])) / π) norm(sol4[2][(k+1):n, 1:k])
end

function test_grhor_newton_Wproblem(n, k, scale::T; MaxIter=200, AbsTol=1e-8, InitSeed=1234, FullSolver=false, RandAttempt=10) where {T<:Real}
    println("Testing GrHor solver on connecting I_{n,k} and Wk, where Wk is generated by skew-symmetric S_{A,B,C} with exp(S_{A,B,C}) = | Wk | Wp |. This is a general problem and does not expect an available answer.")
    println("S_{A,B,C} is randomly sampled with its θ_1 + θ_2 specified from the inputs.")

    S = rand(n, n)
    S .-= S'
    S_saf = schurAngular_SkewSymm(Ref(S))
    S .*= scale / (abs(S_saf.angle[][1]) + abs(S_saf.angle[][2]))

    Q = exp(S)

    A = copy(S[1:k, 1:k])
    B = copy(S[(k+1):n, 1:k])
    C = copy(S[(k+1):n, (k+1):n])

    Vk = Q[:, 1:k] * exp(-A)
    Vp = Q[:, (k+1):n] * exp(-C)
    Wk = copy(Q[:, 1:k])
    Wp = copy(Q[:, (k+1):n])

    println("==========================Generating SO Geodesic==========================")

    saf = schurAngular_SkewSymm(Ref(S))
    # display(saf3)
    println("θ_1 + θ_2 =\t $(abs(saf.angle[][1]) + abs(saf.angle[][2])) =\t $((abs(saf.angle[][1]) + abs(saf.angle[][2]))/π) π")

    println("==========================Generating SO Geodesic==========================\n")

    sol1, rec1 = nothing, nothing
    sol2, rec2 = nothing, nothing
    sol3, rec3 = nothing, nothing
    sol4, rec4 = nothing, nothing

    saf1 = nothing
    saf2 = nothing
    saf3 = nothing
    saf4 = nothing

    rand_eng = MersenneTwister(InitSeed)





    println("==========================Random initialization==========================")
    println("Random special orthogonal completion to Vk is used.\n")
    attempt::Int = 1

    flags = Vector{Bool}(undef, 3)
    fill!(flags, false)

    while attempt <= RandAttempt
        println("Attempt $(attempt):\n")
        if FullSolver
            sol1, rec1 = grhor_newton_full_core(Wk; Stop=terminator(MaxIter, 100000, AbsTol, 1e-6), RandEng=rand_eng)
        else
            sol1, rec1 = grhor_newton_core(Wk; Stop=terminator(MaxIter, 100000, AbsTol, 1e-6), RandEng=rand_eng)
        end
        println("A ≈ X?\t", sol1[2][1:k, 1:k] ≈ sol1[3], "\t Difference: ", maximum(abs.(sol1[2][1:k, 1:k] .- sol1[3])))

        saf1 = schurAngular_SkewSymm(Ref(sol1[2]))

        if maximum(abs.(sol1[2][1:k, 1:k] .- sol1[3])) < 1e-6
            println("Solution Found")
            display(sol1[2])
            # display(saf1)
            println("θ_1 + θ_2 =\t $(abs(saf1.angle[][1]) + abs(saf1.angle[][2])) =\t $((abs(saf1.angle[][1]) + abs(saf1.angle[][2]))/π) π")
            println("Curve length:\t $(norm(sol1[2][(k+1):n, 1:k])).")
            flags[1] = true
            break
        else
            println("Solver failed.\n")
            attempt += 1
        end
    end

    println("==========================Random initialization==========================\n")


    println("==========================V_perp initialization==========================")
    println("Particular initialization Vp from exp(S_{A,B,C}) = | Wk | Vp exp(C) | is used.\n")
    if FullSolver
        sol2, rec2 = grhor_newton_full_core(Wk, Vp; Stop=terminator(MaxIter, 100000, AbsTol, 1e-6))
    else
        sol2, rec2 = grhor_newton_core(Wk, Vp; Stop=terminator(MaxIter, 100000, AbsTol, 1e-6))
    end
    println("A ≈ X?\t", sol2[2][1:k, 1:k] ≈ sol2[3], "\t Difference: ", maximum(abs.(sol2[2][1:k, 1:k] .- sol2[3])))

    saf2 = schurAngular_SkewSymm(Ref(sol2[2]))

    if maximum(abs.(sol2[2][1:k, 1:k] .- sol2[3])) < 1e-6
        println("Solution found.")
        display(sol2[2])
        # display(saf2)
        println("θ_1 + θ_2 =\t $(abs(saf2.angle[][1]) + abs(saf2.angle[][2])) =\t $((abs(saf2.angle[][1]) + abs(saf2.angle[][2]))/π) π")
        println("Curve length:\t $(norm(sol2[2][(k+1):n, 1:k])).")
        flags[2] = true
    else
        println("Solver failed.")
    end
    println("==========================V_perp initialization==========================\n")


    println("==========================W_perp initialization==========================")

    println("Particular initialization Wp from exp(S_{A,B,C}) = | Wk | Wp | is used.\n")

    if FullSolver
        sol3, rec3 = grhor_newton_full_core(Wk, Wp; Stop=terminator(MaxIter, 100000, AbsTol, 1e-6))
    else
        sol3, rec3 = grhor_newton_core(Wk, Wp; Stop=terminator(MaxIter, 100000, AbsTol, 1e-6))
    end
    println("A ≈ X?\t", sol3[2][1:k, 1:k] ≈ sol3[3], "\t Difference: ", maximum(abs.(sol3[2][1:k, 1:k] .- sol3[3])))

    saf3 = schurAngular_SkewSymm(Ref(sol3[2]))

    if maximum(abs.(sol3[2][1:k, 1:k] .- sol3[3])) < 1e-6
        println("Solution found.")
        display(sol3[2])
        # display(saf3)
        println("θ_1 + θ_2 =\t $(abs(saf3.angle[][1]) + abs(saf3.angle[][2])) =\t $((abs(saf3.angle[][1]) + abs(saf3.angle[][2]))/π) π")
        println("Curve length:\t $(norm(sol3[2][(k+1):n, 1:k])).")
    else
        println("Solver failed.")
    end
    println("==========================W_perp initialization==========================\n")


    println("==========================St_log initialization==========================")

    println("Particular initialization Wp from the Stiefel geodesic that satisfies exp(S_{A,B,0}) = | Vk | Wp | is used.\n")

    StVp = zeros(n, n - k)
    StSkew, flag, iter, = stlog_hybrid_Newton_armijo(Wk, StVp; Init=init_guess_simple, NMLS_Set=NMLS_SET, Stop=terminator(100, 10000, 1e-8, 1e-6))

    if flag > 2
        println("Stlog failed!")
    else
        println("Stlog done!")
        StVp .= exp(StSkew)[:, (k+1):n]
    end

    if FullSolver
        sol4, rec4 = grhor_newton_full_core(Wk, StVp; Stop=terminator(MaxIter, 100000, AbsTol, 1e-6))
    else
        sol4, rec4 = grhor_newton_core(Wk, StVp; Stop=terminator(MaxIter, 100000, AbsTol, 1e-6))
    end
    println("A ≈ X?\t", sol4[2][1:k, 1:k] ≈ sol4[3], "\t Difference: ", maximum(abs.(sol4[2][1:k, 1:k] .- sol4[3])))

    saf4 = schurAngular_SkewSymm(Ref(sol4[2]))

    if maximum(abs.(sol4[2][1:k, 1:k] .- sol4[3])) < 1e-6
        println("Solution found.")
        display(sol4[2])
        # display(saf3)
        println("θ_1 + θ_2 =\t $(abs(saf4.angle[][1]) + abs(saf4.angle[][2])) =\t $((abs(saf4.angle[][1]) + abs(saf4.angle[][2]))/π) π")
        println("Curve length:\t $(norm(sol4[2][(k+1):n, 1:k])).")
    else
        println("Solver failed.")
    end
    println("==========================St_log initialization==========================\n")


    plt = plot(rec1[1], label="Random initial guess", yscale=:log10, ylabel="Objective value", xlabel="Number of iteration")
    plot!(rec2[1], label="V_perp initial guess", yscale=:log10)
    plot!(rec3[1], label="W_perp initial guess", yscale=:log10)
    plot!(rec4[1], label="St_log initial guess", yscale=:log10)

    display(plt)

    @printf "Method\t\t\tConvergence\tθ_1+θ_2\t\tLength\n"
    @printf "Random Initial\t\t%e\t%e π\t%e\n" maximum(abs.(sol1[2][1:k, 1:k] .- sol1[3])) ((abs(saf1.angle[][1]) + abs(saf1.angle[][2])) / π) norm(sol1[2][(k+1):n, 1:k])
    @printf "V_perp Initial\t\t%e\t%e π\t%e\n" maximum(abs.(sol2[2][1:k, 1:k] .- sol2[3])) ((abs(saf2.angle[][1]) + abs(saf2.angle[][2])) / π) norm(sol2[2][(k+1):n, 1:k])
    @printf "W_perp Initial\t\t%e\t%e π\t%e\n" maximum(abs.(sol3[2][1:k, 1:k] .- sol3[3])) ((abs(saf3.angle[][1]) + abs(saf3.angle[][2])) / π) norm(sol3[2][(k+1):n, 1:k])
    @printf "St_log Initial\t\t%e\t%e π\t%e\n" maximum(abs.(sol4[2][1:k, 1:k] .- sol4[3])) ((abs(saf4.angle[][1]) + abs(saf4.angle[][2])) / π) norm(sol4[2][(k+1):n, 1:k])

end

function test_grhor_newton_Wproblem(n, k, S::Matrix{Float64}; MaxIter=200, AbsTol=1e-8, InitSeed=1234, FullSolver=false, RandAttempt=10)
    println("Testing GrHor solver on connecting I_{n,k} and Wk, where Wk is generated by skew-symmetric S_{A,B,C} with exp(S_{A,B,C}) = | Wk | Wp |. This is a general problem and does not expect an available answer.")
    println("S_{A,B,C} is specified from inputs.")

    Q = exp(S)

    A = copy(S[1:k, 1:k])
    B = copy(S[(k+1):n, 1:k])
    C = copy(S[(k+1):n, (k+1):n])

    Vk = Q[:, 1:k] * exp(-A)
    Vp = Q[:, (k+1):n] * exp(-C)
    Wk = copy(Q[:, 1:k])
    Wp = copy(Q[:, (k+1):n])

    println("==========================Generating SO Geodesic==========================")

    saf = schurAngular_SkewSymm(Ref(S))
    # display(saf3)
    println("θ_1 + θ_2 =\t $(abs(saf.angle[][1]) + abs(saf.angle[][2])) =\t $((abs(saf.angle[][1]) + abs(saf.angle[][2]))/π) π")

    println("==========================Generating SO Geodesic==========================\n")

    sol1, rec1 = nothing, nothing
    sol2, rec2 = nothing, nothing
    sol3, rec3 = nothing, nothing
    sol4, rec4 = nothing, nothing

    saf1 = nothing
    saf2 = nothing
    saf3 = nothing
    saf4 = nothing

    rand_eng = MersenneTwister(InitSeed)




    println("==========================Random initialization==========================")
    println("Random special orthogonal completion to Vk is used.\n")
    attempt::Int = 1

    flags = Vector{Bool}(undef, 3)
    fill!(flags, false)

    while attempt <= RandAttempt
        println("Attempt $(attempt):\n")
        if FullSolver
            sol1, rec1 = grhor_newton_full_core(Wk; Stop=terminator(MaxIter, 100000, AbsTol, 1e-6), RandEng=rand_eng)
        else
            sol1, rec1 = grhor_newton_core(Wk; Stop=terminator(MaxIter, 100000, AbsTol, 1e-6), RandEng=rand_eng)
        end
        println("A ≈ X?\t", sol1[2][1:k, 1:k] ≈ sol1[3], "\t Difference: ", maximum(abs.(sol1[2][1:k, 1:k] .- sol1[3])))
        if maximum(abs.(sol1[2][1:k, 1:k] .- sol1[3])) < 1e-6
            println("Solution Found")
            display(sol1[2])
            saf1 = schurAngular_SkewSymm(Ref(sol1[2]))
            # display(saf1)
            println("θ_1 + θ_2 =\t $(abs(saf1.angle[][1]) + abs(saf1.angle[][2])) =\t $((abs(saf1.angle[][1]) + abs(saf1.angle[][2]))/π) π")
            println("Curve length:\t $(norm(sol1[2][(k+1):n, 1:k])).")
            flags[1] = true
            break
        else
            println("Solver failed.\n")
            attempt += 1
        end
    end

    println("==========================Random initialization==========================\n")


    println("==========================V_perp initialization==========================")
    println("Particular initialization Vp from exp(S_{A,B,C}) = | Wk | Vp exp(C) | is used.\n")
    if FullSolver
        sol2, rec2 = grhor_newton_full_core(Wk, Vp; Stop=terminator(MaxIter, 100000, AbsTol, 1e-6))
    else
        sol2, rec2 = grhor_newton_core(Wk, Vp; Stop=terminator(MaxIter, 100000, AbsTol, 1e-6))
    end
    println("A ≈ X?\t", sol2[2][1:k, 1:k] ≈ sol2[3], "\t Difference: ", maximum(abs.(sol2[2][1:k, 1:k] .- sol2[3])))
    if maximum(abs.(sol2[2][1:k, 1:k] .- sol2[3])) < 1e-6
        println("Solution found.")
        display(sol2[2])
        saf2 = schurAngular_SkewSymm(Ref(sol2[2]))
        # display(saf2)
        println("θ_1 + θ_2 =\t $(abs(saf2.angle[][1]) + abs(saf2.angle[][2])) =\t $((abs(saf2.angle[][1]) + abs(saf2.angle[][2]))/π) π")
        println("Curve length:\t $(norm(sol2[2][(k+1):n, 1:k])).")
        flags[2] = true
    else
        println("Solver failed.")
    end
    println("==========================V_perp initialization==========================\n")


    println("==========================W_perp initialization==========================")

    println("Particular initialization Wp from exp(S_{A,B,C}) = | Wk | Wp | is used.\n")

    if FullSolver
        sol3, rec3 = grhor_newton_full_core(Wk, Wp; Stop=terminator(MaxIter, 100000, AbsTol, 1e-6))
    else
        sol3, rec3 = grhor_newton_core(Wk, Wp; Stop=terminator(MaxIter, 100000, AbsTol, 1e-6))
    end
    println("A ≈ X?\t", sol3[2][1:k, 1:k] ≈ sol3[3], "\t Difference: ", maximum(abs.(sol3[2][1:k, 1:k] .- sol3[3])))
    if maximum(abs.(sol3[2][1:k, 1:k] .- sol3[3])) < 1e-6
        println("Solution found.")
        display(sol3[2])
        saf3 = schurAngular_SkewSymm(Ref(sol3[2]))
        # display(saf3)
        println("θ_1 + θ_2 =\t $(abs(saf3.angle[][1]) + abs(saf3.angle[][2])) =\t $((abs(saf3.angle[][1]) + abs(saf3.angle[][2]))/π) π")
        println("Curve length:\t $(norm(sol3[2][(k+1):n, 1:k])).")
    else
        println("Solver failed.")
    end
    println("==========================W_perp initialization==========================\n")


    println("==========================St_log initialization==========================")

    println("Particular initialization Wp from the Stiefel geodesic that satisfies exp(S_{A,B,0}) = | Vk | Wp | is used.\n")

    StVp = zeros(n, n - k)
    StSkew, flag, iter, = stlog_hybrid_Newton_armijo(Wk, StVp; Init=init_guess_simple, NMLS_Set=NMLS_SET, Stop=terminator(100, 10000, 1e-8, 1e-6))

    if flag > 2
        println("Stlog failed!")
    else
        println("Stlog done!")
        StVp .= exp(StSkew)[:, (k+1):n]
    end

    if FullSolver
        sol4, rec4 = grhor_newton_full_core(Wk, StVp; Stop=terminator(MaxIter, 100000, AbsTol, 1e-6))
    else
        sol4, rec4 = grhor_newton_core(Wk, StVp; Stop=terminator(MaxIter, 100000, AbsTol, 1e-6))
    end
    println("A ≈ X?\t", sol4[2][1:k, 1:k] ≈ sol4[3], "\t Difference: ", maximum(abs.(sol4[2][1:k, 1:k] .- sol4[3])))
    if maximum(abs.(sol4[2][1:k, 1:k] .- sol4[3])) < 1e-6
        println("Solution found.")
        display(sol4[2])
        saf4 = schurAngular_SkewSymm(Ref(sol4[2]))
        # display(saf3)
        println("θ_1 + θ_2 =\t $(abs(saf4.angle[][1]) + abs(saf4.angle[][2])) =\t $((abs(saf4.angle[][1]) + abs(saf4.angle[][2]))/π) π")
        println("Curve length:\t $(norm(sol4[2][(k+1):n, 1:k])).")
    else
        println("Solver failed.")
    end
    println("==========================St_log initialization==========================\n")


    plt = plot(rec1[1], label="Random initial guess", yscale=:log10, ylabel="Objective value", xlabel="Number of iteration")
    plot!(rec2[1], label="V_perp initial guess", yscale=:log10)
    plot!(rec3[1], label="W_perp initial guess", yscale=:log10)
    plot!(rec4[1], label="St_log initial guess", yscale=:log10)
    display(plt)

    @printf "Method\t\t\tConvergence\tθ_1+θ_2\t\tLength\n"
    @printf "Random Initial\t\t%e\t%e π\t%e\n" maximum(abs.(sol1[2][1:k, 1:k] .- sol1[3])) ((abs(saf1.angle[][1]) + abs(saf1.angle[][2])) / π) norm(sol1[2][(k+1):n, 1:k])
    @printf "V_perp Initial\t\t%e\t%e π\t%e\n" maximum(abs.(sol2[2][1:k, 1:k] .- sol2[3])) ((abs(saf2.angle[][1]) + abs(saf2.angle[][2])) / π) norm(sol2[2][(k+1):n, 1:k])
    @printf "W_perp Initial\t\t%e\t%e π\t%e\n" maximum(abs.(sol3[2][1:k, 1:k] .- sol3[3])) ((abs(saf3.angle[][1]) + abs(saf3.angle[][2])) / π) norm(sol3[2][(k+1):n, 1:k])
    @printf "St_log Initial\t\t%e\t%e π\t%e\n" maximum(abs.(sol4[2][1:k, 1:k] .- sol4[3])) ((abs(saf4.angle[][1]) + abs(saf4.angle[][2])) / π) norm(sol4[2][(k+1):n, 1:k])

end

function test_grhor_newton_full(n, k, scale::Float64=0.9π; seed=1234)
    rand_eng = MersenneTwister(seed)
    S = rand(rand_eng, n, n)
    S .-= S'
    S .*= scale / opnorm(S)
    Q = exp(S)
    # V = Q[:, 1:k]*exp(-S[1:k, 1:k])
    V = Q[:, 1:k]


    println("==========================Generating Horizontal Curve==========================")

    saf = schurAngular_SkewSymm(Ref(S))
    # display(saf3)
    println("θ_1 + θ_2 =\t $(abs(saf.angle[][1]) + abs(saf.angle[][2])) =\t $((abs(saf.angle[][1]) + abs(saf.angle[][2]))/π) π")

    println("Curve length:\t $(norm(S[(k+1):n, 1:k])).")

    println("==========================Generating Horizontal Curve==========================\n")


    try_cnt = 0

    while try_cnt < 20
        sol = grhor_newton_full_core(V)

        println("A ≈ X?\t", sol[2][1:k, 1:k] ≈ sol[3], "\t Difference: ", maximum(abs.(sol[2][1:k, 1:k] .- sol[3])))
        saf = schurAngular_SkewSymm(Ref(sol[2]))
        # display(saf)
        println("θ_1 + θ_2 =\t $(abs(saf.angle[][1]) + abs(saf.angle[][2])) =\t $((abs(saf.angle[][1]) + abs(saf.angle[][2]))/π) π")

        println("Curve length:\t $(norm(sol[2][(k+1):n, 1:k]))).\n")

        if sol[2][1:k, 1:k] ≈ sol[3]
            display(plot(sol[4], yscale=:log10))
            break
        end
        try_cnt += 1
    end

    # display(norm.(sol[9]))

    return sol
end

function test_grhor_newton_full(n, k, S::Matrix{Float64})
    Q = exp(S)

    A = copy(S[1:k, 1:k])
    B = copy(S[(k+1):n, 1:k])
    C = copy(S[(k+1):n, (k+1):n])

    Vk = Q[:, 1:k] * exp(-A)
    Vp = Q[:, (k+1):n] * exp(-C)

    Up = Q[:, (k+1):n]

    println("==========================Generating Horizontal Curve==========================")

    saf = schurAngular_SkewSymm(Ref(S))
    # display(saf3)
    println("θ_1 + θ_2 =\t $(abs(saf.angle[][1]) + abs(saf.angle[][2])) =\t $((abs(saf.angle[][1]) + abs(saf.angle[][2]))/π) π")

    println("Curve length:\t $(norm(B)).")

    println("==========================Generating Horizontal Curve==========================\n")



    println("==========================Default initialization==========================")
    sol1 = grhor_newton_full_core(Vk)
    println("A ≈ X?\t", sol1[2][1:k, 1:k] ≈ sol1[3], "\t Difference: ", maximum(abs.(sol1[2][1:k, 1:k] .- sol1[3])))
    saf1 = schurAngular_SkewSymm(Ref(sol1[2]))
    # display(saf1)
    println("θ_1 + θ_2 =\t $(abs(saf1.angle[][1]) + abs(saf1.angle[][2])) =\t $((abs(saf1.angle[][1]) + abs(saf1.angle[][2]))/π) π")
    println("Curve length:\t $(norm(sol1[2][(k+1):n, 1:k])).")

    println("==========================Default initialization==========================\n")


    println("==========================Good initialization==========================")
    sol2 = grhor_newton_full_core(Vk, Vp)
    println("A ≈ X?\t", sol2[2][1:k, 1:k] ≈ sol2[3], "\t Difference: ", maximum(abs.(sol2[2][1:k, 1:k] .- sol2[3])))
    saf2 = schurAngular_SkewSymm(Ref(sol2[2]))
    # display(saf2)
    println("θ_1 + θ_2 =\t $(abs(saf2.angle[][1]) + abs(saf2.angle[][2])) =\t $((abs(saf2.angle[][1]) + abs(saf2.angle[][2]))/π) π")
    println("Curve length:\t $(norm(sol2[2][(k+1):n, 1:k])).")
    println("==========================Good initialization==========================\n")


    println("==========================Perfect initialization==========================")


    sol3 = grhor_newton_full_core(Vk, Up)
    println("A ≈ X?\t", sol3[2][1:k, 1:k] ≈ sol3[3], "\t Difference: ", maximum(abs.(sol3[2][1:k, 1:k] .- sol3[3])))
    saf3 = schurAngular_SkewSymm(Ref(sol3[2]))
    # display(saf3)
    println("θ_1 + θ_2 =\t $(abs(saf3.angle[][1]) + abs(saf3.angle[][2])) =\t $((abs(saf3.angle[][1]) + abs(saf3.angle[][2]))/π) π")
    println("Curve length:\t $(norm(sol3[2][(k+1):n, 1:k])).")
    println("==========================Perfect initialization==========================\n")

    plt = plot(sol1[4], label="Default initial guess", yscale=:log10)
    plot!(sol2[4], label="Good initial guess", yscale=:log10)
    plot!(sol3[4], label="Perfect initial guess", yscale=:log10)

    display(plt)

end