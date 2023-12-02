include("stlog_compact_solver.jl")


global STLOG_RESTART_THRESHOLD = 0.1
global STLOG_ENABLE_NEARLOG = false
global STLOG_HYBRID_BCH_MAXITER = 6
global STLOG_HYBRID_BCH_ABSTOL = 1e-7

_STLOG_TEST_SOLVER_STOP = terminator(100, 100000, 1e-7, 1e-7)
_STLOG_TEST_NMLS_SET = NMLS_Paras(0.1, 20.0, 0.9, 0.3, 0)

global bad_set_newton = (0, 0, 9527, 0.0)

function stlog_BCH_analysis!(M::Ref{Matrix{Float64}}, Uk::Ref{Matrix{Float64}}, Up::Ref{Matrix{Float64}}, wsp::WSP;
    Stop=terminator(300, 1000, 1e-8, 1e-8), Init=nothing, order=1, builtin_log=false)

    elapsed_time::Int = 0
    direction_time::Int = 0
    update_time::Int = 0

    Record_ObjVal = Vector{Float64}(undef, Stop.MaxIter)
    Record_Direct = Vector{Matrix{Float64}}(undef, Stop.MaxIter)
    Record_PointU = Vector{Matrix{Float64}}(undef, Stop.MaxIter)
    Record_PointS = Vector{Matrix{Float64}}(undef, Stop.MaxIter)


    MatUk = Uk[]
    MatUp = Up[]
    MatM = M[]

    n::Int, k::Int = size(MatUk)

    MatU::Matrix{Float64} = wsp[1]
    MatB::Matrix{Float64} = wsp[2]
    MatC::Matrix{Float64} = wsp[3]
    MatR::Matrix{Float64} = wsp[4]
    MatZ::Matrix{Float64} = wsp[5]

    M_saf::SAFactor = wsp[6]

    wsp_saf_n::WSP = wsp[7]
    wsp_stlog_ret::WSP = wsp[8]

    MateZ = wsp_stlog_ret[4]
    eZ = wsp_stlog_ret(4)


    U = Ref(MatU)
    Up = Ref(MatUp)
    M = Ref(MatM)
    B = Ref(MatB)
    C = Ref(MatC)
    R = Ref(MatR)
    Z = Ref(MatZ)

    iter::Int = 0
    abserr::Float64 = -1.0
    objval::Float64 = -1.0
    result_flag::Int = 0

    elapsed = @timed begin

        copyto!(view(MatU, :, 1:k), MatUk)
        if Init !== nothing
            copyto!(MatUp, Init(MatUk))
        end
        copyto!(view(MatU, :, (k+1):n), MatUp)

        update = @timed begin

            if builtin_log
                copyto!(MatM, real.(log(MatU)))
                getSkewSymm!(M)
            else
                log_SpecOrth!(M, M_saf, U, wsp_saf_n; order=false, regular=false)
            end
        end
        update_time += Int(round((update.time - update.gctime) * 1e9))


        direction = @timed begin

            if order == 1
                copyto!(MatZ, view(MatM, (k+1):n, (k+1):n))
                lmul!(-1, MatZ)
            elseif order == 5
                stlog_BCH5_direction_lyap!(Z, M, B, C, R)
            else
                throw("BCH order not supported! Please choose from 1 and 5.")
            end
        end

        direction_time += Int(round((direction.time - direction.gctime) * 1e9))

        iter = 1
        objval = stlog_cost(M, k)
        abserr = sqrt(2 * objval)

        Record_PointU[iter] = copy(MatU)
        Record_PointS[iter] = copy(MatM)
        Record_Direct[iter] = copy(MatZ)
        Record_ObjVal[iter] = objval


    end

    elapsed_time += Int(round((elapsed.time - elapsed.gctime) * 1e9))


    while result_flag == 0
        elapsed = @timed begin
            update = @timed begin

                if builtin_log
                    _stlog_ret_exp!(eZ, Z)
                    mul!(view(MatU, :, (k+1):n), MatUp, MateZ)
                    copyto!(MatM, real.(log(MatU)))
                    getSkewSymm!(M)
                    copyto!(MatUp, view(MatU, :, (k+1):n))
                else
                    _stlog_ret_principal!(M, M_saf, Up, Uk, Z, n, k, wsp_stlog_ret)
                end
            end
            update_time += Int(round((update.time - update.gctime) * 1e9))

            direction = @timed begin
                if order == 1
                    copyto!(MatZ, view(MatM, (k+1):n, (k+1):n))
                    lmul!(-1, MatZ)
                elseif order == 5
                    stlog_BCH5_direction_lyap!(Z, M, B, C, R)
                else
                    throw("BCH order not supported! Please choose from 1 and 5.")
                end
            end

            direction_time += Int(round((direction.time - direction.gctime) * 1e9))


            iter += 1
            objval = stlog_cost(M, k)
            abserr = sqrt(2 * objval)

            Record_PointU[iter] = copy(MatU)
            Record_PointS[iter] = copy(MatM)
            Record_Direct[iter] = copy(MatZ)
            Record_ObjVal[iter] = objval

        end

        elapsed_time += Int(round((elapsed.time - elapsed.gctime) * 1e9))


        result_flag = check_termination_val(abserr, nothing, nothing, elapsed_time, nothing, iter, Stop)
    end

    return result_flag, iter, (elapsed_time, update_time, direction_time), (copy(Record_PointU[1:iter]), copy(Record_PointS[1:iter]), copy(Record_Direct[1:iter]), copy(Record_ObjVal[1:iter]))
end

function stlog_Newton_analysis!(M::Ref{Matrix{Float64}}, Uk::Ref{Matrix{Float64}}, Up::Ref{Matrix{Float64}}, wsp::WSP;
    Stop::terminator=terminator(300, 10000, 1e-8, 1e-6), Solver_Stop::terminator=terminator(200, Int(5e6), 1e-12, 1e-9),
    Init=nothing, NMLS_Set=NMLS_Paras(0.1, 20.0, 0.9, 0.3, 0))

    elapsed_time::Int = 0
    direction_time::Int = 0
    update_time::Int = 0
    action_time::Int = 0

    Record_ObjVal = Vector{Float64}(undef, Stop.MaxIter)
    Record_LineSα = Vector{Float64}(undef, Stop.MaxIter)
    Record_PointU = Vector{Matrix{Float64}}(undef, Stop.MaxIter)
    Record_PointS = Vector{Matrix{Float64}}(undef, Stop.MaxIter)
    Record_PointΔ = Vector{Matrix{Float64}}(undef, Stop.MaxIter)
    Record_PointZ = Vector{Matrix{Float64}}(undef, Stop.MaxIter)



    elapsed = @timed begin


        MatM = M[]
        MatUk = Uk[]
        MatUp = Up[]

        n::Int, k::Int = size(MatUk)
        m::Int = n - k

        MatU::Matrix{Float64} = wsp[1]
        MatUp_new::Matrix{Float64} = wsp[2]
        MatM_new::Matrix{Float64} = wsp[3]
        MatΔ::Matrix{Float64} = wsp[4]
        MatB::Matrix{Float64} = wsp[5]
        MatC::Matrix{Float64} = wsp[6]
        MatR::Matrix{Float64} = wsp[7]
        MatZ::Matrix{Float64} = wsp[8]
        MatαZ::Matrix{Float64} = wsp[9]

        M_saf::SAFactor = wsp[10]
        M_sys::dexp_SkewSymm_system = wsp[11]

        blk_it_m::STRICT_LOWER_ITERATOR = wsp[12]
        blk_it_nm::STRICT_LOWER_ITERATOR = wsp[13]
        blk_it_n::STRICT_LOWER_ITERATOR = wsp[14]

        wsp_cong_nm::WSP = wsp[15]
        wsp_cong_n::WSP = wsp[16]
        wsp_saf_n::WSP = wsp[17]
        wsp_stlog_ret::WSP = wsp[18]
        wsp_stlog_newton_gmres::WSP = wsp[19]

        U = Ref(MatU)
        Up = Ref(MatUp)
        Up_new = Ref(MatUp_new)
        M_new = Ref(MatM_new)
        Δ = Ref(MatΔ)
        Z = Ref(MatZ)
        αZ = Ref(MatαZ)

        cost_record = zeros(Stop.MaxIter)
        fval = Ref(cost_record)


        M_ang = M_saf.angle
        VecM_ang = getAngle(M_saf)

        α::Float64 = 1.0
        sq::Float64 = 1.0
        slope::Float64 = -1.0
        α_upper_bound::Float64 = 0.0
        MatΔ_2_norm::Float64 = 0.0

        iter::Int = 0
        objval::Float64 = -1.0
        abserr::Float64 = -1.0
        result_flag::Int = 0


        copyto!(view(MatU, :, 1:k), MatUk)
        if Init !== nothing
            copyto!(MatUp, Init(MatUk))
        end
        copyto!(view(MatU, :, (k+1):n), MatUp)


        update = @timed begin
            log_SpecOrth!(M, M_saf, U, wsp_saf_n; order=true, regular=true)
        end

        update_time += Int(round((update.time - update.gctime) * 1e9))


        iter = 1
        objval = stlog_cost(M, k)
        abserr = sqrt(2 * objval)
        cost_record[iter] = objval

        Record_PointU[iter] = copy(MatU)
        Record_PointS[iter] = copy(MatM)
        Record_ObjVal[iter] = objval


    end

    elapsed_time += Int(round((elapsed.time - elapsed.gctime) * 1e9))

    while result_flag == 0
        elapsed = @timed begin

            if VecM_ang[1] + VecM_ang[2] > 2π
                throw("The active point is beyond the restricted manifold, with angles $(VecM_ang[1]) and $(VecM_ang[2]).")
            end

            if VecM_ang[1] > π + STLOG_RESTART_THRESHOLD
                update = @timed begin
                    log_SpecOrth!(M, M_saf, U, wsp_saf_n; order=true, regular=true)
                end
                update_time += Int(round((update.time - update.gctime) * 1e9))

                objval = stlog_cost(M, k)
                abserr = sqrt(2 * objval)
            else
                direction = @timed begin
                    compute_dexp_SkewSymm_both_system!(M_sys, M_ang)
                    newton_flag, newton_time = stlog_newton_descent_backward!(Z, M, M_sys, M_saf, n, k, div((n - k) * (n - k - 1), 2), blk_it_nm, blk_it_m, blk_it_n, wsp_stlog_newton_gmres; Stop=Solver_Stop)
                    getSkewSymm!(Z)

                    action_time += newton_time[2]

                    # display(MatM)

                    # display(MatZ)

                    if newton_flag > 2
                        # Newton direction failed
                        @printf "\nRestart for bad Newton Direction at step: \t %i \n" iter
                        try
                            # Try to construct escaping direction from the gmres result.
                            lmul!(4π / opnorm(MatZ, 2), MatZ)
                        catch
                            copyto!(MatZ, view(MatM, (k+1):n, (k+1):n))
                        end
                    end

                    # Δ = dexp_{S_{A,B,C}}^{-1}[S_{0, 0, Z}]

                    action = @timed begin
                        cong_dense!(Δ, M_saf.vector, k, Z, 0, m, wsp_cong_nm; trans=true)
                        dexp_SkewSymm!(Δ, Δ, M_sys, M_saf, blk_it_n, wsp_cong_n; inv=true, cong=false)
                        cong_dense!(Δ, M_saf.vector, Δ, wsp_cong_n; trans=false)
                        getSkewSymm!(Δ)
                    end

                    action_time += Int(round((action.time - action.gctime) * 1e9))
                    # display(MatΔ)
                end

                direction_time += Int(round((direction.time - direction.gctime) * 1e9))


                Record_PointZ[iter] = copy(MatZ)
                Record_PointΔ[iter] = copy(MatΔ)

                try
                    MatΔ_2_norm = opnorm(MatΔ, 2)
                    α_upper_bound = (2π - VecM_ang[1] - VecM_ang[2]) / MatΔ_2_norm
                catch
                    @printf "Fail to obtain correct Δ_S in Dexp_S[Δ_S] = exp(S)Δ_Q.\n"

                    display(MatZ)
                    display(MatΔ)

                    return 5, iter, (elapsed_time, update_time, direction_time, action_time), (copy(Record_PointU[1:iter]), copy(Record_PointS[1:iter]), copy(Record_PointΔ[1:iter]), copy(Record_PointZ[1:iter]), copy(Record_LineSα[1:iter]), copy(Record_ObjVal[1:iter]))

                end

                # MatΔ_2_norm = opnorm(MatΔ, 2)
                # α_upper_bound = (2π - VecM_ang[1] - VecM_ang[2]) / MatΔ_2_norm

                lmul!(-1, MatZ)

                if newton_flag > 2
                    slope = -stlog_dcost(M, Δ, k)
                else
                    slope = -2 * objval
                end

                if iter == 1
                    α = 1
                    sq = inner_skew!(Z, Z)
                    copyto!(MatαZ, MatZ)
                else
                    α, sq = hor_BB_step!(α, sq, Z, αZ)
                end

                # @printf "Successful Newton direcion? \t%s,\t |Δ|_2:\t %.16f, \t(2π - θ_1 - θ_2) / 2:\t %.16f, \tBB step α:\t %.16f, \tBound of α:\t %.16f\n" (newton_flag <= 2 ? "True" : "False") MatΔ_2_norm (2π - VecM_ang[1] - VecM_ang[2]) * 0.5 α α_upper_bound



                nearlog_flag = α_upper_bound < 0.1 && MatΔ_2_norm < 2π && STLOG_ENABLE_NEARLOG


                try
                    update = @timed begin
                        if MatΔ_2_norm < 2π

                            α, objval, = stlog_UpZ_NMLS!(fval, slope, α, Z, αZ, Uk, Up, Up_new, M, M_new, M_saf, M_sys, wsp_stlog_ret; paras=NMLS_Set, f_len=iter, nearlog=nearlog_flag, bound=α_upper_bound, fail_step=1.0)
                        else
                            @printf "\nRestart for huge Newton Direction at step: \t %i \n" iter

                            α = π / MatΔ_2_norm
                            copyto!(MatαZ, MatZ)
                            lmul!(α, MatαZ)

                            _stlog_ret_principal!(M, M_saf, Up, Uk, αZ, n, k, wsp_stlog_ret)
                        end
                    end

                    update_time += Int(round((update.time - update.gctime) * 1e9))
                catch
                    @printf "Fail to perform line search.\n"
                    return 5, iter, (elapsed_time, update_time, direction_time, action_time), (copy(Record_PointU[1:iter]), copy(Record_PointS[1:iter]), copy(Record_PointΔ[1:iter]), copy(Record_PointZ[1:iter]), copy(Record_LineSα[1:iter]), copy(Record_ObjVal[1:iter]))

                end

                Record_LineSα[iter] = α
            end

            iter += 1
            objval = stlog_cost(M, k)
            abserr = sqrt(2 * objval)
            cost_record[iter] = objval

            Record_PointU[iter] = hcat(MatUk, MatUp)
            Record_PointS[iter] = copy(MatM)
            Record_ObjVal[iter] = objval

        end

        elapsed_time += Int(round((elapsed.time - elapsed.gctime) * 1e9))





        result_flag = check_termination_val(abserr, nothing, nothing, elapsed_time, nothing, iter, Stop)


    end

    # display(plot(cost_record[1:iter], yscale=:log10))
    # println(direction_time / elapsed_time)

    # @printf "Time profiles: \t GMRES solver / all time = \t %.16f, \t Action / All time  = \t %.16f, \t Action / GMRES solver = \t %.16f\n" direction_time / elapsed_time action_time / elapsed_time action_time / direction_time

    return result_flag, iter, (elapsed_time, update_time, direction_time, action_time), (copy(Record_PointU[1:iter]), copy(Record_PointS[1:iter]), copy(Record_PointΔ[1:iter]), copy(Record_PointZ[1:iter]), copy(Record_LineSα[1:iter]), copy(Record_ObjVal[1:iter]))

end

function stlog_Record_Analysis_Newton(record)

    Record_PointU, Record_PointS, Record_PointΔ, Record_PointZ, Record_LineSα, Record_ObjVal = record

    iter = length(Record_ObjVal) - 1
    n::Int = size(Record_PointU[1], 1)
    m::Int = size(Record_PointZ[1], 1)
    k::Int = n - m

    Z_2norm = zeros(iter)
    Δ_2norm = zeros(iter)
    SQ_diff = zeros(iter)
    ZΔ_diff = zeros(iter)
    CΔ_diff = zeros(iter)


    MatM = zeros(n, n)
    MatU = zeros(n, n)
    MatΔQ = zeros(n, n)
    MatΔS = zeros(n, n)

    M = Ref(MatM)
    U = Ref(MatU)
    ΔQ = Ref(MatΔQ)
    ΔS = Ref(MatΔS)

    M_saf = SAFactor(n)
    M_sys = dexp_SkewSymm_system(n)

    wsp_saf_n = get_wsp_saf(n)
    wsp_cong_n = get_wsp_cong(n)

    blk_it_n = STRICT_LOWER_ITERATOR(n, lower_blk_traversal)



    for ind in 1:iter



        copy!(MatU, Record_PointU[ind])
        copy!(MatM, Record_PointS[ind])
        copy!(view(MatΔQ, (k+1):n, (k+1):n), Record_PointZ[ind])

        schurAngular_SkewSymm!(M_saf, M, wsp_saf_n; regular=true, order=true)
        compute_dexp_SkewSymm_both_system!(M_sys, M_saf.angle)

        dexp_SkewSymm!(ΔS, ΔQ, M_sys, M_saf, blk_it_n, wsp_cong_n; inv=true, cong=true)


        Z_2norm[ind] = opnorm(Record_PointZ[ind])
        Δ_2norm[ind] = opnorm(Record_PointΔ[ind])
        SQ_diff[ind] = norm(exp(MatM) .- MatU)
        ZΔ_diff[ind] = norm(MatΔS .- Record_PointΔ[ind])
        CΔ_diff[ind] = norm(MatM[(k+1):n, (k+1):n] .- MatΔS[(k+1):n, (k+1):n])

    end

    plt_1 = plot(1:iter, [SQ_diff ZΔ_diff CΔ_diff], labels=["Matrix Logarithm: S = log(Q)" "Direction Derivative: Dexp[Δ_S] = QΔ_Q" "Mat-free GMRES: I'_{n,k}⋅Δ_S⋅I_{n,k} = C"], ylabel="Errors in Newton Algorithm", xlabel="Iterations in Newton Algorithm", yscale=:log10, legend=:bottomleft)

    plt_2 = plot(1:iter, [Δ_2norm Z_2norm Record_LineSα[1:iter]], ylabel="Some Quantities in Newton Algorithm", xlabel="Iterations in Newton Algorithm", labels=["Δ_Q 2-norm" "Δ_S 2-norm" "Stepsize"], legend=:bottomleft)

    plot!(twinx(), 1:iter, Record_ObjVal[1:iter], labels="Objective", yscale=:log10, legend=:topright, linestyle=:dash, color=:red)

    display(plt_1)
    display(plt_2)



end

function test_newton_analysis(n, k, σ;
    AbsTol=1e-7, MaxIter=2000, MaxTime=10, seed=rand(1:10000),
    Solver_Stop=_STLOG_TEST_SOLVER_STOP,
    NMLS_Set=_STLOG_TEST_NMLS_SET)

    Stop = terminator(MaxIter, Int(round(MaxTime * 1e9)), AbsTol, AbsTol)

    eng = MersenneTwister(seed)
    MatS = rand(eng, n, n)
    fill!(view(MatS, (k+1):n, (k+1):n), 0.0)
    MatS .-= MatS'
    lmul!(σ, MatS)

    MatU = exp(MatS)

    MatM_Newton = zeros(n, n)
    MatUk_Newton = copy(view(MatU, :, 1:k))
    MatUp_Newton = copy(view(MatU, :, (k+1):n))

    M_Newton = Ref(MatM_Newton)
    Uk_Newton = Ref(MatUk_Newton)
    Up_Newton = Ref(MatUp_Newton)
    wsp_Newton = get_wsp_stlog_Newton(n, k)

    @printf "Seed: \t %i.\n" seed

    flag, iter, profile, record = stlog_Newton_analysis!(M_Newton, Uk_Newton, Up_Newton, wsp_Newton; Stop=Stop, Solver_Stop=Solver_Stop, Init=stlog_init_guess_simple, NMLS_Set=NMLS_Set)

    # stlog_Record_Analysis_Newton(record)

    return flag, iter, profile, record
end

function stlog_simple_newton_analysis(n, k, σ_range, seed)
    for s in σ_range
        @printf "σ: \t %.16f,\t" s
        flag, iter, profile, record = test_newton_analysis(n, k, s; seed=seed)
        if iter > 10 || flag > 2
            if flag > 2
                global bad_set_newton = (n, k, seed, s)
            end
            @printf "Plots made.\n"
            stlog_Record_Analysis_Newton(record)
        end
    end
end

stlog_simple_test(n, k, scatter_num, seed; show_plts=true, save_plts=false) = test_stlog_profile(n, k, range(0.1, 1.5π, scatter_num);
    AbsTol=1e-7, MaxIter=2000, MaxTime=10,
    save_plts=save_plts, show_plts=show_plts, loops=3, seed=seed,
    Solver_Stop=_STLOG_TEST_SOLVER_STOP,
    NMLS_Set=_STLOG_TEST_NMLS_SET)