include("stlog_compact_geometry.jl")
include("stlog_compact_linesearch.jl")
include("stlog_compact_descent.jl")
include("stlog_compact_init_guess.jl")

global STLOG_RESTART_THRESHOLD = 0.5
global STLOG_ENABLE_NEARLOG = true
global STLOG_HYBRID_BCH_MAXITER = 8
global STLOG_HYBRID_BCH_ABSTOL = 1e-4
global STLOG_HYBRID_BCH_SHUTDOWN = 5




@inline get_wsp_stlog_BCH(n, k) = WSP(Matrix{Float64}(undef, n, n), Matrix{Float64}(undef, n - k, k), Matrix{Float64}(undef, n - k, n - k), Matrix{Float64}(undef, n - k, n - k), Matrix{Float64}(undef, n - k, n - k), SAFactor(n), get_wsp_saf(n), get_wsp_stlog_ret(n, k))

get_wsp_stlog_Newton(n, k)::WSP = WSP(Matrix{Float64}(undef, n, n), Matrix{Float64}(undef, n, n - k), Matrix{Float64}(undef, n, n), Matrix{Float64}(undef, n, n), Matrix{Float64}(undef, n - k, k), Matrix{Float64}(undef, n - k, n - k), Matrix{Float64}(undef, n - k, n - k), Matrix{Float64}(undef, n - k, n - k), Matrix{Float64}(undef, n - k, n - k), SAFactor(n), dexp_SkewSymm_system(n), STRICT_LOWER_ITERATOR(n - k, lower_blk_traversal), STRICT_LOWER_ITERATOR(n, k, n, lower_blk_traversal), STRICT_LOWER_ITERATOR(n, lower_blk_traversal), get_wsp_cong(n, n - k), get_wsp_cong(n), get_wsp_saf(n), get_wsp_stlog_ret(n, k), get_wsp_stlog_newton_gmres(n, k, div((n - k) * (n - k - 1), 2), div((n - k) * (n - k - 1), 2)))

function stlog_BCH_2k!(M::Ref{Matrix{Float64}}, Uk::Ref{Matrix{Float64}}, Up::Ref{Matrix{Float64}}, wsp::WSP;
    Stop=terminator(300, 1000, 1e-8, 1e-8), Init=nothing, order=1, builtin_log=false)

    elapsed_time::Int = 0
    direction_time::Int = 0
    update_time::Int = 0

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

        end

        elapsed_time += Int(round((elapsed.time - elapsed.gctime) * 1e9))


        result_flag = check_termination_val(abserr, nothing, nothing, elapsed_time, nothing, iter, Stop)
    end

    return result_flag, iter, (elapsed_time, update_time, direction_time)
end



function stlog_Newton!(M::Ref{Matrix{Float64}}, Uk::Ref{Matrix{Float64}}, Up::Ref{Matrix{Float64}}, wsp::WSP;
    Stop::terminator=terminator(300, 10000, 1e-8, 1e-6), Solver_Stop::terminator=terminator(200, Int(5e6), 1e-12, 1e-9),
    Init=nothing, NMLS_Set=NMLS_Paras(0.1, 20.0, 0.9, 0.3, 0))

    elapsed_time::Int = 0
    direction_time::Int = 0
    update_time::Int = 0
    action_time::Int = 0


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

                    try
                        MatΔ_2_norm = opnorm(MatΔ, 2)
                        α_upper_bound = (2π - VecM_ang[1] - VecM_ang[2]) / MatΔ_2_norm
                    catch
                        copyto!(MatZ, view(MatM, (k+1):n, (k+1):n))
                        action = @timed begin
                            cong_dense!(Δ, M_saf.vector, k, Z, 0, m, wsp_cong_nm; trans=true)
                            dexp_SkewSymm!(Δ, Δ, M_sys, M_saf, blk_it_n, wsp_cong_n; inv=true, cong=false)
                            cong_dense!(Δ, M_saf.vector, Δ, wsp_cong_n; trans=false)
                            getSkewSymm!(Δ)
                        end

                        action_time += Int(round((action.time - action.gctime) * 1e9))

                        MatΔ_2_norm = opnorm(MatΔ, 2)
                        α_upper_bound = (2π - VecM_ang[1] - VecM_ang[2]) / MatΔ_2_norm
                        newton_flag = 5
                    end
                end

                direction_time += Int(round((direction.time - direction.gctime) * 1e9))

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


                update = @timed begin

                    if MatΔ_2_norm < 2π
                        α, objval, = stlog_UpZ_NMLS!(fval, slope, α, Z, αZ, Uk, Up, Up_new, M, M_new, M_saf, M_sys, wsp_stlog_ret; paras=NMLS_Set, f_len=iter, nearlog=nearlog_flag, bound=α_upper_bound, fail_step=1.0)
                    else
                        α = π / MatΔ_2_norm
                        copyto!(MatαZ, MatZ)
                        lmul!(α, MatαZ)

                        _stlog_ret_principal!(M, M_saf, Up, Uk, αZ, n, k, wsp_stlog_ret)
                    end
                end

                update_time += Int(round((update.time - update.gctime) * 1e9))


            end

            iter += 1
            objval = stlog_cost(M, k)
            abserr = sqrt(2 * objval)
            cost_record[iter] = objval

        end

        elapsed_time += Int(round((elapsed.time - elapsed.gctime) * 1e9))





        result_flag = check_termination_val(abserr, nothing, nothing, elapsed_time, nothing, iter, Stop)


    end

    # display(plot(cost_record[1:iter], yscale=:log10))
    # println(direction_time / elapsed_time)

    # @printf "Time profiles: \t GMRES solver / all time = \t %.16f, \t Action / All time  = \t %.16f, \t Action / GMRES solver = \t %.16f\n" direction_time / elapsed_time action_time / elapsed_time action_time / direction_time

    return result_flag, iter, (elapsed_time, update_time, direction_time, action_time)

end


function stlog_hybrid!(M::Ref{Matrix{Float64}}, Uk::Ref{Matrix{Float64}}, Up::Ref{Matrix{Float64}}, wsp::WSP;
    Stop::terminator=terminator(300, 10000, 1e-8, 1e-6), Solver_Stop::terminator=terminator(200, Int(5e6), 1e-12, 1e-9),
    Init=nothing, NMLS_Set=NMLS_Paras(0.1, 20.0, 0.9, 0.3, 0))

    elapsed_time::Int = 0
    direction_time::Int = 0
    update_time::Int = 0
    action_time::Int = 0


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

        B = Ref(MatB)
        C = Ref(MatC)
        R = Ref(MatR)

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
        bch_iter_cnt::Int = 0
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

    end

    elapsed_time += Int(round((elapsed.time - elapsed.gctime) * 1e9))


    while result_flag == 0
        elapsed = @timed begin

            if (0 <= bch_iter_cnt < STLOG_HYBRID_BCH_MAXITER && abserr > STLOG_HYBRID_BCH_ABSTOL)        # BCH step
                if iter != 1 && bch_iter_cnt == 0
                    println("Restart!")
                end

                bch_iter_cnt += 1

                direction = @timed begin
                    stlog_BCH5_direction_lyap!(Z, M, B, C, R)
                end
                direction_time += Int(round((direction.time - direction.gctime) * 1e9))

                update = @timed begin
                    _stlog_ret_principal!(M, M_saf, Up, Uk, Z, n, k, wsp_stlog_ret)
                end
                update_time += Int(round((update.time - update.gctime) * 1e9))



                # ret_UpZ!(U, Up, M, M_saf, Z, Z_saf, wsp_UpZ_ret; nearlog=false)

                α = 1.0
                iter += 1
                objval = stlog_cost(M, k)
                abserr = sqrt(2 * objval)
                cost_record[iter] = objval

                # msgln("BCH step\tIteration: $(iter)\tAbsErr: $(abserr)")
            else                                                                # Opt step
                if bch_iter_cnt > 0
                    # The SAF done in last BCH step was not ordered nor regulaized.
                    SAFactor_order(M_saf, wsp_saf_n)
                    SAFactor_regularize(M_saf, wsp_saf_n)
                    copyto!(MatαZ, MatZ)
                    bch_iter_cnt = -STLOG_HYBRID_BCH_SHUTDOWN
                end

                if VecM_ang[1] + VecM_ang[2] > 2π
                    throw("The active point is beyond the restricted manifold, with angles $(VecM_ang[1]) and $(VecM_ang[2]).")
                end

                if VecM_ang[1] > π + STLOG_RESTART_THRESHOLD
                    update = @timed begin
                        log_SpecOrth!(M, M_saf, U, wsp_saf_n; order=true, regular=true)
                    end
                    update_time += Int(round((update.time - update.gctime) * 1e9))

                    bch_iter_cnt = 0
                    objval = stlog_cost(M, k)
                    abserr = sqrt(2 * objval)
                else
                    direction = @timed begin

                        compute_dexp_SkewSymm_both_system!(M_sys, M_ang)
                        newton_flag, newton_time = stlog_newton_descent_backward!(Z, M, M_sys, M_saf, n, k, div((n - k) * (n - k - 1), 2), blk_it_nm, blk_it_m, blk_it_n, wsp_stlog_newton_gmres; Stop=Solver_Stop)
                        getSkewSymm!(Z)

                        action_time += newton_time[2]


                        if newton_flag > 2
                            # Newton direction failed
                            try
                                # Try to construct escaping direction from the gmres result.
                                lmul!(4π / opnorm(MatZ, 2), MatZ)
                                bch_iter_cnt += 1
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

                        try
                            MatΔ_2_norm = opnorm(MatΔ, 2)
                            α_upper_bound = (2π - VecM_ang[1] - VecM_ang[2]) / MatΔ_2_norm
                        catch
                            copyto!(MatZ, view(MatM, (k+1):n, (k+1):n))


                            action = @timed begin
                                cong_dense!(Δ, M_saf.vector, k, Z, 0, m, wsp_cong_nm; trans=true)
                                dexp_SkewSymm!(Δ, Δ, M_sys, M_saf, blk_it_n, wsp_cong_n; inv=true, cong=false)
                                cong_dense!(Δ, M_saf.vector, Δ, wsp_cong_n; trans=false)
                                getSkewSymm!(Δ)
                            end

                            action_time += Int(round((action.time - action.gctime) * 1e9))

                            MatΔ_2_norm = opnorm(MatΔ, 2)
                            α_upper_bound = (2π - VecM_ang[1] - VecM_ang[2]) / MatΔ_2_norm
                            newton_flag = 5
                        end
                    end

                    direction_time += Int(round((direction.time - direction.gctime) * 1e9))




                    MatΔ_2_norm = opnorm(MatΔ, 2)
                    α_upper_bound = (2π - VecM_ang[1] - VecM_ang[2]) / MatΔ_2_norm

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

                    # @printf "Successful Newton direcion? \t%i,\t |Δ|_2:\t %.16f, \t(2π - θ_1 - θ_2) / 2:\t %.16f, \tBB step α:\t %.16f, \tBound of α:\t %.16f\n" newton_flag MatΔ_2_norm (2π - VecM_ang[1] - VecM_ang[2]) * 0.5 α α_upper_bound



                    nearlog_flag = α_upper_bound < 0.1 && MatΔ_2_norm < 3π && STLOG_ENABLE_NEARLOG


                    update = @timed begin

                        if MatΔ_2_norm < 2π
                            α, objval, = stlog_UpZ_NMLS!(fval, slope, α, Z, αZ, Uk, Up, Up_new, M, M_new, M_saf, M_sys, wsp_stlog_ret; paras=NMLS_Set, f_len=iter, nearlog=nearlog_flag, bound=α_upper_bound, fail_step=1.0)
                        else
                            α = π / MatΔ_2_norm
                            copyto!(MatαZ, MatZ)
                            lmul!(α, MatαZ)

                            _stlog_ret_principal!(M, M_saf, Up, Uk, αZ, n, k, wsp_stlog_ret)
                        end
                    end

                    update_time += Int(round((update.time - update.gctime) * 1e9))


                end
            end

            iter += 1
            objval = stlog_cost(M, k)
            abserr = sqrt(2 * objval)
            cost_record[iter] = objval

        end

        elapsed_time += Int(round((elapsed.time - elapsed.gctime) * 1e9))



        result_flag = check_termination_val(abserr, nothing, nothing, elapsed_time, nothing, iter, Stop)


    end

    # display(plot(cost_record[1:iter], yscale=:log10))
    # println(direction_time / elapsed_time)

    # @printf "Time profiles: \t GMRES solver / all time = \t %.16f, \t Action / All time  = \t %.16f, \t Action / GMRES solver = \t %.16f\n" direction_time / elapsed_time action_time / elapsed_time action_time / direction_time

    return result_flag, iter, (elapsed_time, update_time, direction_time, action_time)

end

#######################################Test functions#######################################

using Random, Printf, Plots, DelimitedFiles

function test_stlog_solver(n, k; seed=9527, MaxIter=200, MaxTime=10, AbsTol=1e-9)
    eng = MersenneTwister(seed)
    MatS = rand(eng, n, n)
    fill!(view(MatS, (k+1):n, (k+1):n), 0.0)
    MatS .-= MatS'

    MatU = exp(MatS)

    MatM_BCH1 = zeros(n, n)
    MatUk_BCH1 = copy(view(MatU, :, 1:k))
    MatUp_BCH1 = copy(view(MatU, :, (k+1):n))

    MatM_BCH5 = zeros(n, n)
    MatUk_BCH5 = copy(view(MatU, :, 1:k))
    MatUp_BCH5 = copy(view(MatU, :, (k+1):n))

    MatM_Newton = zeros(n, n)
    MatUk_Newton = copy(view(MatU, :, 1:k))
    MatUp_Newton = copy(view(MatU, :, (k+1):n))

    MatM_Hybrid = zeros(n, n)
    MatUk_Hybrid = copy(view(MatU, :, 1:k))
    MatUp_Hybrid = copy(view(MatU, :, (k+1):n))

    M_BCH1 = Ref(MatM_BCH1)
    Uk_BCH1 = Ref(MatUk_BCH1)
    Up_BCH1 = Ref(MatUp_BCH1)

    M_BCH5 = Ref(MatM_BCH5)
    Uk_BCH5 = Ref(MatUk_BCH5)
    Up_BCH5 = Ref(MatUp_BCH5)

    M_Newton = Ref(MatM_Newton)
    Uk_Newton = Ref(MatUk_Newton)
    Up_Newton = Ref(MatUp_Newton)

    M_Hybrid = Ref(MatM_Hybrid)
    Uk_Hybrid = Ref(MatUk_Hybrid)
    Up_Hybrid = Ref(MatUp_Hybrid)



    wsp_BCH = get_wsp_stlog_BCH(n, k)
    wsp_Newton = get_wsp_stlog_Newton(n, k)

    Stop = terminator(MaxIter, Int(round(MaxTime * 1e9)), AbsTol, AbsTol)
    Solver_Stop = terminator(100, Int(round(1e6)), 1e-9, 1e-9)


    stlog_BCH_2k!(M_BCH1, Uk_BCH1, Up_BCH1, wsp_BCH; Stop=Stop, Init=stlog_init_guess_simple, order=1)
    stlog_BCH_2k!(M_BCH5, Uk_BCH5, Up_BCH5, wsp_BCH; Stop=Stop, Init=stlog_init_guess_simple, order=5)
    stlog_Newton!(M_Newton, Uk_Newton, Up_Newton, wsp_Newton; Stop=Stop, Solver_Stop=Solver_Stop, Init=stlog_init_guess_simple)
    stlog_hybrid!(M_Hybrid, Uk_Hybrid, Up_Hybrid, wsp_Newton; Stop=Stop, Solver_Stop=Solver_Stop, Init=stlog_init_guess_simple)

    @printf "Correct execution of BCH1 algorithm?\t |C|_F :\t %.16f, |exp(S_{A,B,C})I_{n,k} - Uk|_F :\t %.16f\n" norm(MatM_BCH1[(k+1):n, (k+1):n]) norm(exp(MatM_BCH1)[:, 1:k] .- MatU[:, 1:k])
    @printf "Correct execution of BCH5 algorithm?\t |C|_F :\t %.16f, |exp(S_{A,B,C})I_{n,k} - Uk|_F :\t %.16f\n" norm(MatM_BCH5[(k+1):n, (k+1):n]) norm(exp(MatM_BCH5)[:, 1:k] .- MatU[:, 1:k])
    @printf "Correct execution of Newton algorithm?\t |C|_F :\t %.16f, |exp(S_{A,B,C})I_{n,k} - Uk|_F :\t %.16f\n" norm(MatM_Newton[(k+1):n, (k+1):n]) norm(exp(MatM_Newton)[:, 1:k] .- MatU[:, 1:k])
    @printf "Correct execution of Hybrid algorithm?\t |C|_F :\t %.16f, |exp(S_{A,B,C})I_{n,k} - Uk|_F :\t %.16f\n" norm(MatM_Hybrid[(k+1):n, (k+1):n]) norm(exp(MatM_Hybrid)[:, 1:k] .- MatU[:, 1:k])

end


function test_stlog_profile(n::Int, k::Int, σ; show_plts=false, save_plts=false, seed=9527, MaxIter=200, MaxTime=10, AbsTol=1e-12, loops=10, Solver_Stop=terminator(500, Int(round(1e6)), 1e-10, 1e-8), NMLS_Set=NMLS_Paras(0.2, 20.0, 0.9, 0.3, 0))
    eng = MersenneTwister(seed)
    num_alg = 6
    Record = zeros(Int, length(σ), num_alg, loops)
    RecordIter = zeros(Int, length(σ), num_alg, loops)

    Profile_BCH1_BIL = zeros(Int, length(σ), loops, 3)
    Profile_BCH1_SOL = zeros(Int, length(σ), loops, 3)
    Profile_BCH5_BIL = zeros(Int, length(σ), loops, 3)
    Profile_BCH5_SOL = zeros(Int, length(σ), loops, 3)
    Profile_Newton = zeros(Int, length(σ), loops, 4)
    Profile_Hybrid = zeros(Int, length(σ), loops, 4)




    MatS = rand(eng, n, n)
    fill!(view(MatS, (k+1):n, (k+1):n), 0.0)
    MatS .-= MatS'

    wsp_BCH = get_wsp_stlog_BCH(n, k)
    wsp_Newton = get_wsp_stlog_Newton(n, k)



    Stop = terminator(MaxIter, Int(round(MaxTime * 1e9)), AbsTol, AbsTol)


    for s_ind in eachindex(σ)
        s = σ[s_ind]

        MatS .*= s / opnorm(MatS, 2)

        MatU = exp(MatS)

        MatM_BCH1 = zeros(n, n)
        MatUk_BCH1 = copy(view(MatU, :, 1:k))
        MatUp_BCH1 = copy(view(MatU, :, (k+1):n))

        MatM_BCH5 = zeros(n, n)
        MatUk_BCH5 = copy(view(MatU, :, 1:k))
        MatUp_BCH5 = copy(view(MatU, :, (k+1):n))

        MatM_Newton = zeros(n, n)
        MatUk_Newton = copy(view(MatU, :, 1:k))
        MatUp_Newton = copy(view(MatU, :, (k+1):n))

        MatM_Hybrid = zeros(n, n)
        MatUk_Hybrid = copy(view(MatU, :, 1:k))
        MatUp_Hybrid = copy(view(MatU, :, (k+1):n))

        M_BCH1 = Ref(MatM_BCH1)
        Uk_BCH1 = Ref(MatUk_BCH1)
        Up_BCH1 = Ref(MatUp_BCH1)

        M_BCH5 = Ref(MatM_BCH5)
        Uk_BCH5 = Ref(MatUk_BCH5)
        Up_BCH5 = Ref(MatUp_BCH5)

        M_Newton = Ref(MatM_Newton)
        Uk_Newton = Ref(MatUk_Newton)
        Up_Newton = Ref(MatUp_Newton)

        M_Hybrid = Ref(MatM_Hybrid)
        Uk_Hybrid = Ref(MatUk_Hybrid)
        Up_Hybrid = Ref(MatUp_Hybrid)

        for l_ind in 1:loops
            stats = @timed begin
                flag, RecordIter[s_ind, 1, l_ind], Profile = stlog_BCH_2k!(M_BCH1, Uk_BCH1, Up_BCH1, wsp_BCH; Stop=Stop, Init=stlog_init_guess_simple, order=1, builtin_log=true)
            end
            # Record[s_ind, 1, l_ind] = Int(round((stats.time - stats.gctime) * 1e9))
            Record[s_ind, 1, l_ind] = Profile[1]
            Profile_BCH1_BIL[s_ind, l_ind, :] .= Profile

            stats = @timed begin
                flag, RecordIter[s_ind, 2, l_ind], Profile = stlog_BCH_2k!(M_BCH5, Uk_BCH5, Up_BCH5, wsp_BCH; Stop=Stop, Init=stlog_init_guess_simple, order=5, builtin_log=true)
            end
            # Record[s_ind, 2, l_ind] = Int(round((stats.time - stats.gctime) * 1e9))
            Record[s_ind, 2, l_ind] = Profile[1]
            Profile_BCH5_BIL[s_ind, l_ind, :] .= Profile

            stats = @timed begin
                flag, RecordIter[s_ind, 3, l_ind], Profile = stlog_BCH_2k!(M_BCH1, Uk_BCH1, Up_BCH1, wsp_BCH; Stop=Stop, Init=stlog_init_guess_simple, order=1, builtin_log=false)
            end
            # Record[s_ind, 3, l_ind] = Int(round((stats.time - stats.gctime) * 1e9))
            Record[s_ind, 3, l_ind] = Profile[1]
            Profile_BCH1_SOL[s_ind, l_ind, :] .= Profile

            stats = @timed begin
                flag, RecordIter[s_ind, 4, l_ind], Profile = stlog_BCH_2k!(M_BCH5, Uk_BCH5, Up_BCH5, wsp_BCH; Stop=Stop, Init=stlog_init_guess_simple, order=5, builtin_log=false)
            end
            # Record[s_ind, 4, l_ind] = Int(round((stats.time - stats.gctime) * 1e9))
            Record[s_ind, 4, l_ind] = Profile[1]
            Profile_BCH5_SOL[s_ind, l_ind, :] .= Profile

            stats = @timed begin
                flag, RecordIter[s_ind, 5, l_ind], Profile = stlog_Newton!(M_Newton, Uk_Newton, Up_Newton, wsp_Newton; Stop=Stop, Solver_Stop=Solver_Stop, Init=stlog_init_guess_simple, NMLS_Set=NMLS_Set)
            end
            # Record[s_ind, 5, l_ind] = Int(round((stats.time - stats.gctime) * 1e9))
            Record[s_ind, 5, l_ind] = Profile[1]
            Profile_Newton[s_ind, l_ind, :] .= Profile

            stats = @timed begin
                flag, RecordIter[s_ind, 6, l_ind], Profile = stlog_hybrid!(M_Hybrid, Uk_Hybrid, Up_Hybrid, wsp_Newton; Stop=Stop, Solver_Stop=Solver_Stop, Init=stlog_init_guess_simple, NMLS_Set=NMLS_Set)
            end
            # Record[s_ind, 6, l_ind] = Int(round((stats.time - stats.gctime) * 1e9))
            Record[s_ind, 6, l_ind] = Profile[1]
            Profile_Hybrid[s_ind, l_ind, :] .= Profile

        end
    end
    Profile_BCH1_BIL_AVG = reshape(mean(Profile_BCH1_BIL, dims=2), length(σ), 3)
    Profile_BCH5_BIL_AVG = reshape(mean(Profile_BCH5_BIL, dims=2), length(σ), 3)
    Profile_BCH1_SOL_AVG = reshape(mean(Profile_BCH1_SOL, dims=2), length(σ), 3)
    Profile_BCH5_SOL_AVG = reshape(mean(Profile_BCH5_SOL, dims=2), length(σ), 3)
    Profile_Newton_AVG = reshape(mean(Profile_Newton, dims=2), length(σ), 4)
    Profile_Hybrid_AVG = reshape(mean(Profile_Hybrid, dims=2), length(σ), 4)

    BCH1_BIL_DIR = Profile_BCH1_BIL_AVG[:, 3] ./ Profile_BCH1_BIL_AVG[:, 1]
    BCH1_BIL_UPD = Profile_BCH1_BIL_AVG[:, 2] ./ Profile_BCH1_BIL_AVG[:, 1]

    BCH5_BIL_DIR = Profile_BCH5_BIL_AVG[:, 3] ./ Profile_BCH5_BIL_AVG[:, 1]
    BCH5_BIL_UPD = Profile_BCH5_BIL_AVG[:, 2] ./ Profile_BCH5_BIL_AVG[:, 1]

    BCH1_SOL_DIR = Profile_BCH1_SOL_AVG[:, 3] ./ Profile_BCH1_SOL_AVG[:, 1]
    BCH1_SOL_UPD = Profile_BCH1_SOL_AVG[:, 2] ./ Profile_BCH1_SOL_AVG[:, 1]

    BCH5_SOL_DIR = Profile_BCH5_SOL_AVG[:, 3] ./ Profile_BCH5_SOL_AVG[:, 1]
    BCH5_SOL_UPD = Profile_BCH5_SOL_AVG[:, 2] ./ Profile_BCH5_SOL_AVG[:, 1]

    Newton_UPD = Profile_Newton_AVG[:, 2] ./ Profile_Newton_AVG[:, 1]
    Newton_DIR = Profile_Newton_AVG[:, 3] ./ Profile_Newton_AVG[:, 1]
    Newton_ACT = Profile_Newton_AVG[:, 4] ./ Profile_Newton_AVG[:, 1]

    Hybrid_UPD = Profile_Hybrid_AVG[:, 2] ./ Profile_Hybrid_AVG[:, 1]
    Hybrid_DIR = Profile_Hybrid_AVG[:, 3] ./ Profile_Hybrid_AVG[:, 1]
    Hybrid_ACT = Profile_Hybrid_AVG[:, 4] ./ Profile_Hybrid_AVG[:, 1]


    plt_BCH1_BIL = plot(σ, BCH1_BIL_DIR, label="Update Direction", xlabel="σ in U = exp(σ⋅S)I_{n,p} with |S|_2 = 1", ylabel="Time Profile of BCH1 (Builtin log)", yrange=(0, 1), legend=:outertopright)
    plot!(σ, BCH1_BIL_DIR .+ BCH1_BIL_UPD, label="Exp and Log")
    plot!(σ, ones(length(σ)), label="Others")


    plt_BCH5_BIL = plot(σ, BCH5_BIL_DIR, label="Update Direction", xlabel="σ in U = exp(σ⋅S)I_{n,p} with |S|_2 = 1", ylabel="Time Profile of BCH5 (Builtin log)", yrange=(0, 1), legend=:outertopright)
    plot!(σ, BCH5_BIL_DIR .+ BCH5_BIL_UPD, label="Exp and Log")
    plot!(σ, ones(length(σ)), label="Others")



    plt_BCH1_SOL = plot(σ, BCH1_SOL_DIR, label="Update Direction", xlabel="σ in U = exp(σ⋅S)I_{n,p} with |S|_2 = 1", ylabel="Time Profile of BCH1 (SpecOrth log)", yrange=(0, 1), legend=:outertopright)
    plot!(σ, BCH1_SOL_DIR .+ BCH1_SOL_UPD, label="Exp and Log")
    plot!(σ, ones(length(σ)), label="Others")



    plt_BCH5_SOL = plot(σ, BCH5_SOL_DIR, label="Update Direction", xlabel="σ in U = exp(σ⋅S)I_{n,p} with |S|_2 = 1", ylabel="Time Profile of BCH5 (SpecOrth log)", yrange=(0, 1), legend=:outertopright)
    plot!(σ, BCH5_SOL_DIR .+ BCH5_SOL_UPD, label="Exp and Log")
    plot!(σ, ones(length(σ)), label="Others")



    plt_Newton = plot(σ, Newton_DIR, label="Update Direction", xlabel="σ in U = exp(σ⋅S)I_{n,p} with |S|_2 = 1", ylabel="Time Profile of Newton algorithm", yrange=(0, 1), legend=:outertopright)
    plot!(σ, Newton_ACT, label="Action of Dexp")
    plot!(σ, Newton_DIR .+ Newton_UPD, label="Exp and Log")
    plot!(σ, ones(length(σ)), label="Others")


    plt_Hybrid = plot(σ, Hybrid_DIR, label="Update Direction", xlabel="σ in U = exp(σ⋅S)I_{n,p} with |S|_2 = 1", ylabel="Time Profile of Hybrid algorithm", yrange=(0, 1), legend=:outertopright)
    plot!(σ, Hybrid_ACT, label="Action of Dexp")
    plot!(σ, Hybrid_DIR .+ Hybrid_UPD, label="Exp and Log")
    plot!(σ, ones(length(σ)), label="Others")





    RecordMin = reshape(minimum(Record, dims=3), length(σ), num_alg)
    RecordIterMin = reshape(minimum(RecordIter, dims=3), length(σ), num_alg)

    # plt = plot(σ, RecordMin[:, [4, 7, 5, 6]], xlabel="σ in U = exp(σ⋅S)I_{n,p} with |S|_2 = 1", ylabel="Time(ns): exp(S_{A,B,0})I_{n,k} = U", legend=:topleft, labels=["BCH5 SpecOrth logarithm" "BCH5 (old implementation)" "Newton" "Hybrid"], yscale=:log10)

    # display(plt)

    # plt2 = plot(σ, RecordIterMin[:, [4, 7, 5, 6]], xlabel="σ in U = exp(σ⋅S)I_{n,p} with |S|_2 = 1", ylabel="Iterations to Convergence", legend=:topleft, labels=["BCH5 SpecOrth logarithm" "BCH5 (old implementation)" "Newton" "Hybrid"])

    # display(plt2)

    plt_time = plot(σ, RecordMin, xlabel="σ in U = exp(σ⋅S)I_{n,p} with |S|_2 = 1", ylabel="Time(ns): exp(S_{A,B,0})I_{n,k} = U", legend=:topleft, labels=["BCH1 general logarithm" "BCH5 general logarithm" "BCH1 SpecOrth logarithm" "BCH5 SpecOrth logarithm" "Newton" "Hybrid"], yscale=:log10)

    plt_iter = plot(σ, RecordIterMin, xlabel="σ in U = exp(σ⋅S)I_{n,p} with |S|_2 = 1", ylabel="Number of Iterations to Convergence", legend=:topleft, labels=["BCH1 general logarithm" "BCH5 general logarithm" "BCH1 SpecOrth logarithm" "BCH5 SpecOrth logarithm" "Newton" "Hybrid"])

    if save_plts
        savefig(plt_BCH1_BIL, "figures/Profile_BCH1_BIL_n$(n)_k$(k)_s$(seed).pdf")
        savefig(plt_BCH5_BIL, "figures/Profile_BCH5_BIL_n$(n)_k$(k)_s$(seed).pdf")
        savefig(plt_BCH1_SOL, "figures/Profile_BCH1_SOL_n$(n)_k$(k)_s$(seed).pdf")
        savefig(plt_BCH5_SOL, "figures/Profile_BCH5_SOL_n$(n)_k$(k)_s$(seed).pdf")
        savefig(plt_Newton, "figures/Profile_Newton_n$(n)_k$(k)_s$(seed).pdf")
        savefig(plt_Hybrid, "figures/Profile_Hybrid_n$(n)_k$(k)_s$(seed).pdf")
        savefig(plt_time, "figures/Stlog_Time_n$(n)_k$(k)_s$(seed).pdf")
        savefig(plt_iter, "figures/Stlog_Iter_n$(n)_k$(k)_s$(seed).pdf")
    end

    if show_plts
        display(plt_BCH1_SOL)
        display(plt_BCH1_BIL)
        display(plt_BCH5_BIL)
        display(plt_BCH5_SOL)
        display(plt_Newton)
        display(plt_Hybrid)
        display(plt_time)
        display(plt_iter)
    end

    @printf "Random Seed: \t %i\n" seed
end

function test_stlog_time(n::Int, k::Int, σ, sample=10; show_plts=false, save_plts=false, seed=9527, MaxIter=200, MaxTime=10, AbsTol=1e-12, loops=10, Solver_Stop=terminator(500, Int(round(1e6)), 1e-10, 1e-8), NMLS_Set=NMLS_Paras(0.2, 20.0, 0.9, 0.3, 0))
    eng = MersenneTwister(seed)
    num_alg = 3
    RecordTime = zeros(Int, length(σ), num_alg, loops, sample)
    RecordIter = zeros(Int, length(σ), num_alg, loops, sample)


    for sample_ind in 1:sample
        MatS = rand(eng, n, n)
        fill!(view(MatS, (k+1):n, (k+1):n), 0.0)
        MatS .-= MatS'

        wsp_BCH = get_wsp_stlog_BCH(n, k)
        wsp_Newton = get_wsp_stlog_Newton(n, k)



        Stop = terminator(MaxIter, Int(round(MaxTime * 1e9)), AbsTol, AbsTol)


        for s_ind in eachindex(σ)
            s = σ[s_ind]

            MatS .*= s / opnorm(MatS, 2)

            MatU = exp(MatS)

            MatM_BCH5 = zeros(n, n)
            MatUk_BCH5 = copy(view(MatU, :, 1:k))
            MatUp_BCH5 = copy(view(MatU, :, (k+1):n))

            MatM_Newton = zeros(n, n)
            MatUk_Newton = copy(view(MatU, :, 1:k))
            MatUp_Newton = copy(view(MatU, :, (k+1):n))

            MatM_Hybrid = zeros(n, n)
            MatUk_Hybrid = copy(view(MatU, :, 1:k))
            MatUp_Hybrid = copy(view(MatU, :, (k+1):n))

            M_BCH5 = Ref(MatM_BCH5)
            Uk_BCH5 = Ref(MatUk_BCH5)
            Up_BCH5 = Ref(MatUp_BCH5)

            M_Newton = Ref(MatM_Newton)
            Uk_Newton = Ref(MatUk_Newton)
            Up_Newton = Ref(MatUp_Newton)

            M_Hybrid = Ref(MatM_Hybrid)
            Uk_Hybrid = Ref(MatUk_Hybrid)
            Up_Hybrid = Ref(MatUp_Hybrid)

            for l_ind in 1:loops

                stats = @timed begin
                    flag, RecordIter[s_ind, 1, l_ind, sample_ind], Profile = stlog_BCH_2k!(M_BCH5, Uk_BCH5, Up_BCH5, wsp_BCH; Stop=Stop, Init=stlog_init_guess_simple, order=5, builtin_log=false)
                end
                RecordTime[s_ind, 1, l_ind, sample_ind] = Profile[1]

                stats = @timed begin
                    flag, RecordIter[s_ind, 2, l_ind, sample_ind], Profile = stlog_Newton!(M_Newton, Uk_Newton, Up_Newton, wsp_Newton; Stop=Stop, Solver_Stop=Solver_Stop, Init=stlog_init_guess_simple, NMLS_Set=NMLS_Set)
                end
                RecordTime[s_ind, 2, l_ind, sample_ind] = Profile[1]

                stats = @timed begin
                    flag, RecordIter[s_ind, 3, l_ind, sample_ind], Profile = stlog_hybrid!(M_Hybrid, Uk_Hybrid, Up_Hybrid, wsp_Newton; Stop=Stop, Solver_Stop=Solver_Stop, Init=stlog_init_guess_simple, NMLS_Set=NMLS_Set)
                end
                RecordTime[s_ind, 3, l_ind, sample_ind] = Profile[1]
            end
        end

    end

    RecordTime = reshape(minimum(RecordTime, dims=3), length(σ), num_alg, sample)
    RecordIter = reshape(minimum(RecordIter, dims=3), length(σ), num_alg, sample)

    open("figures/stlog_data_n$(n)_k$(k)_seed$(seed).txt", "w") do io
        write(io, "// n = $(n), k = $(k), seed = $(seed), # of sigma = $(length(σ)), # of algorithms = $(num_alg)\n")
        write(io, "BCH5 SpecOrth logarithm\tNewton Algorithm\tHybrid Algorithm\n")
        write(io, "$(n)\t$(k)\t$(seed)\t$(length(σ))\t$(num_alg)\n")

        writedlm(io, reshape(collect(σ), 1, length(σ)))

        write(io, "\nBCH5 SpecOrth logarithm\n")
        for ind in 1:sample
            writedlm(io, reshape(RecordTime[:, 1, ind], 1, length(σ)))
        end

        write(io, "\nNewton Algorithm\n")
        for ind in 1:sample
            writedlm(io, reshape(RecordTime[:, 2, ind], 1, length(σ)))
        end

        write(io, "\nHybrid Algorithm\n")
        for ind in 1:sample
            writedlm(io, reshape(RecordTime[:, 3, ind], 1, length(σ)))
        end

        write(io, "\nBCH5 SpecOrth logarithm\n")
        for ind in 1:sample
            writedlm(io, reshape(RecordIter[:, 1, ind], 1, length(σ)))
        end

        write(io, "\nNewton Algorithm\n")
        for ind in 1:sample
            writedlm(io, reshape(RecordIter[:, 2, ind], 1, length(σ)))
        end

        write(io, "\nHybrid Algorithm\n")
        for ind in 1:sample
            writedlm(io, reshape(RecordIter[:, 3, ind], 1, length(σ)))
        end
    end

    RecordTime_Avg = reshape(mean(RecordTime, dims=3), length(σ), num_alg)
    RecordIter_Avg = reshape(mean(RecordIter, dims=3), length(σ), num_alg)

    RecordTime_Min = reshape(minimum(RecordTime, dims=3), length(σ), num_alg)
    RecordIter_Min = reshape(minimum(RecordIter, dims=3), length(σ), num_alg)

    RecordTime_Max = reshape(maximum(RecordTime, dims=3), length(σ), num_alg)
    RecordIter_Max = reshape(maximum(RecordIter, dims=3), length(σ), num_alg)

    RecordTime_STD = reshape(std(RecordTime, dims=3), length(σ), num_alg)
    RecordIter_STD = reshape(std(RecordIter, dims=3), length(σ), num_alg)


    # plt = plot(σ, RecordMin[:, [4, 7, 5, 6]], xlabel="σ in U = exp(σ⋅S)I_{n,p} with |S|_2 = 1", ylabel="Time(ns): exp(S_{A,B,0})I_{n,k} = U", legend=:topleft, labels=["BCH5 SpecOrth logarithm" "BCH5 (old implementation)" "Newton" "Hybrid"], yscale=:log10)

    # display(plt)

    # plt2 = plot(σ, RecordIterMin[:, [4, 7, 5, 6]], xlabel="σ in U = exp(σ⋅S)I_{n,p} with |S|_2 = 1", ylabel="Iterations to Convergence", legend=:topleft, labels=["BCH5 SpecOrth logarithm" "BCH5 (old implementation)" "Newton" "Hybrid"])

    # display(plt2)

    plt_time = plot(xlabel="σ in U = exp(σ⋅S)I_{n,p} with |S|_2 = 1", ylabel="Compute Time to Convergence (ns)", yscale=:log10)
    plot!(σ, RecordTime_Avg, ribbon=RecordTime_STD, fillalpha=0.4, label=["BCH5 SpecOrth logarithm" "Newton Algorithm" "Hybrid Algorithm"])

    plt_iter = plot(xlabel="σ in U = exp(σ⋅S)I_{n,p} with |S|_2 = 1", ylabel="Number of Iterations to Convergence (ns)", yscale=:log10)
    plot!(σ, RecordIter_Avg, ribbon=RecordIter_STD, fillalpha=0.4, label=["BCH5 SpecOrth logarithm" "Newton Algorithm" "Hybrid Algorithm"])

    if save_plts
        savefig(plt_time, "figures/Stlog_Time_n$(n)_k$(k)_seed$(seed)_sample$(sample).pdf")
        savefig(plt_iter, "figures/Stlog_Iter_n$(n)_k$(k)_seed$(seed)_sample$(sample).pdf")
    end

    if show_plts
        display(plt_time)
        display(plt_iter)
    end

    @printf "Random Seed: \t %i\n" seed
end