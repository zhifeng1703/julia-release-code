include("../../inc/global_path.jl")

include(joinpath(JULIA_INCLUDE_PATH, "algorithm_port.jl"))

include("stlog_compact_geometry.jl")

using Printf


function hor_BB_step!(α_pre::Float64, sq_pre::Float64, Z_cur_r::Ref{Matrix{Float64}}, αZ_pre_r::Ref{Matrix{Float64}})
    α_cur::Float64 = α_pre * sq_pre / (sq_pre - inner_skew!(Z_cur_r, αZ_pre_r) / α_pre)
    sq_cur::Float64 = inner_skew!(Z_cur_r, Z_cur_r)

    return max(α_cur, 1.0), sq_cur
end

function stlog_UpZ_NMLS!(fval::Ref{Vector{Float64}}, slope::Float64, init::Float64, Z::Ref{Matrix{Float64}}, αZ::Ref{Matrix{Float64}},
    Uk::Ref{Matrix{Float64}}, Up::Ref{Matrix{Float64}}, Up_new::Ref{Matrix{Float64}}, M::Ref{Matrix{Float64}}, M_new::Ref{Matrix{Float64}}, M_saf::SAFactor, M_sys::dexp_SkewSymm_system, wsp_stlog_ret::WSP;
    paras::NMLS_Paras=NMLS_Paras(0.01, 10.0, 0.1, 0.5, 5), f_len::Int=-1, nearlog::Bool=false, bound::Float64=paras.α_max, fail_step::Float64=paras.α_min)
    # This routine updates Up, αZ, S, P and A and returns (stepsize, f_new)

    # println("Start non-monotonic linear search:")
    # println("Initial α $(init), Bound $(bound), α_max $(paras.α_max), α_min $(paras.α_min)")
    Vecfval = fval[]

    MatUp = Up[]
    MatUp_new = Up_new[]
    MatαZ = αZ[]
    MatZ = Z[]
    MatM = M[]
    MatM_new = M_new[]

    # println("\tData Loaded.")



    ind::Int = 0
    α_cur::Float64 = init
    f_cur::Float64 = 0.0
    f_max::Float64 = 0.0
    f_new::Float64 = 0.0

    α_min::Float64 = paras.α_min
    α_max::Float64 = paras.α_max
    paraM::Int = paras.M
    γ::Float64 = paras.γ
    σ::Float64 = paras.σ


    n::Int, m::Int = size(Up[])
    k::Int = n - m

    # println("\tParameters Set.")


    good_search::Bool = false

    if f_len < 0
        f_len = length(Vecfval)
        ind = max(1, f_len - paraM - 1)
        f_max = -1000000
        for ii = ind:f_len
            if f_max < Vecfval[ii]
                @inbounds f_max = Vecfval[ii]
            end
        end
    elseif f_len > 0
        ind = max(1, f_len - paraM)
        f_max = -1000000
        f_cur = Vecfval[f_len]
        for ii = ind:f_len
            if f_max < Vecfval[ii]
                @inbounds f_max = Vecfval[ii]
            end
        end
    else
        # First 2 iterations, use 1.0.

        # println("\tFirst two iterations.")


        good_search = true

        α_cur = 1.0

        copyto!(MatαZ, MatZ)
        # lmul!(α_cur, MatαZ) # α_cur = 1.0

        copyto!(MatUp_new, MatUp)
        _stlog_ret!(M_new, M_saf, Up_new, Uk, αZ, M_sys, n, k, nearlog, wsp_stlog_ret)
        f_new = stlog_cost(M_new, k)



        copyto!(MatUp, MatUp_new)
        copyto!(MatM, MatM_new)

        # println("Exit non-monotonic linear search.")


        return α_cur, f_new, good_search
    end

    if α_cur < α_min
        # use minimal stepsize

        good_search = false

        α_cur = fail_step

        copyto!(MatαZ, MatZ)
        lmul!(α_cur, MatαZ)

        copyto!(MatUp_new, MatUp)
        _stlog_ret!(M_new, M_saf, Up_new, Uk, αZ, M_sys, n, k, nearlog, wsp_stlog_ret)
        f_new = stlog_cost(M_new, k)

        copyto!(MatUp, MatUp_new)
        copyto!(MatM, MatM_new)

        # println("Exit non-monotonic linear search.")


        return α_cur, f_new, good_search
    elseif α_cur > α_max
        α_cur = α_max
    end

    # println("\tMax objective value found.")

    # println("Initial α $(init), Current α $(α_cur), Bound $(bound), α_max $(paras.α_max), α_min $(paras.α_min)")



    # Backtracking

    while α_cur >= α_min
        # println("\tStart Backtracking.")
        copyto!(MatαZ, MatZ)
        lmul!(α_cur, MatαZ)

        copyto!(MatUp_new, MatUp)
        _stlog_ret!(M_new, M_saf, Up_new, Uk, αZ, M_sys, n, k, nearlog, wsp_stlog_ret)

        f_new = stlog_cost(M_new, k)

        d_msgln(@sprintf("\t\tf_cur: %.8f\t f_max: %.8f\t α_cur: %.8f\t f_max + α_curΔ: %8f\t f_new: %.8f.",
            f_cur, f_max, α_cur, f_max + γ * (slope + min(α_cur, 1.5) * f_cur) * α_cur, f_new))

        # update Up_new, αZ, S, P and A;
        if (f_new < max(0.1 * f_max, f_max + γ * (slope + min(α_cur, 1.5) * f_cur) * α_cur))
            good_search = true

            copyto!(MatUp, MatUp_new)
            copyto!(MatM, MatM_new)

            # println("Exit non-monotonic linear search.")


            return α_cur, f_new, good_search
        else
            α_cur *= σ
            if α_cur < α_min
                # use minimal stepsize
                # d_msgln("Line search failed.")


                good_search = false

                α_cur = fail_step

                copyto!(MatαZ, MatZ)
                lmul!(α_cur, MatαZ)

                copyto!(MatUp_new, MatUp)
                _stlog_ret!(M_new, M_saf, Up_new, Uk, αZ, M_sys, n, k, false, wsp_stlog_ret)
                f_new = stlog_cost(M_new, k)

                copyto!(MatUp, MatUp_new)
                copyto!(MatM, MatM_new)

                # println("Exit non-monotonic linear search.")


                return α_cur, f_new, good_search
            end
        end
    end
end
