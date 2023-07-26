include("../../inc/global_path.jl")

include(joinpath(JULIA_MANIFOLD_PATH, "spec_orth/so_nearlog_newton.jl"))


function grhor_NMLS!(fval::Ref{Vector{Float64}}, slope::Float64, init::Float64, Z::Ref{Matrix{Float64}}, αZ::Ref{Matrix{Float64}}, Z_saf::SAFactor,
    U::Ref{Matrix{Float64}}, Up::Ref{Matrix{Float64}}, Up_new::Ref{Matrix{Float64}}, M::Ref{Matrix{Float64}}, M_new::Ref{Matrix{Float64}}, M_saf::SAFactor, wsp_stlog_UpZ_ret::WSP; 
    paras::NMLS_Paras=NMLS_Paras(0.01, 10.0, 0.1, 0.5, 5), f_len::Int=-1, nearlog::Bool = false, bound::Float64 = paras.α_max, fail_step::Float64 = paras.α_min)+

    # println("Start non-monotonic linear search:")
    Vecfval = fval[]

    # Geometry_Set collects routines for scalling velocities, updating points and computing retraction.

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

    good_search::Bool = false;


    if f_len < 0
        f_len = length(Vecfval)
        ind = max(1, f_len - paraM - 1)
        f_max = -1000000
        for ii = ind:f_len
            if f_max < Vecfval[ii]
                @inbounds f_max = Vecfval[ii]
            end
        end
        # f_max = maximum(f[max(1, f_len - M - 1):f_len]);
    elseif f_len > 0
        ind = max(1, f_len - paraM)
        f_max = -1000000
        f_cur = Vecfval[f_len]
        for ii = ind:f_len
            if f_max < Vecfval[ii]
                @inbounds f_max = Vecfval[ii]
            end
        end
        # f_max = maximum(f[max(1, f_len - M - 1):f_len])
    else
        # First 2 iterations, use 1.0.

        good_search = true

        α_cur = 1.0

        nip_ret_UpZ!(U, Up_new, Up, M_new, M, M_saf, Z, Z_saf, α_cur, wsp_stlog_UpZ_ret; nearlog = nearlog && (α_cur < bound))


        f_new = stlog_cost(M_new, k)

        scale_velocity_UpZ!(αZ, Z, α_cur)
        unsafe_copyto!(pointer(Up[]), pointer(Up_new[]), length(Up[]))
        unsafe_copyto!(pointer(M[]), pointer(M_new[]), length(M[]))
        return α_cur, f_new, good_search
    end

    if α_cur < α_min
        # use minimal stepsize

        good_search = false

        α_cur = fail_step


        nip_ret_UpZ!(U, Up_new, Up, M_new, M, M_saf, Z, Z_saf, α_cur, wsp_stlog_UpZ_ret; nearlog = false)

        f_new = stlog_cost(M_new, k)
        
        scale_velocity_UpZ!(αZ, Z, α_cur)
        unsafe_copyto!(pointer(Up[]), pointer(Up_new[]), length(Up[]))
        unsafe_copyto!(pointer(M[]), pointer(M_new[]), length(M[]))
        return α_cur, f_new, good_search
    elseif α_cur > α_max
        α_cur = α_max
    end

    # Backtracking

    while α_cur >= α_min
        # scale_velocity(v_ref, αv_ref, α_cur);
        # f_new = horizontal_nip_ret(U_ref, Up_ref, Up_new_ref, S_ref, S_new_ref, P_ref, Θ_ref, αv_ref, Ret);
        nip_ret_UpZ!(U, Up_new, Up, M_new, M, M_saf, Z, Z_saf, α_cur, wsp_stlog_UpZ_ret; nearlog = nearlog && (α_cur < bound))

        f_new = stlog_cost(M_new, k)

        d_msgln(@sprintf("\t\tf_cur: %.8f\t f_max: %.8f\t α_cur: %.8f\t f_max + α_curΔ: %8f\t f_new: %.8f.", 
            f_cur, f_max, α_cur, f_max + γ * (slope + min(α_cur, 1.5) * f_cur) * α_cur, f_new))
        
        # d_msg(["\t\t\t", f_max, " ", f_new, " ", slope, " ", α_cur, "\n"], true);
        # update Up_new, αZ, S, P and A;
        if (f_new < max(0.1 * f_max, f_max + γ * (slope + min(α_cur, 1.5) * f_cur) * α_cur))
            good_search = true
            scale_velocity_UpZ!(αZ, Z, α_cur)
            unsafe_copyto!(pointer(Up[]), pointer(Up_new[]), length(Up[]))
            unsafe_copyto!(pointer(M[]), pointer(M_new[]), length(M[]))
            # horizontal_overwrite_ret(Up_ref, Up_new_ref, S_ref, S_new_ref);
            return α_cur, f_new, good_search
        else
            α_cur *= σ
            if α_cur < α_min
                # use minimal stepsize
                d_msgln("Line search failed.")


                good_search = false;

                α_cur = fail_step

                nip_ret_UpZ!(U, Up_new, Up, M_new, M, M_saf, Z, Z_saf, α_cur, wsp_stlog_UpZ_ret; nearlog = false)

                f_new = stlog_cost(M_new, k)
                
                scale_velocity_UpZ!(αZ, Z, α_cur)
                unsafe_copyto!(pointer(Up[]), pointer(Up_new[]), length(Up[]))
                unsafe_copyto!(pointer(M[]), pointer(M_new[]), length(M[]))
                # scale_velocity(v_ref, αv_ref, α_cur);
                # f_new = horizontal_nip_ret(U_ref, Up_ref, Up_new_ref, S_ref, S_new_ref, P_ref, Θ_ref, αv_ref, Ret);
                # horizontal_overwrite_ret(Up_ref, Up_new_ref, S_ref, S_new_ref);
                return α_cur, f_new, good_search
            end
        end
    end

    # α_cur = α_min;
    # f_new = log_St_opt_not_in_place_update(Uk_ref, Up_ref, Up_new_ref, Z_ref, αZ_ref, S_ref, P_ref, A_ref, α_cur);
    # Up .= Up_new;
    # return α_cur, f_newa, good_search;
end

# function grhor_NMLS!(fval::Ref{Vector{Float64}}, slope::Float64, init::Float64, Z::Ref{Matrix{Float64}}, αZ::Ref{Matrix{Float64}}, Z_saf::SAFactor,
#     U::Ref{Matrix{Float64}}, Up::Ref{Matrix{Float64}}, Up_new::Ref{Matrix{Float64}}, M::Ref{Matrix{Float64}}, M_new::Ref{Matrix{Float64}}, M_saf::SAFactor, wsp_stlog_UpZ_ret::WSP; 
#     paras::NMLS_Paras=NMLS_Paras(0.01, 10.0, 0.1, 0.5, 5), f_len::Int=-1, nearlog::Bool = false, bound::Float64 = paras.α_max, fail_step::Float64 = paras.α_min)+

#     # println("Start non-monotonic linear search:")
#     Vecfval = fval[]

#     # Geometry_Set collects routines for scalling velocities, updating points and computing retraction.

#     ind::Int = 0
#     α_cur::Float64 = init
#     f_cur::Float64 = 0.0
#     f_max::Float64 = 0.0
#     f_new::Float64 = 0.0

#     α_min::Float64 = paras.α_min
#     α_max::Float64 = paras.α_max
#     paraM::Int = paras.M
#     γ::Float64 = paras.γ
#     σ::Float64 = paras.σ


#     n::Int, m::Int = size(Up[])
#     k::Int = n - m

#     good_search::Bool = false;


#     if f_len < 0
#         f_len = length(Vecfval)
#         ind = max(1, f_len - paraM - 1)
#         f_max = -1000000
#         for ii = ind:f_len
#             if f_max < Vecfval[ii]
#                 @inbounds f_max = Vecfval[ii]
#             end
#         end
#         # f_max = maximum(f[max(1, f_len - M - 1):f_len]);
#     elseif f_len > 0
#         ind = max(1, f_len - paraM)
#         f_max = -1000000
#         f_cur = Vecfval[f_len]
#         for ii = ind:f_len
#             if f_max < Vecfval[ii]
#                 @inbounds f_max = Vecfval[ii]
#             end
#         end
#         # f_max = maximum(f[max(1, f_len - M - 1):f_len])
#     else
#         # First 2 iterations, use 1.0.

#         good_search = true

#         α_cur = 1.0

#         nip_ret_UpZ!(U, Up_new, Up, M_new, M, M_saf, Z, Z_saf, α_cur, wsp_stlog_UpZ_ret; nearlog = nearlog && (α_cur < bound))


#         f_new = stlog_cost(M_new, k)

#         scale_velocity_UpZ!(αZ, Z, α_cur)
#         unsafe_copyto!(pointer(Up[]), pointer(Up_new[]), length(Up[]))
#         unsafe_copyto!(pointer(M[]), pointer(M_new[]), length(M[]))
#         return α_cur, f_new, good_search
#     end

#     if α_cur < α_min
#         # use minimal stepsize

#         good_search = false

#         α_cur = fail_step


#         nip_ret_UpZ!(U, Up_new, Up, M_new, M, M_saf, Z, Z_saf, α_cur, wsp_stlog_UpZ_ret; nearlog = false)

#         f_new = stlog_cost(M_new, k)
        
#         scale_velocity_UpZ!(αZ, Z, α_cur)
#         unsafe_copyto!(pointer(Up[]), pointer(Up_new[]), length(Up[]))
#         unsafe_copyto!(pointer(M[]), pointer(M_new[]), length(M[]))
#         return α_cur, f_new, good_search
#     elseif α_cur > α_max
#         α_cur = α_max
#     end

#     # Backtracking

#     while α_cur >= α_min
#         # scale_velocity(v_ref, αv_ref, α_cur);
#         # f_new = horizontal_nip_ret(U_ref, Up_ref, Up_new_ref, S_ref, S_new_ref, P_ref, Θ_ref, αv_ref, Ret);
#         nip_ret_UpZ!(U, Up_new, Up, M_new, M, M_saf, Z, Z_saf, α_cur, wsp_stlog_UpZ_ret; nearlog = nearlog && (α_cur < bound))

#         f_new = stlog_cost(M_new, k)

#         d_msgln(@sprintf("\t\tf_cur: %.8f\t f_max: %.8f\t α_cur: %.8f\t f_max + α_curΔ: %8f\t f_new: %.8f.", 
#             f_cur, f_max, α_cur, f_max + γ * (slope + min(α_cur, 1.5) * f_cur) * α_cur, f_new))
        
#         # d_msg(["\t\t\t", f_max, " ", f_new, " ", slope, " ", α_cur, "\n"], true);
#         # update Up_new, αZ, S, P and A;
#         if (f_new < max(0.1 * f_max, f_max + γ * (slope + min(α_cur, 1.5) * f_cur) * α_cur))
#             good_search = true
#             scale_velocity_UpZ!(αZ, Z, α_cur)
#             unsafe_copyto!(pointer(Up[]), pointer(Up_new[]), length(Up[]))
#             unsafe_copyto!(pointer(M[]), pointer(M_new[]), length(M[]))
#             # horizontal_overwrite_ret(Up_ref, Up_new_ref, S_ref, S_new_ref);
#             return α_cur, f_new, good_search
#         else
#             α_cur *= σ
#             if α_cur < α_min
#                 # use minimal stepsize
#                 d_msgln("Line search failed.")


#                 good_search = false;

#                 α_cur = fail_step

#                 nip_ret_UpZ!(U, Up_new, Up, M_new, M, M_saf, Z, Z_saf, α_cur, wsp_stlog_UpZ_ret; nearlog = false)

#                 f_new = stlog_cost(M_new, k)
                
#                 scale_velocity_UpZ!(αZ, Z, α_cur)
#                 unsafe_copyto!(pointer(Up[]), pointer(Up_new[]), length(Up[]))
#                 unsafe_copyto!(pointer(M[]), pointer(M_new[]), length(M[]))
#                 # scale_velocity(v_ref, αv_ref, α_cur);
#                 # f_new = horizontal_nip_ret(U_ref, Up_ref, Up_new_ref, S_ref, S_new_ref, P_ref, Θ_ref, αv_ref, Ret);
#                 # horizontal_overwrite_ret(Up_ref, Up_new_ref, S_ref, S_new_ref);
#                 return α_cur, f_new, good_search
#             end
#         end
#     end

#     # α_cur = α_min;
#     # f_new = log_St_opt_not_in_place_update(Uk_ref, Up_ref, Up_new_ref, Z_ref, αZ_ref, S_ref, P_ref, A_ref, α_cur);
#     # Up .= Up_new;
#     # return α_cur, f_newa, good_search;
# end
