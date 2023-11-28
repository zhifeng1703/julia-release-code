include("../../inc/global_path.jl")

include(joinpath(JULIA_INCLUDE_PATH, "workspace.jl"))
include(joinpath(JULIA_INCLUDE_PATH, "algorithm_port.jl"))
include(joinpath(JULIA_LAPACK_PATH, "gmres.jl"))
include(joinpath(JULIA_LAPACK_PATH, "dgesvd.jl"))
include(joinpath(JULIA_MANIFOLD_PATH, "spec_orth/dexp_SkewSymm.jl"))



using LinearMaps, IterativeSolvers


@inline get_wsp_stlog_newton_action(n::Int, k::Int, wsp_cong_n::WSP, wsp_cong_nm::WSP) = WSP(Matrix{Float64}(undef, n, n), Matrix{Float64}(undef, n, n), wsp_cong_n, wsp_cong_nm);
@inline get_wsp_stlog_newton_action(n::Int, k::Int) = WSP(Matrix{Float64}(undef, n, n), Matrix{Float64}(undef, n, n), get_wsp_cong(n), get_wsp_cong(n, n - k));


"""
    get_wsp_stlog_newton_gmres(n::Int, d::Int, rs::Int)

Workspace for solving the Newton direction of the stlog problem of size `n × k` with dimension `d = div((n - k) * (n - k - 1), 2)`. The gmres is set to restart after `rs` iterations
"""
@inline get_wsp_stlog_newton_gmres(d::Int, wsp_action::WSP, wsp_gmres::WSP) = WSP(Vector{Float64}(undef, d), Vector{Float64}(undef, d), wsp_action, wsp_gmres)
@inline get_wsp_stlog_newton_gmres(n::Int, k::Int, d::Int, rs::Int) = WSP(Vector{Float64}(undef, d), Vector{Float64}(undef, d), get_wsp_stlog_newton_action(n, k), get_wsp_gmres(d, rs))


function _stlog_newton_descent_backward_system!(y, x, M_sys, M_saf, blk_it_nm, wsp_stlog_newton_action)
    MatS = wsp_stlog_newton_action[1]
    wsp_cong_n = wsp_stlog_newton_action[3]
    wsp_cong_nm = wsp_stlog_newton_action[4]


    S = wsp_stlog_newton_action(1)
    Δ = wsp_stlog_newton_action(1)


    # S = S_{0, 0, Z}
    fill!(MatS, 0.0)
    mat_SkewSymm_blk!(S, blk_it_nm, x; fil=true)

    # Δ = dexp_{S_{A, B, C}}^{-1}[S_{0, 0, Z}]
    dexp_SkewSymm!(Δ, S, M_sys, M_saf, wsp_cong_n; inv=true, cong=true)

    # Extract Δ_C from Δ = S_{Δ_A, Δ_B, Δ_C}
    vec_SkewSymm_blk!(y, Δ, blk_it_nm; lower=true)
end

function _stlog_newton_descent_backward_system!(y, x, M_sys, M_saf, blk_it_nm, n, k, wsp_stlog_newton_action)
    MatS = wsp_stlog_newton_action[1]
    wsp_cong_n = wsp_stlog_newton_action[3]
    wsp_cong_nm = wsp_stlog_newton_action[4]


    S = wsp_stlog_newton_action(1)
    Δ = wsp_stlog_newton_action(1)




    # S = S_{0, 0, Z}
    fill!(MatS, 0.0)
    mat_SkewSymm_blk!(S, blk_it_nm, x; fil=true)

    # Δ = dexp_{S_{A, B, C}}^{-1}[S_{0, 0, Z}]
    dexp_SkewSymm!(Δ, S, M_sys, M_saf, wsp_cong_n; inv=true, cong=true)

    # Extract Δ_C from Δ = S_{Δ_A, Δ_B, Δ_C}
    vec_SkewSymm_blk!(y, Δ, blk_it_nm; lower=true)
end

function _stlog_newton_descent_backward_system!(y, x, M_sys, M_saf, blk_it_nm, blk_it_n, n, k, wsp_stlog_newton_action)
    MatS = wsp_stlog_newton_action[1]
    wsp_cong_n = wsp_stlog_newton_action[3]
    wsp_cong_nm = wsp_stlog_newton_action[4]


    S = wsp_stlog_newton_action(1)
    Δ = wsp_stlog_newton_action(1)


    # S = S_{0, 0, Z}
    fill!(MatS, 0.0)
    mat_SkewSymm_blk!(S, blk_it_nm, x; fil=true)

    # Δ = dexp_{S_{A, B, C}}^{-1}[S_{0, 0, Z}]
    cong_dense!(S, M_saf.vector, k, S, k, n - k, wsp_cong_nm; trans=true)

    # mul!(MatTmpnm, view(MatR, (k + 1):n, :)', view(MatS, (k + 1):n, (k + 1):n));
    # mul!(MatS, MatTmpnm, view(MatR, (k + 1):n, :))
    # fill_upper_SkewSymm!(Tmpn, blk_it_n)

    dexp_SkewSymm!(Δ, S, M_sys, M_saf, blk_it_n, wsp_cong_n; inv=true, cong=false)
    # fill_upper_SkewSymm!(Δ, blk_it_n)

    cong_dense!(Δ, M_saf.vector, Δ, wsp_cong_n; trans=false)

    # Extract Δ_C from Δ = S_{Δ_A, Δ_B, Δ_C}
    vec_SkewSymm_blk!(y, Δ, blk_it_nm; lower=true)
end




_stlog_newton_wapper(y_r, x_r; kwargs=nothing) = _stlog_newton_descent_backward_system!(y_r, x_r, kwargs...)


function stlog_newton_descent_backward!(Z::Ref{Matrix{Float64}},
    M::Ref{Matrix{Float64}}, M_sys::dexp_SkewSymm_system, M_saf::SAFactor, n::Int, k::Int, restart::Int,
    blk_it_nm::STRICT_LOWER_ITERATOR, blk_it_m::STRICT_LOWER_ITERATOR,
    wsp_bgs::WSP=get_wsp_bgs(n, k, div((n - k) * (n - k - 1), 2), restart); Stop=terminator(50, 1000, 1e-8, 1e-6))

    VecV = wsp_bgs[1]
    wsp_action = wsp_bgs[3]
    wsp_gmres = wsp_bgs[4]

    v = wsp_bgs(1)
    b = wsp_bgs(2)


    # extract C from M = S_{A, B, C} (lower triangular part)
    # b = vec(C) from M = S_{A, B, C}
    vec_SkewSymm_blk!(b, M, blk_it_nm; lower=true)

    fill!(VecV, 0.0)

    flag = gmres_matfree!(_stlog_newton_wapper, v, b, wsp_gmres, div((n - k) * (n - k - 1), 2), restart, Stop;
        is_x_zeros=true, action_kwargs=[M_sys, M_saf, blk_it_nm, wsp_action])

    mat_SkewSymm_blk!(Z, blk_it_m, v; fil=true)

    return flag
end

"""
    stlog_newton_descent_backward!(Z, M, M_sys, M_saf, n, k, restart, blk_it_nm, blk_it_m, blk_it_n, wsp_stlog_newton_gmres; Stop) -> Z::Ref{Matrix{Float64}}

Compute the Z so that Dexp_{S_{A,B,C}}[S_{X, Y, C}] = QS_{0, 0, Z} with unknowns X, Y, Z.

"""
function stlog_newton_descent_backward!(Z::Ref{Matrix{Float64}},
    M::Ref{Matrix{Float64}}, M_sys::dexp_SkewSymm_system, M_saf::SAFactor, n::Int, k::Int, restart::Int,
    blk_it_nm::STRICT_LOWER_ITERATOR, blk_it_m::STRICT_LOWER_ITERATOR, blk_it_n::STRICT_LOWER_ITERATOR,
    wsp_stlog_newton_gmres::WSP=get_wsp_stlog_newton_gmres(n, k, div((n - k) * (n - k - 1), 2), restart); Stop=terminator(50, 1000, 1e-8, 1e-6))

    VecV = wsp_stlog_newton_gmres[1]
    wsp_action = wsp_stlog_newton_gmres[3]
    wsp_gmres = wsp_stlog_newton_gmres[4]

    v = wsp_stlog_newton_gmres(1)
    b = wsp_stlog_newton_gmres(2)


    # extract C from M = S_{A, B, C} (lower triangular part)
    # b = vec(C) from M = S_{A, B, C}
    vec_SkewSymm_blk!(b, M, blk_it_nm; lower=true)

    fill!(VecV, 0.0)

    # Solve Dexp_{S_{A, B, C}}[S_{X, Y, C}] = QS_{0, 0, Z} 
    flag, time = gmres_matfree_analysis!(_stlog_newton_wapper, v, b, wsp_gmres, div((n - k) * (n - k - 1), 2), restart, Stop;
        is_x_zeros=true, action_kwargs=[M_sys, M_saf, blk_it_nm, blk_it_n, n, k, wsp_action])

    mat_SkewSymm_blk!(Z, blk_it_m, v; fil=true)

    return flag, time
end


"""
    stlog_BCH3_direction_naive(Z, B, C, R) -> Z::Ref{Matrix{Float64}}

Compute the BCH third update Z from the sylvester equation Z(BB'/12 - I/2) + (BB'/12 - I/2)Z = -C. Lyapunov solver is used.

"""
function stlog_BCH5_direction_lyap!(Z::Ref{Matrix{Float64}}, M::Ref{Matrix{Float64}}, B::Ref{Matrix{Float64}}, C::Ref{Matrix{Float64}}, R::Ref{Matrix{Float64}})

    MatZ = Z[]
    MatM = M[]
    MatB = B[]
    MatC = C[]
    MatR = R[]

    n = size(MatM, 1)
    m = size(MatZ, 1)
    k = n - m


    fill!(MatR, 0.0)
    @inbounds for d_ind = 1:(n-k)
        MatR[d_ind, d_ind] = -0.5
    end
    @inbounds for c_ind = 1:k
        @inbounds for r_ind = (k+1):n
            MatB[r_ind-k, c_ind] = MatM[r_ind, c_ind]
        end
    end

    mul!(MatR, MatB, MatB', 1.0 / 12, 1.0)

    @inbounds for r_ind = 1:(n-k)
        @inbounds for c_ind = 1:(n-k)
            MatC[r_ind, c_ind] = -MatM[r_ind+k, c_ind+k]
        end
    end

    copyto!(MatZ, lyap(MatR, MatC))
    getSkewSymm!(Z)
    return Z
end

get_wsp_BCH3_direction(n, m) = WSP(get_wsp_cong(m), get_wsp_dgesvd(m, n - m))

"""
    stlog_BCH3_direction_svd(Z, M, B, C, R) -> Z::Ref{Matrix{Float64}}

Compute the BCH third update Z from the sylvester equation ``Z(BB'/12 - I/2) + (BB'/12 - I/2)Z = -C``. SVD of B is expolited to obtained 
the eigensystem of `(BB'/12 - I/2)` so that the equation can be solved more efficiently. dgesvd is used for the SVD. 
"""
function stlog_BCH5_direction_svd!(Z::Ref{Matrix{Float64}}, M::Ref{Matrix{Float64}}, B::Ref{Matrix{Float64}}, BP::Ref{Matrix{Float64}}, BS::Ref{Vector{Float64}}, C::Ref{Matrix{Float64}},
    wsp_BCH3=get_wsp_BCH3_direction(size(M[], 1), size(Z[], 1)))

    MatZ = Z[]
    MatM = M[]
    MatB = B[]
    MatBP = BP[]
    MatC = C[]

    VecBS = BS[]

    wsp_cong = wsp_BCH3[1]
    wsp_dgesvd = wsp_BCH3[2]


    n::Int = size(MatM, 1)
    m::Int = size(MatC, 1)
    k::Int = n - m


    if m > k
        # B is tall-skinny and should be used for the thin SVD 
        for c_ind in axes(MatB, 2)
            for r_ind in axes(MatB, 1)
                @inbounds MatB[r_ind, c_ind] = MatM[r_ind+k, c_ind]
            end
        end


        # B = USV' -> BB' = U S^2 U', only the left vector is needed. Full vector is needed here for later computation.
        fill!(VecBS, 0.0)
        dgesvd!(B, BS, BP, 'U', 'A', wsp_dgesvd)


        # Get the transformed sylvester equation S ^ 2 / 12 - I / 2
        for ind in eachindex(VecBS)
            VecBS[ind] = (VecBS[ind]^2 - 6.0) / 12.0
        end

        for c_ind in axes(MatC, 2)
            for r_ind in axes(MatC, 1)
                @inbounds MatC[r_ind, c_ind] = MatM[r_ind+k, c_ind+k]
            end
        end


        # C ← U' C U, as U is stored, transpose is needed
        cong_dense!(C, BP, wsp_cong; trans=true)

        # Solve the simplied Sylvester equation
        for c_ind in axes(MatC, 2)
            for r_ind in (c_ind+1):m
                @inbounds MatZ[r_ind, c_ind] = MatC[r_ind, c_ind] / (VecBS[r_ind] + VecBS[c_ind])
            end
        end
        fill_upper_SkewSymm!(Z)

        # Convert the solution to the solution of the original problem Z ← U Z U', as U is stored, transpose is not needed.
        cong_dense!(Z, BP, wsp_cong; trans=false)

    else
        # -B' is tall-skinny and should be used for the thin SVD 
        for c_ind in axes(MatB, 2)
            for r_ind in axes(MatB, 1)
                @inbounds MatB[r_ind, c_ind] = MatM[r_ind, c_ind+k]
            end
        end

        # -B' = USV' -> BB' = V S^2 V', only the right vector is needed.
        dgesvd!(B, BS, BP, 'V', 'S', wsp_dgesvd)

        # Get the transformed sylvester equation S ^ 2 / 12 - I / 2
        for ind in eachindex(VecBS)
            VecBS[ind] = (VecBS[ind]^2 - 6.0) / 12.0
        end

        for c_ind in axes(MatC, 2)
            for r_ind in axes(MatC, 1)
                @inbounds MatC[r_ind, c_ind] = MatM[r_ind+k, c_ind+k]
            end
        end

        # C ← V' C V, as V' is stored, no transpose needed
        cong_dense!(C, BP, wsp_cong; trans=false)

        # Solve the simplied Sylvester equation
        for c_ind in axes(MatC, 2)
            for r_ind in (c_ind+1):m
                @inbounds MatZ[r_ind, c_ind] = MatC[r_ind, c_ind] / (VecBS[r_ind] + VecBS[c_ind])
            end
        end
        fill_upper_SkewSymm!(Z)

        # Convert the solution to the solution of the original problem Z ← V Z V', as V' is stored, transpose is needed.
        cong_dense!(Z, BP, wsp_cong; trans=true)

    end

    getSkewSymm!(Z)

    return Z
end




#######################################Test functions#######################################

using Plots

function test_stlog_newton_direction(n, k, rs=div((n - k) * (n - k - 1), 2))
    m = n - k

    MatM = rand(n, n)
    MatM .-= MatM'
    MatM .*= 0.5 / opnorm(MatM)

    MatS = zeros(n, n)
    MatΔ = zeros(n, n)

    MatZ = zeros(m, m)


    M = Ref(MatM)
    Z = Ref(MatZ)
    S = Ref(MatS)
    Δ = Ref(MatΔ)

    wsp_cong_n = get_wsp_cong(n)
    wsp_saf_n = get_wsp_saf(n)

    blk_it_nm = STRICT_LOWER_ITERATOR(n, k, n, lower_blk_traversal)
    blk_it_m = STRICT_LOWER_ITERATOR(m, lower_blk_traversal)
    blk_it_n = STRICT_LOWER_ITERATOR(n, lower_blk_traversal)



    M_saf = SAFactor(n)
    M_sys = dexp_SkewSymm_system(n)


    schurAngular_SkewSymm!(M_saf, M, wsp_saf_n; order=true, regular=true)
    compute_dexp_SkewSymm_both_system!(M_sys, M_saf.angle)

    stlog_newton_descent_backward!(Z, M, M_sys, M_saf, n, k, div((n - k) * (n - k - 1), 2), blk_it_nm, blk_it_m, blk_it_n; Stop=terminator(max(2 * rs, 200), 50000, 1e-12, 1e-9))

    fill!(MatΔ, 0.0)
    copyto!(view(MatΔ, (k+1):n, (k+1):n), MatZ)
    dexp_SkewSymm!(S, Δ, M_sys, M_saf, blk_it_n, wsp_cong_n; inv=true, cong=true)

    MatQ = exp(MatM)
    MatQetZ = similar(MatQ)

    display(MatM)
    display(MatΔ)

    display(MatS)


    println("The matrix 2 norm of S:\t $(opnorm(MatM))")

    println("dexp_{S_{A, B, C}}[S_{X, Y, C}] = exp(S_{A, B, C})S_{0, 0, Z}?\t\t", norm(MatM[(k+1):n, (k+1):n] .- MatS[(k+1):n, (k+1):n]))

    FulZ = zeros(n, n)
    copyto!(view(FulZ, (k+1):n, (k+1):n), MatZ)
    step = 1e-9


    copyto!(MatQetZ, MatQ * exp(step .* FulZ))
    MatDiff = (log(MatQetZ) .- MatM) ./ step

    println("Numerical differentiation check: |(C(t) - C) / t - C| at t = 1e-8:\t", norm(MatDiff[(k+1):n, (k+1):n] .- MatM[(k+1):n, (k+1):n]))
    # display(MatΔ[(k + 1):n, (k + 1):n] .- MatM[(k + 1):n, (k + 1):n])


    # t_grid = range(1e-7, 1e-5, 100)
    # val = Vector{Float64}(undef, length(t_grid))
    # for t_ind in eachindex(t_grid)
    #     t = t_grid[t_ind]
    #     MatQetZ .= MatQ * exp(t .* FulZ)
    #     MatΔ .= (log_SpecOrth(Ref(MatQetZ)) .- MatM) ./ t
    #     val[t_ind] = norm(MatΔ[(k+1):n, (k+1):n] .- MatM[(k+1):n, (k+1):n])^2
    # end


    # val = [norm(MatM[(k + 1):n, (k + 1):n] - (real.(log(MatQ * exp(t .* FulZ)))[(k + 1):n, (k + 1):n] .- MatM[(k + 1):n, (k + 1):n]) ./ t) for t in t_grid]
    # println("Numerical order of |(C(t) - C) / t - C|_F^2 w.r.t. t, \nwhere exp(S_{A(t), B(t), C(t)}) = exp(S_{A, B, C})exp(t ⋅ S_{0, 0, Z})\t", log.(t_grid) \ log.(val))
end

function test_stlog_newton_direciton_speed()

end

# function test_stlog_newton_gmres(n, k, rs=div((n - k) * (n - k - 1), 2))
#     m::Int = n - k
#     d::Int = div(m * (m - 1), 2)

#     MatM = rand(n, n)
#     MatM .-= MatM'
#     MatM .*= rand() * π / opnorm(MatM)

#     MatΔ1 = similar(MatM)
#     MatΔ2 = similar(MatM)
#     MatZ1 = zeros(m, m)
#     MatZ2 = zeros(m, m)

#     M = Ref(MatM)
#     Δ1 = Ref(MatΔ1)
#     Δ2 = Ref(MatΔ2)
#     Z1 = Ref(MatZ1)
#     Z2 = Ref(MatZ2)

#     M_sys = dexp_SkewSymm_system(n)
#     M_saf = SAFactor(n)

#     wsp_saf = get_wsp_saf(n)
#     wsp_bgs = get_wsp_bgs(n, k, d, rs)
#     wsp_action = wsp_bgs[3]

#     blk_it_n = STRICT_LOWER_ITERATOR(n, lower_blk_traversal)
#     blk_it_nm = STRICT_LOWER_ITERATOR(n, k, n, lower_blk_traversal)
#     blk_it_m = STRICT_LOWER_ITERATOR(m, lower_blk_traversal)

#     schurAngular_SkewSymm!(M_saf, M, wsp_saf; regular=true, order=true)
#     compute_dexp_SkewSymm_both_system!(M_sys, M_saf.angle; trans=true)

#     A = _STLOG_BACKWARD_Z_ACTION(M_sys, M_saf, blk_it_nm, blk_it_n, wsp_action)

#     stlog_newton_descent_both!(Δ1, Z1, M, M_sys, M_saf, k, rs, blk_it_nm, blk_it_m, wsp_bgs; Stop=terminator(max(rs, d), 5000, 1e-7, 1e-6))

#     stlog_newton_descent_gmres!(Δ2, Z2, M, A, rs, blk_it_m, wsp_bgs; Stop=terminator(max(rs, d), 5000, 1e-7, 1e-6))

#     println("Same Δ? \t", MatΔ1 ≈ MatΔ2)

#     println("Same Z? \t", MatZ1 ≈ MatZ2)


# end

# function test_BCH3_direction(n, k)
#     m = n - k
#     MatM = rand(n, n)
#     MatM .-= MatM'

#     MatB1 = zeros(m, k)
#     MatB2 = zeros(max(m, k), min(m, k))
#     MatC = zeros(m, m)

#     MatR = zeros(m, m)
#     VecS = zeros(m)

#     MatZ1 = zeros(m, m)
#     MatZ2 = zeros(m, m)

#     M = Ref(MatM)
#     B1 = Ref(MatB1)
#     B2 = Ref(MatB2)
#     C = Ref(MatC)

#     R = Ref(MatR)
#     S = Ref(VecS)

#     Z1 = Ref(MatZ1)
#     Z2 = Ref(MatZ2)

#     stlog_BCH3_direction_lyap!(Z1, M, B1, C, R)

#     stlog_BCH3_direction_svd!(Z2, M, B2, R, S, C)

#     println("Same answer?\t", MatZ1 ≈ MatZ2)
# end

# function test_BCH3_2k_direction_speed(k_grid, runs=10; filename="")
#     RecTime = Matrix{Float64}(undef, length(k_grid) * runs, 2)

#     k_vec = vcat(ones(runs) * k_grid'...)

#     record_ind::Int = 1
#     for k in k_grid
#         m = k
#         n = 2k

#         MatM = zeros(n, n)

#         MatB1 = zeros(m, k)
#         MatB2 = zeros(max(m, k), min(m, k))
#         MatC = zeros(m, m)

#         MatR = zeros(m, m)
#         VecS = zeros(m)

#         MatZ1 = zeros(m, m)
#         MatZ2 = zeros(m, m)

#         M = Ref(MatM)
#         B1 = Ref(MatB1)
#         B2 = Ref(MatB2)
#         C = Ref(MatC)

#         R = Ref(MatR)
#         S = Ref(VecS)

#         Z1 = Ref(MatZ1)
#         Z2 = Ref(MatZ2)

#         for r_ind in 1:runs

#             MatM .= rand(n, n)
#             MatM .-= MatM'

#             stat = @timed stlog_BCH3_direction_lyap!(Z1, M, B1, C, R)
#             RecTime[record_ind, 1] = 1000 * (stat.time - stat.gctime)

#             stat = @timed stlog_BCH3_direction_svd!(Z2, M, B2, R, S, C)
#             RecTime[record_ind, 2] = 1000 * (stat.time - stat.gctime)

#             record_ind += 1
#         end
#     end

#     time_plt = scatter(k_vec, RecTime,
#         label=["BCH3 direction, Lyap implementation" "BCH3 direction, SVD implementation"],
#         # xlabel="2-norm of the generating velocity, |S_{A,B,0}|_2",
#         ylabel="Compute time (ms)",
#         # ylims = (0.0, 8 * median(RecTime)),
#         yscale=:log2,
#         markerstrokewidth=0,
#         lw=0,
#         ms=1.5,
#         ma=0.3
#     )

#     rate_plt = scatter(k_vec, RecTime[:, 1] ./ RecTime,
#         label=:none,
#         xlabel="dimension k in 2k × k system",
#         ylabel="Ratio",
#         # ylims = (0.0, 5),
#         yscale=:log2,
#         markerstrokewidth=0,
#         lw=0,
#         ms=1.5,
#         ma=0.3
#     )

#     plt = plot(layout=(2, 1), size=(800, 600), time_plt, rate_plt)

#     display(plt)

#     if filename != ""
#         savefig(plt, filename)
#     end
# end

# function test_stlog_newton_direction_thread_speed(k_grid, runs=10; filename="")
#     RecTime = Matrix{Float64}(undef, length(k_grid) * runs, 3)

#     k_vec = vcat(ones(runs) * k_grid'...)

#     record_ind::Int = 1
#     for k in k_grid
#         m = k
#         n = 2k
#         rs = div(m * (m - 1), 2)
#         d = rs

#         MatM = zeros(n, n)

#         MatΔ1 = similar(MatM)
#         MatΔ2 = similar(MatM)

#         MatZ1 = zeros(m, m)
#         MatZ2 = zeros(m, m)
#         MatZ3 = zeros(m, m)
#         MatΔ1 = zeros(n, n)
#         MatΔ2 = zeros(n, n)
#         MatΔ3 = zeros(n, n)

#         M = Ref(MatM)

#         Z1 = Ref(MatZ1)
#         Z2 = Ref(MatZ2)
#         Z3 = Ref(MatZ3)
#         Δ1 = Ref(MatΔ1)
#         Δ2 = Ref(MatΔ2)
#         Δ3 = Ref(MatΔ3)

#         M_sys = dexp_SkewSymm_system(n)
#         M_saf = SAFactor(n)

#         wsp_saf = get_wsp_saf(n)
#         wsp_bgs = get_wsp_bgs(n, d, rs)
#         wsp_action = wsp_bgs[3]


#         blk_it_n = STRICT_LOWER_ITERATOR(n, lower_blk_traversal)
#         blk_it_nm = STRICT_LOWER_ITERATOR(n, k, n, lower_blk_traversal)
#         blk_it_m = STRICT_LOWER_ITERATOR(m, lower_blk_traversal)

#         A = _STLOG_BACKWARD_Z_ACTION(M_sys, M_saf, blk_it_nm, blk_it_n, wsp_action)

#         time::Float64 = 0.0


#         Stop = terminator(max(rs, d), 50000, 1e-7, 1e-6)

#         for r_ind in 1:runs

#             MatM .= rand(n, n)
#             MatM .-= MatM'
#             MatM .*= (0.5 + 0.6 * rand()) * π / opnorm(MatM)

#             schurAngular_SkewSymm!(M_saf, M, wsp_saf; regular=true, order=true)
#             compute_dexp_SkewSymm_both_system!(M_sys, M_saf.angle; trans=true)

#             time = 10000000
#             for s_ind = 1:10
#                 time = min(time, @elapsed stlog_newton_descent_both!(Δ1, Z1, M, M_sys, M_saf, k, rs, blk_it_nm, blk_it_m, wsp_bgs; Stop=Stop))
#             end
#             RecTime[record_ind, 1] = time * 1e3

#             time = 10000000
#             for s_ind = 1:10
#                 time = min(time, @elapsed stlog_newton_descent_both!(Δ2, Z2, M, M_sys, M_saf, k, rs, blk_it_nm, blk_it_m, blk_it_n, wsp_bgs; Stop=Stop))
#             end
#             RecTime[record_ind, 2] = time * 1e3

#             time = 10000000
#             for s_ind = 1:10
#                 time = min(time, @elapsed stlog_newton_descent_gmres!(Δ3, Z3, M, A, rs, blk_it_m, wsp_bgs; Stop=Stop))
#             end

#             RecTime[record_ind, 3] = time * 1e3

#             record_ind += 1
#         end
#     end

#     time_plt = scatter(k_vec, RecTime,
#         label=["Customize GMRES, raw loops" "Customize GMRES, threads" "IterativeSolvers GMRES, threads"],
#         # xlabel="2-norm of the generating velocity, |S_{A,B,0}|_2",
#         ylabel="Compute time (ms)",
#         # ylims = (0.0, 8 * median(RecTime)),
#         # yscale=:log2,
#         markerstrokewidth=0,
#         lw=0,
#         ms=1.5,
#         ma=0.3
#     )

#     rate_plt = scatter(k_vec, RecTime[:, 1] ./ RecTime,
#         label=:none,
#         xlabel="dimension k in 2k × k system",
#         ylabel="Ratio",
#         # ylims = (0.0, 5),
#         # yscale=:log2,
#         markerstrokewidth=0,
#         lw=0,
#         ms=1.5,
#         ma=0.3
#     )

#     plt = plot(layout=(2, 1), size=(600, 800), time_plt, rate_plt)

#     plt_log = plot(layout=(2, 1), size=(600, 800), yscale=:log2, time_plt, rate_plt)


#     display(plt)
#     display(plt_log)


#     if filename != ""
#         savefig(plt, filename)
#     end
# end

# function test_stlog_newton_implementation_speed(k_grid, runs=10; filename="")
#     RecTime = Matrix{Float64}(undef, length(k_grid) * runs, 2)

#     k_vec = vcat(ones(runs) * k_grid'...)

#     record_ind::Int = 1
#     for k in k_grid
#         m = k
#         n = 2k
#         rs = div(m * (m - 1), 2)
#         d = rs

#         MatM = zeros(n, n)

#         MatΔ1 = similar(MatM)
#         MatΔ2 = similar(MatM)

#         MatZ1 = zeros(m, m)
#         MatZ2 = zeros(m, m)
#         MatZ3 = zeros(m, m)
#         MatΔ1 = zeros(n, n)
#         MatΔ2 = zeros(n, n)
#         MatΔ3 = zeros(n, n)

#         M = Ref(MatM)

#         Z1 = Ref(MatZ1)
#         Z2 = Ref(MatZ2)
#         Z3 = Ref(MatZ3)
#         Δ1 = Ref(MatΔ1)
#         Δ2 = Ref(MatΔ2)
#         Δ3 = Ref(MatΔ3)

#         M_sys = dexp_SkewSymm_system(n)
#         M_saf = SAFactor(n)

#         wsp_saf = get_wsp_saf(n)
#         wsp_bgs = get_wsp_bgs(n, k, d, rs)
#         wsp_action = wsp_bgs[3]


#         blk_it_n = STRICT_LOWER_ITERATOR(n, lower_blk_traversal)
#         blk_it_nm = STRICT_LOWER_ITERATOR(n, k, n, lower_blk_traversal)
#         blk_it_m = STRICT_LOWER_ITERATOR(m, lower_blk_traversal)

#         A = _STLOG_BACKWARD_Z_ACTION(M_sys, M_saf, blk_it_nm, blk_it_n, wsp_action)

#         time::Float64 = 0.0


#         Stop = terminator(max(rs, d), 50000, 1e-7, 1e-6)

#         for r_ind in 1:runs

#             MatM .= rand(n, n)
#             MatM .-= MatM'
#             MatM .*= (0.5 + 0.6 * rand()) * π / opnorm(MatM)

#             schurAngular_SkewSymm!(M_saf, M, wsp_saf; regular=true, order=true)
#             compute_dexp_SkewSymm_both_system!(M_sys, M_saf.angle; trans=true)

#             # time = 10000000
#             # for s_ind = 1:10
#             #     time = min(time, @elapsed stlog_newton_descent_both!(Δ1, Z1, M, M_sys, M_saf, k, rs, blk_it_nm, blk_it_m, wsp_bgs; Stop=Stop))
#             # end
#             # RecTime[record_ind, 1] = time * 1e3

#             time = 10000000
#             for s_ind = 1:10
#                 time = min(time, @elapsed stlog_newton_descent_both!(Δ2, Z2, M, M_sys, M_saf, k, rs, blk_it_nm, blk_it_m, blk_it_n, wsp_bgs; Stop=Stop))
#             end
#             RecTime[record_ind, 2] = time * 1e3

#             time = 10000000
#             for s_ind = 1:10
#                 time = min(time, @elapsed stlog_newton_descent_gmres!(Δ3, Z3, M, A, rs, blk_it_m, wsp_bgs; Stop=Stop))
#             end

#             RecTime[record_ind, 1] = time * 1e3

#             record_ind += 1
#         end
#     end

#     time_plt = scatter(k_vec, RecTime,
#         label=["IterativeSolvers GMRES, threads" "Customize GMRES, threads"],
#         # xlabel="2-norm of the generating velocity, |S_{A,B,0}|_2",
#         ylabel="Compute time (ms)",
#         # ylims = (0.0, 8 * median(RecTime)),
#         # yscale=:log2,
#         markerstrokewidth=0,
#         lw=0,
#         ms=1.5,
#         ma=0.3
#     )

#     rate_plt = scatter(k_vec, RecTime[:, 1] ./ RecTime,
#         label=:none,
#         xlabel="dimension k in 2k × k system",
#         ylabel="Ratio",
#         # ylims = (0.0, 5),
#         # yscale=:log2,
#         markerstrokewidth=0,
#         lw=0,
#         ms=1.5,
#         ma=0.3
#     )

#     plt = plot(layout=(2, 1), size=(600, 800), time_plt, rate_plt)

#     plt_log = plot(layout=(2, 1), size=(600, 800), yscale=:log2, time_plt, rate_plt)


#     display(plt)
#     display(plt_log)


#     if filename != ""
#         savefig(plt, filename)
#     end
# end

# function test_stlog_newton_bch3_speed(k_grid, runs=10; filename="")
#     RecTime = Matrix{Float64}(undef, length(k_grid) * runs, 2)

#     k_vec = vcat(ones(runs) * k_grid'...)

#     record_ind::Int = 1
#     for k in k_grid
#         m = k
#         n = 2k
#         rs = div(m * (m - 1), 2)
#         d = rs

#         MatM = zeros(n, n)

#         MatΔ1 = similar(MatM)
#         MatΔ2 = similar(MatM)

#         MatZ1 = zeros(m, m)
#         MatZ2 = zeros(m, m)
#         MatZ3 = zeros(m, m)
#         MatΔ1 = zeros(n, n)
#         MatΔ2 = zeros(n, n)
#         MatΔ3 = zeros(n, n)

#         MatB = zeros(m, k)
#         MatC = zeros(m, m)

#         MatR = zeros(m, m)
#         VecS = zeros(m)

#         M = Ref(MatM)


#         Z1 = Ref(MatZ1)
#         Z2 = Ref(MatZ2)
#         Z3 = Ref(MatZ3)
#         Δ1 = Ref(MatΔ1)
#         Δ2 = Ref(MatΔ2)
#         Δ3 = Ref(MatΔ3)

#         B = Ref(MatB)
#         C = Ref(MatC)

#         R = Ref(MatR)
#         S = Ref(VecS)

#         M_sys = dexp_SkewSymm_system(n)
#         M_saf = SAFactor(n)

#         wsp_saf = get_wsp_saf(n)
#         wsp_bgs = get_wsp_bgs(n, k, d, rs)
#         wsp_action = wsp_bgs[3]


#         blk_it_n = STRICT_LOWER_ITERATOR(n, lower_blk_traversal)
#         blk_it_nm = STRICT_LOWER_ITERATOR(n, k, n, lower_blk_traversal)
#         blk_it_m = STRICT_LOWER_ITERATOR(m, lower_blk_traversal)

#         A = _STLOG_BACKWARD_Z_ACTION(M_sys, M_saf, blk_it_nm, blk_it_n, wsp_action)

#         time::Float64 = 0.0


#         Stop = terminator(max(rs, d), 50000, 1e-7, 1e-6)

#         for r_ind in 1:runs

#             MatM .= rand(n, n)
#             MatM .-= MatM'
#             MatM .*= (0.5 + 0.6 * rand()) * π / opnorm(MatM)

#             schurAngular_SkewSymm!(M_saf, M, wsp_saf; regular=true, order=true)
#             compute_dexp_SkewSymm_both_system!(M_sys, M_saf.angle; trans=true)

#             # time = 10000000
#             # for s_ind = 1:10
#             #     time = min(time, @elapsed stlog_newton_descent_both!(Δ1, Z1, M, M_sys, M_saf, k, rs, blk_it_nm, blk_it_m, wsp_bgs; Stop=Stop))
#             # end
#             # RecTime[record_ind, 1] = time * 1e3

#             time = 10000000
#             for s_ind = 1:10
#                 time = min(time, @elapsed stlog_BCH3_direction_lyap!(Z1, M, B, C, R);)
#             end
#             RecTime[record_ind, 1] = time * 1e3

#             time = 10000000
#             for s_ind = 1:10
#                 time = min(time, @elapsed stlog_newton_descent_both!(Δ2, Z2, M, M_sys, M_saf, k, rs, blk_it_nm, blk_it_m, blk_it_n, wsp_bgs; Stop=Stop))
#             end
#             RecTime[record_ind, 2] = time * 1e3



#             record_ind += 1
#         end
#     end

#     time_plt = scatter(k_vec, RecTime,
#         label=["BCH3 direction, Lyapunov solver" "Newton direction, GMRES solver"],
#         # xlabel="2-norm of the generating velocity, |S_{A,B,0}|_2",
#         ylabel="Compute time (ms)",
#         # ylims = (0.0, 8 * median(RecTime)),
#         # yscale=:log2,
#         markerstrokeshape=[:circle :star5],
#         markerstrokecolor=:auto,
#         lw=0,
#         ms=1.5,
#         ma=0.5
#     )

#     rate_plt = scatter(k_vec, RecTime[:, 1] ./ RecTime,
#         label=:none,
#         xlabel="dimension k in 2k × k system",
#         ylabel="Ratio of time to the BCH3 direction solver",
#         # ylims = (0.0, 5),
#         # yscale=:log2,
#         markerstrokeshape=[:circle :star5],
#         markerstrokecolor=:auto,
#         lw=0,
#         ms=1.5,
#         ma=0.5
#     )

#     plt = plot(layout=(2, 1), size=(600, 800), time_plt, rate_plt)

#     plt_log = plot(layout=(2, 1), size=(600, 800), yscale=:log2, time_plt, rate_plt)


#     display(plt)
#     display(plt_log)


#     if filename != ""
#         pos = findlast('.', filename)
#         savefig(plot(time_plt), filename[1:(pos-1)] * "_time." * filename[(pos+1):end])
#         savefig(plot(rate_plt), filename[1:(pos-1)] * "_rate." * filename[(pos+1):end])
#         savefig(plot(yscale=:log2, time_plt), filename[1:(pos-1)] * "_time_logscale." * filename[(pos+1):end])
#         savefig(plot(yscale=:log2, rate_plt), filename[1:(pos-1)] * "_rate_action_logscale." * filename[(pos+1):end])
#     end
# end