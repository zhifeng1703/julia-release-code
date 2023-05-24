include("stlog_geometry.jl")
include("stlog_init_guess.jl")
include("stlog_linesearch.jl")

include("stlog_descent.jl")

include("stlog_prob_process.jl")


# include(homedir() * "/Documents/julia/inc/BLAS_setting/blas_main.jl")

using Printf

BCH_MAX_ITER = 20
BCH_ABSTOL = 1e-4
BCH_SHUTDOWN = -10

GMRES_RS = 50

NMLS_SET = NMLS_Paras(0.1, 20.0, 0.8, 0.3, 0)

SOLVER_STOP = terminator(50, 500, 1e-6, 1e-4)

NEARLOG_THRESHOLD = 2e-1
RESTART_THRESHOLD = 2e-1
DIRECTION_THRESHOLD = 2.0

FAIL_STEP = 2.0

ENABLE_NEARLOG = false
ENABLE_RESTART_BCH = true




default_stepsize_failure(nZ) = min(1.0, 3π / (4 * nZ));
# At most π / 2 when escaping

##########################workspace###########################

function get_wsp_bch1(n, k, MaxIter)

    m::Int = n - k
    n_b::Int = div(n, 2)

    MatU::Matrix{Float64} = Matrix{Float64}(undef, n, n)
    MatM::Matrix{Float64} = Matrix{Float64}(undef, n, n)
    MatB::Matrix{Float64} = Matrix{Float64}(undef, m, k)
    MatC::Matrix{Float64} = Matrix{Float64}(undef, m, m)
    MatR::Matrix{Float64} = Matrix{Float64}(undef, m, m)
    MatZ::Matrix{Float64} = Matrix{Float64}(undef, m, m)
    MatQ::Matrix{Float64} = Matrix{Float64}(undef, m, m)

    Z_saf = SAFactor(m)
    M_saf = SAFactor(n)

    wsp_saf_m::WSP = get_wsp_saf(m)
    wsp_saf_n::WSP = get_wsp_saf(n)
    wsp_ret_UpZ::WSP = get_wsp_stlog_UpZ_ret(n, k, wsp_saf_m, wsp_saf_n)

    return WSP(MatU, MatM, MatB, MatC, MatR, MatZ, MatQ, Z_saf, M_saf, wsp_saf_m, wsp_saf_n, wsp_ret_UpZ)
end

function get_wsp_bch3(n, k, MaxIter)

    m::Int = n - k
    n_b::Int = div(n, 2)

    MatU::Matrix{Float64} = Matrix{Float64}(undef, n, n)
    MatM::Matrix{Float64} = Matrix{Float64}(undef, n, n)
    MatB::Matrix{Float64} = Matrix{Float64}(undef, m, k)
    MatC::Matrix{Float64} = Matrix{Float64}(undef, m, m)
    MatR::Matrix{Float64} = Matrix{Float64}(undef, m, m)
    MatZ::Matrix{Float64} = Matrix{Float64}(undef, m, m)
    MatQ::Matrix{Float64} = Matrix{Float64}(undef, m, m)

    Z_saf = SAFactor(m)
    M_saf = SAFactor(n)


    # Initialize workspace

    wsp_saf_m::WSP = get_wsp_saf(m)
    wsp_saf_n::WSP = get_wsp_saf(n)
    wsp_ret_UpZ::WSP = get_wsp_stlog_UpZ_ret(n, k, wsp_saf_m, wsp_saf_n)

    return WSP(MatU, MatM, MatB, MatC, MatR, MatZ, MatQ, Z_saf, M_saf, wsp_saf_m, wsp_saf_n, wsp_ret_UpZ)
end

function get_wsp_hybrid(n, k, MaxIter)

    m::Int = n - k
    n_b::Int = div(n, 2)

    n_dim::Int = div(n * (n - 1), 2)
    m_dim::Int = div(m * (m - 1), 2)
    restart::Int = m_dim

    MatU::Matrix{Float64} = Matrix{Float64}(undef, n, n)
    MatUp_new::Matrix{Float64} = Matrix{Float64}(undef, n, m)
    MatM::Matrix{Float64} = Matrix{Float64}(undef, n, n)
    MatM_new::Matrix{Float64} = Matrix{Float64}(undef, n, n)
    MatΔ::Matrix{Float64} = Matrix{Float64}(undef, n, n)
    MatB::Matrix{Float64} = Matrix{Float64}(undef, m, k)
    MatC::Matrix{Float64} = Matrix{Float64}(undef, m, m)
    MatR::Matrix{Float64} = Matrix{Float64}(undef, m, m)
    MatZ::Matrix{Float64} = Matrix{Float64}(undef, m, m)
    MatαZ::Matrix{Float64} = Matrix{Float64}(undef, m, m)



    cost_record::Vector{Float64} = zeros(MaxIter)

    Z_saf = SAFactor(m)
    M_saf = SAFactor(n)
    Δ_saf = SAFactor(n)

    M_sys = dexp_SkewSymm_system(n)

    blk_it_m = STRICT_LOWER_ITERATOR(m, lower_blk_traversal)
    blk_it_nm = STRICT_LOWER_ITERATOR(n, k, n, lower_blk_traversal)
    blk_it_n = STRICT_LOWER_ITERATOR(n, lower_blk_traversal)

    wsp_saf_m::WSP = get_wsp_saf(m)
    wsp_saf_n::WSP = get_wsp_saf(n)
    wsp_UpZ_ret::WSP = get_wsp_stlog_UpZ_ret(n, k, wsp_saf_m, wsp_saf_n)
    wsp_bgs::WSP = get_wsp_bgs(n, k, m_dim, m_dim)

    return WSP(MatU, MatUp_new, MatM, MatM_new, MatΔ, MatB, MatC, MatR, MatZ, MatαZ, cost_record, Z_saf, M_saf, Δ_saf, M_sys, blk_it_m, blk_it_nm, blk_it_n, wsp_saf_m, wsp_saf_n, wsp_UpZ_ret, wsp_bgs)
end

function get_wsp_newton(n, k, MaxIter)

    m::Int = n - k
    n_b::Int = div(n, 2)

    n_dim::Int = div(n * (n - 1), 2)
    m_dim::Int = div(m * (m - 1), 2)
    restart::Int = m_dim

    MatU::Matrix{Float64} = Matrix{Float64}(undef, n, n)
    MatUp_new::Matrix{Float64} = Matrix{Float64}(undef, n, m)
    MatM::Matrix{Float64} = Matrix{Float64}(undef, n, n)
    MatM_new::Matrix{Float64} = Matrix{Float64}(undef, n, n)
    MatΔ::Matrix{Float64} = Matrix{Float64}(undef, n, n)
    MatB::Matrix{Float64} = Matrix{Float64}(undef, m, k)
    MatC::Matrix{Float64} = Matrix{Float64}(undef, m, m)
    MatR::Matrix{Float64} = Matrix{Float64}(undef, m, m)
    MatZ::Matrix{Float64} = Matrix{Float64}(undef, m, m)
    MatαZ::Matrix{Float64} = Matrix{Float64}(undef, m, m)



    cost_record::Vector{Float64} = zeros(MaxIter)

    Z_saf = SAFactor(m)
    M_saf = SAFactor(n)
    Δ_saf = SAFactor(n)

    M_sys = dexp_SkewSymm_system(n)

    blk_it_m = STRICT_LOWER_ITERATOR(m, lower_blk_traversal)
    blk_it_nm = STRICT_LOWER_ITERATOR(n, k, n, lower_blk_traversal)
    blk_it_n = STRICT_LOWER_ITERATOR(n, lower_blk_traversal)

    wsp_saf_m::WSP = get_wsp_saf(m)
    wsp_saf_n::WSP = get_wsp_saf(n)
    wsp_UpZ_ret::WSP = get_wsp_stlog_UpZ_ret(n, k, wsp_saf_m, wsp_saf_n)
    wsp_bgs::WSP = get_wsp_bgs(n, k, m_dim, m_dim)

    return WSP(MatU, MatUp_new, MatM, MatM_new, MatΔ, MatB, MatC, MatR, MatZ, MatαZ, cost_record, Z_saf, M_saf, Δ_saf, M_sys, blk_it_m, blk_it_nm, blk_it_n, wsp_saf_m, wsp_saf_n, wsp_UpZ_ret, wsp_bgs)
end

function get_wsp_alg(n, k, MaxIter, alg)
    if alg == stlog_BCH1_2k_analysis || alg == stlog_BCH1_2k || alg == stlog_BCH1_2k_naive
        return get_wsp_bch1(n, k, MaxIter);
    elseif alg == stlog_BCH3_2k_analysis || alg == stlog_BCH3_2k || alg == stlog_BCH3_2k_naive
        return get_wsp_bch3(n, k, MaxIter);
    elseif alg == stlog_hybrid_Newton_armijo_analysis || alg == stlog_hybrid_Newton_armijo
        return get_wsp_hybrid(n, k, MaxIter)
    elseif alg == stlog_Newton_armijo_analysis || alg == stlog_Newton_armijo
        return get_wsp_newton(n, k, MaxIter)
    else
        throw("Algorithm not recognized!")
    end
end
########################core algorithm########################

function stlog_BCH1_2k_analysis(MatUk::Matrix{Float64}, MatUp::Matrix{Float64}; 
    Stop=terminator(300, 10000, 1e-8, 1e-6), Solver_Stop=nothing, Init=nothing, Records=nothing, NMLS_Set=nothing, wsp = get_wsp_bch1(size(MatUk)..., Stop.MaxIter))

    # msgln("Start BCHSolver\n");

    n::Int, k::Int = size(MatUk)
    m::Int = n - k
    n_b::Int = div(n, 2)


    # This must be used after preprocessing_2k that turn (n, k) problem into (2k, k) problem

    time_record::Vector{Float64} = Records[1][]
    cost_record::Vector{Float64} = Records[2][]
    dist_record::Vector{Float64} = Records[3][]
    vect_record::Vector{Float64} = Records[4][]
    step_record::Vector{Float64} = Records[5][]
    angs_record = Records[6][]
    stpt_record = Records[7][]


    MatU::Matrix{Float64} = wsp[1]
    MatM::Matrix{Float64} = wsp[2]
    MatB::Matrix{Float64} = wsp[3]
    MatC::Matrix{Float64} = wsp[4]
    MatR::Matrix{Float64} = wsp[5]
    MatZ::Matrix{Float64} = wsp[6]
    MatQ::Matrix{Float64} = wsp[7]

    Z_saf::SAFactor = wsp[8]
    M_saf::SAFactor = wsp[9]

    wsp_saf_m::WSP = wsp[10]
    wsp_saf_n::WSP = wsp[11]
    wsp_ret_UpZ::WSP = wsp[12]


    U = Ref(MatU)
    Up = Ref(MatUp)
    M = Ref(MatM)
    Z = Ref(MatZ)
    M_vec = M_saf.vector
    M_ang = M_saf.angle
    Z_vec = Z_saf.vector
    Z_ang = Z_saf.angle

    ptrUk = pointer(MatUk)
    ptrUp = pointer(MatUp)
    ptrU = pointer(MatU)

    # Initialize workspace


    iter::Int = 0
    abserr::Float64 = -1.0
    result_flag::Int = 0


    stats = @timed begin
        # U[:, 1:k] .= Uk;
        unsafe_copyto!(ptrU, ptrUk, n * k)

        if Init !== nothing
            MatUp .= Init(MatUk)
        end
        unsafe_copyto!(ptrU + n * k * sizeof(Float64), ptrUp, n * m)
        # U[:, (k + 1):n] .= Up
    end
    time_record[1] = (stats.time - stats.gctime) * 1000
    # completion

    stats = @timed begin

        log_SpecOrth!(M, M_saf, U, wsp_saf_n; order=false, regular=false)

        for r_ind in 1:m
            for c_ind in 1:m
                @inbounds MatZ[r_ind, c_ind] = -MatM[r_ind+k, c_ind+k]
            end
        end

        iter = 1
        abserr = stlog_cost(M, k)
    end


    if stpt_record !== nothing
        stpt_record[iter] = copy(MatU)
    end

    if angs_record !== nothing
        SAFactor_order(M_saf, wsp_saf_n)
        SAFactor_regularize(M_saf, wsp_saf_n)
        angs_record[iter, :] .= M_saf.angle[]
    end

    time_record[iter] += (stats.time - stats.gctime) * 1000
    # cost_record[iter] = norm(C) ^ 2 / 2.0;
    cost_record[iter] = abserr
    dist_record[iter] = sqrt(inner_skew!(M, M) - 2 * abserr)


    while result_flag == 0
        stats = @timed begin

            # As BCH does not make use of the angles information of the update,
            # real matrix exponential is used here, which is faster than the SAF approach.

            # ret_UpZ!(U, Up, M, M_saf, Z, Z_saf, wsp_ret_UpZ; nearlog=false)
            ret_UpZ_builtin_exp!(U, Up, M, M_saf, Z, wsp_ret_UpZ; nearlog=false)

            for r_ind in 1:m
                for c_ind in 1:m
                    @inbounds MatZ[r_ind, c_ind] = -MatM[r_ind+k, c_ind+k]
                end
            end

            iter += 1
            abserr = stlog_cost(M, k)
        end

        if stpt_record !== nothing
            stpt_record[iter] = copy(MatU)
        end

        if angs_record !== nothing
            SAFactor_order(M_saf, wsp_saf_n)
            SAFactor_regularize(M_saf, wsp_saf_n)
            angs_record[iter, :] .= M_saf.angle[]
        end

        time_record[iter] = time_record[iter-1] + (stats.time - stats.gctime) * 1000
        # time_record[iter] = time_record[iter - 1] + (stats.time - 0 * stats.gctime) * 1000;
        cost_record[iter] = abserr
        dist_record[iter] = sqrt(inner_skew!(M, M) - 2 * abserr)
        vect_record[iter-1] = sqrt(inner_skew!(Z, Z))

        result_flag = check_termination_vec(cost_record, nothing, vect_record, time_record, step_record, iter, Stop)
        # msgln(("\t\tIteration:  ", iter - 1, "\tAbsolute error:  ", round(abserr, digits=14), "\n"), true)
    end
    return MatM, result_flag, iter, time_record[iter], cost_record[iter], dist_record[iter], vect_record[iter-1]
end

function stlog_BCH1_2k(MatUk::Matrix{Float64}, MatUp::Matrix{Float64}; 
    Stop=terminator(300, 10000, 1e-8, 1e-6), Solver_Stop=nothing, Init=nothing, NMLS_Set=nothing, wsp = get_wsp_bch1(size(MatUk)..., Stop.MaxIter))

    # msgln("Start BCHSolver\n");

    n::Int, k::Int = size(MatUk)
    m::Int = n - k
    n_b::Int = div(n, 2)

    MatU::Matrix{Float64} = wsp[1]
    MatM::Matrix{Float64} = wsp[2]
    MatB::Matrix{Float64} = wsp[3]
    MatC::Matrix{Float64} = wsp[4]
    MatR::Matrix{Float64} = wsp[5]
    MatZ::Matrix{Float64} = wsp[6]
    MatQ::Matrix{Float64} = wsp[7]

    Z_saf::SAFactor = wsp[8]
    M_saf::SAFactor = wsp[9]

    wsp_saf_m::WSP = wsp[10]
    wsp_saf_n::WSP = wsp[11]
    wsp_ret_UpZ::WSP = wsp[12]

    U = Ref(MatU)
    Up = Ref(MatUp)
    M = Ref(MatM)
    Z = Ref(MatZ)
    M_vec = M_saf.vector
    M_ang = M_saf.angle
    Z_vec = Z_saf.vector
    Z_ang = Z_saf.angle

    ptrUk = pointer(MatUk)
    ptrUp = pointer(MatUp)
    ptrU = pointer(MatU)


    iter::Int = 0
    abserr::Float64 = -1.0
    result_flag::Int = 0

    # U[:, 1:k] .= Uk;
    unsafe_copyto!(ptrU, ptrUk, n * k)

    if Init !== nothing
        MatUp .= Init(MatUk)
    end
    unsafe_copyto!(ptrU + n * k * sizeof(Float64), ptrUp, n * m)
    # U[:, (k + 1):n] .= Up
    # completion


    log_SpecOrth!(M, M_saf, U, wsp_saf_n; order=false, regular=false)

    for r_ind in 1:m
        for c_ind in 1:m
            @inbounds MatZ[r_ind, c_ind] = -MatM[r_ind+k, c_ind+k]
        end
    end

    iter = 1
    abserr = stlog_cost(M, k)


    while result_flag == 0

        # As BCH does not make use of the angles information of the update,
        # real matrix exponential is used here, which is faster than the SAF approach.

        # ret_UpZ!(U, Up, M, M_saf, Z, Z_saf, wsp_ret_UpZ; nearlog=false)
        ret_UpZ_builtin_exp!(U, Up, M, M_saf, Z, wsp_ret_UpZ; nearlog=false)

        for r_ind in 1:m
            for c_ind in 1:m
                @inbounds MatZ[r_ind, c_ind] = -MatM[r_ind+k, c_ind+k]
            end
        end

        iter += 1
        abserr = stlog_cost(M, k)


        result_flag = check_termination_val(abserr, nothing, nothing, nothing, nothing, iter, Stop)
        # msgln(("\t\tIteration:  ", iter - 1, "\tAbsolute error:  ", round(abserr, digits=14), "\n"), true)
    end
    return MatM, result_flag, iter
end

function stlog_BCH1_2k_naive(MatUk::Matrix{Float64}, MatUp::Matrix{Float64}; 
    Stop=terminator(300, 10000, 1e-8, 1e-6), Solver_Stop=nothing, Init=nothing, NMLS_Set=nothing, wsp = get_wsp_bch1(size(MatUk)..., Stop.MaxIter))

    # msgln("Start BCHSolver\n");

    n::Int, k::Int = size(MatUk)
    m::Int = n - k
    n_b::Int = div(n, 2)

    MatU::Matrix{Float64} = wsp[1]
    MatM::Matrix{Float64} = wsp[2]
    MatB::Matrix{Float64} = wsp[3]
    MatC::Matrix{Float64} = wsp[4]
    MatR::Matrix{Float64} = wsp[5]
    MatZ::Matrix{Float64} = wsp[6]
    MatQ::Matrix{Float64} = wsp[7]

    Z_saf::SAFactor = wsp[8]
    M_saf::SAFactor = wsp[9]

    wsp_saf_m::WSP = wsp[10]
    wsp_saf_n::WSP = wsp[11]
    wsp_ret_UpZ::WSP = wsp[12]


    U = Ref(MatU)
    Up = Ref(MatUp)
    M = Ref(MatM)
    Z = Ref(MatZ)
    M_vec = M_saf.vector
    M_ang = M_saf.angle
    Z_vec = Z_saf.vector
    Z_ang = Z_saf.angle

    ptrUk = pointer(MatUk)
    ptrUp = pointer(MatUp)
    ptrU = pointer(MatU)

    MatUpTmp = similar(MatUp)
    ptrUpTmp = pointer(MatUpTmp)

    # Initialize workspace



    iter::Int = 0
    abserr::Float64 = -1.0
    result_flag::Int = 0

    # U[:, 1:k] .= Uk;
    unsafe_copyto!(ptrU, ptrUk, n * k)

    if Init !== nothing
        MatUp .= Init(MatUk)
    end
    unsafe_copyto!(ptrU + n * k * sizeof(Float64), ptrUp, n * m)
    # U[:, (k + 1):n] .= Up
    # completion


    MatM .= real.(log(MatU))

    for r_ind in 1:m
        for c_ind in 1:m
            @inbounds MatZ[r_ind, c_ind] = -MatM[r_ind+k, c_ind+k]
        end
    end

    iter = 1
    abserr = stlog_cost(M, k)


    while result_flag == 0
        # As BCH does not make use of the angles information of the update,
        # real matrix exponential is used here, which is faster than the SAF approach.

        # ret_UpZ!(U, Up, M, M_saf, Z, Z_saf, wsp_ret_UpZ; nearlog=false)
        ret_UpZ_builtin_explog!(U, Up, M, Z, wsp_ret_UpZ)

        for r_ind in 1:m
            for c_ind in 1:m
                @inbounds MatZ[r_ind, c_ind] = -MatM[r_ind+k, c_ind+k]
            end
        end

        iter += 1
        abserr = stlog_cost(M, k)


        result_flag = check_termination_val(abserr, nothing, nothing, nothing, nothing, iter, Stop)
        # msgln(("\t\tIteration:  ", iter - 1, "\tAbsolute error:  ", round(abserr, digits=14), "\n"), true)
    end
    return MatM, result_flag, iter
end

function stlog_BCH3_2k_analysis(MatUk::Matrix{Float64}, MatUp::Matrix{Float64}; 
    Stop=terminator(300, 10000, 1e-8, 1e-6), Solver_Stop=nothing, Init=nothing, Records=nothing, NMLS_Set=nothing, wsp = get_wsp_bch3(size(MatUk)..., Stop.MaxIter))

    # msgln("Start BCHSolver\n");

    n::Int, k::Int = size(MatUk)
    m::Int = n - k
    n_b::Int = div(n, 2)


    # This must be used after preprocessing_2k that turn (n, k) problem into (2k, k) problem

    time_record::Vector{Float64} = Records[1][]
    cost_record::Vector{Float64} = Records[2][]
    dist_record::Vector{Float64} = Records[3][]
    vect_record::Vector{Float64} = Records[4][]
    step_record::Vector{Float64} = Records[5][]
    angs_record = Records[6][]
    stpt_record = Records[7][]


    MatU::Matrix{Float64} = wsp[1]
    MatM::Matrix{Float64} = wsp[2]
    MatB::Matrix{Float64} = wsp[3]
    MatC::Matrix{Float64} = wsp[4]
    MatR::Matrix{Float64} = wsp[5]
    MatZ::Matrix{Float64} = wsp[6]
    MatQ::Matrix{Float64} = wsp[7]

    Z_saf::SAFactor = wsp[8]
    M_saf::SAFactor = wsp[9]

    wsp_saf_m::WSP = wsp[10]
    wsp_saf_n::WSP = wsp[11]
    wsp_ret_UpZ::WSP = wsp[12]


    U = Ref(MatU)
    Up = Ref(MatUp)
    M = Ref(MatM)
    Z = Ref(MatZ)
    B = Ref(MatB)
    C = Ref(MatC)
    R = Ref(MatR)

    M_vec = M_saf.vector
    M_ang = M_saf.angle
    Z_vec = Z_saf.vector
    Z_ang = Z_saf.angle

    ptrUk = pointer(MatUk)
    ptrUp = pointer(MatUp)
    ptrU = pointer(MatU)


    iter::Int = 0
    abserr::Float64 = -1.0
    result_flag::Int = 0

    stats = @timed begin
        # U[:, 1:k] .= Uk;
        unsafe_copyto!(ptrU, ptrUk, n * k)

        if Init !== nothing
            MatUp .= Init(MatUk)
        end
        unsafe_copyto!(ptrU + n * k * sizeof(Float64), ptrUp, n * m)
        # U[:, (k + 1):n] .= Up
    end
    time_record[1] = (stats.time - stats.gctime) * 1000
    # completion

    stats = @timed begin
        # As BCH does not make use of the angles information of the update,
        # real matrix exponential is used here, which is faster than the SAF approach.

        # ret_UpZ!(U, Up, M, M_saf, Z, Z_saf, wsp_ret_UpZ; nearlog=false)

        log_SpecOrth!(M, M_saf, U, wsp_saf_n; order=false, regular=false)

        stlog_BCH3_direction_lyap!(Z, M, B, C, R)

        iter = 1
        abserr = stlog_cost(M, k)
    end


    if stpt_record !== nothing
        stpt_record[iter] = copy(MatU)
    end

    if angs_record !== nothing
        SAFactor_order(M_saf, wsp_saf_n)
        SAFactor_regularize(M_saf, wsp_saf_n)
        angs_record[iter, :] .= M_saf.angle[]
    end

    time_record[iter] += (stats.time - stats.gctime) * 1000
    # cost_record[iter] = norm(C) ^ 2 / 2.0;
    cost_record[iter] = abserr
    dist_record[iter] = sqrt(inner_skew!(M, M) - 2 * abserr)


    while result_flag == 0
        stats = @timed begin

            ret_UpZ_builtin_exp!(U, Up, M, M_saf, Z, wsp_ret_UpZ)

            stlog_BCH3_direction_lyap!(Z, M, B, C, R)

            iter += 1
            abserr = stlog_cost(M, k)
        end

        if stpt_record !== nothing
            stpt_record[iter] = copy(MatU)
        end

        if angs_record !== nothing
            SAFactor_order(M_saf, wsp_saf_n)
            SAFactor_regularize(M_saf, wsp_saf_n)
            angs_record[iter, :] .= M_saf.angle[]
        end

        time_record[iter] = time_record[iter-1] + (stats.time - stats.gctime) * 1000
        # time_record[iter] = time_record[iter - 1] + (stats.time - 0 * stats.gctime) * 1000;
        cost_record[iter] = abserr
        dist_record[iter] = sqrt(inner_skew!(M, M) - 2 * abserr)
        vect_record[iter-1] = sqrt(inner_skew!(Z, Z))

        result_flag = check_termination_vec(cost_record, nothing, vect_record, time_record, step_record, iter, Stop)
        # msgln(("\t\tIteration:  ", iter - 1, "\tAbsolute error:  ", round(abserr, digits=14), "\n"), true)
    end
    return MatM, result_flag, iter, time_record[iter], cost_record[iter], dist_record[iter], vect_record[iter-1]
end

function stlog_BCH3_2k(MatUk::Matrix{Float64}, MatUp::Matrix{Float64}; 
    Stop=terminator(300, 10000, 1e-8, 1e-6), Solver_Stop=nothing, Init=nothing, NMLS_Set=nothing, wsp = get_wsp_bch3(size(MatUk)..., Stop.MaxIter))

    # msgln("Start BCHSolver\n");

    n::Int, k::Int = size(MatUk)
    m::Int = n - k
    n_b::Int = div(n, 2)


    # This must be used after preprocessing_2k that turn (n, k) problem into (2k, k) problem

    MatU::Matrix{Float64} = wsp[1]
    MatM::Matrix{Float64} = wsp[2]
    MatB::Matrix{Float64} = wsp[3]
    MatC::Matrix{Float64} = wsp[4]
    MatR::Matrix{Float64} = wsp[5]
    MatZ::Matrix{Float64} = wsp[6]
    MatQ::Matrix{Float64} = wsp[7]

    Z_saf::SAFactor = wsp[8]
    M_saf::SAFactor = wsp[9]

    wsp_saf_m::WSP = wsp[10]
    wsp_saf_n::WSP = wsp[11]
    wsp_ret_UpZ::WSP = wsp[12]


    U = Ref(MatU)
    Up = Ref(MatUp)
    M = Ref(MatM)
    Z = Ref(MatZ)
    B = Ref(MatB)
    C = Ref(MatC)
    R = Ref(MatR)


    M_vec = M_saf.vector
    M_ang = M_saf.angle
    Z_vec = Z_saf.vector
    Z_ang = Z_saf.angle

    ptrUk = pointer(MatUk)
    ptrUp = pointer(MatUp)
    ptrU = pointer(MatU)



    iter::Int = 0
    abserr::Float64 = -1.0
    result_flag::Int = 0


    # U[:, 1:k] .= Uk;
    unsafe_copyto!(ptrU, ptrUk, n * k)

    if Init !== nothing
        MatUp .= Init(MatUk)
    end
    unsafe_copyto!(ptrU + n * k * sizeof(Float64), ptrUp, n * m)
    # U[:, (k + 1):n] .= Up
    # completion


    log_SpecOrth!(M, M_saf, U, wsp_saf_n; order=false, regular=false)

    stlog_BCH3_direction_lyap!(Z, M, B, C, R)

    iter = 1
    abserr = stlog_cost(M, k)


    while result_flag == 0

        # As BCH does not make use of the angles information of the update,
        # real matrix exponential is used here, which is faster than the SAF approach.

        # ret_UpZ!(U, Up, M, M_saf, Z, Z_saf, wsp_ret_UpZ; nearlog=false)

        ret_UpZ_builtin_exp!(U, Up, M, M_saf, Z, wsp_ret_UpZ)

        stlog_BCH3_direction_lyap!(Z, M, B, C, R)

        iter += 1
        abserr = stlog_cost(M, k)


        result_flag = check_termination_val(abserr, nothing, nothing, nothing, nothing, iter, Stop)
        # msgln(("\t\tIteration:  ", iter - 1, "\tAbsolute error:  ", round(abserr, digits=14), "\n"), true)
    end
    return MatM, result_flag, iter
end

function stlog_BCH3_2k_naive(MatUk::Matrix{Float64}, MatUp::Matrix{Float64}; 
    Stop=terminator(300, 10000, 1e-8, 1e-6), Solver_Stop=nothing, Init=nothing, NMLS_Set=nothing, wsp = get_wsp_bch3(size(MatUk)..., Stop.MaxIter))

    # msgln("Start BCHSolver\n");

    n::Int, k::Int = size(MatUk)
    m::Int = n - k
    n_b::Int = div(n, 2)

    MatU::Matrix{Float64} = wsp[1]
    MatM::Matrix{Float64} = wsp[2]
    MatB::Matrix{Float64} = wsp[3]
    MatC::Matrix{Float64} = wsp[4]
    MatR::Matrix{Float64} = wsp[5]
    MatZ::Matrix{Float64} = wsp[6]
    MatQ::Matrix{Float64} = wsp[7]

    Z_saf::SAFactor = wsp[8]
    M_saf::SAFactor = wsp[9]

    wsp_saf_m::WSP = wsp[10]
    wsp_saf_n::WSP = wsp[11]
    wsp_ret_UpZ::WSP = wsp[12]


    U = Ref(MatU)
    Up = Ref(MatUp)
    M = Ref(MatM)
    Z = Ref(MatZ)
    B = Ref(MatB)
    C = Ref(MatC)
    R = Ref(MatR)

    M_vec = M_saf.vector
    M_ang = M_saf.angle
    Z_vec = Z_saf.vector
    Z_ang = Z_saf.angle

    ptrUk = pointer(MatUk)
    ptrUp = pointer(MatUp)
    ptrU = pointer(MatU)

    MatUpTmp = similar(MatUp)
    ptrUpTmp = pointer(MatUpTmp)

    # Initialize workspace


    iter::Int = 0
    abserr::Float64 = -1.0
    result_flag::Int = 0

    # U[:, 1:k] .= Uk;
    unsafe_copyto!(ptrU, ptrUk, n * k)

    if Init !== nothing
        MatUp .= Init(MatUk)
    end
    unsafe_copyto!(ptrU + n * k * sizeof(Float64), ptrUp, n * m)
    # U[:, (k + 1):n] .= Up
    # completion


    MatM .= real.(log(MatU))

    stlog_BCH3_direction_lyap!(Z, M, B, C, R)


    iter = 1
    abserr = stlog_cost(M, k)


    while result_flag == 0

        # As BCH does not make use of the angles information of the update,
        # real matrix exponential is used here, which is faster than the SAF approach.
        # Naive(slow) implementation of logarithm is used in here.

        # ret_UpZ!(U, Up, M, M_saf, Z, Z_saf, wsp_ret_UpZ; nearlog=false)
        ret_UpZ_builtin_explog!(U, Up, M, Z, wsp_ret_UpZ)

        stlog_BCH3_direction_lyap!(Z, M, B, C, R)


        iter += 1
        abserr = stlog_cost(M, k)


        result_flag = check_termination_val(abserr, nothing, nothing, nothing, nothing, iter, Stop)
        # msgln(("\t\tIteration:  ", iter - 1, "\tAbsolute error:  ", round(abserr, digits=14), "\n"), true)
    end
    return MatM, result_flag, iter
end

function stlog_hybrid_Newton_armijo_analysis(MatUk::Matrix{Float64}, MatUp::Matrix{Float64};
    Stop::terminator=terminator(300, 10000, 1e-8, 1e-6), Solver_Stop::terminator=terminator(200, 50000, 1e-12, 1e-9),
    Init=nothing, Records=nothing, NMLS_Set=nothing, wsp = get_wsp_hybrid(size(MatUk)..., Stop.MaxIter))


    n::Int, k::Int = size(MatUk)
    m::Int = n - k
    n_b::Int = div(n, 2)

    n_dim::Int = div(n * (n - 1), 2)
    m_dim::Int = div(m * (m - 1), 2)

    time_record::Vector{Float64} = Records[1][]
    cost_record::Vector{Float64} = Records[2][]
    dist_record::Vector{Float64} = Records[3][]
    vect_record::Vector{Float64} = Records[4][]
    step_record::Vector{Float64} = Records[5][]
    angs_record = Records[6][]
    stpt_record = Records[7][]



    MatU::Matrix{Float64} = wsp[1]
    MatUp_new::Matrix{Float64} = wsp[2]
    MatM::Matrix{Float64} = wsp[3]
    MatM_new::Matrix{Float64} = wsp[4]
    MatΔ::Matrix{Float64} = wsp[5]
    MatB::Matrix{Float64} = wsp[6]
    MatC::Matrix{Float64} = wsp[7]
    MatR::Matrix{Float64} = wsp[8]
    MatZ::Matrix{Float64} = wsp[9]
    MatαZ::Matrix{Float64} = wsp[10]

    # The cost needs to be stored explicitly.
    # cost_record::Vector{Float64} = wsp[11]

    Z_saf::SAFactor = wsp[12]
    M_saf::SAFactor = wsp[13]
    Δ_saf::SAFactor = wsp[14]
    M_sys::dexp_SkewSymm_system = wsp[15]

    blk_it_m::STRICT_LOWER_ITERATOR = wsp[16]
    blk_it_nm::STRICT_LOWER_ITERATOR = wsp[17]
    blk_it_n::STRICT_LOWER_ITERATOR = wsp[18]

    wsp_saf_m::WSP = wsp[19]
    wsp_saf_n::WSP = wsp[20]
    wsp_UpZ_ret::WSP = wsp[21]
    wsp_bgs::WSP = wsp[22]


    fval = Ref(cost_record)


    U = Ref(MatU)
    Up = Ref(MatUp)
    Up_new = Ref(MatUp_new)
    M = Ref(MatM)
    M_new = Ref(MatM_new)
    Δ = Ref(MatΔ)
    Z = Ref(MatZ)
    αZ = Ref(MatαZ)
    B = Ref(MatB)
    C = Ref(MatC)
    R = Ref(MatR)

    M_vec = M_saf.vector
    M_ang = M_saf.angle
    Z_vec = Z_saf.vector
    Z_ang = Z_saf.angle
    VecM_ang = getAngle(M_saf)
    VecΔ_ang = getAngle(Δ_saf)

    ptrUk = pointer(MatUk)
    ptrUp = pointer(MatUp)
    ptrU = pointer(MatU)

    ptrαZ = pointer(MatαZ)
    ptrZ = pointer(MatZ)

    direction_indicator::Int = 0
    direction_type::Vector{String} = ["Good  descent direction", "Good escaping direction", "Bad  escaping direction"]

    α::Float64 = 1.0
    sq::Float64 = 1.0
    slope::Float64 = -1.0
    α_upper_bound::Float64 = 0.0
    MatΔ_2_norm::Float64 = 0.0

    iter::Int = 0
    abserr::Float64 = -1.0
    result_flag::Int = 0


    bch_iter_cnt::Int = 0


    solver_flag::Bool = true
    nearlog_flag::Bool = false
    search_flag::Bool = false
    fail_step::Float64 = 0.5


    msgln("Newton solver:\n")

    stats = @timed begin
        # U[:, 1:k] .= Uk;
        unsafe_copyto!(ptrU, ptrUk, n * k)

        if Init !== nothing
            MatUp .= Init(MatUk)
        end
        unsafe_copyto!(ptrU + n * k * sizeof(Float64), ptrUp, n * m)
        # U[:, (k + 1):n] .= Up
    end
    time_record[1] = (stats.time - stats.gctime) * 1000
    # completion

    stats = @timed begin
        log_SpecOrth!(M, M_saf, U, wsp_saf_n; order=true, regular=true)

        iter = 1
        abserr = stlog_cost(M, k)
    end


    if stpt_record !== nothing
        stpt_record[iter] = copy(MatU)
    end

    if angs_record !== nothing
        SAFactor_order(M_saf, wsp_saf_n)
        SAFactor_regularize(M_saf, wsp_saf_n)
        angs_record[iter, :] .= M_saf.angle[]
    end

    time_record[iter] += (stats.time - stats.gctime) * 1000
    cost_record[iter] = abserr
    dist_record[iter] = sqrt(inner_skew!(M, M) - 2 * abserr)


    while result_flag == 0
        stats = @timed begin
            # if (abserr > BCH_ABSTOL && 0 <= bch_iter_cnt < BCH_MAX_ITER)        # BCH step
            if (0 <= bch_iter_cnt < BCH_MAX_ITER)        # BCH step
                
                bch_iter_cnt += 1


                    # # BCH1 step
                    # for r_ind in 1:m
                    #     for c_ind in 1:m
                    #         @inbounds MatZ[r_ind, c_ind] = -MatM[r_ind+k, c_ind+k]
                    #     end
                    # end

                    stlog_BCH3_direction_lyap!(Z, M, B, C, R)

                    ret_UpZ_builtin_exp!(U, Up, M, M_saf, Z, wsp_UpZ_ret; nearlog=false)

                    # ret_UpZ!(U, Up, M, M_saf, Z, Z_saf, wsp_UpZ_ret; nearlog=false)

                α = 1.0
                iter += 1
                abserr = stlog_cost(M, k)
                cost_record[iter] = abserr

                msgln("BCH step\tIteration: $(iter)\tAbsErr: $(abserr)")
            else                                                                # Opt step
                if bch_iter_cnt < 0
                    bch_iter_cnt += 1
                elseif bch_iter_cnt > 0
                    # The SAF done in last BCH step was not ordered nor regulaized.
                    SAFactor_order(M_saf, wsp_saf_n)
                    SAFactor_regularize(M_saf, wsp_saf_n)
                    unsafe_copyto!(ptrαZ, ptrZ, length(MatZ))
                    bch_iter_cnt = BCH_SHUTDOWN
                end
        

                if VecM_ang[1] + VecM_ang[2] > 2π
                    throw("The active point is beyond the restricted manifold, with angles $(VecM_ang[1]) and $(VecM_ang[2]).")
                end
        
                if VecM_ang[1] > π + RESTART_THRESHOLD                          # Restart by principal log
                    msgln("stlog solver restarted with principal logarithm!\n")
                    if ENABLE_RESTART_BCH
                        bch_iter_cnt = 0
                    end
                    log_SpecOrth!(M, M_saf, U, wsp_saf_n; order=true, regular=true)
                    
                    # ret_UpZ!(U, Up, M, M_saf, αZ, Z_saf, wsp_UpZ_ret; nearlog = nearlog_flag);
                    # iter = 1                             Complete restart.
                    abserr = stlog_cost(M, k)
                else                                                            # Compute Newton direction
                    compute_dexp_SkewSymm_both_system!(M_sys, M_ang)
                    slope = -2 * abserr


                    stlog_newton_descent_backward!(Z, M, M_sys, M_saf, k, m_dim, blk_it_nm, blk_it_m, blk_it_n, wsp_bgs; Stop=Solver_Stop)

                    solver_flag = all(isfinite, MatZ)
                    if !solver_flag
                        unsafe_copyto!(ptrZ, ptrαZ, length(MatZ))
                    end
                        
    
                    if ENABLE_NEARLOG && solver_flag
    
                        wsp_action = wsp_bgs[3];
    
                        MatTemp = wsp_action[1];
                        wsp_cong_n = wsp_action[3];
                        wsp_cong_nm = wsp_action[4];
    
    
                        Temp = wsp_action(1)
    
                        @turbo for d_ind in (k + 1):n
                            @inbounds MatTemp[d_ind, d_ind] = 0.0;
                        end
    
                        cong_dense!(Temp, M_saf.vector, k, Z, 0, m, wsp_cong_nm; trans = true);
                        dexp_SkewSymm!(Δ, Temp, M_sys, M_saf, blk_it_n, wsp_cong_n; inv = true, cong = false);
                        cong_dense!(Δ, M_saf.vector, Δ, wsp_cong_n; trans = false);
    
                        MatΔ_2_norm = opnorm(MatΔ)
                        α_upper_bound = (2π - VecM_ang[1] - VecM_ang[2]) / 2 * MatΔ_2_norm
    
                        # schurAngular_SkewSymm!(Δ_saf, Δ, wsp_saf_n; order=false, regular = false)
                        # sort!(VecΔ_ang; by = abs, rev = true)
                        # @inbounds MatΔ_2_norm = abs(VecΔ_ang[1])
                        # @inbounds α_upper_bound = (2π - VecM_ang[1] - VecM_ang[2]) / (abs(VecΔ_ang[1]) + abs(VecΔ_ang[2]))
    
                    else
                        α_upper_bound = 1000.0
                        MatΔ_2_norm = 1
                    end
    
                    for z_ind in eachindex(MatZ)
                        @inbounds MatZ[z_ind] = -MatZ[z_ind]
                    end
        
    
                    if !solver_flag || MatΔ_2_norm > 4π                         # Restart by escaping direction
                        # Fail to find the solution, or,
                        # restart using this direction
                        # The founded direction is too large, indicating local nonzero minima
                        # msgln("stlog solver restarted with escaping direction!\n")
                        direction_indicator = 3
                        α = 1.0
                    elseif MatΔ_2_norm > DIRECTION_THRESHOLD * α_upper_bound                    # Large direction, line search with scaled initial guess.
                        # Reasonable large sulution found. (The bound is sufficient but not necessary.)
                        # Line search will be employed but with scaled initial stepsize
                        sq = inner_skew!(Z, Z)
                        α = max(α_upper_bound / MatΔ_2_norm, 1.0)
                        # α = α_upper_bound / MatΔ_2_norm
                        direction_indicator = 2
    
                        # α = 1.0   
                    else                                                        # Good direction, line search with BB step
                        # Ideal solution, use BB stepsize
                        α, sq = hor_BB_step!(α, sq, Z, αZ)
                        direction_indicator = 1
                    end
        
                    @inbounds nearlog_flag = VecM_ang[1] > π - NEARLOG_THRESHOLD && LINESEARCH_CHECK(direction_indicator) && ENABLE_NEARLOG
        
                        if LINESEARCH_CHECK(direction_indicator)                # Line search
                            # Enable line search.
                            # fail_step = 1.0
                            fail_step = π / opnorm(MatZ)
                            # msgln("\t Line search activated with initial stepsize $(α) and slop $(slope) at objective value $(abserr).")
                            msgln("\t Line search activated with initial stepsize $(α) and slop $(slope) at objective value $(cost_record[iter]).")

                            α, abserr, search_flag = stlog_UpZ_NMLS!(fval, slope, α, Z, αZ, Z_saf, U, Up, Up_new, M, M_new, M_saf, wsp_UpZ_ret;
                            paras=NMLS_Set, f_len=iter, nearlog=nearlog_flag, bound=α_upper_bound / MatΔ_2_norm, fail_step = fail_step)

                            if !search_flag && ENABLE_RESTART_BCH
                                msgln("Restarted due to failed line search.\n")
                                bch_iter_cnt = 0    # Restart by failed line search
                            end

                        else                                                    # Restart with given stepsize
                            # Restart along direction Z
                            scale_velocity_UpZ!(αZ, Z, α)
                            ret_UpZ!(U, Up, M, M_saf, αZ, Z_saf, wsp_UpZ_ret; nearlog=false)
                            # ret_UpZ!(U, Up, M, M_saf, αZ, Z_saf, wsp_UpZ_ret; nearlog=nearlog_flag && α < α_upper_bound / MatΔ_2_norm)
                                
                            abserr = stlog_cost(M, k)
                            if ENABLE_RESTART_BCH
                                bch_iter_cnt = 0    # Restart by escaping direction
                            end
                        end
                end

                iter += 1
                cost_record[iter] = abserr

                if MSG
                    @inbounds @printf(
                        "Opt step\tIteration: %i\tAbsErr: %.12f\tDirection type: %s\t\tNearLog: %i\tSlope: %.12f\tStepsize: %.8f\n",
                        iter, abserr, direction_type[direction_indicator], nearlog_flag, slope, α
                    )
                end
            end

            result_flag = check_termination_vec(cost_record, nothing, vect_record, time_record, step_record, iter, Stop)

        end

        if stpt_record !== nothing
            stpt_record[iter] = copy(MatU)
        end

        if angs_record !== nothing
            SAFactor_order(M_saf, wsp_saf_n)
            SAFactor_regularize(M_saf, wsp_saf_n)
            angs_record[iter, :] .= M_saf.angle[]
        end

        time_record[iter] = time_record[iter-1] + (stats.time - stats.gctime) * 1000
        cost_record[iter] = abserr
        dist_record[iter] = sqrt(inner_skew!(M, M) - abserr)
        vect_record[iter-1] = sqrt(sq)
        step_record[iter-1] = α
    end

    return MatM, result_flag, iter, time_record[iter], cost_record[iter], dist_record[iter], vect_record[iter-1]
end

function stlog_hybrid_Newton_armijo(MatUk::Matrix{Float64}, MatUp::Matrix{Float64};
    Stop::terminator=terminator(300, 10000, 1e-8, 1e-6), Solver_Stop::terminator=terminator(200, 50000, 1e-12, 1e-9),
    Init=nothing, NMLS_Set=nothing, wsp = get_wsp_hybrid(size(MatUk)..., Stop.MaxIter))

    n::Int, k::Int = size(MatUk)
    m::Int = n - k
    n_b::Int = div(n, 2)

    n_dim::Int = div(n * (n - 1), 2)
    m_dim::Int = div(m * (m - 1), 2)

    
    MatU::Matrix{Float64} = wsp[1]
    MatUp_new::Matrix{Float64} = wsp[2]
    MatM::Matrix{Float64} = wsp[3]
    MatM_new::Matrix{Float64} = wsp[4]
    MatΔ::Matrix{Float64} = wsp[5]
    MatB::Matrix{Float64} = wsp[6]
    MatC::Matrix{Float64} = wsp[7]
    MatR::Matrix{Float64} = wsp[8]
    MatZ::Matrix{Float64} = wsp[9]
    MatαZ::Matrix{Float64} = wsp[10]

    # The cost is needed for NMLS
    cost_record::Vector{Float64} = wsp[11]

    Z_saf::SAFactor = wsp[12]
    M_saf::SAFactor = wsp[13]
    Δ_saf::SAFactor = wsp[14]
    M_sys::dexp_SkewSymm_system = wsp[15]

    blk_it_m::STRICT_LOWER_ITERATOR = wsp[16]
    blk_it_nm::STRICT_LOWER_ITERATOR = wsp[17]
    blk_it_n::STRICT_LOWER_ITERATOR = wsp[18]

    wsp_saf_m::WSP = wsp[19]
    wsp_saf_n::WSP = wsp[20]
    wsp_UpZ_ret::WSP = wsp[21]
    wsp_bgs::WSP = wsp[22]




    U = Ref(MatU)
    Up = Ref(MatUp)
    Up_new = Ref(MatUp_new)
    M = Ref(MatM)
    M_new = Ref(MatM_new)
    Δ = Ref(MatΔ)
    Z = Ref(MatZ)
    αZ = Ref(MatαZ)
    B = Ref(MatB)
    C = Ref(MatC)
    R = Ref(MatR)

    fval = Ref(cost_record)


    M_vec = M_saf.vector
    M_ang = M_saf.angle
    Z_vec = Z_saf.vector
    Z_ang = Z_saf.angle
    VecM_ang = getAngle(M_saf)
    VecΔ_ang = getAngle(Δ_saf)

    ptrUk = pointer(MatUk)
    ptrUp = pointer(MatUp)
    ptrU = pointer(MatU)

    ptrαZ = pointer(MatαZ)
    ptrZ = pointer(MatZ)


    direction_indicator::Int = 0

    α::Float64 = 1.0
    sq::Float64 = 1.0
    slope::Float64 = -1.0
    α_upper_bound::Float64 = 0.0
    MatΔ_2_norm::Float64 = 0.0

    iter::Int = 0
    abserr::Float64 = -1.0
    result_flag::Int = 0

    bch_iter_cnt::Int = 0

    solver_flag::Bool = true
    nearlog_flag::Bool = false
    search_flag::Bool = false
    fail_step::Float64 = 0.5


    # U[:, 1:k] .= Uk;
    unsafe_copyto!(ptrU, ptrUk, n * k)

    if Init !== nothing
        MatUp .= Init(MatUk)
    end
    unsafe_copyto!(ptrU + n * k * sizeof(Float64), ptrUp, n * m)
    # U[:, (k + 1):n] .= Up
    # completion

    log_SpecOrth!(M, M_saf, U, wsp_saf_n; order=true, regular=true)

    iter = 1
    abserr = stlog_cost(M, k)
    cost_record[iter] = abserr

    while result_flag == 0
        if (0 <= bch_iter_cnt < BCH_MAX_ITER)        # BCH step
        # if (abserr > BCH_ABSTOL && 0 <= bch_iter_cnt < BCH_MAX_ITER)        # BCH step
            bch_iter_cnt += 1

            # BCH1 step
            # for r_ind in 1:m
            #     for c_ind in 1:m
            #         @inbounds MatZ[r_ind, c_ind] = -MatM[r_ind+k, c_ind+k]
            #     end
            # end

            stlog_BCH3_direction_lyap!(Z, M, B, C, R)

            # ret_UpZ!(U, Up, M, M_saf, Z, Z_saf, wsp_UpZ_ret; nearlog=false)
            ret_UpZ_builtin_exp!(U, Up, M, M_saf, Z, wsp_UpZ_ret; nearlog=false)

            
            α = 1.0

            abserr = stlog_cost(M, k)
            # sq = inner_skew!(Z, Z)
        else                                                                # Opt step
            if bch_iter_cnt < 0
                bch_iter_cnt += 1
            elseif bch_iter_cnt > 0
                # The SAF done in last BCH step was not ordered nor regulaized.
                SAFactor_order(M_saf, wsp_saf_n)
                SAFactor_regularize(M_saf, wsp_saf_n)

                unsafe_copyto!(ptrαZ, ptrZ, length(MatZ))
                bch_iter_cnt = BCH_SHUTDOWN
            end
            

            if VecM_ang[1] + VecM_ang[2] > 2π
                throw("The active point is beyond the restricted manifold, with angles $(VecM_ang[1]) and $(VecM_ang[2]).")
            end

            if VecM_ang[1] > π + RESTART_THRESHOLD                          # Restart by principal log
                # msgln("stlog solver restarted with principal logarithm!\n")
                if ENABLE_RESTART_BCH
                    bch_iter_cnt = 0
                end
                log_SpecOrth!(M, M_saf, U, wsp_saf_n; order=true, regular=true)
                # ret_UpZ!(U, Up, M, M_saf, αZ, Z_saf, wsp_UpZ_ret; nearlog = nearlog_flag);
                # iter = 1                             Complete restart.
                abserr = stlog_cost(M, k)
            else                                                            # Compute Newton direction
                compute_dexp_SkewSymm_both_system!(M_sys, M_ang)
                slope = -2 * abserr

                # act = _STLOG_BACKWARD_Z_ACTION(M_sys, M_saf, blk_it_nm, blk_it_n, wsp_action)
                # stlog_newton_descent_gmres!(Δ3, Z3, M, act, rs, blk_it_m, wsp_bgs; Stop=Stop)

                stlog_newton_descent_backward!(Z, M, M_sys, M_saf, k, m_dim, blk_it_nm, blk_it_m, blk_it_n, wsp_bgs; Stop=Solver_Stop)

                solver_flag = all(isfinite, MatZ)
                if !solver_flag
                    unsafe_copyto!(ptrZ, ptrαZ, length(MatZ))
                end
                    

                if ENABLE_NEARLOG && solver_flag

                    wsp_action = wsp_bgs[3];

                    MatTemp = wsp_action[1];
                    wsp_cong_n = wsp_action[3];
                    wsp_cong_nm = wsp_action[4];


                    Temp = wsp_action(1)

                    @turbo for d_ind in (k + 1):n
                        @inbounds MatTemp[d_ind, d_ind] = 0.0;
                    end

                    cong_dense!(Temp, M_saf.vector, k, Z, 0, m, wsp_cong_nm; trans = true);
                    dexp_SkewSymm!(Δ, Temp, M_sys, M_saf, blk_it_n, wsp_cong_n; inv = true, cong = false);
                    cong_dense!(Δ, M_saf.vector, Δ, wsp_cong_n; trans = false);

                    MatΔ_2_norm = opnorm(MatΔ)
                    α_upper_bound = (2π - VecM_ang[1] - VecM_ang[2]) / 2 * MatΔ_2_norm

                    # schurAngular_SkewSymm!(Δ_saf, Δ, wsp_saf_n; order=false, regular = false)
                    # sort!(VecΔ_ang; by = abs, rev = true)
                    # @inbounds MatΔ_2_norm = abs(VecΔ_ang[1])
                    # @inbounds α_upper_bound = (2π - VecM_ang[1] - VecM_ang[2]) / (abs(VecΔ_ang[1]) + abs(VecΔ_ang[2]))

                else
                    α_upper_bound = 1000.0
                    MatΔ_2_norm = 1
                end

                for z_ind in eachindex(MatZ)
                    @inbounds MatZ[z_ind] = -MatZ[z_ind]
                end
    

                if !solver_flag || MatΔ_2_norm > 4π                         # Restart by escaping direction
                    # Fail to find the solution, or,
                    # restart using this direction
                    # The founded direction is too large, indicating local nonzero minima
                    # msgln("stlog solver restarted with escaping direction!\n")
                    direction_indicator = 3
                    α = 1.0
                elseif MatΔ_2_norm > DIRECTION_THRESHOLD * α_upper_bound                    # Large direction, line search with scaled initial guess.
                    # Reasonable large sulution found. (The bound is sufficient but not necessary.)
                    # Line search will be employed but with scaled initial stepsize
                    sq = inner_skew!(Z, Z)
                    α = max(α_upper_bound / MatΔ_2_norm, 1.0)
                    # α = α_upper_bound / MatΔ_2_norm
                    direction_indicator = 2

                    # α = 1.0   
                else                                                        # Good direction, line search with BB step
                    # Ideal solution, use BB stepsize
                    α, sq = hor_BB_step!(α, sq, Z, αZ)
                    direction_indicator = 1
                end

                @inbounds nearlog_flag = VecM_ang[1] > π - NEARLOG_THRESHOLD && LINESEARCH_CHECK(direction_indicator) && ENABLE_NEARLOG


                if LINESEARCH_CHECK(direction_indicator)                    # Line search
                    # Enable line search.
                    # fail_step = 1.0
                    # fail_step = FAIL_STEP
                    fail_step = π / opnorm(MatZ)


                    # fail_step = max(α_upper_bound / MatΔ_2_norm, 1.0)


                    α, abserr, search_flag = stlog_UpZ_NMLS!(fval, slope, α, Z, αZ, Z_saf, U, Up, Up_new, M, M_new, M_saf, wsp_UpZ_ret;
                        paras=NMLS_Set, f_len=iter, nearlog=nearlog_flag, bound=α_upper_bound / MatΔ_2_norm, fail_step = fail_step)
                    
                    if !search_flag && ENABLE_RESTART_BCH
                        bch_iter_cnt = 0    # Restart by failed line search
                    end
                else                                                        # Restart with given stepsize
                    # Restart along direction Z
                    scale_velocity_UpZ!(αZ, Z, α)
                    ret_UpZ!(U, Up, M, M_saf, αZ, Z_saf, wsp_UpZ_ret; nearlog=false)
                    # ret_UpZ!(U, Up, M, M_saf, αZ, Z_saf, wsp_UpZ_ret; nearlog=nearlog_flag && α < α_upper_bound / MatΔ_2_norm)
                    
                    abserr = stlog_cost(M, k)
                    if ENABLE_RESTART_BCH
                        bch_iter_cnt = 0    # Enable BCH step if necessary
                    end
                end
            end
        end

        iter += 1
        cost_record[iter] = abserr
        result_flag = check_termination_val(abserr, nothing, sqrt(sq), nothing, nothing, iter, Stop)
    end
    return MatM, result_flag, iter
end

function stlog_Newton_armijo_analysis(MatUk::Matrix{Float64}, MatUp::Matrix{Float64};
    Stop::terminator=terminator(300, 10000, 1e-8, 1e-6), Solver_Stop::terminator=terminator(200, 50000, 1e-12, 1e-9),
    Init=nothing, Records=nothing, NMLS_Set=nothing, wsp = get_wsp_hybrid(size(MatUk)..., Stop.MaxIter))

    temp = BCH_MAX_ITER;
    global BCH_MAX_ITER = 1;
    Ans = stlog_hybrid_Newton_armijo_analysis(MatUk, MatUp;
        Stop=Stop, Solver_Stop=Solver_Stop,
        Init=Init, Records=Records, NMLS_Set=NMLS_Set, wsp = wsp)
    
    global BCH_MAX_ITER = temp;
    return Ans
end

function stlog_Newton_armijo(MatUk::Matrix{Float64}, MatUp::Matrix{Float64};
    Stop::terminator=terminator(300, 10000, 1e-8, 1e-6), Solver_Stop::terminator=terminator(200, 50000, 1e-12, 1e-9),
    Init=nothing, NMLS_Set=nothing, wsp = get_wsp_hybrid(size(MatUk)..., Stop.MaxIter))

    temp = BCH_MAX_ITER;
    global BCH_MAX_ITER = 1;

    Ans= stlog_hybrid_Newton_armijo(MatUk, MatUp;
        Stop=Stop, Solver_Stop=Solver_Stop,
        Init=Init, NMLS_Set=NMLS_Set, wsp = wsp)
    global BCH_MAX_ITER = temp;

    return Ans
end


########################core algorithm########################


########################algorithm port########################
# function StlogIk_analysis_port(algo, Unk, Up, Stop, Solver_Stop, Init, Geo, NMLS_Set, Require_StPt_Record)
#     Stop_terminator = terminator(Stop[1], Stop[2], Stop[3], nothing; MSS=Stop[4], MAU=Stop[5], MRU=Stop[6], MSBI=Stop[7])
#     Solver_Stop_terminator = terminator(Solver_Stop[1], Solver_Stop[2], Solver_Stop[3], Solver_Stop[4])

#     MaxIter = Stop[1]

#     TimeRec = zeros(MaxIter)
#     AbsERec = zeros(MaxIter)
#     DistRec = zeros(MaxIter)
#     VectRec = zeros(MaxIter)
#     StepRec = ones(MaxIter)
#     StPtRec = Require_StPt_Record ? [Matrix{Float64}(undef, 1, 1) for ii = 1:MaxIter] : nothing

#     Records = [Ref(TimeRec), Ref(AbsERec), Ref(DistRec), Ref(VectRec), Ref(StepRec), Ref(StPtRec)]

#     Ans, flag, iter = algo(Unk, Up;
#         Stop=Stop_terminator, Solver_Stop=Solver_Stop_terminator,
#         Init=Init, Geo=Geo, Records=Records, NMLS_Set=NMLS_Set)

#     if Require_StPt_Record
#         return Ans, flag, TimeRec[1:(iter+1)], AbsERec[1:(iter+1)], DistRec[1:iter], VectRec[1:iter], StepRec[1:iter], StPtRec[1:iter]
#     else
#         return Ans, flag, TimeRec[1:(iter+1)], AbsERec[1:(iter+1)], DistRec[1:iter], VectRec[1:iter], StepRec[1:iter]
#     end
# end

# function StlogIk_2k_analysis_port(algo, Unk, Up, Stop, Solver_Stop, Init, Geo, NMLS_Set, Require_StPt_Record)
#     Stop_terminator = terminator(Stop[1], Stop[2], Stop[3], nothing; MSS=Stop[4], MAU=Stop[5], MRU=Stop[6], MSBI=Stop[7])
#     Solver_Stop_terminator = terminator(Solver_Stop[1], Solver_Stop[2], Solver_Stop[3], Solver_Stop[4])

#     MaxIter = Stop[1]

#     TimeRec = zeros(MaxIter)
#     AbsERec = zeros(MaxIter)
#     DistRec = zeros(MaxIter)
#     VectRec = zeros(MaxIter)
#     StepRec = ones(MaxIter)
#     StPtRec = Require_StPt_Record ? [Matrix{Float64}(undef, 1, 1) for ii = 1:MaxIter] : nothing

#     Records = [Ref(TimeRec), Ref(AbsERec), Ref(DistRec), Ref(VectRec), Ref(StepRec), Ref(StPtRec)]

#     preprocessing_time = @timed begin
#         Vdk, Qrm = preprocessing_Ink_2k(Ref(Unk))
#     end

#     d::Int, k::Int = size(Vdk)
#     if norm(Up, Inf) < 1e-10
#         Vp = zeros(d, d - k)
#     else
#         Vp = zeros(d, d - k)
#     end

#     display(Geo)
#     Ans_V, flag, iter = algo(Vdk, Vp;
#         Stop=Stop_terminator, Solver_Stop=Solver_Stop_terminator,
#         Init=Init, Geo=Geo, Records=Records, NMLS_Set=NMLS_Set)

#     postprocessing_time = @timed begin
#         Ans = post_processing_Ink(Ref(Unk), Ref(Ans_V), Qrm)
#     end
#     if Require_StPt_Record
#         # Mdk::Matrix{Float64} = Matrix{Float64}(undef, d, k);
#         # for ii = 1:iter
#         #     Mdk .= StPtRec[ii];
#         #     StPtRec[ii] = post_processing_Ink(Ref(Unk), Ref(Mdk), Qrm);
#         # end
#         return Ans, flag, TimeRec[1:(iter+1)], AbsERec[1:(iter+1)], DistRec[1:iter], VectRec[1:iter], StepRec[1:iter], StPtRec[1:iter]
#     else
#         return Ans, flag, TimeRec[1:(iter+1)], AbsERec[1:(iter+1)], DistRec[1:iter], VectRec[1:iter], StepRec[1:iter]
#     end
# end

# function StlogIk_rank_analysis_port(algo, Unk, Up, Stop, Solver_Stop, Init, Geo, NMLS_Set, Require_StPt_Record)
#     Stop_terminator = terminator(Stop[1], Stop[2], Stop[3], nothing; MSS=Stop[4], MAU=Stop[5], MRU=Stop[6], MSBI=Stop[7])
#     Solver_Stop_terminator = terminator(Solver_Stop[1], Solver_Stop[2], Solver_Stop[3], Solver_Stop[4])

#     MaxIter = Stop[1]

#     TimeRec = zeros(MaxIter)
#     AbsERec = zeros(MaxIter)
#     DistRec = zeros(MaxIter)
#     VectRec = zeros(MaxIter)
#     StepRec = ones(MaxIter)
#     StPtRec = Require_StPt_Record ? [Matrix{Float64}(undef, 1, 1) for ii = 1:MaxIter] : nothing

#     Records = [Ref(TimeRec), Ref(AbsERec), Ref(DistRec), Ref(VectRec), Ref(StepRec), Ref(StPtRec)]

#     preprocessing_time = @timed begin
#         Vdk, Qrm = preprocessing_Ink_2k(Ref(Unk))
#     end

#     d::Int, k::Int = size(Vdk)
#     if norm(Up, Inf) < 1e-10
#         Vp = zeros(d, d - k)
#     else
#         Vp = zeros(d, d - k)
#     end

#     display(Geo)
#     Ans_V, flag, iter = algo(Vdk, Vp;
#         Stop=Stop_terminator, Solver_Stop=Solver_Stop_terminator,
#         Init=Init, Geo=Geo, Records=Records, NMLS_Set=NMLS_Set)

#     postprocessing_time = @timed begin
#         Ans = post_processing_Ink(Ref(Unk), Ref(Ans_V), Qrm)
#     end
#     if Require_StPt_Record
#         # Mdk::Matrix{Float64} = Matrix{Float64}(undef, d, k);
#         # for ii = 1:iter
#         #     Mdk .= StPtRec[ii];
#         #     StPtRec[ii] = post_processing_Ink(Ref(Unk), Ref(Mdk), Qrm);
#         # end
#         return Ans, flag, TimeRec[1:(iter+1)], AbsERec[1:(iter+1)], DistRec[1:iter], VectRec[1:iter], StepRec[1:iter], StPtRec[1:iter]
#     else
#         return Ans, flag, TimeRec[1:(iter+1)], AbsERec[1:(iter+1)], DistRec[1:iter], VectRec[1:iter], StepRec[1:iter]
#     end
# end
########################algorithm port########################

# test_Newton_speed(10, range(0.2π, 1.2π, 50), 10; MaxIter=500, AbsTol=1e-12, MaxTime=10000, seed=rand(1:10000), scaleby=opnorm)
# test_BCH_naive_speed(10,range(0.2π,1.2π,50), 20; MaxIter=500, AbsTol=1e-14, MaxTime=1000, seed=rand(1:10000), scaleby=opnorm)