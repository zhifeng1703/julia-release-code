using LinearAlgebra, MKL, Printf, DelimitedFiles
using Statistics, Random

include("stlog_analysis.jl")

# Base.libblas_name

BLAS_LOADED_LIB = BLAS.get_config().loaded_libs[1].libname;



UpZ_Geo = (scale_velocity_UpZ!, update_point_UpZ!, ret_UpZ!, nip_ret_UpZ!);

function BCH_42_analysis(Uk::Matrix{Float64}, U_perp::Matrix{Float64};
    Stop=terminator(300, 10000, 1e-10, 1e-6), Solver_Stop=nothing,
    Init=nothing, Geo=nothing, Records=nothing, NMLS_Set=nothing)
    n::Int, k::Int = size(Uk)
    m::Int = n - k
    n_b::Int = div(n, 2)

    # This must be used after preprocessing_2k that turn (n, k) problem into (2k, k) problem

    time_record::Vector{Float64} = Records[1][]
    cost_record::Vector{Float64} = Records[2][]
    dist_record::Vector{Float64} = Records[3][]
    vect_record::Vector{Float64} = Records[4][]
    step_record::Vector{Float64} = Records[5][]
    stpt_record = Records[6][]

    result_flag::Int = 0



    Scale = Geo[1]
    Update = Geo[2]
    Ret = Geo[3]
    NIP_Ret = Geo[4]


    # Define workspace:
    # Up:       1   n x (n-k)       orthogonal completion, the variable
    # U:        2   n x n           the full orthogonal matrix [Uk Up]
    # S:        3   n x n           skew-symmetric log of [Uk Up]
    # B:        4   (n-k) x k       the 1,2-partition from S
    # C:        5   (n-k) x (n-k)   the 2,2-partition from S
    # R:        6   (n-k) x (n-k)   matrix argument used in lyapunov equation 
    # Z:        7   (n-k) x (n-k)   the 2,2-partition form S_{X,Y,Z} on the RHS
    # QZ:       8   (n-k) x (n-k)   the orthogonal QZ = exp(Q)
    # P:        9   n x n           orthogonal schur vectors of S
    # Θ:        10  div(n, 2)       canoncial angles of S
    # PZ:       11  (n-k) x (n-k)   orthogonal schur vectors of Z
    # ΘZ:       12  div(n-k, 2)     canoncial angles of Z
    # TMn:      13  n x n           Temporal real matrix workspace of n x n
    # TMn_k:    14  (n-k) x (n-k)   Temporal real matrix workspace of n-k x n-k
    # TBVn:     15  n               Temporal boolean vector workspace of n
    # TBVn_k:   16  (n-k)           Temporal boolean vector workspace of n-k
    Up::Matrix{Float64} = Matrix{Float64}(undef, n, m)
    U::Matrix{Float64} = Matrix{Float64}(undef, n, n)
    S::Matrix{Float64} = Matrix{Float64}(undef, n, n)
    B::Matrix{Float64} = Matrix{Float64}(undef, m, k)
    C::Matrix{Float64} = Matrix{Float64}(undef, m, m)
    R::Matrix{Float64} = Matrix{Float64}(undef, m, m)
    Z::Matrix{Float64} = Matrix{Float64}(undef, m, m)
    QZ::Matrix{Float64} = Matrix{Float64}(undef, m, m)
    P::Matrix{Float64} = Matrix{Float64}(undef, n, n)
    Θ::Vector{Float64} = Vector{Float64}(undef, n_b)
    PZ::Matrix{Float64} = Matrix{Float64}(undef, m, m)
    ΘZ::Vector{Float64} = Vector{Float64}(undef, div(m, 2))

    Upt::Matrix{Float64} = Matrix{Float64}(undef, n, m)


    U_r = Ref(U)
    Up_r = Ref(Up)
    S_r = Ref(S)
    P_r = Ref(P)
    Θ_r = Ref(Θ)
    Z_r = Ref(Z)

    # Initialize workspace


    wsp_schur_n::WSP = get_wsp_schur(n)
    wsp_schur_m::WSP = get_wsp_schur(m)
    wsp_log_n::WSP = get_wsp_log(n, wsp_schur_n)
    wsp_exp_m::WSP = get_wsp_exp(m, wsp_schur_m)
    wsp_ret::WSP = get_wsp_ret(n, k, wsp_exp_m, wsp_log_n)

    # The TMm is also used in wsp_exp_m, which is not causing conflict in exp_skew!(TMm_r, ..., wsp_exp_m)



    stats = @timed begin
        # U[:, 1:k] .= Uk;
        for ii = 1:n
            for jj = 1:k
                U[ii, jj] = Uk[ii, jj]
            end
        end
        if norm(U_perp, Inf) == 0.0
            Up .= Init(Uk)
            # U[:, (k + 1):n] .= Up;
            for ii = 1:n
                for jj = 1:(n-k)
                    U[ii, k+jj] = Up[ii, jj]
                end
            end
        else
            # Up .= U_perp;
            # U[:, (k + 1):n] .= Up;
            for ii = 1:n
                for jj = 1:k
                    Up[ii, jj] = U_perp[ii, jj]
                    U[ii, k+jj] = Up[ii, jj]
                end
            end
        end
        # U[:, (k + 1):n] .= Up
    end
    time_record[1] = (stats.time - stats.gctime) * 1000
    # completion

    stats = @timed begin

        log_skew!(S_r, P_r, Θ_r, U_r, wsp_log_n)

        R .= 0.0
        for ii = 1:(n-k)
            R[ii, ii] = -0.5
        end
        for ii = 1:(n-k)
            for jj = 1:k
                B[ii, jj] = S[ii+k, jj]
            end
        end
        mul!(R, B, B', 1.0 / 12, 1.0)

        for ii = 1:(n-k)
            for jj = 1:(n-k)
                C[ii, jj] = S[ii+k, jj+k]
            end
        end

        Z .= lyap(R, C)
        Z .*= -1


        iter::Int = 1
        abserr::Float64 = inner_skew_22!(S_r, S_r, k) / 2.0
    end


    if !isnothing(stpt_record)
        stpt_record[iter] = Matrix{Float64}(undef, n, n)
        stpt_record[iter] .= U
    end
    time_record[iter] += (stats.time - stats.gctime) * 1000
    # cost_record[iter] = norm(C) ^ 2 / 2.0;
    cost_record[iter] = abserr
    dist_record[iter] = sqrt(inner_skew!(S_r, S_r) / 2.0 - abserr)


    while result_flag == 0
        stats = @timed begin
            # Ret(U_r, Up_r, S_r, P_r, Θ_r, Z_r, wsp_ret)
            QZ[1, 1] = cos(Z[2, 1])
            QZ[2, 2] = cos(Z[2, 1])
            QZ[1, 2] = -sin(Z[2, 1])
            QZ[2, 1] = sin(Z[2, 1])

            for ind in eachindex(Up)
                Upt[ind] = Up[ind]
            end

            mul!(Up, Upt, QZ)

            for ii = 1:n
                for jj = (k+1):n
                    U[ii, jj] = Up[ii, jj-k]
                end
            end


            log_skew!(S_r, P_r, Θ_r, U_r, wsp_log_n)

            R .= 0.0
            for ii = 1:(n-k)
                R[ii, ii] = -0.5
            end
            for ii = 1:(n-k)
                for jj = 1:k
                    B[ii, jj] = S[ii+k, jj]
                end
            end
            for ii = 1:(n-k)
                for jj = 1:(n-k)
                    C[ii, jj] = S[ii+k, jj+k]
                end
            end

            mul!(R, B, B', 1.0 / 12, 1)

            Z .= lyap(R, C)
            Z .*= -1

            iter += 1
            abserr = inner_skew_22!(S_r, S_r, k) / 2.0
        end

        if !isnothing(stpt_record)
            stpt_record[iter] = Matrix{Float64}(undef, n, n)
            stpt_record[iter] .= U
        end
        time_record[iter] = time_record[iter-1] + (stats.time - stats.gctime) * 1000
        # time_record[iter] = time_record[iter - 1] + (stats.time - 0 * stats.gctime) * 1000;
        cost_record[iter] = abserr
        dist_record[iter] = sqrt(inner_skew!(S_r, S_r) / 2.0 - abserr)
        vect_record[iter-1] = sqrt(0.5 * inner_skew!(Z_r, Z_r))

        result_flag = check_termination(cost_record, nothing, vect_record, time_record, step_record, iter, Stop)
    end


    return S[:, 1:k], result_flag, iter

end


function spherical_angular_grids(ms)
    ans = [range(0, step=π / mesh_size, length=mesh_size) for mesh_size in ms]
    ans[end] *= 2
    return ans
end
# π, 2π not included in the endpoints 

function spherical_to_euclidean!(ec_r, sc)
    ec = ec_r[]

    temp::Float64 = 0.0

    dim = length(sc)
    for ii = 1:dim
        ec[ii] = sc[1]
    end

    for ii = 2:dim
        temp = sin(sc[ii])
        for jj = ii:dim
            ec[jj] *= temp
        end
    end

    for ii = 1:(dim-1)
        ec[ii] *= cos(sc[ii+1])
    end
    return ec_r
end

function spherical_to_euclidean(sc)
    ec = zeros(sc)
    spherical_to_euclidean!(Ref(ec), sc)
    return ec
end

function euclidean_to_skew_sym!(S_r, x)
    S = S_r[]
    S .= 0.0
    n::Int = size(S)[1]
    d::Int = length(x)
    ind::Int = 1
    dim::Int = length(x)
    for col = 1:(n-1)
        for row = (col+1):n
            if ind > dim
                break
            end
            S[row, col] = x[ind]
            S[col, row] = -x[ind]
            ind += 1
        end
    end
    return S_r
end

function euclidean_to_skew_sym(x; n::Int=Int(ceil(sqrt(2 * length(x)))))
    S::Matrix{Float64} = Matrix{Float64}(undef, n, n)
    euclidean_to_skew_sym!(Ref(S), x)
    return S
end


function get_mesh_point!(mesh_r, grids, cartesian_index)
    mesh = mesh_r[]
    for ind in eachindex(mesh)
        mesh[ind] = grids[ind][cartesian_index[ind]]
    end
    return mesh_r
end

function get_mesh_point(grids, cartesian_index)
    mesh = zeros(length(grids))
    get_mesh_point!(Ref(mesh), grids, cartesian_index)
    return mesh
end

function cost_in_test(S_r, n, k)
    S = S_r[]
    ans::Float64 = 0.0
    for ii = (k+1):n
        for jj = (ii+1):n
            ans += S[ii, jj]^2
        end
    end
    return ans
end

avg_good(V, Fail) = sum(.~Fail) == 0 ? 0.0 : sum(V .* .~Fail) / sum(.~Fail);
avg_bad(V, Fail) = sum(Fail) == 0 ? 0.0 : sum(V .* Fail) / sum(Fail);


function test_root_and_connect(n, k; Eval_Cnt=1000, Radius_Mesh=1.0:0.1:2.0, Search_Mesh=100,
    Threshold_Dc=1e-2, Threshold_Rt=1e-2, Threshold_Rt_Dis=1e-1, FilePath="output.txt",
    MaxIter=200, MaxTime=1000, AbsTol=1e-8, RelTol=1e-5)

    m::Int = n - k

    dim::Int = div(m * (m - 1), 2)

    Record_alg::Vector{Any} = Vector{Any}(undef, 6)
    for ii = 1:5
        Record_alg[ii] = Vector{Float64}(undef, MaxIter)
    end
    Record_alg[6] = nothing
    Record_alg_r = [Ref(ele) for ele in Record_alg]

    Record_fail = zeros(Bool, length(Radius_Mesh), Eval_Cnt)
    Record_time = zeros(length(Radius_Mesh), Eval_Cnt)
    Record_gcti = zeros(length(Radius_Mesh), Eval_Cnt)
    Record_ovhe = zeros(length(Radius_Mesh), Eval_Cnt)
    Record_iter = Matrix{Int}(undef, length(Radius_Mesh), Eval_Cnt)

    wsp_schur_n::WSP = get_wsp_dgees(n)
    wsp_schur_m::WSP = get_wsp_dgees(m)
    wsp_log_n::WSP = get_wsp_log(n, wsp_schur_n)
    wsp_exp_m::WSP = get_wsp_exp(m, wsp_schur_m)
    wsp_ret::WSP = get_wsp_ret(n, k, wsp_exp_m, wsp_log_n)

    Failing_Run = Vector{Any}()


    S::Matrix{Float64} = zeros(n, n)
    Q::Matrix{Float64} = zeros(n, n)
    Z::Matrix{Float64} = zeros(n - k, n - k)


    # Ans = eval_on_mesh(x -> StLog_cost_by_Z(Q, rescaled_ssym_by_2norm(x), n, k), meshes);


    msize::Vector{Int} = Vector{Int}(undef, dim)
    msize .= Search_Mesh

    # mend::Vector{Float64} = Vector{Float64}(undef, dim);
    # mend .= π;
    # mend[dim] = 2π;

    # meshes = [range(0, step = mend[ii] / msize[ii], length = msize[ii]) for ii = 1:dim]

    meshes = spherical_angular_grids(msize) # radius mesh is also 0 -> ... -> π.
    if n - k == 2
        meshes[1] = range(0, π - 5e-2, length(meshes[1]))
    end

    cost_in_search::Float64 = 10.0
    scale::Float64 = 0.0
    is_dc::Int = 0
    multi_root::Int = 0
    Printed_Percentage::Int = 0
    found_root::Matrix{Float64} = zeros(k, k)



    sphere_coordinate_Z::Vector{Float64} = zeros(dim)
    euclid_coordinate_Z::Vector{Float64} = zeros(dim)

    S_in_Search = Matrix{Float64}(undef, n, n)
    Z_in_Search = Matrix{Float64}(undef, k, k)
    U_in_Search = Matrix{Float64}(undef, n, n)
    P_in_Search = Matrix{Float64}(undef, n, n)
    Uk_in_Search = Matrix{Float64}(undef, n, k)
    Up_in_Search = Matrix{Float64}(undef, n, m)
    UpZ_in_Search = Matrix{Float64}(undef, n, m)

    rSiS = Ref(S_in_Search)
    rZiS = Ref(Z_in_Search)
    rUiS = Ref(U_in_Search)
    rPiS = Ref(P_in_Search)
    rUkiS = Ref(Uk_in_Search)
    rUpiS = Ref(Up_in_Search)
    rUpZiS = Ref(Up_in_Search)

    Θ_in_Search::Vector{Float64} = Vector{Float64}(undef, div(n, 2))

    rΘiS = Ref(Θ_in_Search)

    @printf "Distance\tAvg.Iters\tAvg.Overhead\tAvg.Time\tAvg.GCtime\tAvg.FailRate\tAccu.FailTime\tAccu.GoodTime\n"

    File = open(FilePath, "w")



    for s_ind in eachindex(Radius_Mesh)
        s = Radius_Mesh[s_ind]
        for e_ind = 1:Eval_Cnt

            Record_alg[1] .= 0.0
            Record_alg[2] .= 0.0
            Record_alg[3] .= 0.0
            Record_alg[4] .= 0.0
            Record_alg[5] .= 0.0

            S = rand(n, n)
            S[(k+1):n, (k+1):n] .= 0.0
            S .-= S'
            S .*= sqrt(2) * s / norm(S)
            Q = exp(S)



            S_ans, flag, iter, GCt = BCH_2k_analysis(Q[:, 1:k], zeros(n, n - k);
                Init=init_guess_simple, Geo=UpZ_Geo, Records=Record_alg_r, Stop=terminator(MaxIter, MaxTime, AbsTol, RelTol))

            Record_fail[s_ind, e_ind] = flag > 2
            Record_time[s_ind, e_ind] = Record_alg[1][iter]
            Record_gcti[s_ind, e_ind] = GCt
            Record_ovhe[s_ind, e_ind] = Record_alg[1][1]
            Record_iter[s_ind, e_ind] = iter

            if flag > 2
                # Search the feasible set.
                # println("BCH solver failed. Brute-force searching feasible set 𝒮𝒪_" * string(k));

                cost_in_search = 10.0
                scale = 0.0
                is_dc = 0
                multi_root = 0
                Printed_Percentage = 0

                found_root .= 0.0

                for ind in eachindex(Uk_in_Search)
                    Uk_in_Search[ind] = Q[ind]
                    U_in_Search[ind] = Q[ind]
                end

                Up_in_Search .= init_guess_simple(Uk_in_Search)



                # initialized problem

                for CarInd in CartesianIndices(tuple(msize...))

                    for ind in eachindex(Up_in_Search)
                        U_in_Search[ind+n*k] = Up_in_Search[ind]
                        UpZ_in_Search[ind] = Up_in_Search[ind]
                    end

                    get_mesh_point!(Ref(sphere_coordinate_Z), meshes, CarInd)
                    spherical_to_euclidean!(Ref(euclid_coordinate_Z), sphere_coordinate_Z)
                    euclidean_to_skew_sym!(rZiS, euclid_coordinate_Z)

                    if sphere_coordinate_Z[1] > 1e-9
                        scale = sphere_coordinate_Z[1] / opnorm(Z_in_Search, 2)
                        for ind in eachindex(Z_in_Search)
                            Z_in_Search[ind] *= scale
                        end
                    end

                    ret_UpZ!(rUiS, rUpZiS, rSiS, rPiS, rΘiS, rZiS, wsp_ret)

                    cost_in_search = inner_skew_22!(rSiS, rSiS, k) / 2.0

                    if is_dc == 0 && maximum(abs.(Θ_in_Search)) > π - Threshold_Dc
                        is_dc = 1
                    end

                    if multi_root == 0 && cost_in_search < Threshold_Rt
                        if sum(found_root) == 0
                            found_root .= Z_in_Search
                        elseif norm(Z_in_Search .- found_root, Inf) > Threshold_Rt_Dis
                            # Base.display(Z .- found_root);
                            multi_root = 1
                        end
                    end

                    if is_dc == 1 && multi_root == 1
                        break
                    end
                end

                # @printf "Boolean value: Discontinous? %i\t Multiple roots? %i.\nTest problem, scale\t%.8f\n" is_dc multi_root Radius_Mesh[s_ind]

                # display(Q);

                @printf(File, "%i %i\n", is_dc, multi_root)

                writedlm(File, Q[:, 1:k])
                write(File, "\n\n")

                push!(Failing_Run, (is_dc, multi_root, Q[:, 1:k]))
            end


        end

        vi = Record_iter[s_ind, :]
        vo = Record_ovhe[s_ind, :]
        vt = Record_time[s_ind, :]
        vg = Record_gcti[s_ind, :]

        vf = Record_fail[s_ind, :]

        @printf(
            "%.3f\t\t%.8f\t%.8f\t%.8f\t%.8f\t%.8f\t%.8f\t%.8f\n",
            Radius_Mesh[s_ind], avg_good(vi, vf), avg_good(vo, vf),
            avg_good(vt, vf), avg_good(vg, vf), mean(Record_fail[s_ind, :]),
            avg_bad(vt, vf) * sum(vf), avg_good(vt, vf) * sum(.~vf)
        )
        # @printf "%f\t%.1f\t%8f\t%.8f\t%.8f\t%.8f\t%.8f\t%i\t%.8f\n" Radius_Mesh[s_ind] mean(Record_iter[s_ind, :]) mean(Record_ovhe[s_ind, :]) mean(Record_time[s_ind, :]) mean(Record_gcti[s_ind, :]) mean(Record_fail[s_ind, :]) maximum(Record_iter[s_ind,:]) maximum(Record_time[s_ind,:]);
    end



    # for s_ind in eachindex(Radius_Mesh)
    #     @printf "%f\t%.1f\t%8f\t%.8f\t%.8f\t%.8f\t%.8f\t%i\t%.8f\n" Radius_Mesh[s_ind] mean(Record_iter[s_ind, :]) mean(Record_ovhe[s_ind, :]) mean(Record_time[s_ind, :] .- Record_ovhe[s_ind, :]) mean(Record_time[s_ind, :]) mean(Record_gcti[s_ind, :]) mean(Record_fail[s_ind, :]) maximum(Record_iter[s_ind,:]) maximum(Record_time[s_ind,:]);
    # end

    close(File)

    return Failing_Run
end

function check_root_and_connect(n, k; Radius_Mesh=1.0:0.1:2.0, Problem_Mesh_Size=5, Search_Mesh=100,
    Threshold_Dc=1e-1, Threshold_Rt=1e-1, Threshold_Rt_Dis=1e-1, FilePath="output.txt",
    MaxIter=200, MaxTime=1000, AbsTol=1e-8, RelTol=1e-5)


    dim::Int = div((n - k) * (n - k - 1), 2)
    dim_prob::Int = div(k * (n - 1 + n - k), 2)

    S::Matrix{Float64} = zeros(n, n)
    Q::Matrix{Float64} = zeros(n, n)
    QexpZ::Matrix{Float64} = zeros(n, n)
    logQZ::Matrix{Float64} = zeros(n, n)

    Z::Matrix{Float64} = zeros(n - k, n - k)
    expZ::Matrix{Float64} = zeros(n - k, n - k)


    Q_1::Matrix{Float64} = zeros(n, n - k)
    Q_2::Matrix{Float64} = zeros(n, n - k)


    # Ans = eval_on_mesh(x -> StLog_cost_by_Z(Q, rescaled_ssym_by_2norm(x), n, k), meshes);



    msize_prob::Vector{Int} = Vector{Int}(undef, dim_prob)
    msize_prob .= Problem_Mesh_Size
    msize_prob[1] = length(Radius_Mesh)


    # mend_prob::Vector{Float64} = Vector{Float64}(undef, dim_prob);
    # mend_prob .= π;
    # mend_prob[dim_prob] = 2π;

    # meshes_prob = [range(0, step = mend_prob[ii] / msize_prob[ii], length = msize_prob[ii]) for ii = 1:dim_prob];
    # meshes_prob = Radius_Mesh;

    meshes_prob = Vector{Any}(undef, length(msize_prob))

    meshes_prob[2:end] = spherical_angular_grids(msize_prob[2:end])
    # display(meshes_prob);
    meshes_prob[1] = Radius_Mesh

    display(meshes_prob)

    Record_dc::Array{Bool,length(msize_prob)} = Array{Bool,length(msize_prob)}(undef, msize_prob...)
    Record_rt::Array{Bool,length(msize_prob)} = Array{Bool,length(msize_prob)}(undef, msize_prob...)
    Record_tm::Array{Float64,length(msize_prob)} = Array{Float64,length(msize_prob)}(undef, msize_prob...)


    Record_alg::Vector{Any} = Vector{Any}(undef, 6)
    for ii = 1:5
        Record_alg[ii] = Vector{Float64}(undef, MaxIter)
    end
    Record_alg[6] = nothing
    Record_alg_r = [Ref(ele) for ele in Record_alg]

    msize::Vector{Int} = Vector{Int}(undef, dim)
    msize .= Search_Mesh

    # mend::Vector{Float64} = Vector{Float64}(undef, dim);
    # mend .= π;
    # mend[dim] = 2π;

    # meshes = [range(0, step = mend[ii] / msize[ii], length = msize[ii]) for ii = 1:dim]

    meshes = spherical_angular_grids(msize) # radius mesh is also 0 -> ... -> π.
    if n - k == 2
        meshes[1] = range(0, π - 5e-2, length(meshes[1]))
    end

    # Ans::Array{Float64, length(msize)} = Array{Float64, length(msize)}(undef, msize...);

    Cost::Float64 = 10.0
    scale::Float64 = 0.0
    is_dc::Int = 0
    multi_root::Int = 0
    Problem_Cnt::Int = 1
    Problem_Total_Cnt::Int = length(CartesianIndices(Record_dc))
    Printed_Percentage::Int = 0

    found_root::Matrix{Float64} = zeros(k, k)

    sphere_coordinate_S::Vector{Float64} = zeros(dim_prob)
    euclid_coordinate_S::Vector{Float64} = zeros(dim_prob)

    sphere_coordinate_Z::Vector{Float64} = zeros(dim)
    euclid_coordinate_Z::Vector{Float64} = zeros(dim)

    File = open(FilePath, "w")

    println("Start checking connectivity and multiple roots existence for test problem.")
    println(join(["-" for ii = 1:100]))

    for CarInd_Prob in CartesianIndices(Record_dc)
        # sphere_coordinate = get_mesh_point(meshes_prob, CarInd_Prob);
        # S .= euclidean_to_skew_sym(spherical_to_euclidean(sphere_coordinate); n = n);
        Cost = 10.0
        is_dc = 0
        multi_root = 0
        found_root .= 0.0

        get_mesh_point!(Ref(sphere_coordinate_S), meshes_prob, CarInd_Prob)
        spherical_to_euclidean!(Ref(euclid_coordinate_S), sphere_coordinate_S)
        euclidean_to_skew_sym!(Ref(S), euclid_coordinate_S)

        Q .= exp(S)

        for row = 1:n
            for col = 1:k
                QexpZ[row, col] = Q[row, col]
            end
        end

        for row = 1:n
            for col = (k+1):n
                Q_1[row, col-k] = Q[row, col]
            end
        end

        for CarInd in CartesianIndices(tuple(msize...))

            get_mesh_point!(Ref(sphere_coordinate_Z), meshes, CarInd)
            spherical_to_euclidean!(Ref(euclid_coordinate_Z), sphere_coordinate_Z)
            euclidean_to_skew_sym!(Ref(Z), euclid_coordinate_Z)

            if sphere_coordinate_Z[1] > 1e-9
                scale = sphere_coordinate_Z[1] / opnorm(Z, 2)
                for ind in eachindex(Z)
                    Z[ind] *= scale
                end
            end

            mul!(Q_2, Q_1, exp(Z))

            for row = 1:n
                for col = (k+1):n
                    QexpZ[row, col] = Q_2[row, col-k]
                end
            end

            logQZ .= real.(log(QexpZ))
            Cost = cost_in_test(Ref(logQZ), n, k)


            if is_dc == 0 && opnorm(logQZ, 2) > π - Threshold_Dc
                is_dc = 1
                # Base.display(eigvals(logQZ));
            end

            if multi_root == 0 && Cost < Threshold_Rt
                if norm(found_root, Inf) < 1e-9
                    found_root .= Z
                elseif norm(Z .- found_root, Inf) > Threshold_Rt_Dis
                    # Base.display(Z .- found_root);
                    multi_root = 1
                end
            end

            if is_dc == 1 && multi_root == 1
                break
            end

        end


        if n - k == 2
            S_ans, flag, iter = BCH_42_analysis(Q[:, 1:k], zeros(n, n - k); Init=init_guess_simple, Geo=UpZ_Geo, Records=Record_alg_r)
        else
            S_ans, flag, iter = BCH_2k_analysis(Q[:, 1:k], zeros(n, n - k); Init=init_guess_simple, Geo=UpZ_Geo, Records=Record_alg_r)
        end

        Record_dc[CarInd_Prob] = is_dc
        Record_rt[CarInd_Prob] = multi_root
        Record_tm[CarInd_Prob] = Record_alg[1][iter]

        # Base.display((is_dc, multi_root, iter, CarInd_Prob, Record_alg[1][iter]))

        # if is_dc == 1 || multi_root == 1
        #     Base.display((is_dc, multi_root, iter, CarInd_Prob, Record_alg[1][iter]))
        # end

        for ii = 1:(div(100 * Problem_Cnt, Problem_Total_Cnt)-Printed_Percentage)
            print("*")
            Printed_Percentage += 1
        end

        write(File, string(is_dc) * " " * string(multi_root) * " " * string(Record_tm[CarInd_Prob]) * "\n")
        Problem_Cnt += 1
    end
    close(File)
    # foreach(x -> Ans[x] = f(get_mesh_point(meshes, x)), CartesianIndices(Ans))


    return Record_dc, Record_rt, Record_tm
end

function check_root_and_connect_rand(n, k; Radius_Mesh=1.0:0.1:2.0, Search_Mesh=100, Seed=9527, Eval_Cnt=10000,
    Threshold_Dc=1e-1, Threshold_Rt=1e-1, Threshold_Rt_Dis=1e-1, FilePath="output.txt",
    MaxIter=200, MaxTime=1000, AbsTol=1e-8, RelTol=1e-5)

    eng = MersenneTwister(Seed)


    dim::Int = div((n - k) * (n - k - 1), 2)
    dim_prob::Int = div(k * (n - 1 + n - k), 2)

    S::Matrix{Float64} = zeros(n, n)
    Q::Matrix{Float64} = zeros(n, n)
    QexpZ::Matrix{Float64} = zeros(n, n)
    logQZ::Matrix{Float64} = zeros(n, n)

    Z::Matrix{Float64} = zeros(n - k, n - k)
    expZ::Matrix{Float64} = zeros(n - k, n - k)


    Q_1::Matrix{Float64} = zeros(n, n - k)
    Q_2::Matrix{Float64} = zeros(n, n - k)


    # Ans = eval_on_mesh(x -> StLog_cost_by_Z(Q, rescaled_ssym_by_2norm(x), n, k), meshes);





    # mend_prob::Vector{Float64} = Vector{Float64}(undef, dim_prob);
    # mend_prob .= π;
    # mend_prob[dim_prob] = 2π;

    # meshes_prob = [range(0, step = mend_prob[ii] / msize_prob[ii], length = msize_prob[ii]) for ii = 1:dim_prob];
    Record_dc::Array{Bool,2} = Array{Bool,2}(undef, Eval_Cnt, length(Radius_Mesh))
    Record_rt::Array{Bool,2} = Array{Bool,2}(undef, Eval_Cnt, length(Radius_Mesh))
    Record_tm::Array{Float64,2} = Array{Float64,2}(undef, Eval_Cnt, length(Radius_Mesh))


    Record_alg::Vector{Any} = Vector{Any}(undef, 6)
    for ii = 1:5
        Record_alg[ii] = Vector{Float64}(undef, MaxIter)
    end
    Record_alg[6] = nothing
    Record_alg_r = [Ref(ele) for ele in Record_alg]

    msize::Vector{Int} = Vector{Int}(undef, dim)
    msize .= Search_Mesh

    # mend::Vector{Float64} = Vector{Float64}(undef, dim);
    # mend .= π;
    # mend[dim] = 2π;

    # meshes = [range(0, step = mend[ii] / msize[ii], length = msize[ii]) for ii = 1:dim]

    meshes = spherical_angular_grids(msize) # radius mesh is also 0 -> ... -> π.
    if n - k == 2
        meshes[1] = range(0, π - 5e-2, length(meshes[1]))
    end

    # Ans::Array{Float64, length(msize)} = Array{Float64, length(msize)}(undef, msize...);

    Cost::Float64 = 10.0
    scale::Float64 = 0.0
    is_dc::Int = 0
    multi_root::Int = 0
    Problem_Cnt::Int = 1
    Problem_Total_Cnt::Int = length(CartesianIndices(Record_dc))
    Printed_Percentage::Int = 0

    found_root::Matrix{Float64} = zeros(k, k)

    sphere_coordinate_S::Vector{Float64} = zeros(dim_prob)
    euclid_coordinate_S::Vector{Float64} = zeros(dim_prob)

    sphere_coordinate_Z::Vector{Float64} = zeros(dim)
    euclid_coordinate_Z::Vector{Float64} = zeros(dim)

    File = open(FilePath, "w")

    println("Start checking connectivity and multiple roots existence for test problem.")
    println(join(["-" for ii = 1:100]))

    e_ind::Int = 0

    for r_ind in eachindex(Radius_Mesh)

        rad = Radius_Mesh[r_ind]
        e_ind = 1

        while e_ind <= Eval_Cnt
            # for CarInd_Prob in CartesianIndices(Record_dc)
            # sphere_coordinate = get_mesh_point(meshes_prob, CarInd_Prob);
            # S .= euclidean_to_skew_sym(spherical_to_euclidean(sphere_coordinate); n = n);
            Cost = 10.0
            is_dc = 0
            multi_root = 0
            found_root .= 0.0

            # get_mesh_point!(Ref(sphere_coordinate_S), meshes_prob, CarInd_Prob);
            # spherical_to_euclidean!(Ref(euclid_coordinate_S), sphere_coordinate_S);
            # euclidean_to_skew_sym!(Ref(S), euclid_coordinate_S);



            S = rand(eng, n, n)
            S[(k+1):n, (k+1):n] .= 0
            S .-= S'
            S .*= sqrt(2) * rad / norm(S)

            Q .= exp(S)

            for row = 1:n
                for col = 1:k
                    QexpZ[row, col] = Q[row, col]
                end
            end

            for row = 1:n
                for col = (k+1):n
                    Q_1[row, col-k] = Q[row, col]
                end
            end

            for CarInd in CartesianIndices(tuple(msize...))

                get_mesh_point!(Ref(sphere_coordinate_Z), meshes, CarInd)
                spherical_to_euclidean!(Ref(euclid_coordinate_Z), sphere_coordinate_Z)
                euclidean_to_skew_sym!(Ref(Z), euclid_coordinate_Z)

                if sphere_coordinate_Z[1] > 1e-9
                    scale = sphere_coordinate_Z[1] / opnorm(Z, 2)
                    for ind in eachindex(Z)
                        Z[ind] *= scale
                    end
                end

                mul!(Q_2, Q_1, exp(Z))

                for row = 1:n
                    for col = (k+1):n
                        QexpZ[row, col] = Q_2[row, col-k]
                    end
                end

                logQZ .= real.(log(QexpZ))
                Cost = cost_in_test(Ref(logQZ), n, k)


                if is_dc == 0 && opnorm(logQZ, 2) > π - Threshold_Dc
                    is_dc = 1
                    # Base.display(eigvals(logQZ));
                end

                if multi_root == 0 && Cost < Threshold_Rt
                    if norm(found_root, Inf) < 1e-9
                        found_root .= Z
                    elseif norm(Z .- found_root, Inf) > Threshold_Rt_Dis
                        # Base.display(Z .- found_root);
                        multi_root = 1
                    end
                end

                if is_dc == 1 && multi_root == 1
                    break
                end
            end

            if n - k == 2
                S_ans, flag, iter, GCt = BCH_42_analysis(Q[:, 1:k], zeros(n, n - k);
                    Init=init_guess_simple, Geo=UpZ_Geo, Records=Record_alg_r, Stop=terminator(MaxIter, MaxTime, AbsTol, RelTol))
            else
                S_ans, flag, iter, GCt = BCH_2k_analysis(Q[:, 1:k], zeros(n, n - k);
                    Init=init_guess_simple, Geo=UpZ_Geo, Records=Record_alg_r, Stop=terminator(MaxIter, MaxTime, AbsTol, RelTol))
            end


            Record_dc[e_ind, r_ind] = is_dc
            Record_rt[e_ind, r_ind] = multi_root
            Record_tm[e_ind, r_ind] = Record_alg[1][iter]

            # Base.display((is_dc, multi_root, iter, CarInd_Prob, Record_alg[1][iter]))

            # if is_dc == 1 || multi_root == 1
            #     Base.display((is_dc, multi_root, iter, CarInd_Prob, Record_alg[1][iter]))
            # end

            for ii = 1:(div(100 * Problem_Cnt, Problem_Total_Cnt)-Printed_Percentage)
                print("*")
                Printed_Percentage += 1
            end

            write(File, string(e_ind) * " " * string(r_ind) * " " * string(is_dc) * " " * string(multi_root) * " " * string(Record_tm[e_ind, r_ind]) * "\n")

            e_ind += 1

        end
    end
    close(File)
    # foreach(x -> Ans[x] = f(get_mesh_point(meshes, x)), CartesianIndices(Ans))
    return Record_dc, Record_rt, Record_tm
end

function check_time_rand(n, k, scales, eval_cnt;
    MaxIter=500, MaxTime=10000, AbsTol=1e-8, RelTol=1e-5, Output_File=nothing, order=0, seed=9527)

    Record_Time = zeros(length(scales), eval_cnt)
    Record_GCti = zeros(length(scales), eval_cnt)
    Record_Cost = zeros(length(scales), eval_cnt)
    Record_Leng = zeros(length(scales), eval_cnt)
    Record_Fail = zeros(Bool, length(scales), eval_cnt)
    Record_Iter = Matrix{Int}(undef, length(scales), eval_cnt)
    Record_OvHe = zeros(length(scales), eval_cnt)



    @printf("Numerical test for BCH solver on %i x %i problems.\n", n, k)
    @printf("Initial Length\tReturn Length\tAvg. Iter\tOverhead\tAvg. Time\tAvg. GCtime\tAvg. FailRate\n")


    Record_alg::Vector{Any} = Vector{Any}(undef, 6)
    for ii = 1:5
        Record_alg[ii] = Vector{Float64}(undef, MaxIter)
    end
    Record_alg[6] = nothing
    Record_alg_r = [Ref(ele) for ele in Record_alg]

    rand_eng = MersenneTwister(seed)

    for s_ind in eachindex(scales)
        s = scales[s_ind]
        for e_ind = 1:eval_cnt

            Record_alg[1] .= 0.0
            Record_alg[2] .= 0.0
            Record_alg[3] .= 0.0
            Record_alg[4] .= 0.0
            Record_alg[5] .= 0.0

            S = rand(rand_eng, n, n)
            S[(k+1):n, (k+1):n] .= 0.0
            S .-= S'
            S .*= sqrt(2) * s / norm(S)
            Q = exp(S)



            if order == 3
                S_ans, flag, iter, GCt = BCH_2k_analysis(Q[:, 1:k], zeros(n, n - k);
                    Init=init_guess_simple, Geo=UpZ_Geo, Records=Record_alg_r, Stop=terminator(MaxIter, MaxTime, AbsTol, RelTol))
            else
                S_ans, flag, iter, GCt = BCH_zero_2k_analysis(Q[:, 1:k], zeros(n, n - k);
                    Init=init_guess_simple, Geo=UpZ_Geo, Records=Record_alg_r, Stop=terminator(MaxIter, MaxTime, AbsTol, RelTol))
            end

            Record_Time[s_ind, e_ind] = Record_alg[1][iter]
            Record_GCti[s_ind, e_ind] = GCt
            Record_Cost[s_ind, e_ind] = Record_alg[2][iter]


            leng = 0.0
            for c_ind = 1:k
                for r_ind = (c_ind+1):n
                    leng += S_ans[r_ind, c_ind]^2
                end
            end
            leng = sqrt(leng)
            Record_Leng[s_ind, e_ind] = leng

            Record_Fail[s_ind, e_ind] = flag > 2
            Record_Iter[s_ind, e_ind] = iter

            Record_OvHe[s_ind, e_ind] = Record_alg[1][1]






        end

        @printf("%.8f\t%.8f\t%8f\t%.8f\t%.8f\t%.8f\t%.8f\n",
            scales[s_ind],
            mean(Record_Leng[s_ind, :]),
            mean(Record_Iter[s_ind, :]),
            mean(Record_OvHe[s_ind, :]),
            mean(Record_Time[s_ind, :] .- Record_OvHe[s_ind, :]),
            mean(Record_GCti[s_ind, :]),
            mean(Record_Fail[s_ind, :]))


        # @printf "%.8f\t%.8f\t%8f\t%.8f\t%.8f\t%.8f\t%.8f\t%i\t%.8f\n" scales[s_ind] mean(Record_leng[s_ind, :]) mean(Record_iter[s_ind, :]) mean(Record_ovhe[s_ind, :]) mean(Record_time[s_ind, :] .- Record_ovhe[s_ind, :]) mean(Record_gcti[s_ind, :]) mean(Record_fail[s_ind, :]);
    end

    if !isnothing(Output_File)
        file = open(Output_File, "w")
        writedlm(file, scales')
        write(file, "\n")
        writedlm(file, Record_Time .- Record_OvHe)
        write(file, "\n")
        writedlm(file, Record_GCti)
        write(file, "\n")
        writedlm(file, Record_Cost)
        write(file, "\n")
        writedlm(file, Record_Leng)
        write(file, "\n")
        writedlm(file, Record_Fail)
        write(file, "\n")
        writedlm(file, Record_Iter)
        write(file, "\n")
        close(file)
    end
    # for s_ind in eachindex(scales)
    #     println(scales[s_ind], "\t", sum(Record_time[s_ind, :]) / eval_cnt);
    # end
end

uneven_mesh(a, b, c, d; big_step=0.1, tiny_step=0.01) = vcat(a:big_step:b, (b+tiny_step):tiny_step:(c-tiny_step), c:big_step:d)

# test_problem_42(filepath) = check_root_and_connect(4, 2; FilePath = filepath,
#     Radius_Mesh = [3.0, 3.1, 3.4], Problem_Mesh_Size = 5, Search_Mesh = 100,
#     Threshold_Rt = 1e-4, Threshold_Rt_Dis = 1e-1, Threshold_Dc = 1e-2);

# test_problem_63(filepath) = check_root_and_connect(6, 3; FilePath = filepath,
#     Radius_Mesh = 0.4:0.1:0.6, Problem_Mesh_Size = 8, Search_Mesh = 50,
#     Threshold_Rt = 1e-2, Threshold_Rt_Dis = 1e-1, Threshold_Dc = 1e-2);

test_problem_k3r(filepath) = test_root_and_connect(6, 3; FilePath=filepath,
    Radius_Mesh=0.1:0.01:0.7, Search_Mesh=100, Eval_Cnt=100,
    Threshold_Rt=1e-2, Threshold_Rt_Dis=5e-1, Threshold_Dc=1e-2);

test_problem_k4r(filepath) = test_root_and_connect(8, 4; FilePath=filepath,
    Radius_Mesh=2.8:0.1:3.5, Search_Mesh=100, Eval_Cnt=100,
    Threshold_Rt=1e-2, Threshold_Rt_Dis=5e-1, Threshold_Dc=1e-2);

# test_problem_k5(filepath) = check_root_and_connect(10, 5; FilePath = filepath,
# Radius_Mesh = 2.5:0.5:3.5, Problem_Mesh_Size = 8, Search_Mesh = 50,
# Threshold_Rt = 1e-2, Threshold_Rt_Dis = 5e-1, Threshold_Dc = 1e-1);

# test_problem_k5r(filepath) = check_root_and_connect_rand(10, 5; FilePath = filepath,
# Radius_Mesh = 2.5:0.5:3.5, Search_Mesh = 50, Eval_Cnt = 1000,
# Threshold_Rt = 1e-2, Threshold_Rt_Dis = 5e-1, Threshold_Dc = 1e-1);




# Rec_DC, Rec_Root, Rec_Time = check_root_and_connect(4, 2; 
# Radius_Mesh = 2.5:0.1:3.0, Problem_Mesh_Size = 10, Search_Mesh = 100,
# Threshold_Rt = 1e-4, Threshold_Rt_Dis = 1e-1, Threshold_Dc = 1e-2);


# Rec_DC, Rec_Root, Rec_Time = check_root_and_connect(6, 3; 
# Radius_Mesh = 3.0:0.1:3.5, Problem_Mesh_Size = 4, Search_Mesh = 50,
# Threshold_Rt = 1e-2, Threshold_Rt_Dis = 1e-1, Threshold_Dc = 1e-2);



# Rec_DC, Rec_Roo, Rec_Time = check_root_and_connect_rand(6, 3; Radius_Mesh = 1.5:0.25:2.5, Angular_Sample_Size = 100, Search_Mesh = 40);



# function eval_on_mesh(f, meshes)
#     msize = [length(mesh) for mesh in meshes];

#     Ans::Array{Any, length(msize)} = Array{Any, length(msize)}(undef, msize...);
#     foreach(x -> Ans[x] = f(get_mesh_point(meshes, x)), CartesianIndices(Ans))
#     return Ans;
# end

# function eval_on_ball_mesh(f, n, rm, ams)
#     ms = [length(rm), ams...];
#     grids = prepend!(spherical_angular_grids(n - 1, ams), rm);

#     Ans::Array{Any, length(ms)} = Array{undef, ms...}();
#     foreach(x -> Ans[x] = f(get_mesh_point(grids, x)), CartesianIndices((ms...)))
#     return Ans;
# end:
