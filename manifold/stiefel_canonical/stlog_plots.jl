using DelimitedFiles, Plots, Statistics, Random

include(homedir() * "/Documents/julia/manifold/spec_orth/spec_orth.jl")

DISPLAY_CNT = 0
DISPLAY_MAX = 5

routine_labels = ["BCH solver" "System Solver" "Hybrid Solver"]
routine_colors = [:black :red :blue]
routine_marker = [:circle :rect :utriangle]

function readbox(f, n)
    local lines = ""
    local i = 1
    for i = 1:n
        lines = lines * "\n" * readline(f)
    end
    return (lines)
end

function readbox(f)
    box = ""
    line = readline(f)
    # Skip empty lines
    while isempty(line)
        line = readline(f)
    end
    # Read until the next empty line
    while !isempty(line)
        box = box * line * "\n"
        line = readline(f)
    end
    # print(box)
    return box
end

function readmatrix(f, T::Type)
    return readdlm(IOBuffer(readbox(f)), T)
end

function read_2k_test_result(File)
    file = open(File, "r")

    scales = Vector{Float64}()
    alg = Vector{Int}()

    str::String = ""

    str = readline(file)

    while !isempty(str)
        push!(scales, parse(Float64, str))
        str = readline(file)
    end

    str = readline(file)

    while !isempty(str)
        push!(alg, parse(Int, str))
        str = readline(file)
    end

    file_pos = position(file)

    str = readline(file)

    temp_mat = Vector{Any}()

    while !isempty(str)
        push!(temp_mat, [parse(Float64, elem) for elem in split(str)])
        str = readline(file)
    end

    seek(file, file_pos)

    Rec_Time = Array{Float64,3}(undef, length(scales), length(temp_mat[1]), length(alg))
    Rec_OvHe = Array{Float64,3}(undef, length(scales), length(temp_mat[1]), length(alg))
    Rec_GCti = Array{Float64,3}(undef, length(scales), length(temp_mat[1]), length(alg))
    Rec_Cost = Array{Float64,3}(undef, length(scales), length(temp_mat[1]), length(alg))
    Rec_Leng = Array{Float64,3}(undef, length(scales), length(temp_mat[1]), length(alg))
    Rec_Fail = Array{Bool,3}(undef, length(scales), length(temp_mat[1]), length(alg))
    Rec_Iter = Array{Int,3}(undef, length(scales), length(temp_mat[1]), length(alg))


    for a_ind = 1:length(alg)
        Rec_Time[:, :, a_ind] .= readdlm(IOBuffer(readbox(file)), Float64)
        Rec_OvHe[:, :, a_ind] .= readdlm(IOBuffer(readbox(file)), Float64)
        Rec_GCti[:, :, a_ind] .= readdlm(IOBuffer(readbox(file)), Float64)
        Rec_Cost[:, :, a_ind] .= readdlm(IOBuffer(readbox(file)), Float64)
        Rec_Leng[:, :, a_ind] .= readdlm(IOBuffer(readbox(file)), Float64)
        Rec_Fail[:, :, a_ind] .= readdlm(IOBuffer(readbox(file)), Bool)
        Rec_Iter[:, :, a_ind] .= readdlm(IOBuffer(readbox(file)), Int)
    end
    close(file)

    return scales, alg, Rec_Time, Rec_OvHe, Rec_GCti, Rec_Cost, Rec_Leng, Rec_Fail, Rec_Iter

end

function read_bch_test_result(File)
    file = open(File, "r")

    scales = vcat(readmatrix(file, Float64)...)
    Rec_Time = readmatrix(file, Float64)
    Rec_GCti = readmatrix(file, Float64)
    Rec_Cost = readmatrix(file, Float64)
    Rec_Leng = readmatrix(file, Float64)
    Rec_Fail = readmatrix(file, Bool)
    Rec_Iter = readmatrix(file, Int)


    close(file)

    return scales, Rec_Time, Rec_GCti, Rec_Cost, Rec_Leng, Rec_Fail, Rec_Iter
end

function plot_bch_result(File, order, n, k)
    s, rt, rg, rc, rl, rf, ri = read_bch_test_result(File)

    rt_m = [mean(row) for row in eachrow(rt)]
    rl_m = [mean(row) for row in eachrow(rl)]


    rt_σ = [std(row) for row in eachrow(rt)]

    plt = plot(s, rt_m,
        yscale=:log10,
        label="BCH order $(order), $(size(rt, 2)) runs",
        legend=:topleft,
        title="Performance of BCH solver on $(n) × $(k) problems."
        # ribbon=rt_σ,
        # fillalpha=.5,
    )

    mat_s = s * ones(size(rt, 2))'
    vec_s = vcat(mat_s...)

    scatter!(vec_s, vcat(rt...),
        label=:none,
        color=:blue,
        lw=0,
        ms=1,
        mc=:blue,
        ma=0.01
    )

    l_diff = vcat(abs.(rl .- mat_s)...)
    nz_ind = l_diff .> 1e-10

    ylims!(1e-2, 1e2)
    xlabel!("The length of the generating geodesic.")
    ylabel!("Time (ms).")

    nz_ind_mean = abs.(rl_m .- s) .> 1e-10

    plot!(twinx(), s[nz_ind_mean], abs.(rl_m[nz_ind_mean] .- s[nz_ind_mean]),
        label="Length difference",
        # label="BCH, Length difference between \ngenerating and returning geodesics",
        legend=:bottomleft,
        color=:red,
        ylims=(0.0, 1.0),
        ylabel="Length difference."
        # ribbon=rt_σ,
        # fillalpha=.5,
    )


    scatter!(twinx(), vec_s[nz_ind], l_diff[nz_ind],
        label=:none,
        color=:red,
        markershape=:circle,
        ms=1,
        ma=0.01,
        ylims=(0.0, 1.0),
        xticks=:none,
        yticks=:none
    )


    return plt


    # savefig(plt, SubString(File, 1, findlast('.', File)) * "png")
    # display(plt)
end

function plot_bch_03_result(File0, File3, n, k)

    plt0 = plot_bch_result(File0, 0, n, k)
    plt3 = plot_bch_result(File3, 3, n, k)

    plt = plot(plt0, plt3, layout=(2, 1), link=:all)


    savefig(plt, SubString(File0, 1, findlast('.', File0)) * "png")
    display(plt)
end

function recur_max(A)
    mA = maximum(maximum(a) for a in A)
    if mA != A
        return recur_max(mA)
    else
        return mA
    end
end

function recur_min(A)
    mA = minimum(minimum(a) for a in A)
    if mA != A
        return recur_min(mA)
    else
        return mA
    end
end

function plot_multi_algo_2k(File, n, k; data=nothing, title="")
    if isnothing(data)
        s, alg, rt, ro, rg, rc, rl, rf, ri = read_2k_test_result(File)
    else
        s, alg, rt, ro, rg, rc, rl, rf, ri = data
    end

    rt_mean = [[mean(row) for row in eachrow(rt[:, :, a_ind])] for a_ind in eachindex(alg)]
    rt_max = [[maximum(row) for row in eachrow(rt[:, :, a_ind])] for a_ind in eachindex(alg)]
    rt_min = [[minimum(row) for row in eachrow(rt[:, :, a_ind])] for a_ind in eachindex(alg)]

    rt_rate_mean = [[mean(rt[s_ind, :, a_ind] ./ rt[s_ind, :, 1]) for s_ind in axes(rt)[1]] for a_ind in eachindex(alg)]
    rt_rate_max = [[maximum(rt[s_ind, :, a_ind] ./ rt[s_ind, :, 1]) for s_ind in axes(rt)[1]] for a_ind in eachindex(alg)]
    rt_rate_min = [[minimum(rt[s_ind, :, a_ind] ./ rt[s_ind, :, 1]) for s_ind in axes(rt)[1]] for a_ind in eachindex(alg)]

    mat_s = s * ones(size(rt, 2))'
    vec_s = vcat(mat_s...)


    plt_time = plot(s, rt_mean,
        title="Performance on $(n) × $(k) problems" * title,
        xlabel="The length of the generating geodesic.",
        ylabel="Time (ms).",
        linewidth=2,
        color=reshape(routine_colors[collect(alg)], 1, length(alg)),
        label=reshape(routine_labels[collect(alg)] .* " Compute time", 1, length(alg)),
        legend=:topleft,
        yscale=:identity,
        # ylims=(10^(floor(log10(recur_min(rt_min)))), 10^(ceil(log10(recur_max(rt_max)))))
    )

    # for a_ind in eachindex(alg)

    #     scatter!(twinx(), vec_s, vcat((rt[:, :, 1] ./ rt[:, :, a_ind])...),
    #         linewidth=1,
    #         fillrange=rt_min[a_ind],
    #         label=:none,
    #         lw=0,
    #         ms=3,
    #         ma=0.1,
    #         mc=routine_colors[alg[a_ind]],
    #         m=routine_marker[alg[a_ind]],
    #         yscale=:log2,
    #         # ylims=(2^-4, 2^5),
    #         ylabel="Time ratio BCH solver over other solver",
    #         xlabel=:none,
    #     )
    # end
    scatter!(twinx(), vec_s, [vcat((rt[:, :, 1] ./ rt[:, :, a_ind])...) for a_ind in eachindex(alg)],
        linewidth=1,
        lw=0,
        ms=3,
        ma=0.1,
        # mc=routine_colors[alg[a_ind]],
        # m=routine_marker[alg[a_ind]],
        color=reshape(routine_colors[collect(alg)], 1, length(alg)),
        label=reshape(routine_labels[collect(alg)] .* " Compute time ratio", 1, length(alg)),
        marker=reshape(routine_marker[collect(alg)], 1, length(alg)),
        yscale=:log2,
        # ylims=(2^-4, 2^5),
        ylabel="Time ratio BCH solver over other solver",
        # xlabel=:none,
        legend=:topright,
        grid=true,
        minorgrid=true,
        minorticks=true,
    )
    # display(plt_time)

    return plt_time
end

function plot_multi_algo_against_bch_2k(File, n, k; data=nothing, title="")
    if isnothing(data)
        s, alg, rt, ro, rg, rc, rl, rf, ri = read_2k_test_result(File)
    else
        s, alg, rt, ro, rg, rc, rl, rf, ri = data
    end

    plt_time = scatter(vcat(rt[:, :, 1]...), [vcat(rt[:, :, a_ind]...) for a_ind in eachindex(alg)],
        title="Performance on $(n) × $(k) problems" * title,
        xlabel="The BCH solver compute time(ms).",
        ylabel="Compute Time (ms).",
        color=reshape(routine_colors[collect(alg)], 1, length(alg)),
        label=reshape(routine_labels[collect(alg)] .* " Compute time", 1, length(alg)),
        marker=reshape(routine_marker[collect(alg)], 1, length(alg)),
        legend=:topleft,
        yscale=:identity,
        ms=3,
        ma=0.1,
        # ylims=(10^(floor(log10(recur_min(rt_min)))), 10^(ceil(log10(recur_max(rt_max)))))
    )

    plt_ratio = scatter(vcat(rt[:, :, 1]...), [vcat((rt[:, :, 1] ./ rt[:, :, a_ind])...) for a_ind in eachindex(alg)],
        title="Performance on $(n) × $(k) problems" * title,
        xlabel="The BCH solver compute time(ms).",
        ylabel="Compute Time ratio.",
        color=reshape(routine_colors[collect(alg)], 1, length(alg)),
        label=reshape(routine_labels[collect(alg)] .* " Ratio", 1, length(alg)),
        marker=reshape(routine_marker[collect(alg)], 1, length(alg)),
        legend=:topleft,
        yscale=:log2,
        ms=3,
        ma=0.1,
        # ylims=(10^(floor(log10(recur_min(rt_min)))), 10^(ceil(log10(recur_max(rt_max)))))
    )

    plt = plot(plt_time, plt_ratio, layout=(2, 1))
    savefig(plt, File[1:findlast('.', File)] * ".svg")
    display(plt)
    return plt_time, plt_ratio
end

function match_angle!(A_r, PA_r, B_r, PB_r; threshold=1e-1)
    # input angles are nonnegative. The match is based on the angles.
    # if there is a block R_i[0 -θ; θ 0]R_i mismatched around π, it means 
    # θ = -(ω - 2π) = 2π - ω. Then there was an interchange happened to R_i.

    A = A_r[]
    B = B_r[]
    PA = PA_r[]
    PB = PB_r[]
    diff::Float64 = 100.0
    boolA = zeros(Bool, length(A))
    boolB = zeros(Bool, length(B))
    temp = zeros(size(PA, 1))
    flag_pm::Int = 0
    updateS::Bool = false

    matchedA::Int = 0
    matchedB::Int = 0



    for a_ind in eachindex(A)
        diff = 100.0
        matchedA = a_ind
        for b_ind in eachindex(B)
            if boolB[b_ind]
                continue
            end

            if abs(A[a_ind] - B[b_ind]) < diff
                diff = abs(A[a_ind] - B[b_ind])
                matchedB = b_ind
            end
        end

        if diff < threshold
            boolA[matchedA] = true
            boolB[matchedB] = true

            # For the matched pair, one should verify if they needed to be flipped.
            if norm(PA[:, 2*matchedA-1] - PB[:, 2*matchedB-1]) > 1e-1
                updateS = true
                A[matchedA] = 2π - A[matchedA]

                temp .= PA[:, 2*matchedA-1]
                PA[:, 2*matchedA-1] .= PA[:, 2*matchedA]
                PA[:, 2*matchedA] .= temp
            end
        end
    end

    for a_ind in eachindex(A)
        if boolA[a_ind]
            continue
        end
        diff = 100
        matchedA = a_ind
        for b_ind in eachindex(B)
            if boolB[b_ind]
                continue
            end

            if abs(A[a_ind] - B[b_ind] + 2π) < diff
                flag_pm = 1
                diff = abs(A[a_ind] - B[b_ind] + 2π)
                matchedB = b_ind
            end
            if abs(A[a_ind] - B[b_ind] - 2π) < diff
                flag_pm = -1
                diff = abs(A[a_ind] - B[b_ind] - 2π)
                matchedB = b_ind
            end
            if abs(A[a_ind] - B[b_ind]) < diff
                flag_pm = 0
                diff = abs(A[a_ind] - B[b_ind])
                matchedB = b_ind
            end
        end

        boolA[matchedA] = true
        boolB[matchedB] = true
        if flag_pm == 0
            continue
        else
            A[matchedA] = flag_pm > 0 ? A[matchedA] + 2π : A[matchedA] - 2π
            updateS = true
        end
    end
    return updateS
end

function log_skew_near_by!(S_r, P_r, Θ_r, U_r, lastΘ_r, wsp_log)
    lastP = similar(P_r[])
    lastP .= P_r[]

    log_skew!(S_r, P_r, Θ_r, U_r, wsp_log; orderP=true)
    Θ_before_match = similar(Θ_r[])
    Θ_before_match .= Θ_r[]

    if match_angle!(Θ_r, lastΘ_r)
        if DISPLAY_CNT < DISPLAY_MAX
            display(lastP)
            display(lastΘ_r[])
            println()
            display(P_r[])
            display(Θ_before_match)
            println()
            display(Θ_r[])
            println("------------------------------------")
            global DISPLAY_CNT = DISPLAY_CNT + 1
        end
        S = S_r[]
        S .= 0.0
        get_S!(P_r, Θ_r, S_r)
    end
end

function plot_2d_cost(n, rad; seed=9527, meshsize=1000)
    k = n - 2

    eng = MersenneTwister(seed)
    global DISPLAY_CNT = 0


    X = rand(eng, n, n)
    X[(k+1):n, (k+1):n] .= 0
    X .-= X'
    X .*= sqrt(2) * rad / norm(X)
    Q = exp(X)

    xmesh_p = range(0, π + π, 2 * meshsize)
    xmesh_n = range(0, -π - π, 2 * meshsize)
    xmesh = range(-π, π, 2 * meshsize - 1)
    val = Vector{Float64}(undef, length(xmesh))
    val_p = Vector{Float64}(undef, length(xmesh_p))
    val_n = Vector{Float64}(undef, length(xmesh_n))

    U::Matrix{Float64} = Matrix{Float64}(undef, n, n)
    S::Matrix{Float64} = Matrix{Float64}(undef, n, n)
    P::Matrix{Float64} = Matrix{Float64}(undef, n, n)
    Θ::Vector{Float64} = Vector{Float64}(undef, div(n, 2))
    lastΘ::Vector{Float64} = Vector{Float64}(undef, div(n, 2))
    wsp_log = get_wsp_log(n)

    U_r = Ref(U)
    S_r = Ref(S)
    P_r = Ref(P)
    Θ_r = Ref(Θ)
    lastΘ_r = Ref(lastΘ)

    ind::Int = 1

    for x in xmesh
        U .= Q
        U[:, (k+1):n] .= U[:, (k+1):n] * [cos(x) -sin(x); sin(x) cos(x)]
        log_skew!(S_r, P_r, Θ_r, U_r, wsp_log)
        val[ind] = S[n, k+1]
        # val[ind] = abs(S[n, k+1])
        # val[ind] = S[n, k+1]^2 / 4
        ind = ind + 1
    end

    # ind = 1
    # for x in xmesh_p
    #     U .= Q
    #     U[:, (k+1):n] .= U[:, (k+1):n] * [cos(x) -sin(x); sin(x) cos(x)]
    #     if ind == 1
    #         log_skew!(S_r, P_r, Θ_r, U_r, wsp_log; orderP=true)
    #         lastΘ .= Θ
    #     else
    #         log_skew_near_by!(S_r, P_r, Θ_r, U_r, lastΘ_r, wsp_log)
    #         lastΘ .= Θ
    #     end
    #     # val_p[ind] = S[n, k+1]^2 / 4
    #     val_p[ind] = S[n, k+1]

    #     ind = ind + 1
    # end

    # ind = 1
    # for x in xmesh_n
    #     U .= Q
    #     U[:, (k+1):n] .= U[:, (k+1):n] * [cos(x) -sin(x); sin(x) cos(x)]
    #     if ind == 1
    #         log_skew!(S_r, P_r, Θ_r, U_r, wsp_log; orderP=true)
    #         lastΘ .= Θ
    #     else
    #         log_skew_near_by!(S_r, P_r, Θ_r, U_r, lastΘ_r, wsp_log)
    #         lastΘ .= Θ
    #     end
    #     # val_n[ind] = S[n, k+1]^2 / 4
    #     val_n[ind] = S[n, k+1]
    #     ind = ind + 1
    # end

    plt1 = scatter(xmesh, val,
        markerstrokewidth=0,
        markersize=1,
        label=:none,
        xlabel="θ",
        ylabel="C_{2,1}"
    )
    plt2 = scatter(xmesh_p, val_p)
    scatter!(xmesh_n, val_n)

    # display(plot(plt1, plt1, link=:all))
    savefig(plot(plt1, plt1, link=:all), "discontinuity.pdf")


end



# rk30q = read_2k_test_result("data/k30r_q.txt");
# rk30p = read_2k_test_result("data/k30r_p.txt");
# rk30g = read_2k_test_result("data/k30r_g.txt");
# rk50q = read_2k_test_result("data/k50r_q.txt");
# rk50p = read_2k_test_result("data/k50r_p.txt");
# rk50g = read_2k_test_result("data/k50r_g.txt");

