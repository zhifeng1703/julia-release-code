
include("stlog_solver.jl")

#######################################Tuning setting#######################################

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

FAIL_STEP = 10.0



DETAILED_RUN = 5

LINESEARCH_CHECK = x -> (x != 3)

#######################################Test setting#######################################


USE_BENCHMARK = true
USE_ANALYSIS = true

USE_BENCHMARK = false
USE_ANALYSIS = false

BENCHMARK_EVALS = 1
BENCHMARK_SAMPLES = 100

SHORT_RUN = max(8, BCH_MAX_ITER + 5)

#######################################Test functions#######################################

using Plots, Random
 
skewFnorm(X) = sqrt(dot(X, X) / 2.0);

function test_stlog_alg(MatUk, alg_analysis, alg_label; MaxIter=500, AbsTol=1e-10, MaxTime=20000)
    n, k = size(MatUk)
    MatUp = similar(MatUk)


    TimeRec = zeros(MaxIter)
    AbsERec = zeros(MaxIter)
    DistRec = zeros(MaxIter)
    VectRec = zeros(MaxIter)
    StepRec = ones(MaxIter)
    AngsRec = zeros(MaxIter, k)
    StPtRec = Vector{Any}(undef, MaxIter)

    Records = [Ref(TimeRec), Ref(AbsERec), Ref(DistRec), Ref(VectRec), Ref(StepRec), Ref(AngsRec), Ref(StPtRec)]

    MatS, flag, iter, time, = alg_analysis(MatUk, MatUp; Stop=terminator(MaxIter, MaxTime, AbsTol, 1e-5), Init=init_guess_simple, Records=Records, Solver_Stop=SOLVER_STOP, NMLS_Set=NMLS_SET)


    MatQ = exp(MatS)

    println(MatUk ≈ MatQ[:, 1:k])

    plt_cvi = plot(1:iter, AbsERec[1:iter],
        yaxis=:log,
        title="Objective value vs iterations"
    )

    plt_avi = plot(1:iter, AngsRec[1:iter, 1:min(k, 5)],
        title="Angles vs iteration",
        label=:none
    )

    plot!(1:iter, 0.5 .* (AngsRec[1:iter, 1] .+ AngsRec[1:iter, 2]),
        label="(Θ_1 + Θ_2) / 2",
        markershape=:xcross
    )

    plot!(1:iter, π .* ones(iter),
        label="θ = π",
        line=:dash
    )

    plt_cvt = plot(TimeRec[1:iter], AbsERec[1:iter],
        yaxis=:log,
        title="Objective value vs time"
    )

    display(plot(plt_cvi, plt_avi, plt_cvt, layout=(3, 1), size = (1000, 500),title=alg_label))

    return MatS
end

function test_stlog_multi_alg(MatUk, alg_analysis, alg_label; MaxIter=500 .* ones(Int, length(alg_analysis)), AbsTol=1e-10, MaxTime=20000)
    n, k = size(MatUk)
    blk_dim = div(n, 2)
    MatUp = similar(MatUk)

    alg_len = length(alg_analysis)

    TimeRec = zeros(maximum(MaxIter))
    AbsERec = zeros(maximum(MaxIter))
    DistRec = zeros(maximum(MaxIter))
    VectRec = zeros(maximum(MaxIter))
    StepRec = ones(maximum(MaxIter))
    AngsRec = zeros(maximum(MaxIter), k)
    StPtRec = Vector{Any}(undef, maximum(MaxIter))

    Records = [Ref(TimeRec), Ref(AbsERec), Ref(DistRec), Ref(VectRec), Ref(StepRec), Ref(AngsRec), Ref(StPtRec)]


    AllFlag = zeros(Int, alg_len)
    AllStopInd = zeros(Int, alg_len)
    AllTimeRec = zeros(maximum(MaxIter), alg_len)
    AllAbsERec = zeros(maximum(MaxIter), alg_len)
    AllDistRec = zeros(maximum(MaxIter), alg_len)
    AllVectRec = zeros(maximum(MaxIter), alg_len)
    AllStepRec = ones(maximum(MaxIter), alg_len)
    AllAngsRec = zeros(maximum(MaxIter), k, alg_len)


    for a_ind in eachindex(alg_analysis)
        TimeRec .= 0.0
        MatS, flag, iter, time, = alg_analysis[a_ind](MatUk, MatUp; Stop=terminator(MaxIter[a_ind], MaxTime, AbsTol, 1e-5), Init=init_guess_simple, Records=Records, Solver_Stop=SOLVER_STOP, NMLS_Set=NMLS_SET)

        AllFlag[a_ind] = flag
        AllStopInd[a_ind] = iter
        AllTimeRec[1:iter, a_ind] .= TimeRec[1:iter]
        AllAbsERec[1:iter, a_ind] .= AbsERec[1:iter]
        AllAngsRec[1:iter, :, a_ind] .= AngsRec[1:iter, :]
    end


    plt_cvi = plot(
        title="Objective value vs iterations"
    )

    for a_ind in eachindex(alg_analysis)
        iter = AllStopInd[a_ind]
        plot!(1:iter, AllAbsERec[1:iter, a_ind],
            label=alg_label[a_ind],
            yaxis=:log10,
        )
    end

    plt_cvt = plot(
        title="Objective value vs time"
    )

    for a_ind in eachindex(alg_analysis)
        iter = AllStopInd[a_ind]
        plot!(AllTimeRec[1:iter, a_ind], AllAbsERec[1:iter, a_ind],
            label=:none,
            yaxis=:log10,
        )
    end

    plt_avi = plot(
        title="Angles vs iteration",
    )

    for a_ind in eachindex(alg_analysis)
        iter = AllStopInd[a_ind]
        MatAs = AllAngsRec[1:AllStopInd[a_ind], :, a_ind] .^ 2 ./ blk_dim
        plot!(1:iter, hcat(AllAngsRec[1:iter, 1:2, a_ind], sum(MatAs, dims = 2)),
            label=alg_label[a_ind] .*[" first angle" " second angle" " avg. squared sum"],legend=:topright
        )
        # plot!(1:iter, 0.5 .* (AngsRec[1:iter, 1] .+ AngsRec[1:iter, 1]), 
        # label = alg_label[a_ind] * " (Θ_1 + Θ_2) / 2",
        # markershape  = :xcross
        # )
        plot!(1:iter, π .* ones(iter),
            label=:none,
            line=:dash
        )
    end

    # display(plot(plt_cvi, plt_cvt, plt_avi, layout=(3, 1), formatter = (x -> @sprintf("%.2f", x))))
    # display(plot(plt_cvi, plt_cvt, plt_avi, layout=(3, 1), size = (600, 1000)))

    display(plot(plt_cvi, plt_cvt, layout=(2, 1), size = (600, 800)))

end

function test_BCH1(k=10; MaxIter=500, AbsTol=1e-10, MaxTime=10000, seed=9527)
    rand_eng = MersenneTwister(seed)
    X = rand(rand_eng, 2k, 2k)
    X .-= X'
    X[(k+1):2k, (k+1):2k] .= 0.0
    X .*= (π - 0.01) / opnorm(X)

    MatU = exp(X)

    MatUk = zeros(2k, k)
    MatUp = zeros(2k, k)


    MatUk .= MatU[:, 1:k]

    TimeRec = zeros(MaxIter)
    AbsERec = zeros(MaxIter)
    DistRec = zeros(MaxIter)
    VectRec = zeros(MaxIter)
    StepRec = ones(MaxIter)
    AngsRec = zeros(MaxIter, k)
    StPtRec = Vector{Any}(undef, MaxIter)

    Records = [Ref(TimeRec), Ref(AbsERec), Ref(DistRec), Ref(VectRec), Ref(StepRec), Ref(AngsRec), Ref(StPtRec)]

    MatS, flag, iter, time, = stlog_BCH1_2k_analysis(MatUk, MatUp; Stop=terminator(MaxIter, MaxTime, AbsTol, 1e-5), Init=init_guess_simple, Records=Records)

    MatQ = exp(MatS)

    println(MatU[:, 1:k] ≈ MatQ[:, 1:k])

    plt_cvi = plot(1:iter, AbsERec[1:iter],
        yaxis=:log,
        title="Objective value vs iterations"
    )

    plt_avi = plot(1:iter, AngsRec[1:iter, 1:min(k, 5)],
        title="Angles vs iteration",
        label=:none
    )

    plt_cvt = plot(TimeRec[1:iter], AbsERec[1:iter],
        yaxis=:log,
        title="Objective value vs time"
    )

    display(plot(plt_cvi, plt_avi, plt_cvt, layout=(3, 1)))

end

function test_BCH3(k=10; MaxIter=500, AbsTol=1e-10, MaxTime=10000, seed=9527)
    rand_eng = MersenneTwister(seed)
    X = rand(rand_eng, 2k, 2k)
    X .-= X'
    X[(k+1):2k, (k+1):2k] .= 0.0
    X .*= (π - 0.01) / opnorm(X)

    MatU = exp(X)

    MatUk = zeros(2k, k)
    MatUp = zeros(2k, k)


    MatUk .= MatU[:, 1:k]

    TimeRec = zeros(MaxIter)
    AbsERec = zeros(MaxIter)
    DistRec = zeros(MaxIter)
    VectRec = zeros(MaxIter)
    StepRec = ones(MaxIter)
    AngsRec = zeros(MaxIter, k)
    StPtRec = Vector{Any}(undef, MaxIter)

    Records = [Ref(TimeRec), Ref(AbsERec), Ref(DistRec), Ref(VectRec), Ref(StepRec), Ref(AngsRec), Ref(StPtRec)]

    MatS, flag, iter, time, = stlog_BCH3_2k_analysis(MatUk, MatUp; Stop=terminator(MaxIter, MaxTime, AbsTol, 1e-5), Init=init_guess_simple, Records=Records)

    MatQ = exp(MatS)

    println(MatU[:, 1:k] ≈ MatQ[:, 1:k])

    plt_cvi = plot(1:iter, AbsERec[1:iter],
        yaxis=:log,
        title="Objective value vs iterations"
    )

    plt_avi = plot(1:iter, AngsRec[1:iter, 1:min(k, 5)],
        title="Angles vs iteration",
        label=:none
    )

    plt_cvt = plot(TimeRec[1:iter], AbsERec[1:iter],
        yaxis=:log,
        title="Objective value vs time"
    )

    display(plot(plt_cvi, plt_avi, plt_cvt, layout=(3, 1)))

end

function test_stlog_Newton(k=10; MaxIter=500, AbsTol=1e-10, MaxTime=10000, seed=9527)
    rand_eng = MersenneTwister(seed)
    X = rand(rand_eng, 2k, 2k)
    X .-= X'
    X[(k+1):2k, (k+1):2k] .= 0.0
    X .*= (π + 0.1) / opnorm(X)

    MatU = exp(X)

    MatUk = zeros(2k, k)
    MatUp = zeros(2k, k)


    MatUk .= MatU[:, 1:k]

    TimeRec = zeros(MaxIter)
    AbsERec = zeros(MaxIter)
    DistRec = zeros(MaxIter)
    VectRec = zeros(MaxIter)
    StepRec = ones(MaxIter)
    AngsRec = zeros(MaxIter, k)
    StPtRec = Vector{Any}(undef, MaxIter)

    Records = [Ref(TimeRec), Ref(AbsERec), Ref(DistRec), Ref(VectRec), Ref(StepRec), Ref(AngsRec), Ref(StPtRec)]

    MatS, flag, iter, time, = stlog_hybrid_Newton_armijo_analysis(MatUk, MatUp;
        Stop=terminator(MaxIter, MaxTime, AbsTol, 1e-5), Solver_Stop=SOLVER_STOP,
        NMLS_Set=NMLS_SET, Init=init_guess_simple, Records=Records)

    MatQ = exp(MatS)

    println(MatU[:, 1:k] ≈ MatQ[:, 1:k])

    plt_cvi = plot(1:iter, AbsERec[1:iter],
        yaxis=:log,
        title="Objective value vs iterations"
    )

    plt_avi = plot(1:iter, AngsRec[1:iter, 1:min(k, 5)],
        title="Angles vs iteration",
        label=:none
    )

    plt_cvt = plot(TimeRec[1:iter], AbsERec[1:iter],
        yaxis=:log,
        title="Objective value vs time"
    )

    display(plot(plt_cvi, plt_avi, plt_cvt, layout=(3, 1)))

end

function test_alg_speed(k, scale_grid, algs, algs_analysis, labels, runs=100;
    markershapes = fill(:circle, 1, length(algs)), 
    MaxIter=500, AbsTol=1e-10, MaxTime=10000, seed=9527, scaleby=opnorm, filename="")
    n = 2k
    rand_eng = MersenneTwister(seed)

    MatU = zeros(2k, 2k)
    MatUk = zeros(2k, k)
    MatUp = zeros(2k, k)

    flags = zeros(Int, length(algs))
    iters = zeros(Int, length(algs))

    TimeRec = [zeros(MaxIter[ind]) for ind in eachindex(MaxIter)]
    AbsERec = [zeros(MaxIter[ind]) for ind in eachindex(MaxIter)]
    DistRec = [zeros(MaxIter[ind]) for ind in eachindex(MaxIter)]
    VectRec = [zeros(MaxIter[ind]) for ind in eachindex(MaxIter)]
    StepRec = [ones(MaxIter[ind]) for ind in eachindex(MaxIter)]
    AngsRec = [zeros(MaxIter[ind], k) for ind in eachindex(MaxIter)]
    StPtRec = [Vector{Any}(undef, MaxIter[ind]) for ind in eachindex(MaxIter)]

    Records = [[Ref(TimeRec[ind]), Ref(AbsERec[ind]), Ref(DistRec[ind]), Ref(VectRec[ind]), Ref(StepRec[ind]), Ref(AngsRec[ind]), Ref(StPtRec[ind])] for ind in eachindex(MaxIter)]

    wsp = [get_wsp_alg(n, k, MaxIter[a_ind], algs[a_ind]) for a_ind in eachindex(algs)]

    RecTime = zeros(length(scale_grid) * runs, length(algs))
    RecIter = zeros(Int, length(scale_grid) * runs, length(algs))


    scale_vec = vcat(ones(runs) * scale_grid'...)

    FailCnt = zeros(length(algs))

    FailRec = []
    ShortRec = []



    detailed_run_cnt = 0

    for s_ind = eachindex(scale_grid)
        for r_ind = 1:runs
            scale = scale_grid[s_ind]
            X = rand(rand_eng, 2k, 2k)
            X .-= X'
            X[(k+1):2k, (k+1):2k] .= 0.0
            X .*= scale / scaleby(X)

            MatU .= exp(X)

            MatUk .= MatU[:, 1:k]

            for a_ind in eachindex(algs)
                record_ind = (s_ind-1)*runs+r_ind

                fill!(TimeRec[a_ind], 0.0)
                fill!(AbsERec[a_ind], 0.0)
                fill!(DistRec[a_ind], 0.0)
                fill!(VectRec[a_ind], 0.0)
                fill!(StepRec[a_ind], 1.0)

                if USE_BENCHMARK
                    # benchmark = @benchmarkable $algs[$a_ind]($MatUk, $MatUp; wsp = $wsp[$a_ind],
                    # Stop= terminator($MaxIter[$a_ind], $MaxTime, $AbsTol, 1e-5), 
                    # Solver_Stop=$SOLVER_STOP, Init=$init_guess_simple, NMLS_Set=$NMLS_SET) evals = BENCHMARK_EVALS samples = BENCHMARK_SAMPLES
                    # profile = run(benchmark)
                    # time = minimum(profile.times .- profile.gctimes)
                    # time = time / 1e6

                    time = 10000000
                    for sample = 1:BENCHMARK_SAMPLES
                        stat = @timed MatS, flag, iter, = algs[a_ind](MatUk, MatUp; wsp = wsp[a_ind],
                            Stop= terminator(MaxIter[a_ind], MaxTime, AbsTol, 1e-5), 
                            Solver_Stop=SOLVER_STOP, Init=init_guess_simple, NMLS_Set=NMLS_SET)
                        time = min(time, (stat.time - stat.gctime) * 1e3)
                        flags[a_ind] = flag
                        iters[a_ind] = iter
                    end

                    # MatS, flag, iter, = algs[a_ind](MatUk, MatUp; wsp = wsp[a_ind],
                    # Stop= terminator(MaxIter[a_ind], MaxTime, AbsTol, 1e-5), 
                    # Solver_Stop=SOLVER_STOP, Init=init_guess_simple, NMLS_Set=NMLS_SET)
                elseif USE_ANALYSIS
                    MatS, flag, iter, time = algs_analysis[a_ind](MatUk, MatUp; wsp = wsp[a_ind],
                    Stop= terminator(MaxIter[a_ind], MaxTime, AbsTol, 1e-5), Records = Records[a_ind],
                    Solver_Stop=SOLVER_STOP, Init=init_guess_simple, NMLS_Set=NMLS_SET)
                    flags[a_ind] = flag
                    iters[a_ind] = iter
                else
                    stats = @timed begin
                        MatS, flag, iter = algs[a_ind](MatUk, MatUp; wsp = wsp[a_ind],
                        Stop= terminator(MaxIter[a_ind], MaxTime, AbsTol, 1e-5),
                        Solver_Stop=SOLVER_STOP, Init=init_guess_simple, NMLS_Set=NMLS_SET)
                    end
                    time = (stats.time - stats.gctime) * 1000
                    flags[a_ind] = flag
                    iters[a_ind] = iter
                end

                if flag > 2
                    FailCnt[a_ind] += 1
                    push!(FailRec, record_ind)
                end

                if iter < SHORT_RUN
                    push!(ShortRec, record_ind)
                end

                RecTime[(s_ind-1)*runs+r_ind, a_ind] = time
                RecIter[(s_ind-1)*runs+r_ind, a_ind] = iter


                # if detailed_run_cnt < DETAILED_RUN && (flag > 2 || iter > 50) && a_ind == 2
                if detailed_run_cnt < DETAILED_RUN && a_ind == 2
                    if (RecTime[record_ind, 2] > 1.5 * RecTime[record_ind, 1] || RecIter[record_ind, 1] > 200) && RecIter[record_ind, 1] > SHORT_RUN
                        if !USE_BENCHMARK && USE_ANALYSIS
                            for ind = 1:iter
                                println(@sprintf("Iteration: %i, \tTime: %.2f, \tCost: %.12f", ind, TimeRec[a_ind][ind], AbsERec[a_ind][ind]))
                            end
                        end
                        
                        detailed_run_cnt += 1


                        println("\n========================================================\n")
                        println("Plot ind: $(detailed_run_cnt)\t Flag1: $(flags[1])\t Flag2: $(flags[2])\t Iteration1: $(iters[1])\t Iteration2: $(iters[2])\t Time, Alg1: $(RecTime[record_ind, 1])\t Time, Alg2: $(RecTime[record_ind, 2])")
                        println("\n========================================================\n")


                        global DEBUG = true
                        global MSG = true

                        test_stlog_multi_alg(MatUk, algs_analysis, labels; MaxIter=MaxIter, AbsTol=AbsTol, MaxTime=MaxTime)
                        global DEBUG = false
                        global MSG = false

                        # test_stlog_alg(MatUk, algs_analysis[a_ind], labels[a_ind]; MaxIter=MaxIter, AbsTol=AbsTol, MaxTime=MaxTime)
                    end
                end
            end
        end
    end


    display(FailCnt)

    # display(FailRec)
    # display(RecTime)

    GoodRun = setdiff(eachindex(scale_vec), FailRec)

    GoodLongRun = setdiff(eachindex(scale_vec), union(ShortRec, FailRec))


    scale_vec_good = scale_vec[GoodRun]
    RecTime_good = RecTime[GoodRun, :]
    RecIter_good = RecIter[GoodRun, :]
    Ratio_good = hcat([RecTime[:, 1] ./ c for c in eachcol(RecTime)]...)[GoodRun, 2:end]

    scale_vec_good_long = scale_vec[GoodLongRun]
    RecTime_good_long = RecTime[GoodLongRun, :]
    RecIter_good_long = RecIter[GoodLongRun, :]
    Ratio_good_long = hcat([RecTime[:, 1] ./ c for c in eachcol(RecTime)]...)[GoodLongRun, 2:end]


    iter_plt = scatter(scale_vec_good, RecIter_good,
        label=labels,
        # xlabel="2-norm of the generating velocity, |S_{A,B,0}|_2",
        xlabel="σ",
        ylabel="Iteration",
        # ylims = (1e-2, 8 * median(RecTime)),
        yscale=:log2,
        markershape = markershapes,
        markerstrokewidth=0,
        markerstrokecolor = :auto,
        lw=0,
        ms=1.5,
        ma=0.5
    )

    display(iter_plt)

    time_plt = scatter(scale_vec_good, RecTime_good,
        label=labels,
        # xlabel="2-norm of the generating velocity, |S_{A,B,0}|_2",
        ylabel="Time (ms)",
        xlabel="σ",
        # ylims = (1e-2, 8 * median(RecTime)),
        # yscale=:log2,
        markerstrokewidth=0,
        markershape = markershapes,
        markerstrokecolor = :auto,
        lw=0,
        ms=1.5,
        ma=0.5
    )

    time_logplt = scatter(scale_vec_good, RecTime_good,
        label=labels,
        # xlabel="2-norm of the generating velocity, |S_{A,B,0}|_2",
        ylabel="Time (ms)",
        xlabel="σ",
        # ylims = (1e-2, 8 * median(RecTime)),
        yscale=:log2,
        markerstrokewidth=0,
        markershape = markershapes,
        markerstrokecolor = :auto,
        lw=0,
        ms=1.5,
        ma=0.5
    )

    ratio_plt = plot(scale_grid, ones(length(scale_grid)), label = :none, linestyle = :dash);

    scatter!(scale_vec_good, Ratio_good,
        label=length(markershapes) == 2 ? labels[2] : labels[:, 2:end],
        xlabel="σ",
        ylabel="Speedup to the $(labels[1]) solver",
        # ylims = (0.0, 10),
        yscale=:log2,
        markerstrokewidth=0,
        markershape = length(markershapes) == 2 ? markershapes[2] : markershapes[:, 2:end],
        markerstrokecolor = :auto,
        lw=0,
        ms=1.5,
        ma=0.5
    )



    plt = plot(layout=(2, 1), size = (800, 1000), time_plt, ratio_plt, yscale = :identity, formatter = (x -> @sprintf("%.2f", x)))

    plt_log = plot(layout=(2, 1), size = (800, 1000), time_logplt, ratio_plt, yscale = :identity, formatter = (x -> @sprintf("%.2f", x)))

    display(plt)

    time_long_plt = scatter(scale_vec_good_long, RecTime_good_long,
        label=labels,
        # xlabel="2-norm of the generating velocity, |S_{A,B,0}|_2",
        xlabel="σ",
        ylabel="Time (ms)",
        # ylims = (1e-2, 8 * median(RecTime)),
        # yscale=:log2,
        markerstrokewidth=0,
        markershape = markershapes,
        markerstrokecolor = :auto,
        lw=0,
        ms=1.5,
        ma=0.5
    )

    time_long_logplt = scatter(scale_vec_good_long, RecTime_good_long,
        label=:none,
        # xlabel="2-norm of the generating velocity, |S_{A,B,0}|_2",
        xlabel="σ",
        ylabel="Time (ms)",
        # ylims = (1e-2, 8 * median(RecTime)),
        yscale=:log2,
        markerstrokewidth=0,
        markershape = markershapes,
        markerstrokecolor = :auto,
        lw=0,
        ms=1.5,
        ma=0.5
    )

    ratio_long_plt = plot(scale_grid, ones(length(scale_grid)), label = :none, linestyle = :dash);


    scatter!(scale_vec_good_long, Ratio_good_long,
        label= length(markershapes) == 2 ? labels[2] : labels[:, 2:end],
        xlabel="σ",
        ylabel="Speedup to the $(labels[1]) solver",
        # ylims = (0.0, 10),
        yscale=:log2,
        markerstrokewidth=0,
        markershape = length(markershapes) == 2 ? markershapes[2] : markershapes[:, 2:end],
        markerstrokecolor = :auto,
        lw=0,
        ms=1.5,
        ma=0.5
    )



    long_plt = plot(layout=(2, 1), size = (800, 1000), time_long_logplt, ratio_long_plt, formatter = (x -> @sprintf("%.2f", x)))

    display(long_plt)

    if filename != ""
        pos = findlast('.', filename);
        savefig(iter_plt, filename[1:(pos-1)] * "_iter." * filename[(pos+1):end])
        savefig(time_plt, filename[1:(pos-1)] * "_time." * filename[(pos+1):end])
        savefig(time_logplt, filename[1:(pos-1)] * "_time_logscale." * filename[(pos+1):end])
        savefig(ratio_plt, filename[1:(pos-1)] * "_rate." * filename[(pos+1):end])

        savefig(time_long_plt, filename[1:(pos-1)] * "_time_long." * filename[(pos+1):end])
        savefig(time_long_logplt, filename[1:(pos-1)] * "_time_long_logscale." * filename[(pos+1):end])
        savefig(ratio_long_plt, filename[1:(pos-1)] * "_rate_long." * filename[(pos+1):end])
        writedlm(filename[1:(pos-1)] * "_data.txt", hcat(scale_vec_good, RecIter_good, RecTime_good))
    end
end

function test_BCH_naive_speed(k, scale_grid, runs=100; MaxIter=500, AbsTol=1e-10, MaxTime=10000, seed=9527, scaleby=opnorm, filename="")
    test_alg_speed(k, scale_grid,
        [stlog_BCH1_2k_naive, stlog_BCH3_2k_naive, stlog_BCH1_2k, stlog_BCH3_2k], [stlog_BCH1_2k_analysis, stlog_BCH3_2k_analysis, stlog_BCH1_2k_analysis, stlog_BCH3_2k_analysis],
        ["BCH1" "BCH3" "BCH1-Schur" "BCH3-Schur"], runs; markershapes = [:circle :circle :star5 :star5],
        MaxIter=MaxIter .* ones(Int, 4), AbsTol=AbsTol, MaxTime=MaxTime, seed=seed, scaleby=scaleby, filename=filename)
end

function test_BCH_speed(k, scale_grid, runs=100; MaxIter=500, AbsTol=1e-10, MaxTime=10000, seed=9527, scaleby=opnorm)
    test_alg_speed(k, scale_grid,
        [stlog_BCH1_2k, stlog_BCH3_2k], [stlog_BCH1_2k_analysis, stlog_BCH3_2k_analysis],
        ["BCH1-Schur" "BCH3-Schur"], runs;
        MaxIter=MaxIter, AbsTol=AbsTol, MaxTime=MaxTime, seed=seed, scaleby=scaleby)
end

function test_hybrid_Newton_speed(k, scale_grid, runs=100;
    MaxIter=500, AbsTol=1e-10, MaxTime=10000, seed=9527, scaleby=opnorm, filename="")
    test_alg_speed(k, scale_grid,
        [stlog_BCH3_2k, stlog_hybrid_Newton_armijo],
        [stlog_BCH3_2k_analysis, stlog_hybrid_Newton_armijo_analysis],
        ["BCH3-Schur" "Hybrid-Newton"], runs; markershapes = [:circle :star5],
        MaxIter=[10 * MaxIter, MaxIter], AbsTol=AbsTol, MaxTime=MaxTime,
        seed=seed, scaleby=scaleby, filename=filename)
end

function test_Newton_speed(k, scale_grid, runs=100;
    MaxIter=500, AbsTol=1e-10, MaxTime=10000, seed=9527, scaleby=opnorm, filename="")
    test_alg_speed(k, scale_grid,
        [stlog_BCH3_2k, stlog_Newton_armijo, stlog_hybrid_Newton_armijo],
        [stlog_BCH3_2k_analysis, stlog_Newton_armijo_analysis, stlog_hybrid_Newton_armijo_analysis],
        ["BCH3-Schur" "Newton" "Hybrid-Newton"], runs; markershapes = [:circle :star5 :star5],
        MaxIter=[10 * MaxIter, MaxIter, MaxIter], AbsTol=AbsTol, MaxTime=MaxTime,
        seed=seed, scaleby=scaleby, filename=filename)
end


function test_BCH13_Newton_speed(k, scale_grid, runs=100;
    MaxIter=500, AbsTol=1e-10, MaxTime=10000, seed=9527, scaleby=opnorm, filename="")
    test_alg_speed(k, scale_grid,
        [stlog_BCH1_2k, stlog_BCH3_2k, stlog_hybrid_Newton_armijo],
        [stlog_BCH1_2k_analysis, stlog_BCH3_2k_analysis, stlog_hybrid_Newton_armijo_analysis],
        ["BCH1" "BCH3" "Hybird-Newton"], runs;
        MaxIter=[10 * MaxIter, 10 * MaxIter, MaxIter], AbsTol=AbsTol, MaxTime=MaxTime,
        seed=seed, scaleby=scaleby, filename=filename)
end


function test_stlog_BCH_Newton_single_prob(k::Int, scale; MaxIter=500, AbsTol=1e-10, MaxTime=20000, scaleby=opnorm, seed=9527)
    rand_eng = MersenneTwister(seed)
    X = rand(rand_eng, 2k, 2k)
    X .-= X'
    X[(k+1):2k, (k+1):2k] .= 0.0
    X .*= scale / scaleby(X)

    Q = exp(X)
    Uk = zeros(2k, k)
    Uk .= Q[:, 1:k]

    test_stlog_multi_alg(Uk, [stlog_BCH3_2k_analysis, stlog_hybrid_Newton_armijo_analysis], ["BCH3 solver" "Newton Solver"]; MaxIter=MaxIter, AbsTol=AbsTol, MaxTime=MaxTime)
end


function test_fail_BCH3(k::Int, attemps = 1000; goal = 10, MaxIter = 1000, AbsTol = 1e-12, filename = "")
    n = 2 * k;
    found_failure::Int = 0

    TimeRec = zeros(MaxIter)
    AbsERec = zeros(MaxIter)
    DistRec = zeros(MaxIter)
    VectRec = zeros(MaxIter)
    StepRec = ones(MaxIter)
    AngsRec = zeros(MaxIter, k)
    StPtRec = Vector{Any}(undef, MaxIter)

    Records = [Ref(TimeRec), Ref(AbsERec), Ref(DistRec), Ref(VectRec), Ref(StepRec), Ref(AngsRec), Ref(StPtRec)]

    TimeRec2 = zeros(MaxIter)
    AbsERec2 = zeros(MaxIter)
    DistRec2 = zeros(MaxIter)
    VectRec2 = zeros(MaxIter)
    StepRec2 = ones(MaxIter)
    AngsRec2 = zeros(MaxIter, k)
    StPtRec2 = Vector{Any}(undef, MaxIter)

    Records2 = [Ref(TimeRec2), Ref(AbsERec2), Ref(DistRec2), Ref(VectRec2), Ref(StepRec2), Ref(AngsRec2), Ref(StPtRec2)]

    wsp = get_wps_bch3(n, k, 1000)
    V = zeros(n, k)
    Vp = zeros(n, k)

    for a_ind = 1:attemps
        X = rand(n, n);
        X[(k + 1):n, (k + 1):n] .= 0.0;
        X .-= X';
        X .*= (π - 0.1 + 0.2 * rand()) / opnorm(X);

        Q = exp(X);
        V .= Q[:, 1:k];

        Ans, flag, iter, time, = stlog_BCH3_2k_analysis(V, Vp; Stop = terminator(MaxIter, 100000, AbsTol, 1e-8), Init = init_guess_simple, Records = Records);

        if flag > 2
            Ans2, flag2, iter2, time, = stlog_hybrid_Newton_armijo_analysis(V, Vp; Stop = terminator(div(MaxIter, 10), 100000, AbsTol, 1e-8), Init = init_guess_simple, Records = Records);

            cvi_plot = scatter(AbsERec[1:iter], 
                label = "BCH3 solver",
                ylabel="Objective value",
                xlabel="Iteration",
                yscale=:log2,
                markerstrokewidth=0,
                lw=0,
                ms=1.5,
                ma=0.5
            )
            scatter!(AbsERec2[1:iter2], 
                label = "Newton solver",
                ylabel="Objective value",
                yscale=:log2,
                markerstrokewidth=0,
                lw=0,
                ms=1.5,
                ma=0.5
            )

            display(cvi_plot);

            found_failure += 1;
            if filename != ""
                pos = findlast('.', filename);
                savefig(cvi_plot, filename[1:(pos - 1)] * "_$(found_failure)." * filename[(pos + 1):end])
            end
        end

        if found_failure > goal
            break
        end
    end
end

function test_BCH3_Newton_condition_plot(k::Int, plt_num::Int, scale::Float64, cond, attemps, title = ""; MaxIter = 1000, AbsTol = 1e-14, MaxTime = 20000, seed = 9527)

    n = 2k

    rand_eng = MersenneTwister(seed)

    plt_cnt = 0;
    plts = [];

    alg_len = 2

    TimeRec = [zeros(MaxIter) for ind in 1:alg_len]
    AbsERec = [zeros(MaxIter) for ind in 1:alg_len]
    DistRec = [zeros(MaxIter) for ind in 1:alg_len]
    VectRec = [zeros(MaxIter) for ind in 1:alg_len]
    StepRec = [ones(MaxIter) for ind in 1:alg_len]
    AngsRec = [zeros(MaxIter, k) for ind in 1:alg_len]
    StPtRec = [Vector{Any}(undef, MaxIter) for ind in 1:alg_len]

    Records = [[Ref(TimeRec[ind]), Ref(AbsERec[ind]), Ref(DistRec[ind]), Ref(VectRec[ind]), Ref(StepRec[ind]), Ref(AngsRec[ind]), Ref(StPtRec[ind])] for ind in 1:alg_len]

    wsp = [get_wsp_alg(n, k, MaxIter, stlog_BCH3_2k_analysis), get_wsp_alg(n, k, MaxIter, stlog_Newton_armijo_analysis)]

    iter1::Int = 0
    iter2::Int = 0


    MatU = zeros(2k, 2k)
    MatUk = zeros(2k, k)
    MatUp = zeros(2k, k)


    for ind in 1:attemps
        X = rand(rand_eng, 2k, 2k)
        X .-= X'
        X[(k+1):2k, (k+1):2k] .= 0.0
        X .*= scale / opnorm(X)

        MatU .= exp(X)

        MatUk .= MatU[:, 1:k]

        fill!(TimeRec[1], 0.0)
        fill!(AbsERec[1], 0.0)
        fill!(DistRec[1], 0.0)
        fill!(VectRec[1], 0.0)
        fill!(StepRec[1], 1.0)

        fill!(TimeRec[2], 0.0)
        fill!(AbsERec[2], 0.0)
        fill!(DistRec[2], 0.0)
        fill!(VectRec[2], 0.0)
        fill!(StepRec[2], 1.0)

        MatS, flag, iter1 = stlog_BCH3_2k_analysis(MatUk, MatUp; wsp = wsp[1],
            Stop= terminator(MaxIter, MaxTime, AbsTol, 1e-5), Records = Records[1],
            Solver_Stop=SOLVER_STOP, Init=init_guess_simple, NMLS_Set=NMLS_SET)
                        
        MatS, flag, iter2 = stlog_Newton_armijo_analysis(MatUk, MatUp; wsp = wsp[2],
            Stop= terminator(MaxIter, MaxTime, AbsTol, 1e-5), Records = Records[2],
            Solver_Stop=SOLVER_STOP, Init=init_guess_simple, NMLS_Set=NMLS_SET)

        if cond(iter1, iter2)
            plt = plot(1:iter1, AbsERec[1][1:iter1], yscale = :log10, label = "BCH3-Schur", xlabel = "Iteration", ylabel = "Objective value", title = title * "σ = $(scale)")
            plot!(1:iter2, AbsERec[2][1:iter2], yscale = :log10, label = "Newton", xlabel = "Iteration", ylabel = "Objective value")
            display(plt)
            push!(plts, plt);
            plt_cnt += 1;
            if plt_cnt >= plt_num
                break;
            end
        end
    end

    return plts
end

function test_BCH3_Newton_iteration_plot(k::Int, scales, title = ""; MaxIter = 1000, AbsTol = 1e-14, MaxTime = 20000, seed = 9527)

    n = 2k

    rand_eng = MersenneTwister(seed)

    plts = [];

    alg_len = 2

    TimeRec = [zeros(MaxIter) for ind in 1:alg_len]
    AbsERec = [zeros(MaxIter) for ind in 1:alg_len]
    DistRec = [zeros(MaxIter) for ind in 1:alg_len]
    VectRec = [zeros(MaxIter) for ind in 1:alg_len]
    StepRec = [ones(MaxIter) for ind in 1:alg_len]
    AngsRec = [zeros(MaxIter, k) for ind in 1:alg_len]
    StPtRec = [Vector{Any}(undef, MaxIter) for ind in 1:alg_len]

    Records = [[Ref(TimeRec[ind]), Ref(AbsERec[ind]), Ref(DistRec[ind]), Ref(VectRec[ind]), Ref(StepRec[ind]), Ref(AngsRec[ind]), Ref(StPtRec[ind])] for ind in 1:alg_len]

    wsp = [get_wsp_alg(n, k, MaxIter, stlog_BCH3_2k_analysis), get_wsp_alg(n, k, MaxIter, stlog_Newton_armijo_analysis)]

    iter1::Int = 0
    iter2::Int = 0


    MatU = zeros(2k, 2k)
    MatUk = zeros(2k, k)
    MatUp = zeros(2k, k)

    X = rand(rand_eng, 2k, 2k)
    X .-= X'
    X[(k+1):2k, (k+1):2k] .= 0.0
    for s_ind in eachindex(scales)
        scale = scales[s_ind]
        X .*= scale / opnorm(X)

        MatU .= exp(X)

        MatUk .= MatU[:, 1:k]

        fill!(TimeRec[1], 0.0)
        fill!(AbsERec[1], 0.0)
        fill!(DistRec[1], 0.0)
        fill!(VectRec[1], 0.0)
        fill!(StepRec[1], 1.0)

        fill!(TimeRec[2], 0.0)
        fill!(AbsERec[2], 0.0)
        fill!(DistRec[2], 0.0)
        fill!(VectRec[2], 0.0)
        fill!(StepRec[2], 1.0)

        MatS, flag, iter1 = stlog_BCH3_2k_analysis(MatUk, MatUp; wsp = wsp[1],
            Stop= terminator(MaxIter, MaxTime, AbsTol, 1e-5), Records = Records[1],
            Solver_Stop=SOLVER_STOP, Init=init_guess_simple, NMLS_Set=NMLS_SET)
                        
        MatS, flag, iter2 = stlog_Newton_armijo_analysis(MatUk, MatUp; wsp = wsp[2],
            Stop= terminator(MaxIter, MaxTime, AbsTol, 1e-5), Records = Records[2],
            Solver_Stop=SOLVER_STOP, Init=init_guess_simple, NMLS_Set=NMLS_SET)

        plt = plot(1:iter1, AbsERec[1][1:iter1], yscale = :log10, label = "BCH3-Schur", xlabel = "Iteration", ylabel = "Objective value", title = title * "σ = $(scale)")
        plot!(1:iter2, AbsERec[2][1:iter2], yscale = :log10, label = "Newton", xlabel = "Iteration", ylabel = "Objective value")
        display(plt)
        push!(plts, plt);
    end

    return plts
end



function test_preprocessing(n, k, d, scale_grid, alg, alg_label, runs=100;
    MaxIter=100, AbsTol=1e-14, MaxTime=50000, seed=9527, filename="")
    rand_eng = MersenneTwister(seed)

    MatU = zeros(n, n)
    MatUk = zeros(n, k)
    MatUp = zeros(n, n - k)
    MatVk_2k = zeros(2k, k)
    MatVp_2k = zeros(2k, k)
    MatVk_rk = zeros(2k - d, k)
    MatVp_rk = zeros(2k - d, k - d)

    Uk = Ref(MatUk)
    Up = Ref(MatUp)
    Vk_2k = Ref(MatVk_2k)
    Vp_2k = Ref(MatVp_2k)
    Vk_rk = Ref(MatVk_rk)
    Vp_rk = Ref(MatVp_rk)

    MaxIterList = MaxIter .* [1, 1, 1]
    labels = [alg_label alg_label*"-2k" alg_label*"-rank"]
    markershapes = [:circle :circle :circle]


    flags = zeros(Int, length(MaxIterList))
    iters = zeros(Int, length(MaxIterList))


    wsp1 = get_wsp_alg(n, k, MaxIterList[1], alg);
    wsp2 = get_wsp_alg(2k, k, MaxIterList[2], alg);
    wsp3 = get_wsp_alg(2k - d, k, MaxIterList[3], alg);




    TimeRec = [zeros(MaxIterList[ind]) for ind in eachindex(MaxIterList)]
    AbsERec = [zeros(MaxIterList[ind]) for ind in eachindex(MaxIterList)]
    DistRec = [zeros(MaxIterList[ind]) for ind in eachindex(MaxIterList)]
    VectRec = [zeros(MaxIterList[ind]) for ind in eachindex(MaxIterList)]
    StepRec = [ones(MaxIterList[ind]) for ind in eachindex(MaxIterList)]
    AngsRec = [zeros(MaxIterList[ind], k) for ind in eachindex(MaxIterList)]
    StPtRec = [Vector{Any}(undef, MaxIterList[ind]) for ind in eachindex(MaxIterList)]

    Records = [[Ref(TimeRec[ind]), Ref(AbsERec[ind]), Ref(DistRec[ind]), Ref(VectRec[ind]), Ref(StepRec[ind]), Ref(AngsRec[ind]), Ref(StPtRec[ind])] for ind in eachindex(MaxIterList)]

    # wsp = [get_wsp_alg(n, k, MaxIter[a_ind], algs[a_ind]) for a_ind in eachindex(algs)]

    RecTime = zeros(length(scale_grid) * runs, length(MaxIterList))
    RecIter = zeros(Int, length(scale_grid) * runs, length(MaxIterList))


    scale_vec = vcat(ones(runs) * scale_grid'...)

    FailCnt = zeros(length(MaxIterList))

    FailRec = []
    ShortRec = []

    detailed_run_cnt = 0

    flag::Int = -1;
    iter::Int = -1;
    time1::Float64 = 0.0;
    time2::Float64 = 0.0;


    for s_ind = eachindex(scale_grid)
        for r_ind = 1:runs
            scale = scale_grid[s_ind]
            X = rand(rand_eng, n, n)
            X .-= X'
            X[(k+1):n, (k+1):n] .= 0.0
            X[(d+1):n, 1:d] .= 0.0
            X[1:d, (d+1):n] .= 0.0
            X .*= scale / opnorm(X)

            MatU .= exp(X)

            MatUk .= MatU[:, 1:k]

            for a_ind in eachindex(MaxIterList)
                record_ind = (s_ind-1)*runs+r_ind

                fill!(TimeRec[a_ind], 0.0)
                fill!(AbsERec[a_ind], 0.0)
                fill!(DistRec[a_ind], 0.0)
                fill!(VectRec[a_ind], 0.0)
                fill!(StepRec[a_ind], 1.0)

                if a_ind == 1
                    time1 = 10000000.
                    time2 = 10000000.
                    for sample = 1:BENCHMARK_SAMPLES
                        time1 = 0.
                        stat = @timed MatS, flag, iter, = alg(MatUk, MatUp; wsp = wsp1,
                            Stop= terminator(MaxIterList[a_ind], MaxTime, AbsTol, 1e-5), 
                            Solver_Stop=SOLVER_STOP, Init=init_guess_simple, NMLS_Set=NMLS_SET)
                        time2 = min(time2, (stat.time - stat.gctime) * 1e3)
                        flags[a_ind] = flag
                        iters[a_ind] = iter
                    end
                elseif a_ind == 2
                    time1 = 10000000.
                    time2 = 10000000.
                    for sample = 1:BENCHMARK_SAMPLES
                        stat = @timed preprocessing_Ink_with_rank(Vk_2k, Uk);
                        time1 = min(time1, (stat.time - stat.gctime) * 1e3)

                        stat = @timed MatS, flag, iter, = alg(MatVk_2k, MatVp_2k; wsp = wsp2,
                            Stop= terminator(MaxIterList[a_ind], MaxTime, AbsTol, 1e-5), 
                            Solver_Stop=SOLVER_STOP, Init=init_guess_simple, NMLS_Set=NMLS_SET)
                        time2 = min(time2, (stat.time - stat.gctime) * 1e3)
                        flags[a_ind] = flag
                        iters[a_ind] = iter
                    end
                elseif a_ind == 3
                    time1 = 10000000.
                    time2 = 10000000.
                    for sample = 1:BENCHMARK_SAMPLES
                        stat = @timed  preprocessing_Ink_with_rank(Vk_rk, Uk);
                        time1 = min(time1, (stat.time - stat.gctime) * 1e3)

                        stat = @timed MatS, flag, iter, = alg(MatVk_rk, MatVp_rk; wsp = wsp3,
                            Stop= terminator(MaxIterList[a_ind], MaxTime, AbsTol, 1e-5), 
                            Solver_Stop=SOLVER_STOP, Init=init_guess_simple, NMLS_Set=NMLS_SET)
                        time2 = min(time2, (stat.time - stat.gctime) * 1e3)
                        flags[a_ind] = flag
                        iters[a_ind] = iter
                    end
                end

                if flag > 2
                    FailCnt[a_ind] += 1
                    push!(FailRec, record_ind)
                end

                if iter < SHORT_RUN
                    push!(ShortRec, record_ind)
                end

                RecTime[(s_ind-1)*runs+r_ind, a_ind] = time1 + time2
                RecIter[(s_ind-1)*runs+r_ind, a_ind] = iter


                # if detailed_run_cnt < DETAILED_RUN && (flag > 2 || iter > 50) && a_ind == 2
                # if detailed_run_cnt < DETAILED_RUN && a_ind == 2
                #     if (RecTime[record_ind, 2] > 1.5 * RecTime[record_ind, 1] || RecIter[record_ind, 1] > 200) && RecIter[record_ind, 1] > SHORT_RUN
                #         if !USE_BENCHMARK && USE_ANALYSIS
                #             for ind = 1:iter
                #                 println(@sprintf("Iteration: %i, \tTime: %.2f, \tCost: %.12f", ind, TimeRec[a_ind][ind], AbsERec[a_ind][ind]))
                #             end
                #         end
                        
                #         detailed_run_cnt += 1


                #         println("\n========================================================\n")
                #         println("Plot ind: $(detailed_run_cnt)\t Flag1: $(flags[1])\t Flag2: $(flags[2])\t Iteration1: $(iters[1])\t Iteration2: $(iters[2])\t Time, Alg1: $(RecTime[record_ind, 1])\t Time, Alg2: $(RecTime[record_ind, 2])")
                #         println("\n========================================================\n")


                #         global DEBUG = true
                #         global MSG = true

                #         test_stlog_multi_alg(MatUk, algs_analysis, labels; MaxIter=MaxIter, AbsTol=AbsTol, MaxTime=MaxTime)
                #         global DEBUG = false
                #         global MSG = false

                #         # test_stlog_alg(MatUk, algs_analysis[a_ind], labels[a_ind]; MaxIter=MaxIter, AbsTol=AbsTol, MaxTime=MaxTime)
                #     end
                # end
            end
        end
    end


    display(FailCnt)

    # display(FailRec)
    # display(RecTime)

    GoodRun = setdiff(eachindex(scale_vec), FailRec)

    GoodLongRun = setdiff(eachindex(scale_vec), union(ShortRec, FailRec))


    scale_vec_good = scale_vec[GoodRun]
    RecTime_good = RecTime[GoodRun, :]
    RecIter_good = RecIter[GoodRun, :]
    Ratio_good = hcat([RecTime[:, 1] ./ c for c in eachcol(RecTime)]...)[GoodRun, 2:end]

    scale_vec_good_long = scale_vec[GoodLongRun]
    RecTime_good_long = RecTime[GoodLongRun, :]
    RecIter_good_long = RecIter[GoodLongRun, :]
    Ratio_good_long = hcat([RecTime[:, 1] ./ c for c in eachcol(RecTime)]...)[GoodLongRun, 2:end]


    iter_plt = scatter(scale_vec_good, RecIter_good,
        label=labels,
        # xlabel="2-norm of the generating velocity, |S_{A,B,0}|_2",
        xlabel="σ",
        ylabel="Iteration",
        # ylims = (1e-2, 8 * median(RecTime)),
        yscale=:log2,
        markershape = markershapes,
        markerstrokewidth=0,
        markerstrokecolor = :auto,
        lw=0,
        ms=1.5,
        ma=0.5
    )

    display(iter_plt)

    time_plt = scatter(scale_vec_good, RecTime_good,
        label=labels,
        # xlabel="2-norm of the generating velocity, |S_{A,B,0}|_2",
        ylabel="Time (ms)",
        xlabel="σ",
        # ylims = (1e-2, 8 * median(RecTime)),
        # yscale=:log2,
        markerstrokewidth=0,
        markershape = markershapes,
        markerstrokecolor = :auto,
        lw=0,
        ms=1.5,
        ma=0.5
    )

    time_logplt = scatter(scale_vec_good, RecTime_good,
        label=labels,
        # xlabel="2-norm of the generating velocity, |S_{A,B,0}|_2",
        ylabel="Time (ms)",
        xlabel="σ",
        # ylims = (1e-2, 8 * median(RecTime)),
        yscale=:log2,
        markerstrokewidth=0,
        markershape = markershapes,
        markerstrokecolor = :auto,
        lw=0,
        ms=1.5,
        ma=0.5
    )

    ratio_plt = plot(scale_grid, ones(length(scale_grid)), label = :none, linestyle = :dash);

    scatter!(scale_vec_good, Ratio_good,
        label=length(markershapes) == 2 ? labels[2] : labels[:, 2:end],
        xlabel="σ",
        ylabel="Speedup to the $(labels[1]) solver",
        # ylims = (0.0, 10),
        yscale=:log2,
        markerstrokewidth=0,
        markershape = length(markershapes) == 2 ? markershapes[2] : markershapes[:, 2:end],
        markerstrokecolor = :auto,
        lw=0,
        ms=1.5,
        ma=0.5
    )



    plt = plot(layout=(2, 1), size = (800, 1000), time_plt, ratio_plt, yscale = :identity, formatter = (x -> @sprintf("%.2f", x)))

    plt_log = plot(layout=(2, 1), size = (800, 1000), time_logplt, ratio_plt, yscale = :identity, formatter = (x -> @sprintf("%.2f", x)))

    display(plt)

    time_long_plt = scatter(scale_vec_good_long, RecTime_good_long,
        label=labels,
        # xlabel="2-norm of the generating velocity, |S_{A,B,0}|_2",
        xlabel="σ",
        ylabel="Time (ms)",
        # ylims = (1e-2, 8 * median(RecTime)),
        # yscale=:log2,
        markerstrokewidth=0,
        markershape = markershapes,
        markerstrokecolor = :auto,
        lw=0,
        ms=1.5,
        ma=0.5
    )

    time_long_logplt = scatter(scale_vec_good_long, RecTime_good_long,
        label=:none,
        # xlabel="2-norm of the generating velocity, |S_{A,B,0}|_2",
        xlabel="σ",
        ylabel="Time (ms)",
        # ylims = (1e-2, 8 * median(RecTime)),
        yscale=:log2,
        markerstrokewidth=0,
        markershape = markershapes,
        markerstrokecolor = :auto,
        lw=0,
        ms=1.5,
        ma=0.5
    )

    ratio_long_plt = plot(scale_grid, ones(length(scale_grid)), label = :none, linestyle = :dash);


    scatter!(scale_vec_good_long, Ratio_good_long,
        label= length(markershapes) == 2 ? labels[2] : labels[:, 2:end],
        xlabel="σ",
        ylabel="Speedup to the $(labels[1]) solver",
        # ylims = (0.0, 10),
        yscale=:log2,
        markerstrokewidth=0,
        markershape = length(markershapes) == 2 ? markershapes[2] : markershapes[:, 2:end],
        markerstrokecolor = :auto,
        lw=0,
        ms=1.5,
        ma=0.5
    )



    long_plt = plot(layout=(2, 1), size = (800, 1000), time_long_logplt, ratio_long_plt, formatter = (x -> @sprintf("%.2f", x)))

    display(long_plt)

    if filename != ""
        pos = findlast('.', filename);
        savefig(iter_plt, filename[1:(pos-1)] * "_iter." * filename[(pos+1):end])
        savefig(time_plt, filename[1:(pos-1)] * "_time." * filename[(pos+1):end])
        savefig(time_logplt, filename[1:(pos-1)] * "_time_logscale." * filename[(pos+1):end])
        savefig(ratio_plt, filename[1:(pos-1)] * "_rate." * filename[(pos+1):end])

        savefig(time_long_plt, filename[1:(pos-1)] * "_time_long." * filename[(pos+1):end])
        savefig(time_long_logplt, filename[1:(pos-1)] * "_time_long_logscale." * filename[(pos+1):end])
        savefig(ratio_long_plt, filename[1:(pos-1)] * "_rate_long." * filename[(pos+1):end])
        writedlm(filename[1:(pos-1)] * "_data.txt", hcat(scale_vec_good, RecIter_good, RecTime_good))
    end
end


function test_preprocessing_rank(n, k, d, s)

    zout = x -> abs(x) > 1e-7 ? x : 0.0;

    MatX = rand(n, n);
    MatX .-= MatX';
    MatX[(k + 1):n, (k + 1):n] .= 0.0;
    MatX[(d + 1):n, 1:d] .= 0.0;
    MatX[1:d, (d + 1):n] .= 0.0;
    MatX .*= s / opnorm(MatX);

    MatP = Matrix{Float64}(I, n, n);
    MatQ = MatP * exp(MatX);

    MatUk = copy(view(MatQ, :, 1:k));
    MatUp = zeros(n, n - k);


    MatVk_2k = zeros(2k, k)
    MatVp_2k = zeros(2k, k)
    MatVk_rk = zeros(2k - d, k)
    MatVp_rk = zeros(2k - d, k - d)

    Uk = Ref(MatUk)
    Vk_2k = Ref(MatVk_2k)
    Vp_2k = Ref(MatVp_2k)
    Vk_rk = Ref(MatVk_rk)
    Vp_rk = Ref(MatVp_rk)

    preprocessing_Ink_with_rank(Vk_2k, Uk)

    preprocessing_Ink_with_rank(Vk_rk, Uk)

    MatS, flag, iter, = stlog_BCH3_2k(MatUk, MatUp; 
        wsp = get_wsp_alg(n, k, 1000, stlog_BCH3_2k),
        Stop= terminator(1000, 200000, 1e-14, 1e-5), 
        Solver_Stop=terminator(500, 500, 1e-8, 1e-6), Init=init_guess_simple, 
        NMLS_Set=NMLS_Paras(0.1, 20.0, 0.9, 0.3, 0))

    println("BCH3:")
    display(zout.(MatS))
    display(flag)
    display(iter)

    MatS, flag, iter, = stlog_hybrid_Newton_armijo(MatUk, MatUp; 
        wsp = get_wsp_alg(n, k, 1000, stlog_hybrid_Newton_armijo),
        Stop= terminator(100, 200000, 1e-14, 1e-5), 
        Solver_Stop=terminator(500, 500, 1e-8, 1e-6), Init=init_guess_simple, 
        NMLS_Set=NMLS_Paras(0.1, 20.0, 0.9, 0.3, 0))

    println("Hybrid-Newton:")
    display(zout.(MatS))
    display(flag)
    display(iter)

    MatS, flag, iter, = stlog_BCH3_2k(MatVk_2k, MatVp_2k; 
        wsp = get_wsp_alg(2k, k, 1000, stlog_BCH3_2k),
        Stop= terminator(1000, 200000, 1e-14, 1e-5), 
        Solver_Stop=terminator(500, 500, 1e-8, 1e-6), Init=init_guess_simple, 
        NMLS_Set=NMLS_Paras(0.1, 20.0, 0.9, 0.3, 0))

    println("BCH3-2k:")
    display(zout.(MatS))
    display(flag)
    display(iter)

    MatS, flag, iter, = stlog_hybrid_Newton_armijo(MatVk_2k, MatVp_2k; 
        wsp = get_wsp_alg(2k, k, 1000, stlog_hybrid_Newton_armijo),
        Stop= terminator(100, 200000, 1e-14, 1e-5), 
        Solver_Stop=terminator(500, 500, 1e-8, 1e-6), Init=init_guess_simple, 
        NMLS_Set=NMLS_Paras(0.1, 20.0, 0.9, 0.3, 0))

    println("Hybrid-Newton-2k:")
    display(zout.(MatS))
    display(flag)
    display(iter)

    MatS, flag, iter, = stlog_BCH3_2k(MatVk_rk, MatVp_rk; wsp = get_wsp_alg(2k - d, k, 1000, stlog_BCH3_2k),
        Stop= terminator(1000, 200000, 1e-14, 1e-5), 
        Solver_Stop=terminator(500, 5000, 1e-8, 1e-6), Init=init_guess_simple, 
        NMLS_Set=NMLS_Paras(0.1, 20.0, 0.9, 0.3, 0))

    println("BCH3-rank:")
    display(zout.(MatS))
    display(flag)
    display(iter)

    MatS, flag, iter, = stlog_hybrid_Newton_armijo(MatVk_rk, MatVp_rk; 
        wsp = get_wsp_alg(2k - d, k, 1000, stlog_hybrid_Newton_armijo),
        Stop= terminator(100, 200000, 1e-14, 1e-5), 
        Solver_Stop=terminator(500, 500, 1e-8, 1e-6), Init=init_guess_simple, 
        NMLS_Set=NMLS_Paras(0.1, 20.0, 0.9, 0.3, 0))

    println("Hybrid-Newton-rank:")
    display(zout.(MatS))
    display(flag)
    display(iter)

    # temp_msg = MSG;
    # temp_dbg = DEBUG;

    # global MSG = true
    # global DEBUG = true
    # test_stlog_alg(MatVk_2k, stlog_Newton_armijo_analysis, "Newton-2k"; MaxIter = 100, AbsTol = 1e-14, MaxTime = 200000)
    # global MSG = temp_msg
    # global DEBUG = temp_dbg

end

# ENABLE_RESTART_BCH = true
# ENABLE_NEARLOG = false
# USE_BENCHMARK = true
# test_Newton_speed(10, range(0.9π, 1.2π, 100), 50; MaxIter = 100, AbsTol = 1e-14, seed = 1234, filename = "figures/newton_hard_range_k10.pdf")
# test_Newton_speed(10, range(0.2π, 1.2π, 100), 50; MaxIter = 100, AbsTol = 1e-14, seed = 1234, filename = "figures/newton_all_range_k10.pdf")


# ENABLE_RESTART_BCH = true
# ENABLE_NEARLOG = true
# USE_BENCHMARK = true
# test_hybrid_Newton_speed(10, range(0.9π, 1.2π, 100), 50; MaxIter = 100, AbsTol = 1e-14, seed = 1234, filename = "figures/newton_nearlog_k10.pdf")