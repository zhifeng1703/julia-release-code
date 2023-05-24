include("../../inc/global_path.jl")
include(joinpath(JULIA_INCLUDE_PATH, "workspace.jl"))

include("so_safactor.jl")



function exp_SkewSymm!(M::Ref{Matrix{Float64}}, saf::SAFactor, S::Ref{Matrix{Float64}}, scale::Float64, wsp_saf = get_wsp_saf(size(M[], 1)); order::Bool = true, regular::Bool = false)
    schurAngular_SkewSymm!(saf, S, wsp_saf; order = order, regular = regular);
    computeSpecOrth!(M, saf, scale);
end

function exp_SkewSymm!(M::Ref{Matrix{Float64}}, saf::SAFactor, S::Ref{Matrix{Float64}}, wsp_saf = get_wsp_saf(size(M[], 1)); order::Bool = true, regular::Bool = false)
    schurAngular_SkewSymm!(saf, S, wsp_saf; order = order, regular = regular);
    computeSpecOrth!(M, saf);
end

function log_SpecOrth!(M::Ref{Matrix{Float64}}, saf::SAFactor, Q::Ref{Matrix{Float64}}, scale::Float64, wsp_saf = get_wsp_saf(size(M[], 1)); order::Bool = true, regular::Bool = false)
    schurAngular_SpecOrth!(saf, Q, wsp_saf; order = order, regular = regular);
    computeSkewSymm!(M, saf, scale);
end

function log_SpecOrth!(M::Ref{Matrix{Float64}}, saf::SAFactor, Q::Ref{Matrix{Float64}}, wsp_saf = get_wsp_saf(size(M[], 1)); order::Bool = true, regular::Bool = false)
    schurAngular_SpecOrth!(saf, Q, wsp_saf; order = order, regular = regular);
    computeSkewSymm!(M, saf);
end

function nearlog_SpecOrth!(M::Ref{Matrix{Float64}}, saf::SAFactor, Q::Ref{Matrix{Float64}}, S::Ref{Matrix{Float64}}, wsp_saf = get_wsp_saf(size(Q[], 1)); order::Bool = true, regular::Bool = false)
    MatM = M[];
    MatS = S[];

    schurAngular_SpecOrth!(saf, Q, wsp_saf; order = true, regular = true);

    MatTmp = wsp_saf[1];
    Tmp = wsp_saf(1)

    unsafe_copyto!(pointer(MatTmp), pointer(MatS), length(MatS));


    computeSkewSymm!(M, saf);


    VecA = getAngle(saf);
    axpy!(-1, MatTmp, MatM); # M = M - S
    ind::Int = 0;
    temp::Float64 = 0.0;
    while opnorm(MatM) > π
        ind += 1;
        if ind > saf.nza_cnt
            # display(VecA')
            # computeSkewSymm!(M, saf);
            # axpy!(-1, MatTmp, MatM); # M = M - S
            # display(MatM)
            # display(opnorm(MatM))

            # for jj = 1:saf.nza_cnt
            #     temp = VecA[jj];
            #     VecA[jj] -= 2π;
            #     display(VecA')
            #     computeSkewSymm!(M, saf);
            #     axpy!(-1, MatTmp, MatM); # M = M - S
            #     display(MatM)
            #     display(opnorm(MatM))
            #     VecA[jj] = temp
            # end

            # throw(1);
            break;
        end
        temp = VecA[ind];
        VecA[ind] -= 2π;
        computeSkewSymm!(M, saf);
        axpy!(-1, MatTmp, MatM); # M = M - S
        if opnorm(MatM) < π
            break;
        else
            VecA[ind] = temp
        end
    end

    axpy!(1, MatTmp, MatM); # recover M

    if order
        SAFactor_order(saf, wsp_saf);
    end

    if regular
        SAFactor_regularize(saf, wsp_saf);
    end

end

function exp_SkewSymm(S::Ref{Matrix{Float64}}, scale::Float64, wsp_saf = get_wsp_saf(size(S[], 1)))
    MatW = copy(S[]);
    MatM = similar(MatW);
    W = Ref(MatW);
    M = Ref(MatM);
    saf = SAFactor(size(MatM, 1))
    exp_SkewSymm!(M, saf, W, scale, wsp_saf; order = false, regular = false);
    return MatM;
end

function exp_SkewSymm(S::Ref{Matrix{Float64}}, wsp_saf = get_wsp_saf(size(S[], 1)))
    MatW = copy(S[]);
    MatM = similar(MatW);
    W = Ref(MatW);
    M = Ref(MatM);
    saf = SAFactor(size(MatM, 1))
    exp_SkewSymm!(M, saf, W, wsp_saf; order = false, regular = false);
    return MatM;
end

function log_SpecOrth(Q::Ref{Matrix{Float64}}, scale::Float64, wsp_saf = get_wsp_saf(size(Q[], 1)))
    MatW = copy(Q[]);
    MatM = similar(MatW);
    W = Ref(MatW);
    M = Ref(MatM);
    saf = SAFactor(size(MatM, 1))
    log_SpecOrth!(M, saf, W, scale, wsp_saf; order = false, regular = false);
    return MatM;
end

function log_SpecOrth(Q::Ref{Matrix{Float64}}, wsp_saf = get_wsp_saf(size(Q[], 1)))
    MatW = copy(Q[]);
    MatM = similar(MatW);
    W = Ref(MatW);
    M = Ref(MatM);
    saf = SAFactor(size(MatM, 1))
    log_SpecOrth!(M, saf, W, wsp_saf; order = false, regular = false);
    return MatM;
end

function nearlog_SpecOrth(Q::Ref{Matrix{Float64}}, S::Ref{Matrix{Float64}}, wsp_saf = get_wsp_saf(size(Q[], 1)))
    MatW = copy(Q[]);
    MatM = similar(MatW);
    W = Ref(MatW);
    M = Ref(MatM);
    saf = SAFactor(size(MatM, 1))
    nearlog_SpecOrth!(M, saf, W, S, wsp_saf);
    return MatM
end

#######################################Test functions#######################################

using Plots, Random

function test_so_explog(n = 10)
    X = rand(n, n);
    X .-= X';
    X .*= 4π;

    Q = exp(X);

    logQ = log_SpecOrth(Ref(Q));
    explogQ = exp_SkewSymm(Ref(logQ));

    println(Q ≈ explogQ);
end

function test_so_nearlog(n = 10)
    X = rand(n, n);
    X .-= X';
    X .*= (2π - 0.1) / opnorm(X);

    Q = exp(X);
    S = log_SpecOrth(Ref(Q));

    S_saf = schurAngular_SkewSymm(Ref(S); order = true, regular = true)
    S_ang = getAngle(S_saf);
    bound = min(π, (2π - S_ang[1] - S_ang[2]) / 2);

    Δ = rand(n, n);
    Δ .-= Δ';
    Δ .*= bound * rand() / opnorm(Δ);
    Ω = S + Δ;
    eΩ = exp(Ω);
    leΩ = log_SpecOrth(Ref(eΩ))
    nleΩ = try
        nearlog_SpecOrth(Ref(eΩ), Ref(S))
    catch err
        Ω_saf = schurAngular_SkewSymm(Ref(Ω); order = true, regular = true)
        display(S_saf)
        display(bound)
        display(opnorm(Δ))
        display(Ω_saf)
        throw(err)
    end

    # println(bound)
    if opnorm(Ω) > π + 0.01
        println("*");
    end
    # println(exp(nleΩ) ≈ eΩ)
    # println(exp(leΩ) ≈ exp(nleΩ))
    # println(leΩ ≈ nleΩ)
    # println(Ω ≈ nleΩ)

    return Ω ≈ nleΩ
end



function test_explog_speed_vesus_dim(dim_grid, runs = 10; filename = "", seed = 9527)

    RecTime = zeros(runs * length(dim_grid), 5)

    rand_eng = MersenneTwister(seed)
    record_ind::Int = 1
    for dim_ind in eachindex(dim_grid)
        n = dim_grid[dim_ind]
        Q = zeros(n, n)
        S = zeros(n, n)
        expS = similar(Q)
        logQ = similar(Q)

        Q_saf = SAFactor(n)
        S_saf = SAFactor(n)

        wsp_saf = get_wsp_saf(n)

        for run_ind = 1:runs
            S .= rand(rand_eng, n, n);
            S .-= S';

            S .*= 2π

            stat = @timed Q .= exp(S)
            RecTime[record_ind, 1] = 1000 * (stat.time - stat.gctime)

            stat = @timed exp_SkewSymm!(Ref(Q), Q_saf, Ref(S), wsp_saf; regular = false, order = false)
            RecTime[record_ind, 2] = 1000 * (stat.time - stat.gctime)

            schurAngular_SkewSymm!(Q_saf, Ref(S), wsp_saf; regular = false, order = false)
            stat = @timed computeSpecOrth!(Ref(Q), Q_saf)
            RecTime[record_ind, 3] = 1000 * (stat.time - stat.gctime)

            stat = @timed S .= real.(log(Q));
            RecTime[record_ind, 4] = 1000 * (stat.time - stat.gctime)

            stat = @timed log_SpecOrth!(Ref(S), S_saf, Ref(Q), wsp_saf; regular = false, order = false)
            RecTime[record_ind, 5] = 1000 * (stat.time - stat.gctime)

            record_ind += 1;
        end
    end

    dim_vec = vcat(ones(runs) * dim_grid'...)
    
    time_plt = scatter(dim_vec, RecTime,
        label=["General matrix exponential" "SkewSymm matrix exponential" "SkewSymm matrix exponential with factorization" "General matrix logarithm" "SpecOrth matrix logarithm"],
        xlabel="dimension, n",
        ylabel="Compute time (ms)",
        markershape = [:circle :circle :circle :star5 :star5],
        # yscale=:log2,
        markerstrokewidth=0,
        lw=0,
        ms=1.5,
        markeralpha=0.4
    )

    display(time_plt)
    
    # plt = plot(
    #         layout=(2, 1),
    #         # scatter(scale_vec, RecTime,
    #         #     label=labels,
    #         #     # xlabel="2-norm of the generating velocity, |S_{A,B,0}|_2",
    #         #     ylabel="Compute time (ms)",
    #         #     markerstrokewidth=0,
    #         #     lw=0,
    #         #     ms=1.5,
    #         #     ma=0.1
    #         # ),
            
    #         scatter(dim_vec, DerTime./DerTime[:, 1],
    #             label=:none,
    #             xlabel="2-norm of the generating velocity, |S_{A,B,0}|_2",
    #             ylabel="Ratio to the first solver",
    #             yscale=:log2,
    #             markerstrokewidth=0,
    #             lw=0,
    #             ms=1.5,
    #             ma=0.1
    #         )
    #     )
    # display(plt)

    if filename != ""
        savefig(time_plt, filename)
    end
end


function test_matmul_order(dim_grid, runs = 10)

    RecTime = zeros(length(dim_grid))
    for dim_ind in eachindex(dim_grid)
        n = dim_grid[dim_ind]

        for run_ind = 1:runs
            Mat1 = rand(n, n)
            Mat2 = rand(n, n)
            Mat3 = rand(n, n)
            stat = @timed mul!(Mat1, Mat2, Mat3)
            RecTime[dim_ind] += 1000 * (stat.time - stat.gctime)
        end

        RecTime[dim_ind] /= runs;
    end
    
    plt = plot(dim_grid, RecTime,
        xlabel="dimension, n",
        ylabel="Compute time (ms)",
        xscale=:log10,
        yscale=:log10
    )

    display(plt)

    plt2 = plot(log.(dim_grid), log.(RecTime),
        xlabel="dimension, n",
        ylabel="Compute time (ms)",

    )

    display(plt2)


    println("Estimate order = \t",  dot(log.(dim_grid), log.(dim_grid)) \ dot(log.(dim_grid) , log.(RecTime)))
    
end