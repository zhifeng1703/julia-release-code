using DelimitedFiles, Printf

function stlog_read_file(fname)
    raw_data = readdlm(fname);
    # display(raw_data);

    pts_num = size(raw_data, 1);
    alg_num = div(size(raw_data, 2) - 1, 2);

    RecSigm = Vector{Float64}(undef, pts_num);
    RecSigm .= raw_data[:, 1];

    RecIter = Matrix{Int}(undef, pts_num, alg_num);
    RecIter .= raw_data[:, 2:(alg_num + 1)];

    RecTime = Matrix{Float64}(undef, pts_num, alg_num);
    RecTime .= raw_data[:, (alg_num + 2):(2 * alg_num + 1)];

    return RecSigm, RecIter, RecTime
end

function get_speedup(data)
    su = Matrix{Any}(undef, length(data), length(data));
    fill!(su, "-");
    
    for ii = eachindex(data)
        for jj = eachindex(data)
            su[ii, jj] = data[ii] / data[jj];
        end
    end

    return su;
end

function get_extreme_ratio(data)
    # display(data)
    su = Matrix{Any}(undef, size(data, 2), size(data, 2));

    for d_ind = 1:size(data, 2)
        su[d_ind, d_ind] = 1.0;
    end

    m_ind::Int = 0;
    r::Float64 = 0.0;

    for r_ind = axes(data, 2)
        for c_ind = axes(data, 2);
            if r_ind < c_ind
                r, m_ind = findmax(i -> data[i, r_ind] / data[i, c_ind], axes(data, 1));
                su[r_ind, c_ind] = (r, data[m_ind, r_ind], data[m_ind, c_ind]);
            elseif r_ind > c_ind
                r, m_ind = findmin(i -> data[i, c_ind] / data[i, r_ind], axes(data, 1));
                su[r_ind, c_ind] = (1.0 / r, data[m_ind, r_ind], data[m_ind, c_ind]);
            end
        end
    end

    return su;



end


function stlog_data_analysis(fname, labels)
    RecSigm, RecIter, RecTime = stlog_read_file(fname);

    pts_num, alg_num = size(RecIter);
    
    prob_set1 = RecSigm .<= 1.0;
    prob_set2 = 1.0 .< RecSigm .<= 2.5;
    prob_set3 = 2.5 .< RecSigm;
    
    AvgIter1 = mean(RecIter[prob_set1, :], dims = 1);
    AvgIter2 = mean(RecIter[prob_set2, :], dims = 1);
    AvgIter3 = mean(RecIter[prob_set3, :], dims = 1);

    AvgTime1 = mean(RecTime[prob_set1, :], dims = 1);
    AvgTime2 = mean(RecTime[prob_set2, :], dims = 1);
    AvgTime3 = mean(RecTime[prob_set3, :], dims = 1);

    println(labels)
    println("Easy problems")

    @printf "Average iter:\t"
    for ind in eachindex(AvgIter1)
        @printf "%.3f\t" AvgIter1[ind];
    end
    @printf "\n";

    @printf "Average time:\t"
    for ind in eachindex(AvgTime1)
        @printf "%.3f\t" AvgTime1[ind];
    end
    @printf "\n";

    @printf "Max time:\t"
    maxt = maximum(RecTime[prob_set1, :], dims = 1)
    for ind in eachindex(maxt)
        @printf "%.3f\t" maxt[ind]
    end
    @printf "\n";

    @printf "Min time:\t"
    mint = minimum(RecTime[prob_set1, :], dims = 1)
    for ind in eachindex(mint)
        @printf "%.3f\t" mint[ind]
    end
    @printf "\n";
    
    @printf "Average Speedup:\t"
    display(get_speedup(AvgTime1))

    @printf "Max Speedup:\t"
    display(get_extreme_ratio(copy(RecTime[prob_set1, :])))

    println("Medium problems")
    
    @printf "Average iter:\t"
    for ind in eachindex(AvgIter2)
        @printf "%.3f\t" AvgIter2[ind];
    end
    @printf "\n";

    @printf "Average time:\t"
    for ind in eachindex(AvgTime2)
        @printf "%.3f\t" AvgTime2[ind];
    end
    @printf "\n";

    @printf "Max time:\t"
    maxt = maximum(RecTime[prob_set2, :], dims = 1)
    for ind in eachindex(maxt)
        @printf "%.3f\t" maxt[ind]
    end
    @printf "\n";

    @printf "Min time:\t"
    mint = minimum(RecTime[prob_set2, :], dims = 1)
    for ind in eachindex(mint)
        @printf "%.3f\t" mint[ind]
    end
    @printf "\n";

    @printf "Average Speedup:\t"
    display(get_speedup(AvgTime2))

    @printf "Max Speedup:\t"
    display(get_extreme_ratio(copy(RecTime[prob_set2, :])))

    @printf "Average iter:\t"
    for ind in eachindex(AvgIter3)
        @printf "%.3f\t" AvgIter3[ind];
    end
    @printf "\n";

    @printf "Average time:\t"
    for ind in eachindex(AvgTime3)
        @printf "%.3f\t" AvgTime3[ind];
    end
    @printf "\n";

    @printf "Max time:\t"
    maxt = maximum(RecTime[prob_set3, :], dims = 1)
    for ind in eachindex(maxt)
        @printf "%.3f\t" maxt[ind]
    end
    @printf "\n";

    @printf "Min time:\t"
    mint = minimum(RecTime[prob_set3, :], dims = 1)
    for ind in eachindex(mint)
        @printf "%.3f\t" mint[ind]
    end
    @printf "\n";

    @printf "Average Speedup:\t"
    display(get_speedup(AvgTime3))

    @printf "Max Speedup:\t"
    display(get_extreme_ratio(copy(RecTime[prob_set3, :])))
end

function stlog_data_plot(fname, labels; filename = "")
    RecSigm, RecIter, RecTime = stlog_read_file(fname);

    pts_num, alg_num = size(RecIter);

    time_plt = scatter(RecSigm, RecTime,
        label=labels,
        # xlabel="2-norm of the generating velocity, |S_{A,B,0}|_2",
        ylabel="Time (ms)",
        xlabel="σ",
        # ylims = (1e-2, 8 * median(RecTime)),
        # yscale=:log2,
        markerstrokewidth=0,
        markershape = :circle,
        markerstrokecolor = :auto,
        lw=0,
        ms=1.5,
        ma=0.5,
        legend = :topleft,
    )

    iter_plt = scatter(RecSigm, RecIter,
        label=labels,
        # xlabel="2-norm of the generating velocity, |S_{A,B,0}|_2",
        xlabel="σ",
        ylabel="Iteration",
        # ylims = (1e-2, 8 * median(RecTime)),
        yscale=:log2,
        markershape = :circle,
        markerstrokewidth=0,
        markerstrokecolor = :auto,
        lw=0,
        ms=1.5,
        ma=0.5,
        legend = :topleft,
    )

    display(time_plt)

    display(iter_plt)


    if filename != ""
        pos = findlast('.', filename);
        savefig(iter_plt, filename[1:(pos-1)] * "_iter." * filename[(pos+1):end])
        savefig(time_plt, filename[1:(pos-1)] * "_time." * filename[(pos+1):end])
        savefig(plot(time_plt, yscale = :log2), filename[1:(pos-1)] * "_time_logscale." * filename[(pos+1):end])
    end
end