using Printf
using LinearAlgebra
using Plots

include("treeDriver.jl")

complexity_weight(m, n) = (m + n)^2 * sqrt(m * n)

function maxindex_histogram(results::AbstractMatrix)
    hist = Dict{Int,Int}()
    for x in results
        for (m, n) in x.mincut_sizes
            k = max(m, n)
            hist[k] = get(hist, k, 0) + 1
        end
    end
    return hist
end

function maxindex_complexity(results::AbstractMatrix)
    hist = Dict{Int,Float64}()
    for x in results
        for (m, n) in x.mincut_sizes
            k = max(m, n)
            hist[k] = get(hist, k, 0.0) + complexity_weight(m, n)
        end
    end
    return hist
end

total_mincuts(results::AbstractMatrix) = sum(length(x.mincut_sizes) for x in results)

function total_complexity(results::AbstractMatrix)
    s = 0.0
    for x in results
        for (m, n) in x.mincut_sizes
            s += complexity_weight(m, n)
        end
    end
    return s
end

function dict_to_vector(hist::Dict{Int,T}) where T
    isempty(hist) && return Int[], T[]
    kmax = maximum(keys(hist))
    x = collect(1:kmax)
    y = [get(hist, k, zero(T)) for k in x]
    return x, y
end

function plot_count_curve(hist_exact::Dict{Int,Int}, hist_tol::Dict{Int,Int})
    x1, y1 = dict_to_vector(hist_exact)
    x2, y2 = dict_to_vector(hist_tol)
    xmax = max(isempty(x1) ? 0 : maximum(x1), isempty(x2) ? 0 : maximum(x2))
    x = collect(1:xmax)
    ye = [get(hist_exact, k, 0) for k in x]
    yt = [get(hist_tol, k, 0) for k in x]
    p = plot(x, ye; label="exact", marker=:circle, xlabel="problem size", ylabel="count", title="Min-cut count")
    plot!(p, x, yt; label="tolerant", marker=:square)
    return p
end

function plot_cost_curve(hist_exact::Dict{Int,Float64}, hist_tol::Dict{Int,Float64})
    x1, y1 = dict_to_vector(hist_exact)
    x2, y2 = dict_to_vector(hist_tol)
    xmax = max(isempty(x1) ? 0 : maximum(x1), isempty(x2) ? 0 : maximum(x2))
    x = collect(1:xmax)
    ye = [get(hist_exact, k, 0.0) for k in x]
    yt = [get(hist_tol, k, 0.0) for k in x]
    p = plot(x, ye; label="exact", marker=:circle, xlabel="problem size", ylabel="cost", title="Estimated min-cut cost")
    plot!(p, x, yt; label="tolerant", marker=:square)
    return p
end

function print_summary_table(out)
    @printf("\n")
    @printf("%-18s %14s %14s\n", "metric", "exact", "tolerant")
    @printf("%-18s %14.6f %14.6f\n", "time (s)", out.time_exact, out.time_tol)
    @printf("%-18s %14d %14d\n", "min-cut count", out.mincuts_exact_total, out.mincuts_tol_total)
    @printf("%-18s %14.6e %14.6e\n", "est. complexity", out.complexity_exact_total, out.complexity_tol_total)
    @printf("\n")
    @printf("%-18s %14.6f\n", "speedup", out.time_exact / out.time_tol)
    @printf("%-18s %14d\n", "saved min-cuts", out.mincuts_exact_total - out.mincuts_tol_total)
    @printf("%-18s %14.6e\n", "saved complexity", out.complexity_exact_total - out.complexity_tol_total)
    @printf("%-18s %14.6e\n", "max abs diff", maximum(abs.(out.D_tol .- out.D_exact)))
    @printf("%-18s %14.6e\n", "mean abs diff", sum(abs.(out.D_tol .- out.D_exact)) / length(out.D_exact))
    @printf("%-18s %14.6e\n", "frobenius diff", norm(out.D_tol .- out.D_exact))
end

function test_geodesic_matrix(
    setA::AbstractVector{PhyloTree},
    setB::AbstractVector{PhyloTree};
    abstol::Real=1e-6,
    reltol::Real=1e-2,
    nrepeat::Integer=10,
    verbose::Bool=true,
)
    m, n = length(setA), length(setB)

    D_exact = Matrix{Float64}(undef, m, n)
    D_tol = Matrix{Float64}(undef, m, n)
    results_exact = Matrix{Any}(undef, m, n)
    results_tol = Matrix{Any}(undef, m, n)
    geo_data = Matrix{Any}(undef, m, n)

    @inbounds for i in 1:m, j in 1:n
        geo_data[i, j] = geodesic_data(setA[i], setB[j])
    end

    t_exact_rec = zeros(nrepeat)
    for r in 1:nrepeat
        t_exact_rec[r] = @elapsed begin
            @inbounds for i in 1:m, j in 1:n
                data = geo_data[i, j]
                out = refine_support(
                    data.c, data.a, data.b, data.edgea, data.edgeb, data.inc;
                    abstol=0.0, reltol=0.0
                )
                D_exact[i, j] = out.curve_length
                results_exact[i, j] = out
            end
        end
    end
    t_exact = minimum(t_exact_rec)

    t_tol_rec = zeros(nrepeat)
    for r in 1:nrepeat
        t_tol_rec[r] = @elapsed begin
            @inbounds for i in 1:m, j in 1:n
                data = geo_data[i, j]
                out = refine_support(
                    data.c, data.a, data.b, data.edgea, data.edgeb, data.inc;
                    abstol=abstol, reltol=reltol
                )
                D_tol[i, j] = out.curve_length
                results_tol[i, j] = out
            end
        end
    end
    t_tol = minimum(t_tol_rec)

    hist_count_exact = maxindex_histogram(results_exact)
    hist_count_tol = maxindex_histogram(results_tol)

    hist_cost_exact = maxindex_complexity(results_exact)
    hist_cost_tol = maxindex_complexity(results_tol)

    mincuts_exact_total = total_mincuts(results_exact)
    mincuts_tol_total = total_mincuts(results_tol)

    complexity_exact_total = total_complexity(results_exact)
    complexity_tol_total = total_complexity(results_tol)

    out = (
        D_exact=D_exact,
        D_tol=D_tol,
        time_exact=t_exact,
        time_tol=t_tol,
        results_exact=results_exact,
        results_tol=results_tol,
        mincuts_exact_total=mincuts_exact_total,
        mincuts_tol_total=mincuts_tol_total,
        complexity_exact_total=complexity_exact_total,
        complexity_tol_total=complexity_tol_total,
        hist_count_exact=hist_count_exact,
        hist_count_tol=hist_count_tol,
        hist_cost_exact=hist_cost_exact,
        hist_cost_tol=hist_cost_tol,
        plt_count=plot_count_curve(hist_count_exact, hist_count_tol),
        plt_cost=plot_cost_curve(hist_cost_exact, hist_cost_tol),
    )

    verbose && print_summary_table(out)
    return out
end