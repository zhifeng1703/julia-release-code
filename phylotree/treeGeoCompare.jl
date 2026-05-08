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

    println("\nTop test cases by saved min-cut calls:")
    println(" i   j   exact   tolerant   saved   ratio     abs error")
    for x in out.top_saved_cases
        @printf("%2d  %2d  %6d  %8d  %6d  %7.2f%%  %.3e\n",
            x.i, x.j, x.exact_complexity, x.tol_complexity,
            x.saved, 100*x.save_ratio, x.abs_error)
    end
end

function top_mincut_savings(results_exact, results_tol; ntop::Integer=10)
    m, n = size(results_exact)
    rows = NamedTuple[]

    @inbounds for i in 1:m, j in 1:n
        ne = length(results_exact[i, j].mincut_sizes)
        nt = length(results_tol[i, j].mincut_sizes)
        saved = ne - nt
        saved <= 0 && continue

        push!(rows, (
            i=i,
            j=j,
            exact_mincuts=ne,
            tol_mincuts=nt,
            saved=saved,
            save_ratio=saved / max(ne, 1),
            exact_dist=results_exact[i, j].curve_length,
            tol_dist=results_tol[i, j].curve_length,
            abs_error=abs(results_tol[i, j].curve_length - results_exact[i, j].curve_length),
        ))
    end

    sort!(rows, by = x -> (-x.saved, -x.save_ratio, x.i, x.j))
    return rows[1:min(ntop, length(rows))]
end

function result_complexity(out)
    total = 0.0
    for (m, n) in out.mincut_sizes
        total += (m + n)^2 * sqrt(m * n)
    end
    return total
end

function top_complexity_savings(results_exact, results_tol; ntop::Integer=10)
    m, n = size(results_exact)
    rows = NamedTuple[]

    @inbounds for i in 1:m, j in 1:n
        ce = result_complexity(results_exact[i, j])
        ct = result_complexity(results_tol[i, j])
        saved = ce - ct
        (saved <= 0 || ct == 0) && continue

        push!(rows, (
            i=i,
            j=j,
            exact_complexity=ce,
            tol_complexity=ct,
            saved=saved,
            exact_mincuts=length(results_exact[i, j].mincut_sizes),
            tol_mincuts=length(results_tol[i, j].mincut_sizes),
            save_ratio=saved / max(ce, 1e-12),
            abs_error=abs(results_tol[i, j].curve_length -
                          results_exact[i, j].curve_length),
        ))
    end

    sort!(rows, by = x -> (-x.saved, -x.save_ratio, x.i, x.j))
    return rows[1:min(ntop, length(rows))]
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

    # top_saved_cases = top_mincut_savings(results_exact, results_tol; ntop=10)
    top_saved_cases = top_complexity_savings(results_exact, results_tol; ntop=10)

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
        top_saved_cases=top_saved_cases,
        geo_data = geo_data,
    )

    verbose && print_summary_table(out)
    return out
end

function geodesic_weight_map(data, res; t::Real, atol::Real=1e-14)
    w = Dict{Any, Float64}()

    edgea, edgeb = data.edgea, data.edgeb
    wa, wb = data.a, data.b

    for pair in _support_pairs(res)
        Aidx, Bidx = _pair_indices(pair)

        anorm = sqrt(sum(wa[k]^2 for k in Aidx))
        bnorm = sqrt(sum(wb[k]^2 for k in Bidx))

        α = anorm <= atol ? 0.0 : max(0.0, (1 - t) - t * bnorm / anorm)
        β = bnorm <= atol ? 0.0 : max(0.0, t - (1 - t) * anorm / bnorm)

        for k in Aidx
            val = α * wa[k]
            val > atol && (w[edgea[k]] = val)
        end

        for k in Bidx
            val = β * wb[k]
            val > atol && (w[edgeb[k]] = val)
        end
    end

    # common edges, if stored separately in data.c
    if hasproperty(data, :c)
        for e in data.c
            # if data.c stores weighted pairs, adjust here if needed
            haskey(w, e) || (w[e] = 1.0)
        end
    end

    return w
end

# function plot_exact_geodesic_with_tol_skip(
#     treeA::PhyloTree,
#     treeB::PhyloTree;
#     abstol::Real=1e-6,
#     reltol::Real=1e-2,
# )
#     data = geodesic_data(treeA, treeB)

#     exact = refine_support(data.c, data.a, data.b, data.edgea, data.edgeb, data.inc;
#         abstol=0.0, reltol=0.0)

#     tol = refine_support(data.c, data.a, data.b, data.edgea, data.edgeb, data.inc;
#         abstol=abstol, reltol=reltol)

#     # supp_exact = exact.support
#     # supp_tol = tol.support

#     supp_exact = collect(zip(exact.supp_a, exact.supp_b))
#     supp_tol   = collect(zip(tol.supp_a, tol.supp_b))

#     red_edges = Set{Any}()
#     for s in supp_exact[(length(supp_tol)+1):end]
#         Aidx, Bidx = s
#         foreach(k -> push!(red_edges, data.edgea[k]), Aidx)
#         foreach(k -> push!(red_edges, data.edgeb[k]), Bidx)
#     end

#     color_map = Dict(e => :red for e in red_edges)

#     crit = Float64[]
#     for (Aidx, Bidx) in supp_exact
#         an = sqrt(sum(data.a[k]^2 for k in Aidx))
#         bn = sqrt(sum(data.b[k]^2 for k in Bidx))
#         push!(crit, an / (an + bn))
#     end
#     ts = vcat(0.0, sort(crit), 1.0)

#     plots = Any[]
#     for t in ts
#         w = Dict{Any, Float64}()

#         for (Aidx, Bidx) in supp_exact
#             an = sqrt(sum(data.a[k]^2 for k in Aidx))
#             bn = sqrt(sum(data.b[k]^2 for k in Bidx))

#             α = max(0.0, (1 - t) - t * bn / an)
#             β = max(0.0, t - (1 - t) * an / bn)

#             for k in Aidx
#                 α > 0 && (w[data.edgea[k]] = α * data.a[k])
#             end
#             for k in Bidx
#                 β > 0 && (w[data.edgeb[k]] = β * data.b[k])
#             end
#         end

#         for e in data.c
#             haskey(w, e) || (w[e] = 1.0)
#         end

#         # push!(plots, draw_tree(collect(keys(w)); weight_map=w, color_map=color_map))
#         clade = Bipart[collect(keys(w))...]
#         w_bip = Dict{Bipart,Float64}(e => v for (e, v) in w)
#         c_bip = Dict{Bipart,Symbol}(e => v for (e, v) in color_map)

#         push!(plots, draw_tree(clade; weight_map=w_bip, color_map=c_bip))
#     end

#     return plots
# end

function plot_exact_geodesic_with_tol_skip(
    treeA::PhyloTree,
    treeB::PhyloTree;
    abstol::Real=1e-6,
    reltol::Real=1e-2,
)
    data = geodesic_data(treeA, treeB)

    exact = refine_support(data.c, data.a, data.b, data.edgea, data.edgeb, data.inc;
        abstol=0.0, reltol=0.0)

    tol = refine_support(data.c, data.a, data.b, data.edgea, data.edgeb, data.inc;
        abstol=abstol, reltol=reltol)

    supp_exact = collect(zip(exact.supp_a, exact.supp_b))
    supp_tol = collect(zip(tol.supp_a, tol.supp_b))

    red_edges = Set{Bipart}()
    if length(supp_tol) < length(supp_exact)
        for (Aidx, Bidx) in supp_exact[(length(supp_tol)+1):end]
            foreach(k -> push!(red_edges, data.edgea[k]), Aidx)
            foreach(k -> push!(red_edges, data.edgeb[k]), Bidx)
        end
    end

    color_map = Dict{Bipart,Symbol}(e => :red for e in red_edges)

    crit = Float64[]
    for (Aidx, Bidx) in supp_exact
        an = sqrt(sum(data.a[k]^2 for k in Aidx))
        bn = sqrt(sum(data.b[k]^2 for k in Bidx))
        push!(crit, an / (an + bn))
    end
    ts = vcat(0.0, sort(crit), 1.0)

    plots = Any[]

    for t in ts
        w = Dict{Bipart,Float64}()

        for (Aidx, Bidx) in supp_exact
            an = sqrt(sum(data.a[k]^2 for k in Aidx))
            bn = sqrt(sum(data.b[k]^2 for k in Bidx))

            α = an == 0 ? 0.0 : max(0.0, (1 - t) - t * bn / an)
            β = bn == 0 ? 0.0 : max(0.0, t - (1 - t) * an / bn)

            for k in Aidx
                val = α * data.a[k]
                val > 0 && (w[data.edgea[k]] = val)
            end

            for k in Bidx
                val = β * data.b[k]
                val > 0 && (w[data.edgeb[k]] = val)
            end
        end

        # If data.c is a vector of common Bipart edges, keep them visible.
        if data.c isa AbstractVector
            for e in data.c
                e isa Bipart && !haskey(w, e) && (w[e] = 1.0)
            end
        end

        weight_map = e -> get(w, e, 0.0)

        clade = collect(keys(w))
        push!(plots, draw_tree(clade; weight_map=weight_map, color_map=color_map))
    end

    return plots
end