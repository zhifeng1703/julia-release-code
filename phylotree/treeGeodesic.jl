using Graphs, GraphsFlows
using SimpleWeightedGraphs

using LinearAlgebra
include("mincut.jl")
include("treeObj.jl")


struct SupportPair
    A::Vector{Bipart}
    WA::Vector{Float64}
    B::Vector{Bipart}
    WB::Vector{Float64}
end

struct SharedPair
    C::Vector{Bipart}
    W0::Vector{Float64}
    W1::Vector{Float64}
end

struct PathInfo
    curve_length::Float64
    status::Symbol
    abserr::Float64
    relerr::Float64
    abstol::Float64
    reltol::Float64
    mincut_sizes::Vector{Tuple{Int,Int}}
end

function _shared_sqdist(shared::SharedPair)
    return sum((shared.W0[i] - shared.W1[i])^2 for i in eachindex(shared.C))
end

function _leaf_biparts(n::Int)
    leaf = n == BITSTR_SIZE ? typemax(BITSTR_TYPE) : (BITSTR_TYPE(1) << n) - 1
    return [Bipart(BITSTR_TYPE(1) << (i - 1), leaf) for i in 1:n]
end

function geodesic_initial(treeA::PhyloTree, treeB::PhyloTree)
    length(treeA.lw) == length(treeB.lw) || throw(ArgumentError("leaf weight size mismatch"))

    leaf_edges = _leaf_biparts(length(treeA.lw))

    mapA = Dict{Bipart,Float64}()
    mapB = Dict{Bipart,Float64}()

    for i in eachindex(treeA.ib)
        mapA[treeA.ib[i]] = treeA.iw[i]
    end
    for i in eachindex(treeB.ib)
        mapB[treeB.ib[i]] = treeB.iw[i]
    end
    for i in eachindex(leaf_edges)
        mapA[leaf_edges[i]] = treeA.lw[i]
        mapB[leaf_edges[i]] = treeB.lw[i]
    end

    allA = collect(keys(mapA))
    allB = collect(keys(mapB))

    common_set = Set{Bipart}()

    for e in allA
        if haskey(mapB, e) || all(_bipart_comp(e, f) for f in allB)
            push!(common_set, e)
        end
    end

    for e in allB
        if haskey(mapA, e) || all(_bipart_comp(f, e) for f in allA)
            push!(common_set, e)
        end
    end

    C = collect(common_set)

    shared = SharedPair(
        C,
        [get(mapA, e, 0.0) for e in C],
        [get(mapB, e, 0.0) for e in C],
    )

    A = Bipart[]
    WA = Float64[]
    for e in allA
        e in common_set && continue
        push!(A, e)
        push!(WA, mapA[e])
    end

    B = Bipart[]
    WB = Float64[]
    for e in allB
        e in common_set && continue
        push!(B, e)
        push!(WB, mapB[e])
    end

    pa = sortperm(WA; rev=true)
    pb = sortperm(WB; rev=true)

    A, WA = A[pa], WA[pa]
    B, WB = B[pb], WB[pb]

    initial = SupportPair(A, WA, B, WB)

    inc = falses(length(A), length(B))
    for i in eachindex(A), j in eachindex(B)
        inc[i, j] = !_bipart_comp(A[i], B[j])
    end

    return initial, shared, inc
end

function refine_support(
    initial::SupportPair,
    shared::SharedPair,
    inc::AbstractMatrix{Bool};
    abstol::Real=0.0,
    reltol::Real=0.0,)

    size(inc, 1) == length(initial.A) || throw(ArgumentError("inc row size mismatch"))
    size(inc, 2) == length(initial.B) || throw(ArgumentError("inc col size mismatch"))

    path = SupportPair[initial]
    sqL_cur = _shared_sqdist(shared)
    mincut_sizes = Tuple{Int,Int}[]

    while true
        sp = path[end]

        if isempty(sp.A) || isempty(sp.B)
            pop!(path)
            return path, PathInfo(
               sqrt(sqL_cur), :resolved, 0.0, 0.0,
               float(abstol), float(reltol), mincut_sizes,
            )
            throw("Error!")
        end

        ak = sp.WA
        bk = sp.WB

        L2 = sqL_cur + (norm(ak) + norm(bk))^2
        LB2 = sqL_cur + phi_best_sorted(ak, bk)

        L = sqrt(L2)
        LB = sqrt(LB2)

        abserr = L - LB
        relerr = iszero(LB) ? 0.0 : abserr / LB

        if abserr <= abstol || relerr <= reltol
            return path, PathInfo(
                L, abserr == 0.0 ? :exact : :tolerance,
                abserr, relerr,
                float(abstol), float(reltol), mincut_sizes,
            )
        end

        push!(mincut_sizes, (length(sp.A), length(sp.B)))

        inck = falses(length(sp.A), length(sp.B))
        for i in eachindex(sp.A), j in eachindex(sp.B)
            inck[i, j] = inc[
                findfirst(==(sp.A[i]), initial.A),
                findfirst(==(sp.B[j]), initial.B)
            ]
        end

        out = refine_pair_mincut(ak, bk, inck)

        if out.cut_value >= 1.0 - 1e-12
            return path, PathInfo(
                L, :exact, 0.0, 0.0,
                float(abstol), float(reltol), mincut_sizes,
            )
        end

        R = SupportPair(sp.A[out.Ridx], sp.WA[out.Ridx], sp.B[out.Uidx], sp.WB[out.Uidx])
        S = SupportPair(sp.A[out.Sidx], sp.WA[out.Sidx], sp.B[out.Vidx], sp.WB[out.Vidx])

        pop!(path)
        push!(path, R)
        push!(path, S)

        sqL_cur += (norm(R.WA) + norm(R.WB))^2
    end
end

"""
    refine_pair_mincut(a, b, inc)

Solve the Owen–Provan refinement min-cut problem for one support pair.

Inputs
------
- `a::AbstractVector{<:Real}`:
    sorted weights for 𝒜ᵢ
- `b::AbstractVector{<:Real}`:
    sorted weights for ℬᵢ
- `inc::AbstractMatrix{Bool}`:
    incompatibility matrix, where `inc[p,q] == true` means
    edge `a[p]` is incompatible with edge `b[q]`

Returns
-------
A named tuple with fields

- `cut_value::Float64`
- `Ridx::Vector{Int}`
- `Sidx::Vector{Int}`
- `Uidx::Vector{Int}`
- `Vidx::Vector{Int}`

The split is
    𝒜ᵢ = R ∪ S,   ℬᵢ = U ∪ V,
with S and U compatible.

The returned index vectors are increasing, so if `a` and `b` were sorted,
the subvectors `a[Ridx]`, `a[Sidx]`, `b[Uidx]`, `b[Vidx]` remain sorted.
"""
function refine_pair_mincut(a::AbstractVector{Ta}, b::AbstractVector{Tb}, inc::AbstractMatrix{Bool}) where {Ta<:Real,Tb<:Real}

    M = length(a)
    N = length(b)

    size(inc, 1) == M || throw(ArgumentError("inc has wrong number of rows"))
    size(inc, 2) == N || throw(ArgumentError("inc has wrong number of columns"))
    M > 0 || throw(ArgumentError("a must be nonempty"))
    N > 0 || throw(ArgumentError("b must be nonempty"))
    # Normalized vertex weights
    A2 = sum(abs2, a)
    B2 = sum(abs2, b)
    A2 > 0 || throw(ArgumentError("a must not be identically zero"))
    B2 > 0 || throw(ArgumentError("b must not be identically zero"))

    wa = Float64.(abs2.(a) ./ A2)
    wb = Float64.(abs2.(b) ./ B2)

    # Node numbering:
    # 1                = source s
    # 2:1+M            = A-side vertices
    # 2+M:1+M+N        = B-side vertices
    # 2+M+N            = sink t
    s = 1
    astart = 2
    bstart = 2 + M
    t = 2 + M + N
    nv = t

    # Any number > total finite capacity (=2) is enough for "infinity"
    INF = 3.0

    # Build Dinic graph
    dg = DinicGraph(nv)

    # source -> A
    @inbounds for i in 1:M
        add_edge!(dg, s, astart + i - 1, wa[i])
    end

    # B -> sink
    @inbounds for j in 1:N
        add_edge!(dg, bstart + j - 1, t, wb[j])
    end

    # A -> B incompatibility arcs with "infinite" capacity
    @inbounds for i in 1:M
        ui = astart + i - 1
        for j in 1:N
            if inc[i, j]
                vj = bstart + j - 1
                add_edge!(dg, ui, vj, INF)
            end
        end
    end

    # Solve min s-t cut by Dinic
    cut_value = dinic_maxflow!(dg, s, t)

    # Reachable set from s in the residual graph
    Sset = _reachable_from_source(dg, s)

    # println("Min-cut / max-flow value: ", cut_value)

    # for u in 1:length(dg.g)
    #     for e in dg.g[u]
    #         println("residual cap ", u, " -> ", e.to, " = ", e.cap)
    #     end
    # end

    # Recover min vertex cover:
    # R = A ∩ Tside,  V = B ∩ Sside
    # and complement independent set:
    # S = A ∩ Sside,  U = B ∩ Tside
    Ridx = Int[]
    Sidx = Int[]
    Uidx = Int[]
    Vidx = Int[]

    @inbounds for i in 1:M
        v = astart + i - 1
        if Sset[v]
            push!(Sidx, i)
        else
            push!(Ridx, i)
        end
    end

    @inbounds for j in 1:N
        v = bstart + j - 1
        if Sset[v]
            push!(Vidx, j)
        else
            push!(Uidx, j)
        end
    end

    return (
        cut_value=cut_value,
        Ridx=Ridx,
        Sidx=Sidx,
        Uidx=Uidx,
        Vidx=Vidx,
    )
end


"""
    _residual_reachable(g, F, s)

Given a directed weighted graph `g`, a flow matrix `F`, and source `s`,
return a Bool vector marking vertices reachable from `s` in the residual graph.
"""
function _residual_reachable(g::SimpleWeightedDiGraph, F, s::Int)
    n = nv(g)
    seen = falses(n)
    queue = Vector{Int}(undef, n)
    head = 1
    tail = 1
    queue[1] = s
    seen[s] = true

    while head <= tail
        u = queue[head]
        head += 1

        # Forward residual arcs: c(u,v) - f(u,v) > 0
        for v in outneighbors(g, u)
            cap = get_weight(g, u, v)
            if !seen[v] && cap - F[u, v] > 1e-12
                tail += 1
                queue[tail] = v
                seen[v] = true
            end
        end

        # Backward residual arcs: f(v,u) > 0
        for v in inneighbors(g, u)
            if !seen[v] && F[v, u] > 1e-12
                tail += 1
                queue[tail] = v
                seen[v] = true
            end
        end
    end

    return seen
end

"""
    phi_best_sorted(a, b)

Given two nonincreasing vectors `a` and `b` of positive weights corresponding to
(𝒜_i, ℬ_i), return the lower-bound contribution for the best possible refinement
of the last support pair.

The inputs are assumed sorted as
    a[1] ≥ a[2] ≥ ... > 0,
    b[1] ≥ b[2] ≥ ... > 0.
"""
function phi_best_sorted(a::AbstractVector{T}, b::AbstractVector{T}) where {T<:Real}
    M = length(a)
    N = length(b)

    if M == 0 || N == 0
        throw(ArgumentError("Input vectors must be nonempty."))
    end

    if M == N
        s = zero(promote_type(T, Float64))
        @inbounds for r in 1:M
            s += (a[r] + b[M-r+1])^2
        end
        return s

    elseif M > N
        k = M - N + 1
        s = sum(abs2, @view a[1:k])
        val = (sqrt(s) + b[N])^2
        @inbounds for j in 1:(N-1)
            val += (a[k+j] + b[N-j])^2
        end
        return val

    else
        k = N - M + 1
        s = sum(abs2, @view b[1:k])
        val = (a[M] + sqrt(s))^2
        @inbounds for j in 1:(M-1)
            val += (a[M-j] + b[k+j])^2
        end
        return val
    end
end


function geodesic_tree_at(
    path::Vector{SupportPair},
    shared::SharedPair,
    t::Real;
    atol::Float64=1e-16,)

    K = length(path)
    τ = clamp(float(t), 0.0, 1.0)
    W = Dict{Bipart,Float64}()

    for (k, sp) in enumerate(path)
        tk = k / (K + 1)

        α = τ <= tk ? 1.0 - τ / tk : 0.0
        β = τ >= tk ? (τ - tk) / (1.0 - tk) : 0.0

        for (e, w) in zip(sp.A, sp.WA)
            ew = α * w
            ew > atol && (W[e] = ew)
        end

        for (e, w) in zip(sp.B, sp.WB)
            ew = β * w
            ew > atol && (W[e] = ew)
        end
    end

    for (e, w0, w1) in zip(shared.C, shared.W0, shared.W1)
        ew = (1.0 - τ) * w0 + τ * w1
        ew > atol && (W[e] = ew)
    end

    return W
end