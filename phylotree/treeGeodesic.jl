using Graphs, GraphsFlows
using SimpleWeightedGraphs

using LinearAlgebra
include("mincut.jl")
include("treeObj.jl")

"""
    geodesic_data(treeA::PhyloTree, treeB::PhyloTree)

Build the support-refinement input for the BHV geodesic computation between
two phylogenetic trees.

Returns a named tuple with fields

- `c`      : squared contribution from common internal bipartitions
- `a`      : weights of internal bipartitions present only in treeA
- `b`      : weights of internal bipartitions present only in treeB
- `edgea`  : bipartitions corresponding to `a`
- `edgeb`  : bipartitions corresponding to `b`
- `inc`    : incompatibility matrix between `edgea` and `edgeb`
"""
function geodesic_data(treeA::PhyloTree, treeB::PhyloTree)
    # lookup for exact common bipartitions
    mapA = Dict{Bipart,Float64}(treeA.ib[i] => treeA.iw[i] for i in eachindex(treeA.ib))
    mapB = Dict{Bipart,Float64}(treeB.ib[i] => treeB.iw[i] for i in eachindex(treeB.ib))

    common = intersect(Set(treeA.ib), Set(treeB.ib))

    # squared contribution from common edges
    c = 0.0
    for e in common
        c += (mapA[e] - mapB[e])^2
    end

    # edges unique to A
    edgea = Bipart[]
    a = Float64[]
    for i in eachindex(treeA.ib)
        e = treeA.ib[i]
        if !(e in common)
            push!(edgea, e)
            push!(a, treeA.iw[i])
        end
    end

    # edges unique to B
    edgeb = Bipart[]
    b = Float64[]
    for j in eachindex(treeB.ib)
        e = treeB.ib[j]
        if !(e in common)
            push!(edgeb, e)
            push!(b, treeB.iw[j])
        end
    end

    # incompatibility matrix
    inc = falses(length(edgea), length(edgeb))
    for i in eachindex(edgea), j in eachindex(edgeb)
        inc[i, j] = !_bipart_comp(edgea[i], edgeb[j])
    end

    return (c=c, a=a, b=b, edgea=edgea, edgeb=edgeb, inc=inc)
end


"""
    geodesic_support(treeA::PhyloTree, treeB::PhyloTree; abstol=0.0, reltol=0.0)

Compute the support sequence for the BHV geodesic between two trees by calling
`refine_support`.

Returns a named tuple with fields

- all outputs of `refine_support`
- plus `c`, `a`, `b`, `edgea`, `edgeb`, `inc`
"""
function geodesic_support(treeA::PhyloTree, treeB::PhyloTree; abstol::Real=0.0, reltol::Real=0.0)
    data = geodesic_data(treeA, treeB)

    out = refine_support(
        data.c,
        data.a,
        data.b,
        data.edgea,
        data.edgeb,
        data.inc;
        abstol=abstol,
        reltol=reltol,
    )

    return merge(data, out)
end

"""
    geodesic_distance(treeA, treeB; abstol=0.0, reltol=0.0)

Call `refine_support` and return the curve length together with the raw output.
"""
function geodesic_distance(treeA::PhyloTree, treeB::PhyloTree; abstol::Real=0.0, reltol::Real=0.0)
    data = geodesic_data(treeA, treeB)

    out = refine_support(
        data.c,
        data.a,
        data.b,
        data.edgea,
        data.edgeb,
        data.inc;
        abstol=abstol,
        reltol=reltol,
    )

    return out.curve_length, merge(data, out)
end
"""
    refine_support(c, a, b, edgea, edgeb, inc; abstol=0.0, reltol=0.0)

Refine one support pair into a support sequence, with early termination.

In addition to the previous outputs, this version also returns

- `mincut_sizes::Vector{Tuple{Int,Int}}`

where each entry `(m,n)` records one min-cut problem that was actually solved
during refinement, with `m = |A_i|`, `n = |B_i|` for the refined last pair.
"""
function refine_support(
    c::Real,
    a::AbstractVector{<:Real},
    b::AbstractVector{<:Real},
    edgea::AbstractVector,
    edgeb::AbstractVector,
    inc::AbstractMatrix{Bool};
    abstol::Real=0.0,
    reltol::Real=0.0,
)
    length(edgea) == length(a) || throw(ArgumentError("edgea must match a"))
    length(edgeb) == length(b) || throw(ArgumentError("edgeb must match b"))
    size(inc, 1) == length(a) || throw(ArgumentError("inc row size mismatch"))
    size(inc, 2) == length(b) || throw(ArgumentError("inc col size mismatch"))
    abstol >= 0 || throw(ArgumentError("abstol must be nonnegative"))
    reltol >= 0 || throw(ArgumentError("reltol must be nonnegative"))

    # ------------------------------------------------------------
    # Initial sort
    # ------------------------------------------------------------
    pa = sortperm(a; rev=true)
    pb = sortperm(b; rev=true)

    a1 = Float64.(a[pa])
    b1 = Float64.(b[pb])
    edgea1 = edgea[pa]
    edgeb1 = edgeb[pb]
    inc1 = inc[pa, pb]

    idxA1 = collect(pa)
    idxB1 = collect(pb)

    # ------------------------------------------------------------
    # Initial peel
    # ------------------------------------------------------------
    peeled = peel_fully_compatible(c, a1, b1, edgea1, edgeb1, inc1)
    ccur = peeled.c

    keepA = peeled.idx_a
    keepB = peeled.idx_b

    idxA2 = idxA1[keepA]
    idxB2 = idxB1[keepB]

    mincut_sizes = Tuple{Int,Int}[]

    # If peel resolves everything
    if isempty(idxA2) || isempty(idxB2)
        return (
            supp_a=Vector{Vector{Int}}(),
            supp_b=Vector{Vector{Int}}(),
            curve_length=sqrt(ccur),
            abstol=float(abstol),
            reltol=float(reltol),
            abserr=0.0,
            relerr=0.0,
            status=:resolved,
            mincut_sizes=mincut_sizes,
        )
    end

    # Support sequence stored in original indices.
    supp_a = [collect(idxA2)]
    supp_b = [collect(idxB2)]

    sqL_cur = ccur
    sqK_cur = 0.0

    while true
        k = length(supp_a)

        # Current last pair
        idxA = supp_a[k]
        idxB = supp_b[k]

        ak = Float64.(a[idxA])
        bk = Float64.(b[idxB])

        # --------------------------------------------------------
        # Current squared curve length
        # --------------------------------------------------------
        sqK_cur = (norm(ak) + norm(bk))^2
        L2 = sqL_cur + sqK_cur

        # --------------------------------------------------------
        # Lower bound: fixed earlier pairs + best refinement of last
        # --------------------------------------------------------
        LB2 = sqL_cur + phi_best_sorted(ak, bk)

        L = sqrt(L2)
        LB = sqrt(LB2)

        abserr = L - LB
        relerr = iszero(LB) ? 0.0 : abserr / LB

        # --------------------------------------------------------
        # Early termination
        # --------------------------------------------------------
        if abserr <= abstol || relerr <= reltol
            return (
                supp_a=[sort(copy(x)) for x in supp_a],
                supp_b=[sort(copy(x)) for x in supp_b],
                curve_length=L,
                abstol=float(abstol),
                reltol=float(reltol),
                abserr=abserr,
                relerr=relerr,
                status=(abserr == 0 ? :exact : :tolerance),
                mincut_sizes=mincut_sizes,
            )
        end

        # --------------------------------------------------------
        # Min-cut refinement on the last pair
        # --------------------------------------------------------
        # Record the size of the min-cut problem that is ACTUALLY solved
        push!(mincut_sizes, (length(idxA), length(idxB)))

        inck = inc[idxA, idxB]
        out = refine_pair_mincut(ak, bk, inck)

        # Mathematical convention:
        #   refine iff cut_value < 1
        #   terminate iff cut_value >= 1
        if out.cut_value >= 1.0 - 1e-12
            return (
                supp_a=[sort(copy(x)) for x in supp_a],
                supp_b=[sort(copy(x)) for x in supp_b],
                curve_length=L,
                abstol=float(abstol),
                reltol=float(reltol),
                abserr=0.0,
                relerr=0.0,
                status=:exact,
                mincut_sizes=mincut_sizes,
            )
        end

        # Split last pair:
        #   A = R ∪ S,  B = U ∪ V
        Rorig = collect(idxA[out.Ridx])
        Sorig = collect(idxA[out.Sidx])
        Uorig = collect(idxB[out.Uidx])
        Vorig = collect(idxB[out.Vidx])

        supp_a[end] = Rorig
        supp_b[end] = Uorig
        push!(supp_a, Sorig)
        push!(supp_b, Vorig)

        sqL_cur += (norm(a[Rorig]) + norm(b[Uorig]))^2
    end
end

"""
    peel_fully_compatible(c, a, b, edgea, edgeb, inc)

Peel off bipartitions that are compatible with all bipartitions on the other side.

If a peeled bipartition occurs on both sides, say with weights a[i] and b[j],
then its contribution to `c` is (a[i] - b[j])^2.

If it occurs only on one side, its contribution is just the squared weight.

Inputs
------
- `c      :: Real`
- `a      :: AbstractVector{<:Real}`   sorted weights on A-side
- `b      :: AbstractVector{<:Real}`   sorted weights on B-side
- `edgea  :: AbstractVector`           bipartition/edge labels for `a`
- `edgeb  :: AbstractVector`           bipartition/edge labels for `b`
- `inc    :: AbstractMatrix{Bool}`     incompatibility matrix:
                                       inc[i,j] = true means edgea[i] incompatible with edgeb[j]

Returns
-------
A named tuple with fields

- `c`
- `idx_a`
- `idx_b`

where `idx_a`, `idx_b` are the remaining indices in increasing order.
"""
function peel_fully_compatible(c::Real, a::AbstractVector{Ta}, b::AbstractVector{Tb}, edgea::AbstractVector, edgeb::AbstractVector, inc::AbstractMatrix{Bool}) where {Ta<:Real,Tb<:Real}

    M = length(a)
    N = length(b)

    length(edgea) == M || throw(ArgumentError("edgea must have same length as a"))
    length(edgeb) == N || throw(ArgumentError("edgeb must have same length as b"))
    size(inc, 1) == M || throw(ArgumentError("inc has wrong number of rows"))
    size(inc, 2) == N || throw(ArgumentError("inc has wrong number of columns"))

    # Compatible with everything on the other side
    peel_a = [!any(@view inc[i, :]) for i in 1:M]
    peel_b = [!any(@view inc[:, j]) for j in 1:N]

    c_new = float(c)

    removed_a = falses(M)
    removed_b = falses(N)

    # Match peeled identical bipartitions first
    b_lookup = Dict{Any,Int}()
    for j in 1:N
        if peel_b[j]
            b_lookup[edgeb[j]] = j
        end
    end

    for i in 1:M
        if peel_a[i]
            key = edgea[i]
            if haskey(b_lookup, key)
                j = b_lookup[key]
                if !removed_b[j]
                    c_new += (a[i] - b[j])^2
                    removed_a[i] = true
                    removed_b[j] = true
                end
            end
        end
    end

    # Unmatched peeled entries on A-side
    for i in 1:M
        if peel_a[i] && !removed_a[i]
            c_new += a[i]^2
            removed_a[i] = true
        end
    end

    # Unmatched peeled entries on B-side
    for j in 1:N
        if peel_b[j] && !removed_b[j]
            c_new += b[j]^2
            removed_b[j] = true
        end
    end

    idx_a = findall(!, removed_a)
    idx_b = findall(!, removed_b)

    return (c=c_new, idx_a=idx_a, idx_b=idx_b)
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