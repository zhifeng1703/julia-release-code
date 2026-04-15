# ============================================================
# Dinic max-flow / min-cut for the phylogenetic refinement step
#
# Input:
#   a   :: Vector{Float64}      positive weights, length M
#   b   :: Vector{Float64}      positive weights, length N
#   inc :: Matrix{Bool}         M x N incompatibility matrix
#
# Output:
#   (
#       cut_value :: Float64,
#       Ridx :: Vector{Int},
#       Sidx :: Vector{Int},
#       Uidx :: Vector{Int},
#       Vidx :: Vector{Int}
#   )
#
# Convention:
#   A = R ∪ S,  B = U ∪ V,
#   and S, U are compatible.
#
# Refinement condition:
#   cut_value < 1
# ============================================================

mutable struct DinicEdge
    to::Int
    rev::Int
    cap::Float64
end

mutable struct DinicGraph
    g::Vector{Vector{DinicEdge}}
end

function DinicGraph(n::Int)
    return DinicGraph([DinicEdge[] for _ in 1:n])
end

function add_edge!(dg::DinicGraph, u::Int, v::Int, c::Real)
    cf = Float64(c)
    push!(dg.g[u], DinicEdge(v, length(dg.g[v]) + 1, cf))
    push!(dg.g[v], DinicEdge(u, length(dg.g[u]), 0.0))
    return nothing
end

function _bfs_level!(dg::DinicGraph, s::Int, t::Int, level::Vector{Int}; tol::Float64=1e-12)
    fill!(level, -1)
    level[s] = 0

    q = Vector{Int}(undef, length(level))
    head = 1
    tail = 1
    q[1] = s

    while head <= tail
        u = q[head]
        head += 1
        for e in dg.g[u]
            if e.cap > tol && level[e.to] < 0
                level[e.to] = level[u] + 1
                tail += 1
                q[tail] = e.to
            end
        end
    end

    return level[t] >= 0
end

function _dfs_block!(
    dg::DinicGraph,
    u::Int,
    t::Int,
    f::Float64,
    level::Vector{Int},
    it::Vector{Int};
    tol::Float64=1e-12,
)
    u == t && return f

    gu = dg.g[u]
    m = length(gu)

    while it[u] <= m
        i = it[u]
        e = gu[i]

        if e.cap > tol && level[u] + 1 == level[e.to]
            pushed = _dfs_block!(dg, e.to, t, min(f, e.cap), level, it; tol=tol)
            if pushed > tol
                dg.g[u][i].cap -= pushed
                rev = e.rev
                dg.g[e.to][rev].cap += pushed
                return pushed
            end
        end

        it[u] += 1
    end

    return 0.0
end

function dinic_maxflow!(dg::DinicGraph, s::Int, t::Int; tol::Float64=1e-12)
    n = length(dg.g)
    level = fill(-1, n)
    it = ones(Int, n)
    flow = 0.0

    while _bfs_level!(dg, s, t, level; tol=tol)
        fill!(it, 1)
        while true
            pushed = _dfs_block!(dg, s, t, Inf, level, it; tol=tol)
            pushed <= tol && break
            flow += pushed
        end
    end

    return flow
end

function _reachable_from_source(dg::DinicGraph, s::Int; tol::Float64=1e-12)
    n = length(dg.g)
    seen = falses(n)

    q = Vector{Int}(undef, n)
    head = 1
    tail = 1
    q[1] = s
    seen[s] = true

    while head <= tail
        u = q[head]
        head += 1
        for e in dg.g[u]
            if e.cap > tol && !seen[e.to]
                seen[e.to] = true
                tail += 1
                q[tail] = e.to
            end
        end
    end

    return seen
end

function refine_pair_mincut(
    a::AbstractVector{Ta},
    b::AbstractVector{Tb},
    inc::AbstractMatrix{Bool};
    tol::Float64 = 1e-12,
) where {Ta<:Real,Tb<:Real}

    M = length(a)
    N = length(b)

    M > 0 || throw(ArgumentError("a must be nonempty"))
    N > 0 || throw(ArgumentError("b must be nonempty"))
    size(inc, 1) == M || throw(ArgumentError("inc has wrong number of rows"))
    size(inc, 2) == N || throw(ArgumentError("inc has wrong number of columns"))

    A2 = sum(abs2, a)
    B2 = sum(abs2, b)
    A2 > 0 || throw(ArgumentError("a must not be identically zero"))
    B2 > 0 || throw(ArgumentError("b must not be identically zero"))

    wa = Float64.(abs2.(a) ./ A2)
    wb = Float64.(abs2.(b) ./ B2)

    # Node numbering:
    # 1             = source
    # 2 : 1+M       = A-side nodes
    # 2+M : 1+M+N   = B-side nodes
    # 2+M+N         = sink
    s = 1
    astart = 2
    bstart = 2 + M
    t = 2 + M + N
    nv = t

    dg = DinicGraph(nv)

    # Since total finite capacity is 2, any number > 2 is enough for "infinity".
    INF = 3.0

    # source -> A
    @inbounds for i in 1:M
        add_edge!(dg, s, astart + i - 1, wa[i])
    end

    # B -> sink
    @inbounds for j in 1:N
        add_edge!(dg, bstart + j - 1, t, wb[j])
    end

    # incompatibility arcs A -> B
    @inbounds for i in 1:M
        ui = astart + i - 1
        for j in 1:N
            if inc[i, j]
                vj = bstart + j - 1
                add_edge!(dg, ui, vj, INF)
            end
        end
    end

    cut_value = dinic_maxflow!(dg, s, t; tol=tol)
    reachable = _reachable_from_source(dg, s; tol=tol)

    # Recover split:
    #   S = A ∩ reachable,   R = A \ S
    #   V = B ∩ reachable,   U = B \ V
    Ridx = Int[]
    Sidx = Int[]
    @inbounds for i in 1:M
        v = astart + i - 1
        if reachable[v]
            push!(Sidx, i)
        else
            push!(Ridx, i)
        end
    end

    Uidx = Int[]
    Vidx = Int[]
    @inbounds for j in 1:N
        v = bstart + j - 1
        if reachable[v]
            push!(Vidx, j)
        else
            push!(Uidx, j)
        end
    end

    return (
        cut_value = cut_value,
        Ridx = Ridx,
        Sidx = Sidx,
        Uidx = Uidx,
        Vidx = Vidx,
    )
end