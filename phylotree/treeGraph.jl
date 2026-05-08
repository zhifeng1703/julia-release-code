using Colors, Plots

include("treeObj.jl")
include("treeGeodesic.jl")

struct GraphTree
    bipart::Bipart
    rside::Vector{Bipart}
    lside::Vector{Bipart}
end

_subset(a::BITSTR_TYPE, b::BITSTR_TYPE) = (a & b) == a

function _complete_leaf_edges(Biparts::Vector{Bipart}, leaf::BITSTR_TYPE)
    S = Set(Biparts)
    for i in 0:BITSTR_SIZE-1
        b = BITSTR_TYPE(1) << i
        (b & leaf) == 0 && continue
        push!(S, Bipart(b, leaf))
    end
    return collect(S)
end

function Compute_GraphTree(Biparts::Vector{Bipart}, root::Bipart)
    leaf = root.lower + root.upper
    root = Bipart(root.lower, leaf)
    Biparts = _complete_leaf_edges(Biparts, leaf)
    root in Biparts || push!(Biparts, root)

    others = [e for e in Biparts if e != root]

    rootside = Dict{Bipart,Int}()
    clade = Dict{Bipart,BITSTR_TYPE}()

    for e in others
        if _subset(e.lower, root.lower)
            rootside[e] = 1
            clade[e] = e.lower
        elseif _subset(e.upper, root.lower)
            rootside[e] = 1
            clade[e] = e.upper
        elseif _subset(e.lower, root.upper)
            rootside[e] = 2
            clade[e] = e.lower
        elseif _subset(e.upper, root.upper)
            rootside[e] = 2
            clade[e] = e.upper
        else
            error("Bipartition incompatible with root.")
        end
    end

    parent = Dict{Bipart,Bipart}()
    child = Dict(e => Bipart[] for e in Biparts)

    for e in others
        cand = [
            f for f in others
            if rootside[f] == rootside[e] &&
            clade[e] != clade[f] &&
            _subset(clade[e], clade[f])
        ]

        if isempty(cand)
            parent[e] = root
            push!(child[root], e)
        else
            p = cand[argmin([count_ones(clade[f]) for f in cand])]
            parent[e] = p
            push!(child[p], e)
        end
    end

    gt = Dict{Bipart,GraphTree}()

    gt[root] = GraphTree(
        root,
        [e for e in child[root] if rootside[e] == 1],
        [e for e in child[root] if rootside[e] == 2],
    )

    for e in others
        gt[e] = GraphTree(e, [parent[e]], child[e])
    end

    return gt
end



_getw(weights, e) = get(weights, e, 1.0)
_getc(colors, e) = get(colors, e, colorant"black")

function _leafcount(gt::Dict{Bipart,GraphTree}, e::Bipart, p::Bipart)
    cs = [x for x in gt[e].rside ∪ gt[e].lside if x != p]
    isempty(cs) && return 1
    return sum(_leafcount(gt, c, e) for c in cs)
end

function _depth(gt::Dict{Bipart,GraphTree}, e::Bipart, p::Bipart)
    cs = [x for x in gt[e].rside ∪ gt[e].lside if x != p]
    isempty(cs) && return 1
    return 1 + maximum(_depth(gt, c, e) for c in cs)
end

function _draw_branch!(plt, gt, cs, p0, x0, y0, a, b,
    weights, colors, default_length, gap_ratio)

    isempty(cs) && return

    sort!(cs, by=e -> (_depth(gt, e, p0), _leafcount(gt, e, p0), e.lower, e.upper))

    nleaf = [_leafcount(gt, e, p0) for e in cs]
    total = sum(nleaf)

    δ = gap_ratio * abs(b - a)
    s = sign(b - a)
    aa = a + s * δ
    bb = b - s * δ

    r0 = hypot(x0, y0)
    acc = 0

    for (e, J) in zip(cs, nleaf)
        t0 = aa + (acc / total) * (bb - aa)
        t1 = aa + ((acc + J) / total) * (bb - aa)
        t = (t0 + t1) / 2

        l = default_length * _getw(weights, e)
        x1 = (r0 + l) * cos(t)
        y1 = (r0 + l) * sin(t)

        plot!(plt, [x0, x1], [y0, y1], color=_getc(colors, e), lw=2)

        next_cs = [x for x in union(gt[e].rside, gt[e].lside) if x != p0]
        _draw_branch!(plt, gt, next_cs, e, x1, y1, t0, t1,
            weights, colors, default_length, gap_ratio)

        acc += J
    end
end

function draw_tree(
    gt::Dict{Bipart,GraphTree},
    root::Bipart;
    weights::Dict{Bipart,Float64}=Dict{Bipart,Float64}(),
    colors::Dict{Bipart,Colorant}=Dict{Bipart,Colorant}(),
    default_length::Float64=1.0,
    gap_ratio::Float64=0.05,
    size=(800, 800),)

    root = gt[root].bipart
    lroot = default_length * _getw(weights, root)

    xL, yL = -lroot / 2, 0.0
    xR, yR = lroot / 2, 0.0

    plt = plot(
        size=size,
        xlims=(-1.1, 1.1),
        ylims=(-1.1, 1.1),
        framestyle=:none,
        legend=:none,
        aspect_ratio=1,
    )

    plot!(plt, [xL, xR], [yL, yR], color=_getc(colors, root), lw=2)

    _draw_branch!(
        plt, gt, copy(gt[root].rside), root,
        xL, yL, π / 2, 3π / 2,
        weights, colors, default_length, gap_ratio,
    )

    _draw_branch!(
        plt, gt, copy(gt[root].lside), root,
        xR, yR, π / 2, -π / 2,
        weights, colors, default_length, gap_ratio,
    )

    return plt
end


function support_colors(K::Int)
    warm = [RGB(HSV(15 + 45 * (k - 1) / max(K - 1, 1), 0.75, 0.90)) for k in 1:K]
    cool = [RGB(HSV(200 + 60 * (k - 1) / max(K - 1, 1), 0.65, 0.90)) for k in 1:K]
    return warm, cool
end

function plot_support_path(
    path::Vector{SupportPair},
    shared::SharedPair;
    size=(850, 450),
    lw=3,
)
    K = length(path)
    warm, cool = support_colors(K)

    ymax = maximum(vcat(
        [norm(sp.WA) for sp in path],
        [norm(sp.WB) for sp in path],
        [norm(shared.W0)],
        [norm(shared.W1)],
        [1e-12],
    ))

    plt = plot(
        size=size,
        xlims=(0, 1),
        framestyle=:box,
        legend=:inside,
        legend_position=:topright,
        xlabel="t",
        ylabel="aggregate weight",
    )

    for k in 1:K
        tk = k / (K + 1)

        plot!(
            plt,
            [0, tk],
            [norm(path[k].WA), 0.0],
            color=warm[k],
            lw=lw,
            label="A$k",
        )
    end

    for k in 1:K
        tk = k / (K + 1)

        plot!(
            plt,
            [tk, 1],
            [0.0, norm(path[k].WB)],
            color=cool[k],
            lw=lw,
            label="B$k",
        )
    end

    plot!(
        plt,
        [0, 1],
        [norm(shared.W0), norm(shared.W1)],
        color=colorant"black",
        lw=lw,
        label="C",
    )

    return plt
end


function test_draw_graphtree()

    # leaf = BITSTR_TYPE(31)   # binary 11111 = leaves 1,2,3,4,5

    # root = Bipart(BITSTR_TYPE(0b00011), leaf)  # {1,2}|{3,4,5}
    # e45  = Bipart(BITSTR_TYPE(0b11000), leaf)  # {4,5}|{1,2,3}

    # Biparts = [root, e45]

    # gt = Compute_GraphTree(Biparts, root)

    # weights = Dict{Bipart,Float64}(
    #     root => 0.6,
    #     e45  => 0.5,
    # )

    # colors = Dict{Bipart,Symbol}(
    #     root => :black,
    #     e45  => :red,
    # )

    # plt = draw_tree(
    #     gt,
    #     root;
    #     weights=weights,
    #     colors=colors,
    #     default_length=1.0,
    #     gap_ratio=0.05,
    # )

    # display(plt)

    leaf = BITSTR_TYPE(2^10 - 1)

    root = Bipart(BITSTR_TYPE(0b0000011111), leaf)  # {1,2,3,4,5}|{6,7,8,9,10}

    e12 = Bipart(BITSTR_TYPE(0b0000000011), leaf)  # {1,2}
    e34 = Bipart(BITSTR_TYPE(0b0000001100), leaf)  # {3,4}
    e1234 = Bipart(BITSTR_TYPE(0b0000001111), leaf)  # {1,2,3,4}

    e67 = Bipart(BITSTR_TYPE(0b0001100000), leaf)  # {6,7}
    e89 = Bipart(BITSTR_TYPE(0b0110000000), leaf)  # {8,9}
    e6789 = Bipart(BITSTR_TYPE(0b0111100000), leaf)  # {6,7,8,9}

    Biparts = [root, e12, e34, e1234, e67, e89, e6789]

    gt = Compute_GraphTree(Biparts, root)

    weights = Dict{Bipart,Float64}(
        root => 0.45,
        e1234 => 0.35,
        e12 => 0.25,
        e34 => 0.25,
        e6789 => 0.35,
        e67 => 0.25,
        e89 => 0.25,
    )

    colors = Dict{Bipart,Colorant}(
        root => colorant"black",
        e1234 => RGB(0.9, 0.2, 0.2),
        e12 => RGB(0.2, 0.4, 0.9),
        e34 => HSV(120, 0.7, 0.8),
    )

    plt = draw_tree(
        gt,
        root;
        weights=weights,
        colors=colors,
        default_length=1.0,
        gap_ratio=0.05,
    )

    display(plt)
end



