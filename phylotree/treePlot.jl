using Plots,LaTeXStrings

include("treeObj.jl")
include("bipartComp.jl")


pyplot()
# plotlyjs()

_cov_max(x, y) = min(x, y) - x*y
_cov_min(x, y) = max(0, x+y-1) - x*y

_cov_max_same(x) = _cov_max(x, x)
_cov_min_same(x) = _cov_min(x, x)

function plot2d_curves_with_region(f, g, plot_title)
    # Create x values
    x = LinRange(0, 1, 100)

    # Compute y values for both functions
    y_f = f.(x)
    y_g = g.(x)

    # Create the plot
    p = plot(x, y_f, label="Maximum", linewidth=2, color=:blue)
    plot!(x, y_g, label="Minimum", linewidth=2, color=:red)

    # Color the region between the curves
    fillrange = min.(y_f, y_g)
    plot!(x, y_f, fillrange=fillrange, fillalpha=0.3, label="Possible values", color=:gray)

    # Set the plot labels and title
    xlabel!("Frequency "*L"f_x = f_y = p")
    ylabel!("Covariance "*L"f_{x, y} - f_x \cdot f_y")
    title!(plot_title)

    return p
end


function plot3d_surfaces_with_region(f, g, plot_title; pts = 200)
    # Define the grid for x and y
    x = LinRange(0, 1, pts)
    y = LinRange(0, 1, pts)
    
    # Generate the meshgrid
    X, Y = [x[i] for i in 1:length(x), j in 1:length(y)], [y[j] for i in 1:length(x), j in 1:length(y)]

    # Compute function values
    Z_f = f.(X, Y)
    Z_g = g.(X, Y)

    # Create the plot
    p = surface(X, Y, Z_g, label="Minimum",fillalpha = 0.5)  # Semi-transparent surface
    surface!(X, Y, Z_f, label="Maximum",fillalpha = 0.5)  # Semi-transparent surface

    # Fill the region between using a transparent middle surface
    #Z_fill = min.(Z_f, Z_g)  # The lowest boundary
    #surface!(X, Y, Z_fill, label="Possible values", color=:gray, alpha=0.3)

    # Set labels and title
    xlabel!(L"f_x")
    ylabel!(L"f_y")
    zlabel!("Covariance Range")
    title!(plot_title)

    return p
end


struct GraphTreeEdge
    label::BITSTR_TYPE
    weight::Float64
    outter::Vector{GraphTreeEdge}
    GraphTreeEdge(l, w, o) = new(l, w, o)
end

struct GraphTree
    left::GraphTreeEdge
    right::GraphTreeEdge
    GraphTree(l, r) = new(l, r)
end

function _build_weight_map(tree::PhyloTree)
    weights::Dict{BITSTR_TYPE, Float64} = Dict{BITSTR_TYPE, Float64}()
    for ind in eachindex(tree.ib)
        setindex!(weights, tree.iw[ind], tree.ib[ind].upper)
        setindex!(weights, tree.iw[ind], tree.ib[ind].lower)
    end

    for ind in eachindex(tree.lw)
        leafbitstr = BITSTR_TYPE(1) << (ind - 1)
        setindex!(weights, tree.lw[ind], leafbitstr)
    end
    return weights
end

function _build_biparts(tree::PhyloTree, leaf::BITSTR_TYPE)
    biparts=Bipart[]

    for b in tree.ib
        push!(biparts, b)
    end

    for ind in eachindex(tree.lw)
        leafbitstr = BITSTR_TYPE(1) << (ind - 1)
        push!(biparts, Bipart(leafbitstr, leaf))
    end
    
    return biparts
end

function _build_biparts(VecBipart::Vector{Bipart})
    compatible = _bipart_comp(VecBipart)

    if !compatible
        display(VecBipart)
        throw("Error! Incompatible bipartitions encountered!")
    end

    leaf::BITSTR_TYPE = VecBipart[1].lower | VecBipart[1].upper

    biparts=Bipart[]

    for b in VecBipart
        push!(biparts, b)
    end

    temp_leaf::BITSTR_TYPE = leaf
    ind::Int = 1
    while temp_leaf != 0
        if temp_leaf % 2 == 1
            leafbitstr = BITSTR_TYPE(1) << (ind - 1)
            push!(biparts, Bipart(leafbitstr, leaf))
        end
        temp_leaf = temp_leaf >> 1
        ind += 1
    end
    
    return biparts
end

function Compute_GraphTreeEdge(label::BITSTR_TYPE, biparts::Vector{Bipart}, weights::Dict{BITSTR_TYPE, Float64})

    # find the smallest combinations (a,b,c...), in terms of the counts of BITSTR_TYPE, from biparts, such that label = a | b | c 
    # (trivial label = label should be excluded)
    # Construct the current edge with label and the outter be the vector of edges by the bipartitions who hold a, b and c.
    # The outter is not specified. therefore, this routine executes recursively to construct them so that the current edge can be declared. 
    # For example, let y = {a, a'} be the next recursive call, then it will be Compute_GraphTreeEdge(a, label, biparts, weights)

    
    if count_ones(label) == 1
        return GraphTreeEdge(label, get(weights, label, 1.0), GraphTreeEdge[])
    end


    valid_bitstrs = BITSTR_TYPE[]
    for b in biparts
        if b.lower | label == label && b.lower != label
            push!(valid_bitstrs, b.lower)
        elseif b.upper | label == label && b.upper != label
            push!(valid_bitstrs, b.upper)
        end
    end

    bad_indices = Int[]
    for (i, bstr) in enumerate(valid_bitstrs)
        for (j, cstr) in enumerate(valid_bitstrs)
            if (bstr | cstr == bstr) && bstr != cstr && j ∉ bad_indices
                # display((bstr, cstr))
                push!(bad_indices, j)
            elseif (bstr | cstr == cstr) && bstr != cstr && i ∉ bad_indices
                # display((bstr, cstr))
                push!(bad_indices, i)
            end
        end
    end
    bad_indices = sort(bad_indices)
    # display(bad_indices)
    deleteat!(valid_bitstrs, bad_indices)

    current_outter = GraphTreeEdge[]
    for b in valid_bitstrs
        push!(current_outter, Compute_GraphTreeEdge(b, biparts, weights))
    end

    sorted_outter = sort(current_outter, by = x -> count_ones(x.label))

    current_weight = get(weights, label, 0.0)
    return GraphTreeEdge(label, current_weight, sorted_outter)
end

function Compute_GraphTree(bipart::Bipart, tree::PhyloTree, leaf::BITSTR_TYPE)
    biparts = _build_biparts(tree, leaf)
    weights = _build_weight_map(tree)

    left = Compute_GraphTreeEdge(bipart.upper, biparts, weights)
    right = Compute_GraphTreeEdge(bipart.lower, biparts, weights)

    return GraphTree(left, right)
end

function Compute_GraphTree(bipart::Bipart, VecBipart::Vector{Bipart})
    biparts = _build_biparts(VecBipart)
    weights::Dict{BITSTR_TYPE, Float64} = Dict{BITSTR_TYPE, Float64}()

    left = Compute_GraphTreeEdge(bipart.upper, biparts, weights)
    right = Compute_GraphTreeEdge(bipart.lower, biparts, weights)

    return GraphTree(left, right)
end

function _balance_bipartition(VecBipart::Vector{Bipart})
    diff = abs(count_ones(VecBipart[1].upper) - count_ones(VecBipart[1].lower))
    bipart = VecBipart[1];
    for b in VecBipart
        curr_diff = abs(count_ones(b.upper) - count_ones(b.lower))
        if curr_diff < diff
            diff = curr_diff
            bipart = b
        end
    end
    return bipart
end

_balance_bipartition(tree::PhyloTree) = _balance_bipartition(tree.ib)

Compute_GraphTree(tree::PhyloTree, leaf::BITSTR_TYPE) = Compute_GraphTree(_balance_bipartition(tree), tree, leaf)

Compute_GraphTree(VecBipart::Vector{Bipart}) = Compute_GraphTree(_balance_bipartition(VecBipart), VecBipart)

# leafset = BITSTR_TYPE(31)
# biparts = [Bipart(BITSTR_TYPE(5), leafset), Bipart(BITSTR_TYPE(7), leafset)]
# innerws = [0.5,0.8]
# leafws = [0.6,0.6,0.6,0.6,0.6]
# tree = PhyloTree(biparts, innerws, leafws)

function _draw_tree_depth(edge::GraphTreeEdge, cur_depth::Int, max_depth::Ref{Int})
    cur_depth += 1;
    # println("Entering an edge at depth $(cur_depth), the current maximum depth is $(max_depth[])")

    if max_depth[] < cur_depth
        max_depth[] = cur_depth
    end

    for e in edge.outter
        _draw_tree_depth(e, cur_depth, max_depth)
    end
    cur_depth -= 1;
end

function _draw_tree_depth(rootedge::GraphTreeEdge)
    max_depth::Int = 0
    cur_depth::Int = 0
    max_depth_ref = Ref(max_depth)

    for e in rootedge.outter
        _draw_tree_depth(e, cur_depth, max_depth_ref)
    end

    return max_depth_ref[]
end

_draw_tree_polar_coordinate(radius::Float64, theta::Float64) = radius * cos(theta), radius * sin(theta)

function _draw_tree(canvas, curr_pos::Tuple{Float64, Float64}, edge::GraphTreeEdge, radius_level::Float64, theta_lb::Float64, theta_ub::Float64;
    color_map = Dict{BITSTR_TYPE, Symbol}(), weight_map::Function = (x -> 1.0))
    leaf_cnt = count_ones(edge.label)

    tlb = theta_lb
    tub = theta_ub

    tlevel = (tub - tlb) / leaf_cnt

    # println("Current point: ($(curr_pos[1]), $(curr_pos[2])), \tAngles Range: $(tlb)~$(tlevel)~$(tub)")

    # display(edge.outter)

    for e in edge.outter
        if count_ones(e.label) == 1
            pt = _draw_tree_polar_coordinate(1.0, tlb + tlevel / 2)
            scatter!(pt)

            # display(pt)

            plot!([curr_pos[1], pt[1]], [curr_pos[2], pt[2]], linewidth = weight_map(e.weight), color=color=get(color_map, e.label, :black))
            tlb += tlevel
        else
            cnt = count_ones(e.label)
            depth = _draw_tree_depth(e)
            pt = _draw_tree_polar_coordinate(1.0 - radius_level * depth, tlb + cnt * tlevel / 2)
            # display((depth, 1.0 - radius_level * depth, pt))
            scatter!(pt)
            plot!([curr_pos[1], pt[1]], [curr_pos[2], pt[2]], linewidth = weight_map(e.weight), color=color=get(color_map, e.label, :black))
            canvas = _draw_tree(canvas, pt, e, radius_level, tlb, tlb + cnt * tlevel; weight_map = weight_map, color_map = color_map)
            tlb += cnt * tlevel
        end
    end

    return canvas
end

function draw_tree(gt::GraphTree; color_map = Dict{BITSTR_TYPE, Symbol}(), weight_map::Function = (x -> 1.0))
    canvas = plot(size=(800,800), 
        xlims=(-1.1,1.1), ylims=(-1.1, 1.1), 
        framestyle=:none,
        legend=:none,
        aspect_ratio=1
    )

    left_depth = _draw_tree_depth(gt.left)
    left_rad_level = 1.0 / (left_depth + 1)
    left_pos = _draw_tree_polar_coordinate(left_rad_level, 1.0π)
    # display((left_depth, left_rad_level, left_pos))
    scatter!(left_pos)
    if left_depth != 0
        canvas = _draw_tree(canvas, left_pos, gt.left, left_rad_level, 0.6π, 1.4π; color_map = color_map, weight_map = weight_map)
    end

    right_depth = _draw_tree_depth(gt.right)
    right_rad_level = 1.0 / (right_depth + 1)
    right_pos = _draw_tree_polar_coordinate(right_rad_level, 0.0)
    # display((right_depth, right_rad_level, right_pos))
    scatter!(right_pos)
    if right_depth != 0
        canvas = _draw_tree(canvas, right_pos, gt.right, right_rad_level, -0.4π, 0.4π; color_map = color_map, weight_map = weight_map)
    end
    plot!([left_pos[1], right_pos[1]], [left_pos[2], right_pos[2]], linewidth = weight_map(gt.left.weight), color=get(color_map, gt.left.label, :black))
    return canvas
end

draw_tree(tree::PhyloTree; color_map = Dict{BITSTR_TYPE, Symbol}(), weight_map::Function = (x -> 1.0)) = draw_tree(Compute_GraphTree(tree, tree.ib[1].lower|tree.ib[1].upper); color_map = color_map, weight_map = weight_map)

function _weight_scaler(data::AbstractVector{<:Real})
    # Extract the values that are not default (i.e. less than 1.0)
    small_vals = filter(x -> x < 1.0, data)
    
    # If no small values exist, return a constant function
    if isempty(small_vals)
        return x -> 2.0
    end

    # Compute the maximum among the small values
    max_small = maximum(small_vals)
    
    # Return a function that scales non-default values relative to max_small
    # and leaves default values (1.0) unchanged.
    return x -> (x == 1.0 ? 2.0 : 2 * x / max_small)
end

_weight_scaler(dict::Dict{BITSTR_TYPE, Float64}) = _weight_scaler(collect(values(dict)))
_weight_scaler(tree::PhyloTree) = _weight_scaler(collect(values(_build_weight_map(tree))))

function draw_tree(VecBipart::Vector{Bipart}; color_map = Dict{BITSTR_TYPE, Symbol}(), weight_map::Function = (x -> 1.0))
    gt = Compute_GraphTree(VecBipart)
    
    canvas = plot(size=(800,800), 
        xlims=(-1.1,1.1), ylims=(-1.1, 1.1), 
        framestyle=:none,
        legend=:none,
        aspect_ratio=1
    )

    left_depth = _draw_tree_depth(gt.left)
    left_rad_level = 1.0 / (left_depth + 1)
    left_pos = _draw_tree_polar_coordinate(left_rad_level, 1.0π)
    # display((left_depth, left_rad_level, left_pos))
    scatter!(left_pos)
    if left_depth != 0
        canvas = _draw_tree(canvas, left_pos, gt.left, left_rad_level, 0.5π, 1.5π; color_map = color_map, weight_map = weight_map)
    end

    right_depth = _draw_tree_depth(gt.right)
    right_rad_level = 1.0 / (right_depth + 1)
    right_pos = _draw_tree_polar_coordinate(right_rad_level, 0.0)
    # display((right_depth, right_rad_level, right_pos))
    scatter!(right_pos)
    if right_depth != 0
        canvas = _draw_tree(canvas, right_pos, gt.right, right_rad_level, -0.5π, 0.5π; color_map = color_map, weight_map = weight_map)
    end
    plot!([left_pos[1], right_pos[1]], [left_pos[2], right_pos[2]],  linewidth = weight_map(gt.left.weight), color=get(color_map, gt.left.label, :black))
    return canvas
end