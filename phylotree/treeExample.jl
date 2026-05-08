include("treeDriver.jl")
#include("treeGeoCompare.jl")
include("treeGraph.jl")

catTree, catTaxa = read_trees("test/allCatTrees.tre")

cov = bipartition_covariance(catTree)
write_lower_triangle("test/covariance.out", cov.covariance)

initial, shared, inc = geodesic_initial(catTree[13], catTree[69])
path, info = refine_support(
    initial, shared, inc;
    abstol=0.0,
    reltol=0.0,
)

plt = plot_support_path(path, shared)
display(plt)


Wt = geodesic_tree_at(path, shared, 0.35)

root = first(shared.C)              # or choose a specific stable root
gt = Compute_GraphTree(collect(keys(Wt)), root)

plt = draw_tree(
    gt,
    root;
    weights=Wt,
    default_length=1.0,
    gap_ratio=0.05,
)

display(plt)

#dist_rf = rf_distance_matrix(catTree)
#write_lower_triangle("test/rf_distance.out", dist_rf)

#cat_out = test_geodesic_matrix(catTree[1:50], catTree[51:100]; abstol=0, reltol=5e-2, nrepeat=10)

#display(cat_out.plt_count)
#display(cat_out.plt_cost)

#plots_13_19 = plot_exact_geodesic_with_tol_skip(
#    catTree[13],
#    catTree[50 + 19];
#    abstol=0,
#    reltol=5e-2,
#)

#for p in plots_13_19
#    display(p)
#end


geodata = geodesic_data(catTree[13], catTree[69])

geodesic = refine_support(
    geodata.c, geodata.a, geodata.b, geodata.edgea, geodata.edgeb, geodata.inc;
    abstol=0.0,
    reltol=0.0,
)

supp_pairs = collect(zip(geodesic.supp_a, geodesic.supp_b))

clades, weights = geodesic_tree_at(0.35, merge(geodata, geodesic), supp_pairs)

p = draw_tree(clades)


# mammalTree, mammalTaxa = read_trees("test/mammalTrees.tre")
# mammal_out = test_geodesic_matrix(mammalTree[1:50], mammalTree[51:100]; abstol=1e-6, reltol=5e-2, nrepeat=10)

# display(mammal_out.plt_count)
# display(mammal_out.plt_cost)