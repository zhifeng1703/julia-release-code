include("treeDriver.jl")
include("treeGeoCompare.jl")
include("treeGraph.jl")

catTree, catTaxa = read_trees("test/allCatTrees.tre")

cov = bipartition_covariance(catTree)
write_lower_triangle("test/covariance.out", cov.covariance)

# initial, shared, inc = geodesic_initial(catTree[13], catTree[69]) 
# geodesic, info = refine_support(
#     initial, shared, inc;
#     abstol=0.0,
#     reltol=0.0,
# )# Geodesic computation

# geodesic_plots = path_snapshot(geodesic, shared, range(0,1,100);depth=1.5) #depth control the canvas size
# anim_geodesic = @animate for t in range(0,1,100)
#     path_snapshot(geodesic, shared, t;depth=1.5)
# end
# gif(anim_geodesic, "tree_geodesic.gif"; fps=10)

# path, info = refine_support(
#     initial, shared, inc;
#     abstol=0.0,
#     reltol=5e-2,
# )# Geodesic computation

# anim_path = @animate for t in range(0,1,100)
#     path_snapshot(path, shared, t;depth = 1.5)
# end
# gif(anim_path, "tree_path.gif"; fps=10)

compare_geodesic(catTree[13], catTree[69]; abstol=0.0, reltol=5e-2)


#dist_rf = rf_distance_matrix(catTree)
#write_lower_triangle("test/rf_distance.out", dist_rf)

cat_out = test_geodesic_matrix(catTree[1:50], catTree[51:100]; abstol=0, reltol=5e-2, nrepeat=10);


# mammalTree, mammalTaxa = read_trees("test/mammalTrees.tre")
# mammal_out = test_geodesic_matrix(mammalTree[1:50], mammalTree[51:100]; abstol=1e-6, reltol=5e-2, nrepeat=10)

# display(mammal_out.plt_count)
# display(mammal_out.plt_cost)