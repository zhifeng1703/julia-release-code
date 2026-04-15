include("treeDriver.jl")
include("treeGeoCompare.jl")

catTree, catTaxa = read_trees("test/allCatTrees.tre")

cov = bipartition_covariance(catTree)
write_lower_triangle("test/covariance.out", cov.covariance)

#dist_rf = rf_distance_matrix(catTree)
#write_lower_triangle("test/rf_distance.out", dist_rf)

cat_out = test_geodesic_matrix(catTree[1:50], catTree[51:100]; abstol=1e-6, reltol=5e-2, nrepeat=10)

display(cat_out.plt_count)
display(cat_out.plt_cost)


mammalTree, mammalTaxa = read_trees("test/mammalTrees.tre")
mammal_out = test_geodesic_matrix(mammalTree[1:50], mammalTree[51:100]; abstol=1e-6, reltol=5e-2, nrepeat=10)

display(mammal_out.plt_count)
display(mammal_out.plt_cost)