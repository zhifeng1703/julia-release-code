#module PhyloTreesAPI

using Printf
using LinearAlgebra
using DelimitedFiles

include("treeObj.jl")
include("bipartComp.jl")
include("treeGeodesic.jl")

#export read_trees,
#    tree_collection,
#    rf_distance,
#    rf_distance_matrix,
#    geodesic_distance_value,
#    geodesic_distance_matrix,
#    bipartition_covariance,
#    write_lower_triangle,
#    save_rf_distance,
#    save_geodesic_distance,
#    save_covariance

function read_trees(fname::AbstractString)
    treestr = String[]
    open(fname, "r") do io
        for line in eachline(io)
            s = strip(line)
            isempty(s) && continue
            push!(treestr, s)
        end
    end
    isempty(treestr) && throw(ArgumentError("empty tree file: $fname"))
    taxa = TaxonList(treestr[1])
    trees = [PhyloTree(extract_bipart(s, taxa; leaf_weight=true)...) for s in treestr]
    return trees, taxa
end

function tree_collection(fname::AbstractString)
    trees, taxa = read_trees(fname)
    biparts = Bipart[]
    bipart_index = Dict{Bipart,Int}()
    for t in trees
        for b in t.ib
            if !haskey(bipart_index, b)
                push!(biparts, b)
                bipart_index[b] = length(biparts)
            end
        end
    end
    return (trees=trees, taxa=taxa, biparts=biparts, bipart_index=bipart_index)
end

function rf_distance(treeA::PhyloTree, treeB::PhyloTree)
    setA = Set(treeA.ib)
    setB = Set(treeB.ib)
    return length(setdiff(setA, setB)) + length(setdiff(setB, setA))
end

function rf_distance_matrix(trees::Vector{PhyloTree})
    n = length(trees)
    D = zeros(Float64, n, n)
    for j in 1:n
        for i in j:n
            d = float(rf_distance(trees[i], trees[j]))
            D[i, j] = d
            D[j, i] = d
        end
    end
    return D
end

function geodesic_distance_value(treeA::PhyloTree, treeB::PhyloTree; abstol::Real=0.0, reltol::Real=0.0)
    d, _ = geodesic_distance(treeA, treeB; abstol=abstol, reltol=reltol)
    return d
end

function geodesic_distance_matrix(trees::Vector{PhyloTree}; abstol::Real=0.0, reltol::Real=0.0)
    n = length(trees)
    D = zeros(Float64, n, n)
    for j in 1:n
        for i in j:n
            d = geodesic_distance_value(trees[i], trees[j]; abstol=abstol, reltol=reltol)
            D[i, j] = d
            D[j, i] = d
        end
    end
    return D
end

function bipartition_covariance(trees::Vector{PhyloTree})
    biparts = Bipart[]
    bipart_index = Dict{Bipart,Int}()
    for t in trees
        for b in t.ib
            if !haskey(bipart_index, b)
                push!(biparts, b)
                bipart_index[b] = length(biparts)
            end
        end
    end
    sf, jf, cov = bipart_unweighted_cov(trees, bipart_index)
    comp = [_bipart_comp(a, b) for a in biparts, b in biparts]
    return (single_freq=sf, joint_freq=jf, covariance=cov, compatibility=comp,
        biparts=biparts, bipart_index=bipart_index)
end

function write_lower_triangle(io::IO, A::AbstractMatrix)
    n = size(A, 1)
    for i in 1:n
        for j in 1:i
            print(io, A[i, j])
            if j < i
                print(io, ' ')
            end
        end
        print(io, '\n')
    end
end

function write_lower_triangle(fname::AbstractString, A::AbstractMatrix)
    open(fname, "w") do io
        write_lower_triangle(io, A)
    end
end

function save_rf_distance(infile::AbstractString, outfile::AbstractString; digits::Int=16)
    trees, _ = read_trees(infile)
    D = rf_distance_matrix(trees)
    write_lower_triangle(outfile, D; digits=digits)
    return D
end

function save_geodesic_distance(infile::AbstractString, outfile::AbstractString;
    abstol::Real=0.0, reltol::Real=0.0, digits::Int=16)
    trees, _ = read_trees(infile)
    D = geodesic_distance_matrix(trees; abstol=abstol, reltol=reltol)
    write_lower_triangle(outfile, D; digits=digits)
    return D
end

function save_covariance(infile::AbstractString, outfile::AbstractString; digits::Int=16)
    trees, _ = read_trees(infile)
    out = bipartition_covariance(trees)
    write_lower_triangle(outfile, out.covariance; digits=digits)
    return out
end

#end