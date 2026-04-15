using Combinatorics, StatsBase, Random
include("treeObj.jl")

function _bipart_comp(a::Bipart, b::Bipart) 
    if (a.upper | a.lower) != (b.upper | b.lower)
        return false
    elseif ((a.lower & b.lower) == a.lower) || ((b.lower & a.lower) == b.lower) || ((a.lower & b.upper) == a.lower)
        return true
    else
        return false
    end
end

function _bipart_comp(v::Vector{Bipart})
    for p in combinations(v, 2)
        if !_bipart_comp(p[1], p[2])
            return false
        end
    end
    return true
end

function _bipart_comp(v::Vector{Bipart}, a::Bipart)
    for b in v
        if !_bipart_comp(a, b)
            return false
        end
    end
    return true
end

function _bipart_comp(g::Vector{Int}, bipart::Vector{Bipart}, ind::Int)
    for i in g
        if !_bipart_comp(bipart[i], bipart[ind])
            return false
        end
    end
    return true
end

function _maximum_clade(v::Vector{Bipart})
    clade = Vector{Bipart}()
    push!(clade, v[1])
    for ind = 2:length(v)
        if _bipart_comp(clade, v[ind])
            push!(clade, v[ind])
        end
    end
    return clade
end 

function _build_clade(ind::Vector{Int}, bipart::Vector{Bipart})
    # println("Exhausting order:\t", ind);
    clade = Vector{Int}()
    push!(clade, ind[1])
    for i = 2:length(ind)
        if _bipart_comp(clade, bipart, ind[i])
            push!(clade, ind[i])
        end
    end
    return clade
end 

_same_clade(v1::Vector{Bipart}, v2::Vector{Bipart}) = (countmap(v1) == countmap(v2))
_same_clade(v1::Vector{Int}, v2::Vector{Int}) = (countmap(v1) == countmap(v2))

_found_max_clade(v1::Vector{Int}, max_clade::Vector{Vector{Int}}) = any(countmap(c) == countmap(v1[1:length(c)]) for c in max_clade)


function build_all_clade(ind::Vector{Int}, bipart::Vector{Bipart}; max_attempt = 1e6)
    clades = Vector{Vector{Int}}()
    count = 0
    if length(ind) < 8
        for indp in permutations(ind)
            if length(clades) == 0 || !_found_max_clade(indp, clades)
                clade = _build_clade(indp, bipart)
                if !any(countmap(c) == countmap(clade) for c in clades)
                    push!(clades, clade)
                end
            end
        end
    else
        for count = 1:max_attempt
            indp = shuffle(ind)
            if length(clades) == 0 || !_found_max_clade(indp, clades)
                clade = _build_clade(indp, bipart)
                if !any(countmap(c) == countmap(clade) for c in clades)
                    push!(clades, clade)
                end
            end
        end
    end
    # display(clades)
    return clades
end

function build_all_clade(v::Vector{Bipart}; max_attempt = 1e6)
    clades = Vector{Vector{Bipart}}()
    count = 0
    for vp in permutations(v)
        clade = _maximum_clade(vp)
        # display(clade)
        if !any(_same_clade(clade, c) for c in clades)
            push!(clades, clade)
        end
        count += 1
        if count > 1e8
            break
        end
    end
    return clades
end