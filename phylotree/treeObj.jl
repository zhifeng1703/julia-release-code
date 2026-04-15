using DelimitedFiles

# include("covariance_plot.jl")

BITSTR = 64

if BITSTR == 32
    BITSTR_TYPE = UInt32
    BITSTR_SIZE = 32
elseif BITSTR == 64
    BITSTR_TYPE = UInt64
    BITSTR_SIZE = 64
elseif BITSTR == 128
    BITSTR_TYPE = UInt128
    BITSTR_SIZE = 128
else
    BITSTR_TYPE = UInt16
    BITSTR_SIZE = 16
end

struct TaxonList
    n::Int
    leaf::BITSTR_TYPE
    vect::Vector{String}
    dict::Dict{String, BITSTR_TYPE}
    # This struct is only used for trees that share the same leaf set.
    #   n records the size of the leaf set;
    #   leaf records the bitstring of the leaf set;
    #   vect records each leaf label in a vector,                   for the map: index ↦ label;
    #   dict records the index of each leaf label in a dictionary,  for the map: label ↦ index;
    function TaxonList(str::String)
        vect = extract_taxon(str)
        dict = Dict(x => BITSTR_TYPE(i) for (i, x) in enumerate(vect))
        if length(vect) == BITSTR
            leaf = typemax(BITSTR_TYPE)
        else
            leaf = (BITSTR_TYPE(1) << length(vect)) - 1
        end
        return new(length(vect), leaf, vect, dict)
    end
end


struct Bipart
    lower::BITSTR_TYPE
    upper::BITSTR_TYPE

    function Bipart(a::BITSTR_TYPE, l::BITSTR_TYPE)
        b = l - a
        if a > b
            return new(b, a)
        else
            return new(a, b)
        end
    end
    
    function Bipart(str::String, taxa::TaxonList)
        sub_taxa = extract_taxon(str);
        a = BITSTR_TYPE(0)
        for taxon in sub_taxa
            offset = taxa.dict[taxon] - 1
            a += BITSTR_TYPE(1) << offset
        end
        b = taxa.leaf - a
        if a > b
            return new(b, a)
        else
            return new(a, b)
        end
    end
end

struct PhyloTree
    ib::Vector{Bipart}
    iw::Vector{Float64}
    lw::Vector{Float64}

    PhyloTree(ib, iw, lw) = new(ib, iw, lw)
end

import Base: in

# Define membership for MyCollection
function in(x::BITSTR_TYPE, tree::PhyloTree)
    for b in tree.ib
        if x == b.lower || x == b.upper
            return true
        end
    end
    return false
end

in(x::Bipart, tree::PhyloTree) = in(x.lower, tree)


#struct PhyloTree
#    VecBipart::Vector{Bipart}
#    VecWeight::Vector{Float64}
#    LeafWeight::Vector{Float64}
#    LeafSet::BITSTR_TYPE
#    PhyloTree(vb, vw, lw, ls) = new(vb, vw, lw, ls)
#    function PhyloTree(str::String, taxa::TaxonList; leaf_weight::Bool = false)

#    end
#end

# Alternatively, the bitstring can be implemented by the BitArray type, but extra care is needed for the 

terminate_findnext(ch, string, start::Int) = (found = findnext(ch, string, start)) === nothing ? length(string) + 1 : found[1]

function extract_taxon(str::String)
    # Regex patterns to match substrings
    #between_parentheses = r"\(([^(),:]*):"
    #between_commas = r",([^(),:]*):"

    ## ([^(),:]*) captures any substring that does not contain control characters (), ,, or :
    ## The character '(' needs to use \(, but ',' does not.

    #substrings_parentheses = [m[1] for m in eachmatch(between_parentheses, str)]
    #substrings_commas = [m[1] for m in eachmatch(between_commas, str)]

    taxon = String[]

    ctrl_sym = r"[(,:)]"

    idx_curr = 0
    ch_curr = '\0'

    idx_next = terminate_findnext(ctrl_sym, str, idx_curr+1)

    while idx_next < length(str)
        ch_next = str[idx_next]

        if ch_next == ':' && ch_curr != ')'
            push!(taxon, str[idx_curr+1:idx_next-1])
        end

        idx_curr = idx_next
        idx_next = terminate_findnext(ctrl_sym, str, idx_curr + 1)
        ch_curr = str[idx_curr]
    end

    return taxon
end

function extract_bipart(str::String, taxa::TaxonList; leaf_weight::Bool = false)
    bstr = String[]
    bipa = Bipart[]
    inte_weights = Float64[]
    stack = Int[]
    ctrl_sym = r"[(;:,)]"

    
    if leaf_weight
        leaf_weights = zeros(taxa.n)
        
        idx_curr = 0
        ch_curr = '\0'
    
        idx_next = terminate_findnext(ctrl_sym, str, idx_curr+1)

        while idx_next < length(str)
            ch_next = str[idx_next]
            #display((ch_curr,ch_next))


            if ch_next == ':' && ch_curr != ')'
                idx_weig = terminate_findnext(ctrl_sym, str, idx_next+1)
                taxon = str[idx_curr+1:idx_next-1]  
                w = parse(Float64, str[idx_next+1:idx_weig-1])
                leaf_weights[taxa.dict[taxon]] = w
            end
            idx_curr = idx_next
            idx_next = terminate_findnext(ctrl_sym, str, idx_curr + 1)
            ch_curr = str[idx_curr]
        end
    else
        leaf_weights = Float64[]
    end 
    
    idx = terminate_findnext(ctrl_sym, str, 1)

    while idx < length(str)
        c = str[idx]
        #display((idx, c))
        if c == '('
            push!(stack, idx)  # Record index of '('
            idx = terminate_findnext(ctrl_sym, str, idx + 1)    # Get the matching ')' with the weights behind it.
        elseif c == ')'
            if !isempty(stack)
                idx_beg = pop!(stack)                           # Get the matching '(' index
                idx_col = terminate_findnext(ctrl_sym, str, idx+1)           # Get the colon ':'
                if str[idx_col] == ';'
                    break;
                end
                idx_end = terminate_findnext(ctrl_sym, str, idx_col+1)       # Get the weight behind colon.
                taxon = str[idx_beg+1:idx-1]                                # Extract the string in between
                b = Bipart(taxon, taxa)
                if b != taxa.leaf && findfirst(x->x==b, bipa)===nothing
                    # display(b)
                    # println(str[idx_beg+1:idx-1])
                    w = parse(Float64, str[idx_col+1:idx_end-1])
                    push!(bipa, b)
                    push!(inte_weights, w)
                end
                idx = idx_end
            else
                break
            end
        else
            idx = terminate_findnext(ctrl_sym, str, idx + 1)    # Get the matching ')' with the weights behind it.
        end
    end
    # return bstr, bipa, inte_weights, leaf_weights

    return bipa, inte_weights, leaf_weights
end


function bipart_unweighted_cov(trees::Vector{PhyloTree}, dic_bipart::Dict{Bipart, Int64})
    tree_cnt = length(trees)
    bipart_cnt = length(dic_bipart)

    single_f = zeros(bipart_cnt)
    joint_f = zeros(bipart_cnt, bipart_cnt)
    cov = zeros(bipart_cnt, bipart_cnt)

    for t in trees
        for b in t.ib
            ind_b = dic_bipart[b]
            single_f[ind_b] += 1
            for c in t.ib
                joint_f[ind_b, dic_bipart[c]] += 1
            end
        end
    end
    single_f ./= tree_cnt
    joint_f ./= tree_cnt

    for col_ind = 1:bipart_cnt
        for row_ind = 1:bipart_cnt
            cov[row_ind, col_ind] = joint_f[row_ind, col_ind] - single_f[row_ind]*single_f[col_ind]
        end
    end
    return single_f,joint_f, cov
end

function create_cov_pts(sf::Vector{Float64}, cov::Matrix{Float64})

    cov_pts = zeros(sizeof(cov), 3)
    cov_ind = 1
    for row_ind in eachindex(sf)
        for col_ind in eachindex(sf)
            cov_pts[cov_ind, 1] = sf[row_ind]
            cov_pts[cov_ind, 2] = sf[col_ind]
            cov_pts[cov_ind, 3] = cov[row_ind, col_ind]
            cov_ind += 1
        end
    end
    return cov_pts
end


