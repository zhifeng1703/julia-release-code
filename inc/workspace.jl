# There is no boundary check in this object. User are supposed to check before accessing WSP contents.
# WSP collect a set of preallocated memories of arbitrary type. The references cannot be changed.

import Base: getindex, setindex!

mutable struct WSP
    # wsp: the type that stores references of workspace 
    vec::Vector{Any}
    function WSP(c...)
        n = length(c)
        if n == 1 && typeof(c[1]) <: Int
            vec = Vector{Any}(undef, c)
        else
            vec = Vector{Any}(undef, n)
            for ii = 1:n
                vec[ii] = Ref(c[ii])
            end
        end
        return new(vec)
    end
end

(workspace::WSP)(i::Int) = @inbounds workspace.vec[i];
getindex(workspace::WSP, i) = @inbounds workspace.vec[i][];
setindex!(workspace::WSP, val, i) = workspace.vec[i][] = val;



retrieve(workspace::WSP, i) = workspace.vec[i][]
safe_access(workspace::WSP, i) = workspace.vec[i][]


