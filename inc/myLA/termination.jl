include("../debug.jl")

struct Terminator
    MaxIter
    MaxTime
    AbsTol
    RelTol
    Terminator(i, t, a, r) = new(i, t, a, r)
    Terminator(; i=nothing, t=nothing, a=nothing, r=nothing) = new(i, t, a, r)
end

function terminate(time, iter, abserr, relerr, Stop)
    if !isnothing(time) && !isnothing(Stop.MaxTime)
        if time > Stop.MaxTime
            d_msg("Terminate for exceeding maximum compute time.\n")
            return true
        end
    end

    if !isnothing(iter) && !isnothing(Stop.MaxIter)
        if iter > Stop.MaxIter
            d_msg("Terminate for exceeding maximum iteration.\n")
            return true
        end
    end

    if !isnothing(abserr) && !isnothing(Stop.AbsTol)
        if abserr < Stop.AbsTol
            return true
        end
    end

    if !isnothing(relerr) && !isnothing(Stop.RelTol)
        if relerr < Stop.RelTol
            return true
        end
    end

    return false

end