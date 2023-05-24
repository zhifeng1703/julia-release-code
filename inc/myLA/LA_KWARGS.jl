include(homedir() * "/Documents/julia/inc/debug.jl")

using LinearAlgebra

KWARGS_LA = [1e-10, 1e-4];

LA_ABSTOL_ = KWARGS_LA[1];
LA_RELTOL_ = KWARGS_LA[2];