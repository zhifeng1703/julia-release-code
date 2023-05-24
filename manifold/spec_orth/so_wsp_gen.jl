using LinearAlgebra

include(homedir() * "/Documents/julia/inc/workspace.jl")
include(homedir() * "/Documents/julia/inc/LAPACK_type.jl")


BLAS_LOADED_LIB = BLAS.get_config().loaded_libs[1].libname;



# The logical function required in LAPACK implemented in MKL by cpp should be 
# returning Cint type (4 bytes), but the LAPACK LOGICAL VECTOR container for 
# storing and operating the boolean value should be Bool type (1 byte).


function get_wsp_log(n::Int, wsp_dgees::WSP)
    Mn::Matrix{Float64} = Matrix{Float64}(undef, n, n);
    return WSP(Mn, wsp_dgees);
end

function get_wsp_log(n::Int)
    Mn::Matrix{Float64} = Matrix{Float64}(undef, n, n);
    wsp_dgees = get_wsp_dgees(n);
    return WSP(Mn, wsp_dgees);
end

function get_wsp_exp(n::Int, wsp_dgees::WSP)
    nb::Int = div(n, 2);

    Mn_1::Matrix{Float64} = Matrix{Float64}(undef, n, n);
    Mn_2::Matrix{Float64} = Matrix{Float64}(undef, n, n);
    Vnb::Vector{Float64} = Vector{Float64}(undef, nb);

    return WSP(Mn_1, Mn_2, Vnb, wsp_dgees);
end

function get_wsp_exp(n::Int)
    nb::Int = div(n, 2);

    Mn_1::Matrix{Float64} = Matrix{Float64}(undef, n, n);
    Mn_2::Matrix{Float64} = Matrix{Float64}(undef, n, n);
    Vnb::Vector{Float64} = Vector{Float64}(undef, nb);
    wsp_dgees = get_wsp_dgees(n);

    return WSP(Mn_1, Mn_2, Vnb, wsp_dgees);
end