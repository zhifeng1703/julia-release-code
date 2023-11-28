include("../global_path.jl")
include(joinpath(JULIA_INCLUDE_PATH, "workspace.jl"))
include(joinpath(JULIA_LAPACK_PATH, "LAPACK_setup.jl"))

function get_wsp_zgemm!(m::Int, n::Int, k::Int)
    # Allocated objects:
    # JOBVS, SORTS, (SELECT), N, {A}, LDA, SDIM, WR, WI, {VS}, LDVS, WORK, LWORK, (BWORK), INFO
    transA::Cuchar = 'V'
    transB::Cuchar = 'S'
    mm::lapack_int = m
    nn::lapack_int = n
    kk::lapack_int = k

    lda::lapack_int = n
    sdim::lapack_int = n
    vR::Vector{lapack_double} = Vector{lapack_double}(undef, n)
    vI::Vector{lapack_double} = Vector{lapack_double}(undef, n)
    ldvs::lapack_int = n
    mW::Matrix{lapack_double} = Matrix{lapack_double}(undef, n, n)
    lw::lapack_int = length(mW)
    vB::Vector{lapack_logical} = Vector{lapack_logical}(undef, n)
    info::lapack_int = 0
    return WSP(jobvs, sorts, n32, lda, sdim, vR, vI, ldvs, mW, lw, vB, info)
end

function zgemm!(A::Ref{Matrix{ComplexF64}}, B::Ref{Matrix{ComplexF64}}, C::Ref{Matrix{ComplexF64}}, wsp::WSP; job='V', sort='S')



    wsp[1] = Cuchar(job)
    wsp[2] = Cuchar(sort)

    if BLAS_LOADED_LIB_IND == 1
        @ccall BLAS_LOADED_LIB.:dgees64_(wsp.vec[1]::Ptr{Cuchar}, wsp.vec[2]::Ptr{Cuchar},
            alg::Ptr{Cvoid},
            wsp.vec[3]::Ptr{lapack_int}, M[]::Ptr{lapack_double}, wsp.vec[4]::Ptr{lapack_int},
            wsp.vec[5]::Ptr{lapack_int},
            wsp.vec[6][]::Ptr{lapack_double}, wsp.vec[7][]::Ptr{lapack_double},
            P[]::Ptr{lapack_double}, wsp.vec[8]::Ptr{lapack_int},
            wsp.vec[9][]::Ptr{lapack_double}, wsp.vec[10]::Ptr{lapack_int},
            wsp.vec[11][]::Ptr{lapack_logical},
            wsp.vec[12]::Ptr{lapack_int})::Cvoid
    elseif BLAS_LOADED_LIB_IND == 2
        @ccall BLAS_LOADED_LIB.:dgees_(wsp.vec[1]::Ptr{Cuchar}, wsp.vec[2]::Ptr{Cuchar},
            alg::Ptr{Cvoid},
            wsp.vec[3]::Ptr{lapack_int}, M[]::Ptr{lapack_double}, wsp.vec[4]::Ptr{lapack_int},
            wsp.vec[5]::Ptr{lapack_int},
            wsp.vec[6][]::Ptr{lapack_double}, wsp.vec[7][]::Ptr{lapack_double},
            P[]::Ptr{lapack_double}, wsp.vec[8]::Ptr{lapack_int},
            wsp.vec[9][]::Ptr{lapack_double}, wsp.vec[10]::Ptr{lapack_int},
            wsp.vec[11][]::Ptr{lapack_logical},
            wsp.vec[12]::Ptr{lapack_int})::Cvoid
    end
    # display(M[]);

    # println("------------------------Finish DGEES------------------------")


    # for ind in eachindex(wsp.vec)
    #     display(wsp.vec[ind]);
    # end
    # println("------------------------Finish DGEES------------------------")

end