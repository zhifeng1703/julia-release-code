include("../global_path.jl")
include(joinpath(JULIA_INCLUDE_PATH, "workspace.jl"))
include(joinpath(JULIA_LAPACK_PATH, "LAPACK_setup.jl"))


function dgetrf!(A::Ref{Matrix{Float64}}, IPIV::Ref{Vector{Int64}}, m::Int, n::Int)
    MatA = A[];
    VecIPIV = IPIV[];

    mkl_layout::mkl_int = mkl_int(LAPACK_COL_MAJOR)
    lapack_m::lapack_int = lapack_int(m)
    lapack_n::lapack_int = lapack_int(n)
    lapack_ldA::lapack_int = lapack_int(size(MatA, 1))


    PtrA = pointer(MatA)
    PtrIPIV = pointer(VecIPIV)

    if BLAS_LOADED_LIB_IND == 1
        @ccall BLAS_LOADED_LIB.:dgetrf64_(
            Ref(lapack_m)::Ptr{lapack_int}, Ref(lapack_n)::Ptr{lapack_int},
            PtrA::Ptr{lapack_double}, Ref(lapack_ldA)::Ptr{lapack_int}, PtrIPIV::Ptr{lapack_int}
        )::Cvoid
    elseif BLAS_LOADED_LIB_IND == 2
        # @ccall BLAS_LOADED_LIB.:dgetrf(
        #     Ref(lapack_layout)::Ptr{mkl_int}, Ref(lapack_m)::Ptr{lapack_int}, Ref(lapack_n)::Ptr{lapack_int},
        #     PtrA::Ptr{lapack_double}, Ref(lapack_ldA)::Ptr{lapack_int}, PtrIPIV::Ptr{lapack_int}
        # )::Cvoid

        @ccall BLAS_LOADED_LIB.:LAPACKE_dgetrf(
            Ref(mkl_layout)::Ptr{mkl_int}, Ref(lapack_m)::Ptr{lapack_int}, Ref(lapack_n)::Ptr{lapack_int},
            PtrA::Ptr{lapack_double}, Ref(lapack_ldA)::Ptr{lapack_int}, PtrIPIV::Ptr{lapack_int}
        )::lapack_int
    else
        throw("Incorrect loaded BLAS library, loaded index $(BLAS_LOADED_LIB_IND)")
    end 
end

function dgetrf2!(A::Ref{Matrix{Float64}}, IPIV::Ref{Vector{lapack_int}}, m::Int, n::Int)
    MatA = A[];
    VecIPIV = IPIV[];

    lapack_layout::mkl_int = mkl_int(LAPACK_COL_MAJOR)
    lapack_m::lapack_int = lapack_int(m)
    lapack_n::lapack_int = lapack_int(n)
    lapack_ldA::lapack_int = lapack_int(size(MatA, 1))


    PtrA = pointer(MatA)
    PtrIPIV = pointer(VecIPIV)

    if BLAS_LOADED_LIB_IND == 1
        @ccall BLAS_LOADED_LIB.:dgetrf64_(
            Ref(lapack_layout)::Ptr{mkl_int}, Ref(lapack_m)::Ptr{lapack_int}, Ref(lapack_n)::Ptr{lapack_int},
            PtrA::Ptr{lapack_double}, Ref(lapack_ldA)::Ptr{lapack_int}, PtrIPIV::Ptr{lapack_int}
        )::Cvoid
    elseif BLAS_LOADED_LIB_IND == 2
        # @ccall BLAS_LOADED_LIB.:dgetrf(
        #     Ref(lapack_layout)::Ptr{mkl_int}, Ref(lapack_m)::Ptr{lapack_int}, Ref(lapack_n)::Ptr{lapack_int},
        #     PtrA::Ptr{lapack_double}, Ref(lapack_ldA)::Ptr{lapack_int}, PtrIPIV::Ptr{lapack_int}
        # )::Cvoid

        @ccall BLAS_LOADED_LIB.:dgetrf2_(
            Ref(lapack_m)::Ptr{lapack_int}, Ref(lapack_n)::Ptr{lapack_int},
            PtrA::Ptr{lapack_double}, Ref(lapack_ldA)::Ptr{lapack_int}, PtrIPIV::Ptr{lapack_int}
        )::Cvoid
    else
        throw("Incorrect loaded BLAS library, loaded index $(BLAS_LOADED_LIB_IND)")
    end 
end