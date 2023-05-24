# This is the julia API for dgees, the real schur decomposition in double precision.
include("../global_path.jl")
include(joinpath(JULIA_INCLUDE_PATH, "workspace.jl"))
include(joinpath(JULIA_LAPACK_PATH, "LAPACK_setup.jl"))


function get_wsp_dgesvd(n_row::Int, n_col::Int)
    # Allocated objects:
    # {jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt,} work, lwork, info
    lw::lapack_int = max(8, n_row) * max(8, n_col);
    vW::Vector{lapack_double} = Vector{lapack_double}(undef, lw);
    return WSP(vW)
end

function dgesvd!(A::Ref{Matrix{Float64}}, m::Int, n::Int, aos::Int, S::Ref{Vector{Float64}}, U::Ref{Matrix{Float64}}, uos::Int, V::Ref{Matrix{Float64}}, vos::Int, jobu::Char, jobvt::Char, wsp_dgesvd::WSP = get_wsp_dgesvd(m, n))
    MatA = A[];
    MatU = U[];
    MatV = V[];
    VecS = S[];
    VecW = wsp_dgesvd[1];

    UcharJobu = Cuchar(jobu)
    UcharJobvt = Cuchar(jobvt)

    lapack_m::lapack_int = lapack_int(m)
    lapack_n::lapack_int = lapack_int(n)
    lapack_ldA::lapack_int = lapack_int(size(MatA, 1))
    lapack_ldU::lapack_int = lapack_int(size(MatU, 1))
    lapack_ldV::lapack_int = lapack_int(size(MatV, 1))
    lapack_ldW::lapack_int = lapack_int(length(VecW))
    lapack_info::lapack_int = lapack_int(0)



    PtrA = pointer(MatA) + aos * sizeof(Float64)
    PtrU = pointer(MatU) + uos * sizeof(Float64)
    PtrV = pointer(MatV) + vos * sizeof(Float64)

    if BLAS_LOADED_LIB_IND == 1
        @ccall BLAS_LOADED_LIB.:dgesvd64_(
            Ref(UcharJobu)::Ptr{Cuchar}, Ref(UcharJobvt)::Ptr{Cuchar}, Ref(lapack_m)::Ptr{lapack_int}, Ref(lapack_n)::Ptr{lapack_int},
            PtrA::Ptr{lapack_double}, Ref(lapack_ldA)::Ptr{lapack_int}, VecS::Ptr{lapack_double},
            PtrU::Ptr{lapack_double}, Ref(lapack_ldU)::Ptr{lapack_int}, PtrV::Ptr{lapack_double}, Ref(lapack_ldV)::Ptr{lapack_int},
            wsp_dgesvd(1)::Ptr{lapack_double}, Ref(lapack_ldW)::Ptr{lapack_int}, Ref(lapack_info)::Ptr{lapack_int}
        )::Cvoid
    elseif BLAS_LOADED_LIB_IND == 2
        @ccall BLAS_LOADED_LIB.:dgesvd_(
            Ref(UcharJobu)::Ptr{Cuchar}, Ref(UcharJobvt)::Ptr{Cuchar}, Ref(lapack_m)::Ptr{lapack_int}, Ref(lapack_n)::Ptr{lapack_int},
            PtrA::Ptr{lapack_double}, Ref(lapack_ldA)::Ptr{lapack_int}, VecS::Ptr{lapack_double},
            PtrU::Ptr{lapack_double}, Ref(lapack_ldU)::Ptr{lapack_int}, PtrV::Ptr{lapack_double}, Ref(lapack_ldV)::Ptr{lapack_int},
            VecW::Ptr{lapack_double}, Ref(lapack_ldW)::Ptr{lapack_int}, Ref(lapack_info)::Ptr{lapack_int}
        )::Cvoid
    else
        throw("Incorrect loaded BLAS library, loaded index $(BLAS_LOADED_LIB_IND)")
    end 
end

function dgesvd!(A::Ref{Matrix{Float64}}, m::Int, n::Int, aos::Int, S::Ref{Vector{Float64}}, P::Ref{Matrix{Float64}}, pos::Int, jobp::Char, job::Char, wsp_dgesvd::WSP = get_wsp_dgesvd(m, n))
    MatA = A[];
    MatP = P[];

    VecS = S[];
    VecW = wsp_dgesvd[1];

    UcharJobu = Cuchar('N')
    UcharJobvt = Cuchar('N')

    lapack_m::lapack_int = lapack_int(m)
    lapack_n::lapack_int = lapack_int(n)
    lapack_ldA::lapack_int = lapack_int(size(MatA, 1))
    lapack_ldU::lapack_int = lapack_int(1)
    lapack_ldV::lapack_int = lapack_int(1)
    lapack_ldW::lapack_int = lapack_int(length(VecW))
    lapack_info::lapack_int = lapack_int(0)

    if jobp == 'U'
        UcharJobu = Cuchar(job)
        PtrU = pointer(MatP) + pos * sizeof(Float64)
        lapack_ldU = lapack_int(size(MatP, 1))
        PtrV = C_NULL
    elseif jobp == 'V'
        UcharJobvt = Cuchar(job)
        PtrV = pointer(MatP) + pos * sizeof(Float64)
        lapack_ldV = lapack_int(size(MatP, 1))
        PtrU = C_NULL
    else
        throw("Not recognized `jobp' flag $(jobp)");
    end

    PtrA = pointer(MatA) + aos * sizeof(Float64)

    if BLAS_LOADED_LIB_IND == 1
        @ccall BLAS_LOADED_LIB.:dgesvd64_(
            Ref(UcharJobu)::Ptr{Cuchar}, Ref(UcharJobvt)::Ptr{Cuchar}, Ref(lapack_m)::Ptr{lapack_int}, Ref(lapack_n)::Ptr{lapack_int},
            PtrA::Ptr{lapack_double}, Ref(lapack_ldA)::Ptr{lapack_int}, VecS::Ptr{lapack_double},
            PtrU::Ptr{lapack_double}, Ref(lapack_ldU)::Ptr{lapack_int}, PtrV::Ptr{lapack_double}, Ref(lapack_ldV)::Ptr{lapack_int},
            VecW::Ptr{lapack_double}, Ref(lapack_ldW)::Ptr{lapack_int}, Ref(lapack_info)::Ptr{lapack_int}
        )::Cvoid
    elseif BLAS_LOADED_LIB_IND == 2
        @ccall BLAS_LOADED_LIB.:dgesvd_(
            Ref(UcharJobu)::Ptr{Cuchar}, Ref(UcharJobvt)::Ptr{Cuchar}, Ref(lapack_m)::Ptr{lapack_int}, Ref(lapack_n)::Ptr{lapack_int},
            PtrA::Ptr{lapack_double}, Ref(lapack_ldA)::Ptr{lapack_int}, VecS::Ptr{lapack_double},
            PtrU::Ptr{lapack_double}, Ref(lapack_ldU)::Ptr{lapack_int}, PtrV::Ptr{lapack_double}, Ref(lapack_ldV)::Ptr{lapack_int},
            VecW::Ptr{lapack_double}, Ref(lapack_ldW)::Ptr{lapack_int}, Ref(lapack_info)::Ptr{lapack_int}
        )::Cvoid
    else
        throw("Incorrect loaded BLAS library, loaded index $(BLAS_LOADED_LIB_IND)")
    end 

end

dgesvd!(A::Ref{Matrix{Float64}}, m::Int, n::Int, S::Ref{Vector{Float64}}, U::Ref{Matrix{Float64}}, V::Ref{Matrix{Float64}}, jobu::Char, jobvt::Char, wsp_dgesvd::WSP = get_wsp_dgesvd(m, n); aos::Int = 0, uos::Int = 0, vos::Int = 0) = dgesvd!(A, m, n, aos, S, U, uos, V, vos, jobu, jobvt);

dgesvd!(A::Ref{Matrix{Float64}}, m::Int, n::Int, S::Ref{Vector{Float64}}, P::Ref{Matrix{Float64}}, jobp::Char, job::Char, wsp_dgesvd::WSP = get_wsp_dgesvd(m, n); aos::Int = 0, pos::Int = 0) = dgesvd!(A, m, n, aos, S, P, pos, jobp, job, wsp_dgesvd)

dgesvd!(A::Ref{Matrix{Float64}}, S::Ref{Vector{Float64}}, U::Ref{Matrix{Float64}}, V::Ref{Matrix{Float64}}, jobu::Char, jobvt::Char, wsp_dgesvd::WSP = get_wsp_dgesvd(size(A[])...); aos::Int = 0, uos::Int = 0, vos::Int = 0) = dgesvd!(A, size(A[])..., aos, S, U, uos, V, vos, jobu, jobvt)

dgesvd!(A::Ref{Matrix{Float64}}, S::Ref{Vector{Float64}}, P::Ref{Matrix{Float64}}, jobp::Char, job::Char, wsp_dgesvd::WSP = get_wsp_dgesvd(size(A[])...); aos::Int = 0, pos::Int = 0) = dgesvd!(A, size(A[])..., aos, S, P, pos, jobp, job)


# function dgees!(M::Ref{Matrix{Float64}}, P::Ref{Matrix{Float64}}; job = 'V', sort = 'S', alg::Ptr{Cvoid} = dgees_select_nz_angles_c)
#     n::Int = size(M[], 1);
#     wsp = get_wsp_dgees(n);
# end

BLAS_LOADED_LIB = BLAS.get_config().loaded_libs[1].libname;

function test_dgesvd(m, n)
    MatM = rand(m, n);
    svdM = svd(MatM)

    MatA = copy(MatM)



    VecS = Vector{Float64}(undef, n)
    MatU = Matrix{Float64}(undef, m, n)
    MatV = Matrix{Float64}(undef, n, n)

    M = Ref(MatM)
    A = Ref(MatA)
    S = Ref(VecS)
    U = Ref(MatU)
    V = Ref(MatV)

    wsp_dgesvd = get_wsp_dgesvd(m, n);

    dgesvd!(M, m, n, 0, S, U, 0, V, 0, 'S', 'N', wsp_dgesvd)

    println("Correct singular values?\t", VecS ≈ svdM.S)

    println("Correct left vectors?\t\t", MatU ≈ svdM.U)

    dgesvd!(A, S, V, 'V', 'O', wsp_dgesvd)
    
    println("Correct right vectors?\t\t", MatA[1:n, 1:n] ≈ svdM.V')
end