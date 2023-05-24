# This is the julia API for dgees, the real schur decomposition in double precision.
include("../global_path.jl")
include(joinpath(JULIA_INCLUDE_PATH, "workspace.jl"))
include(joinpath(JULIA_LAPACK_PATH, "LAPACK_setup.jl"))


function get_wsp_dgees(n::Int)
    # Allocated objects:
    # JOBVS, SORTS, (SELECT), N, {A}, LDA, SDIM, WR, WI, {VS}, LDVS, WORK, LWORK, (BWORK), INFO
    jobvs::Cuchar = 'V';
    sorts::Cuchar = 'S';
    n32::lapack_int = n;
    lda::lapack_int = n;
    sdim::lapack_int = n;
    vR::Vector{lapack_double} = Vector{lapack_double}(undef, n);
    vI::Vector{lapack_double} = Vector{lapack_double}(undef, n);
    ldvs::lapack_int = n;
    mW::Matrix{lapack_double} = Matrix{lapack_double}(undef, n, n);
    lw::lapack_int = length(mW);
    vB::Vector{lapack_logical} = Vector{lapack_logical}(undef, n)
    info::lapack_int = 0;
    return WSP(jobvs, sorts, n32, lda, sdim, vR, vI, ldvs, mW, lw, vB, info)
end

dgees_select_nz_angles(re::lapack_double, im::lapack_double)::lapack_int = im > 1e-15 || im < -1e-15;

dgees_select_nz_angles_c = @cfunction(dgees_select_nz_angles, lapack_int, (Ref{lapack_double}, Ref{lapack_double}));


# function dgees!(M::Ref{T}, P::Ref{T}, wsp::WSP; job::Cuchar = 'V', sort::Cuchar = 'S', alg::Ptr{Cvoid} = select_nz_angles_c) where T<: VecOrMat{Float64}
function dgees!(M::Ref{Matrix{Float64}}, P::Ref{Matrix{Float64}}, wsp::WSP; job = 'V', sort = 'S', alg = dgees_select_nz_angles_c)
    wsp[1] = Cuchar(job);
    wsp[2] = Cuchar(sort);

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

function dgees!(M::Ref{Matrix{Float64}}, P::Ref{Matrix{Float64}}; job = 'V', sort = 'S', alg::Ptr{Cvoid} = dgees_select_nz_angles_c)
    n::Int = size(M[], 1);
    wsp = get_wsp_dgees(n);
    dgees!(M, P, wsp; job = job, sort = sort, alg = alg);
end


using LinearAlgebra, MKL

BLAS_LOADED_LIB = BLAS.get_config().loaded_libs[1].libname;

function test_dgees(n)
    n = 10
    S = rand(n, n);
    S .-= S';

    D = similar(S); 
    copy!(D, S);
    P = similar(S);

    wsp_dgees = get_wsp_dgees(n);
    dgees!(Ref(D), Ref(P), wsp_dgees);
    println(S ≈ P * D * P')

    copy!(D, S)
    dgees!(Ref(D), Ref(P));
    println(S ≈ P * D * P')
end