include("blas_main.jl")
include("../workspace.jl")

using BenchmarkTools, Printf, Random

function schur_dgees!(M_r::Base.RefValue{Matrix{Float64}}, P_r::Base.RefValue{Matrix{Float64}}, wsp_Schur::WSP)
    M::Matrix{Float64} = M_r[]
    P::Matrix{Float64} = P_r[]
    VR::Vector{Float64} = retrieve(wsp_Schur, 1)
    VI::Vector{Float64} = retrieve(wsp_Schur, 2)
    MW::Matrix{Float64} = retrieve(wsp_Schur, 3)
    BV::Vector{Bool} = retrieve(wsp_Schur, 4)
    jobvs::Vector{Cuchar} = retrieve(wsp_Schur, 5)
    sort::Vector{Cuchar} = retrieve(wsp_Schur, 6)

    nM::Integer = size(M, 1)
    ldM::Integer = size(M, 1)
    sdim::Integer = 0
    ldP::Integer = size(M, 1)
    lwork::Integer = size(MW, 1)^2
    info::Integer = 0

    ccall((:dgees_, BLAS.get_config().loaded_libs[1].libname), Cvoid,
        (Ptr{Cuchar}, Ptr{Cuchar}, Ptr{Cvoid}, Integer, Ptr{Cdouble}, Integer, Integer, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Integer, Ptr{Cdouble}, Integer, Ptr{Bool}, Integer),
        jobvs, sort, C_NULL, nM, M, ldM, sdim, VR, VI, P, ldP, MW, lwork, C_NULL, info)

    sdim = 0   # Record the number of all selected eigenvalues.
    ldM = nM + 1    # Record the first nonselected eigenvalue position
    ldP = 0    # Record the number of selected eigenvalues that needed to be pivoted.
    lwork = 0  # Record the active column number in P and M
    info = 0   # Record the active column number in MW

    skip::Bool = false

    for jj = 1:nM
        BV[jj] = abs(VI[jj]) > 1e-15
        if BV[jj]
            sdim += 1
        else
            if ldM == 0
                ldM = jj
            end
        end
    end

    ldP = sdim - ldM + 1


    if ldP == 0
        return info
    else
        # println(BV)
        # println(sdim, ldM, ldP)
        lwork = ldM
        info = 1
        # Collect selected vectors that needed to be pivoted.
        for jj = lwork:nM
            if BV[jj]
                for ii = 1:nM
                    MW[ii, info] = P[ii, jj]
                end
                info += 1
            end
        end
        # Collect unselected vectors that needed to be pivoted.
        for jj = lwork:nM
            if !BV[jj]
                for ii = 1:nM
                    MW[ii, info] = P[ii, jj]
                end
                info += 1
            end
        end
        # Write the pivoted vectors in MW to the back of P
        info = 1
        for jj = lwork:nM
            for ii = 1:nM
                P[ii, jj] = MW[ii, info]
            end
            info += 1
        end

        info = 1
        # Collect selected blocks that needed to be pivoted.
        for jj = lwork:nM
            if BV[jj]
                if skip
                    skip = false
                else
                    # display([info, jj])
                    # display(M[jj:(jj+1), jj:(jj+1)])
                    MW[info, info] = M[jj, jj]
                    MW[info, info+1] = M[jj, jj+1]
                    MW[info+1, info] = M[jj+1, jj]
                    MW[info+1, info+1] = M[jj+1, jj+1]
                    M[jj, jj] = 0.0
                    M[jj, jj+1] = 0.0
                    M[jj+1, jj] = 0.0
                    M[jj+1, jj+1] = 0.0
                    info += 2
                    skip = true
                end
            end
        end

        # Collect unselected diagonals that needed to be pivoted.
        for jj = lwork:nM
            if !BV[jj]
                MW[info, info] = M[jj, jj]
                M[jj, jj] = 0.0
                info += 1
            end
        end

        # Write the pivoted blocks and diagonals in MW to the lower right of M
        info = 1
        for jj = lwork:2:(sdim-1)
            M[jj, jj] = MW[info, info]
            M[jj, jj+1] = MW[info, info+1]
            M[jj+1, jj] = MW[info+1, info]
            M[jj+1, jj+1] = MW[info+1, info+1]
            info += 2
        end
        for jj = (sdim+1):nM
            M[jj, jj] = MW[info, info]
            info += 1
        end
        return info
    end
end

function schur_dgees(M_r::Base.RefValue{Matrix{Float64}}, wsp_Schur::WSP)
    M = M_r[]
    A::Matrix{Float64} = similar(M)
    P::Matrix{Float64} = similar(M)

    A .= M

    schur_dgees!(Ref(A), Ref(P), wsp_Schur)

    return A, P
end

function get_WSP_schur(n::Int)
    Mn::Matrix{Float64} = Matrix{Float64}(undef, n, n)
    Vn::Vector{Vector{Float64}} = [Vector{Float64}(undef, n) for ii = 1:2]
    BVn::Vector{Bool} = Vector{Bool}(undef, n)
    CV1::Vector{Vector{Cuchar}} = [Vector{Cuchar}(undef, 1) for ii = 1:2]
    CV1[1][1] = 'V'
    CV1[2][1] = 'N'
    return WSP(Vn[1], Vn[2], Mn, BVn, CV1[1], CV1[2])
end


# function degees_test(n::Int)

#     global Mat = rand(n, n);

#     global WSP_schur = get_WSP_schur(n);


#     # display(M);

#     SWITCH_TO_MKL()
#     @printf "Computing Schur decomposition with degees under %s\n" BLAS.get_config().loaded_libs[1].libname;
#     @benchmark S_OPENBLAS, P_OPENBLAS = schur_dgees(Ref($Mat), $WSP_schur)


#     SWITCH_TO_OPENBLAS()
#     @printf "Computing Schur decomposition with degees under %s\n" BLAS.get_config().loaded_libs[1].libname;
#     @benchmark S_MKL, P_MKL = schur_dgees(Ref($Mat), $WSP_schur)

#     return nothing
# end

N_test = 200
Mat = rand(N_test, N_test);
Vec1 = rand(N_test);
Vec2 = rand(N_test);
Vec3 = rand(N_test);
WSP_schur = get_WSP_schur(N_test);

# display(M);

SWITCH_TO_MKL()
@printf "Computing Schur decomposition with degees under %s\n" BLAS.get_config().loaded_libs[1].libname;
rng = MersenneTwister(1234);
@benchmark begin
    Mat .= rand(rng, N_test, N_test)
    schur_dgees(Ref($Mat), $WSP_schur)
end

@printf "Basic linear algebra opeartions under %s\n" BLAS.get_config().loaded_libs[1].libname;
@benchmark $Vec1 .= $Mat * $Vec2 .+ $Vec3
@benchmark begin
    $Vec1 .= $Vec3
    mul!($Vec1, $Mat, $Vec2, 1.0, 1.0)
end

@benchmark begin
    $Vec1 .= $Vec3
    LAPACK.gels!('N', $Mat, $Vec1)
    mul!($Vec3, $Mat, $Vec3)
end





SWITCH_TO_OPENBLAS()
@printf "Computing Schur decomposition with degees under %s\n" BLAS.get_config().loaded_libs[1].libname;
rng = MersenneTwister(1234);
@benchmark begin
    Mat .= rand(rng, N_test, N_test)
    schur_dgees(Ref($Mat), $WSP_schur)
end

@printf "Basic linear algebra opeartions under %s\n" BLAS.get_config().loaded_libs[1].libname;
@benchmark $Vec1 .= $Mat * $Vec2 .+ $Vec3
@benchmark begin
    $Vec1 .= $Vec3
    mul!($Vec1, $Mat, $Vec2, 1.0, 1.0)
end

@benchmark begin
    $Vec1 .= $Vec3
    LAPACK.gels!('N', $Mat, $Vec1)
    mul!($Vec3, $Mat, $Vec3)
end

