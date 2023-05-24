# This is a code that implement essential operation on special orthogonal group.

include(homedir() * "/Documents/julia/inc/debug.jl")
include(homedir() * "/Documents/julia/inc/workspace.jl")
include(homedir() * "/Documents/julia/inc/myLA/matrix.jl")
include("so_wsp_gen.jl")

msg("Loading spec_orth.jl in " * (DEBUG ? "debug" : "normal") * " mode.\n");

# ALLOCATION_SCHUR = 0

KWARGS_SPEC_ORTH = [1e-10];

SO_ABSTOL_ = KWARGS_SPEC_ORTH[1];

SO_Transformer = (sqrt(2) / 2) .* [1 1; -im im];

using LinearAlgebra, DelimitedFiles

function schur_chk_eigenval(r::Float64, i::Float64)
    return abs(i) > 1e-15
end
# schur_skew_select = @cfunction(schur_chk_eigenval, Bool, (Cdouble, Cdouble))

# function schur_dgees!(M_r::Base.RefValue{Matrix{Float64}}, P_r::Base.RefValue{Matrix{Float64}}, wsp_Schur::WSP)
#     M::Matrix{Float64} = M_r[];
#     P::Matrix{Float64} = P_r[];
#     VR::Vector{Float64} = retrieve(wsp_Schur, 1);
#     VI::Vector{Float64} = retrieve(wsp_Schur, 2);
#     MW::Matrix{Float64} = retrieve(wsp_Schur, 3);
#     BV::Vector{Bool} = retrieve(wsp_Schur, 4);
#     jobvs::Vector{Cchar} = retrieve(wsp_Schur, 5);
#     sortV::Vector{Cchar} = retrieve(wsp_Schur, 6);

#     nM::Int = size(M, 1);
#     ldM::Int = size(M, 1);
#     sdim::Int = 0;
#     ldP::Int = size(M, 1);
#     lwork::Int = length(MW);
#     info::Int = 0;

#     tM = similar(M);
#     for M_ind in eachindex(M)
#         tM[M_ind] = M[M_ind];
#     end

#     # open("dgees_input_matrix.txt", "w") do file
#     #     writedlm(file, tM)
#     #     write(file, '\n');
#     #     writedlm(file, size(VR));
#     #     write(file, '\n');
#     #     writedlm(file, VR);
#     #     write(file, '\n');
#     #     writedlm(file, size(VI));
#     #     write(file, '\n');
#     #     writedlm(file, VI)
#     #     write(file, '\n');
#     #     writedlm(file, size(MW));
#     #     write(file, '\n');
#     #     writedlm(file, BV)
#     #     write(file, '\n');
#     #     writedlm(file, jobvs)
#     #     write(file, '\n');
#     #     writedlm(file, sortV)
#     # end


#     ccall((:dgees_,BLAS.get_config().loaded_libs[1].libname),
#         Cvoid,
#         (Ptr{Cchar}, Ptr{Cchar}, Ptr{Cvoid}, Int, Ptr{Cdouble}, Int, Int, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Int, Ptr{Cdouble}, Int, Ptr{Bool}, Int),
#         jobvs, sortV, C_NULL, nM, M, ldM, sdim, VR, VI, P, ldP, MW, lwork, C_NULL, info)


#     sdim = 0;   # Record the number of all selected eigenvalues.
#     ldM = nM + 1;    # Record the first nonselected eigenvalue position
#     ldP = 0;    # Record the number of selected eigenvalues that needed to be pivoted.
#     lwork = 0;  # Record the active column number in P and M
#     info = 0;   # Record the active column number in MW

#     skip::Bool = false;

#     for jj = 1:nM
#         BV[jj] = abs(VI[jj]) > 1e-15
#         if BV[jj]
#             sdim += 1;
#         else
#             if ldM == 0
#                 ldM = jj
#             end
#         end
#     end

#     ldP = sdim - ldM + 1;


#     if ldP == 0
#         return info
#     else
#         lwork = ldM;
#         info = 1
#         # Collect selected vectors that needed to be pivoted.
#         for jj = lwork:nM
#             if BV[jj]
#                 for ii = 1:nM
#                     MW[ii, info] = P[ii, jj];
#                 end
#                 info += 1;
#             end
#         end
#         # Collect unselected vectors that needed to be pivoted.
#         for jj = lwork:nM
#             if !BV[jj]
#                 for ii = 1:nM
#                     MW[ii, info] = P[ii, jj];
#                 end
#                 info += 1;
#             end
#         end
#         # Write the pivoted vectors in MW to the back of P
#         info = 1
#         for jj = lwork:nM
#             for ii = 1:nM
#                 P[ii, jj] = MW[ii, info];
#             end
#             info += 1;
#         end

#         info = 1
#         # Collect selected blocks that needed to be pivoted.
#         for jj = lwork:nM
#             if BV[jj]
#                 if skip
#                     skip = false
#                 else
#                     # display([info, jj])
#                     # display(M[jj:(jj+1), jj:(jj+1)])
#                     MW[info, info] = M[jj, jj];
#                     MW[info, info + 1] = M[jj, jj + 1];
#                     MW[info + 1, info] = M[jj + 1, jj];
#                     MW[info + 1, info + 1] = M[jj + 1, jj + 1];
#                     M[jj, jj] = 0.0;
#                     M[jj, jj + 1] = 0.0;
#                     M[jj + 1, jj] = 0.0;
#                     M[jj + 1, jj + 1] = 0.0;
#                     info += 2
#                     skip = true;
#                 end
#             end
#         end

#         # Collect unselected diagonals that needed to be pivoted.
#         for jj = lwork:nM
#             if !BV[jj]
#                 MW[info, info] = M[jj, jj];
#                 M[jj, jj] = 0.0;
#                 info += 1
#             end
#         end

#         # Write the pivoted blocks and diagonals in MW to the lower right of M
#         info = 1
#         for jj = lwork:2:(sdim - 1)
#             M[jj, jj] = MW[info, info]
#             M[jj, jj + 1] = MW[info, info + 1];
#             M[jj + 1, jj] = MW[info + 1, info];
#             M[jj + 1, jj + 1] = MW[info + 1, info + 1];
#             info += 2;
#         end
#         for jj = (sdim + 1):nM
#             M[jj, jj] = MW[info, info]
#             info += 1;
#         end
#         return info
#     end
# end

function schur_built_in!(M_r::Base.RefValue{Matrix{Float64}}, P_r::Base.RefValue{Matrix{Float64}}, wsp_dgees::WSP)
    M = M_r[]
    P = P_r[]

    VR = retrieve(wsp_dgees, 6)
    VI = retrieve(wsp_dgees, 7)
    MW = retrieve(wsp_dgees, 9)
    BV = retrieve(wsp_dgees, 11)

    nM::Int = size(M, 1)
    ldM::Int = size(M, 1)
    sdim::Int = 0
    ldP::Int = size(M, 1)
    lwork::Int = length(MW)
    info::Int = 0


    sdim = 0   # Record the number of all selected eigenvalues.
    ldM = nM + 1    # Record the first nonselected eigenvalue position
    ldP = 0    # Record the number of selected eigenvalues that needed to be pivoted.
    lwork = 0  # Record the active column number in P and M
    info = 0   # Record the active column number in MW

    skip::Bool = false



    F = schur!(M)

    for ind_P in eachindex(P)
        P[ind_P] = F.vectors[ind_P]
    end

    # display(M);

    for jj = 1:nM
        if isreal(F.values[jj])
            VI[jj] = 0
            VR[jj] = F.values[jj]
        else
            VI[jj] = F.values[jj].im
            VR[jj] = F.values[jj].re
        end
    end

    for jj = 1:nM
        BV[jj] = abs(VI[jj]) > 1e-15
        if BV[jj] != 0
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
            if BV[jj] != 0
                for ii = 1:nM
                    MW[ii, info] = P[ii, jj]
                end
                info += 1
            end
        end
        # Collect unselected vectors that needed to be pivoted.
        for jj = lwork:nM
            if BV[jj] == 0
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
            if BV[jj] != 0
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
            if BV[jj] == 0
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


function schur_dgees!(Mr::Ref{Matrix{Float64}}, P::Ref{Matrix{Float64}}, wsp_dgees::WSP)

    dgees!(Mr, P, wsp_dgees)

    # We manually select blocks.

    # M = Mr[];

    # nM = size(Mr[])[1]
    # sdim = 0;   # Record the number of all selected eigenvalues.
    # ldM = size(Mr[])[1] + 1;    # Record the first nonselected eigenvalue position
    # ldP = 0;    # Record the number of selected eigenvalues that needed to be pivoted.
    # lwork = 0;  # Record the active column number in P and M
    # info = 0;   # Record the active column number in MW

    # skip::Bool = false;

    # VI = retrieve(wsp_dgees, 7);
    # MW = retrieve(wsp_dgees, 9)
    # BV = retrieve(wsp_dgees, 11);


    # for jj = 1:nM
    #     BV[jj] = abs(VI[jj]) > 1e-15
    #     if BV[jj] != 0
    #         sdim += 1;
    #     else
    #         if ldM == 0
    #             ldM = jj
    #         end
    #     end
    # end

    # ldP = sdim - ldM + 1;


    # if ldP == 0
    #     return info
    # else
    #     lwork = ldM;
    #     info = 1
    #     # Collect selected vectors that needed to be pivoted.
    #     for jj = lwork:nM
    #         if BV[jj] != 0
    #             for ii = 1:nM
    #                 MW[ii, info] = P[ii, jj];
    #             end
    #             info += 1;
    #         end
    #     end
    #     # Collect unselected vectors that needed to be pivoted.
    #     for jj = lwork:nM
    #         if BV[jj] == 0
    #             for ii = 1:nM
    #                 MW[ii, info] = P[ii, jj];
    #             end
    #             info += 1;
    #         end
    #     end
    #     # Write the pivoted vectors in MW to the back of P
    #     info = 1
    #     for jj = lwork:nM
    #         for ii = 1:nM
    #             P[ii, jj] = MW[ii, info];
    #         end
    #         info += 1;
    #     end

    #     info = 1
    #     # Collect selected blocks that needed to be pivoted.
    #     for jj = lwork:nM
    #         if BV[jj] != 0
    #             if skip
    #                 skip = false
    #             else
    #                 # display([info, jj])
    #                 # display(M[jj:(jj+1), jj:(jj+1)])
    #                 MW[info, info] = M[jj, jj];
    #                 MW[info, info + 1] = M[jj, jj + 1];
    #                 MW[info + 1, info] = M[jj + 1, jj];
    #                 MW[info + 1, info + 1] = M[jj + 1, jj + 1];
    #                 M[jj, jj] = 0.0;
    #                 M[jj, jj + 1] = 0.0;
    #                 M[jj + 1, jj] = 0.0;
    #                 M[jj + 1, jj + 1] = 0.0;
    #                 info += 2
    #                 skip = true;
    #             end
    #         end
    #     end

    #     # Collect unselected diagonals that needed to be pivoted.
    #     for jj = lwork:nM
    #         if BV[jj] == 0
    #             MW[info, info] = M[jj, jj];
    #             M[jj, jj] = 0.0;
    #             info += 1
    #         end
    #     end

    #     # Write the pivoted blocks and diagonals in MW to the lower right of M
    #     info = 1
    #     for jj = lwork:2:(sdim - 1)
    #         M[jj, jj] = MW[info, info]
    #         M[jj, jj + 1] = MW[info, info + 1];
    #         M[jj + 1, jj] = MW[info + 1, info];
    #         M[jj + 1, jj + 1] = MW[info + 1, info + 1];
    #         info += 2;
    #     end
    #     for jj = (sdim + 1):nM
    #         M[jj, jj] = MW[info, info]
    #         info += 1;
    #     end
    #     return info
    # end
end

# schur_dgees!(Mr::Ref{Matrix{Float64}}, P::Ref{Matrix{Float64}}, wsp_dgees::WSP) = schur_built_in!(Mr, P, wsp_dgees)



function reOrth(U, D::AbstractVecOrMat{T}=nothing) where {T}
    # This function reorthogonalizes vectors U into V. The argument D inidicates that U is coming from eigendecomposition U D inv(U).
    # When D is nothing, it returns Q, R from qr decomposition U = QR.
    # When D is an array or matrix, it returns Q, (R D inv(R)).
    n, k = size(U)
    Q, R = qr(U)
    if isnothing(D)
        return Q[:, 1:k], R
    else
        if typeof(D) <: AbstractVector{T}
            return Q[:, 1:k], R * diagm(D) * inv(R)
        else
            return Q[:, 1:k], R * D * inv(R)
        end
    end
end

function so_completion(U)
    Ans = Matrix(qr(U).Q[:, :])
    if det(Ans) < 0
        n, = size(Ans)
        Ans[:, n] .*= -1
    end
    return Ans
end

function get_Psi(D)
    n = length(D)
    Ans::Matrix{ComplexF64} = zeros(n, n)
    for ii = 1:n
        for jj = 1:n
            if norm(D[ii] - D[jj], Inf) < SO_ABSTOL_
                Ans[ii, jj] = 1
            else
                Ans[ii, jj] = (1 - exp(D[jj] - D[ii])) / (D[ii] - D[jj])
            end
        end
    end
    return Ans
end

mutable struct Skew_Factor
    S::Ref{Matrix{Float64}}
    Z
    D
    Psi
    Skew_Factor(S) = new(S, nothing, nothing, nothing)
end
function Base.show(io::IO, p::Skew_Factor)
    print(io, "Decomposition info of skewsymmetric matrix:\n")
    display(p.S)
    print(io, "Eigenvectors\n")
    display(p.Z)
    print(io, "Eigenvalues:\n")
    display(p.D)

    print(io, "Hermitian Matrix constructed from D, Psi_ij = (D[i] == D[j]) ? 1 : (1 - exp(D[j] - D[i])) / (D[j] - D[i]):\n")
    display(p.Psi)
end

function init_Skew_Factor(sof)
    S = sof.S[]
    n, = size(S)

    if isnothing(sof.Z)
        Z = Matrix{ComplexF64}(undef, n, n)
        D = Vector{ComplexF64}(undef, n)

        sof_e = eigen(S)
        Z .= sof_e.vectors
        D .= sof_e.values
        if !isOrthonormal(Z)
            msg("Lost of orthogonality. Attemp to make Reorthogonalization.\n")
            Z, D_temp = reOrth(Z, D)
            assert(isDiagonal, D_temp,
                debug_msg=(["Failed to resume diagonal D from U D inv(U) in process of reorthogonalizing U.", D_temp]))
            D .= diag(D_temp)
        end
        sof.Z = Z
        sof.D = D
    end

    if isnothing(sof.Psi)
        Psi = get_Psi(sof.D)
        sof.Psi = Psi
    end
    return
end

function compute_Skew_Factor(sof)
    S = sof.S[]
    n, = size(S)

    Z = Matrix{ComplexF64}(undef, n, n)
    D = Vector{ComplexF64}(undef, n)

    sof_e = eigen(S)
    Z .= sof_e.vectors
    D .= sof_e.values
    if !isOrthonormal(Z)
        msg("Lost of orthogonality. Attemp to make Reorthogonalization.")
        Z, D_temp = reOrth(Z, D)
        assert(isDiagonal, D_temp,
            debug_msg(["Failed to resume diagonal D from U D inv(U) in process of reorthogonalizing U.", D_temp]))
        D .= diag(D_temp)
    end
    sof.Z = Z
    sof.D = D

    Psi = get_Psi(sof.D)
    sof.Psi = Psi
    return
end

function Lambda2Phi(Λ)
    n = length(Λ)
    Ans::Matrix{ComplexF64} = [
        isapprox(Λ[ii], Λ[jj]) ? exp(Λ[ii]) : (exp(Λ[ii]) - exp(Λ[jj])) / (Λ[ii] - Λ[jj])
        for ii = 1:n, jj = 1:n]
    return Ans
end

function dExp(S, dS)
    # Compute d exp(S + t * dS)/ dt = P( (P' dS P)⊙Φ(Λ) )P' where P Λ P' = S.
    n = size(S)[1]

    assert(isSkewSym, S)
    assert(isSkewSym, dS)

    eigen_F = eigen(S)
    perm = sortperm(abs.(imag.(eigen_F.values)), rev=true)

    U = eigen_F.vectors[:, perm]
    D = eigen_F.values[perm]

    ind_c = 1
    ind_e = ind_c + 1
    ang_c = abs(D[ind_c])

    # Orthogonality check
    while !isapprox(0, ang_c, atol=SO_ABSTOL_)
        # nonzero angle found at current index : ind_c
        while ind_e < n && isapprox(ang_c, abs(D[ind_e]), atol=SO_ABSTOL_)
            # collect all associated vectors;
            ind_e += 1
        end
        if !isOrthonormal(U[:, ind_c:ind_e])
            d_msg("spec_orth, dExp(S, dS) : Lost of orthogonality detected in the eigen decomposition of S. \n Performing reorthogonalization...")
            U_new, D_new = reOrth(U[:, ind_c:ind_e], D[ind_c:ind_e])
            assert(isdiag, D_new,
                debug_msg=(D_new, "spec_orth.jl , dExp(S, dS) : Cannot perserve nonzero structure in diagonal matrix of the eigenvalues.\n"))
            assert(SO_ABSTOL_, >, x -> norm(abs.(x[1]) .- abs.(diag(x[2])), Inf), (D, D_new),
                debug_msg=(D, D_new, "spec_orth.jl , dExp(S, dS) :Cannot preserve eigenvalues in the reorthogonalization.\n"))
            U[:, ind_c:ind_e] .= U_new
            D[ind_c:ind_e] .= diag(D_new)
        end
        ind_c = ind_e + 1
        ind_e = ind_c + 1
        if ind_c >= n
            break
        else
            ang_c = abs(D[ind_c])
        end
    end


    Φ = Lambda2Phi(D)
    Ans = U' * dS * U
    Ans .*= Φ
    Ans = U * Ans * U'

    return Ans
end

struct AngularDecomposition
    vectors::Matrix{Float64}
    angles::Array{Float64,1}
    AngularDecomposition(vec, ang) = new(vec, ang)
end

function Base.show(io::IO, F::AngularDecomposition)
    println("nonzero angles:")
    display(F.angles)
    println("Basis:")
    display(F.vectors)
end

function PSPT(P, S)
    n, = size(P)
    k, = size(S)

    Ans = zeros(n, n)
    temp = 0.0

    for Si = 1:k
        for Sj = 1:(Si-1)
            for Xi = 1:n
                for Xj = 1:(Xi-1)
                    temp = S[Si, Sj] * (P[Xi, Si] * P[Xj, Sj] - P[Xj, Si] * P[Xi, Sj])
                    Ans[Xi, Xj] += temp
                    Ans[Xj, Xi] -= temp
                end
            end
        end
    end
    return Ans
end

function PSPT!(P_ref, S_ref, Ans_ref)
    P = P_ref[]
    S = S_ref[]
    n, = size(P)
    k, = size(S)

    Ans = Ans_ref[]
    temp = 0.0

    for Si = 1:k
        for Sj = 1:(Si-1)
            for Xi = 1:n
                for Xj = 1:(Xi-1)
                    temp = S[Si, Sj] * (P[Xi, Si] * P[Xj, Sj] - P[Xj, Si] * P[Xi, Sj])
                    Ans[Xi, Xj] += temp
                    Ans[Xj, Xi] -= temp
                end
            end
        end
    end
end

function PCSPT!(vectors_ref::Base.RefValue{Matrix{Float64}}, angle::Float64, Ans_Ref::Base.RefValue{Matrix{Float64}})
    # This function is usually called with 
    # |:  : | | c  s| | .. v1^T .. |
    # |v1 v2| | -s c| | .. v2^T .. | = c(v1 * v1^T + v2 * v2^T) + s(v1 * v2^T - v2 * v1^T).
    # |:  : |

    vectors = vectors_ref[]

    Ans = Ans_Ref[]
    n::Int = size(Ans)[1]

    c::Float64 = cos(angle)
    s::Float64 = sin(angle)
    t1::Float64 = 0.0
    t2::Float64 = 0.0

    for ii = 1:n
        for jj = 1:(ii-1)
            t1 = c * (vectors[ii, 1] * vectors[jj, 1] + vectors[ii, 2] * vectors[jj, 2])
            t2 = s * (vectors[ii, 1] * vectors[jj, 2] - vectors[ii, 2] * vectors[jj, 1])
            Ans[ii, jj] += t1 + t2
            Ans[jj, ii] += t1 - t2
        end
        Ans[ii, ii] += c * (vectors[ii, 1] * vectors[ii, 1] + vectors[ii, 2] * vectors[ii, 2])
    end
end

function PCSPT!(vectors_ref::Base.RefValue{Matrix{Float64}}, angle::Float64, Ans_Ref::Base.RefValue{Matrix{Float64}}, col_ind::Int)
    # This function is usually called with 
    # |:  : | | c  s| | .. v1^T .. |
    # |v1 v2| | -s c| | .. v2^T .. | = c(v1 * v1^T + v2 * v2^T) + s(v1 * v2^T - v2 * v1^T).
    # |:  : |

    vectors = vectors_ref[]

    Ans = Ans_Ref[]
    n::Int = size(Ans)[1]

    c::Float64 = cos(angle)
    s::Float64 = sin(angle)
    t1::Float64 = 0.0
    t2::Float64 = 0.0

    for ii = 1:n
        for jj = 1:(ii-1)
            t1 = c * (vectors[ii, col_ind] * vectors[jj, col_ind] + vectors[ii, col_ind+1] * vectors[jj, col_ind+1])
            t2 = s * (vectors[ii, col_ind] * vectors[jj, col_ind+1] - vectors[ii, col_ind+1] * vectors[jj, col_ind])
            Ans[ii, jj] += t1 + t2
            Ans[jj, ii] += t1 - t2
        end
        Ans[ii, ii] += c * (vectors[ii, col_ind] * vectors[ii, col_ind] + vectors[ii, col_ind+1] * vectors[ii, col_ind+1])
    end
end
# function PCSPT!(Q_r, U_r, V_r, CS_r)
#     # This function is usually called with 
#     # |:  : | | c  s| | .. v1^T .. |
#     # |v1 v2| | -s c| | .. v2^T .. | = c(v1 * v1^T + v2 * v2^T) + s(v1 * v2^T - v2 * v1^T).
#     # |:  : |

#     vectors = vectors_ref[];

#     Ans = Ans_Ref[];
#     n = size(Ans)[1];

#     c = cos(angle);
#     s = sin(angle);

#     for ii = 1:n
#         for jj = 1:(ii - 1)
#             t1 = c * (vectors[ii, 1] * vectors[jj, 1] + vectors[ii, 2] * vectors[jj, 2]);
#             t2 = s * (vectors[ii, 1] * vectors[jj, 2] - vectors[ii, 2] * vectors[jj, 1]);
#             Ans[ii, jj] += t1 + t2;
#             Ans[jj, ii] += t1 - t2;
#         end
#         Ans[ii, ii] += c * (vectors[ii, 1] * vectors[ii, 1] + vectors[ii, 2] * vectors[ii, 2]);
#     end
# end

function get_A(F::AngularDecomposition)
    Ans = zeros(size(F.vectors))
    n_p = length(F.angles)
    for ii = 1:n_p
        Ans[2*ii-1, 2*ii] = F.angles[ii]
        Ans[2*ii, 2*ii-1] = -F.angles[ii]
    end
    return Ans
end

function get_eA(F::AngularDecomposition)
    Ans = zeros(size(F.vectors))
    n_p = length(F.angles)
    for ii = 1:n_p
        Ans[2*ii-1, 2*ii-1] = cos(F.angles[ii])
        Ans[2*ii-1, 2*ii] = sin(F.angles[ii])
        Ans[2*ii, 2*ii-1] = -sin(F.angles[ii])
        Ans[2*ii, 2*ii] = -cos(F.angles[ii])
    end
    for ii = (2*n_p):size(Ans)[1]
        Ans[ii, ii] = 1.0
    end
    return Ans
end

function get_S(F::AngularDecomposition)
    n, = size(F.vectors)
    n_p = length(F.angles)

    Ans = zeros(n, n)

    for kk = 1:n_p
        for ii = 1:n
            for jj = 1:(ii-1)
                ss = F.angles[kk] * (F.vectors[ii, 2*kk-1] * F.vectors[jj, 2*kk] - F.vectors[ii, 2*kk] * F.vectors[jj, 2*kk-1])
                Ans[ii, jj] += ss
                Ans[jj, ii] -= ss
            end
        end
    end
    # Ans = V * Ans * V';
    return Ans
end

function get_S(P, A)
    n, = size(P)
    n_p = length(A)

    Ans = zeros(n, n)
    temp = 0

    for kk = 1:n_p
        for ii = 1:n
            for jj = 1:(ii-1)
                temp = A[kk] * (P[ii, 2*kk-1] * P[jj, 2*kk] - P[ii, 2*kk] * P[jj, 2*kk-1])
                Ans[ii, jj] += temp
                Ans[jj, ii] -= temp
            end
        end
    end
    return Ans
end

function get_S!(P_ref::Base.RefValue{Matrix{Float64}}, A_ref::Base.RefValue{Vector{Float64}}, Ans_ref::Base.RefValue{Matrix{Float64}})
    # X must be initialized with 0.
    P::Matrix{Float64} = P_ref[]
    A::Vector{Float64} = A_ref[]
    n::Int, = size(P)
    # n_p = length(A);
    Ans::Matrix{Float64} = Ans_ref[]

    temp::Float64 = 0.0

    for kk = 1:length(A)
        if A[kk] == 0.0
            break
        end
        for ii = 1:n
            for jj = 1:(ii-1)
                temp = A[kk] * (P[ii, 2*kk-1] * P[jj, 2*kk] - P[ii, 2*kk] * P[jj, 2*kk-1])
                Ans[ii, jj] += temp
                Ans[jj, ii] -= temp
            end
        end
    end
end


function get_Q(F::AngularDecomposition)
    V = similar(F.vectors)
    V .= F.vectors
    Ans = zeros(size(V))
    n_p = length(F.angles)
    for ii = 1:n_p
        Ans[2*ii-1, 2*ii-1] = cos(F.angles[ii])
        Ans[2*ii-1, 2*ii] = sin(F.angles[ii])
        Ans[2*ii, 2*ii-1] = -sin(F.angles[ii])
        Ans[2*ii, 2*ii] = cos(F.angles[ii])
    end
    for ii = (2*n_p+1):size(V)[1]
        Ans[ii, ii] = 1.0
    end
    Ans = V * Ans * V'
    return Ans
end


function get_Q!(P_ref, Θ_ref, Ans_ref)
    P = P_ref[]
    Θ = Θ_ref[]
    Ans = Ans_ref[]

    n, = size(Ans)

    # n_p = length(angles);
    n_p = count(x -> x != 0, Θ)

    V = Matrix{Float64}(undef, n, 2)
    V_ref = Ref(V)


    for ii = 1:n_p
        V .= P[:, (2*ii-1):(2*ii)]
        PCSPT!(V_ref, Θ[ii], Ans_ref)
    end

    for ii = (2*n_p+1):n
        for jj = 1:n
            for kk = 1:(jj-1)
                t = P[jj, ii] * P[kk, ii]
                Ans[jj, kk] += t
                Ans[kk, jj] += t
            end
            Ans[jj, jj] += P[jj, ii] * P[jj, ii]
        end
    end
end

function get_wsp_schur_Q(n::Int)
    Mn2::Matrix{Float64} = Matrix{Float64}(undef, n, 2)
    return WSP(Mn2)
end

function get_Q!(Q_r::Base.RefValue{Matrix{Float64}}, P_r::Base.RefValue{Matrix{Float64}}, A_r::Base.RefValue{Vector{Float64}}, wsp_schur_Q)
    # X must be initialized with 0.
    P::Matrix{Float64} = P_r[]
    A::Array{Float64} = A_r[]
    Q::Matrix{Float64} = Q_r[]

    n::Int = size(Q)[1]

    # n_p = length(angles);
    # n_p = count(x -> x != 0, Θ)

    n_p::Int = 0
    for ii = 1:length(A)
        if abs(A[ii]) > 1e-16
            n_p = n_p + 1
        end
    end


    # V = Matrix{Float64}(undef, n, 2)
    # V_ref = Ref(V)

    Mn2 = retrieve(wsp_schur_Q, 1)
    Mn2_r = wsp_schur_Q.vec[1]

    Q .= 0.0

    for ii = 1:n_p
        for jj = 1:n
            # V .= P[:, (2*ii-1):(2*ii)]
            Mn2[jj, 1] = P[jj, 2*ii-1]
            Mn2[jj, 2] = P[jj, 2*ii]
        end
        PCSPT!(Mn2_r, A[ii], Q_r)
    end

    for ii = (2*n_p+1):n
        for jj = 1:n
            for kk = 1:(jj-1)
                t = P[jj, ii] * P[kk, ii]
                Q[jj, kk] += t
                Q[kk, jj] += t
            end
            Q[jj, jj] += P[jj, ii] * P[jj, ii]
        end
    end
end

Matrix_Factor(F::AngularDecomposition) = F.vectors, get_A(F);

function IS_REAL_OR_PURE_IMAG(U)::Int
    # Return 1 if the complex matrix U is real, return -1 if U is purely imaginary and return 0 otherwise.
    if isapprox(0, norm(imag.(U), Inf), atol=SO_ABSTOL_)
        return 1
    elseif isapprox(0, norm(real.(U), Inf), atol=SO_ABSTOL_)
        return -1
    else
        return 0
    end
end

function COMPLEX_2_REAL_BASIS(U; paired=false, ang=nothing)
    # This function convert the complex basis U to real basis V by unitary matrix Q: U = VQ or V = UQ';
    # When U has 2 columns and it is indicated as a pair (usually means it is a paired vector associated to a nonzero angle)
    # Q = SO_Transformer or Q = im * SO_Transformer.
    # When U has 2p columns and it is indicated as pairs, there are p pairs of vectors that can be transformed with
    # Q = SO_Transformer or Q = im * SO_Transformer. Note that we assume the first half are associated with negativa values.
    # When U is not indicated as pairs (usually seen from vectors associated to zero angle)
    # V is solved by the X vectors given in svd of U * U' = X * Y'. As UU' being real, Q = U' * Y and V = X.

    n, k = size(U)
    # if isnothing(ang)
    #     assert(SO_ABSTOL_, >, x -> norm(imag.(x * x'), Inf), U;
    #         debug_msg = (U, "The vector being transformed does not form real UU'."));
    # else
    #     assert(SO_ABSTOL_, >, x -> norm(imag.(x[1] * diagm(0=>exp.(x[2])) * x[1]'), Inf), (U, ang);
    #         debug_msg = (U, "The vector being transformed does not form real U exp(Λ) U'."));
    # end

    assert(SO_ABSTOL_, >, x -> norm(imag.(x * x'), Inf), U;
        debug_msg=(U, "The vector being transformed does not form real UU'."))


    if paired
        assert(0, ==, x -> mod(x[1], x[2]), (k, 2))
        m = div(k, 2)

        if m == 1
            V = U * SO_Transformer'
            check = IS_REAL_OR_PURE_IMAG(V)
            if check == 0
                if DEBUG && MSG
                    display(U)
                end
                throw("Error: Complex vectors cannot be transformed into real by sqrt(2)/2 .* [1 1; im - im].")
            end
            assert(SO_ABSTOL_, >, x -> norm(real.(x[1] * x[1]') .- x[2] * x[2]', Inf), (U, V))
            return (check == 1) ? real.(V) : imag.(V)
        else
            perm = 1:k
            p_end = k
            if !isnothing(ang)
                perm = sortperm(ang)
            end
            Ans = zeros(n, k)
            for ii = 1:m
                check = 0
                for jj = (m+1):p_end
                    V = U[:, [perm[ii], perm[jj]]] * SO_Transformer'
                    check = IS_REAL_OR_PURE_IMAG(V)
                    if check == 0
                        continue
                    end
                    Ans[:, (2*ii-1):(2*ii)] .= (check == 1) ? real.(V) : imag.(V)
                    perm[jj] = perm[p_end]
                    p_end -= 1
                    break
                end
                assert(0, !=, check;
                    debug_msg=(ii, U[:, perm[ii:p_end]], ang[perm[ii:p_end]],
                        "Complex vectors cannot be transformed into real by sqrt(2)/2 .* [1 1; im - im]."))
            end
            assert(SO_ABSTOL_, >, x -> norm(real.(x[1] * x[1]') .- x[2] * x[2]', Inf), (U, Ans);
                debug_msg=(k, U, Ans))
            return Ans
        end
    else
        UUT = real.(U * U')
        V = svd(UUT).U[:, 1:k]
        assert(sqrt(SO_ABSTOL_), >, x -> norm(real.(x[1] * x[1]') .- x[2] * x[2]', Inf), (U, V);
            debug_msg=(U, V, "Failed to recoverd UUT"))
        return V
    end
end


function real_schur_s(X)
    F = schur(X)
    v = norm.(F.values) .> 1e-15
    # ordschur!(F, v);
    F = ordschur(F, v)

    ang_n = div(sum(v), 2)

    angles = [F.T[2*ii-1, 2*ii] for ii = 1:ang_n]
    Z = F.Z

    return AngularDecomposition(Z, angles)
end

function real_schur_q(X)
    F = schur(X)
    v = norm.(F.values .- 1) .> 1e-15
    # ordschur!(F, v);
    F = ordschur(F, v)

    ang_n = div(sum(v), 2)

    angles = [atan(F.T[2*ii-1, 2*ii], F.T[2*ii, 2*ii]) for ii = 1:ang_n]
    Z = F.Z

    return AngularDecomposition(Z, angles)
end

function real_schur_s!(X_ref, P_ref, A_ref)
    # Note that for inplace real_schur, A carries 0.
    X::Matrix{Float64} = X_ref[]
    F = schur(X)
    v = norm.(F.values) .> 1e-15
    # ordschur!(F, v);
    F = ordschur(F, v)

    ang_n::Int = div(sum(v), 2)

    A::Vector{Float64} = A_ref[]
    P::Matrix{Float64} = P_ref[]

    A .= 0.0
    for ii = 1:ang_n
        A[ii] = F.T[2*ii-1, 2*ii]
    end
    n::Int, = size(X)
    for ii = 1:n
        for jj = 1:n
            P[ii, jj] = F.Z[ii, jj]
        end
    end
    # P .= F.Z
end

function real_schur_s!(X_ref, P_ref, A_ref, v_ref)
    # Note that for inplace real_schur, A carries 0.
    X::Matrix{Float64} = X_ref[]
    v = v_ref[]
    n, = size(X)
    F = schur!(X)
    # v = norm.(F.values) .> 1e-10;
    for ii = 1:n
        v[ii] = norm(F.values) > 1e-15
    end
    ordschur!(F, v)
    # F = ordschur(F, v);

    ang_n = div(sum(v), 2)

    A = A_ref[]
    P = P_ref[]

    A .= 0.0
    for ii = 1:ang_n
        A[ii] = F.T[2*ii-1, 2*ii]
    end
    # P .= F.Z;
    for ii = 1:n
        for jj = 1:n
            P[ii, jj] = F.Z[ii, jj]
        end
    end
end

function real_schur_s!(X_ref::Base.RefValue{Matrix{Float64}}, P_ref::Base.RefValue{Matrix{Float64}}, A_ref::Base.RefValue{Vector{Float64}}, wsp_schur::WSP)
    # Note that for inplace real_schur, A carries 0.
    X::Matrix{Float64} = X_ref[]
    n, = size(X)

    # F = schur!(X)
    # P::Matrix{Float64} = P_ref[]
    # for ii = 1:n
    #     for jj = 1:n
    #         P[ii, jj] = F.Z[ii, jj];
    #     end
    # end
    # # v = norm.(F.values) .> 1e-10;
    # for ii = 1:n
    #     v[ii] = norm(F.values) > 1e-10
    # end
    # ordschur!(F, v)
    # F = ordschur(F, v);


    schur_dgees!(X_ref, P_ref, wsp_schur)

    # print("Real schur on skew-symmetric: ");
    # @time schur_dgees!(X_ref, P_ref, wsp_schur);

    ang_n::Int = 0
    for ii = 1:div(n, 2)
        if abs(X[2*ii-1, 2*ii]) > 1e-15
            ang_n += 1
        end
    end

    A = A_ref[]

    A .= 0.0
    for ii = 1:ang_n
        A[ii] = X[2*ii-1, 2*ii]
    end
end

function real_schur_q!(X_ref, P_ref, A_ref)
    X = X_ref[]
    F = schur(X)
    v = norm.(F.values .- 1) .> 1e-15

    # ordschur!(F, v);
    F = ordschur(F, v)

    ang_n = div(sum(v), 2)


    A = A_ref[]
    P = P_ref[]

    A .= 0.0
    for ii = 1:ang_n
        A[ii] = atan(F.T[2*ii-1, 2*ii], F.T[2*ii, 2*ii])
    end
    P .= F.Z
end

function real_schur_q!(X_ref::Base.RefValue{Matrix{Float64}}, P_ref::Base.RefValue{Matrix{Float64}}, A_ref::Base.RefValue{Vector{Float64}}, v_ref::Base.RefValue{Vector{Bool}})
    # This routine overwrites X
    X::Matrix{Float64} = X_ref[]
    v::Vector{Bool} = v_ref[]
    n::Int, = size(X)

    F = schur!(X)
    for ii = 1:n
        v[ii] = norm(F.values[ii] - 1) > 1e-15
    end
    # v = norm.(F.values .- 1) .> 1e-10;
    ordschur!(F, v)
    # F = ordschur(F, v);

    ang_n::Int = div(sum(v), 2)


    A::Vector{Float64} = A_ref[]
    P::Matrix{Float64} = P_ref[]

    A .= 0.0
    for ii = 1:ang_n
        A[ii] = atan(F.T[2*ii-1, 2*ii], F.T[2*ii, 2*ii])
    end
    # P .= F.Z;
    for ii = 1:n
        for jj = 1:n
            P[ii, jj] = F.Z[ii, jj]
        end
    end
end

function real_schur_q!(X_ref::Base.RefValue{Matrix{Float64}}, P_ref::Base.RefValue{Matrix{Float64}}, A_ref::Base.RefValue{Vector{Float64}}, wsp_schur::WSP)
    # This routine overwrites X
    X::Matrix{Float64} = X_ref[]
    n::Int, = size(X)

    # F = schur!(X)
    # P::Matrix{Float64} = P_ref[]
    # for ii = 1:n
    #     for jj = 1:n
    #         P[ii, jj] = F.Z[ii, jj];
    #     end
    # end
    # for ii = 1:n
    #     v[ii] = norm(F.values[ii] - 1) > 1e-10
    # end
    # # v = norm.(F.values .- 1) .> 1e-10;
    # ordschur!(F, v)
    # F = ordschur(F, v);

    schur_dgees!(X_ref, P_ref, wsp_schur)

    # print("Real schur on special-orthogonal: ");
    # @time schur_dgees!(X_ref, P_ref, wsp_schur);


    ang_n::Int = 0
    for ii = 1:div(n, 2)
        if abs(X[2*ii-1, 2*ii-1] - 1) > 1e-15
            ang_n += 1
        end
    end

    A::Vector{Float64} = A_ref[]

    A .= 0.0
    for ii = 1:ang_n
        A[ii] = atan(X[2*ii-1, 2*ii], X[2*ii, 2*ii])
    end
end

function log_skew!(S_r::Base.RefValue{Matrix{Float64}}, P_r::Base.RefValue{Matrix{Float64}}, Θ_r::Base.RefValue{Vector{Float64}}, U_r::Base.RefValue{Matrix{Float64}}, wsp_log_skew::WSP; orderP=false)
    # wsp_log_skew has n x n real matrix M and a boolean n vector v

    S::Matrix{Float64} = S_r[]
    P::Matrix{Float64} = P_r[]
    Θ::Vector{Float64} = Θ_r[]
    U::Matrix{Float64} = U_r[]



    if size(U)[1] == 2
        P[1, 1] = 1.0
        P[2, 2] = 1.0
        P[1, 2] = 0.0
        P[2, 1] = 0.0
        Θ[1] = atan(U[2, 1], U[1, 1])

        S[1, 1] = 0.0
        S[2, 2] = 0.0
        S[1, 2] = -Θ[1]
        S[2, 1] = Θ[1]
        return nothing
    end

    M::Matrix{Float64} = retrieve(wsp_log_skew, 1)
    wsp_schur::WSP = retrieve(wsp_log_skew, 2)

    n::Int, = size(U)

    for ii = 1:n
        for jj = 1:n
            M[ii, jj] = U[ii, jj]
        end
    end
    # Obtain real schur decomposition
    real_schur_q!(wsp_log_skew.vec[1], P_r, Θ_r, wsp_schur)

    # order the Schur vectors pair by their first element.
    if orderP
        for ind in eachindex(Θ)
            if abs(Θ[ind]) < 1e-15
                continue
            end
            # Positive angles
            if Θ[ind] < 0
                for r_ind = 1:n
                    M[r_ind, 1] = P[r_ind, 2*ind-1]
                    P[r_ind, 2*ind-1] = P[r_ind, 2*ind]
                end
                for r_ind = 1:n
                    P[r_ind, 2*ind] = M[r_ind, 1]
                end
                Θ[ind] = -Θ[ind]
            end
            # Positive leading number in the first vector
            if P[1, 2*ind-1] < 0
                for r_ind = 1:n
                    P[r_ind, 2*ind-1] = -P[r_ind, 2*ind-1]
                    P[r_ind, 2*ind] = -P[r_ind, 2*ind]
                end
            end

            # if abs(P[1, 2*ind-1]) > abs(P[1, 2*ind])
            #     for r_ind = 1:n
            #         M[r_ind, 1] = P[r_ind, 2*ind-1]
            #         P[r_ind, 2*ind-1] = P[r_ind, 2*ind]
            #     end
            #     for r_ind = 1:n
            #         P[r_ind, 2*ind] = M[r_ind, 1]
            #     end
            #     Θ[ind] = -Θ[ind]
            # end
            # # Positive leading number in the first vector
            # if P[1, 2*ind-1] < 0
            #     for r_ind = 1:n
            #         P[r_ind, 2*ind-1] = -P[r_ind, 2*ind-1]
            #         P[r_ind, 2*ind] = -P[r_ind, 2*ind]
            #     end
            # end

        end
    end


    # Compute skew-symmetric S from P and Θ
    S .= 0.0
    get_S!(P_r, Θ_r, S_r)
end

function exp_skew!(X_ref; α=1)
    X = X_ref[]
    F = schur(X)
    v = norm.(F.values) .> 1e-15
    # ordschur!(F, v);
    F = ordschur(F, v)

    ang_n = div(sum(v), 2)

    angles = [F.T[2*ii-1, 2*ii] for ii = 1:ang_n]
    Z = F.Z

    if α != 1
        angles .*= α
    end


    Ans = zeros(size(Z))
    n = size(Ans)[1]
    # n_p = length(angles);
    n_p = count(x -> x != 0, angles)
    Z2 = Matrix{Float64}(undef, n, 2)
    Z2_ref = Ref(Z2)

    for ii = 1:n_p
        # Ans[2 * ii - 1, 2 * ii - 1] = cos(angles[ii]);
        # Ans[2 * ii - 1, 2 * ii] = sin(angles[ii]);
        # Ans[2 * ii, 2 * ii - 1] = -sin(angles[ii]);
        # Ans[2 * ii, 2 * ii] = cos(angles[ii]);
        Z2 .= Z[:, (2*ii-1):(2*ii)]
        PCSPT!(Z2_ref, angles[ii], Ref(Ans))
    end
    for ii = (2*n_p+1):n
        for jj = 1:n
            for kk = 1:(jj-1)
                t = Z[jj, ii] * Z[kk, ii]
                Ans[jj, kk] += t
                Ans[kk, jj] += t
            end
            Ans[jj, jj] += Z[jj, ii] * Z[jj, ii]
        end
    end
    # Ans = V * Ans * V';
    return Ans
end

function exp_skew_22!(Q::Ref{Matrix{Float64}}, S::Ref{Matrix{Float64}}; α::Float64=1.0)
    mQ = Q[]
    mS = S[]
    c = cos(α * mS[2, 1])
    s = sin(α * mS[2, 1])
    mQ[1, 1] = c
    mQ[2, 2] = c
    mQ[2, 1] = s
    mQ[1, 2] = -s
end


function exp_skew!(Q_r::Base.RefValue{Matrix{Float64}}, S_r::Base.RefValue{Matrix{Float64}}, wsp_skew_exp::WSP; α::Float64=1.0)
    # wsp_skew_exp carry n x n real matrices M, P, n x 2 real matrix U, V
    # real div(n, 2) vector Θ, boolean n vector v
    # M gets S
    Q = Q_r[]
    S = S_r[]

    if size(Q)[1] == 2
        exp_skew_22!(Q_r, S_r; α=α)
        return 0
    end

    M::Matrix{Float64} = retrieve(wsp_skew_exp, 1)
    P::Matrix{Float64} = retrieve(wsp_skew_exp, 2)
    Θ::Vector{Float64} = retrieve(wsp_skew_exp, 3)
    # v::Vector{Bool} = retrieve(wsp_skew_exp, 4)
    wsp_schur::WSP = retrieve(wsp_skew_exp, 4)

    P_r = wsp_skew_exp.vec[2]


    n::Int, = size(S)
    col_ind::Int = 0
    n_p::Int = 0
    temp::Float64 = 0.0
    n_Θ::Int = length(Θ)

    for ii = 1:n
        for jj = 1:n
            M[ii, jj] = S[ii, jj]
        end
    end

    real_schur_s!(wsp_skew_exp.vec[1], wsp_skew_exp.vec[2], wsp_skew_exp.vec[3], wsp_schur)


    if α != 1.0
        Θ .*= α
    end


    for ii = 1:n_Θ
        if Θ[ii] != 0.0
            n_p += 1
        end
    end

    Q .= 0.0
    for jj = 1:n_p
        # col_ind = 2 * jj - 1
        # for ii = 1:n
        #     U[ii, 1] = M[ii, col_ind]
        #     U[ii, 2] = M[ii, col_ind+1]
        #     # V[ii, 1] = U[ii, 1];
        #     # V[ii, 2] = V[ii, 2];
        # end
        PCSPT!(P_r, Θ[jj], Q_r, 2 * jj - 1)
    end

    i_beg::Int = (2 * n_p + 1)
    for ii = i_beg:n
        for jj = 1:n
            for kk = 1:(jj-1)
                temp = P[jj, ii] * P[kk, ii]

                Q[jj, kk] += temp
                Q[kk, jj] += temp

            end
            temp = P[jj, ii] * P[jj, ii]
            Q[jj, jj] += temp
        end

    end


end

function exp_skew(X; α=1)
    A = α
    X_ref = Ref(X)
    return exp_skew!(X_ref; α=A)
end


# function exp_skew(X; α = 1)0
#     return exp(X);
# end

function ret_skew(Uk, Up, X; α=-1)
    Ans = similar(Up)
    Ans .= Up
    if α == -1
        Ans .-= Up * X
    else
        Ans .+= α .* (Up * X)
    end

    Ans .-= Uk * (Uk' * Ans)
    factor = svd(Ans)
    Ans .= factor.U * factor.Vt
    return Ans
end

function real_schur_to_complex_eigen(Schur::AngularDecomposition)
    P = Schur.vectors

    n, = size(P)

    Z = Matrix{ComplexF64}(undef, n, n)
    Λ = Vector{ComplexF64}(undef, n)
    n_b = div(n, 2)
    A = zeros(n_b + 1)
    A[1:length(Schur.angles)] .= Schur.angles

    for ii = 1:n_b
        Z[:, (2*ii-1):(2*ii)] .= P[:, (2*ii-1):(2*ii)] * SO_Transformer
        Λ[2*ii-1] = -im * A[ii]
        Λ[2*ii] = im * A[ii]
    end
    if 2 * n_b != n
        Z[:, n] .= P[:, n]
        Λ[n] = 0
    end

    return Z, Λ
end


# function vad_s(X)
#     # This function return the real orthogonal matrix U and 2-2-block-diagonal real matrix E
#     # such that X = UEU', where E has blocks [0 θ_i; -θ_i 0] for some θ_i > 0.

#     assert(isSkewSym, X);
#     n, = size(X);

#     eigen_F = eigen(X);
#     perm = sortperm(abs.(imag.(eigen_F.values)))

#     Λ = eigen_F.values[perm];
#     U = eigen_F.vectors[:, perm];

#     if !isapprox(0, norm(U * U' .- diagm(0 => ones(n)), Inf), atol = KWARGS_SPEC_ORTH[1])
#         # The eigen-decomposition may lose the orthogonality in U, i.e., U U' \neq Id.
#         # It means U Λ inv(U) = X. Consider reorthogonalize U with qr: U = VW, then
#         # X = V (W Λ inv(W)) V' = V D V'. We assume D to be diagonal here.
#         d_msg("Loss of orthogonality detected in the eigendecomposition of skew-symmetric matrix. Performing reorthogonalization.")
#         V = Matrix(qr(U).Q);
#         W = V' * U;
#         D = W * diagm(0 => Λ) * inv(W);

#         assert(SO_ABSTOL_, >, x -> norm(x .- diagm(0 => diag(x)), Inf), D; 
#             debug_msg = (D, "The matrix W Λ inv(W) with U = VW is not diagonal. Unable to resume orthogonality."))

#         Λ .= diag(D);
#         perm = sortperm(abs.(imag.(Λ)));
#         Λ = Λ[perm];
#         U .= V[:, perm];
#         assert(SO_ABSTOL_, >, x -> norm(real.(x[1] * diagm(0 => x[2]) * x[1]') .- X, Inf), (U, Λ);
#             debug_msg = (U, Λ, "Unable to obtain eigendecomposition with unitary vectors!"));
#         d_msg(" Done!\n");
#     end


#     a_i = imag.(Λ);

#     vectors = zeros(size(U));
#     angles = zeros(n);

#     P = similar(U);
#     P .= zeros(n, n);
#     P_ind = 1;

#     U_col = 1;
#     V_col_p = 1;
#     V_col_0 = n;

#     while U_col <= n
#         if isapprox(0, a_i[U_col], atol = SO_ABSTOL_)
#             # The column is associated to 0 angles.
#             if isapprox(0, norm(imag.(U[:, U_col]), Inf), atol = SO_ABSTOL_)
#                 # The associated vector is real.
#                 # put it in V[:, V_col_0].
#                 vectors[:, V_col_0] .= real.(U[:, U_col]);
#                 V_col_0 -= 1;
#             else
#                 # The associated vector is complex. 
#                 P[:, P_ind] .= U[:, U_col];
#                 P_ind += 1;
#             end
#             U_col += 1;
#         else

#             k = 1;
#             while((U_col + k <= n) && isapprox(abs(a_i[U_col]), abs(a_i[U_col + k]), atol = SO_ABSTOL_))
#                 k += 1;
#             end
#             vectors[:, V_col_p:(V_col_p + k - 1)] .= COMPLEX_2_REAL_BASIS(U[:, U_col:(U_col + k - 1)]; 
#                 ang = a_i[U_col:(U_col + k - 1)]);
#             # For V is real basis converted from U,  UΛU' = VSV' where S should be real. 
#             # Needed to further translate V into VQ such that VSV' = VQ E Q'V' where E is block-diagonal.
#             Q = vectors[:, V_col_p:(V_col_p + k - 1)]' * U[:, U_col:(U_col + k - 1)];
#             S = Q * diagm(0 => Λ[U_col:(U_col + k - 1)]) * Q';
#             assert(SO_ABSTOL_, >, x -> norm(imag.(x), Inf), S;
#                 debug_msg = (S, "Real decomposition VSV' failed at getting real S."));

#             S_e = eigen(real.(S)); 
#             S_U = S_e.vectors;
#             S_L = S_e.values;
#             S_UUT = real.(S_U * S_U');
#             S_V = svd(S_UUT).U;
#             S_Q = S_V' * S_U;
#             S_E = S_Q * diagm(0 => S_L) * inv(S_Q);
#             assert(SO_ABSTOL_, >, x -> norm(x .- diagm(-1 => diag(x, -1), 1 => diag(x, 1)), Inf), S_E;
#                 debug_msg = (S, S_E, "Fail to convert S into block diagonal form."));
#             for ii = 1:div(k, 2)
#                 if S_E[2 * ii - 1, 2 * ii].re < 0
#                     temp = zeros(k);
#                     temp .= S_V[:, 2 * ii];
#                     S_V[:, 2 * ii] .= S_V[:, 2 * ii - 1];
#                     S_V[:, 2 * ii - 1] .= temp;
#                 end
#             end
#             vectors[:, V_col_p:(V_col_p + k - 1)] .= vectors[:, V_col_p:(V_col_p + k - 1)] * S_V;
#             angles[div(V_col_p + 1, 2):div(V_col_p + k - 1, 2)] .= abs(a_i[U_col]) .* ones(div(k, 2));

#             V_col_p += k;
#             U_col += k;
#         end
#     end

#     if P_ind > 1
#         vectors[:, (V_col_0 - P_ind + 2):V_col_0] = COMPLEX_2_REAL_BASIS(P[:, 1:(P_ind - 1)]);
#     end

#     return AngularDecomposition(vectors, angles[1:div(V_col_p - 1, 2)]);
# end

# function vad_q(X)
#     # This function return the real orthogonal matrix U and 2-2-block-diagonal real matrix E
#     # such that X = UEU', where E has blocks [cos(θ_i) sin(θ_i); -sin(θ_i) cos(θ_i)] for some θ_i > 0.

#     assert(isSpecOrth, X);
#     n, = size(X);

#     eigen_F = eigen(X);

#     Λ = eigen_F.values;
#     U = eigen_F.vectors;

#     if !isapprox(0, norm(U * U' .- diagm(0 => ones(n)), Inf), atol = SO_ABSTOL_)
#         # The eigen-decomposition may lose the orthogonality in U, i.e., U U' \neq Id.
#         # It means U Λ inv(U) = X. Consider reorthogonalize U with qr: U = VW, then
#         # X = V (W Λ inv(W)) V' = V D V'. We assume D to be diagonal here.
#         d_msg("Loss of orthogonality detected in the eigendecomposition of special orthogonal matrix. Performing reorthogonalization.")
#         V = Matrix(qr(U).Q);
#         W = V' * U;
#         D = W * diagm(0 => Λ) * inv(W);

#         assert(SO_ABSTOL_, >, x -> norm(x .- diagm(0 => diag(x)), Inf), D, 
#             debug_msg = (D, "The matrix is not diagonal. Unable to resume orthogonality!"));

#         Λ .= diag(D);
#         U .= V;
#         d_msg(" Done!\n");
#     end

#     Λ = log.(Λ);

#     a_i = zeros(n);
#     #a_i .= atan.(imag.(Λ) ./ real.(Λ));
#     a_i .= imag.(Λ);

#     perm = sortperm(abs.(a_i));
#     a_i = a_i[perm];
#     U = U[:, perm];
#     Λ = Λ[perm];

#     vectors = zeros(size(U));
#     angles = zeros(n);

#     U_col = 1;
#     V_col_p = 1;
#     V_col_0 = n;

#     P = similar(U);
#     P .= zeros(n, n);
#     P_ind = 1;


#     while U_col <= n
#         if isapprox(0, a_i[U_col], atol = SO_ABSTOL_)
#             # The column is associated to 0 angles.
#             if !isapprox(0, norm(imag.(U[:, U_col]), Inf), atol = SO_ABSTOL_)
#                 # The associated vector is not real, collect them and translate together.
#                 P[:, P_ind] .= U[:, U_col];
#                 P_ind += 1;
#             else
#                 # The associated vector is real
#                 vectors[:, V_col_0] .= real.(U[:, U_col]);
#                 V_col_0 -= 1;
#             end
#             U_col += 1;
#         else

#             k = 1;
#             while((U_col + k <= n) && isapprox(abs(a_i[U_col]), abs(a_i[U_col + k]), atol = SO_ABSTOL_))
#                 k += 1;
#             end

#             vectors[:, V_col_p:(V_col_p + k - 1)] .= COMPLEX_2_REAL_BASIS(U[:, U_col:(U_col + k - 1)]; 
#                 ang = a_i[U_col:(U_col + k - 1)]);

#             Q = vectors[:, V_col_p:(V_col_p + k - 1)]' * U[:, U_col:(U_col + k - 1)];
#             S = Q * diagm(0 => Λ[U_col:(U_col + k - 1)]) * Q';
#             assert(SO_ABSTOL_, >, x -> norm(imag.(x), Inf), S;
#                 debug_msg = (S, "Real decomposition VSV' failed at getting real S."));

#             S_e = eigen(real.(S)); 
#             S_U = S_e.vectors;
#             S_L = S_e.values;
#             S_UUT = real.(S_U * S_U');
#             S_V = svd(S_UUT).U;
#             S_Q = S_V' * S_U;
#             S_E = S_Q * diagm(0 => S_L) * inv(S_Q);
#             assert(SO_ABSTOL_, >, x -> norm(x .- diagm(-1 => diag(x, -1), 1 => diag(x, 1)), Inf), S_E;
#                    debug_msg = (S, S_E, "Fail to convert S into block diagonal form."));
#             for ii = 1:div(k, 2)
#                 if S_E[2 * ii - 1, 2 * ii].re < 0
#                     temp = zeros(k);
#                     temp .= S_V[:, 2 * ii];
#                     S_V[:, 2 * ii] .= S_V[:, 2 * ii - 1];
#                     S_V[:, 2 * ii - 1] .= temp;
#                 end
#             end
#             vectors[:, V_col_p:(V_col_p + k - 1)] .= vectors[:, V_col_p:(V_col_p + k - 1)] * S_V;

#             angles[div(V_col_p + 1, 2):div(V_col_p + k - 1, 2)] .= abs(a_i[U_col]) .* ones(div(k, 2));

#             V_col_p += k;
#             U_col += k;

#         end
#     end

#     if P_ind > 1
#         vectors[:, (V_col_0 - P_ind + 2):V_col_0] = COMPLEX_2_REAL_BASIS(P[:, 1:(P_ind - 1)]);
#     end

#     return AngularDecomposition(vectors, angles[1:div(V_col_p - 1, 2)]);
# end
