# This file implements the Schur angular factorization (SAFactor) of special orthogonal matrices and skew symmetric matrices.
# The Schur factorization 

using LoopVectorization

include("../../inc/global_path.jl")
include(joinpath(JULIA_LAPACK_PATH, "congruence.jl"))
include(joinpath(JULIA_INCLUDE_PATH, "workspace.jl"))
include(joinpath(JULIA_LAPACK_PATH, "dgees.jl"))

import Base: show


SAF_ANGLE_ABSTOL = 1e-15

mutable struct SAFactor
    vector::Ref{Matrix{Float64}}
    angle::Ref{Vector{Float64}}
    nza_cnt::Int
    SAFactor(n::Int) = new(Ref(Matrix{Float64}(undef, n, n)), Ref(Vector{Float64}(undef, div(n, 2))), 0)
    SAFactor(V::Ref{Matrix{Float64}}, A::Ref{Vector{Float64}}, n::Int) = new(V, A, n)
end

function Base.show(io::IO, p::SAFactor)
    print(io, "Real Schur Angular Decomposition of skew symmetric/special orthogonal matrices:\n")
    print(io, "Real Schur vectors\n")
    display(p.vector[])
    print(io, "Angles:\n")
    display(p.angle[])
    print(io, "Nonzero angles count:\n")
    display(p.nza_cnt)
end


getVector(SAF::SAFactor) = SAF.vector[]
getAngle(SAF::SAFactor) = SAF.angle[]

SO_Transformer = (sqrt(2) / 2) .* [1.0 1.0; -1.0im 1.0im];

get_wsp_saf(n, wsp_dgees) = WSP(Matrix{Float64}(undef, n, n), Vector{Int}(undef, div(n, 2)), wsp_dgees)
get_wsp_saf(n) = WSP(Matrix{Float64}(undef, n, n), Vector{Int}(undef, div(n, 2)), get_wsp_dgees(n))

get_wsp_saf_reg(n) = WSP(Matrix{Float64}(undef, n, n))
get_wsp_saf_ord(n) = WSP(Matrix{Float64}(undef, n, n), Vector{Int}(undef, div(n, 2)))





function SAFactor_order(saf::SAFactor, wsp_saf_ord::WSP=WSP(similar(saf.vector[]), similar(saf.angle[], Int)))
    # Flip negative angles to positve by flipping the corresponding pair of real Schur vectors.
    # M will be reused as workspace as the angles information has been properly stored in saf.
    Mat = wsp_saf_ord[1]
    # Order the angles ay magnitude
    p = wsp_saf_ord[2]             # div(n, 2) int vector.
    len = 2 * size(Mat, 1)     # operating pairs of vectors
    sVec = getVector(saf)
    ang = getAngle(saf)
    sortperm!(p, ang, by=abs, rev=true)
    unsafe_copyto!(pointer(Mat), pointer(ang), length(ang))
    for ind in eachindex(ang)   # Reorder angles
        @inbounds ang[ind] = Mat[p[ind]]
    end

    unsafe_copyto!(pointer(Mat), pointer(sVec), length(sVec))
    for ind in eachindex(ang)   # Reorder pairs of Schur vectors
        @inbounds if ind == p[ind]
            continue
        else
            unsafe_copyto!(sVec, len * (ind - 1) + 1, Mat, len * (p[ind] - 1) + 1, len)
        end
    end
end

function SAFactor_regularize(saf::SAFactor, wsp_saf_reg::WSP=WSP(similar(saf.vector[])))
    # Flip negative angles to positve by flipping the corresponding pair of real Schur vectors.
    # M will be reused as workspace as the angles information has been properly stored in saf.
    Mat = wsp_saf_reg[1]
    len = size(Mat, 1)         # operating individal vectors within a pair
    sVec = getVector(saf)
    ang = getAngle(saf)

    unsafe_copyto!(pointer(Mat), pointer(sVec), length(sVec))
    for ind in 1:saf.nza_cnt   # Reorder pairs of Schur vectors
        @inbounds if ang[ind] < 0
            @inbounds ang[ind] = -ang[ind]
            @inbounds unsafe_copyto!(Mat, 1, sVec, len * (2 * ind - 2) + 1, len)
            @inbounds unsafe_copyto!(sVec, len * (2 * ind - 2) + 1, sVec, len * (2 * ind - 1) + 1, len)
            @inbounds unsafe_copyto!(sVec, len * (2 * ind - 1) + 1, Mat, 1, len)
        end
    end
end


"""
    schurAngular!(saf, M, wsp; order, regular) -> saf

    Real Schur factorization for skew symmetric matrices, M, and return the `SAFactor` object with real Schur vectors and angles. 
    The matrix M is used as workspace also and become block diagonal matrix on exit.
"""
function schurAngular_SkewSymm!(saf::SAFactor, M::Ref{Matrix{Float64}}, wsp_saf::WSP; order::Bool=true, regular::Bool=false)

    MatTmp = wsp_saf[1]
    MatM = M[]
    unsafe_copyto!(pointer(MatTmp), pointer(MatM), length(MatM))

    dgees!(wsp_saf(1), saf.vector, wsp_saf[3]; job='V', sort='S')


    saf.nza_cnt = 0
    ang = getAngle(saf)
    for ind in eachindex(ang)
        @inbounds ang[ind] = MatTmp[2*ind, 2*ind-1]
    end

    for ind in eachindex(ang)
        @inbounds if abs(ang[ind]) > SAF_ANGLE_ABSTOL
            saf.nza_cnt += 1
        else
            @inbounds ang[ind] = 0.0
        end
    end

    if order
        SAFactor_order(saf, wsp_saf)
    end

    if regular
        SAFactor_regularize(saf, wsp_saf)
    end

    # if order
    #     # Order the angles ay magnitude
    #     # M will be reused as workspace as the angles information has been properly stored in saf.
    #     p = wsp_saf[1];             # div(n, 2) int vector.
    #     len = 2 * size(Mat, 1);     # operating pairs of vectors

    #     sVec = getVector(saf)

    #     sortperm!(p, ang, by=abs, rev=true);

    #     # display(ang')
    #     # display(p')

    #     unsafe_copyto!(pointer(Mat), pointer(ang), length(ang));
    #     for ind in eachindex(ang)   # Reorder angles
    #         @inbounds ang[ind] = Mat[p[ind]];
    #     end

    #     unsafe_copyto!(pointer(Mat), pointer(sVec), length(sVec));
    #     for ind in eachindex(ang)   # Reorder pairs of Schur vectors
    #         @inbounds if ind == p[ind]
    #             continue;
    #         else
    #             # println("Index:\t\t$(ind)")
    #             # println("destination offset:\t$(len * (ind - 1) + 1)")
    #             # println("source offset:\t\t$(len * (p[ind] - 1) + 1)")
    #             unsafe_copyto!(sVec, len * (ind - 1) + 1, Mat, len * (p[ind] - 1) + 1, len);
    #         end
    #     end
    # end

    # if regular
    #     # Flip negative angles to positve by flipping the corresponding pair of real Schur vectors.
    #     # M will be reused as workspace as the angles information has been properly stored in saf.

    #     len = size(Mat, 1);         # operating individal vectors within a pair

    #     sVec = getVector(saf)

    #     unsafe_copyto!(Mat, 1, sVec, 1, length(sVec));
    #     for ind in 1:saf.nza_cnt   # Reorder pairs of Schur vectors
    #         @inbounds if ang[ind] < 0
    #             @inbounds ang[ind] = -ang[ind];
    #             @inbounds unsafe_copyto!(Mat, 1, sVec, len * (2 * ind - 2) + 1, len);
    #             @inbounds unsafe_copyto!(sVec, len * (2 * ind - 2) + 1, sVec, len * (2 * ind - 1) + 1, len);
    #             @inbounds unsafe_copyto!(sVec, len * (2 * ind - 1) + 1, Mat, 1, len);
    #         end
    #     end
    # end
end

function schurAngular_SpecOrth!(saf::SAFactor, M::Ref{Matrix{Float64}}, wsp_saf::WSP; order::Bool=true, regular::Bool=false)

    MatTmp = wsp_saf[1]
    MatM = M[]
    unsafe_copyto!(pointer(MatTmp), pointer(MatM), length(MatM))

    dgees!(wsp_saf(1), saf.vector, wsp_saf[3]; job='V', sort='S')

    saf.nza_cnt = 0
    ang = getAngle(saf)
    for ind in eachindex(ang)
        @inbounds ang[ind] = atan(MatTmp[2*ind, 2*ind-1], MatTmp[2*ind, 2*ind])
    end

    for ind in eachindex(ang)
        @inbounds if abs(ang[ind]) > SAF_ANGLE_ABSTOL
            saf.nza_cnt += 1
        else
            @inbounds ang[ind] = 0.0
        end
    end

    if order
        SAFactor_order(saf, wsp_saf)
    end

    if regular
        SAFactor_regularize(saf, wsp_saf)
    end

    # if order
    #     # Order the angles ay magnitude
    #     # M will be reused as workspace as the angles information has been properly stored in saf.
    #     p = wsp_saf[1];             # div(n, 2) int vector.
    #     len = 2 * size(Mat, 1);     # operating pairs of vectors

    #     sVec = getVector(saf)

    #     sortperm!(p, ang, by=abs, rev=true);

    #     unsafe_copyto!(pointer(Mat), pointer(ang), length(ang));
    #     for ind in eachindex(ang)   # Reorder angles
    #         @inbounds ang[ind] = Mat[p[ind]];
    #     end

    #     unsafe_copyto!(pointer(Mat), pointer(sVec), length(sVec));
    #     for ind in eachindex(ang)   # Reorder pairs of Schur vectors
    #         @inbounds if ind == p[ind]
    #             continue;
    #         else
    #             @inbounds unsafe_copyto!(sVec, len * (ind - 1) + 1, Mat, len * (p[ind] - 1) + 1, len);
    #         end
    #     end
    # end

    # if regular
    #     # Flip negative angles to positve by flipping the corresponding pair of real Schur vectors.
    #     # M will be reused as workspace as the angles information has been properly stored in saf.

    #     len = size(Mat, 1);         # operating individal vectors within a pair

    #     sVec = getVector(saf)

    #     unsafe_copyto!(pointer(Mat), pointer(sVec), length(sVec));
    #     for ind in 1:saf.nza_cnt   # Reorder pairs of Schur vectors
    #         @inbounds if ang[ind] < 0
    #             @inbounds ang[ind] = -ang[ind];
    #             @inbounds unsafe_copyto!(Mat, 1, sVec, len * (2 * ind - 2) + 1, len);
    #             @inbounds unsafe_copyto!(sVec, len * (2 * ind - 2) + 1, sVec, len * (2 * ind - 1) + 1, len);
    #             @inbounds unsafe_copyto!(sVec, len * (2 * ind - 1) + 1, Mat, 1, len);
    #         end
    #     end
    # end
end


function schurAngular_SkewSymm(M::Ref{Matrix{Float64}}, wsp_saf::WSP=get_wsp_saf(size(M[], 1)); order::Bool=true, regular::Bool=false)
    MatM = M[]
    MatW = copy(MatM)
    W = Ref(MatW)
    saf = SAFactor(size(MatM, 1))
    schurAngular_SkewSymm!(saf, W, wsp_saf; order=order, regular=regular)
    return saf
end

function schurAngular_SpecOrth(M::Ref{Matrix{Float64}}, wsp_saf::WSP=get_wsp_saf(size(M[], 1)); order::Bool=true, regular::Bool=false)
    MatM = M[]
    MatW = copy(MatM)
    W = Ref(MatW)
    saf = SAFactor(size(MatM, 1))
    schurAngular_SpecOrth!(saf, W, wsp_saf; order=order, regular=regular)
    return saf
end

# function schurAngular_SpecOrth!(saf::SAFactor, M::Ref{Matrix{Float64}}, wsp_saf::WSP; order::Bool=true, regular::Bool=false)
#     dgees!(M, saf.vector, wsp_saf[2]; job = 'V', sort = 'S')

#     Mat = M[];

#     saf.nza_cnt = 0
#     ang = getAngle(saf);
#     @tturbo for ind in eachindex(ang)
#         ang[ind] = atan(Mat[2*ind - 1, 2*ind], Mat[2*ind, 2*ind])
#     end

#     @inbounds @simd for ind in eachindex(ang)
#         if abs(ang[ind]) > SAF_ANGLE_ABSTOL
#             saf.nza_cnt += 1
#         else
#             ang[ind] = 0.0;
#         end
#     end

#     if order
#         # Order the angles ay magnitude
#         # M will be reused as workspace as the angles information has been properly stored in saf.
#         p = wsp_saf[1];             # div(n, 2) int vector.
#         len = 2 * size(Mat, 1);     # operating pairs of vectors

#         sVec = getVector(saf)

#         sortperm!(p, ang, by=abs, rev=true);

#         copyto!(Mat, ang);
#         @tturbo for ind in eachindex(ang)   # Reorder angles
#             ang[ind] = W[p[ind]];
#         end

#         copyto!(Mat, sVec);
#         @tturbo for ind in eachindex(ang)   # Reorder pairs of Schur vectors
#             if ind == p[ind]
#                 continue;
#             end
#             copyto!(sVec, len * (ind - 1) + 1, Mat, len * (p[ind] - 1) + 1, len);
#         end
#     end

#     if regular
#         # Flip negative angles to positve by flipping the corresponding pair of real Schur vectors.
#         # M will be reused as workspace as the angles information has been properly stored in saf.

#         len = size(Mat, 1);         # operating individal vectors within a pair

#         sVec = getVector(saf)

#         copyto!(Mat, sVec);
#         @tturbo for ind in eachindex(ang)   # Reorder pairs of Schur vectors
#             if ang[ind] < 0
#                 ang[ind] = -ang[ind];
#                 copyto!(Mat, 1, sVec, len * (2 * p[ind] - 2) + 1, len);
#                 copyto!(sVec, len * (2 * p[ind] - 2) + 1, sVec, len * (2 * p[ind] - 1) + 1, len);
#                 copyto!(sVec, len * (2 * p[ind] - 1) + 1, Mat, 1, len);
#         end
#     end
# end

# function congruenceAngle!(M::Ref{Matrix{Float64}}, P::Ref{Matrix{Float64}}, A::Ref{Vector{Float64}}, scale::Float64 = 1.0, m::Int = length(A[]); transpose::Bool = false)
#     MatM = M[];
#     MatP = P[];
#     VecA = A[];

#     fill!(MatM, 0.0)

#     if transpose
#         for a_ind = 1:m
#             # Using view to access column. As Julia stores matrix in column major order, it is possible to access them by raw pointers
#             # and then call a self-warpped ger from BLAS, just like how dgees is done in myLA. However, when it comes to P^T A P,
#             # accessing rows needs extra care. So it is not just a ger warp, but also row access issue. Therefore, we put this on hold
#             # and simply use view that has both access. In this case, the Julia built-in ger! can be used.
#             @inbounds BLAS.ger!(scale * VecA[a_ind], view(MatP, 2 * a_ind, :), view(MatP, 2 * a_ind - 1, :), MatM)
#             # @inbounds BLAS.ger!(-VecA[a_ind], view(MatP, 2 * a_ind - 1, :), view(MatP, 2 * a_ind, :), MatM) # Reduced by skew-symmetry.

#         end
#     else
#         for a_ind = 1:m
#             # Using view to access column. As Julia stores matrix in column major order, it is possible to access them by raw pointers
#             # and then call a self-warpped ger from BLAS, just like how dgees is done in myLA. However, when it comes to P^T A P,
#             # accessing rows needs extra care. So it is not just a ger warp, but also row access issue. Therefore, we put this on hold
#             # and simply use view that has both access. In this case, the Julia built-in ger! can be used.

#             @inbounds BLAS.ger!(scale * VecA[a_ind], view(MatP, :, 2 * a_ind), view(MatP, :, 2 * a_ind - 1), MatM)
#             # @inbounds BLAS.ger!(-VecA[a_ind], view(MatP, :, 2 * a_ind - 1), view(MatP, :, 2 * a_ind), MatM) # Reduced by skew-symmetry

#         end
#     end

#     for c_ind = axes(MatM, 2)
#         for r_ind = (c_ind + 1):size(MatM, 1)
#             @inbounds MatM[r_ind, c_ind] -= MatM[c_ind, r_ind];
#             @inbounds MatM[c_ind, r_ind] = -MatM[r_ind, c_ind];
#         end
#         MatM[c_ind, c_ind] = 0.0;
#     end
# end

# function congruenceCosSin!(M::Ref{Matrix{Float64}}, P::Ref{Matrix{Float64}}, A::Ref{Vector{Float64}}, m::Int = length(A[]); transpose::Bool = false, scale::Float64 = 1.0)
#     MatM = M[];
#     MatP = P[];
#     VecA = A[];

#     fill!(MatM, 0.0)

#     if transpose
#         for a_ind = 1:m
#             # Using view to access column. As Julia stores matrix in column major order, it is possible to access them by raw pointers
#             # and then call a self-warpped ger from BLAS, just like how dgees is done in myLA. However, when it comes to P^T A P,
#             # accessing rows needs extra care. So it is not just a ger warp, but also row access issue. Therefore, we put this on hold
#             # and simply use view that has both access. In this case, the Julia built-in ger! can be used.
#             @inbounds BLAS.ger!(sin(scale * VecA[a_ind]), view(MatP, 2 * a_ind, :), view(MatP, 2 * a_ind - 1, :), MatM)
#             # @inbounds BLAS.ger!(-VecA[a_ind], view(MatP, 2 * a_ind - 1, :), view(MatP, 2 * a_ind, :), MatM) # Reduced by skew-symmetry.

#         end
#     else
#         for a_ind = 1:m
#             # Using view to access column. As Julia stores matrix in column major order, it is possible to access them by raw pointers
#             # and then call a self-warpped ger from BLAS, just like how dgees is done in myLA. However, when it comes to P^T A P,
#             # accessing rows needs extra care. So it is not just a ger warp, but also row access issue. Therefore, we put this on hold
#             # and simply use view that has both access. In this case, the Julia built-in ger! can be used.

#             @inbounds BLAS.ger!(sin(scale * VecA[a_ind]), view(MatP, :, 2 * a_ind), view(MatP, :, 2 * a_ind - 1), MatM)
#             # @inbounds BLAS.ger!(-VecA[a_ind], view(MatP, :, 2 * a_ind - 1), view(MatP, :, 2 * a_ind), MatM) # Reduced by skew-symmetry

#         end
#     end

#     for c_ind = axes(MatM, 2)
#         for r_ind = (c_ind + 1):size(MatM, 1)
#             @inbounds MatM[r_ind, c_ind] -= MatM[c_ind, r_ind];
#             @inbounds MatM[c_ind, r_ind] = -MatM[r_ind, c_ind];
#         end
#         MatM[c_ind, c_ind] = 0.0;
#     end

#     d_ind::Int = 1
#     for a_ind in 1:m
#         @inbounds BLAS.ger!(cos(scale * VecA[a_ind]), view(MatP, :, 2 * a_ind - 1), view(MatP, :, 2 * a_ind - 1), MatM)
#         @inbounds BLAS.ger!(cos(scale * VecA[a_ind]), view(MatP, :, 2 * a_ind), view(MatP, :, 2 * a_ind), MatM)
#     end
#     for d_ind in (2m + 1):size(MatM, 1)
#         @inbounds BLAS.ger!(1.0, view(MatP, :, d_ind), view(MatP, :, d_ind), MatM);
#     end

#     # if transpose
#         # for a_ind in 1:m
#         #     @inbounds cosine = cos(scale * VecA[a_ind])
#         #     @inbounds sine = sin(scale * VecA[a_ind])
#         #     for r_ind in axes(MatM, 1)
#         #         for c_ind in 1:(r_ind - 1)
#         #             @inbounds t1 = cosine * (MatP[2 * a_ind - 1, r_ind] * MatP[2 * a_ind - 1, c_ind] + MatP[2 * a_ind, r_ind] * MatP[2 * a_ind, c_ind]);
#         #             @inbounds t2 = sine * (MatP[2 * a_ind, r_ind] * MatP[2 * a_ind - 1, c_ind] - MatP[2 * a_ind - 1, r_ind] * MatP[2 * a_ind, c_ind]);
#         #             @inbounds MatM[r_ind, c_ind] += t1 + t2;
#         #             @inbounds MatM[c_ind, r_ind] += t1 - t2;
#         #         end
#         #         # c_ind = r_ind, t_2 = 0
#         #         @inbounds t1 = cosine * (MatP[2 * a_ind - 1, r_ind] * MatP[2 * a_ind - 1, r_ind] + MatP[2 * a_ind, r_ind] * MatP[2 * a_ind, r_ind]);
#         #         MatM[r_ind, r_ind] += t1;
#         #     end
#         # end

#         # if 2 * m != n
#         #     for v_ind = (2 * m + 1):n
#         #         for r_ind in axes(MatM, 1)
#         #             for c_ind in 1:(r_ind - 1)
#         #                 @inbounds temp = MatP[v_ind, r_ind] * MatP[v_ind, c_ind];
#         #                 @inbounds MatM[r_ind, c_ind] += temp;
#         #                 @inbounds MatM[c_ind, r_ind] += temp;
#         #             end
#         #             # c_ind = r_ind
#         #             @inbounds temp = MatP[v_ind, r_ind] * MatP[v_ind, r_ind]
#         #             MatM[r_ind, r_ind] += temp;
#         #         end
#         #     end
#         # end
#     # else

#         # for a_ind in 1:m
#         #     @inbounds cosine = cos(scale * VecA[a_ind])
#         #     @inbounds sine = sin(scale * VecA[a_ind])
#         #     for r_ind in axes(MatM, 1)
#         #         for c_ind in 1:(r_ind - 1)
#         #             @inbounds t1 = cosine * (MatP[r_ind, 2 * a_ind - 1] * MatP[c_ind, 2 * a_ind - 1] + MatP[r_ind, 2 * a_ind] * MatP[c_ind, 2 * a_ind]);
#         #             @inbounds t2 = sine * (MatP[r_ind, 2 * a_ind] * MatP[c_ind, 2 * a_ind - 1] - MatP[r_ind, 2 * a_ind - 1] * MatP[c_ind, 2 * a_ind]);
#         #             @inbounds MatM[r_ind, c_ind] += t1 + t2;
#         #             @inbounds MatM[c_ind, r_ind] += t1 - t2;
#         #         end
#         #         # c_ind = r_ind, t_2 = 0
#         #         @inbounds t1 = cosine * (MatP[r_ind, 2 * a_ind - 1] * MatP[r_ind, 2 * a_ind - 1] + MatP[r_ind, 2 * a_ind] * MatP[r_ind, 2 * a_ind]);
#         #         MatM[r_ind, r_ind] += t1;
#         #     end
#         # end

#         # if 2 * m != n
#         #     for v_ind = (2 * m + 1):n
#         #         for r_ind in axes(MatM, 1)
#         #             for c_ind in 1:(r_ind - 1)
#         #                 @inbounds temp = MatP[r_ind, v_ind] * MatP[c_ind, v_ind];
#         #                 @inbounds MatM[r_ind, c_ind] += temp;
#         #                 @inbounds MatM[c_ind, r_ind] += temp;
#         #             end
#         #             # c_ind = r_ind
#         #             @inbounds temp = MatP[r_ind, v_ind] * MatP[r_ind, v_ind]
#         #             MatM[r_ind, r_ind] += temp;
#         #         end
#         #     end
#         # end
#     # end
# end

# function congruenceSkewSymm!(M::Ref{Matrix{Float64}}, P::Ref{Matrix{Float64}}, S::Ref{Matrix{Float64}}; transpose::Bool = false, scale::Float64 = 1.0)
#     MatM = M[];
#     MatP = P[];
#     MatS = S[];

#     fill!(MatM, 0.0)

#     if transpose
#         for Si in axes(MatS, 1)
#             for Sj = 1:(Si-1)
#                 for Xi in axes(MatP, 1)
#                     for Xj = 1:(Xi-1)
#                         @inbounds temp = scale * MatS[Si, Sj] * (MatP[Xi, Si] * MatP[Xj, Sj] - MatP[Xj, Si] * MatP[Xi, Sj])
#                         @inbounds MatM[Xi, Xj] += temp
#                         @inbounds MatM[Xj, Xi] -= temp
#                     end
#                 end
#             end
#         end
#     else
#         for Si in axes(MatS, 1)
#             for Sj = 1:(Si-1)
#                 for Xi in axes(MatP, 1)
#                     for Xj = 1:(Xi-1)
#                         @inbounds temp = scale * MatS[Si, Sj] * (MatP[Si, Xi] * MatP[Sj, Xj] - MatP[Si, Xj] * MatP[Sj, Xi])
#                         @inbounds MatM[Xi, Xj] += temp
#                         @inbounds MatM[Xj, Xi] -= temp
#                     end
#                 end
#             end
#         end
#     end
# end


@inline computeSkewSymm!(M::Ref{Matrix{Float64}}, saf::SAFactor, scale::Float64) = cong_SkewSymm_Angle!(M, saf.vector, saf.angle, saf.nza_cnt, scale; trans=false)
@inline computeSkewSymm!(M::Ref{Matrix{Float64}}, saf::SAFactor) = cong_SkewSymm_Angle!(M, saf.vector, saf.angle, saf.nza_cnt; trans=false)

@inline computeSpecOrth!(M::Ref{Matrix{Float64}}, saf::SAFactor, scale::Float64) = cong_SpecOrth_Angle!(M, saf.vector, saf.angle, saf.nza_cnt, scale; trans=false)
@inline computeSpecOrth!(M::Ref{Matrix{Float64}}, saf::SAFactor) = cong_SpecOrth_Angle!(M, saf.vector, saf.angle, saf.nza_cnt; trans=false)

function computeSkewSymm(saf::SAFactor, scale::Float64)
    n = size(saf.vector[], 1)
    M = Matrix{Float64}(undef, n, n)
    computeSkewSymm!(Ref(M), saf, scale)
    return M
end

function computeSkewSymm(saf::SAFactor)
    n = size(saf.vector[], 1)
    M = Matrix{Float64}(undef, n, n)
    computeSkewSymm!(Ref(M), saf)
    return M
end

function computeSpecOrth(saf::SAFactor, scale::Float64)
    n = size(saf.vector[], 1)
    M = Matrix{Float64}(undef, n, n)
    computeSpecOrth!(Ref(M), saf, scale)
    return M
end

function computeSpecOrth(saf::SAFactor)
    n = size(saf.vector[], 1)
    M = Matrix{Float64}(undef, n, n)
    computeSpecOrth!(Ref(M), saf)
    return M
end

#######################################Test functions#######################################

using BenchmarkTools

function test_congruence(n=10)
    A = rand(div(n, 2))
    P = rand(n, n)
    S = rand(n, n)
    S .-= S'

    DA = zeros(n, n)
    DCS = zeros(n, n)
    for a_ind in eachindex(A)
        ang = A[a_ind]
        DA[2*a_ind, 2*a_ind-1] = ang
        DA[2*a_ind-1, 2*a_ind] = -ang

        DCS[2*a_ind, 2*a_ind-1] = sin(ang)
        DCS[2*a_ind-1, 2*a_ind] = -sin(ang)
        DCS[2*a_ind-1, 2*a_ind-1] = cos(ang)
        DCS[2*a_ind, 2*a_ind] = cos(ang)
    end
    if isodd(n)
        DCS[n, n] = 1.0
    end

    congA = similar(P)
    congruenceAngle!(Ref(congA), Ref(P), Ref(A))
    PAPT = P * DA * P'

    congCS = similar(P)
    congruenceCosSin!(Ref(congCS), Ref(P), Ref(A))
    PCSPT = P * DCS * P'


    println(PAPT ≈ congA)
    println(PCSPT ≈ congCS)

    congruenceAngle!(Ref(congA), Ref(P), Ref(A); transpose=true)
    PTAP = P' * DA * P

    congruenceCosSin!(Ref(congCS), Ref(P), Ref(A); transpose=true)
    PTCSP = P' * DCS * P

    println(PTAP ≈ congA)
    println(PTCSP ≈ congCS)

end

function test_congSkewSymm_speed(n=10)
    # congruenceSkewSymm is bad for n > 12
    MatP = rand(n, n)
    MatS = rand(n, n)
    MatS .-= MatS'

    MatM = similar(MatS)

    cS1 = similar(MatS)
    cS2 = similar(MatS)

    S = Ref(MatS)
    P = Ref(MatP)
    cS = Ref(cS1)


    println("congruenceSkewSymm computation time:")
    @time congruenceSkewSymm!(cS, P, S; transpose=false)

    println("direct matrix multiplication time:")
    @time begin
        mul!(MatM, MatP', MatS)
        mul!(cS2, MatM, MatP)
    end

    println("Same result?\t", cS1 ≈ cS2)
end

function test_congAngle_speed(n=10)
    # congruenceSkewSymm is bad for n > 12
    MatP = rand(n, n)
    VecA = rand(div(n, 2))

    MatA = similar(MatP)
    MatM = similar(MatP)

    cS1 = similar(MatP)
    cS2 = similar(MatP)

    A = Ref(VecA)
    P = Ref(MatP)
    cS = Ref(cS1)


    println("congruenceAngle computation time:")
    @time congruenceAngle!(cS, P, A; transpose=false)

    println("direct matrix multiplication time:")
    @time begin
        fill!(MatA, 0.0)
        for ind in eachindex(VecA)
            MatA[2*ind-1, 2*ind] = -VecA[ind]
            MatA[2*ind, 2*ind-1] = VecA[ind]
        end
        mul!(MatM, MatP, MatA)
        mul!(cS2, MatM, MatP')
    end

    println("Same result?\t", cS1 ≈ cS2)
end

function test_congCosSin_speed(n=10)
    # congruenceSkewSymm is bad for n > 12
    MatP = rand(n, n)
    VecA = rand(div(n, 2))

    MatA = similar(MatP)
    MatM = similar(MatP)

    cS1 = similar(MatP)
    cS2 = similar(MatP)

    A = Ref(VecA)
    P = Ref(MatP)
    cS = Ref(cS1)


    println("congruenceAngle computation time:")
    @time congruenceCosSin!(cS, P, A; transpose=false)

    println("direct matrix multiplication time:")
    @time begin
        fill!(MatA, 0.0)
        for ind in eachindex(VecA)
            MatA[2*ind-1, 2*ind] = -sin(VecA[ind])
            MatA[2*ind, 2*ind-1] = sin(VecA[ind])
            MatA[2*ind-1, 2*ind-1] = cos(VecA[ind])
            MatA[2*ind, 2*ind] = cos(VecA[ind])
        end
        if isodd(n)
            MatA[n, n] = 1.0
        end
        mul!(MatM, MatP, MatA)
        mul!(cS2, MatM, MatP')
    end

    println("Same result?\t", cS1 ≈ cS2)
end

function test_saf(n=10)
    M = rand(n, n)
    M .-= M'
    M .*= 4π
    eM = exp(M)
    M_saf = schurAngular_SkewSymm(Ref(M))
    eM_saf = schurAngular_SpecOrth(Ref(eM))
    eMr_saf = schurAngular_SpecOrth(Ref(eM); order=true, regular=true)



    M_f = computeSkewSymm(M_saf)
    eM_f = computeSpecOrth(eM_saf)
    eMr_f = computeSpecOrth(eMr_saf)

    leM_f = computeSkewSymm(eM_saf)


    println(M_f ≈ M)
    println(eM_f ≈ eM)
    println(leM_f ≈ M)
    println(exp(leM_f) ≈ eM)

    display(eM_saf)
    display(eMr_saf)
    println(eM_f ≈ eMr_f)

end

function test_saf_speed(n=10)
    M = rand(n, n)
    M .-= M'

    println("Eigen decomposition computation time:")
    @btime eigen($M)

    println("Real Schur decomposition computation time:")
    @btime schurAngular_SkewSymm(Ref($M); regular=false, order=false)
    return nothing
end






