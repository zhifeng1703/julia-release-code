include(joinpath(dirname(dirname(@__FILE__)), "workspace.jl"))
include("LAPACK_setup.jl")



@inline get_wsp_cong(n::Int, m::Int) = WSP(Matrix{Float64}(undef, n, m));
@inline get_wsp_cong(n::Int) = WSP(Matrix{Float64}(undef, n, n));

function cong_dense!(M::Ref{Matrix{Float64}}, P::Ref{Matrix{Float64}}, pos::Int, C::Ref{Matrix{Float64}}, cos::Int, dim::Int, wsp_cong::WSP=get_wsp_cong(size(P[], 1), dim); trans=false)
    MatTemp = wsp_cong[1]
    MatM = M[]
    MatP = P[]
    MatC = C[]

    # full_dim = size(MatM, 1);
    # core_dim = ed - os;
    if trans
        mul!(MatTemp, view(MatP, (pos+1):(pos+dim), :)', view(MatC, (cos+1):(cos+dim), (cos+1):(cos+dim)))
        mul!(MatM, MatTemp, view(MatP, (pos+1):(pos+dim), :))
    else
        mul!(MatTemp, view(MatP, :, (pos+1):(pos+dim)), view(MatC, (cos+1):(cos+dim), (cos+1):(cos+dim)))
        mul!(MatM, MatTemp, view(MatP, :, (pos+1):(pos+dim))')
    end
    return M
end

function cong_dense!(M::Ref{Matrix{Float64}}, P::Ref{Matrix{Float64}}, C::Ref{Matrix{Float64}}, wsp_cong::WSP=get_wsp_cong(size(P[], 1), size(C[], 1)); trans=false)

    MatTemp = wsp_cong[1]

    MatM = M[]
    MatP = P[]
    MatC = C[]

    # full_dim = size(MatM, 1);
    # core_dim = ed - os;
    if trans
        mul!(MatTemp, MatP', MatC)
        mul!(MatM, MatTemp, MatP)
    else
        mul!(MatTemp, MatP, MatC)
        mul!(MatM, MatTemp, MatP')
    end
    return M
end

function cong_dense!(M::Ref{Matrix{Float64}}, P::Ref{Matrix{Float64}}, wsp_cong::WSP=get_wsp_cong(size(P[], 1), size(C[], 1)); trans=false)

    MatTemp = wsp_cong[1]

    MatM = M[]
    MatP = P[]

    # full_dim = size(MatM, 1);
    # core_dim = ed - os;
    if trans
        mul!(MatTemp, MatP', MatM)
        mul!(MatM, MatTemp, MatP)
    else
        mul!(MatTemp, MatP, MatM)
        mul!(MatM, MatTemp, MatP')
    end
    return M
end

function cong_dense!(M::Ref{Matrix{Float64}}, P::Ref{Matrix{Float64}}, C::Ref{Matrix{Float64}}, os::Int, ed::Int, wsp_cong::WSP=get_wsp_cong(size(P[], 1), size(C[], 1)); trans=false)

    MatTemp = wsp_cong[1]

    MatM = M[]
    MatP = P[]
    MatC = C[]

    # full_dim = size(MatM, 1);
    # core_dim = ed - os;

    if os == 0 && ed == size(M, 1)
        return cong_dense!(M, P, C, wsp_cong; trans=trans)
    end


    if trans
        viewP = view(MatP, (os+1):ed, :)
        mul!(view(MatTemp, :, (os+1):ed), viewP', view(MatC, (os+1):ed, (os+1):ed))
        mul!(MatM, view(MatTemp, :, (os+1):ed), viewP)
    else
        viewP = view(MatP, :, (os+1):ed)
        mul!(view(MatTemp, :, (os+1):ed), viewP, view(MatC, (os+1):ed, (os+1):ed))
        mul!(MatM, view(MatTemp, :, (os+1):ed), viewP')
    end
    return M
end

function cong_SkewSymm_dense!(M::Ref{Matrix{Float64}}, P::Ref{Matrix{Float64}}, C::Ref{Matrix{Float64}}, wsp_cong::WSP=get_wsp_cong(size(P[], 1), size(C[], 1)); trans=false)

    MatTemp = wsp_cong[1]

    MatM = M[]
    MatP = P[]
    MatC = C[]

    # full_dim = size(MatM, 1);
    # core_dim = ed - os;
    if trans
        mul!(MatTemp, MatP', LowerTriangular(MatC))
        mul!(MatM, MatTemp, MatP)
    else
        mul!(MatTemp, MatP, LowerTriangular(MatC))
        mul!(MatM, MatTemp, MatP')
    end

    # Recover skew-symmetry in the output

    for c_ind in axes(MatM, 1)
        for r_ind in 1:(c_ind-1)
            @inbounds MatM[r_ind, c_ind] -= MatM[c_ind, r_ind]
            @inbounds MatM[c_ind, r_ind] = -MatM[r_ind, c_ind]
        end
        @inbounds MatM[c_ind, c_ind] = 0.0
    end

    return M
end

function cong_Eigen!(M::Ref{Matrix{Float64}}, E::Eigen{Float64,Float64,Matrix{Float64},Vector{Float64}}, wsp_cong::WSP=get_wsp_cong(size(M[])...); trans=false)

    MatM = M[]

    MatTemp = wsp_cong[1]

    if trans
        copy!(MatTemp, E.vectors)
        lmul!(Diagonal(E.values), MatTemp)
        mul!(MatM, E.vectors', MatTemp)
    else
        copy!(MatTemp, E.vectors')
        lmul!(Diagonal(E.values), MatTemp)
        mul!(MatM, E.vectors, MatTemp)
    end

    return M

end

# function cong_SkewSymm_DenseBlk!(M::Ref{Matrix{Float64}}, P::Ref{Matrix{Float64}}, C::Ref{Matrix{Float64}}, wsp_cong::WSP = get_wsp_cong(size(P[], 1), size(C[], 1)); trans = false, r_os::Int = 0, c_os::Int = 0, r_ed::Int = size(C[], 1), c_ed::Int = size(C[], 1))

#     MatTemp = wsp_cong[1];

#     MatM = M[];
#     MatP = P[];
#     MatC = C[];

#     # full_dim = size(MatM, 1);
#     # core_dim = ed - os;


#     if trans
#         viewP = view(MatP, (os + 1):ed, :);
#         mul!(view(MatTemp, :, (os + 1):ed), viewP', view(MatC, (os + 1):ed, (os + 1):ed));
#         mul!(MatM, view(MatTemp, :, (os + 1):ed), viewP);
#     else
#         viewP = view(MatP, :, (os + 1):ed);
#         mul!(view(MatTemp, :, (os + 1):ed), viewP, view(MatC, (os + 1):ed, (os + 1):ed));
#         mul!(MatM, view(MatTemp, :, (os + 1):ed), viewP');
#     end
# end

"""
    cong_SkewSymm_Angle!(M, P, A, [m = length(A[])]; trans = false) -> M::Ref{Matrix{Float64}}

Compute the congruence ``PDP'``, by default with `trans = false`, or ``P'DP`` where D is skew symmetric block diagonal with nonzero diagonal blocks in forms of
```julia-repl
2×2 Matrix{Float64}:
0. -θ
θ  0.
```
where the angles `θ`'s are specified as the first `m` entries in `A::Ref{Vector{Float64}}`. 
The computation is done by `m` rank-1 update. Numerical test indicates that this implementation is `2` to `3` times faster, for `n < 100`, than the dense matrix implementation (possibly with short-circuit strategy). 

Run `test_cong_SkewSymm_Angle_speed(n)` to see the performances.


---------------------------------------------------

    cong_SkewSymm_Angle!(M, P, A, m, α<:Real; trans = false) -> M::Ref{Matrix{Float64}}

Same task as ``cong_SkewSymm_Angle!(M, P, A, [m = length(A[])]; trans = false)`` with `D` scaled by `α`.


cong_SkewSymm_Angle!(M, P, A, m, α<:Real; trans = false) -> M::Ref{Matrix{Float64}}


"""

function cong_SkewSymm_Angle!(M::Ref{Matrix{Float64}}, P::Ref{Matrix{Float64}}, A::Ref{Vector{Float64}}, m::Int=length(A[]); trans::Bool=false)
    MatM = M[]
    MatP = P[]
    VecA = A[]

    fill!(MatM, 0.0)

    if trans
        for a_ind = 1:m
            # Using view to access column. As Julia stores matrix in column major order, it is possible to access them by raw pointers
            # and then call a self-warpped ger from BLAS, just like how dgees is done in myLA. However, when it comes to P^T A P,
            # accessing rows needs extra care. So it is not just a ger warp, but also row access issue. Therefore, we put this on hold
            # and simply use view that has both access. In this case, the Julia built-in ger! can be used.
            @inbounds BLAS.ger!(VecA[a_ind], view(MatP, 2 * a_ind, :), view(MatP, 2 * a_ind - 1, :), MatM)
            # @inbounds BLAS.ger!(-VecA[a_ind], view(MatP, 2 * a_ind - 1, :), view(MatP, 2 * a_ind, :), MatM) # Reduced by skew-symmetry.

        end
    else
        for a_ind = 1:m
            # Using view to access column. As Julia stores matrix in column major order, it is possible to access them by raw pointers
            # and then call a self-warpped ger from BLAS, just like how dgees is done in myLA. However, when it comes to P^T A P,
            # accessing rows needs extra care. So it is not just a ger warp, but also row access issue. Therefore, we put this on hold
            # and simply use view that has both access. In this case, the Julia built-in ger! can be used.

            @inbounds BLAS.ger!(VecA[a_ind], view(MatP, :, 2 * a_ind), view(MatP, :, 2 * a_ind - 1), MatM)
            # @inbounds BLAS.ger!(-VecA[a_ind], view(MatP, :, 2 * a_ind - 1), view(MatP, :, 2 * a_ind), MatM) # Reduced by skew-symmetry

        end
    end

    for c_ind = axes(MatM, 2)
        for r_ind = (c_ind+1):size(MatM, 1)
            @inbounds MatM[r_ind, c_ind] -= MatM[c_ind, r_ind]
            @inbounds MatM[c_ind, r_ind] = -MatM[r_ind, c_ind]
        end
        MatM[c_ind, c_ind] = 0.0
    end
end

function cong_SkewSymm_Angle!(M::Ref{Matrix{Float64}}, P::Ref{Matrix{Float64}}, A::Ref{Vector{Float64}}, m::Int, scale::Float64; trans::Bool=false)
    MatM = M[]
    MatP = P[]
    VecA = A[]

    fill!(MatM, 0.0)

    if trans
        for a_ind = 1:m
            # Using view to access column. As Julia stores matrix in column major order, it is possible to access them by raw pointers
            # and then call a self-warpped ger from BLAS, just like how dgees is done in myLA. However, when it comes to P^T A P,
            # accessing rows needs extra care. So it is not just a ger warp, but also row access issue. Therefore, we put this on hold
            # and simply use view that has both access. In this case, the Julia built-in ger! can be used.
            @inbounds BLAS.ger!(scale * VecA[a_ind], view(MatP, 2 * a_ind, :), view(MatP, 2 * a_ind - 1, :), MatM)
            # @inbounds BLAS.ger!(-VecA[a_ind], view(MatP, 2 * a_ind - 1, :), view(MatP, 2 * a_ind, :), MatM) # Reduced by skew-symmetry.

        end
    else
        for a_ind = 1:m
            # Using view to access column. As Julia stores matrix in column major order, it is possible to access them by raw pointers
            # and then call a self-warpped ger from BLAS, just like how dgees is done in myLA. However, when it comes to P^T A P,
            # accessing rows needs extra care. So it is not just a ger warp, but also row access issue. Therefore, we put this on hold
            # and simply use view that has both access. In this case, the Julia built-in ger! can be used.

            @inbounds BLAS.ger!(scale * VecA[a_ind], view(MatP, :, 2 * a_ind), view(MatP, :, 2 * a_ind - 1), MatM)
            # @inbounds BLAS.ger!(-VecA[a_ind], view(MatP, :, 2 * a_ind - 1), view(MatP, :, 2 * a_ind), MatM) # Reduced by skew-symmetry

        end
    end

    for c_ind = axes(MatM, 2)
        for r_ind = (c_ind+1):size(MatM, 1)
            @inbounds MatM[r_ind, c_ind] -= MatM[c_ind, r_ind]
            @inbounds MatM[c_ind, r_ind] = -MatM[r_ind, c_ind]
        end
        MatM[c_ind, c_ind] = 0.0
    end
end


"""
    cong_SpecOrth_Angle!(M, P, A, [m = length(A[])]; trans = false) -> M::Ref{Matrix{Float64}}

Compute the congruence ``P exp(D) P'``, by default with `trans = false`, or ``P' exp(D) P`` where D is skew symmetric block diagonal with nonzero diagonal blocks in forms of
```julia-repl
2×2 Matrix{Float64}:
0. -θ
θ  0.
```
where the angles `θ`'s are specified as the first `m` entries in `A::Ref{Vector{Float64}}`.
The computation is done by `m + 2n` rank-1 update. Numerical test indicates that this implementation is `1.2` to `1.5` times faster, for `n < 100`, than the dense matrix implementation (possibly with short-circuit strategy). 


Run `test_cong_SpecOrth_Angle_speed(n)` to see the performances.

---------------------------------------------------

    cong_SpecOrth_Angle!(M, P, A, m, α<:Real; trans = false) -> M::Ref{Matrix{Float64}}

Same task as ``cong_SpecOrth_Angle!(M, P, A, [m = length(A[])]; trans = false)`` with `D` scaled by `α`.

"""
function cong_SpecOrth_Angle!(M::Ref{Matrix{Float64}}, P::Ref{Matrix{Float64}}, A::Ref{Vector{Float64}}, m::Int=length(A[]); trans::Bool=false)
    MatM = M[]
    MatP = P[]
    VecA = A[]

    fill!(MatM, 0.0)

    if trans
        for a_ind = 1:m
            # Using view to access column. As Julia stores matrix in column major order, it is possible to access them by raw pointers
            # and then call a self-warpped ger from BLAS, just like how dgees is done in myLA. However, when it comes to P^T A P,
            # accessing rows needs extra care. So it is not just a ger warp, but also row access issue. Therefore, we put this on hold
            # and simply use view that has both access. In this case, the Julia built-in ger! can be used.
            @inbounds BLAS.ger!(sin(VecA[a_ind]), view(MatP, 2 * a_ind, :), view(MatP, 2 * a_ind - 1, :), MatM)
            # @inbounds BLAS.ger!(-VecA[a_ind], view(MatP, 2 * a_ind - 1, :), view(MatP, 2 * a_ind, :), MatM) # Reduced by skew-symmetry.

        end
    else
        for a_ind = 1:m
            # Using view to access column. As Julia stores matrix in column major order, it is possible to access them by raw pointers
            # and then call a self-warpped ger from BLAS, just like how dgees is done in myLA. However, when it comes to P^T A P,
            # accessing rows needs extra care. So it is not just a ger warp, but also row access issue. Therefore, we put this on hold
            # and simply use view that has both access. In this case, the Julia built-in ger! can be used.

            @inbounds BLAS.ger!(sin(VecA[a_ind]), view(MatP, :, 2 * a_ind), view(MatP, :, 2 * a_ind - 1), MatM)
            # @inbounds BLAS.ger!(-VecA[a_ind], view(MatP, :, 2 * a_ind - 1), view(MatP, :, 2 * a_ind), MatM) # Reduced by skew-symmetry

        end
    end

    for c_ind = axes(MatM, 2)
        for r_ind = (c_ind+1):size(MatM, 1)
            @inbounds MatM[r_ind, c_ind] -= MatM[c_ind, r_ind]
            @inbounds MatM[c_ind, r_ind] = -MatM[r_ind, c_ind]
        end
        MatM[c_ind, c_ind] = 0.0
    end

    d_ind::Int = 1
    for a_ind in 1:m
        @inbounds BLAS.ger!(cos(VecA[a_ind]), view(MatP, :, 2 * a_ind - 1), view(MatP, :, 2 * a_ind - 1), MatM)
        @inbounds BLAS.ger!(cos(VecA[a_ind]), view(MatP, :, 2 * a_ind), view(MatP, :, 2 * a_ind), MatM)
    end
    for d_ind in (2m+1):size(MatM, 1)
        @inbounds BLAS.ger!(1.0, view(MatP, :, d_ind), view(MatP, :, d_ind), MatM)
    end
end

function cong_SpecOrth_Angle!(M::Ref{Matrix{Float64}}, P::Ref{Matrix{Float64}}, A::Ref{Vector{Float64}}, m::Int, scale::Float64; trans::Bool=false)
    MatM = M[]
    MatP = P[]
    VecA = A[]

    fill!(MatM, 0.0)

    if trans
        for a_ind = 1:m
            # Using view to access column. As Julia stores matrix in column major order, it is possible to access them by raw pointers
            # and then call a self-warpped ger from BLAS, just like how dgees is done in myLA. However, when it comes to P^T A P,
            # accessing rows needs extra care. So it is not just a ger warp, but also row access issue. Therefore, we put this on hold
            # and simply use view that has both access. In this case, the Julia built-in ger! can be used.
            @inbounds BLAS.ger!(sin(scale * VecA[a_ind]), view(MatP, 2 * a_ind, :), view(MatP, 2 * a_ind - 1, :), MatM)
            # @inbounds BLAS.ger!(-VecA[a_ind], view(MatP, 2 * a_ind - 1, :), view(MatP, 2 * a_ind, :), MatM) # Reduced by skew-symmetry.

        end
    else
        for a_ind = 1:m
            # Using view to access column. As Julia stores matrix in column major order, it is possible to access them by raw pointers
            # and then call a self-warpped ger from BLAS, just like how dgees is done in myLA. However, when it comes to P^T A P,
            # accessing rows needs extra care. So it is not just a ger warp, but also row access issue. Therefore, we put this on hold
            # and simply use view that has both access. In this case, the Julia built-in ger! can be used.

            @inbounds BLAS.ger!(sin(scale * VecA[a_ind]), view(MatP, :, 2 * a_ind), view(MatP, :, 2 * a_ind - 1), MatM)
            # @inbounds BLAS.ger!(-VecA[a_ind], view(MatP, :, 2 * a_ind - 1), view(MatP, :, 2 * a_ind), MatM) # Reduced by skew-symmetry

        end
    end

    for c_ind = axes(MatM, 2)
        for r_ind = (c_ind+1):size(MatM, 1)
            @inbounds MatM[r_ind, c_ind] -= MatM[c_ind, r_ind]
            @inbounds MatM[c_ind, r_ind] = -MatM[r_ind, c_ind]
        end
        MatM[c_ind, c_ind] = 0.0
    end

    d_ind::Int = 1
    for a_ind in 1:m
        @inbounds BLAS.ger!(cos(scale * VecA[a_ind]), view(MatP, :, 2 * a_ind - 1), view(MatP, :, 2 * a_ind - 1), MatM)
        @inbounds BLAS.ger!(cos(scale * VecA[a_ind]), view(MatP, :, 2 * a_ind), view(MatP, :, 2 * a_ind), MatM)
    end
    for d_ind in (2m+1):size(MatM, 1)
        @inbounds BLAS.ger!(1.0, view(MatP, :, d_ind), view(MatP, :, d_ind), MatM)
    end
end

function congruenceSkewSymm!(M::Ref{Matrix{Float64}}, P::Ref{Matrix{Float64}}, S::Ref{Matrix{Float64}}; transpose::Bool=false, scale::Float64=1.0)
    MatM = M[]
    MatP = P[]
    MatS = S[]

    fill!(MatM, 0.0)

    if transpose
        for Si in axes(MatS, 1)
            for Sj = 1:(Si-1)
                for Xi in axes(MatP, 1)
                    for Xj = 1:(Xi-1)
                        @inbounds temp = scale * MatS[Si, Sj] * (MatP[Xi, Si] * MatP[Xj, Sj] - MatP[Xj, Si] * MatP[Xi, Sj])
                        @inbounds MatM[Xi, Xj] += temp
                        @inbounds MatM[Xj, Xi] -= temp
                    end
                end
            end
        end
    else
        for Si in axes(MatS, 1)
            for Sj = 1:(Si-1)
                for Xi in axes(MatP, 1)
                    for Xj = 1:(Xi-1)
                        @inbounds temp = scale * MatS[Si, Sj] * (MatP[Si, Xi] * MatP[Sj, Xj] - MatP[Si, Xj] * MatP[Sj, Xi])
                        @inbounds MatM[Xi, Xj] += temp
                        @inbounds MatM[Xj, Xi] -= temp
                    end
                end
            end
        end
    end
end


#######################################Test functions#######################################

using BenchmarkTools

"""
    test_cong_SkewSymm_Angle(n = 10)
This test function verifies the congruence implementation `cong_SkewSymm_Angle` that evaluates ``P D P'``, where D is a `n × n` skew symmetric block diagonal with `m = div(n, 2)` nonzero diagonal blocks in forms of
```julia-repl
2×2 Matrix{Float64}:
0. -θ
θ  0.
```
"""
function test_cong_SkewSymm_Angle(n=10)
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

"""
    test_cong_SpecOrth(n = 10)
This test function verifies the congruence implementation `cong_SpecOrth_Angle` that evaluates ``P exp(D) P'``, where D is a `n × n` skew symmetric block diagonal with `m = div(n, 2)` nonzero diagonal blocks in forms of
```julia-repl
2×2 Matrix{Float64}:
0. -θ
θ  0.
```
"""
function test_cong_SpecOrth_Angle(n=10)
    A = rand(div(n, 2))
    P = rand(n, n)
    S = rand(n, n)
    S .-= S'

    DCS = zeros(n, n)
    for a_ind in eachindex(A)
        ang = A[a_ind]

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


function test_cong_SkewSymm_dense_speed(n=10)
    MatP = rand(n, n)
    MatS = rand(n, n)
    MatS .-= MatS'

    MatcS1 = similar(MatS)
    MatcS2 = similar(MatS)

    P = Ref(MatP)
    S = Ref(MatS)

    cS1 = Ref(MatcS1)
    cS2 = Ref(MatcS2)



    println("Dense congurence computation time:")
    @btime cong_dense!($cS1, $P, $S)

    println("Dense SkewSymm congurence computation time:")
    @btime cong_SkewSymm_dense!($cS2, $P, $S)

    println("Same result?\t", MatcS1 ≈ MatcS2)

end

"""
    test_cong_SkewSymm_Angle_speed(n = 10)
Compare the speed of rank-1 update implementation of `cong_SkewSymm_Angle!` with the dense matrix implementation.
```
"""
function test_cong_SkewSymm_Angle_speed(n=10)
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


    println("cong_SkewSymm_Angle! computation time:")
    @time cong_SkewSymm_Angle!(cS, P, A; trans=false)

    println("dense matrix multiplication time:")
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

"""
    test_cong_SpecOrth_Angle_speed(n = 10)
Compare the speed of rank-1 update implementation of `cong_SpecOrth_Angle!` with the dense matrix implementation.
```
"""
function test_cong_SpecOrth_Angle_speed(n=10)
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


    println("cong_SpecOrth_Angle! computation time:")
    @time cong_SpecOrth_Angle!(cS, P, A; trans=false)

    println("dense matrix multiplication time:")
    @time begin
        fill!(MatA, 0.0)
        for ind in eachindex(VecA)
            @inbounds MatA[2*ind-1, 2*ind] = -sin(VecA[ind])
            @inbounds MatA[2*ind, 2*ind-1] = sin(VecA[ind])
            @inbounds MatA[2*ind-1, 2*ind-1] = cos(VecA[ind])
            @inbounds MatA[2*ind, 2*ind] = cos(VecA[ind])
        end
        if isodd(n)
            @inbounds MatA[n, n] = 1.0
        end
        mul!(MatM, MatP, MatA)
        mul!(cS2, MatM, MatP')
    end

    println("Same result?\t", cS1 ≈ cS2)
end