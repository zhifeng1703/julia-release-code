include("../../inc/global_path.jl")

include(joinpath(JULIA_INCLUDE_PATH, "workspace.jl"))
include(joinpath(JULIA_INCLUDE_PATH, "algorithm_port.jl"))
include(joinpath(JULIA_LAPACK_PATH, "gmres.jl"))
include("./grhor_system.jl")

using IterativeSolvers

@inline grhor_dim(n, k) = div(k * (k - 1), 2) + div((n - k) * (n - k - 1), 2)
# @inline get_wsp_grhor_gmres(n, k, d, r) = WSP(Vector{Float64}(undef, d), Vector{Float64}(undef, d), get_wsp_action(n, k), get_wsp_gmres(d, r))

# grhor_sys_wapper(y, x; kwargs = nothing) = _grhor_system_action!(y_r, x_r, kwargs...)


# function gmres_newton_descent!(ΔV::Ref{Matrix{Float64}}, ΔVp::Ref{Matrix{Float64}},
#     S::Ref{Matrix{Float64}}, S_sys::dexp_SkewSymm_system, S_saf::SAFactor, 
#     X::Ref{Matrix{Float64}}, X_sys::dexp_SkewSymm_system, X_saf::SAFactor,  
#     blk_it_n::STRICT_LOWER_ITERATOR, blk_it_k::STRICT_LOWER_ITERATOR,
#     k::Int, restart::Int, 
#     wsp_grhor_gmres::WSP = get_wsp_grhor_gmres(n, k, div(k * (k - 1), 2), restart); Stop=terminator(50, 1000, 1e-8, 1e-6))

#     MatS = S[];
#     MatX = X[];

#     Vecv = wsp_grhor_gmres[1]
#     Vecb = wsp_grhor_gmres[2]
#     wsp_grhor_sys = wsp_grhor_gmres[3];
#     wsp_gmres = wsp_grhor_gmres[4]

#     v = wsp_grhor_gmres(1)
#     b = wsp_grhor_gmres(2)

#     n::Int = size(MatM, 1);
#     m::Int = n - k;
#     d::Int = div(k * (k - 1), 2)

#     b_ind::Int = 1;
#     for c_ind in axes(MatX, 2)
#         for r_ind in (c_ind + 1):k
#             @inbounds Vecb[b_ind] = MatX[r_ind, c_ind] - MatS[r_ind, c_ind];
#             b_ind += 1;
#         end
#     end 

#     fill!(VecV, 0.0);

#     flag = gmres_matfree!(grhor_sys_wapper, v, b, wsp_gmres, div(k * (k - 1), 2), restart, Stop; 
#         is_x_zeros=true, action_kwargs=[M, ΔS, ΔX, ΔV, ΔVp, S_sys, S_saf, X_sys, X_saf, blk_it_n, blk_it_k, wsp_grhor_sys]);

#     _grhor_mat!(ΔV, ΔVp, v);

#     return flag
# end



struct _GRHOR_ACTION
    M::Ref{Matrix{Float64}}
    ΔS::Ref{Matrix{Float64}}
    ΔX::Ref{Matrix{Float64}}
    ΔV::Ref{Matrix{Float64}}
    ΔVp::Ref{Matrix{Float64}}
    S_sys::dexp_SkewSymm_system
    S_saf::SAFactor
    X_sys::dexp_SkewSymm_system
    X_saf::SAFactor
    blk_it_n::STRICT_LOWER_ITERATOR
    blk_it_k::STRICT_LOWER_ITERATOR
    col_it_k::STRICT_LOWER_ITERATOR
    wsp_action::WSP
    dim1::Int
    dim2::Int
end

Base.eltype(A::_GRHOR_ACTION) = eltype(A.M[]);

Base.size(A::_GRHOR_ACTION, d::Core.Integer) = (d == 1) ? A.dim1 : A.dim2;

function LinearAlgebra.mul!(y::AbstractVector{T}, A::_GRHOR_ACTION, x::AbstractVector{T}) where {T<:Real}
    _grhor_system_action(y, x, A.M, A.ΔS, A.ΔX, A.ΔV, A.ΔVp, A.S_sys, A.S_saf, A.X_sys, A.X_saf, A.blk_it_n, A.blk_it_k, A.col_it_k, A.wsp_action)
end

function Base.:(*)(A::_GRHOR_ACTION, x::AbstractVector{T}) where {T<:Real}
    y = zeros(eltype(A.M[]), A.dim2)
    mul!(y, A, x)
    return y
end


struct _GRHOR_FULL_ACTION
    ΔS::Ref{Matrix{Float64}}
    ΔX::Ref{Matrix{Float64}}
    ΔZ::Ref{Matrix{Float64}}
    ΔV::Ref{Matrix{Float64}}
    ΔVp::Ref{Matrix{Float64}}
    S_sys::dexp_SkewSymm_system
    S_saf::SAFactor
    X_sys::dexp_SkewSymm_system
    X_saf::SAFactor
    Z_sys::dexp_SkewSymm_system
    Z_saf::SAFactor
    blk_it_n::STRICT_LOWER_ITERATOR
    blk_it_k::STRICT_LOWER_ITERATOR
    blk_it_m::STRICT_LOWER_ITERATOR
    wsp_action::WSP
    dim::Int
end

Base.eltype(A::_GRHOR_FULL_ACTION) = eltype(A.ΔS[]);

Base.size(A::_GRHOR_FULL_ACTION, d::Core.Integer) = A.dim

function LinearAlgebra.mul!(y::AbstractVector{T}, A::_GRHOR_FULL_ACTION, x::AbstractVector{T}) where {T<:Real}
    _grhor_system_full_action(y, x, A.ΔS, A.ΔX, A.ΔZ, A.ΔV, A.ΔVp, A.S_sys, A.S_saf, A.X_sys, A.X_saf, A.Z_sys, A.Z_saf, A.blk_it_n, A.blk_it_k, A.blk_it_m, A.wsp_action)
end

function Base.:(*)(A::_GRHOR_FULL_ACTION, x::AbstractVector{T}) where {T<:Real}
    y = zeros(eltype(A.M[]), A.dim)
    mul!(y, A, x)
    return y
end



function grhor_gmres_newton_descent_itsol(ΔV::Ref{Matrix{Float64}}, ΔVp::Ref{Matrix{Float64}}, S::Ref{Matrix{Float64}}, X::Ref{Matrix{Float64}},
    A::_GRHOR_ACTION, restart::Int;
    Stop=terminator(50, 1000, 1e-8, 1e-6))


    # VecV = wsp_bgs[1];
    # VecB = wsp_bgs[2];
    # wsp_action = wsp_bgs[3];
    # wsp_gmres = wsp_bgs[4]

    # v = wsp_bgs(1)
    # b = wsp_bgs(2)

    # blk_it_nm = A.blk_it_nm
    # blk_it_n = A.blk_it_n


    # n::Int = blk_it_nm.leading_dim
    # k::Int = blk_it_nm.offset
    # m::Int = n - k;

    # vec_SkewSymm_blk!(b, M, blk_it_nm; lower = true);
    # fill!(VecV, 0.0);

    # flag = gmres_matfree!(backward_F_wapper, v, b, wsp_gmres, div(m * (m - 1), 2), restart, Stop; 
    # is_x_zeros=true, action_kwargs=[M_sys, M_saf, blk_it_nm, blk_it_n, wsp_action]);


    MatS = S[]
    MatX = X[]

    k = size(MatX, 1)

    Vecv = zeros(A.dim1)
    Vecb = zeros(A.dim2)

    b_ind::Int = 1
    for c_ind in axes(MatX, 2)
        for r_ind in (c_ind+1):k
            @inbounds Vecb[b_ind] = MatX[r_ind, c_ind] - MatS[r_ind, c_ind]
            b_ind += 1
        end
    end

    gmres!(Vecv, A, Vecb; initially_zero=true, abstol=Stop.AbsTol, restart=restart)
    _grhor_mat!(ΔV, ΔVp, Ref(Vecv))
end



function grhor_gmres_newton_descent_full_itsol(ΔV::Ref{Matrix{Float64}}, ΔVp::Ref{Matrix{Float64}},
    S::Ref{Matrix{Float64}}, X::Ref{Matrix{Float64}}, Z::Ref{Matrix{Float64}},
    A::_GRHOR_FULL_ACTION, restart::Int;
    Stop=terminator(50, 1000, 1e-10, 1e-6))


    # VecV = wsp_bgs[1];
    # VecB = wsp_bgs[2];
    # wsp_action = wsp_bgs[3];
    # wsp_gmres = wsp_bgs[4]

    # v = wsp_bgs(1)
    # b = wsp_bgs(2)

    # blk_it_nm = A.blk_it_nm
    # blk_it_n = A.blk_it_n


    # n::Int = blk_it_nm.leading_dim
    # k::Int = blk_it_nm.offset
    # m::Int = n - k;

    # vec_SkewSymm_blk!(b, M, blk_it_nm; lower = true);
    # fill!(VecV, 0.0);

    # flag = gmres_matfree!(backward_F_wapper, v, b, wsp_gmres, div(m * (m - 1), 2), restart, Stop; 
    # is_x_zeros=true, action_kwargs=[M_sys, M_saf, blk_it_nm, blk_it_n, wsp_action]);


    MatS = S[]
    MatX = X[]
    MatZ = Z[]

    k = size(MatX, 1)

    Vecv = zeros(A.dim)
    Vecb = zeros(A.dim)

    b_ind::Int = 1
    for c_ind in axes(MatX, 2)
        for r_ind in (c_ind+1):k
            @inbounds Vecb[b_ind] = MatX[r_ind, c_ind] - MatS[r_ind, c_ind]
            b_ind += 1
        end
    end

    for c_ind in axes(MatZ, 2)
        for r_ind in (c_ind+1):size(MatZ, 1)
            @inbounds Vecb[b_ind] = MatZ[r_ind, c_ind] - MatS[r_ind+k, c_ind+k]
            b_ind += 1
        end
    end

    gmres!(Vecv, A, Vecb; initially_zero=true, abstol=Stop.AbsTol, restart=restart)
    _grhor_mat!(ΔV, ΔVp, Ref(Vecv))
end

function check_grhor_descent(S, X, ΔUk, ΔUp, step)
    MatS = S[]
    MatX = X[]
    MatΔUk = ΔUk[]
    MatΔUp = ΔUp[]



    S_saf = schurAngular_SkewSymm(S; regular=true)
    S_sys = dexp_SkewSymm_system(n)
    compute_dexp_SkewSymm_both_system!(S_sys, S_saf.angle)

    X_saf = schurAngular_SkewSymm(X; regular=true)
    X_sys = dexp_SkewSymm_system(k)
    compute_dexp_SkewSymm_both_system!(X_sys, X_saf.angle)


    blk_it_n = STRICT_LOWER_ITERATOR(n, lower_blk_traversal)
    blk_it_k = STRICT_LOWER_ITERATOR(k, lower_blk_traversal)


    wsp = get_wsp_grhor_sys(n, k)
    wsp_saf_n = get_wsp_saf(n)
    wsp_saf_k = get_wsp_saf(k)

    MatM = similar(MatS)
    MatΔS = similar(MatS)
    MatΔX = similar(MatX)

    M = Ref(MatM)
    ΔS = Ref(MatΔS)
    ΔX = Ref(MatΔX)


    _grhor_system!(M, ΔS, ΔX, ΔUk, ΔUp, S_sys, S_saf, X_sys, X_saf, blk_it_n, blk_it_k, wsp)

    eS = exp(MatS)
    eX = exp(MatX)



    MatQ_new = hcat(eS[:, 1:k] * exp(step .* MatΔUk), eS[:, (k+1):n] * exp(step .* MatΔUp))
    # MatS_new = log(MatQ_new)
    # MatS_new .-= MatS_new'
    # MatS_new .*= 0.5
    MatS_new = similar(MatS)
    Q_new = Ref(MatQ_new)
    S_new = Ref(MatS_new)
    nearlog_SpecOrth!(S_new, S_saf, Q_new, S, wsp_saf_n; order=true, regular=true)

    MatEX_new = eX * exp(step .* MatΔUk)
    # MatX_new = log(MatEX_new)
    # MatX_new .-= MatX_new'
    # MatX_new .*= 0.5

    MatX_new = similar(MatX)
    EX_new = Ref(MatEX_new)
    X_new = Ref(MatX_new)
    nearlog_SpecOrth!(X_new, X_saf, EX_new, X, wsp_saf_k; order=true, regular=true)

    println("exp(S + t ΔS) ≈ | Uk exp(t ΔV) | Up exp(t ΔVp)|?\t", exp(MatS .+ step .* MatΔS) ≈ MatQ_new, "\tDifference:\t", maximum(abs.(exp(MatS .+ step .* MatΔS) .- MatQ_new)))

    println("exp(X + t ΔX) ≈ exp(X) * exp(t ΔV)?\t\t\t", exp(MatX .+ step .* MatΔX) ≈ MatEX_new, "\tDifference:\t", maximum(abs.(exp(MatX .+ step .* MatΔX) .- MatEX_new)))

    println("A + ΔA ≈ X + ΔX?\t", MatS[1:k, 1:k] .+ MatΔS[1:k, 1:k] ≈ MatX .+ MatΔX)


    f_cur = norm(MatS[1:k, 1:k] .- MatX)^2 / 2.0
    f_new = norm(MatS_new[1:k, 1:k] .- MatX_new)^2 / 2.0
    println("Current obj. value: $(f_cur),\t New obj. value: $(f_new), \t Descent slope: $((f_new - f_cur) / step)")
end


function check_grhor_descent(S, X, Z, ΔUk, ΔUp, step)
    MatS = S[]
    MatX = X[]
    MatZ = Z[]
    MatΔUk = ΔUk[]
    MatΔUp = ΔUp[]

    n = size(MatS, 1)
    k = size(MatX, 1)
    m = size(MatZ, 1)



    S_saf = schurAngular_SkewSymm(S; regular=true)
    S_sys = dexp_SkewSymm_system(n)
    compute_dexp_SkewSymm_both_system!(S_sys, S_saf.angle)

    X_saf = schurAngular_SkewSymm(X; regular=true)
    X_sys = dexp_SkewSymm_system(k)
    compute_dexp_SkewSymm_both_system!(X_sys, X_saf.angle)

    Z_saf = schurAngular_SkewSymm(Z; regular=true)
    Z_sys = dexp_SkewSymm_system(m)
    compute_dexp_SkewSymm_both_system!(Z_sys, Z_saf.angle)


    blk_it_n = STRICT_LOWER_ITERATOR(n, lower_blk_traversal)
    blk_it_k = STRICT_LOWER_ITERATOR(k, lower_blk_traversal)
    blk_it_m = STRICT_LOWER_ITERATOR(m, lower_blk_traversal)


    wsp = get_wsp_grhor_sys(n, k)
    wsp_saf_n = get_wsp_saf(n)
    wsp_saf_k = get_wsp_saf(k)
    wsp_saf_m = get_wsp_saf(m)

    MatM = similar(MatS)
    MatΔS = similar(MatS)
    MatΔX = similar(MatX)
    MatΔZ = similar(MatZ)


    M = Ref(MatM)
    ΔS = Ref(MatΔS)
    ΔX = Ref(MatΔX)
    ΔZ = Ref(MatΔZ)



    _grhor_system_full!(M, ΔS, ΔX, ΔZ, ΔUk, ΔUp, S_sys, S_saf, X_sys, X_saf, Z_sys, Z_saf, blk_it_n, blk_it_k, blk_it_m, wsp)

    eS = exp(MatS)
    eX = exp(MatX)
    eZ = exp(MatZ)




    MatQ_new = hcat(eS[:, 1:k] * exp(step .* MatΔUk), eS[:, (k+1):n] * exp(step .* MatΔUp))
    MatS_new = similar(MatS)
    Q_new = Ref(MatQ_new)
    S_new = Ref(MatS_new)
    nearlog_SpecOrth!(S_new, S_saf, Q_new, S, wsp_saf_n; order=true, regular=true)

    MatEX_new = eX * exp(step .* MatΔUk)
    MatX_new = similar(MatX)
    EX_new = Ref(MatEX_new)
    X_new = Ref(MatX_new)
    nearlog_SpecOrth!(X_new, X_saf, EX_new, X, wsp_saf_k; order=true, regular=true)

    MatEZ_new = eZ * exp(step .* MatΔUp)
    MatZ_new = similar(MatZ)
    EZ_new = Ref(MatEZ_new)
    Z_new = Ref(MatZ_new)
    nearlog_SpecOrth!(Z_new, Z_saf, EZ_new, Z, wsp_saf_m; order=true, regular=true)

    println("exp(S + t ΔS) ≈ | Uk exp(t ΔV) | Up exp(t ΔVp)|?\t", exp(MatS .+ step .* MatΔS) ≈ MatQ_new, "\tDifference:\t", maximum(abs.(exp(MatS .+ step .* MatΔS) .- MatQ_new)))

    println("exp(X + t ΔX) ≈ exp(X) * exp(t ΔV)?\t\t\t", exp(MatX .+ step .* MatΔX) ≈ MatEX_new, "\tDifference:\t", maximum(abs.(exp(MatX .+ step .* MatΔX) .- MatEX_new)))

    println("exp(Z + t ΔZ) ≈ exp(Z) * exp(t ΔVp)?\t\t\t", exp(MatZ .+ step .* MatΔZ) ≈ MatEZ_new, "\tDifference:\t", maximum(abs.(exp(MatZ .+ step .* MatΔZ) .- MatEZ_new)))

    println("A + ΔA ≈ X + ΔX?\t", MatS[1:k, 1:k] .+ MatΔS[1:k, 1:k] ≈ MatX .+ MatΔX)
    println("C + ΔC ≈ Z + ΔZ?\t", MatS[(k+1):n, (k+1):n] .+ MatΔS[(k+1):n, (k+1):n] ≈ MatZ .+ MatΔZ)


    f_cur = (norm(MatS[1:k, 1:k] .- MatX)^2 + norm(MatS[(k+1):n, (k+1):n] .- MatZ)^2) / 2.0
    f_new = (norm(MatS_new[1:k, 1:k] .- MatX_new)^2 + norm(MatS_new[(k+1):n, (k+1):n] .- MatZ_new)^2) / 2.0
    println("Current obj. value: $(f_cur),\t New obj. value: $(f_new), \t Descent slope: $((f_new - f_cur) / step)")
end



#######################################Test functions#######################################



function test_grhor_descent(n, k, step=1e-2; seed=1234)

    rand_eng = MersenneTwister(seed)
    m::Int = n - k

    S = rand(rand_eng, n, n)
    S .-= S'
    eS = exp(S)

    # X = rand(rand_eng, k, k);
    X = zeros(k, k)

    X .-= X'
    eX = exp(X)

    A = copy(S[1:k, 1:k])
    B = copy(S[(k+1):n, 1:k])
    C = copy(S[(k+1):n, (k+1):n])

    M = X .- A


    eC = exp(C)

    V = eS[:, 1:k] * eX'
    Vp = eS[:, (k+1):n] * eC'

    # exp(S) = V exp(X) | Vp exp(C)

    S_saf = schurAngular_SkewSymm(Ref(S); regular=true)
    S_sys = dexp_SkewSymm_system(n)
    compute_dexp_SkewSymm_both_system!(S_sys, S_saf.angle)

    X_saf = schurAngular_SkewSymm(Ref(X); regular=true)
    X_sys = dexp_SkewSymm_system(k)
    compute_dexp_SkewSymm_both_system!(X_sys, X_saf.angle)

    ΔS = zeros(n, n)
    ΔX = zeros(k, k)

    ΔV = zeros(k, k)
    ΔVp = zeros(m, m)

    blk_it_n = STRICT_LOWER_ITERATOR(n, lower_blk_traversal)
    blk_it_k = STRICT_LOWER_ITERATOR(k, lower_blk_traversal)
    col_it_k = STRICT_LOWER_ITERATOR(k, lower_col_traversal)


    wsp = get_wsp_grhor_sys(n, k)

    grhor_action = _GRHOR_ACTION(Ref(similar(M)), Ref(similar(ΔS)), Ref(similar(ΔX)), Ref(similar(ΔV)), Ref(similar(ΔVp)),
        S_sys, S_saf, X_sys, X_saf, blk_it_n, blk_it_k, col_it_k, wsp, grhor_dim(n, k), div(k * (k - 1), 2))

    grhor_gmres_newton_descent_itsol(Ref(ΔV), Ref(ΔVp), Ref(S), Ref(X), grhor_action, size(grhor_action, 2))

    # display(ΔV)
    # display(ΔVp)

    # display(S)
    # display(X)
    # display(A .- X)


    _grhor_system!(Ref(M), Ref(ΔS), Ref(ΔX), Ref(ΔV), Ref(ΔVp), S_sys, S_saf, X_sys, X_saf, blk_it_n, blk_it_k, wsp)

    # display(ΔS[1:k, 1:k] .- ΔX)



    Q_new = hcat(eS[:, 1:k] * exp(step .* ΔV), eS[:, (k+1):n] * exp(step .* ΔVp))
    S_new = log(Q_new)
    S_new .-= S_new'
    S_new .*= 0.5

    EX_new = eX * exp(step .* ΔV)
    X_new = log(EX_new)
    X_new .-= X_new'
    X_new .*= 0.5

    # display(ΔS)
    # display((S_new .- S) ./ step)
    # display((S_new .- S) ./ step .- ΔS)
    println("exp(S + t ΔS) ≈ | Vk exp(t ΔV) | Vp exp(t ΔVp)|?\t", ΔS ≈ (S_new .- S) ./ step, "\tDifference:\t", maximum(ΔS .- (S_new .- S) ./ step))

    println("exp(X + t ΔX) ≈ exp(X) * exp(t ΔV)?\t", ΔX ≈ (X_new .- X) ./ step, "\tDifference:\t", maximum(ΔX .- (X_new .- X) ./ step))


    println("A + ΔA ≈ X + ΔX?\t", A .+ ΔS[1:k, 1:k] ≈ X .+ ΔX)

    display(ΔV)
    display(ΔVp)

    display(ΔX)
    display(ΔS[1:k, 1:k])



    println("Current difference: $(norm(A .- X)),\t New difference: $(norm(S_new[1:k, 1:k] .- X_new))")

    return ΔV
end



function test_grhor_descent_full(n, k, step=1e-2; seed=1234)

    rand_eng = MersenneTwister(seed)
    m::Int = n - k

    S = rand(rand_eng, n, n)
    S .-= S'
    eS = exp(S)

    X = rand(rand_eng, k, k)
    X .-= X'
    eX = exp(X)

    Z = rand(rand_eng, m, m)
    Z .-= Z'
    eZ = exp(Z)

    A = copy(S[1:k, 1:k])
    B = copy(S[(k+1):n, 1:k])
    C = copy(S[(k+1):n, (k+1):n])

    M = zeros(n, n)
    M[1:k, 1:k] .= X .- A
    M[(k+1):n, (k+1):n] .= Z .- C

    Uk = eS[:, 1:k]
    Up = eS[:, (k+1):n]

    Vk = Uk * eX'
    Vp = Up * eZ'

    # exp(S) = U = | Uk | Up | = | Vk exp(X) | Vp exp(Z) |

    # exp(S) = V exp(X) | Vp exp(C)

    S_saf = schurAngular_SkewSymm(Ref(S); regular=true)
    S_sys = dexp_SkewSymm_system(n)
    compute_dexp_SkewSymm_both_system!(S_sys, S_saf.angle)

    X_saf = schurAngular_SkewSymm(Ref(X); regular=true)
    X_sys = dexp_SkewSymm_system(k)
    compute_dexp_SkewSymm_both_system!(X_sys, X_saf.angle)

    Z_saf = schurAngular_SkewSymm(Ref(Z); regular=true)
    Z_sys = dexp_SkewSymm_system(m)
    compute_dexp_SkewSymm_both_system!(Z_sys, Z_saf.angle)

    ΔS = zeros(n, n)
    ΔX = zeros(k, k)
    ΔZ = zeros(m, m)


    ΔV = zeros(k, k)
    ΔVp = zeros(m, m)

    blk_it_n = STRICT_LOWER_ITERATOR(n, lower_blk_traversal)
    blk_it_k = STRICT_LOWER_ITERATOR(k, lower_blk_traversal)
    blk_it_m = STRICT_LOWER_ITERATOR(m, lower_blk_traversal)

    wsp = get_wsp_grhor_sys(n, k)

    grhor_full_action = _GRHOR_FULL_ACTION(Ref(similar(ΔS)), Ref(similar(ΔX)), Ref(similar(ΔZ)), Ref(similar(ΔV)), Ref(similar(ΔVp)),
        S_sys, S_saf, X_sys, X_saf, Z_sys, Z_saf, blk_it_n, blk_it_k, blk_it_m, wsp, grhor_dim(n, k))

    grhor_gmres_newton_descent_full_itsol(Ref(ΔV), Ref(ΔVp), Ref(S), Ref(X), Ref(Z), grhor_full_action, size(grhor_full_action, 2))

    _grhor_system_full!(Ref(M), Ref(ΔS), Ref(ΔX), Ref(ΔZ), Ref(ΔV), Ref(ΔVp), S_sys, S_saf, X_sys, X_saf, Z_sys, Z_saf, blk_it_n, blk_it_k, blk_it_m, wsp)

    # display(ΔS[1:k, 1:k] .- ΔX)

    Q_new = hcat(Uk * exp(step .* ΔV), Up * exp(step .* ΔVp))
    S_new = log(Q_new)
    S_new .-= S_new'
    S_new .*= 0.5

    EX_new = eX * exp(step .* ΔV)
    X_new = log(EX_new)
    X_new .-= X_new'
    X_new .*= 0.5

    EZ_new = eZ * exp(step .* ΔVp)
    Z_new = log(EZ_new)
    Z_new .-= Z_new'
    Z_new .*= 0.5


    println("\nΔS:")
    display(ΔS)

    println("\nΔUk:")
    display(ΔV)

    println("\nΔUp:")
    display(ΔVp)

    println("\nΔX:")
    display(ΔX)

    println("\nΔZ:")
    display(ΔZ)

    println("exp(S + t ΔS) ≈ | Uk exp(t ΔV) | Up exp(t ΔVp)|?\t", exp(S .+ step .* ΔS) ≈ Q_new, "\tDifference:\t", maximum(abs.(exp(S .+ step .* ΔS) .- Q_new)))

    println("exp(X + t ΔX) ≈ exp(X) * exp(t ΔV)?\t\t\t", exp(X .+ step .* ΔX) ≈ EX_new, "\tDifference:\t", maximum(abs.(exp(X .+ step .* ΔX) .- EX_new)))
    println("exp(Z + t ΔZ) ≈ exp(Z) * exp(t ΔVp)?\t\t\t", exp(Z .+ step .* ΔZ) ≈ EZ_new, "\tDifference:\t", maximum(abs.(exp(Z .+ step .* ΔZ) .- EZ_new)))




    println("A + ΔA ≈ X + ΔX?\t", A .+ ΔS[1:k, 1:k] ≈ X .+ ΔX)
    println("C + ΔC ≈ Z + ΔZ?\t", C .+ ΔS[(k+1):n, (k+1):n] ≈ Z .+ ΔZ)



    println("Current difference: $(norm(A .- X)),\t New difference: $(norm(S_new[1:k, 1:k] .- X_new)), \t Descent slope: $((norm(S_new[1:k, 1:k] .- X_new) - norm(A .- X)) / step)")
    println("Current difference: $(norm(C .- Z)),\t New difference: $(norm(S_new[(k+1):n, (k+1):n] .- Z_new)), \t Descent slope: $((norm(S_new[(k+1):n, (k+1):n] .- Z_new) - norm(C .- Z)) / step)")


    return ΔV, ΔVp
end