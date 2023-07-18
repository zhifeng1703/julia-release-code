include("../../inc/global_path.jl")

include(joinpath(JULIA_INCLUDE_PATH, "workspace.jl"))
include(joinpath(JULIA_MANIFOLD_PATH, "spec_orth/dexp_SkewSymm.jl"))

"""
    _grhor_mat(DV, DVp, x) -> DV, DVp

Retrieve the skew-symmetric matrices DV, DVp that determines the tangent vector |[V exp(X)] * DV | [Vp exp(Z)] * DVp| from the real vector x that records their lower half entries.

"""
function _grhor_mat!(DV, DVp, x::Ref{Vector{Float64}})
    Vecx = x[]
    MatDV = DV[]
    MatDVp = DVp[]

    k::Int = size(MatDV, 1)
    m::Int = size(MatDVp, 1)

    x_ind::Int = 1

    for c_ind = 1:k
        MatDV[c_ind, c_ind] = 0.0
        for r_ind = (c_ind+1):k
            @inbounds MatDV[r_ind, c_ind] = Vecx[x_ind]
            @inbounds MatDV[c_ind, r_ind] = -Vecx[x_ind]
            x_ind += 1
        end
    end

    for c_ind = 1:m
        MatDVp[c_ind, c_ind] = 0.0
        for r_ind = (c_ind+1):m
            @inbounds MatDVp[r_ind, c_ind] = Vecx[x_ind]
            @inbounds MatDVp[c_ind, r_ind] = -Vecx[x_ind]
            x_ind += 1
        end
    end

end

function _grhor_mat!(DV, DVp, x::AbstractVector{Float64})
    Vecx = x
    MatDV = DV[]
    MatDVp = DVp[]

    k::Int = size(MatDV, 1)
    m::Int = size(MatDVp, 1)

    x_ind::Int = 1

    for c_ind = 1:k
        MatDV[c_ind, c_ind] = 0.0
        for r_ind = (c_ind+1):k
            @inbounds MatDV[r_ind, c_ind] = Vecx[x_ind]
            @inbounds MatDV[c_ind, r_ind] = -Vecx[x_ind]
            x_ind += 1
        end
    end

    for c_ind = 1:m
        MatDVp[c_ind, c_ind] = 0.0
        for r_ind = (c_ind+1):m
            @inbounds MatDVp[r_ind, c_ind] = Vecx[x_ind]
            @inbounds MatDVp[c_ind, r_ind] = -Vecx[x_ind]
            x_ind += 1
        end
    end

end




"""
    _grhor_vec(x, DV, DVp) -> x

Convert the lower half of the skew-symmetric matrices DV, DVp from the tangent vector |[V exp(X)] * DV | [Vp exp(Z)] * DVp| to a real vector x.

"""
function _grhor_vec!(x, DV, DVp)
    MatDV = DV[]
    MatDVp = DVp[]
    Vecx = x[]


    k::Int = size(MatDV, 1)
    m::Int = size(MatDVp, 1)

    x_ind::Int = 1

    for c_ind = 1:k
        for r_ind = (c_ind+1):k
            @inbounds Vecx[x_ind] = MatDV[r_ind, c_ind]
            x_ind += 1
        end
    end

    for c_ind = 1:m
        for r_ind = (c_ind+1):m
            @inbounds Vecx[x_ind] = MatDVp[r_ind, c_ind]
            x_ind += 1
        end
    end
end

get_wsp_grhor_sys(n, k) = WSP(get_wsp_cong(n), get_wsp_cong(k), get_wsp_cong(n - k));

"""
    _grhor_system!(M, ΔS, ΔX, ΔV, ΔVp, S_sys, S_saf, X_sys, X_saf, blk_it_n, blk_it_k, wsp_grhor_sys) -> M, ΔS, ΔX

Compute ΔS, ΔX so that dexp_S[ΔS] = | V dexp_X[ΔX] | Vp exp(Z) ΔVp | = | V exp(X) ΔV | Vp exp(Z) ΔVp | and M = ΔA - ΔX.

"""
function _grhor_system!(M, ΔS, ΔX, ΔV, ΔVp, S_sys, S_saf, X_sys, X_saf, blk_it_n, blk_it_k, wsp_grhor_sys)
    MatM = M[]
    MatΔS = ΔS[]
    MatΔX = ΔX[]
    MatΔV = ΔV[]
    MatΔVp = ΔVp[]

    wsp_cong_n = wsp_grhor_sys[1]
    wsp_cong_k = wsp_grhor_sys[2]



    k::Int = size(MatΔV, 1)
    m::Int = size(MatΔVp, 1)


    @inbounds MatΔS .= 0.0
    @inbounds MatΔX .= 0.0

    for c_ind in axes(MatΔV, 2)
        for r_ind in axes(MatΔV, 1)
            @inbounds MatΔS[r_ind, c_ind] = MatΔV[r_ind, c_ind]
        end
    end

    for c_ind in axes(MatΔVp, 2)
        for r_ind in axes(MatΔVp, 1)
            @inbounds MatΔS[r_ind+k, c_ind+k] = MatΔVp[r_ind, c_ind]
        end
    end

    dexp_SkewSymm!(ΔS, ΔS, S_sys, S_saf, blk_it_n, wsp_cong_n; inv=true, cong=true)
    dexp_SkewSymm!(ΔX, ΔV, X_sys, X_saf, blk_it_k, wsp_cong_k; inv=true, cong=true)

    for c_ind in axes(MatΔV, 2)
        for r_ind in axes(MatΔV, 1)
            @inbounds MatM[r_ind, c_ind] = MatΔS[r_ind, c_ind] - MatΔX[r_ind, c_ind]
        end
    end
end

function _grhor_system_full!(M, ΔS, ΔX, ΔZ, ΔV, ΔVp, S_sys, S_saf, X_sys, X_saf, Z_sys, Z_saf, blk_it_n, blk_it_k, blk_it_m, wsp_grhor_sys)
    MatM = M[]
    MatΔS = ΔS[]
    MatΔX = ΔX[]
    MatΔZ = ΔZ[]
    MatΔV = ΔV[]
    MatΔVp = ΔVp[]

    wsp_cong_n = wsp_grhor_sys[1]
    wsp_cong_k = wsp_grhor_sys[2]
    wsp_cong_m = wsp_grhor_sys[3]




    k::Int = size(MatΔV, 1)
    m::Int = size(MatΔVp, 1)

    @inbounds fill!(MatM, 0.0)

    @inbounds fill!(MatΔS, 0.0)
    @inbounds fill!(MatΔX, 0.0)
    @inbounds fill!(MatΔZ, 0.0)


    for c_ind in 1:k
        for r_ind in 1:k
            @inbounds MatΔS[r_ind, c_ind] = MatΔV[r_ind, c_ind]
        end
    end

    for c_ind in 1:m
        for r_ind in 1:m
            @inbounds MatΔS[r_ind+k, c_ind+k] = MatΔVp[r_ind, c_ind]
        end
    end

    dexp_SkewSymm!(ΔS, ΔS, S_sys, S_saf, blk_it_n, wsp_cong_n; inv=true, cong=true)
    dexp_SkewSymm!(ΔX, ΔV, X_sys, X_saf, blk_it_k, wsp_cong_k; inv=true, cong=true)
    dexp_SkewSymm!(ΔZ, ΔVp, Z_sys, Z_saf, blk_it_m, wsp_cong_m; inv=true, cong=true)


    for c_ind in 1:k
        for r_ind in 1:k
            @inbounds MatM[r_ind, c_ind] = MatΔS[r_ind, c_ind] - MatΔX[r_ind, c_ind]
        end
    end

    for c_ind in 1:m
        for r_ind in 1:m
            @inbounds MatM[r_ind+k, c_ind+k] = MatΔS[r_ind+k, c_ind+k] - MatΔZ[r_ind, c_ind]
        end
    end
end


"""
    _grhor_system_action!(y, x, M, ΔS, ΔX, ΔV, ΔVp, S_sys, S_saf, X_sys, X_saf, blk_it_n, blk_it_k, wsp_grhor_sys) -> y, M, ΔS, ΔX

The vectorized version of `_grhor_system!` that has `ΔV, ΔVp` specified in `x` and copy `M` in `y`.

"""
function _grhor_system_action!(y, x, M, ΔS, ΔX, ΔV, ΔVp, S_sys, S_saf, X_sys, X_saf, blk_it_n, blk_it_k, wsp_grhor_sys)
    vecy = y[]
    vecx = x[]
    MatM = M[]
    MatΔS = ΔS[]
    MatΔX = ΔX[]
    MatΔV = ΔV[]
    MatΔVp = ΔVp[]

    _grhor_mat(ΔV, ΔVp, x)
    _grhor_system(M, ΔS, ΔX, ΔV, ΔVp, S_sys, S_saf, X_sys, X_saf, blk_it_n, blk_it_k, wsp_grhor_sys)
    _SkewSymm_mat2vec_by_iterator!(y, M, blk_it_k; lower=true)
end


function _grhor_system_action(y, x, M, ΔS, ΔX, ΔV, ΔVp, S_sys, S_saf, X_sys, X_saf, blk_it_n, blk_it_k, col_it_k, wsp_grhor_sys)
    _grhor_mat!(ΔV, ΔVp, x)
    _grhor_system!(M, ΔS, ΔX, ΔV, ΔVp, S_sys, S_saf, X_sys, X_saf, blk_it_n, blk_it_k, wsp_grhor_sys)
    # vec_SkewSymm_blk!(y, M[], col_it_k; lower = true); # Something wrong in this line, not going in column major order.
    MatM = M[]
    y_ind::Int = 1
    for c_ind in axes(MatM, 2)
        for r_ind in (c_ind+1):size(MatM, 1)
            @inbounds y[y_ind] = MatM[r_ind, c_ind]
            y_ind += 1
        end
    end
end

function _grhor_system_full_action(y, x, ΔS, ΔX, ΔZ, ΔV, ΔVp, S_sys, S_saf, X_sys, X_saf, Z_sys, Z_saf, blk_it_n, blk_it_k, blk_it_m, wsp_grhor_sys)
    n::Int = size(ΔS[], 1)
    k::Int = size(ΔX[], 1)
    m::Int = n - k

    MatM = Matrix{Float64}(undef, n, n)
    M = Ref(MatM)

    _grhor_mat!(ΔV, ΔVp, x)
    _grhor_system_full!(M, ΔS, ΔX, ΔZ, ΔV, ΔVp, S_sys, S_saf, X_sys, X_saf, Z_sys, Z_saf, blk_it_n, blk_it_k, blk_it_m, wsp_grhor_sys)
    # vec_SkewSymm_blk!(y, M[], col_it_k; lower = true); # Something wrong in this line, not going in column major order.

    y_ind::Int = 1
    for c_ind in 1:k
        for r_ind in (c_ind+1):k
            @inbounds y[y_ind] = MatM[r_ind, c_ind]
            y_ind += 1
        end
    end

    for c_ind in 1:m
        for r_ind in (c_ind+1):m
            @inbounds y[y_ind] = MatM[r_ind+k, c_ind+k]
            y_ind += 1
        end
    end
end



#######################################Test functions#######################################

function test_grhor_sys(n, k)
    m::Int = n - k

    S = rand(n, n)
    S .-= S'
    eS = exp(S)

    X = rand(k, k)
    X .-= X'
    eX = exp(X)

    A = copy(S[1:k, 1:k])
    B = copy(S[(k+1):n, 1:k])
    C = copy(S[(k+1):n, (k+1):n])

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
    M = zeros(k, k)

    ΔV = rand(k, k)
    ΔV .-= ΔV'
    ΔVp = rand(m, m)
    ΔVp .-= ΔVp'

    blk_it_n = STRICT_LOWER_ITERATOR(n, lower_blk_traversal)
    blk_it_k = STRICT_LOWER_ITERATOR(k, lower_blk_traversal)

    wsp = get_wsp_grhor_sys(n, k)

    _grhor_system!(Ref(M), Ref(ΔS), Ref(ΔX), Ref(ΔV), Ref(ΔVp), S_sys, S_saf, X_sys, X_saf, blk_it_n, blk_it_k, wsp)

    t = 1e-7

    Qt1 = exp(S .+ t .* ΔS)
    Qt2 = hcat(V * eX * exp(t .* ΔV), Vp * eC * exp(t .* ΔVp))
    Qt3 = hcat(V * exp(X .+ t .* ΔX), Vp * eC * exp(t .* ΔVp))

    println("Correct ΔS?\t", Qt1 ≈ Qt2)
    println("Correct ΔX?\t", Qt1 ≈ Qt3)
end