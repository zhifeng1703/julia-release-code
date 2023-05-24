include("../../inc/global_path.jl")

include(joinpath(JULIA_INCLUDE_PATH, "workspace.jl"))
include(joinpath(JULIA_INCLUDE_PATH, "algorithm_port.jl"))
include(joinpath(JULIA_LAPACK_PATH, "gmres.jl"))
include(joinpath(JULIA_LAPACK_PATH, "dgesvd.jl"))
include(joinpath(JULIA_MANIFOLD_PATH, "spec_orth/dexp_SkewSymm.jl"))



using LinearMaps, IterativeSolvers


@inline get_wsp_action(n::Int, k::Int, wsp_cong_n::WSP, wsp_cong_nm::WSP) = WSP(Matrix{Float64}(undef, n, n), Matrix{Float64}(undef, n, n), wsp_cong_n, wsp_cong_nm);
@inline get_wsp_action(n::Int, k::Int) = WSP(Matrix{Float64}(undef, n, n), Matrix{Float64}(undef, n, n), get_wsp_cong(n), get_wsp_cong(n, n - k));


"""
    get_wsp_bgs(n::Int, d::Int, rs::Int)

Workspace for solving the Newton direction of the stlog problem of size `n × k` with dimension `d = div((n - k) * (n - k - 1), 2)`. The gmres is set to restart after `rs` iterations
"""
@inline get_wsp_bgs(d::Int, wsp_action::WSP, wsp_gmres::WSP) = WSP(Vector{Float64}(undef, d), Vector{Float64}(undef, d), wsp_action, wsp_gmres)
@inline get_wsp_bgs(n::Int, k::Int, d::Int, rs::Int) = WSP(Vector{Float64}(undef, d), Vector{Float64}(undef, d), get_wsp_action(n, k), get_wsp_gmres(d, rs))


"""
    _stlog_newton_descent_forward_system!()

This function implements the forward action of the linear system derived from the the Newton direction, described by
``dexp_{S_{A,B,C}}[S_{D_A, D_B, C}] = exp(S_{A,B,C})S_{0, 0, Z}``.

For a given perturbation `S_{D_A, D_B, D_C}`, compute dexp_{S_{A,B,C}}[S_{D_A, D_B, 0}] - S_{0, 0, D_C}
"""
function _stlog_newton_descent_forward_system!(y, x, M_sys, M_saf, n, k, wsp_action)
    S = wsp_action(1);
    Δ = wsp_action(2);

    MatS = wsp_action[1];
    MatΔ = wsp_action[2];
    wsp_cong_n = wsp_action[3];

    mat_SkewSymm_blk!(S, x, fil = true)

    # S = S_{D_A, D_B, 0}
    for c_ind = (k + 1):n
        for r_ind = (k + 1):n
            @inbounds MatS[r_ind, c_ind] = 0.;
        end
    end

    # Δ = dexp_{S_{A,B,C}}[S]
    dexp_SkewSymm!(Δ, S, M_sys, M_saf, wsp_cong_n; inv = false, cong = true)

    # S = S_{D_A, D_B, D_C} (lower triangular part)
    # Note that the vectorization x is not column major, which makes it difficult to only extract D_C from x
    # Since this is O(n) operation that does not significantly impact the performance, 
    # the entire (lower triangular) matrix S_{D_A,D_B,D_C} is recovered.
    mat_SkewSymm_blk!(S, x, fil = false);

    # Δ = Δ - S_{0, 0, DC} (lower triangular part)
    for c_ind = (k + 1):n
        for r_ind = (c_ind + 1):n
            @inbounds MatΔ[r_ind, c_ind] -= MatS[r_ind, c_ind];
        end
    end

    vec_SkewSymm_blk!(y, Δ)
end

function _stlog_newton_descent_backward_system!(y, x, M_sys, M_saf, blk_it_nm, wsp_action)
    MatS = wsp_action[1];
    wsp_cong_n = wsp_action[3];
    wsp_cong_nm = wsp_action[4];


    S = wsp_action(1)
    Δ = wsp_action(1)


    # S = S_{0, 0, Z}
    fill!(MatS, 0.0);
    mat_SkewSymm_blk!(S, blk_it_nm, x; fil = true);

    # Δ = dexp_{S_{A, B, C}}^{-1}[S_{0, 0, Z}]
    dexp_SkewSymm!(Δ, S, M_sys, M_saf, wsp_cong_n; inv = true, cong = true)

    # Extract Δ_C from Δ = S_{Δ_A, Δ_B, Δ_C}
    vec_SkewSymm_blk!(y, Δ, blk_it_nm; lower = true);
end

function _stlog_newton_descent_backward_system!(y::Ref{Vector{Float64}}, x::Ref{Vector{Float64}}, M_sys, M_saf, blk_it_nm, blk_it_n, wsp_action)
    MatS = wsp_action[1];
    wsp_cong_n = wsp_action[3];
    wsp_cong_nm = wsp_action[4];


    MatS = wsp_action[1]
    S = wsp_action(1)
    MatΔ = wsp_action[1]
    Δ = wsp_action(1)


    MatR = M_saf.vector[];

    MatTmpn = wsp_cong_n[1];
    Tmpn = wsp_cong_n(1)

    n::Int = blk_it_n.mat_dim;
    m::Int = blk_it_nm.mat_dim;
    k::Int = n - m;

    # S = S_{0, 0, Z}
    # fill!(MatS, 0.0);
    @turbo for d_ind in (k + 1):n
        @inbounds MatS[d_ind, d_ind] = 0.0;
    end
    mat_SkewSymm_blk!(S, blk_it_nm, x; fil = true);

    # Δ = dexp_{S_{A, B, C}}^{-1}[S_{0, 0, Z}]
    # dexp_SkewSymm!(Δ, S, M_sys, M_saf, blk_it_n, wsp_cong; inv = true, cong = true)

    cong_dense!(S, M_saf.vector, k, S, k, m, wsp_cong_nm; trans = true);

    # mul!(MatTmpnm, view(MatR, (k + 1):n, :)', view(MatS, (k + 1):n, (k + 1):n));
    # mul!(MatS, MatTmpnm, view(MatR, (k + 1):n, :))
    # fill_upper_SkewSymm!(Tmpn, blk_it_n)

    dexp_SkewSymm!(Δ, S, M_sys, M_saf, blk_it_n, wsp_cong_n; inv = true, cong = false)
    # fill_upper_SkewSymm!(Δ, blk_it_n)

    cong_dense!(Δ, M_saf.vector, Δ, wsp_cong_n; trans = false)
    # fill_upper_SkewSymm!(Δ, blk_it_n)

    # Extract Δ_C from Δ = S_{Δ_A, Δ_B, Δ_C}
    vec_SkewSymm_blk!(y, Δ, blk_it_nm; lower = true);
end

function _stlog_newton_descent_backward_system!(y::AbstractVector{Float64}, x::AbstractVector{Float64}, M_sys, M_saf, blk_it_nm, blk_it_n, wsp_action)
    MatS = wsp_action[1];
    wsp_cong_n = wsp_action[3];
    wsp_cong_nm = wsp_action[4];


    S = wsp_action(1)
    Δ = wsp_action(1)

    MatS = wsp_action[1]
    MatΔ = wsp_action[1]

    MatR = M_saf.vector[];
    
    n::Int = blk_it_n.mat_dim;
    m::Int = blk_it_nm.mat_dim;
    k::Int = n - m;

    # S = S_{0, 0, Z}
    @turbo for d_ind in (k + 1):n
        @inbounds MatS[d_ind, d_ind] = 0.0;
    end
    mat_SkewSymm_blk!(MatS, blk_it_nm, x; fil = true);

    # Δ = dexp_{S_{A, B, C}}^{-1}[S_{0, 0, Z}]
    # dexp_SkewSymm!(Δ, S, M_sys, M_saf, blk_it_n, wsp_cong; inv = true, cong = true)


    # mul!(MatTmpnm, view(MatR, (k + 1):n, :)', view(MatS, (k + 1):n, (k + 1):n));
    # mul!(MatS, MatTmpnm, view(MatR, (k + 1):n, :))
    # fill_upper_SkewSymm!(Tmpn, blk_it_n)

    cong_dense!(S, M_saf.vector, k, S, k, m, wsp_cong_nm; trans = true);

    dexp_SkewSymm!(Δ, S, M_sys, M_saf, blk_it_n, wsp_cong_n; inv = true, cong = false)
    # fill_upper_SkewSymm!(Δ, blk_it_n)

    cong_dense!(Δ, M_saf.vector, Δ, wsp_cong_n; trans = false)
    # fill_upper_SkewSymm!(Δ, blk_it_n)

    # Extract Δ_C from Δ = S_{Δ_A, Δ_B, Δ_C}
    vec_SkewSymm_blk!(y, MatΔ, blk_it_nm; lower = true);
end

forward_F_wapper(y_r, x_r; kwargs = nothing) = _stlog_newton_descent_forward_system!(y_r, x_r, kwargs...)
backward_F_wapper(y_r, x_r; kwargs = nothing) = _stlog_newton_descent_backward_system!(y_r, x_r, kwargs...)

function stlog_newton_descent_both!(S::Ref{Matrix{Float64}}, Z::Ref{Matrix{Float64}}, 
    M::Ref{Matrix{Float64}}, M_sys::dexp_SkewSymm_system, M_saf::SAFactor, k::Int, restart::Int, 
    blk_it_nm::STRICT_LOWER_ITERATOR, blk_it_m::STRICT_LOWER_ITERATOR,
    wsp_bgs::WSP = get_wsp_bgs(n, k, div((n - k) * (n - k - 1), 2), restart); Stop=terminator(50, 1000, 1e-8, 1e-6))

    MatM = M[];

    VecV = wsp_bgs[1];
    wsp_action = wsp_bgs[3];
    wsp_gmres = wsp_bgs[4]

    v = wsp_bgs(1)
    b = wsp_bgs(2)


    n::Int = size(MatM, 1);
    m::Int = n - k;
    d::Int = div(m * (m - 1), 2)


    # extract C from M = S_{A, B, C} (lower triangular part)
    # b = vec(C) from M = S_{A, B, C}
    vec_SkewSymm_blk!(b, M, blk_it_nm; lower = true);
    fill!(VecV, 0.0);

    flag = gmres_matfree!(backward_F_wapper, v, b, wsp_gmres, div(m * (m - 1), 2), restart, Stop; 
        is_x_zeros=true, action_kwargs=[M_sys, M_saf, blk_it_nm, wsp_action]);

    mat_SkewSymm_blk!(Z, blk_it_m, v; fil = true)

    MatTemp = wsp_action[1];
    wsp_cong_n = wsp_action[3];

    Temp = wsp_action(1)

    fill!(MatTemp, 0.0);
    mat_SkewSymm_blk!(Temp, blk_it_nm, v; fil = true)

    dexp_SkewSymm!(S, Temp, M_sys, M_saf, wsp_cong_n; inv = true, cong = true)

    return flag
end

function stlog_newton_descent_backward!(Z::Ref{Matrix{Float64}}, 
    M::Ref{Matrix{Float64}}, M_sys::dexp_SkewSymm_system, M_saf::SAFactor, k::Int, restart::Int,
    blk_it_nm::STRICT_LOWER_ITERATOR, blk_it_m::STRICT_LOWER_ITERATOR,
    wsp_bgs::WSP = get_wsp_bgs(n, k, div((n - k) * (n - k - 1), 2), restart); Stop=terminator(50, 1000, 1e-8, 1e-6))

    MatM = M[];

    VecV = wsp_bgs[1];
    wsp_action = wsp_bgs[3];
    wsp_gmres = wsp_bgs[4]

    v = wsp_bgs(1)
    b = wsp_bgs(2)


    n::Int = size(MatM, 1);
    m::Int = n - k;
    d::Int = div(m * (m - 1), 2)


    # extract C from M = S_{A, B, C} (lower triangular part)
    # b = vec(C) from M = S_{A, B, C}
    vec_SkewSymm_blk!(b, M, blk_it_nm; lower = true);

    fill!(VecV, 0.0);

    flag = gmres_matfree!(backward_F_wapper, v, b, wsp_gmres, div(m * (m - 1), 2), restart, Stop; 
        is_x_zeros=true, action_kwargs=[M_sys, M_saf, blk_it_nm, wsp_action]);

    mat_SkewSymm_blk!(Z, blk_it_m, v; fil = true)

    return flag
end

function stlog_newton_descent_both!(S::Ref{Matrix{Float64}}, Z::Ref{Matrix{Float64}}, 
    M::Ref{Matrix{Float64}}, M_sys::dexp_SkewSymm_system, M_saf::SAFactor, k::Int, restart::Int, 
    blk_it_nm::STRICT_LOWER_ITERATOR, blk_it_m::STRICT_LOWER_ITERATOR, blk_it_n::STRICT_LOWER_ITERATOR,
    wsp_bgs::WSP; Stop=terminator(50, 1000, 1e-8, 1e-6))

    MatM = M[];

    VecV = wsp_bgs[1];
    wsp_action = wsp_bgs[3];
    wsp_gmres = wsp_bgs[4]

    v = wsp_bgs(1)
    b = wsp_bgs(2)


    n::Int = size(MatM, 1);
    m::Int = n - k;
    d::Int = div(m * (m - 1), 2)


    # extract C from M = S_{A, B, C} (lower triangular part)
    # b = vec(C) from M = S_{A, B, C}
    vec_SkewSymm_blk!(b, M, blk_it_nm; lower = true);

    fill!(VecV, 0.0);

    flag = gmres_matfree!(backward_F_wapper, v, b, wsp_gmres, div(m * (m - 1), 2), restart, Stop; 
        is_x_zeros=true, action_kwargs=[M_sys, M_saf, blk_it_nm, blk_it_n, wsp_action]);

    mat_SkewSymm_blk!(Z, blk_it_m, v; fil = true)

    MatTemp = wsp_action[1];
    wsp_cong_n = wsp_action[3];
    Temp = wsp_action(1)

    fill!(MatTemp, 0.0);
    mat_SkewSymm_blk!(Temp, blk_it_nm, v; fil = true)
    dexp_SkewSymm!(S, Temp, M_sys, M_saf, blk_it_n, wsp_cong_n; inv = true, cong = true)

    return flag
end

function stlog_newton_descent_backward!(Z::Ref{Matrix{Float64}}, 
    M::Ref{Matrix{Float64}}, M_sys::dexp_SkewSymm_system, M_saf::SAFactor, k::Int, restart::Int, 
    blk_it_nm::STRICT_LOWER_ITERATOR, blk_it_m::STRICT_LOWER_ITERATOR, blk_it_n::STRICT_LOWER_ITERATOR,
    wsp_bgs::WSP = get_wsp_bgs(n, k, div((n - k) * (n - k - 1), 2), restart); Stop=terminator(50, 1000, 1e-8, 1e-6))

    MatM = M[];

    # display(M_saf)
    # display(M_sys.mat_system[])
    # display(M_sys.mat_trasys[])



    VecV = wsp_bgs[1];
    wsp_action = wsp_bgs[3];
    wsp_gmres = wsp_bgs[4]

    v = wsp_bgs(1)
    b = wsp_bgs(2)


    n::Int = size(MatM, 1);
    m::Int = n - k;
    d::Int = div(m * (m - 1), 2)


    # extract C from M = S_{A, B, C} (lower triangular part)
    # b = vec(C) from M = S_{A, B, C}
    vec_SkewSymm_blk!(b, M, blk_it_nm; lower = true);

    fill!(VecV, 0.0);

    flag = gmres_matfree!(backward_F_wapper, v, b, wsp_gmres, div(m * (m - 1), 2), restart, Stop; 
        is_x_zeros=true, action_kwargs=[M_sys, M_saf, blk_it_nm, blk_it_n, wsp_action]);

    mat_SkewSymm_blk!(Z, blk_it_m, v; fil = true)

    return flag
end

function stlog_newton_descent_both(M::Ref{Matrix{Float64}}, k::Int, restart::Int, 
    wsp_bgs::WSP = get_wsp_bgs(size(M[], 1), k, div((size(M[], 1) - k) * (size(M[], 1) - k - 1), 2), restart);
    Stop=terminator(50, 1000, 1e-8, 1e-6))
    n = size(M[], 1)
    MatS = Matrix{Float64}(undef, n, n)
    MatZ = Matrix{Float64}(undef, n - k, n - k)
    blk_it_nm = STRICT_LOWER_ITERATOR(n, k, n, lower_blk_traversal)
    blk_it_m = STRICT_LOWER_ITERATOR(n - k, lower_blk_traversal)
    blk_it_n = STRICT_LOWER_ITERATOR(n, lower_blk_traversal)

    M_saf = schurAngular_SkewSymm(M);
    M_sys = dexp_SkewSymm_system(n)
    compute_dexp_SkewSymm_both_system!(M_sys, M_saf.angle)
    stlog_newton_descent_both!(Ref(MatS), Ref(MatZ), M, M_sys, M_saf, k, restart, blk_it_nm, blk_it_m, blk_it_n, wsp_bgs; Stop = Stop)
    return MatS, MatZ
end

struct _STLOG_BACKWARD_Z_ACTION
    M_sys::dexp_SkewSymm_system;
    M_saf::SAFactor;
    blk_it_nm::STRICT_LOWER_ITERATOR;
    blk_it_n::STRICT_LOWER_ITERATOR;
    wsp_action::WSP
end

function LinearAlgebra.mul!(y::Ref{Vector{T}}, A::_STLOG_BACKWARD_Z_ACTION, x::Ref{Vector{T}}) where T <: Real
    _stlog_newton_descent_backward_system!(y[], x[], A.M_sys, A.M_saf, A.blk_it_nm, A.blk_it_n, A.wsp_action)
end

function LinearAlgebra.mul!(y::AbstractVector{T}, A::_STLOG_BACKWARD_Z_ACTION, x::AbstractVector{T}) where T <: Real
    _stlog_newton_descent_backward_system!(y, x, A.M_sys, A.M_saf, A.blk_it_nm, A.blk_it_n, A.wsp_action)
end

function Base.:(*)(A::_STLOG_BACKWARD_Z_ACTION, x::AbstractVector{T}) where T <: Real
    y = similar(x);
    mul!(y, A, x);
    return y;
end

Base.eltype(A::_STLOG_BACKWARD_Z_ACTION) = eltype(A.M_saf.vector[]);

Base.size(A::_STLOG_BACKWARD_Z_ACTION, d::Core.Integer) = length(A.blk_it_nm.vec2lower[]);

function stlog_newton_descent_gmres!(S::Ref{Matrix{Float64}}, Z::Ref{Matrix{Float64}}, M::Ref{Matrix{Float64}}, 
    A::_STLOG_BACKWARD_Z_ACTION, restart::Int, blk_it_m::STRICT_LOWER_ITERATOR, wsp_bgs::WSP; 
    Stop=terminator(50, 1000, 1e-8, 1e-6))

    MatM = M[];

    VecV = wsp_bgs[1];
    VecB = wsp_bgs[2];
    wsp_action = wsp_bgs[3];
    wsp_gmres = wsp_bgs[4]

    v = wsp_bgs(1)
    b = wsp_bgs(2)

    blk_it_nm = A.blk_it_nm
    blk_it_n = A.blk_it_n


    n::Int = blk_it_nm.leading_dim
    k::Int = blk_it_nm.offset
    m::Int = n - k;

    vec_SkewSymm_blk!(b, M, blk_it_nm; lower = true);
    fill!(VecV, 0.0);

    # flag = gmres_matfree!(backward_F_wapper, v, b, wsp_gmres, div(m * (m - 1), 2), restart, Stop; 
    # is_x_zeros=true, action_kwargs=[M_sys, M_saf, blk_it_nm, blk_it_n, wsp_action]);



    gmres!(VecV, A, VecB; initially_zero = true, abstol = Stop.AbsTol, restart = restart)

    mat_SkewSymm_blk!(Z, blk_it_m, v; fil = true)

    MatTemp = wsp_action[1];
    wsp_cong_n = wsp_action[3];
    Temp = wsp_action(1)

    fill!(MatTemp, 0.0);
    mat_SkewSymm_blk!(Temp, blk_it_nm, v; fil = true)
    dexp_SkewSymm!(S, Temp, A.M_sys, A.M_saf, blk_it_n, wsp_cong_n; inv = true, cong = true)
end

"""
    stlog_BCH3_direction_naive(Z, B, C, R) -> Z::Ref{Matrix{Float64}}

Compute the BCH third update Z from the sylvester equation Z(BB'/12 - I/2) + (BB'/12 - I/2)Z = -C. Lyapunov solver is used.

"""
function stlog_BCH3_direction_lyap!(Z::Ref{Matrix{Float64}}, M::Ref{Matrix{Float64}}, B::Ref{Matrix{Float64}}, C::Ref{Matrix{Float64}}, R::Ref{Matrix{Float64}})

    MatZ = Z[];
    MatM = M[];
    MatB = B[];
    MatC = C[];
    MatR = R[];

    n = size(MatM, 1)
    m = size(MatZ, 1)
    k = n - m


    fill!(MatR, 0.0)
    for d_ind = 1:(n-k)
        @inbounds MatR[d_ind, d_ind] = -0.5
    end
    for c_ind = 1:k
        for r_ind = (k+1):n
            @inbounds MatB[r_ind-k, c_ind] = MatM[r_ind, c_ind]
        end
    end

    mul!(MatR, MatB, MatB', 1.0 / 12, 1.0)

    for r_ind = 1:(n-k)
        for c_ind = 1:(n-k)
            @inbounds MatC[r_ind, c_ind] = -MatM[r_ind+k, c_ind+k]
        end
    end

    MatZ .= lyap(MatR, MatC)
    getSkewSymm!(Z)
    return Z;
end

get_wsp_BCH3_direction(n, m) = WSP(get_wsp_cong(m), get_wsp_dgesvd(m, n - m))

"""
    stlog_BCH3_direction_svd(Z, M, B, C, R) -> Z::Ref{Matrix{Float64}}

Compute the BCH third update Z from the sylvester equation ``Z(BB'/12 - I/2) + (BB'/12 - I/2)Z = -C``. SVD of B is expolited to obtained 
the eigensystem of `(BB'/12 - I/2)` so that the equation can be solved more efficiently. dgesvd is used for the SVD. 
"""
function stlog_BCH3_direction_svd!(Z::Ref{Matrix{Float64}}, M::Ref{Matrix{Float64}}, B::Ref{Matrix{Float64}}, BP::Ref{Matrix{Float64}}, BS::Ref{Vector{Float64}}, C::Ref{Matrix{Float64}}, 
    wsp_BCH3 = get_wsp_BCH3_direction(size(M[], 1), size(Z[], 1)))

    MatZ = Z[];
    MatM = M[];
    MatB = B[];
    MatBP = BP[];
    MatC = C[];

    VecBS = BS[];

    wsp_cong = wsp_BCH3[1]
    wsp_dgesvd = wsp_BCH3[2]


    n::Int = size(MatM, 1)
    m::Int = size(MatC, 1)
    k::Int = n - m


    if m > k  
        # B is tall-skinny and should be used for the thin SVD 
        for c_ind in axes(MatB, 2)
            for r_ind in axes(MatB, 1)
                @inbounds MatB[r_ind, c_ind] = MatM[r_ind + k, c_ind];
            end
        end


        # B = USV' -> BB' = U S^2 U', only the left vector is needed. Full vector is needed here for later computation.
        fill!(VecBS, 0.0);
        dgesvd!(B, BS, BP, 'U', 'A', wsp_dgesvd)


        # Get the transformed sylvester equation S ^ 2 / 12 - I / 2
        for ind in eachindex(VecBS)
            VecBS[ind] = (VecBS[ind] ^ 2 - 6.0) / 12.0;
        end
            
        for c_ind in axes(MatC, 2)
            for r_ind in axes(MatC, 1)
                @inbounds MatC[r_ind, c_ind] = MatM[r_ind + k, c_ind + k]
            end
        end


        # C ← U' C U, as U is stored, transpose is needed
        cong_dense!(C, BP, wsp_cong; trans = true)

        # Solve the simplied Sylvester equation
        for c_ind in axes(MatC, 2)
            for r_ind in (c_ind + 1):m
                @inbounds MatZ[r_ind, c_ind] = MatC[r_ind, c_ind] / (VecBS[r_ind] + VecBS[c_ind]);
            end
        end
        fill_upper_SkewSymm!(Z)

        # Convert the solution to the solution of the original problem Z ← U Z U', as U is stored, transpose is not needed.
        cong_dense!(Z, BP, wsp_cong; trans = false)

    else
        # -B' is tall-skinny and should be used for the thin SVD 
        for c_ind in axes(MatB, 2)
            for r_ind in axes(MatB, 1)
                @inbounds MatB[r_ind, c_ind] = MatM[r_ind, c_ind + k];
            end
        end

        # -B' = USV' -> BB' = V S^2 V', only the right vector is needed.
        dgesvd!(B, BS, BP, 'V', 'S', wsp_dgesvd)

        # Get the transformed sylvester equation S ^ 2 / 12 - I / 2
        for ind in eachindex(VecBS)
            VecBS[ind] = (VecBS[ind] ^ 2 - 6.0) / 12.0;
        end
            
        for c_ind in axes(MatC, 2)
            for r_ind in axes(MatC, 1)
                @inbounds MatC[r_ind, c_ind] = MatM[r_ind + k, c_ind + k]
            end
        end

        # C ← V' C V, as V' is stored, no transpose needed
        cong_dense!(C, BP, wsp_cong; trans = false)

        # Solve the simplied Sylvester equation
        for c_ind in axes(MatC, 2)
            for r_ind in (c_ind + 1):m
                @inbounds MatZ[r_ind, c_ind] = MatC[r_ind, c_ind] / (VecBS[r_ind] + VecBS[c_ind]);
            end
        end
        fill_upper_SkewSymm!(Z)

        # Convert the solution to the solution of the original problem Z ← V Z V', as V' is stored, transpose is needed.
        cong_dense!(Z, BP, wsp_cong; trans = true)

    end

    getSkewSymm!(Z)

    return Z
end




#######################################Test functions#######################################

using Plots

function test_stlog_newton_direction(n, k, rs = div((n - k) * (n - k - 1), 2))
    MatM = rand(n, n);
    MatM .-= MatM';

    MatM .*= rand() * π / opnorm(MatM);

    MatQ = exp(MatM);
    MatQetZ = similar(MatQ);

    MatS, MatZ = stlog_newton_descent_both(Ref(MatM), k, rs; Stop=terminator(max(2 * rs, 200), 50000, 1e-12, 1e-9))

    MatΔ = similar(MatS);

    # display((MatM .- MatS)[k + 1:n, k+1 : n])

    println("The matrix 2 norm of S:\t $(opnorm(MatM))")

    println("dexp_{S_{A, B, C}}[S_{X, Y, C}] = exp(S_{A, B, C})S_{0, 0, Z}?\t", MatM[(k + 1):n, (k + 1):n] ≈ MatS[(k + 1):n, (k + 1):n])

    FulZ = zeros(n, n);
    FulZ[(k + 1):n, (k + 1):n] .= MatZ;

    MatQetZ .= MatQ * exp(1e-8 .* FulZ);
    MatΔ .= (log_SpecOrth(Ref(MatQetZ)) .- MatM) ./ 1e-8

    println("Numerical differentiation check: |(C(t) - C) / t - C| at t = 1e-8:\t", norm(MatΔ[(k + 1):n, (k + 1):n] .- MatM[(k + 1):n, (k + 1):n]))
    # display(MatΔ[(k + 1):n, (k + 1):n] .- MatM[(k + 1):n, (k + 1):n])


    t_grid = range(1e-7, 1e-5, 100);
    val = Vector{Float64}(undef, length(t_grid));
    for t_ind in eachindex(t_grid)
        t = t_grid[t_ind]
        MatQetZ .= MatQ * exp(t .* FulZ);
        MatΔ .= (log_SpecOrth(Ref(MatQetZ)) .- MatM) ./ t
        val[t_ind] = norm(MatΔ[(k + 1):n, (k + 1):n] .- MatM[(k + 1):n, (k + 1):n]) ^2;
    end


    # val = [norm(MatM[(k + 1):n, (k + 1):n] - (real.(log(MatQ * exp(t .* FulZ)))[(k + 1):n, (k + 1):n] .- MatM[(k + 1):n, (k + 1):n]) ./ t) for t in t_grid]
    println("Numerical order of |(C(t) - C) / t - C|_F^2 w.r.t. t, \nwhere exp(S_{A(t), B(t), C(t)}) = exp(S_{A, B, C})exp(t ⋅ S_{0, 0, Z})\t", log.(t_grid) \ log.(val))
end

function test_stlog_newton_gmres(n, k,  rs = div((n - k) * (n - k - 1), 2))
    m::Int = n - k
    d::Int = div(m * (m - 1), 2);

    MatM = rand(n, n);
    MatM .-= MatM';
    MatM .*= rand() * π / opnorm(MatM);

    MatΔ1 = similar(MatM)
    MatΔ2 = similar(MatM)
    MatZ1 = zeros(m, m)
    MatZ2 = zeros(m, m)

    M = Ref(MatM)
    Δ1 = Ref(MatΔ1)
    Δ2 = Ref(MatΔ2)
    Z1 = Ref(MatZ1)
    Z2 = Ref(MatZ2)

    M_sys = dexp_SkewSymm_system(n);
    M_saf = SAFactor(n);

    wsp_saf = get_wsp_saf(n);
    wsp_bgs = get_wsp_bgs(n, k, d, rs);
    wsp_action = wsp_bgs[3]

    blk_it_n = STRICT_LOWER_ITERATOR(n, lower_blk_traversal)
    blk_it_nm = STRICT_LOWER_ITERATOR(n, k, n, lower_blk_traversal)
    blk_it_m = STRICT_LOWER_ITERATOR(m, lower_blk_traversal)

    schurAngular_SkewSymm!(M_saf, M, wsp_saf; regular = true, order = true)
    compute_dexp_SkewSymm_both_system!(M_sys, M_saf.angle; trans = true)

    A = _STLOG_BACKWARD_Z_ACTION(M_sys, M_saf, blk_it_nm, blk_it_n, wsp_action)

    stlog_newton_descent_both!(Δ1, Z1, M, M_sys, M_saf, k, rs, blk_it_nm, blk_it_m, wsp_bgs; Stop=terminator(max(rs, d), 5000, 1e-7, 1e-6))

    stlog_newton_descent_gmres!(Δ2, Z2, M, A, rs, blk_it_m, wsp_bgs; Stop=terminator(max(rs, d), 5000, 1e-7, 1e-6))

    println("Same Δ? \t", MatΔ1 ≈ MatΔ2)

    println("Same Z? \t", MatZ1 ≈ MatZ2)


end

function test_stlog_newton_threading(n, k, rs = div((n - k) * (n - k - 1), 2))
    m = n - k;
    d = div(m * (m - 1), 2)
    MatM = rand(n, n);
    MatM .-= MatM';
    MatM .*= (rand()) * π / opnorm(MatM);

    MatΔ1 = similar(MatM)
    MatΔ2 = similar(MatM)

    MatZ1 = zeros(n - k, n - k)
    MatZ2 = zeros(n - k, n - k)
    MatZ3 = zeros(n - k, n - k)

    MatB = zeros(m, k)
    MatC = zeros(m, m)
    MatR = zeros(m, m)


    M = Ref(MatM)
    Δ1 = Ref(MatΔ1)
    Δ2 = Ref(MatΔ2)
    Z1 = Ref(MatZ1)
    Z2 = Ref(MatZ2)
    Z3 = Ref(MatZ3)
    B = Ref(MatB)
    C = Ref(MatC)
    R = Ref(MatR)

    M_sys = dexp_SkewSymm_system(n);
    M_saf = SAFactor(n);

    wsp_saf = get_wsp_saf(n);
    wsp_bgs = get_wsp_bgs(n, k, d, rs)

    blk_it_n = STRICT_LOWER_ITERATOR(n, lower_blk_traversal)
    blk_it_nm = STRICT_LOWER_ITERATOR(n, k, n, lower_blk_traversal)
    blk_it_m = STRICT_LOWER_ITERATOR(m, lower_blk_traversal)


    schurAngular_SkewSymm!(M_saf, M, wsp_saf; regular = true, order = true)
    compute_dexp_SkewSymm_both_system!(M_sys, M_saf.angle; trans = true)

    @time stlog_newton_descent_both!(Δ1, Z1, M, M_sys, M_saf, k, rs, blk_it_nm, blk_it_m, wsp_bgs; Stop=terminator(max(rs, 200), 50000, 1e-7, 1e-6))
    @time stlog_newton_descent_both!(Δ1, Z1, M, M_sys, M_saf, k, rs, blk_it_nm, blk_it_m, blk_it_n, wsp_bgs; Stop=terminator(max(rs, 200), 50000, 1e-7, 1e-6))

    
    println("solving newton descent direction with raw loops.")
    @btime stlog_newton_descent_both!($Δ1, $Z1, $M, $M_sys, $M_saf, $k, $rs, $blk_it_nm, $blk_it_m, $wsp_bgs; Stop=terminator(max($rs, 200), 50000, 1e-7, 1e-6))

    println("solving newton descent direction with $(Threads.nthreads()) threads.")
    @btime stlog_newton_descent_both!($Δ2, $Z2, $M, $M_sys, $M_saf, $k, $rs, $blk_it_nm, $blk_it_m, $blk_it_n, $wsp_bgs; Stop=terminator(max($rs, 200), 50000, 1e-7, 1e-6))

    println("Same result? \t", MatZ1 ≈ MatZ2)

    println("solving BCH third order update.")
    @btime stlog_BCH3_direction_lyap!($Z3, $M, $B, $C, $R);


end

function test_BCH3_direction(n, k)
    m = n - k
    MatM = rand(n, n)
    MatM .-= MatM'

    MatB1 = zeros(m, k)
    MatB2 = zeros(max(m, k), min(m, k))
    MatC = zeros(m, m)

    MatR = zeros(m, m)
    VecS = zeros(m)

    MatZ1 = zeros(m, m)
    MatZ2 = zeros(m, m)

    M = Ref(MatM)
    B1 = Ref(MatB1)
    B2 = Ref(MatB2)
    C = Ref(MatC)

    R = Ref(MatR)
    S = Ref(VecS)

    Z1 = Ref(MatZ1)
    Z2 = Ref(MatZ2)

    stlog_BCH3_direction_lyap!(Z1, M, B1, C, R);

    stlog_BCH3_direction_svd!(Z2, M, B2, R, S, C)

    println("Same answer?\t", MatZ1 ≈ MatZ2)
end

function test_BCH3_2k_direction_speed(k_grid, runs = 10; filename = "")
    RecTime = Matrix{Float64}(undef, length(k_grid) * runs, 2)

    k_vec = vcat(ones(runs) * k_grid'...)

    record_ind::Int = 1
    for k in k_grid
        m = k
        n = 2k

        MatM = zeros(n, n);

        MatB1 = zeros(m, k)
        MatB2 = zeros(max(m, k), min(m, k))
        MatC = zeros(m, m)

        MatR = zeros(m, m)
        VecS = zeros(m)

        MatZ1 = zeros(m, m)
        MatZ2 = zeros(m, m)
    
        M = Ref(MatM)
        B1 = Ref(MatB1)
        B2 = Ref(MatB2)
        C = Ref(MatC)
    
        R = Ref(MatR)
        S = Ref(VecS)
    
        Z1 = Ref(MatZ1)
        Z2 = Ref(MatZ2)
        
        for r_ind in 1:runs

            MatM .= rand(n, n);
            MatM .-= MatM';
            
            stat = @timed stlog_BCH3_direction_lyap!(Z1, M, B1, C, R);
            RecTime[record_ind, 1] = 1000 * (stat.time - stat.gctime)

            stat = @timed stlog_BCH3_direction_svd!(Z2, M, B2, R, S, C)
            RecTime[record_ind, 2] = 1000 * (stat.time - stat.gctime)

            record_ind += 1
        end
    end

    time_plt = scatter(k_vec, RecTime,
        label=["BCH3 direction, Lyap implementation" "BCH3 direction, SVD implementation"],
        # xlabel="2-norm of the generating velocity, |S_{A,B,0}|_2",
        ylabel="Compute time (ms)",
        # ylims = (0.0, 8 * median(RecTime)),
        yscale=:log2,
        markerstrokewidth=0,
        lw=0,
        ms=1.5,
        ma=0.3
    )

    rate_plt = scatter(k_vec, RecTime[:, 1] ./ RecTime,
        label=:none,
        xlabel="dimension k in 2k × k system",
        ylabel="Ratio",
        # ylims = (0.0, 5),
        yscale=:log2,
        markerstrokewidth=0,
        lw=0,
        ms=1.5,
        ma=0.3
    )

    plt = plot(layout = (2, 1), size = (800, 600), time_plt, rate_plt)

    display(plt)

    if filename != ""
        savefig(plt, filename)
    end
end

function test_stlog_newton_direction_thread_speed(k_grid, runs = 10; filename = "")
    RecTime = Matrix{Float64}(undef, length(k_grid) * runs, 3)

    k_vec = vcat(ones(runs) * k_grid'...)

    record_ind::Int = 1
    for k in k_grid
        m = k
        n = 2k
        rs = div(m * (m - 1), 2)
        d = rs

        MatM = zeros(n, n);

        MatΔ1 = similar(MatM)
        MatΔ2 = similar(MatM)

        MatZ1 = zeros(m, m)
        MatZ2 = zeros(m, m)
        MatZ3 = zeros(m, m)
        MatΔ1 = zeros(n, n)
        MatΔ2 = zeros(n, n)
        MatΔ3 = zeros(n, n)
    
        M = Ref(MatM)
    
        Z1 = Ref(MatZ1)
        Z2 = Ref(MatZ2)
        Z3 = Ref(MatZ3)
        Δ1 = Ref(MatΔ1)
        Δ2 = Ref(MatΔ2)
        Δ3 = Ref(MatΔ3)

        M_sys = dexp_SkewSymm_system(n);
        M_saf = SAFactor(n);
    
        wsp_saf = get_wsp_saf(n);
        wsp_bgs = get_wsp_bgs(n, d, rs)
        wsp_action = wsp_bgs[3]

    
        blk_it_n = STRICT_LOWER_ITERATOR(n, lower_blk_traversal)
        blk_it_nm = STRICT_LOWER_ITERATOR(n, k, n, lower_blk_traversal)
        blk_it_m = STRICT_LOWER_ITERATOR(m, lower_blk_traversal)

        A = _STLOG_BACKWARD_Z_ACTION(M_sys, M_saf, blk_it_nm, blk_it_n, wsp_action)

        time::Float64 = 0.0;


        Stop = terminator(max(rs, d), 50000, 1e-7, 1e-6)
    
        for r_ind in 1:runs

            MatM .= rand(n, n);
            MatM .-= MatM';
            MatM .*= (0.5 + 0.6 * rand()) * π / opnorm(MatM)

            schurAngular_SkewSymm!(M_saf, M, wsp_saf; regular = true, order = true)
            compute_dexp_SkewSymm_both_system!(M_sys, M_saf.angle; trans = true)

            time = 10000000
            for s_ind = 1:10
                time = min(time, @elapsed stlog_newton_descent_both!(Δ1, Z1, M, M_sys, M_saf, k, rs, blk_it_nm, blk_it_m, wsp_bgs; Stop=Stop))
            end
            RecTime[record_ind, 1] = time * 1e3

            time = 10000000
            for s_ind = 1:10
                time = min(time, @elapsed stlog_newton_descent_both!(Δ2, Z2, M, M_sys, M_saf, k, rs, blk_it_nm, blk_it_m, blk_it_n, wsp_bgs; Stop=Stop))
            end
            RecTime[record_ind, 2] = time * 1e3

            time = 10000000
            for s_ind = 1:10
                time = min(time, @elapsed stlog_newton_descent_gmres!(Δ3, Z3, M, A, rs, blk_it_m, wsp_bgs; Stop=Stop))
            end

            RecTime[record_ind, 3] = time * 1e3

            record_ind += 1
        end
    end

    time_plt = scatter(k_vec, RecTime,
        label=["Customize GMRES, raw loops" "Customize GMRES, threads" "IterativeSolvers GMRES, threads"],
        # xlabel="2-norm of the generating velocity, |S_{A,B,0}|_2",
        ylabel="Compute time (ms)",
        # ylims = (0.0, 8 * median(RecTime)),
        # yscale=:log2,
        markerstrokewidth=0,
        lw=0,
        ms=1.5,
        ma=0.3
    )

    rate_plt = scatter(k_vec, RecTime[:, 1] ./ RecTime,
        label=:none,
        xlabel="dimension k in 2k × k system",
        ylabel="Ratio",
        # ylims = (0.0, 5),
        # yscale=:log2,
        markerstrokewidth=0,
        lw=0,
        ms=1.5,
        ma=0.3
    )

    plt = plot(layout = (2, 1), size = (600, 800), time_plt, rate_plt)

    plt_log = plot(layout = (2, 1), size = (600, 800), yscale = :log2, time_plt, rate_plt)


    display(plt)
    display(plt_log)


    if filename != ""
        savefig(plt, filename)
    end
end

function test_stlog_newton_implementation_speed(k_grid, runs = 10; filename = "")
    RecTime = Matrix{Float64}(undef, length(k_grid) * runs, 2)

    k_vec = vcat(ones(runs) * k_grid'...)

    record_ind::Int = 1
    for k in k_grid
        m = k
        n = 2k
        rs = div(m * (m - 1), 2)
        d = rs

        MatM = zeros(n, n);

        MatΔ1 = similar(MatM)
        MatΔ2 = similar(MatM)

        MatZ1 = zeros(m, m)
        MatZ2 = zeros(m, m)
        MatZ3 = zeros(m, m)
        MatΔ1 = zeros(n, n)
        MatΔ2 = zeros(n, n)
        MatΔ3 = zeros(n, n)
    
        M = Ref(MatM)
    
        Z1 = Ref(MatZ1)
        Z2 = Ref(MatZ2)
        Z3 = Ref(MatZ3)
        Δ1 = Ref(MatΔ1)
        Δ2 = Ref(MatΔ2)
        Δ3 = Ref(MatΔ3)

        M_sys = dexp_SkewSymm_system(n);
        M_saf = SAFactor(n);
    
        wsp_saf = get_wsp_saf(n);
        wsp_bgs = get_wsp_bgs(n, k, d, rs)
        wsp_action = wsp_bgs[3]

    
        blk_it_n = STRICT_LOWER_ITERATOR(n, lower_blk_traversal)
        blk_it_nm = STRICT_LOWER_ITERATOR(n, k, n, lower_blk_traversal)
        blk_it_m = STRICT_LOWER_ITERATOR(m, lower_blk_traversal)

        A = _STLOG_BACKWARD_Z_ACTION(M_sys, M_saf, blk_it_nm, blk_it_n, wsp_action)

        time::Float64 = 0.0;


        Stop = terminator(max(rs, d), 50000, 1e-7, 1e-6)
    
        for r_ind in 1:runs

            MatM .= rand(n, n);
            MatM .-= MatM';
            MatM .*= (0.5 + 0.6 * rand()) * π / opnorm(MatM)

            schurAngular_SkewSymm!(M_saf, M, wsp_saf; regular = true, order = true)
            compute_dexp_SkewSymm_both_system!(M_sys, M_saf.angle; trans = true)

            # time = 10000000
            # for s_ind = 1:10
            #     time = min(time, @elapsed stlog_newton_descent_both!(Δ1, Z1, M, M_sys, M_saf, k, rs, blk_it_nm, blk_it_m, wsp_bgs; Stop=Stop))
            # end
            # RecTime[record_ind, 1] = time * 1e3

            time = 10000000
            for s_ind = 1:10
                time = min(time, @elapsed stlog_newton_descent_both!(Δ2, Z2, M, M_sys, M_saf, k, rs, blk_it_nm, blk_it_m, blk_it_n, wsp_bgs; Stop=Stop))
            end
            RecTime[record_ind, 2] = time * 1e3

            time = 10000000
            for s_ind = 1:10
                time = min(time, @elapsed stlog_newton_descent_gmres!(Δ3, Z3, M, A, rs, blk_it_m, wsp_bgs; Stop=Stop))
            end

            RecTime[record_ind, 1] = time * 1e3

            record_ind += 1
        end
    end

    time_plt = scatter(k_vec, RecTime,
        label=["IterativeSolvers GMRES, threads" "Customize GMRES, threads"],
        # xlabel="2-norm of the generating velocity, |S_{A,B,0}|_2",
        ylabel="Compute time (ms)",
        # ylims = (0.0, 8 * median(RecTime)),
        # yscale=:log2,
        markerstrokewidth=0,
        lw=0,
        ms=1.5,
        ma=0.3
    )

    rate_plt = scatter(k_vec, RecTime[:, 1] ./ RecTime,
        label=:none,
        xlabel="dimension k in 2k × k system",
        ylabel="Ratio",
        # ylims = (0.0, 5),
        # yscale=:log2,
        markerstrokewidth=0,
        lw=0,
        ms=1.5,
        ma=0.3
    )

    plt = plot(layout = (2, 1), size = (600, 800), time_plt, rate_plt)

    plt_log = plot(layout = (2, 1), size = (600, 800), yscale = :log2, time_plt, rate_plt)


    display(plt)
    display(plt_log)


    if filename != ""
        savefig(plt, filename)
    end
end

function test_stlog_newton_bch3_speed(k_grid, runs = 10; filename = "")
    RecTime = Matrix{Float64}(undef, length(k_grid) * runs, 2)

    k_vec = vcat(ones(runs) * k_grid'...)

    record_ind::Int = 1
    for k in k_grid
        m = k
        n = 2k
        rs = div(m * (m - 1), 2)
        d = rs

        MatM = zeros(n, n);

        MatΔ1 = similar(MatM)
        MatΔ2 = similar(MatM)

        MatZ1 = zeros(m, m)
        MatZ2 = zeros(m, m)
        MatZ3 = zeros(m, m)
        MatΔ1 = zeros(n, n)
        MatΔ2 = zeros(n, n)
        MatΔ3 = zeros(n, n)

        MatB = zeros(m, k)
        MatC = zeros(m, m)
    
        MatR = zeros(m, m)
        VecS = zeros(m)

        M = Ref(MatM)
    
    
        Z1 = Ref(MatZ1)
        Z2 = Ref(MatZ2)
        Z3 = Ref(MatZ3)
        Δ1 = Ref(MatΔ1)
        Δ2 = Ref(MatΔ2)
        Δ3 = Ref(MatΔ3)

        B = Ref(MatB)
        C = Ref(MatC)
    
        R = Ref(MatR)
        S = Ref(VecS)

        M_sys = dexp_SkewSymm_system(n);
        M_saf = SAFactor(n);
    
        wsp_saf = get_wsp_saf(n);
        wsp_bgs = get_wsp_bgs(n, k, d, rs)
        wsp_action = wsp_bgs[3]

    
        blk_it_n = STRICT_LOWER_ITERATOR(n, lower_blk_traversal)
        blk_it_nm = STRICT_LOWER_ITERATOR(n, k, n, lower_blk_traversal)
        blk_it_m = STRICT_LOWER_ITERATOR(m, lower_blk_traversal)

        A = _STLOG_BACKWARD_Z_ACTION(M_sys, M_saf, blk_it_nm, blk_it_n, wsp_action)

        time::Float64 = 0.0;


        Stop = terminator(max(rs, d), 50000, 1e-7, 1e-6)
    
        for r_ind in 1:runs

            MatM .= rand(n, n);
            MatM .-= MatM';
            MatM .*= (0.5 + 0.6 * rand()) * π / opnorm(MatM)

            schurAngular_SkewSymm!(M_saf, M, wsp_saf; regular = true, order = true)
            compute_dexp_SkewSymm_both_system!(M_sys, M_saf.angle; trans = true)

            # time = 10000000
            # for s_ind = 1:10
            #     time = min(time, @elapsed stlog_newton_descent_both!(Δ1, Z1, M, M_sys, M_saf, k, rs, blk_it_nm, blk_it_m, wsp_bgs; Stop=Stop))
            # end
            # RecTime[record_ind, 1] = time * 1e3

            time = 10000000
            for s_ind = 1:10
                time = min(time, @elapsed stlog_BCH3_direction_lyap!(Z1, M, B, C, R);)
            end
            RecTime[record_ind, 1] = time * 1e3

            time = 10000000
            for s_ind = 1:10
                time = min(time, @elapsed stlog_newton_descent_both!(Δ2, Z2, M, M_sys, M_saf, k, rs, blk_it_nm, blk_it_m, blk_it_n, wsp_bgs; Stop=Stop))
            end
            RecTime[record_ind, 2] = time * 1e3

            

            record_ind += 1
        end
    end

    time_plt = scatter(k_vec, RecTime,
        label=["BCH3 direction, Lyapunov solver" "Newton direction, GMRES solver"],
        # xlabel="2-norm of the generating velocity, |S_{A,B,0}|_2",
        ylabel="Compute time (ms)",
        # ylims = (0.0, 8 * median(RecTime)),
        # yscale=:log2,
        markerstrokeshape=[:circle :star5],
        markerstrokecolor=:auto,
        lw=0,
        ms=1.5,
        ma=0.5
    )

    rate_plt = scatter(k_vec, RecTime[:, 1] ./ RecTime,
        label=:none,
        xlabel="dimension k in 2k × k system",
        ylabel="Ratio of time to the BCH3 direction solver",
        # ylims = (0.0, 5),
        # yscale=:log2,
        markerstrokeshape=[:circle :star5],
        markerstrokecolor=:auto,
        lw=0,
        ms=1.5,
        ma=0.5
    )

    plt = plot(layout = (2, 1), size = (600, 800), time_plt, rate_plt)

    plt_log = plot(layout = (2, 1), size = (600, 800), yscale = :log2, time_plt, rate_plt)


    display(plt)
    display(plt_log)


    if filename != ""
        pos = findlast('.', filename)
        savefig(plot(time_plt), filename[1:(pos - 1)] * "_time." * filename[(pos + 1):end])
        savefig(plot(rate_plt), filename[1:(pos - 1)] * "_rate." * filename[(pos + 1):end])
        savefig(plot(yscale = :log2, time_plt), filename[1:(pos - 1)] * "_time_logscale." * filename[(pos + 1):end])
        savefig(plot(yscale = :log2, rate_plt), filename[1:(pos - 1)] * "_rate_action_logscale." * filename[(pos + 1):end])
    end
end