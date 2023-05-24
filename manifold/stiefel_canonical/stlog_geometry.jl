# This code implements the basic geometry operations needed for 
# the stlog working on the space of all special orthogonal
# completion to Uk, which is essentially SO_{n-k}.
# For a point Up with velocity Up⋅Z where Z = -Z' is skew-sym,
# the geodesic(retraction)
# is simply given by the matrix exponential:
#   ret_{Up}(Up⋅Z) = Up⋅exp(Z).
# Since for ever point Up, the skew symmetric matrix
#   S   = log([Uk Up])
# is required for stlog. The retraction also computes it via
# schur decomposition implemented in 
#   manifold/spec_orth/spec_orth.jl
# which is calling dgees implemented in LAPCAK.

include("../../inc/global_path.jl")
include(joinpath(JULIA_MANIFOLD_PATH, "spec_orth/so_explog.jl"))


# include("stlog_wsp_gen.jl")
# include(homedir() * "/Documents/julia/manifold/spec_orth/spec_orth.jl")

@inline inner_skew!(S1::Ref{Matrix{Float64}}, S2::Ref{Matrix{Float64}}) = dot(S1[], S2[]) / 2.0;
@inline inner_skew!(S1::Ref{Vector{Float64}}, S2::Ref{Vector{Float64}}; lower = true) = if lower return dot(S1[], S2[]); else return dot(S1[], S2[]) / 2.0; end

@inline function stlog_cost(S::Ref{Matrix{Float64}}, k::Int)
    MatS = S[];
    n = size(MatS, 1);
    fval::Float64 = 0.0;
    for c_ind = (k + 1):n
        for r_ind = (c_ind + 1):n
            @inbounds fval += MatS[r_ind, c_ind] * MatS[r_ind, c_ind];
        end
    end
    return fval / 2.0;
end


function scale_velocity_UpZ!(αZ::Ref{Matrix{Float64}}, Z::Ref{Matrix{Float64}}, α::Float64)
    MatαZ = αZ[]
    MatZ = Z[]

    if α == 1.0
        unsafe_copyto!(pointer(MatαZ), pointer(MatZ), length(MatZ))
    else
        for ind in eachindex(MatZ)
            @inbounds MatαZ[ind] = α * MatZ[ind];
        end
    end
end

# @inline update_point_UpZ!(Up_r::Ref{Matrix{Float64}}, Up_new_r::Ref{Matrix{Float64}}) = unsafe_copyto!(pointer(Up_r[]), pointer(Up_new_r[]), length(Up_r[]))

@inline get_wsp_stlog_UpZ_ret(n::Int, k::Int) = WSP(Matrix{Float64}(undef, n - k, n - k), Matrix{Float64}(undef, n, n - k), get_wsp_saf(n - k), get_wsp_saf(n))
@inline get_wsp_stlog_UpZ_ret(n::Int, k::Int, wsp_saf_m::WSP, wsp_saf_n::WSP) = WSP(Matrix{Float64}(undef, n - k, n - k), Matrix{Float64}(undef, n, n - k), wsp_saf_m, wsp_saf_n)

function ret_UpZ!(U::Ref{Matrix{Float64}}, Up::Ref{Matrix{Float64}}, M::Ref{Matrix{Float64}}, M_saf::SAFactor, Z::Ref{Matrix{Float64}}, Z_saf::SAFactor, wsp_stlog_UpZ_ret::WSP; nearlog::Bool = false)
    # wsp_ret carrys real n-k x n-k matrix Mm, n x m matrix Mnm, workspaces wsp_exp and wsp_log
    MatU = U[];
    MatZ = Z[];
    MatUp = Up[];
    MatTmpm = wsp_stlog_UpZ_ret[1];
    MatTmpnm = wsp_stlog_UpZ_ret[2];
    wsp_saf_m = wsp_stlog_UpZ_ret[3];
    wsp_saf_n = wsp_stlog_UpZ_ret[4];

    Tmpm = wsp_stlog_UpZ_ret(1);


    n::Int, m::Int = size(MatUp);
    k::Int = n - m;

    # exp_SkewSymm!(Tmpm, Z_saf, Z, wsp_saf_m)                                    # Q <- exp(Z)
    copy!(MatTmpm, exp(MatZ))


    unsafe_copyto!(pointer(MatTmpnm), pointer(MatUp), length(MatUp));

    mul!(MatUp, MatTmpnm, MatTmpm)                                              # Up <- Up * Q

    unsafe_copyto!(MatU, n * k + 1, MatUp, 1, length(MatUp))                    # Write Up to U

    if nearlog
        nearlog_SpecOrth!(M, M_saf, U, M, wsp_saf_n; order = true, regular = true);        # Get the nearest log of U from M and overwrite it to M
    else
        log_SpecOrth!(M, M_saf, U, wsp_saf_n; order = true, regular = true);
    end
end

function ret_UpZ!(U::Ref{Matrix{Float64}}, Up::Ref{Matrix{Float64}}, M::Ref{Matrix{Float64}}, M_saf::SAFactor, Z::Ref{Matrix{Float64}}, Z_saf::SAFactor, scale::Float64, wsp_stlog_UpZ_ret::WSP; nearlog::Bool = false)
    # wsp_ret carrys real n-k x n-k matrix Mm, n x m matrix Mnm, workspaces wsp_exp and wsp_log

    # The SAFactorization of Z is assumed to be done and therefore this routine directly operates on Z_saf

    MatU = U[];
    MatZ = Z[];
    MatUp = Up[];
    MatTmpm = wsp_stlog_UpZ_ret[1];
    MatTmpnm = wsp_stlog_UpZ_ret[2];
    wsp_saf_n = wsp_stlog_UpZ_ret[4];
    Tmpm = wsp_stlog_UpZ_ret(1);


    n::Int, m::Int = size(MatUp);
    k::Int = n - m;

    # exp_SkewSymm!(Tmpm, Z_saf, Z, wsp_saf_m)                                   # Q <- exp(Z)                    
    # computeSpecOrth!(Tmpm, Z_saf, scale)                                         # Q <- exp(Z)

    for ind in eachindex(MatZ)
        @inbounds MatTmpm[ind] = scale * MatZ[ind];
    end
    copy!(MatTmpm, exp(MatTmpm))

    unsafe_copyto!(pointer(MatTmpnm), pointer(MatUp), length(MatUp));
    mul!(MatUp, MatTmpnm, MatTmpm)                                              # Up <- Up * Q

    unsafe_copyto!(MatU, n * k + 1, MatUp, 1, length(MatUp))                    # Write Up to U

    if nearlog
        nearlog_SpecOrth!(M, M_saf, U, M, wsp_saf_n; order = true, regular = true);        # Get the nearest log of U from M and overwrite it to M
    else
        log_SpecOrth!(M, M_saf, U, wsp_saf_n; order = true, regular = true);
    end
end

function ret_UpZ_builtin_exp!(U::Ref{Matrix{Float64}}, Up::Ref{Matrix{Float64}}, M::Ref{Matrix{Float64}}, M_saf::SAFactor, Z::Ref{Matrix{Float64}}, wsp_stlog_UpZ_ret::WSP; nearlog::Bool = false)
    # wsp_ret carrys real n-k x n-k matrix Mm, n x m matrix Mnm, workspaces wsp_exp and wsp_log
    MatU = U[];
    MatUp = Up[];
    MatTmpm = wsp_stlog_UpZ_ret[1];
    MatTmpnm = wsp_stlog_UpZ_ret[2];
    wsp_saf_m = wsp_stlog_UpZ_ret[3];
    wsp_saf_n = wsp_stlog_UpZ_ret[4];

    Tmpm = wsp_stlog_UpZ_ret(1);


    n::Int, m::Int = size(MatUp);
    k::Int = n - m;

    # Q <- exp(Z), using scaling and squaring algorithm
    # Z needs to ensure skew symmetry to guarantee special orthogonality in this exp(Z).
    MatZ = Z[]
    for r_ind = 1:m
        for c_ind = 1:m
            @inbounds MatZ[r_ind, c_ind] = (MatZ[r_ind, c_ind] - MatZ[c_ind, r_ind]) / 2.0
            @inbounds MatZ[c_ind, r_ind] = -MatZ[r_ind, c_ind]
        end
    end
    MatTmpm .= exp(MatZ)                                
    # exp_SkewSymm!(Tmpm, Z_saf, Z, wsp_saf_m)                                   
    

    unsafe_copyto!(pointer(MatTmpnm), pointer(MatUp), length(MatUp));

    mul!(MatUp, MatTmpnm, MatTmpm)                                              # Up <- Up * Q

    unsafe_copyto!(MatU, n * k + 1, MatUp, 1, length(MatUp))                    # Write Up to U

    if nearlog
        nearlog_SpecOrth!(M, M_saf, U, M, wsp_saf_n; order = true, regular = true);        # Get the nearest log of U from M and overwrite it to M
    else
        log_SpecOrth!(M, M_saf, U, wsp_saf_n; order = false, regular = false);
    end
end

function ret_UpZ_builtin_explog!(U::Ref{Matrix{Float64}}, Up::Ref{Matrix{Float64}}, M::Ref{Matrix{Float64}}, Z::Ref{Matrix{Float64}}, wsp_stlog_UpZ_ret::WSP)
    # wsp_ret carrys real n-k x n-k matrix Mm, n x m matrix Mnm, workspaces wsp_exp and wsp_log
    MatU = U[];
    MatUp = Up[];
    MatTmpm = wsp_stlog_UpZ_ret[1];
    MatTmpnm = wsp_stlog_UpZ_ret[2];
    wsp_saf_m = wsp_stlog_UpZ_ret[3];
    wsp_saf_n = wsp_stlog_UpZ_ret[4];

    Tmpm = wsp_stlog_UpZ_ret(1);


    n::Int, m::Int = size(MatUp);
    k::Int = n - m;

    # Q <- exp(Z), using scaling and squaring algorithm
    # Z needs to ensure skew symmetry to guarantee special orthogonality in this exp(Z).
    MatZ = Z[]
    for r_ind = 1:m
        for c_ind = 1:m
            @inbounds MatZ[r_ind, c_ind] = (MatZ[r_ind, c_ind] - MatZ[c_ind, r_ind]) / 2.0
            @inbounds MatZ[c_ind, r_ind] = -MatZ[r_ind, c_ind]
        end
    end
    MatTmpm .= exp(MatZ)                                
    # exp_SkewSymm!(Tmpm, Z_saf, Z, wsp_saf_m)                                   
    

    unsafe_copyto!(pointer(MatTmpnm), pointer(MatUp), length(MatUp));

    mul!(MatUp, MatTmpnm, MatTmpm)                                              # Up <- Up * Q

    unsafe_copyto!(MatU, n * k + 1, MatUp, 1, length(MatUp))                    # Write Up to U

    MatM = M[]
    MatM .= real.(log(MatU))

end



function nip_ret_UpZ!(U::Ref{Matrix{Float64}}, Up_new::Ref{Matrix{Float64}}, Up::Ref{Matrix{Float64}}, M_new::Ref{Matrix{Float64}}, M::Ref{Matrix{Float64}}, M_saf::SAFactor, 
    Z::Ref{Matrix{Float64}}, Z_saf::SAFactor, wsp_stlog_UpZ_ret::WSP; nearlog::Bool = false)
    copy!(Up_new[], Up[]);
    copy!(M_new[], M[]);

    ret_UpZ!(U, Up_new, M_new, M_saf, Z, Z_saf, wsp_stlog_UpZ_ret; nearlog = nearlog)
end

function nip_ret_UpZ!(U::Ref{Matrix{Float64}}, Up_new::Ref{Matrix{Float64}}, Up::Ref{Matrix{Float64}}, M_new::Ref{Matrix{Float64}}, M::Ref{Matrix{Float64}}, M_saf::SAFactor, 
    Z::Ref{Matrix{Float64}}, Z_saf::SAFactor, scale::Float64, wsp_stlog_UpZ_ret::WSP; nearlog::Bool = false)
    copy!(Up_new[], Up[]);
    copy!(M_new[], M[]);

    ret_UpZ!(U, Up_new, M_new, M_saf, Z, Z_saf, scale, wsp_stlog_UpZ_ret; nearlog = nearlog)
end


#######################################Test functions#######################################

function test_stlog_ret_UpZ(n, k)
    m = n - k;

    X = rand(n, n);
    X .-= X';

    Q = exp_SkewSymm(Ref(X));
    M = log_SpecOrth(Ref(Q));
    Qm = zeros(n, m);
    Qm .= Q[:, (k + 1):n];

    Z = rand(m, m);
    Z .-= Z';

    M_saf = SAFactor(n);
    Z_saf = SAFactor(m);

    Q1 = exp_SkewSymm(Ref(X));
    Q2 = exp_SkewSymm(Ref(X));


    ret_UpZ!(Ref(Q1), Ref(Qm), Ref(M), M_saf, Ref(Z), Z_saf, get_wsp_stlog_UpZ_ret(n, k))

    FulZ = zeros(n, n);
    FulZ[(k + 1):n, (k + 1):n] .= Z;
    Q2 .= Q * exp(FulZ);

    println(Q1 ≈ Q2)

end 
