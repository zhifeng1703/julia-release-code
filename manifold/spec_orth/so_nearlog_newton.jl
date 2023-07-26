include("../../inc/global_path.jl")
include(joinpath(JULIA_INCLUDE_PATH, "workspace.jl"))

include("so_safactor.jl")
include("dexp_SkewSymm.jl")

function nearlog_SpecOrth_newton(eX::Matrix{Float64}, S::Matrix{Float64}, eS::Matrix{Float64})
    # solve X from exp(X) = eX near exp(S) = eS.
    # eX_iter = exp(S);

    n = size(eX, 1)

    X_iter = copy(S);
    eX_iter = copy(eS);

    ΔQ_iter = eX_iter' * (eX .- eX_iter);
    ΔQ_iter .-= ΔQ_iter';
    ΔQ_iter .*= 0.5;

    ΔX_iter = zeros(n, n)

    X_saf_iter = schurAngular_SkewSymm(Ref(X_iter); order = true, regular = true);
    X_sys_iter = dexp_SkewSymm_system(n);
    compute_dexp_SkewSymm_both_system!(X_sys_iter, X_saf_iter.angle);

    iter = 1;
    blk_it_n = STRICT_LOWER_ITERATOR(n, lower_blk_traversal);
    wsp_cong_n = get_wsp_cong(n);
    wsp_saf = get_wsp_saf(n);

    while norm(ΔQ_iter) > 1e-8
        if iter > 100
            println("Near log by Newton's method failed.")
            return false, iter, X_iter;
        end
        dexp_SkewSymm!(Ref(ΔX_iter), Ref(ΔQ_iter), X_sys_iter, X_saf_iter, blk_it_n, wsp_cong_n; inv = true, cong = true, compact = true);

        X_iter .+= ΔX_iter;
        eX_iter .= exp(X_iter);

        schurAngular_SkewSymm!(X_saf_iter, Ref(X_iter), wsp_saf; order = true, regular = true);
        compute_dexp_SkewSymm_both_system!(X_sys_iter, X_saf_iter.angle);


        ΔQ_iter = eX_iter' * (eX .- eX_iter);
        ΔQ_iter .-= ΔQ_iter';
        ΔQ_iter .*= 0.5;

        iter += 1;
    end

    return true, iter, X_iter
end


#######################################Test functions#######################################


function test_nearlog_SpecOrth_newton(n, s, e)
    S = rand(n, n);
    S .-= S';
    S .*= s / opnorm(S);

    ΔS = rand(n, n);
    ΔS .-= ΔS';
    ΔS .*= e / opnorm(ΔS);

    eX = exp(S .+ ΔS);
    eS = exp(S);

    flag, iter, X = nearlog_SpecOrth_newton(eX, S, eS);
    println("Number of iterations used: $(iter)")
    display(X .- (S .+ ΔS))
end