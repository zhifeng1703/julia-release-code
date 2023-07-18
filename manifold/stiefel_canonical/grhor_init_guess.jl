include("../../inc/global_path.jl")

include(joinpath(JULIA_INCLUDE_PATH, "algorithm_port.jl"))

include("stlog_geometry.jl")

using Random

function grhor_init_guess_random(Uk; RandEng=nothing)
    n, k = size(Uk)
    m = n - k
    Up = zeros(n, m)
    qr_factor = qr(Uk)
    Q = qr_factor.Q * Matrix(I, n, n)
    for ii = 1:n
        for jj = 1:m
            Up[ii, jj] = Q[ii, jj+k]
        end
    end

    if det(hcat(Uk, Up)) < 0
        # Up[:, 1] .*= -1;
        for ii = 1:n
            Up[ii, 1] *= -1
        end
    end

    if isnothing(RandEng)
        Z = rand(m, m)
        Z .-= Z'
        Z .*= π * rand() / opnorm(Z)
    else
        Z = rand(RandEng, m, m)
        Z .-= Z'
        Z .*= π * rand(RandEng) / opnorm(Z)
    end

    Up = Up * exp(Z)
    Q = hcat(Uk, Up)
    A = log(Q)
    A .-= A'
    A .*= 0.5
    return A, Q
end

function grhor_init_guess_simple(Uk)
    n, k = size(Uk)
    m = n - k
    Up = zeros(n, m)
    qr_factor = qr(Uk)
    Q = qr_factor.Q * Matrix(I, n, n)
    for ii = 1:n
        for jj = 1:m
            Up[ii, jj] = Q[ii, jj+k]
        end
    end

    if det(hcat(Uk, Up)) < 0
        # Up[:, 1] .*= -1;
        for ii = 1:n
            Up[ii, 1] *= -1
        end
    end

    Z = rand(m, m)
    Z .-= Z'
    Z .*= π * rand() / opnorm(Z)
    Up = Up * exp(Z)

    Q = hcat(Uk, Up)
    A = log(Q)
    A .-= A'
    A .*= 0.5
    return A, Q
end

function grhor_init_guess_grassmann(Uk)
    n, k = size(Uk)
    factor1 = svd(Uk[1:k, :])
    factor2 = svd(Uk[(k+1):n, :])

    angles = -asin.(factor2.S)
    U2 = factor2.U
    Q = factor2.V' * factor1.V
    U1 = factor1.U * Q

    # display(asin.(factor2.S))
    # display(acos.(factor1.S[k:-1:1]))

    # display(hcat(cos.(angles), -sin.(angles)))
    # display(hcat(factor1.S[k:-1:1], factor2.S))

    A = zeros(n, n)
    A[(k+1):n, 1:k] .= -U2 * Diagonal(angles) * U1'

    # display(U1);
    # A[(k + 1):n, 1:k] .= Uk[(k + 1):n, :] * factor2.V * Diagonal(angles ./ factor2.S) * U1';
    A[1:k, (k+1):n] .= -A[(k+1):n, 1:k]'
    Q = exp(A)

    Vp = Q[:, (k+1):n]

    Vp .-= Uk * (Uk' * Vp)
    factor = svd(Vp)
    Up = factor.U * factor.Vt

    display(Uk' * Q)

    display(Uk' * Q[:, 1:k])
    display(det(Uk' * Q[:, 1:k]))



    return A, Q
end