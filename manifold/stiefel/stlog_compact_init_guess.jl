
########################initial guess########################

# For given Uk in St_{n,k}, stlog search on the fiber over Uk 
# in SO_n, which is essentially searching on the space of all
# special orthogonal completion (SOC) to Uk. The routines 
# provided here are returning specific SOC to the input Uk, 
# which will be used as the initial guess for the stlog search.
# 
# qrBlkHH obtains the SOC by the qr decomposition of Uk with
# block Householder reflection:
#   Uk  = QI_{n,k} 
#       = (I - V_{n×(k-1)} C_{(k-1)×(k-1)} V_{n×(k-1)}^T)I_{n,k}
# where V, C are obtained by the dgeqrf implemented in LAPACK.
# The k - 1 comes from the number of sequential Householder
# reflectors H(1), H(2), ..., H(k-1) used in the row-elimination,
# which further forms Q = H(1)H(2)...H(k - 1). The I_{n,k} comes
# from the nature of upper triangular orthonormal basis. 
# Therefore, Q is an orthogonal completion to Uk with determinant
# det(Q) = (-1) ^ (k - 1).
# When k is odd, Q is returned as the demanded SOC to Uk.
# When k is even, the k + 1 column of Q is flipped to make it
# special orthogonal. Then the flipped Q is returned.

# project obtians the SOC via a projection process. For the given
# Uk, project the Eucldiean difference (flat velocity) 
#   D   = Uk - I_{n,k}
# to the tangent space at I_{n,k}, (becomes manifold velocity)
#   ξ   = Proj_{TSt_{n,k} _ {I_n,k}}(D).
# The velocity ξ is then lifted to the tangent space of SO_n
# at I_n over I_{n, k} as
#   η   = Lift(ξ).
# Following the geodesic along η, it gives a special orthogonal
#   V   = exp(η) = [Vk Vp]
# which in general has Vk ≠ Uk. Then project the Vp
#   Up  = Proj_{ℛ_{⟂} (Uk)}(Vp)
# to get the SOC to Uk as demanded.

# grassmann obtains the SOC via the lifted grassmann geodesic.
# For the given Uk, find the Grassmann geodesic between the
# subspaces generated by I_{n,k} and U_k:
#   ξ = log_{ℛ(I_{n,k})}(ℛ(U_k))
# and lift it to the special orthogonal group at I_n
#   η = Lift(ξ).
# Following η, one can get
#   V = exp(η) = [Vk Vp]
# where Vk ⊂ ℛ(Uk) and Vp ⊂ ℛ_{⟂}(Uk), i.e., Vp is a SOC
# to Uk as demanded.

########################initial guess########################

function stlog_init_guess_simple(Uk)
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
    return Up
end

function stlog_init_guess_complex(Uk)
    n, k = size(Uk)
    S = zeros(n, n)
    S[(k+1):n, 1:k] .= Uk[(k+1):n, 1:k]
    S[1:k, (k+1):n] .= -Uk[(k+1):n, 1:k]'
    S[1:k, 1:k] .= 0.5 .* (Uk[1:k, 1:k] .- Uk[1:k, 1:k]')
    V = exp(S)
    qr_factor = qr(hcat(V[:, (k+1):n] .- Uk * (Uk' * V[:, (k+1):n])))
    Up = qr_factor.Q * Matrix(I, n, n - k)
    if det(hcat(Uk, Up)) < 0
        min_i = 1
        min_r = qr_factor.R[1, 1]
        for ii = 2:(n-k)
            if qr_factor.R[ii, ii] < min_r
                min_r = qr_factor.R[ii, ii]
                min_i = ii
            end
        end
        Up[:, min_i] .*= -1
    end
    return Up
end

function stlog_init_guess_grassmann(Uk)
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
    return Up
end