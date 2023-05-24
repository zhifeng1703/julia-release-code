
using LinearAlgebra
########################preprocessing########################

# For problem of finding geodesic U_nk -> V_nk, the preprocessing convert the problem of finding orthogonal Q = diag(I_k, Q_r), 
# such that the target geodesic can be written as Q_nd * I_dk -> Q_nd * W_dk, where Q = [Q_nd, Q_rest]. In this way, the problem 
# is converted to finding geodesic I_dk -> W_dk. 
# To be consistent with other notation, we have n = k + r, d = k + m and, therefore, 1 < m <= k, k + 1 < d <= 2k.
# For core computation where no conversion is applied on I_nk -> U_nk, we have n = k + m, i.e., r = m. 


########################preprocessing########################


function preprocessing_Ink_2k(Uk_r::Ref{Matrix{Float64}})

    Uk::Matrix{Float64} = Uk_r[]
    n::Int, k::Int = size(Uk)
    r::Int = k

    Vk::Matrix{Float64} = Matrix{Float64}(undef, k + r, k)

    @inbounds copy!(view(Vk, 1:k, 1:k), view(Uk, 1:k, 1:k))

    # Vk[1:k, :] .= Uk[1:k, :];

    QRFactor_rk = qr(Uk[(k+1):n, :])

    # display(QRFactor_rk.R)

    @inbounds copy!(view(Vk, (k + 1):(k + r), 1:k), QRFactor_rk.R)

    # Vk[(k + 1):(2k), :] .= N_qr.R;

    # returning Q_rr, which can also interpreted as Q_rm
    return Vk, QRFactor_rk.Q
end

function preprocessing_Ink_rank(Uk_r::Ref{Matrix{Float64}})

    Uk::Matrix{Float64} = Uk_r[]
    n::Int, k::Int = size(Uk)
    r = rank(Uk[(k + 1):n, :]);

    Vk::Matrix{Float64} = zeros(k + r, k)

    @inbounds copy!(view(Vk, 1:k, 1:k), view(Uk, 1:k, 1:k))

    QRFactor_rk = qr(Uk[(k+1):n, (k - r + 1):k])

    # display(Vk)
    # display(QRFactor_rk.R)

    @inbounds copy!(view(Vk, (k + 1):(k + r), (k - r + 1):k), QRFactor_rk.R)

    # Vk[(k + 1):(2k), :] .= N_qr.R;

    # returning Q_rr, which can also interpreted as Q_rm
    return Vk, QRFactor_rk.Q
end


function preprocessing_Ink_with_rank(Vk::Ref{Matrix{Float64}}, Uk::Ref{Matrix{Float64}})

    MatUk::Matrix{Float64} = Uk[]
    MatVk::Matrix{Float64} = Vk[]

    n::Int, k::Int = size(MatUk)
    r::Int = size(MatVk, 1) - k

    @inbounds copy!(view(MatVk, 1:k, 1:k), view(MatUk, 1:k, 1:k))

    # Vk[1:k, :] .= Uk[1:k, :];

    @inbounds QRFactor_rk = qr!(copy(view(MatUk, (k+1):n, (k-r+1):k)))
    # display(QRFactor_rk.R)

    @inbounds copy!(view(MatVk, (k+1):(k+r), (k-r+1):k), QRFactor_rk.R)

    # Vk[(k + 1):(2k), :] .= N_qr.R;

    # returning Q_rr, which can also interpreted as Q_rm
    return Vk
end

function post_processing_Ink(Unk_r::Ref{Matrix{Float64}}, Adk_r::Ref{Matrix{Float64}}, Q_rm)

    Unk::Matrix{Float64} = Unk_r[]
    Adk::Matrix{Float64} = Adk_r[]

    n::Int, k::Int = size(Unk)
    d::Int = size(Adk, 1)
    m::Int = d - k
    r::Int = n - k

    Ank::Matrix{Float64} = Matrix{Float64}(undef, n, k)

    for ii = 1:k
        for jj = 1:k
            Ank[ii, jj] = Adk[ii, jj]
        end
    end
    # Ans[1:k, :] .= Ans_2k[1:k, :];
    Mmk::Matrix{Float64} = Matrix{Float64}(undef, m, k)
    for ii = 1:m
        for jj = 1:k
            Mmk[ii, jj] = Adk[ii+k, jj]
        end
    end

    Mrk::Matrix{Float64} = Q_rm * Mmk
    for ii = 1:r
        for jj = 1:k
            Ank[ii+k, jj] = Mrk[ii, jj]
        end
    end

    # Adk = [Adk_kk // Adk_mk]
    # Q = diag(Ik, Qrr) = [Qnd & Qnx] 
    # Qnd = diag(Ik, Qrm)
    # Ank = [Adk_kk // Mrk = Qrm * Adk_mk] = Qnd * Adk

    return Ank
end
