include("spec_orth.jl")

function inner_skew(A, B)
    n, = size(A)
    Ans = 0.0
    for ii = 1:n
        for jj = 1:(ii-1)
            Ans += A[ii, jj] * B[ii, jj]
        end
    end
    Ans *= 2
    return Ans
end

function inner_flat(A, B)
    return sum(A .* B)
end

function inner_skew_22(A, B, k)
    n, = size(A)
    Ans = 0.0
    for ii = (k+1):n
        for jj = (k+1):(ii-1)
            Ans += A[ii, jj] * B[ii, jj]
        end
    end
    Ans *= 2
    return Ans
end

function inner_skew!(A_ref::Base.RefValue{Matrix{Float64}}, B_ref::Base.RefValue{Matrix{Float64}})
    A::Matrix{Float64} = A_ref[]
    B::Matrix{Float64} = B_ref[]
    n::Int, = size(A)
    Ans::Float64 = 0.0
    for ii = 1:n
        for jj = 1:(ii-1)
            Ans += A[ii, jj] * B[ii, jj]
        end
    end
    Ans *= 2.0
    return Ans
end

function inner_flat!(A_ref::Base.RefValue{Matrix{Float64}}, B_ref::Base.RefValue{Matrix{Float64}})
    A::Matrix{Float64} = A_ref[]
    B::Matrix{Float64} = B_ref[]
    return sum(A .* B)
end

function inner_skew_22!(A_ref::Base.RefValue{Matrix{Float64}}, B_ref::Base.RefValue{Matrix{Float64}}, k::Int)
    A::Matrix{Float64} = A_ref[]
    B::Matrix{Float64} = B_ref[]
    n::Int, = size(A)
    Ans::Float64 = 0.0
    for ii = (k+1):n
        for jj = (k+1):(ii-1)
            Ans += A[ii, jj] * B[ii, jj]
        end
    end
    Ans *= 2.0
    return Ans
end

norm2_skew(A) = inner_skew(A, A);
norm2_skew!(A_ref::Base.RefValue{Matrix{Float64}}) = inner_skew!(A_ref, A_ref);



function dexp_para(θi::Number, θj::Number)
    su = θi + θj
    di = θi - θj
    Ans = zeros(4)
    Ans[1] = (abs(di) < 1e-15) ? 1 : sin(di) / di
    Ans[2] = (abs(su) < 1e-15) ? 1 : sin(su) / su
    Ans[3] = (abs(di) < 1e-15) ? 0 : (cos(di) - 1) / di
    Ans[4] = (abs(su) < 1e-15) ? 0 : (cos(su) - 1) / su

    return Ans
end

function dexp_inv_para(θi::Number, θj::Number)
    f_para = dexp_para(θi, θj)
    Ans = zeros(4)
    n1 = f_para[1]^2 + f_para[3]^2
    n2 = f_para[2]^2 + f_para[4]^2

    Ans[1] = f_para[1] / n1
    Ans[2] = f_para[2] / n2
    Ans[3] = -f_para[3] / n1
    Ans[4] = -f_para[4] / n2
    return Ans
end

function dexp_para(θ::Number)
    Ans = zeros(2)
    Ans[1] = (abs(θ) < 1e-15) ? 1 : sin(θ) / θ
    Ans[2] = (abs(θ) < 1e-15) ? 0 : (cos(θ) - 1) / θ
    return Ans
end

function dexp_inv_para(θ::Number)
    f_para = dexp_para(θ)
    n = f_para[1]^2 + f_para[2]^2
    Ans = zeros(2)
    Ans[1] = f_para[1] / n
    Ans[2] = -f_para[2] / n
    return Ans
end

function dexp_para!(θi::Number, θj::Number, p1_r, ind)
    su = θi + θj
    di = θi - θj
    p1 = p1_r[]
    p1[ind, 1] = (abs(di) < 1e-15) ? 1 : sin(di) / di
    p1[ind, 2] = (abs(su) < 1e-15) ? 1 : sin(su) / su
    p1[ind, 3] = (abs(di) < 1e-15) ? 0 : (cos(di) - 1) / di
    p1[ind, 4] = (abs(su) < 1e-15) ? 0 : (cos(su) - 1) / su
end

function dexp_para!(θ::Number, p2_r, ind)
    p2 = p2_r[]
    p2[ind, 1] = (abs(θ) < 1e-15) ? 1 : sin(θ) / θ
    p2[ind, 2] = (abs(θ) < 1e-15) ? 0 : (cos(θ) - 1) / θ
end

function dexp_inv_para(θ::Number)
    f_para = dexp_para(θ)
    n = f_para[1]^2 + f_para[2]^2
    Ans = zeros(2)
    Ans[1] = f_para[1] / n
    Ans[2] = -f_para[2] / n
    return Ans
end

function dexp_action_22(S, para; Workspace=nothing)
    a, b, c, d = para
    Ans = zeros(2, 2)
    if !isnothing(Workspace)
        M = Workspace[]
    else
        M = zeros(4, 4)
    end
    M[1, 1] = a + b
    M[1, 2] = d - c
    M[1, 3] = c + d
    M[1, 4] = a - b
    M[2, 1] = c - d
    M[2, 2] = a + b
    M[2, 3] = b - a
    M[2, 4] = c + d
    M[3, 1] = -c - d
    M[3, 2] = b - a
    M[3, 3] = a + b
    M[3, 4] = d - c
    M[4, 1] = a - b
    M[4, 2] = -c - d
    M[4, 3] = c - d
    M[4, 4] = a + b
    x = zeros(4)
    x[1] = S[1, 1]
    x[2] = S[1, 2]
    x[3] = S[2, 1]
    x[4] = S[2, 2]
    # display(M)
    # println(x)
    x = M * x
    x .*= 0.5
    # println(x)
    Ans[1, 1] = x[1]
    Ans[1, 2] = x[2]
    Ans[2, 1] = x[3]
    Ans[2, 2] = x[4]
    return Ans
end

function dexp_action_12(S, para; Workspace=nothing)
    a, b = para
    Ans = zeros(2)
    if !isnothing(Workspace)
        M = Workspace[]
    else
        M = zeros(2, 2)
    end
    M[1, 1] = a
    M[1, 2] = b
    M[2, 1] = -b
    M[2, 2] = a
    x = zeros(2)
    x .= S
    x = M * x
    return x
end

function dexp_system22!(M4_r::Base.RefValue{Matrix{Float64}}, a::Float64, b::Float64, c::Float64, d::Float64)
    M = M4_r[]
    M[1, 1] = a + b
    M[2, 2] = M[1, 1]
    M[3, 3] = M[1, 1]
    M[4, 4] = M[1, 1]

    M[1, 4] = a - b
    M[4, 1] = M[1, 4]
    M[2, 3] = -M[1, 4]
    M[3, 2] = M[2, 3]

    M[1, 2] = d - c
    M[2, 1] = -M[1, 2]
    M[3, 4] = M[1, 2]
    M[4, 3] = -M[3, 4]

    M[1, 3] = c + d
    M[3, 1] = -M[1, 3]
    M[2, 4] = M[1, 3]
    M[4, 2] = -M[2, 4]
end

function dexp_system12!(M2_r::Base.RefValue{Matrix{Float64}}, a::Float64, b::Float64)
    M = M2_r[]
    M[1, 1] = a
    M[1, 2] = b
    M[2, 1] = -b
    M[2, 2] = a
end

function dexp_action_22!(Y_r::Base.RefValue{Matrix{Float64}}, X_r::Base.RefValue{Matrix{Float64}}, i::Int, j::Int, 
    a::Float64, b::Float64, c::Float64, d::Float64, 
    M4_r::Base.RefValue{Matrix{Float64}}, x4_r::Base.RefValue{Vector{Float64}}, y4_r::Base.RefValue{Vector{Float64}})

    Y::Matrix{Float64} = Y_r[]
    X::Matrix{Float64} = X_r[]
    M4::Matrix{Float64} = M4_r[]
    x4::Vector{Float64} = x4_r[]
    y4::Vector{Float64} = y4_r[]
    

    # form M
    dexp_system22!(M4_r, a, b, c, d)

    ind_row::Int = 2 * i - 1
    ind_col::Int = 2 * j - 1
    x4[1] = X[ind_row, ind_col]
    x4[2] = X[ind_row, ind_col+1]
    x4[3] = X[ind_row+1, ind_col]
    x4[4] = X[ind_row+1, ind_col+1]
    mul!(y4, M4, x4)
    # y4 <- M4 * x4
    y4 .*= 0.5
    # display(M4)
    # println(x4)
    # println(y4)
    
    Y[ind_row, ind_col] = y4[1]
    Y[ind_row, ind_col+1] = y4[2]
    Y[ind_row+1, ind_col] = y4[3]
    Y[ind_row+1, ind_col+1] = y4[4]
    # println(x4)
    # println(y4)
    # println([Y[ind_row, ind_col], Y[ind_row, ind_col+1], Y[ind_row+1, ind_col], Y[ind_row+1, ind_col+1]], "\n")
end

function dexp_action_12!(Y_r::Base.RefValue{Matrix{Float64}}, X_r::Base.RefValue{Matrix{Float64}}, n::Int, j::Int, 
    a::Float64, b::Float64,
    M2_r::Base.RefValue{Matrix{Float64}}, x2_r::Base.RefValue{Vector{Float64}}, y2_r::Base.RefValue{Vector{Float64}})
    Y::Matrix{Float64} = Y_r[]
    X::Matrix{Float64} = X_r[]
    x2::Vector{Float64} = x2_r[]
    y2::Vector{Float64} = y2_r[]

    dexp_system12!(M2_r, a, b)

    M2 = M2_r[]

    ind_col::Int = 2 * j - 1
    x2[1] = X[n, ind_col]
    x2[2] = X[n, ind_col+1]

    mul!(y2, M2, x2)

    Y[n, ind_col] = y2[1]
    Y[n, ind_col+1] = y2[2]
end

function dexp_transpose_action_22(S, para)
    a, b, c, d = para
    c = -c
    d = -d
    Ans = zeros(2, 2)
    M = zeros(4, 4)
    # M[1, 1] = a + b; M[1, 2] = -d- c; M[1, 3] = c - d; M[1, 4] = a - b;
    # M[2, 1] = c + d; M[2, 2] = a + b; M[2, 3] = b - a; M[2, 4] = c - d;
    # M[3, 1] = d - c; M[3, 2] = b - a; M[3, 3] = a + b; M[3, 4] = -d- c;
    # M[4, 1] = a - b; M[4, 2] = d - c; M[4, 3] = c + d; M[4, 4] = a + b;
    M[1, 1] = a + b
    M[1, 2] = d - c
    M[1, 3] = c + d
    M[1, 4] = a - b
    M[2, 1] = c - d
    M[2, 2] = a + b
    M[2, 3] = b - a
    M[2, 4] = c + d
    M[3, 1] = -c - d
    M[3, 2] = b - a
    M[3, 3] = a + b
    M[3, 4] = d - c
    M[4, 1] = a - b
    M[4, 2] = -c - d
    M[4, 3] = c - d
    M[4, 4] = a + b
    x = zeros(4)
    x[1] = S[1, 1]
    x[2] = S[1, 2]
    x[3] = S[2, 1]
    x[4] = S[2, 2]
    x = M * x
    x .*= 0.5
    Ans[1, 1] = x[1]
    Ans[1, 2] = x[2]
    Ans[2, 1] = x[3]
    Ans[2, 2] = x[4]

    # if norm(x) > 1e10
    #     println(para)
    #     display(M)
    # end
    return Ans
end

function dexp_transpose_action_12(S, para)
    a, b = para
    b = -b
    Ans = zeros(2)
    M = zeros(2, 2)
    M[1, 1] = a
    M[1, 2] = b
    M[2, 1] = -b
    M[2, 2] = a
    x = zeros(2)
    x .= S
    x = M * x
    return x
end

function compute_dexp_para(S; inv::Bool=false)
    n, = size(S)
    n_b = div(n, 2)
    S_schur = real_schur_s(S)
    P = S_schur.vectors
    A = zeros(n_b)
    A[1:length(S_schur.angles)] .= S_schur.angles
    para1 = Matrix{Float64}(undef, div(n_b * (n_b + 1), 2), 4)
    para2 = nothing
    if inv
        ind = 1
        for ii = 1:n_b
            for jj = 1:ii
                para1[ind, :] = dexp_inv_para(A[ii], A[jj])
                ind += 1
            end
        end
        if n != 2 * n_b
            para2 = Matrix{Float64}(undef, n_b, 2)
            for ii = 1:n_b
                para2[ii, :] = dexp_inv_para(A[ii])
            end
        end
    else
        ind = 1
        for ii = 1:n_b
            for jj = 1:ii
                para1[ind, :] = dexp_para(A[ii], A[jj])
                ind += 1
            end
        end
        if n != 2 * n_b
            para2 = Matrix{Float64}(undef, n_b, 2)
            for ii = 1:n_b
                para2[ii, :] = dexp_para(A[ii])
            end
        end
    end
    return P, para1, para2
end

function compute_dexp_para_all(S::Matrix{T}) where {T}
    n, = size(S)
    n_b = div(n, 2)
    P = nothing
    S_schur = real_schur_s(S)
    P = S_schur.vectors
    A = zeros(n_b)
    A[1:length(S_schur.angles)] .= S_schur.angles
    para1 = Matrix{Float64}(undef, div(n_b * (n_b + 1), 2), 4)
    parainv1 = Matrix{Float64}(undef, div(n_b * (n_b + 1), 2), 4)
    para2 = nothing
    parainv2 = nothing
    ind = 1
    for ii = 1:n_b
        for jj = 1:ii
            para1[ind, :] = dexp_para(A[ii], A[jj])

            den1 = para1[ind, 1]^2 + para1[ind, 3]^2
            den2 = para1[ind, 2]^2 + para1[ind, 4]^2

            parainv1[ind, 1] = para1[ind, 1] / den1
            parainv1[ind, 2] = para1[ind, 2] / den2
            parainv1[ind, 3] = -para1[ind, 3] / den1
            parainv1[ind, 4] = -para1[ind, 4] / den2
            # if norm(para1[ind, :]) > 1e10
            #     print("para1:\t")
            #     println(para1[ind, :])
            # end
            # if norm(parainv1[ind, :]) > 1e10
            #     println("parainv1:")
            #     println(para1[ind, :])
            #     println(parainv1[ind, :])
            #     println(den1, "\t", den2, "\t", A[ii], "\t", A[jj])
            # end
            ind += 1
        end
    end
    if n != 2 * n_b
        para2 = Matrix{Float64}(undef, n_b, 2)
        parainv2 = Matrix{Float64}(undef, n_b, 2)
        for ii = 1:n_b
            para2[ii, :] = dexp_para(A[ii])

            den = para2[ii, 1]^2 + para2[ii, 2]^2

            parainv2[ii, 1] = para2[ii, 1] / den
            parainv2[ii, 2] = -para2[ii, 2] / den
        end
    end
    return P, para1, para2, parainv1, parainv2
end

function compute_dexp_para_all(A::Vector{T}) where {T}
    n_b = length(A)
    para1 = Matrix{Float64}(undef, div(n_b * (n_b + 1), 2), 4)
    parainv1 = Matrix{Float64}(undef, div(n_b * (n_b + 1), 2), 4)
    para2 = Matrix{Float64}(undef, n_b, 2)
    parainv2 = Matrix{Float64}(undef, n_b, 2)
    den1::Float64 = 0.0
    den2::Float64 = 0.0

    ind = 1
    for ii = 1:n_b
        for jj = 1:ii
            para1[ind, :] .= dexp_para(A[ii], A[jj])

            den1 = para1[ind, 1]^2 + para1[ind, 3]^2
            den2 = para1[ind, 2]^2 + para1[ind, 4]^2

            parainv1[ind, 1] = para1[ind, 1] / den1
            parainv1[ind, 2] = para1[ind, 2] / den2
            parainv1[ind, 3] = -para1[ind, 3] / den1
            parainv1[ind, 4] = -para1[ind, 4] / den2
            # if norm(para1[ind, :]) > 1e10
            #     display(para1[ind, :])
            # end
            # if norm(parainv1[ind, :]) > 1e10
            #     display(parainv1[ind, :])
            # end
            ind += 1
        end
    end
    para2 = Matrix{Float64}(undef, n_b, 2)
    parainv2 = Matrix{Float64}(undef, n_b, 2)
    for ii = 1:n_b
        para2[ii, :] = dexp_para(A[ii])
        den = para2[ii, 1]^2 + para2[ii, 2]^2

        parainv2[ii, 1] = para2[ii, 1] / den
        parainv2[ii, 2] = -para2[ii, 2] / den
    end
    ###################################################
    # ind = 1;
    # for ii = 1:n_b;
    #     for jj = 1:ii
    #         para1[ind, :] = dexp_para(A[ii], A[jj]);

    #         den1 = para1[ind, 1] ^ 2 + para1[ind, 3] ^ 2;
    #         den2 = para1[ind, 2] ^ 2 + para1[ind, 4] ^ 2;

    #         parainv1[ind, 1] = para1[ind, 1] / den1;
    #         parainv1[ind, 2] = para1[ind, 2] / den2;
    #         parainv1[ind, 3] = -para1[ind, 3] / den1;
    #         parainv1[ind, 4] = -para1[ind, 4] / den2;
    #         ind += 1;
    #     end
    # end
    # if n != 2 * n_b
    #     para2 = Matrix{Float64}(undef, n_b, 2);
    #     parainv2 = Matrix{Float64}(undef, n_b, 2);
    #     for ii = 1:n_b
    #         para2[ii, :] =  dexp_para(A[ii]);

    #         den = para2[ii, 1] ^ 2 + para2[ii, 2] ^ 2;

    #         parainv2[ii, 1] = para2[ii, 1] / den;
    #         parainv2[ii, 2] = -para2[ii, 2] / den;
    #     end
    # end
    ###################################################
    return para1, para2, parainv1, parainv2
end

function compute_dexp_para_all!(S::Matrix{T}, P_ref, p1_ref, p2_ref, p1inv_ref, p2inv_ref) where {T}
    n, = size(S)
    n_b = div(n, 2)

    A = zeros(n_b)

    real_schur_s!(S, P_ref, Ref(A))

    # para1 = Matrix{Float64}(undef, div(n_b * (n_b + 1), 2), 4);
    # parainv1 = Matrix{Float64}(undef, div(n_b * (n_b + 1), 2), 4);
    # para2 = nothing;
    # parainv2 = nothing;

    para1 = p1_ref[]
    para2 = p2_ref[]
    parainv1 = p1inv_ref[]
    parainv2 = p2inv_ref[]

    ind = 1
    for ii = 1:n_b
        for jj = 1:ii
            para1[ind, :] .= dexp_para(A[ii], A[jj])

            den1 = para1[ind, 1]^2 + para1[ind, 3]^2
            den2 = para1[ind, 2]^2 + para1[ind, 4]^2

            parainv1[ind, 1] = para1[ind, 1] / den1
            parainv1[ind, 2] = para1[ind, 2] / den2
            parainv1[ind, 3] = -para1[ind, 3] / den1
            parainv1[ind, 4] = -para1[ind, 4] / den2
            ind += 1
        end
    end
    if n != 2 * n_b
        for ii = 1:n_b
            para2[ii, :] .= dexp_para(A[ii])

            den = para2[ii, 1]^2 + para2[ii, 2]^2

            parainv2[ii, 1] = para2[ii, 1] / den
            parainv2[ii, 2] = -para2[ii, 2] / den
        end
    end
end

function compute_dexp_para_all!(A::Vector{Float64}, p1_ref::Base.RefValue{Matrix{Float64}}, p2_ref::Base.RefValue{Matrix{Float64}}, 
    p1inv_ref::Base.RefValue{Matrix{Float64}}, p2inv_ref::Base.RefValue{Matrix{Float64}}) 
    n_b::Int = length(A)
    # para1 = Matrix{Float64}(undef, div(n_b*(n_b + 1), 2), 4);
    # parainv1 = Matrix{Float64}(undef, div(n_b*(n_b + 1), 2), 4);
    # para2 = Matrix{Float64}(undef, n_b, 2);
    # parainv2 = Matrix{Float64}(undef, n_b, 2);

    para1::Matrix{Float64} = p1_ref[]
    para2::Matrix{Float64} = p2_ref[]
    parainv1::Matrix{Float64} = p1inv_ref[]
    parainv2::Matrix{Float64} = p2inv_ref[]

    den::Float64 = 0.0
    den1::Float64 = 0.0
    den2::Float64 = 0.0


    ind::Int = 1
    for ii = 1:n_b
        for jj = 1:ii
            # para1[ind, :] .= dexp_para(A[ii], A[jj])
            dexp_para!(A[ii], A[jj], p1_ref, ind)

            den1 = para1[ind, 1]^2 + para1[ind, 3]^2
            den2 = para1[ind, 2]^2 + para1[ind, 4]^2

            parainv1[ind, 1] = para1[ind, 1] / den1
            parainv1[ind, 2] = para1[ind, 2] / den2
            parainv1[ind, 3] = -para1[ind, 3] / den1
            parainv1[ind, 4] = -para1[ind, 4] / den2
            ind += 1
        end
    end
    for ii = 1:n_b
        # para2[ii, :] .= dexp_para(A[ii])
        dexp_para!(A[ii], p2_ref, ii)
        den = para2[ii, 1]^2 + para2[ii, 2]^2

        parainv2[ii, 1] = para2[ii, 1] / den
        parainv2[ii, 2] = -para2[ii, 2] / den
    end
    ###################################################
    # ind = 1;
    # for ii = 1:n_b;
    #     for jj = 1:ii
    #         para1[ind, :] = dexp_para(A[ii], A[jj]);

    #         den1 = para1[ind, 1] ^ 2 + para1[ind, 3] ^ 2;
    #         den2 = para1[ind, 2] ^ 2 + para1[ind, 4] ^ 2;

    #         parainv1[ind, 1] = para1[ind, 1] / den1;
    #         parainv1[ind, 2] = para1[ind, 2] / den2;
    #         parainv1[ind, 3] = -para1[ind, 3] / den1;
    #         parainv1[ind, 4] = -para1[ind, 4] / den2;
    #         ind += 1;
    #     end
    # end
    # if n != 2 * n_b
    #     para2 = Matrix{Float64}(undef, n_b, 2);
    #     parainv2 = Matrix{Float64}(undef, n_b, 2);
    #     for ii = 1:n_b
    #         para2[ii, :] =  dexp_para(A[ii]);

    #         den = para2[ii, 1] ^ 2 + para2[ii, 2] ^ 2;

    #         parainv2[ii, 1] = para2[ii, 1] / den;
    #         parainv2[ii, 2] = -para2[ii, 2] / den;
    #     end
    # end
    ###################################################
    # return para1, para2, parainv1, parainv2;
end

function dexp_block_update(X, para1, para2)
    n, = size(X)
end

function dexp_real(S, Δ; P=nothing, para1=nothing, para2=nothing, transformed=false)
    # Compute X in d/dt exp(S + tΔ)|_{t=0} = exp(S)X
    n, = size(S)
    n_b = div(n, 2)

    if isnothing(P)
        P, para1, para2 = compute_dexp_para(S)
    end

    Ans = Matrix{Float64}(undef, n, n)
    if transformed
        Ans .= Δ
    else
        Ans .= P' * Δ * P
    end

    X22 = zeros(2, 2)
    X12 = zeros(1, 2)

    ind = 1
    for ii = 1:n_b
        for jj = 1:ii
            X22 .= Ans[(2*ii-1):(2*ii), (2*jj-1):(2*jj)]
            Ans[(2*ii-1):(2*ii), (2*jj-1):(2*jj)] .= dexp_action_22(X22, para1[ind, :])
            ind += 1
        end
    end

    if n != 2 * n_b
        for jj = 1:n_b
            X12 = Ans[n, (2*jj-1):(2*jj)]
            Ans[n, (2*jj-1):(2*jj)] .= dexp_action_12(X12, para2[jj, :])
        end
    end

    for ii = 1:n
        for jj = (ii+1):n
            Ans[ii, jj] = -Ans[jj, ii]
        end
    end

    Ans = P * Ans * P'

    return Ans
end

function dexp_real(P, para1, para2, Δ; transformed=false)
    # Compute X in d/dt exp(S + tΔ)|_{t=0} = exp(S)X
    n, = size(Δ)
    n_b = div(n, 2)

    Ans = Matrix{Float64}(undef, n, n)
    if transformed
        Ans .= Δ
    else
        Ans .= P' * Δ * P
    end

    X22 = zeros(2, 2)
    X12 = zeros(1, 2)

    ind = 1
    for ii = 1:n_b
        for jj = 1:ii
            X22 .= Ans[(2*ii-1):(2*ii), (2*jj-1):(2*jj)]
            Ans[(2*ii-1):(2*ii), (2*jj-1):(2*jj)] .= dexp_action_22(X22, para1[ind, :])
            ind += 1
        end
    end

    if n != 2 * n_b
        for jj = 1:n_b
            X12 = Ans[n, (2*jj-1):(2*jj)]
            Ans[n, (2*jj-1):(2*jj)] .= dexp_action_12(X12, para2[jj, :])
        end
    end

    for ii = 1:n
        for jj = (ii+1):n
            Ans[ii, jj] = -Ans[jj, ii]
        end
    end

    Ans = P * Ans * P'

    return Ans
end

# function dexp_real!(P_ref, para1_ref, para2_ref, Δ_ref, Ans_ref; transformed=false)
#     # Compute X in d/dt exp(S + tΔ)|_{t=0} = exp(S)X
#     P = P_ref[]
#     para1 = para1_ref[]
#     para2 = para2_ref[]
#     Δ = Δ_ref[]
#     n, = size(Δ)
#     n_b = div(n, 2)

#     X = Matrix{Float64}(undef, n, n)
#     Ans = Ans_ref[]

#     if transformed
#         X .= Δ
#     else
#         X .= P' * Δ * P
#         # PSPT!(P', Δ, X_ref);
#     end

#     X22 = zeros(2, 2)
#     X12 = zeros(1, 2)

#     ind = 1
#     for ii = 1:n_b
#         for jj = 1:ii
#             X22 .= X[(2*ii-1):(2*ii), (2*jj-1):(2*jj)]
#             X[(2*ii-1):(2*ii), (2*jj-1):(2*jj)] .= dexp_action_22(X22, para1[ind, :])
#             ind += 1
#         end
#     end

#     if n != 2 * n_b
#         for jj = 1:n_b
#             X12 = X[n, (2*jj-1):(2*jj)]
#             X[n, (2*jj-1):(2*jj)] .= dexp_action_12(X12, para2[jj, :])
#         end
#     end

#     for ii = 1:n
#         for jj = (ii+1):n
#             X[ii, jj] = -X[jj, ii]
#         end
#     end

#     Ans .= P * X * P'
#     # PSPT!(P, X, Ans_ref);
# end

function dexp_real!(P_ref, para1_ref, para2_ref, Δ_ref, Ans_ref; transformed=false, Workspace=nothing, Sys44=nothing, Sys22=nothing)
    # Compute X in d/dt exp(S + tΔ)|_{t=0} = exp(S)X
    P = P_ref[]
    para1 = para1_ref[]
    para2 = para2_ref[]
    Δ = Δ_ref[]
    n, = size(Δ)
    n_b = div(n, 2)

    if !isnothing(Workspace)
        X = Workspace[]
    else
        X = zeros(n, n)
        Workspace = Ref(X)
    end
    Ans = Ans_ref[]

    if isnothing(Sys44)
        Sys44 = Ref(zeros(4, 4))
    end

    if isnothing(Sys22)
        Sys22 = Ref(zeros(2, 2))
    end

    if transformed
        X .= Δ
    else
        X .= P' * Δ * P
        # PSPT!(P', Δ, X_ref);
    end

    X22 = zeros(2, 2)
    X12 = zeros(1, 2)

    x22 = zeros(4)
    x12 = zeros(2)

    x22_ref = Ref(x22)
    x12_ref = Ref(x12)

    ind = 1
    for ii = 1:n_b
        for jj = 1:ii
            X22 .= X[(2*ii-1):(2*ii), (2*jj-1):(2*jj)]
            X[(2*ii-1):(2*ii), (2*jj-1):(2*jj)] .= dexp_action_22(X22, para1[ind, :]; Workspace=Sys44)
            # dexp_action_22!(Workspace, para1_ref, Workspace, ii, jj, ind, Sys44, x22_ref)
            ind += 1
        end
    end

    if n != 2 * n_b
        for jj = 1:n_b
            X12 = X[n, (2*jj-1):(2*jj)]
            X[n, (2*jj-1):(2*jj)] .= dexp_action_12(X12, para2[jj, :]; Workspace=Sys22)
            # dexp_action_12!(Workspace, para2_ref, Workspace, n, jj, jj, Sys22, x12_ref)

        end
    end

    # display(X)

    for ii = 1:n
        for jj = (ii+1):n
            X[ii, jj] = -X[jj, ii]
        end
    end

    # display(X)

    Ans .= P * X * P'

    # display(Ans)
    # PSPT!(P, X, Ans_ref);
end


function dexp_real!(Y_r::Base.RefValue{Matrix{Float64}}, X_r::Base.RefValue{Matrix{Float64}}, P_r::Base.RefValue{Matrix{Float64}}, 
    para1_r::Base.RefValue{Matrix{Float64}}, para2_r::Base.RefValue{Matrix{Float64}}, wsp_dexp::WSP; transformed=false)
    # Compute Δ or X in d/dt exp(S + tΔ)|_{t=0} = exp(S)X
    # wsp_dexp carries n x n real matrices Mn_1, Mn_2, 4 x 4 real matrix M4, 2 x 2 real matrix M2
    Y::Matrix{Float64} = Y_r[]
    P::Matrix{Float64} = P_r[]
    X::Matrix{Float64} = X_r[]
    p1::Matrix{Float64} = para1_r[]
    p2::Matrix{Float64} = para2_r[]

    n::Int, = size(X)
    n_b::Int = div(n, 2)

    Mn_1::Matrix{Float64} = retrieve(wsp_dexp, 1)
    Mn_2::Matrix{Float64} = retrieve(wsp_dexp, 2)
    M4::Matrix{Float64} = retrieve(wsp_dexp, 3)
    x4::Vector{Float64} = retrieve(wsp_dexp, 4)
    y4::Vector{Float64} = retrieve(wsp_dexp, 5)
    M2::Matrix{Float64} = retrieve(wsp_dexp, 6)
    x2::Vector{Float64} = retrieve(wsp_dexp, 7)
    y2::Vector{Float64} = retrieve(wsp_dexp, 8)


    Mn_1_r = wsp_dexp.vec[1]
    Mn_2_r = wsp_dexp.vec[2]
    M4_r = wsp_dexp.vec[3]
    x4_r = wsp_dexp.vec[4]
    y4_r = wsp_dexp.vec[5]
    M2_r = wsp_dexp.vec[6]
    x2_r = wsp_dexp.vec[7]
    y2_r = wsp_dexp.vec[8]

    a::Float64 = 0.0
    b::Float64 = 0.0
    c::Float64 = 0.0
    d::Float64 = 0.0


    if transformed
        for ii = 1:n
            for jj = 1:n
                Mn_1[ii, jj] = X[ii, jj]
            end
        end
    else
        for ii = 1:n
            for jj = 1:n
                Mn_1[ii, jj] = X[ii, jj]
            end
        end
        mul!(Mn_2, P', Mn_1)
        mul!(Mn_1, Mn_2, P)
    end

    # display(Mn_1)

    alloc_dexp_22::Int = 0

    ind::Int = 1
    for ii = 1:n_b
        for jj = 1:ii
            # X22 .= X[(2*ii-1):(2*ii), (2*jj-1):(2*jj)]
            # X[(2*ii-1):(2*ii), (2*jj-1):(2*jj)] .= dexp_action_22(X22, para1[ind, :]; Workspace=Sys44)
            a = p1[ind, 1]
            b = p1[ind, 2]
            c = p1[ind, 3]
            d = p1[ind, 4]
            dexp_action_22!(Mn_2_r, Mn_1_r, ii, jj, a, b, c, d, M4_r, x4_r, y4_r)

            # stat = @timed dexp_action_22!(Mn_2_r, Mn_1_r, ii, jj, a, b, c, d, M4_r, x4_r, y4_r)
            # alloc_dexp_22 += stat.bytes

            # dexp_action_22!(Workspace, para1_ref, Workspace, ii, jj, ind, Sys44, x22_ref)
            ind += 1
        end
    end

    if n != 2 * n_b
        for jj = 1:n_b
            # X12 = X[n, (2*jj-1):(2*jj)]
            # X[n, (2*jj-1):(2*jj)] .= dexp_action_12(X12, para2[jj, :]; Workspace=Sys22)
            a = p2[jj, 1]
            b = p2[jj, 2]
            dexp_action_12!(Mn_2_r, Mn_1_r, n, jj, a, b, M2_r, x2_r, y2_r)
        end
    end

    # display(Mn_2)


    for ii = 1:n
        for jj = (ii+1):n
            Mn_2[ii, jj] = -Mn_2[jj, ii]
        end
        Mn_2[ii, ii] = 0.0;
    end


    mul!(Mn_1, P, Mn_2)
    mul!(Y, Mn_1, P')

    for jj = 1:n
        for ii = (jj+1):n
            Y[ii, jj] = (Y[ii, jj] - Y[jj, ii]) / 2;
            Y[jj, ii] = -Y[ii, jj];
        end
        Y[jj, jj] = 0.0;
    end

    # println("\t\t Allocation in the 22 action of dexp:\t", alloc_dexp_22)


end

function dexp_real_transpose(P, para1, para2, Δ; transformed=false)
    # Compute X in d/dt exp(S + tΔ)|_{t=0} = exp(S)X
    n, = size(Δ)
    n_b = div(n, 2)

    Ans = Matrix{Float64}(undef, n, n)
    if transformed
        Ans .= Δ
    else
        Ans .= P' * Δ * P
    end

    X22 = zeros(2, 2)
    X12 = zeros(1, 2)

    ind = 1
    for ii = 1:n_b
        for jj = 1:ii
            X22 .= Ans[(2*ii-1):(2*ii), (2*jj-1):(2*jj)]
            Ans[(2*ii-1):(2*ii), (2*jj-1):(2*jj)] .= dexp_transpose_action_22(X22, para1[ind, :])
            ind += 1
        end
    end

    if n != 2 * n_b
        for jj = 1:n_b
            X12 = Ans[n, (2*jj-1):(2*jj)]
            Ans[n, (2*jj-1):(2*jj)] .= dexp_transpose_action_12(X12, para2[jj, :])
        end
    end

    for ii = 1:n
        for jj = (ii+1):n
            Ans[ii, jj] = -Ans[jj, ii]
        end
    end

    Ans = P * Ans * P'

    return Ans
end

function dexp_real_transpose_inv(P, para1, para2, Δ; transformed=false)
    # Compute X in d/dt exp(S + tΔ)|_{t=0} = exp(S)X
    n, = size(Δ)
    n_b = div(n, 2)

    Ans = Matrix{Float64}(undef, n, n)
    if transformed
        Ans .= Δ
    else
        Ans .= P' * Δ * P
    end

    X22 = zeros(2, 2)
    X12 = zeros(1, 2)

    ind = 1
    for ii = 1:n_b
        for jj = 1:ii
            X22 .= Ans[(2*ii-1):(2*ii), (2*jj-1):(2*jj)]
            Ans[(2*ii-1):(2*ii), (2*jj-1):(2*jj)] .= dexp_transpose_action_22(X22, para1[ind, :])
            ind += 1
        end
    end

    if n != 2 * n_b
        for jj = 1:n_b
            X12 = Ans[n, (2*jj-1):(2*jj)]
            Ans[n, (2*jj-1):(2*jj)] .= dexp_transpose_action_12(X12, para2[jj, :])
        end
    end

    for ii = 1:n
        for jj = (ii+1):n
            Ans[ii, jj] = -Ans[jj, ii]
        end
    end

    Ans = P * Ans * P'

    return Ans
end

function dexp_inv_real(S, X; P=nothing, para1=nothing, para2=nothing, transformed=false)
    # Compute Δ in d/dt exp(S + tΔ)|_{t=0} = exp(S)X
    n, = size(S)
    n_b = div(n, 2)

    if isnothing(P)
        P, para1, para2 = compute_dexp_para(S; inv=true)
    end

    Ans = Matrix{Float64}(undef, n, n)
    if transformed
        Ans .= X
    else
        Ans .= P' * X * P
    end

    X22 = zeros(2, 2)
    X12 = zeros(1, 2)

    ind = 1
    for ii = 1:n_b
        for jj = 1:ii
            X22 .= Ans[(2*ii-1):(2*ii), (2*jj-1):(2*jj)]
            Ans[(2*ii-1):(2*ii), (2*jj-1):(2*jj)] .= dexp_action_22(X22, para1[ind, :])
            ind += 1
        end
    end

    if n != 2 * n_b
        for jj = 1:n_b
            X12 = Ans[n, (2*jj-1):(2*jj)]
            Ans[n, (2*jj-1):(2*jj)] .= dexp_action_12(X12, para2[jj, :])
        end
    end

    for ii = 1:n
        for jj = (ii+1):n
            Ans[ii, jj] = -Ans[jj, ii]
        end
    end

    Ans = P * Ans * P'

    return Ans
end

function PTXP_22_zero(P, X, k)
    n, = size(P)
    Ans = Matrix{Float64}(undef, n, n)

    Ans[:, 1:k] .= P' * X[:, 1:k]
    Ans[:, (k+1):n] .= P[1:k, :]' * X[1:k, (k+1):n]

    Ans = Ans * P

    return Ans
end

function PTXP_22_zero!(P_ref, X_ref, k, Ans_ref)
    P = P_ref[]
    X = X_ref[]
    n, = size(P)
    Ans = Ans_ref[]

    Ans = Ans_ref[]
    temp = 0.0

    # for Sj = 1:k
    #     for Si = (Sj + 1):n
    #         for Xi = 1:n
    #             for Xj = 1:(Xi - 1) 
    #                 # temp = S[Si, Sj] * (P[Xi, Si] * P[Xj, Sj] - P[Xj, Si] * P[Xi, Sj])
    #                 temp = S[Si, Sj] * (P[Si, Xi] * P[Sj, Xj] - P[Si, Xj] * P[Sj, Xi])
    #                 Ans[Xi, Xj] += temp;
    #                 Ans[Xj, Xi] -= temp;
    #             end
    #         end
    #     end
    # end

    for ii = 1:n
        for jj = 1:k
            Ans[ii, jj] = X[ii, jj]
            Ans[jj, ii] = -X[ii, jj]
        end
        Ans[ii, ii] = 0
    end
    Ans .= P' * Ans * P

    # Ans[:, 1:k] .= P' * X[:, 1:k]
    # Ans[:, (k+1):n] .= P[1:k, :]' * X[1:k, (k+1):n]

    # Ans .= Ans * P

    # return Ans;
end

function PTXP_22_zero!(Y_r::Base.RefValue{Matrix{Float64}}, P_r::Base.RefValue{Matrix{Float64}}, X_r::Base.RefValue{Matrix{Float64}}, 
    k::Int, wsp_PTXP::WSP)

    # wsp_PTXP carries m x m real matrix Mm, m x n real matrix Mmn, n x m real matrices Mnm_1, Mnm_2
    # n x n real matrix Mn, n x k real matrices Mnk_1, Mnk_2, k x m real matrix Mkm
    
    Y::Matrix{Float64} = Y_r[]
    P::Matrix{Float64} = P_r[]
    X::Matrix{Float64} = X_r[]

    Mnm::Matrix{Float64} = retrieve(wsp_PTXP, 4)
    Mn::Matrix{Float64} = retrieve(wsp_PTXP, 5)
    Mnk_1::Matrix{Float64} = retrieve(wsp_PTXP, 6)
    Mnk_2::Matrix{Float64} = retrieve(wsp_PTXP, 7)
    Mkm::Matrix{Float64} = retrieve(wsp_PTXP, 8)

    n::Int, = size(X)
    m::Int = n - k

    

    for ii = 1:n
        for jj = 1:n
            Mn[ii, jj] = P[jj, ii]         # Mn gets P'
        end
    end

    for ii = 1:n
        for jj = 1:k
            Mnk_1[ii, jj] = X[ii, jj]      # Mnk_1 gets AB
        end
    end

    for ii = 1:k
        for jj = 1:m
            Mkm[ii, jj] = X[ii, jj+k]    # Mkm gets -B'
        end
    end


    mul!(Mnk_2, Mn, Mnk_1)                 # Mnk_2 gets P' * [A//B]
    for ii = 1:n
        for jj = 1:k
            Mnk_1[ii, jj] = Mn[ii, jj]     # Mnk_1 gets the first k columns of P' in Mn
            Mn[ii, jj] = Mnk_2[ii, jj]     # Then the first k columns of Mn gets Mnk_2 and frees Mnk_2
        end
    end

    mul!(Mnm, Mnk_1, Mkm)
    for ii = 1:n
        for jj = 1:m
            Mn[ii, jj+k] = Mnm[ii, jj]   # Entire Mn gets updated
        end
    end
    mul!(Y, Mn, P)

    for jj = 1:n
        for ii = (jj + 1):n
            Y[ii, jj] = (Y[ii, jj] - Y[jj, ii]) / 2;
            Y[jj, ii] = -Y[ii, jj];
        end
        Y[jj, jj] = 0.0;
    end
    
end

function PTXP_only_22_nonzero(P, X, k)
    n, = size(P)
    Ans = Matrix{Float64}(undef, n, n)

    Ans = P[(k+1):n, :]' * X[(k+1):n, (k+1):n] * P[(k+1):n, :]

    return Ans
end

function PTXP_only_22_nonzero!(P_ref, X_ref, k, Ans_ref)
    P = P_ref[]
    X = X_ref[]
    n, = size(P)

    Ans = Ans_ref[]
    # Ans = zeros(n, n)
    temp = 0.0

    Ans .= P[(k+1):n, :]' * X[(k+1):n, (k+1):n] * P[(k+1):n, :]
end

function PTXP_only_22_nonzero!(Y_r::Base.RefValue{Matrix{Float64}}, P_r::Base.RefValue{Matrix{Float64}}, X_r::Base.RefValue{Matrix{Float64}},
    k::Int, wsp_PTXP::WSP)
    # wsp_PTXP carries m x m real matrix Mm, m x n real matrix Mmn, n x m real matrices Mnm_1, Mnm_2
    # n x n real matrix Mn, n x k real matrices Mnk_1, Mnk_2, k x m real matrix Mkm
    P::Matrix{Float64} = P_r[]
    X::Matrix{Float64} = X_r[]
    Y::Matrix{Float64} = Y_r[]
    n::Int, = size(P)
    m::Int = n - k

    Mm::Matrix{Float64} = retrieve(wsp_PTXP, 1)
    Mmn::Matrix{Float64} = retrieve(wsp_PTXP, 2)
    Mnm_1::Matrix{Float64} = retrieve(wsp_PTXP, 3)
    Mnm_2::Matrix{Float64} = retrieve(wsp_PTXP, 4)

    for ii = 1:m
        for jj = 1:n
            Mmn[ii, jj] = P[ii+k, jj]
            Mnm_1[jj, ii] = Mmn[ii, jj]
        end
    end

    for ii = 1:m
        for jj = 1:m
            Mm[ii, jj] = X[ii+k, jj+k]
        end
    end

    mul!(Mnm_2, Mnm_1, Mm)
    mul!(Y, Mnm_2, Mmn)

    for jj = 1:n
        for ii = (jj + 1):n
            Y[ii, jj] = (Y[ii, jj] - Y[jj, ii]) / 2
            Y[jj, ii] = - Y[ii, jj];
        end
        Y[jj, jj] = 0.0;
    end
    # Ans .= P[(k+1):n, :]' * X[(k+1):n, (k+1):n] * P[(k+1):n, :]
end

# function PTXP_only_22_nonzero!(P, X, k, Ans_ref)
#     n, = size(P);

#     Ans = Ans_ref[];
#     # Ans = zeros(n, n)


#     # for Si = (k + 1):n
#     #     for Sj = (k + 1):(Si - 1)
#     #         for Xi = 1:n
#     #             for Xj = 1:(Xi - 1) 
#     #                 temp = S[Si, Sj] * (P[Si, Xi] * P[Sj, Xj] - P[Si, Xj] * P[Sj, Xi])
#     #                 Ans[Xi, Xj] += temp;
#     #                 Ans[Xj, Xi] -= temp;
#     #             end
#     #         end
#     #     end
#     # end
#     # return Ans;
#     Ans .= P[(k + 1):n, :]' * X[(k + 1):n, (k + 1):n] * P[(k + 1):n, :];
# end

function solve_log_St_gradient(S, Δ, k; MaxIter=100, AbsTol=1e-8)
    n, = size(S)
    sP, spara1, spara2, sparainv1, sparainv2 = compute_dexp_para_all(S)
    #sP, sparainv1, sparainv2 = compute_dexp_para(S, inv = true);

    Δ_iter = zeros(n, n)
    Δ_iter .= Δ

    X_iter = Matrix{Float64}(undef, n, n)
    Z_iter = zeros(n - k, n - k)

    abserr = 100
    iter = 1

    while iter < MaxIter && abserr > AbsTol
        PTCP_iter = PTXP_only_22_nonzero(sP, Δ_iter, k)

        X_iter .= dexp_real(S, PTCP_iter; P=sP, para1=spara1, para2=spara2, transformed=true)
        #X_iter .= dexp_real(S, PTCP_iter, transformed = true);
        Z_iter .+= X_iter[(k+1):n, (k+1):n]

        #display(X_iter .- dexp_real(S, PTCP_iter))

        PTXP_iter = PTXP_22_zero(sP, X_iter, k)
        Δ_iter .= dexp_real(S, PTXP_iter, P=sP, para1=sparainv1, para2=sparainv2, transformed=true)
        #Δ_iter .= dexp_inv_real(S, PTXP_iter, transformed = true);

        abserr = norm(Δ_iter[(k+1):n, (k+1):n], Inf)
        #println("Iteration:\t", iter, "\t,AbsErr:\t", abserr);
        iter += 1
    end

    temp = 0
    for ii = 1:(n-k)
        for jj = 1:ii
            temp = 0.5 * (Z_iter[ii, jj] - Z_iter[jj, ii])
            Z_iter[ii, jj] = temp
            Z_iter[jj, ii] = -temp
        end
        Z_iter[ii, ii] = 0
    end

    return Z_iter
end

function check_solver(n, k)
    S = rand(n, n)
    S .-= S'
    Δ = rand(n, n)
    Δ .-= Δ'

    Z = solve_log_St_gradient(S, Δ, k)
    Z_f = zeros(n, n)
    Z_f[(k+1):n, (k+1):n] .= Z

    display(Δ)
    display(dexp_inv_real(S, Z_f))
end

function solve_log_St_gradient(S, sP, sA, Δ, k; MaxIter=5, AbsTol=1e-2)
    n, = size(sP)
    spara1, spara2, sparainv1, sparainv2 = compute_dexp_para_all(sA)
    #sP, sparainv1, sparainv2 = compute_dexp_para(S, inv = true);

    Δ_iter = zeros(n, n)
    Δ_iter .= Δ

    X_iter = Matrix{Float64}(undef, n, n)
    Z_iter = zeros(n - k, n - k)

    abserr = 100
    iter = 1

    while iter < MaxIter && abserr > AbsTol
        PTCP_iter = PTXP_only_22_nonzero(sP, Δ_iter, k)

        X_iter .= dexp_real(S, PTCP_iter; P=sP, para1=spara1, para2=spara2, transformed=true)
        Z_iter .+= X_iter[(k+1):n, (k+1):n]

        PTXP_iter = PTXP_22_zero(sP, X_iter, k)
        Δ_iter .= dexp_real(S, PTXP_iter, P=sP, para1=sparainv1, para2=sparainv2, transformed=true)
        abserr = norm(Δ_iter[(k+1):n, (k+1):n], Inf)
        println("Iteration:\t", iter, "\t,AbsErr:\t", abserr)
        iter += 1
    end

    if abserr > 1
        throw("Error!")
    end

    return Z_iter
end

function grad_log_St_opt(S, k; MaxIter=5, AbsTol=1e-2, dexp_para=nothing)
    n, = size(S)

    if isnothing(dexp_para)
        dexp_para = compute_dexp_para_all(S)
    end

    Δ_iter = zeros(n, n)
    Δ_iter[(k+1):n, (k+1):n] .= S[(k+1):n, (k+1):n]

    X_iter = Matrix{Float64}(undef, n, n)
    Z_iter = zeros(n - k, n - k)

    abserr = 100
    iter = 1



    while iter < MaxIter && abserr > AbsTol
        PTCP_iter = PTXP_only_22_nonzero(dexp_para[1], Δ_iter, k)

        X_iter .= dexp_real(dexp_para[1], dexp_para[2], dexp_para[3], PTCP_iter; transformed=true)
        Z_iter .+= X_iter[(k+1):n, (k+1):n]

        PTXP_iter = PTXP_22_zero(dexp_para[1], X_iter, k)
        Δ_iter .= dexp_real(dexp_para[1], dexp_para[4], dexp_para[5], PTXP_iter; transformed=true)
        err_pre = abserr
        abserr = norm(Δ_iter[(k+1):n, (k+1):n], Inf)
        if abserr > err_pre
            break
        end
        #println("Iteration:\t", iter, "\t,AbsErr:\t", abserr);
        iter += 1
    end

    # if abserr > 1
    #     throw("Error!");
    # end

    d_msg(["Gradient Solver: Termination:\t", iter, "\t,AbsErr:\t", abserr, "\n"], true)

    # temp = 0;
    # for ii = 1:(n - k)
    #     for jj = 1:(ii - 1)
    #         temp = 0.5 * (Z_iter[ii, jj] - Z_iter[jj, ii]);
    #         Z_iter[ii, jj] = temp;
    #         Z_iter[jj, ii] = - temp;
    #     end
    #     Z_iter[ii, ii] = 0;
    # end

    return Z_iter
end



function grad_log_St_opt2(Z_r, S, k; dexp_para=nothing)
    n, = size(S)
    Z = Z_r[];

    if isnothing(dexp_para)
        dexp_para = compute_dexp_para_all(S)
    end

    P = dexp_para[1]
    slope::Float64 = 0.0

    PTCP = PTXP_only_22_nonzero(dexp_para[1], S, k)
    X = dexp_real_transpose(dexp_para[1], dexp_para[4], dexp_para[5], PTCP; transformed=true)
    for ii = 1:(n-k)
        for jj = 1:(ii - 1)
            Z[ii, jj] = X[k + ii, k + jj]
            Z[jj, ii] = - Z[ii, jj]
            slope += Z[ii, jj] ^ 2
        end
        Z[ii, ii] = 0.0;
    end

    slope = -slope
    return slope
end

function grad_log_St_opt2(Z_r, S, k, P_r, dexp_para_r)
    n, = size(S)
    Z = Z_r[];
    P = P_r[]

    p1inv = dexp_para_r[3][]
    p2inv = dexp_para_r[4][]


    PTCP = PTXP_only_22_nonzero(P, S, k)
    X = dexp_real_transpose(P, p1inv, p2inv, PTCP; transformed=true)
    for ii = 1:(n-k)
        for jj = 1:(ii - 1)
            Z[ii, jj] = X[k + ii, k + jj]
            Z[jj, ii] = - Z[ii, jj]
        end
        Z[ii, ii] = 0.0;
    end

    PTXP = P[(k + 1):n, :]' * Z * P[(k + 1):n, :];
    C = dexp_real(P, p1inv, p2inv, PTXP; transformed=true);
    slope = -inner_skew_22(X, C, k)
    return slope
end



function grad_log_St_opt_NMLS(S, k; MaxIter=5, AbsTol=1e-2, dexp_para_ref=nothing)
    n, = size(S)

    if isnothing(dexp_para_ref)
        dexp_para = compute_dexp_para_all(S)
    else
        dexp_para = dexp_para_ref[]
    end

    Δ_iter = zeros(n, n)
    Δ_iter[(k+1):n, (k+1):n] .= S[(k+1):n, (k+1):n]

    X_iter = Matrix{Float64}(undef, n, n)
    Z_iter = zeros(n - k, n - k)

    abserr = 100
    iter = 1

    flag_good_direction = true



    while iter < MaxIter && abserr > AbsTol
        PTCP_iter = PTXP_only_22_nonzero(dexp_para[1], Δ_iter, k)

        X_iter .= dexp_real(dexp_para[1], dexp_para[2], dexp_para[3], PTCP_iter; transformed=true)
        Z_iter .+= X_iter[(k+1):n, (k+1):n]

        PTXP_iter = PTXP_22_zero(dexp_para[1], X_iter, k)
        Δ_iter .= dexp_real(dexp_para[1], dexp_para[4], dexp_para[5], PTXP_iter; transformed=true)
        err_pre = abserr
        abserr = norm(Δ_iter[(k+1):n, (k+1):n], Inf)
        if abserr > err_pre
            break
            flag_good_direction = false
        end
        #println("Iteration:\t", iter, "\t,AbsErr:\t", abserr);
        iter += 1
    end

    if abserr > 0.1
        flag_good_direction = false
    end

    #d_msg(["Gradient Solver: Termination:\t", iter, "\t,AbsErr:\t", abserr, "\n"], true);

    temp = 0
    for ii = 1:(n-k)
        for jj = 1:(ii-1)
            temp = 0.5 * (Z_iter[ii, jj] - Z_iter[jj, ii])
            Z_iter[ii, jj] = temp
            Z_iter[jj, ii] = -temp
        end
        Z_iter[ii, ii] = 0
    end

    # display(flag_good_direction);

    return Z_iter, flag_good_direction
end

function grad_log_St_opt_NMLS!(S_ref, k, P_ref, p1_ref, p2_ref, p1inv_ref, p2inv_ref, Z_ref; MaxIter=5, AbsTol=1e-2)
    S = S_ref[]
    n, = size(S)

    Δ_iter = zeros(n, n)
    Δ_iter[(k+1):n, (k+1):n] .= S[(k+1):n, (k+1):n]

    P = P_ref[]

    X_iter = Matrix{Float64}(undef, n, n)
    PTCP_iter = Matrix{Float64}(undef, n, n)
    PTXP_iter = Matrix{Float64}(undef, n, n)

    Z_iter = Z_ref[]
    Z_iter .= 0.0

    PTCP_ref = Ref(PTCP_iter)
    PTXP_ref = Ref(PTXP_iter)
    X_ref = Ref(X_iter)
    Δ_ref = Ref(Δ_iter)

    p1 = p1_ref[]
    p2 = p2_ref[]
    p1inv = p1inv_ref[]
    p2inv = p2inv_ref[]

    abserr = 100
    iter = 1


    flag_good_direction = true



    while iter < MaxIter && abserr > AbsTol
        # PTCP_iter = PTXP_only_22_nonzero(dexp_para[1], Δ_iter, k);
        # PTXP_only_22_nonzero!(P, Δ_iter, k, PTCP_ref);
        # @time PTXP_only_22_nonzero!(P_ref, Δ_ref, k, PTCP_ref);
        PTXP_only_22_nonzero!(P_ref, Δ_ref, k, PTCP_ref)

        # PTCP_iter .= 0.0;
        # # PSPT!(P[(k + 1):n, :]', Δ_iter[(k + 1):n, (k + 1):n], PTCP_ref);
        # PTXP_22_zero!(P, Δ_iter, k, PTCP_ref);

        # X_iter .= dexp_real(dexp_para[1], dexp_para[2], dexp_para[3], PTCP_iter; transformed = true);
        # X_iter .= 0.0;

        # dexp_real!(P, p1, p2, PTCP_iter, X_ref; transformed = true);
        # @time dexp_real!(P_ref, p1_ref, p2_ref, PTCP_ref, X_ref; transformed = true);
        dexp_real!(P_ref, p1_ref, p2_ref, PTCP_ref, X_ref; transformed=true)
        Z_iter .+= X_iter[(k+1):n, (k+1):n]

        # PTXP_iter = PTXP_22_zero(dexp_para[1], X_iter, k);
        # PTXP_iter .= 0.0;
        # PTXP_22_zero!(P, X_iter, k, PTXP_ref);
        # @time PTXP_22_zero!(P_ref, X_ref, k, PTXP_ref);
        PTXP_22_zero!(P_ref, X_ref, k, PTXP_ref)



        # Δ_iter .= dexp_real(dexp_para[1], dexp_para[4], dexp_para[5], PTXP_iter; transformed = true);
        # Δ_iter .= 0.0;
        # dexp_real!(P, p1inv, p2inv, PTXP_iter, Δ_ref; transformed = true);
        # @time dexp_real!(P_ref, p1inv_ref, p2inv_ref, PTXP_ref, Δ_ref; transformed = true);
        dexp_real!(P_ref, p1inv_ref, p2inv_ref, PTXP_ref, Δ_ref; transformed=true)

        err_pre = abserr
        abserr = norm(Δ_iter[(k+1):n, (k+1):n], Inf)
        # d_msg(["\t\t\t", abserr, "\n"], true);
        if abserr > err_pre
            flag_good_direction = false
            break
        end
        # println("Iteration:\t", iter, "\t,AbsErr:\t", abserr);
        iter += 1
    end

    if abserr > 0.1
        flag_good_direction = false
    end

    temp = 0
    for ii = 1:(n-k)
        for jj = 1:(ii-1)
            temp = 0.5 * (Z_iter[ii, jj] - Z_iter[jj, ii])
            Z_iter[ii, jj] = temp
            Z_iter[jj, ii] = -temp
        end
        Z_iter[ii, ii] = 0
    end

    # d_msg(["\t\t\tGradient Solver: Termination:\t", iter, "\t,AbsErr:\t", abserr, "\t,Good direction:\t", flag_good_direction, "\n"], true);

    return flag_good_direction
end