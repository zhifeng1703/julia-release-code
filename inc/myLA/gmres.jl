include("./LA_KWARGS.jl")
include("./termination.jl")
include(homedir() * "/Documents/julia/inc/workspace.jl")

using LoopVectorization


function get_wsp_gmres(d::Int, rs::Int)
    # d for dimension, r for restart dimension
    v::Vector{Float64} = Vector{Float64}(undef, d)
    y::Vector{Float64} = Vector{Float64}(undef, d)
    r::Vector{Float64} = Vector{Float64}(undef, d)
    β::Vector{Float64} = Vector{Float64}(undef, rs + 1)
    c::Vector{Float64} = Vector{Float64}(undef, rs)
    s::Vector{Float64} = Vector{Float64}(undef, rs)

    Q::Matrix{Float64} = zeros(d, rs + 1)
    H::Matrix{Float64} = zeros(rs + 1, rs)

    Qdr::Matrix{Float64} = zeros(d, rs)
    Hr::Matrix{Float64} = zeros(rs, rs)

    return WSP(v, y, r, β, c, s, Q, H, Qdr, Hr)
end


function gmres_matfree!(F, x_r::Base.RefValue{Vector{Float64}}, b_r::Base.RefValue{Vector{Float64}}, 
    wsp_gmres::WSP, dim::Int, restart::Int, Stop;
    is_x_zeros::Bool=false, MaxRes::Int = 20, action_kwargs=nothing)
    # Argument list of the routine that implements the linear transformation must admits [y_ref, x_ref; kwargs = nothing]

    restart_cnt::Int = 0
    compute_time::Float64 = 0.0
    m::Int = restart

    # println("\t\t\tGMRES start! Dimension: $(dim)\t Restart: $(restart)")




    v::Vector{Float64} = retrieve(wsp_gmres, 1)
    y::Vector{Float64} = retrieve(wsp_gmres, 2)
    r::Vector{Float64} = retrieve(wsp_gmres, 3)
    β::Vector{Float64} = retrieve(wsp_gmres, 4)
    c::Vector{Float64} = retrieve(wsp_gmres, 5)
    s::Vector{Float64} = retrieve(wsp_gmres, 6)

    Q::Matrix{Float64} = retrieve(wsp_gmres, 7)
    H::Matrix{Float64} = retrieve(wsp_gmres, 8)

    Qdr::Matrix{Float64} = retrieve(wsp_gmres, 9)
    Hr::Matrix{Float64} = retrieve(wsp_gmres, 10)

    x::Vector{Float64} = x_r[];
    b::Vector{Float64} = b_r[];

    v_r = wsp_gmres.vec[1]
    y_r = wsp_gmres.vec[2]

    # x for solution, b for the target, y for the result of the linear action, v for preimage of the linear action
    # r for residual, β for the transfromed target, c, s for the cosine and sine values

    # display(x_r[]);
    # display(b_r[]);

    
    n_vec = length(r)
    @turbo for ii = 1:n_vec
        @inbounds r[ii] = b[ii]
    end
    # r .= b
    if !is_x_zeros
        F(y_r, x_r; kwargs=action_kwargs)
        @turbo for ii = 1:n_vec
            @inbounds r[ii] -= y[ii]
        end
        # r .-= y
    end

    b_norm::Float64 = norm(b)
    r_norm::Float64 = norm(r)

    elsize::Int = sizeof(eltype(Q))



    abserr::Float64 = r_norm
    relerr::Float64 = r_norm / b_norm
    err_per_run::Float64 = 10.0
    temp::Float64 = 0.0
    found::Bool = false

    ptrQ = pointer(Q)
    ptrQdr = pointer(Qdr)
    ptrH = pointer(H)
    ptrHr = pointer(Hr)

    ptry = pointer(y)
    ptrc = pointer(c)
    ptrb = pointer(b)
    ptrr = pointer(r)
    ptrβ = pointer(β)
    ptrv = pointer(v)
    ptrs = pointer(s)

    while !terminate(compute_time, restart_cnt, abserr, relerr, Stop)
        # println("\t\t\tAttemp $(restart_cnt + 1): Norm of initial residual: $(r_norm).")

        stat = @elapsed begin
            r_norm = norm(r)
            # v .= r
            # v ./= r_norm
            # Q[:, 1] .= v
            # β .= 0
            # for ii = 1:n_vec
            #     @inbounds v[ii] = r[ii] / r_norm
            #     @inbounds Q[ii, 1] = v[ii]
            # end

            unsafe_copyto!(ptrv, ptrr, n_vec)
            BLAS.scal!(1.0 / r_norm, v)
            unsafe_copyto!(ptrQ, ptrv, n_vec)

            # β .= 0.0
            fill!(β, 0.0);
            β[1] = r_norm
            d::Int = m
            # tt = @timed begin
            for iter = 1:m
                # Arnoldi process start
                # println("\t\t\t\tArnoldi process, step $(iter), AbsErr: $(abs(β[iter])).")
                # display(hcat(Q[:, 1:iter], y))
                F(y_r, v_r; kwargs=action_kwargs)
                # display(hcat(Q[:, 1:iter], y))

                # y = F(Q[:, iter])
                for i = 1:iter
                    # temp = 0.0;
                    # @turbo for jj = 1:n_vec
                    #     @inbounds temp = temp + y[jj] * Q[jj, i]
                    # end
                    @inbounds temp = dot(y, view(Q, :, i))

                    @inbounds H[i, iter] = temp
                    # @turbo for jj = 1:n_vec
                    #     @inbounds y[jj] = y[jj] - temp * Q[jj, i]
                    # end
                    axpy!(-temp, view(Q, :, i), y)
                    # H[i, iter] = dot(y, Q[:, i])
                    # y .-= H[i, iter] .* Q[:, i]
                    # println("\t\t\t\t\tProjection on residual , step $(i), selected:\t$(temp)\t, remainder:\t$(norm(y)).")
                end
                @inbounds H[iter+1, iter] = norm(y)

                # @inbounds if H[iter+1, iter] < 1e-14
                #     break;
                # end

                @inbounds BLAS.scal!(1.0 / H[iter+1, iter], y) 
                # Q[:, iter+1] .= y
                # for ii = 1:n_vec
                #     Q[ii, iter+1] = y[ii]
                # end
                unsafe_copyto!(ptrQ + iter * n_vec * elsize, ptry, n_vec)
                # Arnoldi process done

                # Givens rotation on Hessenberg matrix H start
                for i = 1:iter-1
                    @inbounds temp = c[i] * H[i, iter] + s[i] * H[i+1, iter]
                    @inbounds H[i+1, iter] = -s[i] * H[i, iter] + c[i] * H[i+1, iter]
                    @inbounds H[i, iter] = temp
                end
                @inbounds temp = sqrt(H[iter, iter]^2 + H[iter+1, iter]^2)
                @inbounds c[iter] = H[iter, iter] / temp
                @inbounds s[iter] = H[iter+1, iter] / temp
                @inbounds H[iter, iter] = c[iter] * H[iter, iter] + s[iter] * H[iter+1, iter]
                @inbounds H[iter+1, iter] = 0
                @inbounds β[iter+1] = -s[iter] * β[iter]
                @inbounds β[iter] = c[iter] * β[iter]
                # Givens rotation on Hessenberg matrix H end

                @inbounds abserr = abs(β[iter+1])
                relerr = abserr / b_norm


                # println("\t", abserr)
                if terminate(compute_time, restart_cnt, abserr, relerr, Stop)
                    # solution found
                    found = true
                    d = iter
                    break
                end
                # for ii = 1:n_vec
                #     v[ii] = Q[ii, iter+1]
                # end
                unsafe_copyto!(ptrv, ptrQ + iter * n_vec * elsize, n_vec)
            end
            # end
            # println("\tAllocation in main body:", tt.bytes / 1024)
            # One run done, solve Hy = β in d dimension and compute solution x = x + Q[:, 1:d] * y
            if d < m
                # @inbounds y[1:d] .= H[1:d, 1:d] \ β[1:d]
                unsafe_copyto!(ptry, ptrβ, d)
                # @inbounds LAPACK.gesv!(copy(view(H, 1:d, 1:d)), view(y, 1:d))
                @inbounds ldiv!(UpperHessenberg(view(H, 1:d, 1:d)), view(y, 1:d))

                @inbounds if maximum(abs, view(y, 1:d)) > 1e6
                    break;
                end
                    
                # @inbounds x .+= Q[:, 1:d] * y[1:d]

                mul!(x, view(Q, :, 1:d), view(y, 1:d), 1.0, 1.0)


            else
                # for ii = 1:m
                #     for jj = 1:m
                #         @inbounds Hr[ii, jj] = H[ii, jj]
                #     end
                # end
                # for ii = 1:dim
                #     for jj = 1:m
                #         @inbounds Qdr[ii, jj] = Q[ii, jj]
                #     end
                # end

                # Hr: rs × rs
                # H:  (rs + 1) × rs
                # for ii = 1:m
                #     unsafe_copyto!(ptrHr + (ii - 1) * m * elsize, ptrH + (ii - 1) * (m + 1)  * elsize , m)
                # end

                copy!(Hr, view(H, 1:m, 1:m))



                # Qdr: d × rs
                # Q:   d × (rs + 1)
                # Column major Q can be directly transfer to Qdr
                unsafe_copyto!(ptrQdr, ptrQ, dim * m)
                # @turbo for ii = 1:m
                #     @inbounds c[ii] = β[ii]
                # end
                unsafe_copyto!(ptrc, ptrβ, m)
                # H_n .= H[1:m, :]
                # Q_n .= Q[:, 1:m]
                # r .= β[1:m]
                # @inbounds s .= Hr \ c

                unsafe_copyto!(ptrs, ptrc, length(c))
                # @inbounds LAPACK.gesv!(copy(Hr), s)
                @inbounds ldiv!(UpperHessenberg(Hr), s)

                if maximum(abs, s) > 1e6
                    break;
                end

                # @inbounds x .+= Qdr * s
                mul!(x, Qdr, s, 1.0, 1.0)
            end
        end
        compute_time += stat * 1000
        if found
            break
        end
        if abs(abserr - err_per_run) / abserr < 1e-5
            break
        end
        err_per_run = abserr
        restart_cnt += 1
        
        F(y_r, x_r; kwargs=action_kwargs)
        # @turbo for ii = 1:n_vec
        #     @inbounds r[ii] = b[ii] - y[ii]
        # end

        unsafe_copyto!(ptrr, ptrb, n_vec)
        axpy!(-1.0, y, r)

        # r .= b
        # r .-= y
        fill!(H, 0.0)
        fill!(Q, 0.0)
        # println(norm(r))
    end
    return found;
end


# function testfunc(y_ref, x_ref; kwargs=nothing)
#     if isnothing(kwargs)
#         throw(1)
#     end

#     A_ref = kwargs[1]
#     A = A_ref[]
#     x = x_ref[]
#     y = y_ref[]
#     y .= A * x
# end

# function gmres_test(n, restart)
#     # A = rand(n, n)
#     A = diagm(100 .* rand(n))
#     A .+= (100 / n) .* rand(n, n)
#     b = rand(n)
#     exact = A \ b

#     x = Vector{Float64}(undef, n)
#     Stop = Terminator(100, 1000, 1e-10, 1e-6)
#     # restart = min(20, n)

#     wsp_gmres = get_wsp_gmres(n, restart);
#     x .= 0

#     gmres_matfree!(testfunc, Ref(x), Ref(b), wsp_gmres, n, restart, Stop;
#         is_x_zeros=true, action_kwargs=[Ref(A)])
#     display(norm(b .- A * x))
# end