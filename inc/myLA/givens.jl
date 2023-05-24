include("./LA_KWARGS.jl")

using LinearAlgebra

mutable struct Givens_pos
    c
    s
    i::Int
    j::Int
    function Givens_pos(a, b, i, j)
        t = a^2 + b^2
        new(a / t, b / t, i, j)
    end
end

mutable struct Givens
    c
    s
    function Givens(a, b)
        t = a^2 + b^2
        new(a / t, b / t)
    end
end

function Action!(R_ref::Base.RefValue{Givens_pos}, x_ref::Base.RefValue{Vector{T}}) where {T<:Number}
    R = R_ref[]
    x = x_ref[]

    i = R.i
    j = R.j
    c = R.c
    s = R.s

    t = x[i] * c + x[j] * s
    x[j] = -s * x[i] + x[j] * c
    x[i] = t
end

function Action!(R_ref::Base.RefValue{Givens}, x_ref::Base.RefValue{Vector{T}}, i) where {T<:Number}
    R = R_ref[]
    x = x_ref[]

    j = i + 1
    c = R.c
    s = R.s

    t = x[i] * c + x[j] * s
    x[j] = -s * x[i] + x[j] * c
    x[i] = t
end

function Action!(R_ref::Base.RefValue{Givens_pos}, x_ref::Base.RefValue{Matrix{T}}, col) where {T<:Number}
    R = R_ref[]
    x = x_ref[]

    i = R.i
    j = R.j
    c = R.c
    s = R.s

    t = x[i, col] * c + x[j, col] * s
    x[j, col] = -s * x[i, col] + x[j, col] * c
    x[i, col] = t
end

function Action!(R_ref::Base.RefValue{Givens}, x_ref::Base.RefValue{Matrix{T}}, i, col) where {T<:Number}
    R = R_ref[]
    x = x_ref[]

    j = i + 1
    c = R.c
    s = R.s

    t = x[i, col] * c + x[j, col] * s
    x[j, col] = -s * x[i, col] + x[j, col] * c
    x[i, col] = t
end

function Action!(R_ref::Base.RefValue{Givens_pos}, x_ref::Base.RefValue{Matrix{T}}, w1_ref::Base.RefValue{Vector{T}}, w2_ref::Base.RefValue{Vector{T}}) where {T<:Number}
    R = R_ref[]
    x = x_ref[]
    w1 = w1_ref[]
    w2 = w2_ref[]

    i = R.i
    j = R.j
    c = R.c
    s = R.s

    n = length(x)

    w1[1:n] .= x[i, :]
    w2[1:n] .= x[]

    t = x[i, col] * c + x[j, col] * s
    x[j, col] = -s * x[i, col] + x[j, col] * c
    x[i, col] = t
end

function Action!(R_ref::Base.RefValue{Givens}, x_ref::Base.RefValue{Matrix{T}}, i, col) where {T<:Number}
    R = R_ref[]
    x = x_ref[]

    j = i + 1
    c = R.c
    s = R.s

    t = x[i, col] * c + x[j, col] * s
    x[j, col] = -s * x[i, col] + x[j, col] * c
    x[i, col] = t
end


function Action(R::Givens_pos, x)
    y = similar(x)
    y .= x
    Action!(Ref(R), Ref(y))
    return y
end

function Action(R::Givens, x, i)
    y = similar(x)
    y .= x
    Action!(Ref(R), Ref(y), i)
    return y
end