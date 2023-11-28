using Random, LinearAlgebra, LoopVectorization, Printf

function dot_product_by_juliaDot(a::Ref{Vector{T}}, b::Ref{Vector{T}}) where {T}
    VecA = a[]
    VecB = b[]
    return dot(VecA, VecB)
end

function dot_product_by_juliaLA(a::Ref{Vector{T}}, b::Ref{Vector{T}}) where {T}
    VecA = a[]
    VecB = b[]
    return VecA' * VecB
end



function dot_product_by_rawloop(a::Ref{Vector{T}}, b::Ref{Vector{T}}) where {T}
    VecA = a[]
    VecB = b[]
    result = zero(eltype(VecA))
    for i in eachindex(VecA)
        result += VecA[i] * VecB[i]
    end
    return result
end

function dot_product_by_vecloop(a::Ref{Vector{T}}, b::Ref{Vector{T}}) where {T}
    VecA = a[]
    VecB = b[]
    result = zero(eltype(VecA))
    @turbo for i in eachindex(VecA)
        result += VecA[i] * VecB[i]
    end
    return result
end

function test_vectorization_on_dot_product(n, type=Float64; seed=1234)
    eng = MersenneTwister(seed)
    VecA = rand(eng, type, n)
    VecB = rand(eng, type, n)

    A = Ref(VecA)
    B = Ref(VecB)

    sample_times = 100
    record = zeros(sample_times, 4)
    x = zeros(4)

    for s_ind in 1:sample_times
        stat = @timed begin
            x[1] = dot_product_by_juliaDot(A, B)
        end
        record[s_ind, 1] = 1000 * (stat.time - stat.gctime)

        stat = @timed begin
            x[2] = dot_product_by_juliaLA(A, B)
        end
        record[s_ind, 2] = 1000 * (stat.time - stat.gctime)

        stat = @timed begin
            x[3] = dot_product_by_rawloop(A, B)
        end
        record[s_ind, 3] = 1000 * (stat.time - stat.gctime)

        stat = @timed begin
            x[4] = dot_product_by_vecloop(A, B)
        end
        record[s_ind, 4] = 1000 * (stat.time - stat.gctime)
    end

    println("Same result?\t", (x[1] ≈ x[2]) && (x[2] ≈ x[3]) && (x[3] ≈ x[4]))

    @printf "Methods\t\t|\t Values \t|\t Min time \t|\t Avg Time \t|\t Max Time \t|\n"
    methods = ["Built-in dot", "Built-in LA", "Raw Loops", "Vec Loops"]

    for ind = 1:4
        @printf "%s\t|\t%.8f\t|\t%.8f\t|\t%.8f\t|\t%.8f\t|\n" methods[ind] x[ind] minimum(record[:, ind]) mean(record[:, ind]) maximum(record[:, ind])
    end

end