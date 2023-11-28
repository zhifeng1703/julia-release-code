include("./LA_KWARGS.jl")

using LinearAlgebra

struct HouseHolder{T<:Number}
    v::Vector{T};
    a::T;
    HouseHolder{T}(v::Vector{T}, a::T) = new(v, a);
    HouseHolder{T}(v::Vector{T}) = new(v, -2.0 / norm(v, 2));
end
function Base.show(is::IO, HouseHolder{T<:Number})
    println("Householder reflector that transformed the following vector");
    display(vector);
end



