struct SGD{T} <: AbstractOptimizer
    x::Vector{T}
    ϵ::T # learning rate
    fixed::Bool # if true, all steps are of length ϵ
end


function SGD(x_start::Vector{T}; ϵ::T = 1e-3, fixed::Bool = true) where T <: Real
    SGD(x_start, ϵ, fixed)
end


function iterate!(O::SGD, g::AbstractVector{T}) where T <: Real
    α = O.fixed ? sqrt(sum(g.^2) + T(1e-9)) : one(T)
    O.x .-= O.ϵ / α * g
end
