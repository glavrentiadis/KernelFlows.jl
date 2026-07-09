mutable struct AMSGrad{T} <: AbstractOptimizer
    # const for safety: minibatching with MultiCenterMinibatch
    # requires persistent pointers, so we make sure we don't change
    # that.
    const x::Vector{T}
    m::Vector{T}
    v::T
    vhat::T
    ϵ::T # learning rate, α in AMSGrad paper
    β1::T
    β2::T
    δ::T # regularization, ϵ in AMSGrad paper
end


function iterate!(O::AMSGrad{T}, g::AbstractVector{T}) where T <: Real
    O.m = O.β1 * O.m + (one(T) - O.β1) * g
    O.v = O.β2 * O.v + (one(T) - O.β2) * dot(g,g) # g.^2
    O.vhat = max(O.vhat, O.v)
    O.x .-= O.ϵ .* O.m / (sqrt(O.vhat) + O.δ)
end


# Standard initializer
function AMSGrad(x_start::Vector{T};
                 ϵ::T = T(1e-3), β1::T = T(.9), β2::T = T(.999),
                 δ::T = T(1e-8)) where T <: Real
    AMSGrad(x_start, zero(x_start), T(0), T(0), ϵ, β1, β2, δ)
end
