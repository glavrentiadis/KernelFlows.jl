struct InertialSGD{T} <: AbstractOptimizer
    x::Vector{T}
    ϵ::T # learning rate
    fixed::Bool # if true, all steps are of length ϵ
    history::Vector{Vector{T}} # for inertia
    history_idx::Vector{Int}
    f::T # history weight; current gradient gets 1-f
end


function InertialSGD(x_start::Vector{T}; ϵ::T = 1e-3, fixed::Bool = true, inertia::Int = 10, f::T = 0.5) where T <: Real
    if inertia == 0
        return SGD(x_start; ϵ, fixed)
    end

    InertialSGD(x_start, ϵ, fixed, [zero(x_start) for _ in 1:inertia], ones(Int, 1), f)
end


vecnorm(g::Vector{T}) where T <: Real = sqrt(sum(g.^2) + T(1e-9))


function iterate!(O::InertialSGD, g::AbstractVector{T}) where T <: Real
    g[isnan.(g)] .= T(0)
    α = O.fixed ? vecnorm(g) : one(T)
    lh = length(O.history)
    hi = O.history_idx[1] # history idx, where we read and write
    if ((lh > 0) || (O.history[hi][1] == zero(T)))
        g_history = O.x - O.history[hi]
        g_history .*= O.f*(O.ϵ / vecnorm(g_history))
        O.history[hi] .= O.x
        O.history_idx[1] = hi % lh + 1
        O.x .+= g_history
    end

    g[isnan.(g)] .= zero(T)
    O.x .-= (T(1)-O.f)*O.ϵ / α * g
end
