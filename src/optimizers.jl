abstract type AbstractOptimizer end

# In this file we have the various optimizers used by
# KernelFlows.jl. Convention: an iterate!(O::AbstractOptimizer{T},
# g::Vector{T}) method should be provided for updating the state of
# any optimizer, returning the updated parameters. For all algorithms,
# the learning rate (step size) should be called ϵ.


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


struct SGD{T} <: AbstractOptimizer
    x::Vector{T}
    ϵ::T # learning rate
    fixed::Bool # if true, all steps are of length ϵ
end


function iterate!(O::SGD, g::AbstractVector{T}) where T <: Real
    α = O.fixed ? sqrt(sum(g.^2) + 1e-9) : 1.0
    O.x .-= O.ϵ / α * g
end


function SGD(x_start::Vector{T}; ϵ::T = 1e-3, fixed::Bool = true) where T <: Real
    SGD(x_start, ϵ, fixed)
end


function get_optimizer(optalg::Symbol, x_start::Vector{T};
                       optargs::Dict{Symbol,H} = Dict{Symbol,Any}()) where {T<:Real, H<:Any}
    optalg == :AMSGrad && (return AMSGrad(x_start; optargs...))
    optalg == :SGD && (return SGD(x_start; optargs...))
end


## Grid optimization code. Rarely works sufficiently well, and largely
## obsolete due to how well SGD and AMSGrad work.
# function gridrounds(X::AbstractMatrix{T}, logα::AbstractVector, ξ::Function, ngridrounds::Int; n::Int = 32, quiet::Bool = false) where T <: Real
#     ndata, nXdims = size(X)
#     s_gridr = get_random_partitions(ndata, n, ngridrounds * nl)
#     s_gridr = collect(eachrow(s_gridr))

#     nl = 5 # number of nodes in grid for each variable
#     test_logα = logα[:]
#     for j ∈ 1:ngridrounds # number of grid optimization rounds
#         quiet || println("Grid optimization round $j")

#         for i ∈ randperm(nXdims + npars) # go through parameters in random order
#             tlogα = repeat(test_logα', nl) # temporary variable

#             tlogα[:,i] .+= collect(range(-2., 2., nl))
#             start_ξ_vals = zeros(nl)
#             ss = [s_gridr[k] for k ∈ (j-1)*nl+1:j*nl]

#             for k ∈ 1:nl
#                 ξ_val = @views sum([ξ(X[s,:], ζ[s], tlogα[k,:]) for s ∈ ss])
#                 start_ξ_vals[k] = ξ_val
#             end
#             test_logα[i] = tlogα[argmin(start_ξ_vals), i]
#         end
#     end
#     return test_logα
# end
