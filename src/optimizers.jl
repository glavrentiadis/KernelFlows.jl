abstract type AbstractOptimizer end

# In this file we have the various optimizers used by
# KernelFlows.jl. Convention: an iterate!(O::AbstractOptimizer{T},
# g::Vector{T}) method should be provided for updating the state of
# any optimizer, returning the updated parameters. For all algorithms,
# the learning rate (step size) should be called ϵ.

include("opt_SGD.jl")
include("opt_AMSGrad.jl")
include("opt_InertialSGD.jl")


function get_optimizer(optalg::Symbol, x_start::Vector{T};
                       optargs::Dict{Symbol,H} = Dict{Symbol,Any}()) where {T<:Real, H<:Any}
    optalg == :AMSGrad && (return AMSGrad(x_start; optargs...))
    optalg == :SGD && (return SGD(x_start; optargs...))
    optalg == :InertialSGD && (return InertialSGD(x_start; optargs...))
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
