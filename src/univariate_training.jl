#  Copyright 2023 California Institute of Technology
#
#  Licensed under the Apache License, Version 2.0 (the \"License\");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an \"AS IS\" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# Author: Jouni Susiluoto, jouni.i.susiluoto@jpl.nasa.gov
#
struct FlowRes{T}
    s_values::Vector{Vector{Int}} # indexes of the minibatch in X_full
    ρ_values::Vector{T} # loss function values
    α_values::Vector{Vector{T}} # scaling and kernel parameters and nugget
end


function best_α_from_flowres(flowres::FlowRes{T};
                        navg::Union{Int, Nothing} = nothing,
                        quiet::Bool = false) where T <: Real

    navg == nothing && return flowres.α_values[end]

    bestidx = argmin(runningmedian(flowres.ρ_values, navg))
    println("Selecting best index $bestidx")
    flowres.α_values[bestidx]
end


function train!(M::GPModel{T}, ρ::Function;
                ϵ::T = .05, niter::Int = 500, n::Int = 48, ngridrounds::Int = 6,
                navg::Union{Nothing, Int} = nothing,
                skip_K_update::Bool = false, quiet::Bool = false) where T <: Real

    α₀ = vcat(M.λ, M.θ)
    nλ = length(M.λ)

    Z = M.Z ./ M.λ'
    flowres = flow(Z, M.ζ, ρ, M.kernel, α₀;
                   ϵ, niter, n, ngridrounds, quiet)

    if niter > 0 # update parameters from training
        α = best_α_from_flowres(flowres; navg, quiet)
    elseif length(M.ρ_values) > 0 # use last training value
        α = vcat(M.λ_training[end], M.θ_training[end])
    else # if no training has been done, go with initial values
        α = vcat(M.λ, M.θ)
    end

    update_GPModel!(M; newλ = α[1:nλ], newθ = α[nλ+1:end], skip_K_update)
    append!(M.ρ_values, flowres.ρ_values)
    append!(M.λ_training, [α[1:nλ] for α in flowres.α_values])
    append!(M.θ_training, [α[nλ+1:end] for α in flowres.α_values])
    M
end


"""Function to do the actual 1-d learning. This does not depend on
GPModel; that way it is more generally usable."""
function flow(X::AbstractMatrix{T}, ζ::AbstractVector{T}, ρ::Function,
              kernel::Kernel, α₀::Vector{T};
              n::Int = min(48, length(ζ) ÷ 2), niter::Int = 500,
              ngridrounds::Int = 6, ϵ = 5e-2,
              quiet::Bool = false) where T <: Real

    Random.seed!(1235)
    ndata, nXdims = size(X) # number of input dimensions
    npars = length(α₀) - nXdims
    logα₀ = log.(α₀)

    # keep reference to the full data set
    X_full = X
    ζ_full = ζ

    reg = 1e-7

    ξ(X, ζ, logα) = ρ(X .* exp.(logα[1:nXdims]'), ζ, kernel, logα[nXdims+1:end]) + reg * sum(exp.(logα))
    ∇ξ(X, ζ, logα) = Zygote.gradient(logα -> ξ(X, ζ, logα), logα)


    nl = 5
    s_gridr = get_random_partitions(ndata, n, ngridrounds * nl)
    s_gridr = collect(eachrow(s_gridr))

    test_logα = logα₀[:]
    for j ∈ 1:ngridrounds # This many rounds of grid optimizations
        quiet || println("Grid optimization round $j")

        for i ∈ randperm(nXdims + npars) # go through parameters in random order
            tlogα = repeat(test_logα', nl) # temporary variable

            tlogα[:,i] .+= collect(range(-2., 2., nl))
            start_ξ_vals = zeros(nl)
            ss = [s_gridr[k] for k ∈ (j-1)*nl+1:j*nl]

            for k ∈ 1:nl
                ξ_val = @views sum([ξ(X[s,:], ζ[s], tlogα[k,:]) for s ∈ ss])
                start_ξ_vals[k] = ξ_val
            end
            test_logα[i] = tlogα[argmin(start_ξ_vals), i]
        end
    end

    logα = test_logα # rename this now that we have a starting point guess

    flowres = FlowRes(Vector{Vector{Int}}(), zeros(T, niter), Vector{Vector{T}}())

    # minibatches
    all_s = get_random_partitions(ndata, n, niter)
    all_s = collect(eachrow(all_s))

    grad = zero(logα) # buffer
    g = similar(grad) # buffer
    b = similar(grad)
    bb = similar(grad)
    for i ∈ 1:niter
        quiet || ((i % 500 == 0) && println("Training round $i/$niter"))
        s = all_s[i]

        push!(flowres.α_values, exp.(logα))
        flowres.ρ_values[i] = @views ξ(X[s,:], ζ[s], logα)[1]
        grad .= @views ∇ξ(X[s,:], ζ[s], logα)[1]
        g .= grad == nothing ? zero(logα) : grad
        # g[isnan.(g)] .= 0 # does not really get triggered.

        # Gradient norm, avoid NaN's if zero grad
        gn(g) = sqrt(sum(g.^2)) + 1e-9

        # gradnorm = gn(g)
        # b .= ϵ * g  / gradnorm

        # Uncomment to allow inertia.
        b .= ϵ * g
        logα .-= b
        if i > 2*20
            Δ = (logα - log.(flowres.α_values[end-20]))/20/2
            logα .+= Δ
        end


    end

    flowres
end
