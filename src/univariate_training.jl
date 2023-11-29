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
                navg::Union{Nothing, Int} = nothing, nXlinear::Int = 1,
                quiet::Bool = false) where T <: Real

    α₀ = vcat(M.λ, M.θ)
    nλ = length(M.λ)

    Z = M.Z ./ M.λ'
    flowres = flow(Z, M.ζ, ρ, M.kernel, α₀;
                   ϵ, niter, n, ngridrounds, nXlinear, quiet)
    α = best_α_from_flowres(flowres; navg, quiet)

    update_GPModel!(M; newλ = α[1:nλ], newθ = α[nλ+1:end], nXlinear)
    M.ρ_values .= 0
    m = min(length(M.ρ_values), length(flowres.ρ_values))
    M.ρ_values[1:m] .= flowres.ρ_values[1:m]
    M
end


"""Function to do the actual 1-d learning. This does not depend on
GPModel; that way it is more generally usable."""
function flow(X::AbstractMatrix{T}, ζ::AbstractVector{T}, ρ::Function,
              kernel::Function, α₀::Vector{T};
              n::Int = min(48, length(ζ) ÷ 2), niter::Int = 500,
              ngridrounds::Int = 6, ϵ = 5e-2, nXlinear::Int = 1,
              quiet::Bool = false) where T <: Real

    Random.seed!(1235)
    ndata, nXdims = size(X) # number of input dimensions
    npars = length(α₀) - nXdims
    logα₀ = log.(α₀)

    # keep reference to the full data set
    X_full = X
    ζ_full = ζ

    reg = 1e-3

    ξ(X, ζ, logα) = ρ(X .* exp.(logα[1:nXdims]'), ζ, kernel, logα[nXdims+1:end]; nXlinear) +
        reg * sum(exp.(logα[1:nXdims]))
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

            Threads.@threads for k ∈ 1:nl
                ξ_val = @views sum([ξ(X[s,:], ζ[s], tlogα[k,:]) for s ∈ ss])
                start_ξ_vals[k] = ξ_val
            end
            test_logα[i] = tlogα[argmin(start_ξ_vals), i]
        end
    end

    logα = test_logα # rename this now that we have a starting point guess
    quiet || println("Starting point for kernel parameters: $logα")

    flowres = FlowRes(Vector{Vector{Int}}(), zeros(T, niter), Vector{Vector{T}}())

    m = 50 # number of past gradients to average over
    avg = zeros(length(logα), m)

    grad = zero(logα) # buffer
    g = similar(grad) # buffer
    b = similar(grad)
    bb = similar(grad)
    for i ∈ 1:niter
        quiet || ((i % 100 == 0) && println("Training round $i/$niter"))
        # (i % 100 == 0) && display(logα')
        s = all_s[i]

        push!(flowres.α_values, exp.(logα))
        flowres.ρ_values[i] = @views ξ(X[s,:], ζ[s], logα)[1]
        grad .= @views ∇ξ(X[s,:], ζ[s], logα)[1]
        g .= grad == nothing ? zero(logα) : grad

        # g[isnan.(g)] .= 0 # does not really get triggered.
        # Gradient norm, avoid NaN's if zero grad
        gn(g) = sqrt.(sum(g.^2)) + 1e-9

        gradnorm = gn(g)
        b .= ϵ * g / gradnorm # cap this timestep's gradient at ϵ
        b .= sign.(b) .* max.(0.01*ϵ, abs.(b)) # move all parameters by at least 0.01*ϵ

        avg[:, i%m + 1] .= b
        bb .= @views sum(avg, dims = 2)[:] # average contribution (unscaled)
        logα .-= b + bb ./ gn(bb) .* ϵ
    end

    quiet || println("Final kernel parameters: $logα")

    flowres
end
