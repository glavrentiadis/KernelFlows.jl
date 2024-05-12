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


function train!(M::GPModel{T};
                ρ::Function = ρ_RMSE, ϵ::T = .05, niter::Int = 500,
                n::Int = 48, navg::Union{Nothing, Int} = nothing,
                skip_K_update::Bool = false, quiet::Bool = false,
                ngridrounds::Int = 0,
                stepping::Symbol = :standard,
                inertia::Int = 0) where T <: Real

    α₀ = vcat(M.λ, M.θ)
    nλ = length(M.λ)

    Z = M.Z ./ M.λ'
    flowres = flow(Z, M.ζ, ρ, M.kernel, α₀;
                   ϵ, niter, n, ngridrounds, quiet,
                   stepping, inertia)

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
              ngridrounds::Int = 0, ϵ::T = 5e-3,
              stepping::Symbol = :standard,
              inertia::Int = 0,
              quiet::Bool = false) where T <: Real

    Random.seed!(1235)
    ndata, nXdims = size(X) # number of input dimensions
    npars = length(α₀) - nXdims
    logα₀ = log.(α₀)

    # keep reference to the full data set
    X_full = X
    ζ_full = ζ

    reg = T(1e-7)

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

    # minibatch_method = :neighborhood
    minibatch_method = :randompartitions

    # minibatches
    if minibatch_method == :randompartitions
        all_s = get_random_partitions(ndata, n, niter)
        all_s = collect(eachrow(all_s))
    end

    grad = zero(logα) # buffer
    g = similar(grad) # buffer
    b = similar(grad)
    bb = similar(grad)

    Ω = zeros(ndata, ndata)
    buf = zeros(ndata, ndata)
    Z = similar(X) # buffer

    for i ∈ 1:niter
        quiet || ((i % 500 == 0) && println("Training round $i/$niter"))

        # Recalculate tree every now and then; otherwise correct
        # observations are not picked
        if i % 200 == 1 && minibatch_method == :neighborhood
            Z .= X
            Z .*= exp.(logα[1:nXdims])'
            kernel_matrix_fast!(kernel, exp.(logα[nXdims+1:end]), Z, buf, Ω;
                                     precision = false)
            Ω .= abs.(Ω)
        end

        # Indexes for data in X below:
        if minibatch_method == :randompartitions
            s = all_s[i]
        elseif minibatch_method == :neighborhood
            k = rand(1:ndata)
            s = sample(1:ndata, Weights(Ω[:,k]), n, replace = false)
        end

        flowres.ρ_values[i] = @views ξ(X[s,:], ζ[s], logα)[1]
        grad .= @views ∇ξ(X[s,:], ζ[s], logα)[1]
        g .= grad == nothing ? zero(logα) : grad
        # g[isnan.(g)] .= 0 # does not really get triggered.

        # Gradient norm, avoid NaN's if zero grad
        gn(g) = sqrt(sum(g.^2)) + 1e-9

        # gradnorm = gn(g)
        b .= stepping == :fixed ? ϵ * g  / gn(g) : ϵ * g

        # if gn(b) > .5
        #     b .= .5*b/gn(b)
        # end

        logα .-= b

        # Inertia in optimization
        if inertia > 0 && i > 2*inertia
            Δ = (logα - log.(flowres.α_values[end-inertia]))/inertia/2
            logα .+= Δ
        end
        push!(flowres.α_values, exp.(logα))
    end

    flowres
end
