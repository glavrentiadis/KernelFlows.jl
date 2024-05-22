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
    n = min(n, length(M.ζ))

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
              k::Kernel, α₀::Vector{T};
              n::Int = min(48, length(ζ) ÷ 2), niter::Int = 500,
              ngridrounds::Int = 0, ϵ::T = 5e-3,
              stepping::Symbol = :standard,
              inertia::Int = 0,
              quiet::Bool = false) where T <: Real

    Random.seed!(1235)
    ndata, nXdims = size(X) # number of input dimensions
    # npars = length(α₀) - nXdims
    logα₀ = log.(α₀)
    nα = length(logα₀)
    nλ = nα - 4
    # Keep reference to the full data set
    X_full = X
    ζ_full = ζ
    reg = T(1e-7)


    # Reference Matern kernels for debugging. Uncomment:
    k_ref = UnaryKernel(Matern32, α₀[end-3:end], nXdims)

    ξ(k::AutodiffKernel, X, ζ, logα) =
        ρ_RMSE(X .* exp.(logα[1:nXdims]'), ζ, k, logα[end-3:end]) +
        reg * sum(exp.(logα))
    ∇ξ(k::AutodiffKernel, X, ζ, logα) = Zygote.gradient(logα -> ξ(k, X, ζ, logα), logα)
    ξ_and_∇ξ(k::AutodiffKernel, X, ζ, logα) = (ξ(k, X, ζ, logα), ∇ξ(k, X, ζ, logα)[1])

    # Only allocate if using AnalyticKernel
    if typeof(k) <: AnalyticKernel
        bufsizes  = ((n,n), (n,n), (nα,), (n,), (n,), (n,), (n,n), (n,n),
                     (n, nXdims), (nα,), (n,), (n,n), (n,n), (n,n))
        Kgradsizes = [(n,n) for _ in 1:nα]
        workbufs = [zeros(T, bs) for bs in bufsizes]
        Kgrads = [zeros(T, bs) for bs in Kgradsizes]
    end
    ξ_and_∇ξ(k::AnalyticKernel, X, ζ, logα) = ρ(X, ζ, k, logα, workbufs, Kgrads)

    # Starting value for optimization. Use grindrounds() if requested
    logα = ngridrounds > 0 ? gridrounds(X, logα₀, ξ, ngridrounds; n, quiet) : logα₀[:]

    flowres = FlowRes(Vector{Vector{Int}}(), zeros(T, niter), Vector{Vector{T}}())

    # minibatch_method = :neighborhood
    # minibatch_method = :randompartitions
    minibatch_method = :hybrid

    # Divide by 2 as we take data from both neighborhood and globally
    n = minibatch_method == :hybrid ? n ÷ 2 : n

    # minibatches
    if minibatch_method in [:randompartitions, :hybrid]
        all_s_rp = get_random_partitions(ndata, n, niter)
        all_s_rp = collect(eachrow(all_s_rp))
    end
    if minibatch_method in [:neighborhood, :hybrid]
        Ω = zeros(ndata, ndata)
        Z = similar(X)
    end

    # buffers
    grad = zero(logα)
    g = similar(grad)
    b = similar(grad)
    bb = similar(grad)
    buf = zeros(ndata, ndata)

    for i ∈ 1:niter
        quiet || ((i % 500 == 0) && println("Training round $i/$niter"))

        # Recalculate tree every now and then; otherwise correct
        # observations are not picked
        if i % 500 == 1 && minibatch_method in [:neighborhood,:hybrid]
            Z .= X
            Z .*= exp.(logα[1:nXdims])'
            kernel_matrix_fast!(k, exp.(logα[nXdims+1:end]), Z, buf, Ω;
                                precision = false)
            # Sometimes most of the weights are zero and there are no
            # n positive points in the selected row. We add eps to be
            # able to sample those at random.
            Ω .= abs.(Ω) .+ eps(T)
        end

        # Indexes for data in X below:
        s = Int[]
        if minibatch_method in [:randompartitions, :hybrid]
            push!(s, all_s_rp[i]...)
        end
        if minibatch_method in [:neighborhood, :hybrid]
            l = rand(1:ndata)
            choices = setdiff(1:ndata,s)
            push!(s, sample(choices, Weights(Ω[choices,l]), n, replace = false)...)
        end

        s = unique(s)

        ρval, gr = ξ_and_∇ξ(k, X[s,:], ζ[s], logα)

        # Debigging block if one wants to compare gradients from ξ_ref()
        # ρval_ref, gr_ref = ξ_and_∇ξ(k_ref, X[s,:], ζ[s], logα)
        # println("Gradient ratio:")
        # display((gr ./ gr_ref)')
        # display(gr)
        # display(gr_ref)

        flowres.ρ_values[i] = ρval
        grad .= gr

        g .= grad == nothing ? zero(logα) : grad
        # g[isnan.(g)] .= 0 # does not really get triggered.

        # Gradient norm, avoid NaN's if zero grad
        gn(g) = sqrt(sum(g.^2)) + 1e-9

        # gradnorm = gn(g)
        b .= stepping == :fixed ? ϵ * g  / gn(g) : ϵ * g
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


function gridrounds(X::AbstractMatrix{T}, logα₀::AbstractVector, ξ::Function, ngridrounds::Int; n::Int = 32, quiet::Bool = false) where T <: Real
    ndata, nXdims = size(X)
    s_gridr = get_random_partitions(ndata, n, ngridrounds * nl)
    s_gridr = collect(eachrow(s_gridr))

    nl = 5 # number of nodes in grid for each variable
    test_logα = logα₀[:]
    for j ∈ 1:ngridrounds # number of grid optimization rounds
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
    return test_logα
end
