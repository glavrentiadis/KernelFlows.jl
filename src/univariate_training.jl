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
    s_values::Vector{Vector{Int}} # indexes of the minibatch in X
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
                optalg::Symbol = :AMSGrad,
                optargs::Dict{Symbol,H} = Dict{Symbol,Any}(),
                ρ::Function = ρ_RMSE, niter::Int = 500, n::Int = 48,
                navg::Union{Nothing, Int} = nothing, quiet::Bool = false,
                skip_K_update::Bool = false) where {T<:Real,H<:Any}

    α₀ = vcat(M.λ, M.θ)
    nλ = length(M.λ)
    n = min(n, length(M.ζ))

    Z = M.Z ./ M.λ'
    O = get_optimizer(optalg, similar(α₀); optargs)

    flowres = flow(Z, M.ζ, ρ, M.kernel, α₀; n, niter, O, quiet)


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
function flow(X::AbstractMatrix{T}, ζ::AbstractVector{T},
              ρ::Function, k::Kernel, α::Vector{T};
              n::Int = min(48, length(ζ) ÷ 2), niter::Int = 500,
              O::AbstractOptimizer = AMSGrad(log.(α)),
              quiet::Bool = false) where T <: Real

    Random.seed!(1235) # fix for reproducibility (minibatching)
    ndata, nλ = size(X) # number of input dimensions
    O.x .= log.(α) # set initial value, optimization in log space
    nα = length(α)
    reg = T(1e-7)

    # Keep a reference to the full data set
    X_full = X
    ζ_full = ζ

    # Reference Matern kernels for debugging. Uncomment:
    k_ref = UnaryKernel(Matern32, α[end-3:end], nλ)

    ξ(k::AutodiffKernel, X, ζ, logα) =
        ρ_RMSE(X .* exp.(logα[1:nλ]'), ζ, k, logα[end-3:end]) +
        reg * sum(exp.(logα))
    ∇ξ(k::AutodiffKernel, X, ζ, logα) = Zygote.gradient(logα -> ξ(k, X, ζ, logα), logα)
    ξ_and_∇ξ(k::AutodiffKernel, X, ζ, logα) = (ξ(k, X, ζ, logα), ∇ξ(k, X, ζ, logα)[1])

    # Only allocate if using AnalyticKernel
    if typeof(k) <: AnalyticKernel
        bufsizes  = ((n,n), (n,n), (nα,), (n,), (n,), (n,), (n,n), (n,n),
                     (n, nλ), (nα,), (n,), (n,n), (n,n), (n,n))
        Kgradsizes = [(n,n) for _ in 1:nα]
        workbufs = [zeros(T, bs) for bs in bufsizes]
        Kgrads = [zeros(T, bs) for bs in Kgradsizes]
    end
    ξ_and_∇ξ(k::AnalyticKernel, X, ζ, logα) = ρ(X, ζ, k, logα, workbufs, Kgrads)

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
        buf = zeros(ndata, ndata)
        Ω = similar(buf)
        Z = similar(X)
    end

    for i ∈ 1:niter
        quiet || ((i % 500 == 0) && println("Training round $i/$niter"))

        # Recalculate tree every now and then; otherwise correct
        # observations are not picked
        if i % 500 == 1 && minibatch_method in [:neighborhood,:hybrid]
            Z .= X
            Z .*= exp.(O.x[1:nλ])'
            kernel_matrix_fast!(k, exp.(O.x[nλ+1:end]), Z, buf, Ω;
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

        ρval, grad = ξ_and_∇ξ(k, X[s,:], ζ[s], O.x)
        iterate!(O, grad) # update parameters in O.x

        # Debigging block if one wants to compare gradients from ξ_ref()
        # ρval_ref, gr_ref = ξ_and_∇ξ(k_ref, X[s,:], ζ[s], logα)
        # println("Gradient ratio:")
        # display((gr ./ gr_ref)')
        # display(gr)
        # display(gr_ref)

        # flowres.ρ_values[i] = ρval
        # grad .= gr

        # g .= grad == nothing ? zero(logα) : grad
        # # g[isnan.(g)] .= 0 # does not really get triggered.

        # # Gradient norm, avoid NaN's if zero grad
        # gn(g) = sqrt(sum(g.^2)) + 1e-9

        # # gradnorm = gn(g)
        # b .= stepping == :fixed ? ϵ * g  / gn(g) : ϵ * g
        # logα .-= b

        # # Inertia in optimization
        # if inertia > 0 && i > 2*inertia
        #     Δ = (logα - log.(flowres.α_values[end-inertia]))/inertia/2
        #     logα .+= Δ
        # end

        flowres.ρ_values[i] = ρval
        push!(flowres.α_values, exp.(O.x))
    end

    flowres
end
