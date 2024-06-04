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


function train!(MVM::MVGPModel{T};
                ρ::Function = ρ_RMSE,
                optalg::Symbol = :AMSGrad,
                optargs::Dict{Symbol,H} = Dict{Symbol,Any}(),
                niter::Int = 500, n::Int = 32,
                quiet::Bool = false,
                navg::Union{Nothing, Int} = nothing,
                ζcomps::AbstractVector{Int} = 1:length(MVM.Ms),
                skip_K_update::Bool = false) where {T<:Real, H<:Any}

    if !quiet
        println("Initial scaling factors (log):")
        display(round.(log.(vcat([M.λ' for M in MVM.Ms]...)), sigdigits = 2))

        println("Initial kernel parameters (log):")
        display(round.(log.(vcat([M.θ' for M in MVM.Ms]...)), sigdigits = 2))
    end

    Threads.@threads :static for k ∈ ζcomps
        train!(MVM.Ms[k]; ρ, optalg, optargs, niter, n, navg,
               quiet, skip_K_update = true)
    end

    if !quiet
        println("Final scaling factors (log):")
        display(round.(log.(vcat([M.λ' for M in MVM.Ms]...)), sigdigits = 2))

        println("Final kernel parameters (log):")
        display(round.(log.(vcat([M.θ' for M in MVM.Ms]...)), sigdigits = 2))
    end

    # Update the MVGPModels sequentially, if requested.
    skip_K_update || update_MVGPModel!(MVM)

    MVM
end


function train!(MVMs::Vector{MVGPModel{T}};
                ρ::Function = ρ_RMSE,
                optalg::Symbol = :AMSGrad,
                optargs::Dict{Symbol,H} = Dict{Symbol,Any}(),
                niter::Int = 500,
                n::Int = 32, quiet::Bool = false,
                navg::Union{Nothing, Int} = nothing,
                skip_K_update::Bool = false) where {T<:Real, H<:Any}

    Mlist = [M for MVM in MVMs for M in MVM.Ms]
    println("Training $(length(Mlist)) GPs...")
    Threads.@threads :static for M in Mlist
        train!(M; ρ, optalg, optargs, niter, n, navg,
               skip_K_update, quiet)
    end
    println("done!")

end


"""Train the univariate MVM.Ms models together to minimize prediction
error. Only uses with the analytic Matern32 loss for now. One should
generally supply the Y_tr used for training, as the recovered
dimension-reduced version is not exact."""
function train_L2_together!(MVM::MVGPModel{T};
                            Y_tr::Union{Nothing, AbstractMatrix{T}} = nothing,
                            optalg::Symbol = :AMSGrad,
                            optargs::Dict{Symbol,H} = Dict{Symbol,Any}(),
                            niter::Int = 500, n::Int = 64, skip_K_update::Bool = false) where {T<:Real, H<:Any}

    Random.seed!(1235)
    ndata, nZdims = size(MVM.Ms[1].Z) # number of data and input dimensions
    nMs = length(MVM.Ms)
    κ = max(min(n÷5, 20),4)
    nt = Threads.nthreads()
    ylen = length(MVM.G.μY)
    nλ = length(MVM.Ms[1].λ)
    nα = nλ + 4
    local ρ

    bufsizes  = ((n,n), (n,n), (nα,), (n,), (n,), (n,), (n,n), (n,n),
                 (n, nZdims), (nα,), (n,), (n,n), (n,n), (n,n))
    Kgradsizes = [(n,n) for _ in 1:nα]
    workbufs_all = [[zeros(T, bs) for bs in bufsizes] for i in 1:nt]
    Kgrads_all = [[zeros(T, bs) for bs in Kgradsizes] for i in 1:nt]

    all_s_rp = get_random_partitions(ndata, n, niter)
    all_s_rp = collect(eachrow(all_s_rp))

    ybuf1 = zeros(T, ylen)
    ybuf2 = zeros(T, ylen)

    logα_tot = log.(vcat([vcat(M.λ, M.θ) for M in MVM.Ms]...))
    ∇logα = similar(logα_tot)

    # Each GPModel predicts a scalar for each LOO obsevation
    allpreds = zeros(κ, nMs)
    # For each left-out observation there are nMs nα-size gradients
    allgrads = zeros(nα, nMs, κ)

    O = get_optimizer(optalg, logα_tot; optargs...)

    for k ∈ 1:niter
        s = all_s_rp[k]
        s_LOO = randperm(n)[1:κ] # The minibatch that we predict

        # Threads.@threads :static
        for j in 1:nMs # for each scalar model
            tid = Threads.threadid()
            M = MVM.Ms[j] # shortand
            logα = logα_tot[(j-1)*nα+1:j*nα]
            Z = @views M.Z[s,:] ./ M.λ' .* exp.(logα[1:nλ])' # GP inputs
            ζ = M.ζ[s] # GP outputs

            # Get predicted GP components, which are then collected
            # together, and used after the for-loop.
            (preds, grads) = ρ_RMSE(Z, ζ, M.kernel, logα,
                                    workbufs_all[tid], Kgrads_all[tid];
                                    s_LOO = s_LOO, return_components = true)
            allgrads[:,j,:] .= grads
            allpreds[:,j] .= preds
        end

        Y_LOO = Y_tr == nothing ? recover_Y(hcat([M.ζ[s_LOO] for M in MVM.Ms]...), MVM.G) : Y_tr[s[s_LOO],:]

        Y_LOO .-= MVM.G.μY'

        ∇logα .= 0
        for j in 1:κ
            ybuf1 .= -Y_LOO[j,:] # ./ MVM.G.σY
            for i in 1:nMs
                # ybuf1 gets 2 x residual in original space
                vec = @view MVM.G.Yproj.vectors[:,i]
                val = MVM.G.Yproj.values[i]
                ybuf1 .+= 2. * val * allpreds[j,i] * vec # .* MVM.G.σY
            end
            # ybuf1 .-= MVM.G.μY #  Y_LOO[j,:]


            for l in 1:nα
                # ybuf2 .= 0.0 # This will have the gradient parts
                for i in 1:nMs
                    vec = @view MVM.G.Yproj.vectors[:,i]
                    val = MVM.G.Yproj.values[i]
                    ybuf2 .= val * allgrads[l,i,j] * vec # .* MVM.G.σY
                    ∇logα[(i-1)*nα+l] += dot(ybuf1, ybuf2)
                end
            end
        end

        logα_tot .= iterate!(O, ∇logα)

        ρ = dot(ybuf1, ybuf1) / 4.0 # negate the 2 above. This is the squared residual.

        for i in 1:nMs
            M = MVM.Ms[i]
            push!(M.ρ_values, ρ)
            newλ = exp.(logα_tot[nα*(i-1)+1:i*nα-4])
            push!(M.λ_training, newλ)
            newθ = exp.(logα_tot[i*nα-3:i*nα])
            push!(M.θ_training, newθ)
        end
    end
    # Save results
    Threads.@threads for M in MVM.Ms
        skip_K_update || update_GPModel!(M; newλ = M.λ_training[end], newθ = M.θ_training[end])
    end

    # display(allgrads)

    return ybuf1, ybuf2
    MVM
end
