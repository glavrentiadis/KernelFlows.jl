using Zygote

"""For a vector of MVGPModels, return log.(α) for all GPModels in all
MVGPModels, concatenated."""
function get_logα(MVMs::Vector{MVGPModel{T}}) where T <: Real
    vcat([get_logα(MVM) for MVM in MVMs]...)
end


"""Given log-parameters in logα, predict the value of M.ζ[s[1]] using
M.Z[s[1],:] as the inputs. The result is returned in out[i]"""
function predict_M(M::GPModel{T}, i::Int, logα, s::Vector{Int}, out::Union{AbstractArray{T}, Zygote.Buffer{T}}) where T <: Real

    λ_new = exp.(logα[1:end-4])
    logθ = logα[end-3:end]
    λ = λ_new ./ M.λ # Factor to scale the inputs properly
    n = length(s)

    Ω = @views kernel_matrix(M.kernel, logθ, M.Z[s,:] .* λ')

    # Training data - we predict the first entry
    L = @views cholesky(Ω[2:end,2:end])
    h = @views L \ M.ζ[s[2:end]]

    # Debug:
    # KI = inv(Ω[2:end,2:end])
    # hh = KI * M.ζ[s[2:end]]
    # println(sum(abs.(h - hh)))

    out[i] = @views dot(h, Ω[2:end,1])
end


"""Autodifferentiable function to predict training data for a vector
of MVMs in parallel"""
function predict_MVMs(MVMs::Vector{MVGPModel{T}}, s::Vector{Int}, logα_tot::Vector{T}, logα_idxs::Vector{UnitRange{Int}}, z_idxs::Vector{UnitRange{Int}}) where T <: Real

    logαs = [logα_tot[idx] for idx in logα_idxs]
    all_Ms = vcat([MVM.Ms for MVM in MVMs]...)
    ntasks = length(all_Ms)
    nMVMs = length(MVMs)
    tasks = collect(zip(all_Ms, 1:ntasks, logαs))

    z = zeros(T, length(tasks))
    z_buf = Zygote.Buffer(z) # vector of single-

    # Threads.@threads
    for t in tasks
        predict_M(t..., s, z_buf)
    end

    all_z = copy(z_buf)

    preds = [recover_y(all_z[z_idxs[i]], MVMs[i].G) for i in 1:nMVMs]
end


"""Helper function to get the indexes of parameters and vectors for
each GPModel and MVGPModel"""
function get_logα_and_z_idxs(MVMs::Vector{MVGPModel{T}}) where T <: Real
    # We need start/stop indexes of each GPModel's logα in
    # logα_tot. Then, we can just iterate over all M in parallel,
    # indexing with an integer j so that the parameters for that M are
    # logα_tot[logα_idx_starts[j]:logα_idx_ends[j]]
    logα_idx = [1] # indexes of parameters for each M
    z_idx = [1] # indexes of which z belongs to which MVM

    for MVM in MVMs
        push!(z_idx, length(MVM.Ms) + z_idx[end])
        for M in MVM.Ms
            push!(logα_idx, length(get_logα(M)) + logα_idx[end])
        end
    end

    logα_idx_starts = logα_idx[1:end-1]
    logα_idx_ends = logα_idx[2:end] .- 1

    z_idx_starts = z_idx[1:end-1]
    z_idx_ends = z_idx[2:end] .- 1

    logα_idxs = [i1:i2 for (i1, i2) in zip(logα_idx_starts, logα_idx_ends)]
    z_idxs = [i1:i2 for (i1, i2) in zip(z_idx_starts, z_idx_ends)]

    return (logα_idxs, z_idxs)
end


"""Convenience functions to Recover training labels, corresponing to
training labels in index set s. This processes all the M.ζ in all Ms
in all MVMs. Returns a vector of matrices."""
function recover_training_labels(MVMs::Vector{MVGPModel{T}}, s::AbstractVector{Int}) where T <: Real
    [recover_training_labels(MVM, s) for MVM in MVMs]
end


function recover_training_labels(MVM::MVGPModel{T}, s::AbstractVector{Int}) where T <: Real
    Z = hcat([M.ζ[s] for M in MVM.Ms]...)
    recover_Y(Z, MVM.G)
end


function recover_training_labels(MVM::MVGPModel{T}, i::Int) where T <: Real
    recover_training_labels(MVM, [i])[:]
end


function recover_training_labels(MVMs::Vector{MVGPModel{T}}, i::Int) where T <: Real
    [recover_training_labels(MVM, i) for MVM in MVMs]
end


function train_MVMVector(MVMs::Vector{MVGPModel{T}};
                         fwdfun_pred::Function = x -> x,
                         fwdfun_true::Function = fwdfun_te,
                         errorsigma::Function = one{T},
                         optalg::Symbol = :AMSGrad,
                         optargs::Dict{Symbol,H} = Dict{Symbol,Any}(),
                         niter::Int = 500, n::Int = 64,
                         skip_K_update::Bool = false) where {T<:Real, H<:Any}

    Random.seed!(1)
    ndata = length(MVMs[1].Ms[1].ζ)
    reg = T(1e-5)

    # Helper variables for indexing MVMs, needed by ξ
    logα_idxs, z_idxs = get_logα_and_z_idxs(MVMs)

    # Loss function, with regularization, and its gradient.
    function ξ(MVMs::Vector{MVGPModel{T}}, s::Vector{Int},
               logα_tot::AbstractVector{T}, y_true_all::Vector{Vector{T}},
               fy_true::Vector{T}, σ::Vector{T}) where T <: Real
        y_preds_all = predict_MVMs(MVMs, s, logα_tot, logα_idxs, z_idxs)

        fy_pred = fwdfun_pred(y_true_all..., y_preds_all...)

        # Debug:
        # k = rand(1:285)
        # fp1 = fy_pred[k]
        # ft1 = fy_true[k]
        # yt1 = y_true_all[3][k]
        # yp1 = y_preds_all[3][k]
        # yd1 = yt1 - yp1
        # println("true / pred / diff: $yt1 $yp1 $yd1")

        tot = zero(T)
        yt = vcat(y_true_all...)
        yp = vcat(y_preds_all...)
        tot += T(1e-2) * sum((yt - yp).^2)

        tot += sum(((fy_pred - fy_true) ./ σ).^2) + reg * sum(exp.(logα_tot))
        return tot
    end

    ∇ξ(MVMs::Vector{MVGPModel{T}}, s::Vector{Int},
       logα_tot::Vector{T}, y_true_all::Vector{Vector{T}}, fy_true::Vector{T}, σ::Vector{T}) =
           Zygote.gradient(logα_tot -> ξ(MVMs, s, logα_tot, y_true_all, fy_true, σ), logα_tot)

    # 2. get minibatches
    all_s = get_random_partitions(ndata, n, niter)
    all_s = collect(collect.(eachrow(all_s)))

    # 3. get optimizer for the combined state
    logα_tot = get_logα(MVMs)
    O = get_optimizer(optalg, logα_tot; optargs)

    # Let's do all the iterations at once.
    all_Ms = vcat([MVM.Ms for MVM in MVMs]...)
    nMs = length(all_Ms)

    # For each iteration:
    for i in 1:niter
        ((i+1) % 25 == 0) && (print("$i "))
        s = all_s[i]

        # The same minibatch is needed for each univariate model M,
        # but they all have different parameters, so choosing the
        # "correct" one is not possible. We use the first GPModel of
        # the first MVGPModel to select the minibatch. This could also
        # be randomized - however, the weights of some models may be
        # small, so sampling weights should probably follow
        # MVM.G.Yproj.values - but then again, the objective function
        # is a nonlinear function of multiple MVGPModels, so that
        # would not be strictly correct either.
        M11 = MVMs[1].Ms[1]
        Zi = M11.Z[s,:] .* (default_λ(M11) ./ M11.λ)'
        dists = pairwise(SqEuclidean(), Zi; dims = 1)
        s2 = sortperm(sum(exp.(-dists), dims = 1)[:])
        s = s[s2[end:-1:1]]

        # True labels for first element in minibatch
        y_true_all = recover_training_labels(MVMs, s[1])
        # True forward-modeled values
        fy_true = fwdfun_true(y_true_all...)
        # Error standard deviation
        σ = errorsigma(y_true_all...)

        loss = ξ(MVMs, s, logα_tot, y_true_all, fy_true, σ)
        ∇logα_tot = ∇ξ(MVMs, s, logα_tot, y_true_all, fy_true, σ)[1]
        iterate!(O, ∇logα_tot)

        # Record parameter path for later
        for j in 1:nMs
            push!(all_Ms[j].λ_training, exp.(O.x[logα_idxs[j][1:end-4]]))
            push!(all_Ms[j].θ_training, exp.(O.x[logα_idxs[j][end-3:end]]))
            push!(all_Ms[j].ρ_values, loss)
        end
    end

    # Update model, and potentially each M.h
    update_GPModel!(all_Ms; skip_K_update)
end
