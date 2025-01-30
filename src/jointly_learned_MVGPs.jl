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
    κ = κ_default

    Ω = @views kernel_matrix(M.kernel, logθ, M.Z[s,:] .* λ')

    # Training data - we predict the first entry
    L = @views cholesky(Ω[κ+1:end,κ+1:end])
    h = @views L \ M.ζ[s[κ+1:end]]

    # Debug:
    # KI = inv(Ω[2:end,2:end])
    # hh = KI * M.ζ[s[2:end]]
    # println(sum(abs.(h - hh)))

    t = Ω[1:κ,κ+1:end] * h

    for j in 1:κ
        out[j,i] = t[j]
    end

    out
end


"""Autodifferentiable function to predict training data for a vector
of MVMs in parallel"""
function predict_MVMs(MVMs::Vector{MVGPModel{T}}, s::Vector{Int}, logα_tot::Vector{T}, logα_idxs::Vector{UnitRange{Int}}, z_idxs::Vector{UnitRange{Int}}) where T <: Real

    logαs = [logα_tot[idx] for idx in logα_idxs]
    all_Ms = vcat([MVM.Ms for MVM in MVMs]...)
    ntasks = length(all_Ms)
    nMVMs = length(MVMs)
    tasks = collect(zip(all_Ms, 1:ntasks, logαs))

    κ = κ_default

    z = zeros(T, (κ, length(tasks)))
    z_buf = Zygote.Buffer(z)

    # Threads.@threads does not work with Zygote
    for t in tasks
        predict_M(t..., s, z_buf)
    end

    all_z = copy(z_buf)
    preds = [recover_Y(all_z[:,z_idxs[i]], MVMs[i].G) for i in 1:nMVMs]
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
                         mbalg::Symbol = :multicenter,
                         mbargs::Dict{Symbol,H2} = Dict{Symbol,Any}(),
                         n::Int = 0, # override :n in mbargs
                         niter::Int = 0, # override :niter in mbargs
                         ϵ::T = zero(T), # override :ϵ in mbargs
                         update_K::Bool = true) where {T<:Real, H<:Any, H2<:Any}

    # Override parameters if those were supplied, like in train!().
    (n != 0) && (mbargs[:n] = n)
    (niter != 0) && (mbargs[:niter] = niter)
    (ϵ != 0.) && (optargs[:ϵ] = ϵ)


    Random.seed!(1)
    ndata = length(MVMs[1].Ms[1].ζ)
    reg = T(1e-5)

    # Helper variables for indexing MVMs, needed by ξ
    logα_idxs, z_idxs = get_logα_and_z_idxs(MVMs)

    # Loss function, with regularization, and its gradient.
    function ξ(MVMs::Vector{MVGPModel{T}}, s::Vector{Int},
               logα_tot::AbstractVector{T}, Y_true_all::Vector{Matrix{T}},
               fy_true::Matrix{T}, σ::Matrix{T}) where T <: Real

        Y_preds_all = predict_MVMs(MVMs, s, logα_tot, logα_idxs, z_idxs)

        tot = zero(T)
        κ = κ_default
        for i in 1:κ
            y_preds_all_i = [YP[i,:] for YP in Y_preds_all]
            y_true_all_i = [YT[i,:] for YT in Y_true_all]

            fy_pred_i = fwdfun_pred(y_true_all_i..., y_preds_all_i...)
            fy_true_i = fy_true[i,:]

            # Debug:
            # k = rand(1:285)
            # fp1 = fy_pred[k]
            # ft1 = fy_true[k]
            # yt1 = y_true_all[3][k]
            # yp1 = y_preds_all[3][k]
            # yd1 = yt1 - yp1
            # println("true / pred / diff: $yt1 $yp1 $yd1")
            # Zygote.@ignore display(fy_pred_i' - fy_true_i')

            # Goodness of fit of total function f
            tot += (sum(fy_pred_i[mm] - fy_true_i[mm]) ./ σ).^2

            # Add regularization
            # tot += reg * sum(exp.(logα_tot))

            # Force individual prediction vectors to be accurate
            # yt = vcat(y_true_all_i...)
            # yp = vcat(y_preds_all_i...)
            # tot += T(1e-2) * sum((yt - yp).^2)

            # Debug
            # Zygote.@ignore display((fy_pred_i[1:5]' - fy_true_i[1:5]') ./ σ[i,1:5]')
            # Zygote.@ignore display(fy_pred_i[1:5]' - fy_true_i[1:5]')
            # Zygote.@ignore display(fy_true_i[1:5]')
        end

        return tot
    end

    ∇ξ(MVMs::Vector{MVGPModel{T}}, s::Vector{Int},
       logα_tot::Vector{T}, y_true_all::Vector{Matrix{T}}, fy_true::Matrix{T}, σ::Matrix{T}) =
           Zygote.gradient(logα_tot -> ξ(MVMs, s, logα_tot, y_true_all, fy_true, σ), logα_tot)

    # 2. get minibatches, use the first univariate model of the first GP for this
    M11 = MVMs[1].Ms[1]
    nλ_M11 = length(M11.λ)
    B = get_minibatcher(mbalg, M11.Z; mbargs)
    κ = κ_default

    # 3. get optimizer for the combined state
    logα_tot = get_logα(MVMs)
    O = get_optimizer(optalg, logα_tot; optargs)

    # Let's do all the iterations at once.
    all_Ms = vcat([MVM.Ms for MVM in MVMs]...)
    nMs = length(all_Ms)

    # For each iteration:
    for i in 1:niter
        if (i+1) % 100 == 0
            (print("\rIteration $(i+1)"))
        end
        s = minibatch(B, exp.(O.x[1:nλ_M11])) # Optimization is in log space

        # True labels for first element in minibatch
        y_true_all_s = recover_training_labels(MVMs, s[1:κ])
        # True forward-modeled values
        fy_true_s = fwdfun_true(y_true_all_s...)
        # Error standard deviation
        σ_s = errorsigma(y_true_all_s...)

        loss = ξ(MVMs, s, logα_tot, y_true_all_s, fy_true_s, σ_s)
        ∇logα_tot = ∇ξ(MVMs, s, logα_tot, y_true_all_s, fy_true_s, σ_s)[1]
        iterate!(O, ∇logα_tot)

        # println("grad:")
        # display(∇logα_tot')

        # Record parameter path for later
        for j in 1:nMs
            push!(all_Ms[j].λ_training, exp.(O.x[logα_idxs[j][1:end-4]]))
            push!(all_Ms[j].θ_training, exp.(O.x[logα_idxs[j][end-3:end]]))
            push!(all_Ms[j].ρ_values, loss)
        end
    end

    # Update model, and potentially each M.h
    update_GPModel!(all_Ms; update_K)
end
