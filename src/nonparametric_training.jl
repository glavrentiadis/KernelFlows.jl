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
function train!(M::NonParametricMVGPModel{T}, ρ::Function; segments = 3) where T <: Real
    # How does training look like?
    # 1. Learn θ, parameters of the parametric base kernel
    train!(M.B_orig, ρ; ngridrounds = 6, n = 12, navgs = nothing, niter = 3000, ϵ = .05)

    navg = 25 # hard code this value; used to find best index for prediction
    for (i,model) ∈ enumerate(M.B_orig.Ms) # for each scalar GP output
        (u, ρ_values) = warp_single(model, ρ, M.kernel) # warp with default options
        bestidx = argmin.(runningmedian.(ρ_values, navg)) .+ (navg ÷ 2)
        # Change the warped MVGP's Z to reflect the warped training inputs
        M.B_warped.Ms[i].Z .= u[bestidx]

        # Training indexes for warping segments:
        warping_idxs = splitrange(1, bestidx, segments)
        w = Vector{MVGPModel{T}}(undef, segments) # Initialize warping for i'th output
        for (i1,i2) ∈ zip(idx[1:end-1],idx[2:end]) # for each warping step
            DXs = dimreduce_basic(u[i1])
            DY = dimreduce_basic(u[i2])
            warping_step = MVGPModel(u[i], u[i+1] - u[i], M.kernel, DXsm, DY)
            # 4. Train warpings
            train!(warping_step, ρ; ngridrounds = 3, n = 12, navgs = nothing, niter = 1000)
            push!(w, warping_step)
        end
        M.warpings[i] = w

    end

    # Train the final GP that does the warped test inputs -> test
    # labels computation:
    train!(M.B_warped, ρ; ngridrounds = 6, n = 12, navgs = nothing,
           niter = 3000, ϵ = .05)

    M
end


function warp_single(M::GPModel{T},
              ρ::Function, # loss function
              k::Function; # kernel function
              n::Int = min(300, length(M.Ms[1]).ζ ÷ 2), # minibatch size for SGD
              niter::Int = 500, # number of iterations
              max_ϵ = 0.01, # step size
              q_gradcap = 0.8, # quantile at which we cap gradients
              reg_interp = 1e-12, # regularization for interpolating batch complement
              maxlag::Int = 0, # lag parameter
              q_lag = 0.5) where T <: Real # this fraction of movement comes from lag

    ndata = length(M.ζ)
    nZcols = size(M.Z)[2]

    logθ = log.(M.θ)
    logθ_interp = logθ[:]
    logθ_interp[end] = log(reg_interp)

    ∇ρ(Z, ζ) = Zygote.gradient(Z -> ρ(Z, ζ, k, logθ), Z)

    q_lag = maxlag == 0 ? 0 : q_lag # no lag effects if there is no lag
    q_∇ρ = 1 - q_lag # proportion of movement from gradient; rest is from lag

    u = Matrix{T}[]
    ρ_values = T[]

    if debug
        ∇ρ_values = Matrix{T}[]
        s_values = Vector{Int}[]
    end

    Z_new = M.Z

    # We can use random partitions or then proximity-based sets.
    all_s = KFCommon.get_random_partitions(ndata, n, niter)

    gradstorage = zero(Z₀) # will be used for gradient storage
    buf_D = zeros(ndata - n, n)

    for i ∈ 1:niter
        # s below contains indexes for data in Zₛ
        s = all_s[i,:]
        sC = setdiff(1:ndata, s) # indexes for data in Z\Zₛ
        push!(u, Z_new)

        Zₛ = @view u[i][s,:]
        yₛ = @view y[s] # shorthands

        ρ_values[i] = ρ(Zₛ, yₛ, k, logθ) # record primal value

        ∇ρZₛ = gradstorage[s,:] = ∇ρ(Zₛ, yₛ)[1] # gradient before renormalizing
        Z_new = u[i][:,:]

        ϵ = maxlag == 0 ? max_ϵ : min(i, maxlag) * max_ϵ / maxlag
        # Compute Zₛ increments:
        ΔZₛ = KFCommon.renormalize_columns(∇ρZₛ; q = q_gradcap, new_scale = ϵ)
        Z_new[s,:] -= q_∇ρ * ΔZₛ # move Zₛ

        # interpolate points not in s
        if length(sC) > 0
            gradstorage[sC,:] = @views GP_predict(Zₛ, ΔZₛ, u[i][sC,:], k, logθ_interp; buf_D)[1]
            Z_new[sC,:] -= @views q_∇ρ * gradstorage[sC,:] # move ZₛC
        end

        if maxlag != 0 && i > 1 # no lag before i > 1
            lag = min(i-1, maxlag)
            mean_ΔZ_over_lag = (u[i] - u[i - lag]) / lag
            Z_new += q_lag * mean_ΔZ_over_lag
        end

        if debug # save some memory
            push!(∇ρ_values, gradstorage[:,:])
            push!(s_values, s)
        end

        gradstorage .= 0. # just to make it easier to catch errors

        quiet || print("\rComputing flow $(round(100*i/niter; digits=2))% complete   \r")
    end

    quiet || println()

    ret = debug ? (u, ρ_values, s_values, ∇ρ_values) : (u, ρ_values)
    return ret
end
