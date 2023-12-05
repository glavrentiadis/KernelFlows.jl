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
export GPModel, update_GPModel!


"""Univariate GP model struct. This struct is sufficient for
prediction of scalar zy at a new x, so that different zys can be
combined to construct the original y."""
struct GPModel{T}
    ζ::Vector{T}        # ζ = zytransf.(zy_tr), needed for updating h
    h::Vector{T}        # K⁻¹ ζ, where K is the kernel matrix
    Z::Matrix{T}        # transformed X_tr *after* applying λ-scaling
    λ::Vector{T}        # X_tr scaling factors for each input dimension
    θ::Vector{T}        # kernel parameters for spherical + linear kernel
    kernel::Function    # kernel function
    zytransf::Function    # nonlinear 1-d output transformations
    zyinvtransf::Function # inverse 1-d output transformations
    ρ_values::Vector{T} # loss function values from latest training
end


# These need struct GPModel => include only after its definition
include("univariate_training.jl")
include("univariate_prediction.jl")


function GPModel(ZX_tr::Matrix{T}, # inputs after reduce()
                 zy_tr::Vector{T}, # outputs after reduce()
                 kernel::Function;
                 λ::Union{Nothing, Vector{T}} = nothing,
                 θ::Union{Nothing, Vector{T}} = nothing,
                 transform_zy::Bool = false) where T <: Real

    # Gaussianize 1-d labels if requested
    transform_zy || (zytransf = zyinvtransf = identity)
    transform_zy && ((zytransf, zyinvtransf) = gaussianize_transforms(zy_tr))
    ζ = zytransf.(zy_tr)

    ntr, nZXdims = size(ZX_tr)

    h = zeros(ntr)
    λ == nothing && (λ = ones(nZXdims))
    θ == nothing && (θ = exp.([0., 0., -4., -7.]))

    @assert length(λ) == nZXdims "Invalid λ length"
    @assert length(θ) == 4 "Invalid θ length"

    return GPModel(ζ, h, ZX_tr, λ, θ, kernel, zytransf, zyinvtransf, zeros(10000))
end


"""Updates 1-d GPmodel M after new parameters newλ and/or newθ have
   been obtained by training the model. Even if these parameters are
   not given, M.h is recomputed (it is not computed when GPModel is
   initialized)."""
function update_GPModel!(M::GPModel{T};
                         newλ::Union{Nothing, Vector{T}} = nothing,
                         newθ::Union{Nothing, Vector{T}} = nothing,
                         buf1::Union{Nothing, Matrix{T}} = nothing,
                         buf2::Union{Nothing, Matrix{T}} = nothing,
                         nXlinear::Int = 1,
                         skip_K_update::Bool = false) where T <: Real

    # Update M.Z, M.λ, and M.θ, if requested
    if newλ != nothing
        newλ[newλ .< -20.] .= -20.
        newλ[newλ .> 20.] .= 20.
        l = newλ ./ M.λ
        M.Z .*= l'
        M.λ .= newλ
    end

    if newθ != nothing
        M.θ .= newθ
    end

    if !skip_K_update
        # Allocate buffers
        ntr = length(M.ζ)
        (buf1 == nothing) && (buf1 = zeros(ntr, ntr))
        (buf2 == nothing) && (buf2 = zeros(ntr, ntr))

        KI = kernel_matrix_fast(M.Z, buf1, buf2, M.kernel, M.θ;
                                precision = true, nXlinear)
        mul!(M.h, KI, M.ζ)
    end

    M
end
