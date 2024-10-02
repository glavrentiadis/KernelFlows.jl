#  Copyright 2023-2024 California Institute of Technology
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
# Author: Grigorios Lavrentiadis, glavrent@caltech.edu
#

include("kernel_functions_path.jl")

# Ground-Motion Model Kernels
#------------------------------
# Aleatory Kernel Functions
# ---   ---   ---   ---   ---
"""
    Binary between event aleatory variability kernel function
"""
function aleat_bevent_binary(EQID1::Union{AbstractMatrix{T}, AbstractArray{T}}, 
                             EQID2::Union{AbstractMatrix{T}, AbstractArray{T}},
                             θ) where T <: Real
    
    #hyperparameters
    τ₀ = @view θ[1] #between event variance
    
    #evaluate kernel for between event residuals
    return group_binary(EQID1, EQID2, τ₀; δ=1e-6)
end

# Individual Non-ergodic Kernels
# ---   ---   ---   ---   ---
"""
    Binary source kernel function
"""
function source_binary(X₁ₗ::AbstractMatrix{T}, X₂ₗ::AbstractMatrix{T},
                       θₗ::AbstractVector{T}) where T <: Real
    
    #non-ergodic source
    return spherical_exp_binary(X₁ₗ,X₂ₗ,θₗ)
end

"""
    Binary site kernel function
"""
function site_binary(X₁ₛ::AbstractMatrix{T}, X₂ₛ::AbstractMatrix{T},
                     θₛ::AbstractVector{T}) where T <: Real

    #non-ergodic site
    return spherical_exp_binary(X₁ₛ,X₂ₛ,θₛ)
end

"""
    Binary path kernel function
"""
function path_binary(X₁::AbstractMatrix{T}, X₂::AbstractMatrix{T},
                     θₚ::AbstractVector{T}; 
                     n_integ_pt=5, flag_normalize=true) where T <: Real
    
    #hyperparameters
    ωₚ² = @views θₚ[1]
    λₚ  = @views θₚ[2]

    #set up path kernel
    # - number of integration points
    # - underling covariance function
    # - path normalization
    # κₚ = ωₚ² * PathKernel(n_integ_pt, d -> exp(-d), 
    #                       flag_normalize) ∘ ScaleTransform(λₚ)
    # κₚ = PathKernel(n_integ_pt, d -> exp(-d), flag_normalize)
    κₚ = ωₚ² * PathKernel(n_integ_pt, 
                          ExponentialKernel(; metric=Euclidean()), 
                          flag_normalize) ∘ ScaleTransform(λₚ)
    #evaluate path kernel
    Kₚ_buff = Zygote.Buffer(zeros(size(X₁)[1], size(X₂)[1]))
    for j1 in 1:size(X₁)[1]
        for j2 in 1:size(X₂)[1]
            Kₚ_buff[j1,j2] = κₚ(X₁[j1,:],X₂[j2,:])
        end
    end
    return Zygote.copy(Kₚ_buff)

    # return kernelmatrix(κₚ, RowVecs(X₁), RowVecs(X₂)) 
end

# Composite Non-ergodic Kernels
# ---   ---   ---   ---   ---
"""
    Binary source & site kernel function
"""
function sourcesite_binary(X₁::AbstractMatrix{T}, X₂::AbstractMatrix{T},
                           θ::AbstractVector{T}) where T <: Real
    
    #hyperparameters
    θₗ = @view θ[1:2] #source parametes
    θₛ = @view θ[3:4] #site parameters

    #coordinates
    X₁ₗ = @view X₁[:,1:2] #1st set of events
    X₂ₗ = @view X₂[:,1:2] #2nd set of events
    X₁ₛ = @view X₁[:,3:4] #1st set of sites
    X₂ₛ = @view X₂[:,3:4] #2nd set of sites

    #evaluate total kernel
    Kₜ  = source_binary(X₁ₗ, X₂ₗ, θₗ)
    Kₜ += site_binary(X₁ₛ,   X₂ₛ, θₛ)
    
    return Kₜ
end

"""
    Binary path & site kernel function
"""
function pathsite_binary(X₁::AbstractMatrix{T}, X₂::AbstractMatrix{T},
                         θ::AbstractVector{T}) where T <: Real
    
    #hyperparameters
    θₚ = @view θ[1:2] #path parametes
    θₛ = @view θ[3:4] #site parameters

    #coordinates
    X₁ₛ = @view X₁[:,3:4] #1st set of sites
    X₂ₛ = @view X₂[:,3:4] #2nd set of sites

    #evaluate total kernel
    Kₜ  = path_binary(X₁,  X₂,  θₚ)
    Kₜ += site_binary(X₁ₛ, X₂ₛ, θₛ)
    
    return Kₜ
end

"""
    Binary source, path & site kernel function
"""
function sourcepathsite_binary(X₁::AbstractMatrix{T}, X₂::AbstractMatrix{T},
                               θ::AbstractVector{T}) where T <: Real
    
    #hyperparameters
    θₗ = @view θ[1:2] #source parametes
    θₚ = @view θ[3:4] #path parametes
    θₛ = @view θ[5:6] #site parameters

    #coordinates
    X₁ₗ = @view X₁[:,1:2] #1st set of events
    X₂ₗ = @view X₂[:,1:2] #2nd set of events
    X₁ₛ = @view X₁[:,3:4] #1st set of sites
    X₂ₛ = @view X₂[:,3:4] #2nd set of sites

    #evaluate total kernel
    Kₜ  = source_binary(X₁ₗ, X₂ₗ, θₗ)
    Kₜ += path_binary(X₁,    X₂,  θₚ)
    Kₜ += site_binary(X₁ₛ,   X₂ₛ, θₛ)
    
    return Kₜ
end

# Non-ergodic Kernels with Aleat Variability
# ---   ---   ---   ---   ---
"""
    Binary source kernel function with between event aleatory variability
"""
function source_aleat_binary(X₁::AbstractMatrix{T}, X₂::AbstractMatrix{T},
                             θ::AbstractVector{T}) where T <: Real
    
    #hyperparameters
    θₗ = @view θ[1:2] #source parametes
    θₐ = @view θ[3]   #aleatory parameters

    #event ids
    EQID₁ = @view X₁[:,1] #1st set of earthquake ids
    EQID₂ = @view X₂[:,1] #2nd set of earthquake ids
    #coordinates
    X₁ₗ   = @view X₁[:,2:3] #1st set of events
    X₂ₗ   = @view X₂[:,2:3] #2nd set of events

    #evaluate total kernel
    Kₜ  = aleat_bevent_binary(EQID₁, EQID₂, θₐ)
    Kₜ += source_binary(X₁ₗ, X₂ₗ, θₗ)
    
    return Kₜ
end

"""
    Binary path kernel function with between event aleatory variability
"""
function path_aleat_binary(X₁::AbstractMatrix{T}, X₂::AbstractMatrix{T},
                           θ::AbstractVector{T}) where T <: Real
    
    #hyperparameters
    θₚ = @view θ[1:2] #path parametes
    θₐ = @view θ[3]   #aleatory parameters

    #event ids
    EQID₁ = @view X₁[:,1] #1st set of earthquake ids
    EQID₂ = @view X₂[:,1] #2nd set of earthquake ids

    #evaluate total kernel
    Kₜ  = aleat_bevent_binary(EQID₁, EQID₂, θₐ)
    Kₜ += path_binary(X₁[:,2:5], X₂[:,2:5], θₚ)
    
    return Kₜ
end

"""
    Binary site kernel function with between event aleatory variability
"""
function site_aleat_binary(X₁::AbstractMatrix{T}, X₂::AbstractMatrix{T},
                           θ::AbstractVector{T}) where T <: Real
    
    #hyperparameters
    θₛ = @view θ[1:2] #site parametes
    θₐ = @view θ[3]   #aleatory parameters

    #event ids
    EQID₁ = @view X₁[:,1] #1st set of earthquake ids
    EQID₂ = @view X₂[:,1] #2nd set of earthquake ids
    #coordinates
    X₁ₛ   = @view X₁[:,2:3] #1st set of sites
    X₂ₛ   = @view X₂[:,2:3] #2nd set of sites

    #evaluate total kernel
    Kₜ  = aleat_bevent_binary(EQID₁, EQID₂, θₐ)
    Kₜ += site_binary(X₁ₛ, X₂ₛ, θₛ)
    
    return Kₜ
end

"""
    Binary source and site kernel function with between event aleatory variability
"""
function sourcesite_aleat_binary(X₁::AbstractMatrix{T}, X₂::AbstractMatrix{T},
                               θ::AbstractVector{T}) where T <: Real
    
    #hyperparameters
    θₙ = @view θ[1:4] #non-ergodic parametes
    θₐ = @view θ[5]   #aleatory parameters

    #event ids
    EQID₁ = @view X₁[:,1] #1st set of earthquake ids
    EQID₂ = @view X₂[:,1] #2nd set of earthquake ids

    #evaluate total kernel
    Kₜ  = aleat_bevent_binary(EQID₁, EQID₂, θₐ)
    Kₜ += sourcesite_binary(X₁[:,2:5], X₂[:,2:5], θₙ)
    
    return Kₜ
end

"""
    Binary path and site kernel function with between event aleatory variability
"""
function pathsite_aleat_binary(X₁::AbstractMatrix{T}, X₂::AbstractMatrix{T},
                               θ::AbstractVector{T}) where T <: Real
    
    #hyperparameters
    θₙ = @view θ[1:4] #non-ergodic parametes
    θₐ = @view θ[5]   #aleatory parameters

    #event ids
    EQID₁ = @view X₁[:,1] #1st set of earthquake ids
    EQID₂ = @view X₂[:,1] #2nd set of earthquake ids

    #evaluate total kernel
    Kₜ  = aleat_bevent_binary(EQID₁, EQID₂, θₐ)
    Kₜ += pathsite_binary(X₁[:,2:5], X₂[:,2:5], θₙ)
    
    return Kₜ
end

"""
    Binary source, path and site kernel function with between event aleatory variability
"""
function sourcepathsite_aleat_binary(X₁::AbstractMatrix{T}, X₂::AbstractMatrix{T},
                                     θ::AbstractVector{T}) where T <: Real
    
    #hyperparameters
    θₙ = @view θ[1:6] #non-ergodic parametes
    θₐ = @view θ[7]   #aleatory parameters

    #event ids
    EQID₁ = @view X₁[:,1] #1st set of earthquake ids
    EQID₂ = @view X₂[:,1] #2nd set of earthquake ids

    #evaluate total kernel
    Kₜ  = aleat_bevent_binary(EQID₁, EQID₂, θₐ)
    Kₜ += sourcepathsite_binary(X₁[:,2:5], X₂[:,2:5], θₙ)
    
    return Kₜ
end
