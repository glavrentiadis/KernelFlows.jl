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
    τ₀² = @views θ[1] #between event variance
    
    #evaluate kernel for between event residuals
    return group_binary(EQID1, EQID2, τ₀²; δ=1e-6)
end

# Individual Non-ergodic Kernels
# ---   ---   ---   ---   ---
"""
    Binary source exponential kernel function
"""
function source_exp_binary(X₁ₗ::AbstractMatrix{T}, X₂ₗ::AbstractMatrix{T},
                           θₗ::AbstractVector{T}) where T <: Real
    
    #non-ergodic source
    return spherical_exp_binary(X₁ₗ,X₂ₗ,θₗ)
end

"""
    Binary source Matern kernel function
"""
function source_matern_binary(X₁ₗ::AbstractMatrix{T}, X₂ₗ::AbstractMatrix{T},
                              θₗ::AbstractVector{T}) where T <: Real
    
    #non-ergodic source
    return spherical_matern_binary(X₁ₗ,X₂ₗ,θₗ)
end

"""
    Binary site exponential kernel function
"""
function site_exp_binary(X₁ₛ::AbstractMatrix{T}, X₂ₛ::AbstractMatrix{T},
                         θₛ::AbstractVector{T}) where T <: Real

    #non-ergodic site
    return spherical_exp_binary(X₁ₛ,X₂ₛ,θₛ)
end

"""
    Binary site Matern kernel function
"""
function site_matern_binary(X₁ₛ::AbstractMatrix{T}, X₂ₛ::AbstractMatrix{T},
                            θₛ::AbstractVector{T}) where T <: Real

    #non-ergodic site
    return spherical_matern_binary(X₁ₛ,X₂ₛ,θₛ)
end

#geometrical spreading exponential kernel
n_integ_pt=5
flag_normalize=true
GSPathExpKernel = PathKernel(n_integ_pt, 
                             KernelFunctions.ExponentialKernel( ;metric=KernelFunctions.Euclidean()), 
                             flag_normalize)

#geometrical spreading Matern kernel
n_integ_pt=5
flag_normalize=true
t, s, w = gaussquad2d(n_integ_pt) #compute integration weights
GSPathMaternKernel = PathKernel(n_integ_pt, 
                                KernelFunctions.Matern32Kernel( ;metric=KernelFunctions.Euclidean()), 
                                flag_normalize) 

#geometrical spreading exponential kernel
n_integ_pt=5
flag_normalize=false
GSPathExpKernelInteg = PathKernel(n_integ_pt, 
                                  KernelFunctions.ExponentialKernel( ;metric=KernelFunctions.Euclidean()), 
                                  flag_normalize)

#geometrical spreading Matern kernel
n_integ_pt=5
flag_normalize=false
t, s, w = gaussquad2d(n_integ_pt) #compute integration weights
GSPathMaternKernelInteg = PathKernel(n_integ_pt, 
                                     KernelFunctions.Matern32Kernel( ;metric=KernelFunctions.Euclidean()), 
                                     flag_normalize) 

"""
    Binary path kernel function
"""
function path_binary(X₁::AbstractMatrix{T}, X₂::AbstractMatrix{T}, θₚ::AbstractVector{T},
                     PKernel) where T <: Real

    #hyperparameters
    ωₚ² = θₚ[1]
    λₚ  = θₚ[2]

    #define path kernel
    κₚ = ωₚ² * PKernel ∘ KernelFunctions.ScaleTransform(λₚ)

    #evaluate kernel matrix
    n1, n2 = size(X₁, 1), size(X₂, 1)
    K_buffer = Zygote.Buffer(Matrix{T}(undef, n1, n2))

    @inbounds for i in 1:n1
        x = @view X₁[i, :]
        @inbounds for j in 1:n2
            y = @view X₂[j, :]
            K_buffer[i, j] = κₚ(x, y)
        end
    end

    return copy(K_buffer)
    # return KernelFunctions.kernelmatrix(κₚ, RowVecs(X₁), RowVecs(X₂))
end

"""
    Binary path exponential kernel function
"""
function path_exp_binary(X₁::AbstractMatrix{T}, X₂::AbstractMatrix{T}, θₚ::AbstractVector{T}) where T <: Real
    
    #evaluate path kernel
    return path_binary(X₁, X₂, θₚ, GSPathExpKernel)
end

"""
    Binary path Matern kernel function
"""
function path_matern_binary(X₁::AbstractMatrix{T}, X₂::AbstractMatrix{T}, θₚ::AbstractVector{T}) where T <: Real
    
    #evaluate path kernel
    return path_binary(X₁, X₂, θₚ, GSPathMaternKernel)
end

"""
    Binary path exponential kernel integral function
"""
function intpath_exp_binary(X₁::AbstractMatrix{T}, X₂::AbstractMatrix{T}, 
                            θₚ::AbstractVector{T}) where T <: Real
    
    #evaluate path kernel
    return path_binary(X₁, X₂, θₚ, GSPathExpKernelInteg)
end

"""
    Binary path Matern kernel integral function
"""
function intpath_matern_binary(X₁::AbstractMatrix{T}, X₂::AbstractMatrix{T}, 
                               θₚ::AbstractVector{T}) where T <: Real
    
    #evaluate path kernel
    return path_binary(X₁, X₂, θₚ, GSPathMaternKernelInteg)
end

# Composite Non-ergodic Kernels
# ---   ---   ---   ---   ---
"""
    Binary source & site exponential kernel function
"""
function sourcesite_exp_binary(X₁::AbstractMatrix{T}, X₂::AbstractMatrix{T},
                               θ::AbstractVector{T}) where T <: Real
    
    #hyperparameters
    θₗ = @views θ[1:2] #source parametes
    θₛ = @views θ[3:4] #site parameters

    #coordinate dimension size
    d = div(size(X₁)[2], 2) 

    #evaluate total kernel
    Kₜ  = @views source_exp_binary(X₁[:,1:d],     X₂[:,1:d],       θₗ)
    Kₜ += @views site_exp_binary(X₁[:,(d+1):end], X₂[:,(d+1):end], θₛ)
    
    return Kₜ
end

"""
    Binary source & site Matern kernel function
"""
function sourcesite_matern_binary(X₁::AbstractMatrix{T}, X₂::AbstractMatrix{T},
                                  θ::AbstractVector{T}) where T <: Real
    
    #hyperparameters
    θₗ = @views θ[1:2] #source parametes
    θₛ = @views θ[3:4] #site parameters

    #coordinate dimension size
    d = div(size(X₁)[2], 2) 

    #evaluate total kernel
    Kₜ  = @views source_matern_binary(X₁[:,1:d],     X₂[:,1:d],       θₗ)
    Kₜ += @views site_matern_binary(X₁[:,(d+1):end], X₂[:,(d+1):end], θₛ)
    
    return Kₜ
end

"""
    Binary path & site exponential kernel function
"""
function pathsite_exp_binary(X₁::AbstractMatrix{T}, X₂::AbstractMatrix{T},
                             θ::AbstractVector{T}) where T <: Real
    
    #hyperparameters
    θₚ = @views θ[1:2] #path parametes
    θₛ = @views θ[3:4] #site parameters

    #coordinate dimension size
    d = div(size(X₁)[2], 2) 

    #evaluate total kernel
    Kₜ  = path_binary(X₁,  X₂,  θₚ)
    Kₜ += @views site_exp_binary(X₁[:,(d+1):end], X₂[:,(d+1):end], θₛ)
    
    return Kₜ
end

"""
    Binary path & site Matern kernel function
"""
function pathsite_matern_binary(X₁::AbstractMatrix{T}, X₂::AbstractMatrix{T},
                                θ::AbstractVector{T}) where T <: Real
    
    #hyperparameters
    θₚ = @views θ[1:2] #path parametes
    θₛ = @views θ[3:4] #site parameters

    #coordinate dimension size
    d = div(size(X₁)[2], 2) 

    #evaluate total kernel
    Kₜ  = path_matern_binary(X₁,  X₂,  θₚ)
    Kₜ += @views site_matern_binary(X₁[:,(d+1):end], X₂[:,(d+1):end], θₛ)
    
    return Kₜ
end

"""
    Binary source, path & site exponential kernel function
"""
function sourcepathsite_exp_binary(X₁::AbstractMatrix{T}, X₂::AbstractMatrix{T},
                                   θ::AbstractVector{T}) where T <: Real
    
    #hyperparameters
    θₗ = @views θ[1:2] #source parametes
    θₚ = @views θ[3:4] #path parametes
    θₛ = @views θ[5:6] #site parameters

    #coordinate dimension size
    d = div(size(X₁)[2], 2) 

    #evaluate total kernel
    Kₜ  = @views source_binary(X₁[:,1:d],     X₂[:,1:d],       θₗ)
    Kₜ += @views path_exp_binary(X₁,              X₂,              θₚ)
    Kₜ += @views site_exp_binary(X₁[:,(d+1):end], X₂[:,(d+1):end], θₛ)
    
    return Kₜ
end

"""
    Binary source, path & site Matern kernel function
"""
function sourcepathsite_matern_binary(X₁::AbstractMatrix{T}, X₂::AbstractMatrix{T},
                                      θ::AbstractVector{T}) where T <: Real
    
    #hyperparameters
    θₗ = @views θ[1:2] #source parametes
    θₚ = @views θ[3:4] #path parametes
    θₛ = @views θ[5:6] #site parameters

    #coordinate dimension size
    d = div(size(X₁)[2], 2) 

    #evaluate total kernel
    Kₜ  = @views source_matern_binary(X₁[:,1:d],     X₂[:,1:d],       θₗ)
    Kₜ += @views path_matern_binary(X₁,              X₂,              θₚ)
    Kₜ += @views site_matern_binary(X₁[:,(d+1):end], X₂[:,(d+1):end], θₛ)
   
    return Kₜ
end

# Non-ergodic Kernels with Aleat Variability
# ---   ---   ---   ---   ---
"""
    Binary source kernel exponential function with between event aleatory variability
"""
function source_exp_aleat_binary(X₁::AbstractMatrix{T}, X₂::AbstractMatrix{T},
                                 θ::AbstractVector{T}) where T <: Real
    
    #hyperparameters
    θₗ = @views θ[1:2] #source parametes
    θₐ = @views θ[3]   #aleatory parameters

    #evaluate total kernel
    Kₜ  = @views aleat_bevent_binary(X₁[:,1], X₂[:,1], θₐ)
    Kₜ += @views source_exp_binary(X₁[:,2:end], X₂[:,2:end], θₗ)
    
    return Kₜ
end

"""
    Binary source kernel Matern function with between event aleatory variability
"""
function source_matern_aleat_binary(X₁::AbstractMatrix{T}, X₂::AbstractMatrix{T},
                                    θ::AbstractVector{T}) where T <: Real
    
    #hyperparameters
    θₗ = @views θ[1:2] #source parametes
    θₐ = @views θ[3]   #aleatory parameters

    #evaluate total kernel
    Kₜ  = @views aleat_bevent_binary(X₁[:,1], X₂[:,1], θₐ)
    Kₜ += @views source_matern_binary(X₁[:,2:end], X₂[:,2:end], θₗ)
    
    return Kₜ
end

"""
    Binary path exponential kernel function with between event aleatory variability
"""
function path_exp_aleat_binary(X₁::AbstractMatrix{T}, X₂::AbstractMatrix{T},
                               θ::AbstractVector{T}) where T <: Real
    
    #hyperparameters
    θₚ = @view θ[1:2] #path parametes
    θₐ = @view θ[3]   #aleatory parameters

    #evaluate total kernel
    Kₜ  = @views aleat_bevent_binary(X₁[:,1], X₂[:,1], θₐ)
    Kₜ += @views path_exp_binary(X₁[:,2:end], X₂[:,2:end], θₚ)
    
    return Kₜ
end

"""
    Binary path Matern kernel function with between event aleatory variability
"""
function path_matern_aleat_binary(X₁::AbstractMatrix{T}, X₂::AbstractMatrix{T},
                                  θ::AbstractVector{T}) where T <: Real
    
    #hyperparameters
    θₚ = @view θ[1:2] #path parametes
    θₐ = @view θ[3]   #aleatory parameters

    #evaluate total kernel
    Kₜ  = @views aleat_bevent_binary(X₁[:,1], X₂[:,1], θₐ)
    Kₜ += @views path_matern_binary(X₁[:,2:end], X₂[:,2:end], θₚ)
    
    return Kₜ
end

"""
    Binary integral path exponential kernel function with between event aleatory variability
"""
function intpath_exp_aleat_binary(X₁::AbstractMatrix{T}, X₂::AbstractMatrix{T},
                                  θ::AbstractVector{T}) where T <: Real
    
    #hyperparameters
    θₚ = @view θ[1:2] #path parametes
    θₐ = @view θ[3]   #aleatory parameters

    #evaluate total kernel
    Kₜ  = @views aleat_bevent_binary(X₁[:,1], X₂[:,1], θₐ)
    Kₜ += @views intpath_exp_binary(X₁[:,2:end], X₂[:,2:end], θₚ)
    
    return Kₜ
end

"""
    Binary integral path Matern kernel function with between event aleatory variability
"""
function intpath_matern_aleat_binary(X₁::AbstractMatrix{T}, X₂::AbstractMatrix{T},
                                       θ::AbstractVector{T}) where T <: Real
    
    #hyperparameters
    θₚ = @view θ[1:2] #path parametes
    θₐ = @view θ[3]   #aleatory parameters

    #evaluate total kernel
    Kₜ  = @views aleat_bevent_binary(X₁[:,1], X₂[:,1], θₐ)
    Kₜ += @views intpath_matern_binary(X₁[:,2:end], X₂[:,2:end], θₚ)
    
    return Kₜ
end

"""
    Binary exponential site kernel function with between event aleatory variability
"""
function site_exp_aleat_binary(X₁::AbstractMatrix{T}, X₂::AbstractMatrix{T},
                               θ::AbstractVector{T}) where T <: Real
    
    #hyperparameters
    θₛ = @view θ[1:2] #site parametes
    θₐ = @view θ[3]   #aleatory parameters

    #evaluate total kernel
    Kₜ  = @views aleat_bevent_binary(X₁[:,1], X₂[:,1], θₐ)
    Kₜ += @views site_exp_binary(X₁[:,2:end], X₂[:,2:end], θₛ)
    
    return Kₜ
end

"""
    Binary matern site kernel function with between event aleatory variability
"""
function site_matern_aleat_binary(X₁::AbstractMatrix{T}, X₂::AbstractMatrix{T},
                                  θ::AbstractVector{T}) where T <: Real
    
    #hyperparameters
    θₛ = @view θ[1:2] #site parametes
    θₐ = @view θ[3]   #aleatory parameters

    #evaluate total kernel
    Kₜ  = @views aleat_bevent_binary(X₁[:,1], X₂[:,1], θₐ)
    Kₜ += @views site_matern_binary(X₁[:,2:end], X₂[:,2:end], θₛ)
    
    return Kₜ
end

"""
    Binary source and site exponential kernel function with between event aleatory variability
"""
function sourcesite_exp_aleat_binary(X₁::AbstractMatrix{T}, X₂::AbstractMatrix{T},
                                     θ::AbstractVector{T}) where T <: Real
    
    #hyperparameters
    θₙ = @view θ[1:4] #non-ergodic parametes
    θₐ = @view θ[5]   #aleatory parameters

    #evaluate total kernel
    Kₜ  = @views aleat_bevent_binary(X₁[:,1], X₂[:,1], θₐ)
    Kₜ += @views sourcesite_exp_binary(X₁[:,2:end], X₂[:,2:end], θₙ)
    
    return Kₜ
end

"""
    Binary source and site Matern kernel function with between event aleatory variability
"""
function sourcesite_matern_aleat_binary(X₁::AbstractMatrix{T}, X₂::AbstractMatrix{T},
                                        θ::AbstractVector{T}) where T <: Real
    
    #hyperparameters
    θₙ = @view θ[1:4] #non-ergodic parametes
    θₐ = @view θ[5]   #aleatory parameters

    #evaluate total kernel
    Kₜ  = @views aleat_bevent_binary(X₁[:,1], X₂[:,1], θₐ)
    Kₜ += @views sourcesite_matern_binary(X₁[:,2:end], X₂[:,2:end], θₙ)
    
    return Kₜ
end

"""
    Binary path and site exponential kernel function with between event aleatory variability
"""
function pathsite_exp_aleat_binary(X₁::AbstractMatrix{T}, X₂::AbstractMatrix{T},
                                   θ::AbstractVector{T}) where T <: Real
    
    #hyperparameters
    θₙ = @view θ[1:4] #non-ergodic parametes
    θₐ = @view θ[5]   #aleatory parameters

    #evaluate total kernel
    Kₜ  = @views aleat_bevent_binary(X₁[:,1], X₂[:,1], θₐ)
    Kₜ += @views pathsite_exp_binary(X₁[:,2:end], X₂[:,2:end], θₙ)
    
    return Kₜ
end

"""
    Binary path and site Matern kernel function with between event aleatory variability
"""
function pathsite_matern_aleat_binary(X₁::AbstractMatrix{T}, X₂::AbstractMatrix{T},
                                      θ::AbstractVector{T}) where T <: Real
    
    #hyperparameters
    θₙ = @view θ[1:4] #non-ergodic parametes
    θₐ = @view θ[5]   #aleatory parameters

    #evaluate total kernel
    Kₜ  = @views aleat_bevent_binary(X₁[:,1], X₂[:,1], θₐ)
    Kₜ += @views pathsite_matern_binary(X₁[:,2:end], X₂[:,2:end], θₙ)
    
    return Kₜ
end

"""
    Binary source, path and site exponential kernel function with between event aleatory variability
"""
function sourcepathsite_exp_aleat_binary(X₁::AbstractMatrix{T}, X₂::AbstractMatrix{T},
                                         θ::AbstractVector{T}) where T <: Real
    
    #hyperparameters
    θₙ = @view θ[1:6] #non-ergodic parametes
    θₐ = @view θ[7]   #aleatory parameters

    #evaluate total kernel
    Kₜ  = @views aleat_bevent_binary(X₁[:,1], X₂[:,1], θₐ)
    Kₜ += @views sourcepathsite_exp_binary(X₁[:,2:end], X₂[:,2:end], θₙ)
    
    return Kₜ
end

"""
    Binary source, path and site Matern kernel function with between event aleatory variability
"""
function sourcepathsite_matern_aleat_binary(X₁::AbstractMatrix{T}, X₂::AbstractMatrix{T},
                                            θ::AbstractVector{T}) where T <: Real
    
    #hyperparameters
    θₙ = @view θ[1:6] #non-ergodic parametes
    θₐ = @view θ[7]   #aleatory parameters

    #evaluate total kernel
    Kₜ  = @views aleat_bevent_binary(X₁[:,1], X₂[:,1], θₐ)
    Kₜ += @views sourcepathsite_matern_binary(X₁[:,2:end], X₂[:,2:end], θₙ)
    
    return Kₜ
end
