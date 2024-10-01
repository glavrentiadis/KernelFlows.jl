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
#
using KernelFunctions 

"""Linear kernel (with mean) for testing BinaryKernel
correctness. This kernel also includes the mean, which is given as the
log of the θ, since θ are always positive. θ[1] is the weight of the
kernel, and θ[end] is the weight of the nugget (this is always the
case). Therefore we have n+2 parameters for this kernel, with n the
number of input dimensions."""
function linear_mean_binary(x1::AbstractVector{T}, x2::AbstractVector{T},
                            θ::AbstractVector{T})::T where T <: Real
    # Note that kernel_matrix_...() functions do not pass along the
    # last entry of θ, as that's always the nugget. Hence the θ here
    # is only n+1 entries long, not n+1 like in BinaryKernel.θ_start
    
    μ = @views log.(θ[2:end])

    return θ[1] * (x1 - μ)' * (x2 - μ)
end

function linear_mean_binary(X1::AbstractMatrix{T}, X2::AbstractMatrix{T},
                            θ::AbstractVector{T})::AbstractMatrix{T} where T <: Real

    #mean
    μ = @views log.(θ[2:end])

    #evaluate kernel matrix
    return linear_binary(X1 .- μ, X2 .- μ, θ[2:end])
end

"""Binary linear binary kernel, but without mean"""
function linear_binary(x1::AbstractVector{T}, x2::AbstractVector{T},
                       θ::AbstractVector{T})::T where T <: Real
    
    return θ[1] * x1' * x2
end

function linear_binary(X1::AbstractMatrix{T}, X2::AbstractMatrix{T},
                       θ::AbstractVector{T})::AbstractMatrix{T} where T <: Real

    #hyperparameters
    σ² = @views θ[1]

    #define kernel function 
    k = σ² * LinearKernel()

    #evaluate kernel matrix
    return kernelmatrix(k, RowVecs(X1), RowVecs(X2)) 
end

"""Binary group kernel, with a correlation scale σ"""
function group_binary(x1::Union{AbstractMatrix{T}, T}, x2::Union{AbstractMatrix{T}, T}, 
                     θ::AbstractVector{T}; δ=1e-6)::T where T <: Real
    
    #evaluate kernel matrix
    return norm(x1 .- x2) < δ ? θ : 0.
end

function group_binary(X1::Union{AbstractVector{T}, T}, X2::Union{AbstractVector{T}, T}, 
                      θ::AbstractVector{T}; δ=1e-6)::AbstractMatrix{T} where T <: Real

    #hyperparameters
    σ² = @views θ[1]

    #define kernel function 
    k = σ² * WhiteKernel()

    #evaluate kernel matrix
    return kernelmatrix(k, RowVecs(X1), RowVecs(X2)) 
end

"""Binary exponential kernel, with a correlation scale σ and lenght λ parameters"""
function spherical_exp_binary(x1::AbstractVector{T}, x2::AbstractVector{T},
                              θ::AbstractVector{T})::T where T <: Real

    #hyperparameters
    σ² = @views θ[1]
    λ  = @views θ[2]

    #evaluate kernel matrix
    return σ² * exp(-λ * norm(x1 .- x2) )
end


function spherical_exp_binary(X1::AbstractMatrix{T}, X2::AbstractMatrix{T},
                              θ::AbstractVector{T})::AbstractMatrix{T} where T <: Real

    #hyperparameters
    σ² = @views θ[1]
    λ  = @views θ[2]

    #define kernel function 
    k = σ² * ExponentialKernel(; metric=Euclidean()) ∘ ScaleTransform(λ)

    #evaluate kernel matrix
    return kernelmatrix(k, RowVecs(X1), RowVecs(X2))
end

#add kernels specific to ngmm
include("kernel_functions_binary_seismic.jl")
