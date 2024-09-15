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
export get_kernels

abstract type Kernel end
abstract type AutodiffKernel <: Kernel end

mutable struct UnaryKernel{T} <: AutodiffKernel
    k::Function # kernel function
    θ_start::Vector{T}
    nXlinear::Int
end

mutable struct BinaryKernel{T} <: AutodiffKernel
    k::Function # kernel function
    θ_start::Vector{T}
end

mutable struct BinaryVectorizedKernel{T} <: AutodiffKernel
    k::Function # kernel function
    θ_start::Vector{T}
end

"""Kernel to be used without autodiff"""
mutable struct AnalyticKernel{T} <: Kernel
    K_and_∂K∂logα!::Function # returns K and its gradients
    θ_start::Vector{T}
end

include("kernel_functions_unary.jl")
include("kernel_functions_binary.jl")
include("kernel_functions_analytic.jl")


function get_UnaryKernel(s::Symbol, G::GPGeometry{T}) where T <: Real
    d = Dict(:spherical_sqexp => spherical_sqexp,
             :spherical_exp => spherical_exp,
             :inverse_quadratic => inverse_quadratic,
             :Matern32 => Matern32,
             :Matern52 => Matern52)
    # Initial θ for UnaryKernels
    θ₀_U = T.(exp.([0., 0., -3., -7.]))
    return [UnaryKernel(d[s], θ₀_U, length(XP.values)) for XP in G.Xprojs]
end

function get_BinaryKernel(s::Symbol, G::GPGeometry{T}) where T <: Real
    d = Dict(:linear_binary        => linear_binary,
             :linear_mean_binary   => linear_mean_binary,
             :spherical_exp_binary => spherical_exp_binary)
    if s  == :linear_binary
        θ₀_B = [exp.([0., -7.]) for XP in G.Xprojs]
    elseif s == :linear_mean_binary
        # get number of transformed X dims, plus nugget and weight
        θ₀_B = [ones(T, length(XP.spec.sparsedims) + 2) for XP in G.Xprojs]
        for θ in θ₀list
            θ[end] = exp(-7.0)
        end
    else
        θ₀_B = T.(exp.([0., 0., -7.]))
    end
    return [BinaryKernel(d[s], θ₀_B) for XP in G.Xprojs]
end

function get_VectBinaryKernel(s::Symbol, G::GPGeometry{T}) where T <: Real
    d = Dict(:linear_binary        => linear_binary,
             :linear_mean_binary   => linear_mean_binary,
             :spherical_exp_binary => spherical_exp_binary)
    if s  == :linear_binary
        θ₀_B = [exp.([0., -7.]) for XP in G.Xprojs]
    elseif s == :linear_mean_binary
        # get number of transformed X dims, plus nugget and weight
        θ₀_B = [ones(T, length(XP.spec.sparsedims) + 2) for XP in G.Xprojs]
        for θ in θ₀list
            θ[end] = exp(-7.0)
        end
    else
        θ₀_B = T.(exp.([0., 0., -7.]))
    end
    return [BinaryVectorizedKernel(d[s], θ₀_B) for XP in G.Xprojs]
end

function get_AnalyticKernel(s::Symbol, G::GPGeometry{T}) where T <: Real
    d = Dict(:Matern32_analytic => Matern32_αgrad!)
    θ₀_A = T.(exp.([0., 0., -4., -7.]))
    return [AnalyticKernel(d[s], θ₀_A) for XP in G.Xprojs]
end

function get_MVGP_kernels(s::Symbol, G::GPGeometry{T}) where T <: Real

    unary_kernels = [:spherical_sqexp, :spherical_exp,
                     :Matern32, :Matern52, :inverse_quadratic]
    binary_vkernels = [:linear_binary, :linear_mean_binary, 
                       :spherical_exp_binary, :group_binary]
    binary_kernels = [:linear_binary, :linear_mean_binary, 
                      :spherical_exp_binary, :group_binary]
    analytic_kernels = [:Matern32_analytic]

    s in unary_kernels    && (return get_UnaryKernel(s,G))
    s in binary_vkernels  && (return get_VectBinaryKernel(s,G))
    s in binary_kernels   && (return get_BinaryKernel(s,G))
    s in analytic_kernels && (return get_AnalyticKernel(s,G))
end
