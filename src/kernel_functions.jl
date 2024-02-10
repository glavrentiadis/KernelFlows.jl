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

mutable struct UnaryKernel{T} <: Kernel
    k::Function # kernel function
    θ_start::Vector{T}
    nXlinear::Int
end

mutable struct BinaryKernel{T} <: Kernel
    k::Function # kernel function
    θ_start::Vector{T}
end


# For UnaryKernel, first parameter (a / θ[1]) is always weight of the
# component, second (b / θ[2]) is the length scale.

spherical_sqexp(d::T, a::T, b::T) where T <: Real = a * exp(T(-.5)*d*d / b)
spherical_sqexp(d::T; θ::AbstractVector{T}) where T <: Real = spherical_sqexp(d, θ[1], θ[2])

spherical_exp(d::T, a::T, b::T) where T <: Real = a * exp(-d / b)
spherical_exp(d::T; θ::AbstractVector{T}) where T <: Real = spherical_exp(d, θ[1], θ[2])

function Matern32(d::T, a::T, b::T) where T <: Real
    h = sqrt(T(3.)) * d / b # d is Euclidean distance
    a * (one(T) + h) * exp(-h)
end
Matern32(d::T; θ::AbstractVector{T}) where T <: Real = Matern32(d, θ[1], θ[2])

function Matern52(d::T, a::T, b::T) where T <: Real
    h = sqrt(T(5.)) * d / b
    a * (T(1.) + h + h^2 / T(3.)) * exp(-h)
end
Matern52(d::T; θ::AbstractVector{T}) where T <: Real = Matern52(d, θ[1], θ[2])

function inverse_quadratic(d::T, a::T, b::T) where T <: Real
    a / sqrt(d^2 + b)
end
inverse_quadratic(d::T; θ::AbstractVector{T}) where T <: Real = inverse_quadratic(d, θ[1], θ[2])


"""Linear kernel (with mean) for testing BinaryKernel
correctness. This kernel also includes the mean, which is given as the
log of the θ, since θ are always positive. θ[1] is the weight of the
kernel, and θ[end] is the weight of the nugget (this is always the
case). Therefore we have n+2 parameters for this kernel, with n the
number of input dimensions."""
function linear_mean(x1::AbstractVector{T}, x2::AbstractVector{T},
                     θ::AbstractVector{T}) where T <: Real
    # Note that kernel_matrix_...() functions do not pass along the
    # last entry of θ, as that's always the nugget. Hence the θ here
    # is only n+1 entries long, not n+1 like in BinaryKernel.θ_start
    μ = log.(θ[2:end])

    θ[1] * (x1 - μ)' * (x2 - μ)
end

"""Linear binary kernel, but without mean"""
function linear(x1::AbstractVector{T}, x2::AbstractVector{T},
                θ::AbstractVector{T}) where T <: Real
    θ[1] * x1' * x2
end


function get_MVGP_kernels(s::Symbol, G::GPGeometry{T}) where T <: Real

    unary_kernels =  [:spherical_sqexp, :spherical_exp,
                      :Matern32, :Matern52, :inverse_quadratic]
    binary_kernels = [:linear, :linear_mean]

    d = Dict(:spherical_sqexp => spherical_sqexp,
             :spherical_exp => spherical_exp,
             :inverse_quadratic => inverse_quadratic,
             :Matern32 => Matern32, :Matern52 => Matern52,
             :linear => linear, :linear_mean => linear_mean)

    # Function for getting initial θ for BinaryKernels
    function get_binary_θs(s::Symbol, G::GPGeometry{T}) where T <: Real
        if s  == :linear
            θ₀list = [exp.([0., -7.]) for XP in G.Xprojs]
        elseif s == :linear_mean
            # get number of transformed X dims, plus nugget and weight
            nθs = [length(XP.spec.sparsedims) + 2 for XP in G.Xprojs]
            θ₀list = [ones(T, nθ) for nθ in nθs]
            for θ in θ₀list
                θ[end] = exp(-7.0)
            end
        end

        return θ₀list
    end

    if s in unary_kernels
        θ₀_U = T.(exp.([0., 0., -4., -7.])) # initial θ for UnaryKernels
        klist = [UnaryKernel(d[s], θ₀_U, XP.spec.nCCA) for XP in G.Xprojs]
    elseif s in binary_kernels
        θ₀s = get_binary_θs(s, G)
        klist = [BinaryKernel(d[s], θ₀) for θ₀ in θ₀s]
    end

    return klist
end
