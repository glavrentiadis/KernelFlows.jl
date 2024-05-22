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
