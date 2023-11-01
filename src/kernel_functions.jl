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
export spherical_sqexp, spherical_exp, Matern32, Matern52

# Parameters: first is always weight of the component, second is the
# length scale.

function spherical_sqexp(d::T; θ::AbstractVector) where T <: Real
    # Assume that dimensions have been scaled before, so that only a
    # spherical kernel will be needed. d is the Euclidean distance.
    θ[1] * exp(-.5d^2 / θ[2])
end

function spherical_exp(d::T; θ::AbstractVector{T}) where T <: Real
    return exp(logθ[1]) * exp(-d / exp(logθ[2]))
end

function Matern32(d::T; θ::AbstractVector{T}) where T <: Real
    h = 3d / θ[2] # d is Euclidean distance
    θ[1] * (1. + h) * exp(-h)
end

function Matern52(d::T; θ::AbstractVector{T}) where T <: Real
    h = 5d / θ[2]
    θ[1] * (1. + h + h^2 / 3) * exp(-h)
end
