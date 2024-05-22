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
