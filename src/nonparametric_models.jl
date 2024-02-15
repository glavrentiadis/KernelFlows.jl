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
"""Nonparametric Kernel Flows model. The warpings is a vector (over
all output dimensions) of vectors (over warping segments) of
individual warpigns. B_orig contains the base kernel with original
data, and B_warped is the same but with training inputs warped."""
struct NonParametricMVGPModel{T}
    warpings::Vector{Vector{MVGPModel{T}}}  # vector of MVGPModels for warping
    B_orig::MVGPModel{T}  # base GP with original inputs (for reference)
    B_warped::MVGPModel{T}  # base GP with warped inputs
end


include("nonparametric_plots.jl")
include("nonparametric_training.jl")
include("nonparametric_predict.jl")


function NonParametricMVGPModel(X_tr::Matrix{T},  # training inputs, with data in rows
                                Y_tr::Matrix{T},  # training outputs, data in rows
                                kernel::Symbol,   # same RBF kernel for all GPModels
                                G::GPGeometry{T};
                                Λ::Union{Nothing, Matrix{T}} = nothing,
                                Ψ::Union{Nothing, Matrix{T}} = nothing,
                                transform_zy::Bool = false) where T <: Real

    B_orig = MVGPModel(X_tr, Y_tr, kernel, G; Λ, Ψ, transform_zy)
    B_warped = MVGPModel(similar(X_tr), Y_tr, kernel, G; Λ, Ψ, transform_zy)

    nGPs = length(B_orig.Ms)
    warpings = Vector{Vector{MVGPModel{T}}}(undef, nGPs)
    NonParametricMVGPModel(warpings, B_orig, B_warped)
end
