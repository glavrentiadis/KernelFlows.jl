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
export MVGPModel, update_MVGPModel!, trim_MVGP_data


"""Multivariate GP for multivariate input - multivariate output relations"""
struct MVGPModel{T}
    Ms::Vector{GPModel{T}} # vector of GPModels
    DY::DimRedStruct{T}     # DimRedStruct for outputs
end


include("multivariate_training.jl")
include("multivariate_prediction.jl")


"""Updates MVGPmodel M by calling update_GPModel!()."""
function update_MVGPModel!(MVM::MVGPModel{T};
                           newΛ::Union{Nothing, Matrix{T}} = nothing,
                           newΨ::Union{Nothing, Matrix{T}} = nothing) where T <: Real

    nZYdims = length(MVM.Ms)
    λs = (newΛ == nothing) ? [nothing for _ ∈ 1:nZYdims] : collect(eachcol(newΛ))
    θs = (newΨ == nothing) ? [nothing for _ ∈ 1:nZYdims] : collect(eachcol(newΨ))

    Threads.@threads  for (i,M) ∈ collect(enumerate(MVM.Ms))
        update_GPModel!(M; newλ = λs[i], newθ = θs[i])
    end

    MVM
end


function MVGPModel(X_tr::Matrix{T},  # training inputs, with data in rows
                   Y_tr::Matrix{T},  # training outputs, data in rows
                   kernel::Function, # same RBF kernel for all GPModels
                   DXs::Vector{DimRedStruct{T}}, # DX for each output dim
                   DY::DimRedStruct{T};
                   Λ::Union{Nothing, Matrix{T}} = nothing, # scaling parameters for input dimensions
                   Ψ::Union{Nothing, Matrix{T}} = nothing, # kernel paramaters, θ in rows

                   transform_zy::Bool = false) where T <: Real

    ntr = size(X_tr)[1]
    buf1 = zeros(ntr, ntr)
    buf2 = zeros(ntr, ntr)

    ZY_tr = original_to_reduced(Y_tr, DY)
    nZYdims = size(ZY_tr)[2]
    @assert (length(DXs) == nZYdims) "DXs length and nZYdims do not match!"
    nXdims = length(DXs[1].F.values)
    λs = (Λ == nothing) ? [nothing for _ ∈ 1:nZYdims] : collect(eachrow(Λ))
    θs = (Ψ == nothing) ? [nothing for _ ∈ 1:nZYdims] : collect(eachrow(Ψ))

    Ms = [GPModel(X_tr, ZY_tr[:,i], kernel, DXs[i];
                  λ = λs[i], θ = θs[i], transform_zy) for i ∈ 1:nZYdims]

    return MVGPModel(Ms, DY)
end


"""Version of MVGPModel where the same DX is used for all output
dimensions"""
function MVGPModel(X_tr::Matrix{T}, Y_tr::Matrix{T}, kernel::Function,
                   DX::DimRedStruct{T}, DY::DimRedStruct{T};
                   Λ::Union{Nothing, Matrix{T}} = nothing,
                   Ψ::Union{Nothing, Matrix{T}} = nothing,
                   transform_zy::Bool = true) where T <: Real

    DXs = [DX for _ ∈ DY.F.values]
    MVGPModel(X_tr, Y_tr, kernel, DXs, DY; Λ, Ψ, transform_zy)
end


"""From MVGP take only observations described by index vector s. Returns an entirely new MVGP object."""
function trim_MVGP_data(MVM::MVGPModel{T}, s::AbstractVector{Int}) where T <: Real
    ntr = length(s)
    Ms = [GPModel(M.ζ[s], zeros(ntr), M.Z[s,:], M.DX, M.λ[:], M.θ[:],
                  M.kernel, M.zytransf, M.zyinvtransf) for M ∈ MVM.Ms]
    MVM_new = MVGPModel(Ms, MVM.DY)
    update_MVGPModel!(MVM_new)
end


# function ZYtr_from_MVGP(MVM::MVGPModel{T}) where T <: Real
#     # hcat([M.zyinvtransf.(M.ζ) for M ∈ MVM.Ms]...)
#     hcat([M.zyinvtransf.(M.ζ) for M ∈ MVM.Ms]...)
# end

# function Ytr_from_MVGP(MVM::MVGPModel{T}) where T <: Real
#     ZY_tr = ZYtr_from_MVGP(MVM)
#     reduced_to_original(ZY_tr, MVM.DY)
# end

# """Assumes that all dimensions use the same training data"""
# function Xtr_from_MVGP(MVM::MVGPModel{T}) where T <: Real
#     reduced_to_original(MVM.Ms[1].Z ./ MVM.Ms[1].λ', MVM.Ms[1].DX)
# end

# """Does not that all dimensions use the same training data"""
# function Xtr_from_MVGP_multidim(MVM::MVGPModel{T}) where T <: Real
#     [reduced_to_original(M.Z ./ M.λ, M.DX) for M ∈ MVM.Ms]
# end

# function Λ_from_MVGP(MVM::MVGPModel{T}) where T <: Real
#     hcat([M.λ for M ∈ MVM.Ms]...)
# end

# function Ψ_from_MVGP(MVM::MVGPModel{T}) where T <: Real
#     hcat([M.θ for M ∈ MVM.Ms]...)
# end

# function DXs_from_MVGP(MVM::MVGPModel{T}) where T <: Real
#     [M.DX for M ∈ MVM.Ms]
# end
