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
export MVGPModel, update_MVGPModel!, trim_MVGP_data, remove_extrapolations


"""Multivariate GP for multivariate input - multivariate output relations"""
struct MVGPModel{T}
    Ms::Vector{GPModel{T}} # vector of GPModels
    G::GPGeometry{T}       # dimension reduction/augmentation spec
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

    parallel = length(MVM.Ms[1].ζ) < 5001
    if parallel
        print("\rUpdating all $(length(MVM.Ms)) GPs in parallel...")
        Threads.@threads :static for (i,M) ∈ collect(enumerate(MVM.Ms))
            update_GPModel!(M; newλ = λs[i], newθ = θs[i])
        end
    else
        print("\rUpdating all $(length(MVM.Ms)) GPs...")
        for (i,M) ∈ collect(enumerate(MVM.Ms))
            update_GPModel!(M; newλ = λs[i], newθ = θs[i])
        end
    end
    println("done!")
    MVM
end


function MVGPModel(X_tr::Matrix{T},  # training inputs, with data in rows
                   Y_tr::Matrix{T},  # training outputs, data in rows
                   kernel::Symbol,   # same kernel for all GPModels
                   G::GPGeometry{T}; # input-output mapping geometry
                   Λ::Union{Nothing, Matrix{T}} = nothing, # scaling parameters for input dimensions
                   Ψ::Union{Nothing, Matrix{T}} = nothing, # kernel paramaters, θ in rows

                   transform_zy::Bool = false) where T <: Real

    ZY_tr = reduce_Y(Y_tr, G)
    nZYdims = size(ZY_tr)[2]

    kernels = get_MVGP_kernels(kernel, G)

    λs = (Λ == nothing) ? [nothing for _ ∈ 1:nZYdims] : collect(eachrow(Λ))
    θs = (Ψ == nothing) ? [nothing for _ ∈ 1:nZYdims] : collect(eachrow(Ψ))

    Ms = [GPModel(reduce_X(X_tr, G, i), ZY_tr[:,i], kernels[i];
                  λ = λs[i], θ = θs[i], transform_zy) for i ∈ 1:nZYdims]

    return MVGPModel(Ms, G)
end



"""Apply standard transformations and dimension reduction as described
in GPGeometry object in G. This function scales the inputs according
to learned kernel parameters. Use this function to produce inputs that
correspond to values in matrix GPModel.Z."""
function reduce_X(X::AbstractMatrix{T}, MVM::MVGPModel{T}, i::Int) where T <: Real
    reduce_X(X, MVM.G, i) .* MVM.Ms[i].λ'
end


"""From MVGP take only observations described by index vector s. Returns an entirely new MVGP object."""
function trim_MVGP_data(MVM::MVGPModel{T}, s::AbstractVector{Int}) where T <: Real
    ntr = length(s)
    Ms = [GPModel(M.ζ[s], zeros(ntr), M.Z[s,:], M.λ[:], M.θ[:],
                  M.kernel, M.zytransf, M.zyinvtransf, T[], Vector{T}[], Vector{T}[]) for M ∈ MVM.Ms]
    MVM_new = MVGPModel(Ms, MVM.G)
    update_MVGPModel!(MVM_new)
end


# """Get the standard number of X dimensions for linear kernel"""
# function nXl(G::GPGeometry{T}, i::Int) where T <: Real
#     spec = G.Xprojs[i].spec
#     nXl_max = spec.nCCA == 0 ? spec.ndummy : spec.nCCA
#     sum(spec.sparsedims .<= nXl_max)
# end


function sparsify_inputs(MVM::MVGPModel{T}, nleave::Int) where T <: Real
    Ms_new = GPModel{T}[]
    G_new = deepcopy(MVM.G)

    for (i,M) in enumerate(MVM.Ms)
        newM, newdims = sparsify_inputs(M, nleave)
        push!(Ms_new, newM)
        keepat!(G_new.Xprojs[i].spec.sparsedims, newdims)
    end

    MVGPModel(Ms_new, G_new)
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
