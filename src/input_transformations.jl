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
export polyexpand, standard_transformations


using StatsBase
using Combinatorics


function polyexpand(X::AbstractMatrix{T}, degs::AbstractVector{Int}) where T <: Real
    ndata, ncols = size(X)
    all_combs = vcat([collect(with_replacement_combinations(1:ncols, d)) for d in degs]...)
    ncols_new = length(all_combs)
    if ncols_new > 10000
        println("ERROR: too many columns: $ncols_new")
        return
    end

    X_new = zeros(ndata, ncols_new)
    for (i,co) ∈ enumerate(all_combs)
        X_new[:,i] .= X[:,co[1]]
        for d ∈ co[2:end]
            X_new[:,i] .*= X[:,d]
        end
    end
    return X_new
end

function polyexpand(X::AbstractMatrix{T}, deg::Int) where T <: Real
    polyexpand(X, 1:deg)
end

function posscale(X::AbstractMatrix{T}; ϵ = 1e-3) where T <: Real
    Z = 5 .*X ./ std(X, dims = 1)
    Z .-= minimum(Z, dims = 1) .- ϵ
end

function meanscale(X::AbstractMatrix{T}; ϵ = 1e-3) where T <: Real
    Z = X ./ std(X, dims = 1)
    Z .-= mean(Z, dims = 1)
end

function transf1(X::AbstractMatrix{T}) where T <: Real
    sqrt.(posscale(X))
end

function transf2(X::AbstractMatrix{T}) where T <: Real
    log.(posscale(X))
end

function standard_transformations(X::AbstractMatrix{T}; deg = 2) where T <: Real
    Z = hcat(meanscale(X), transf1(X), transf2(X))
    Z = polyexpand(Z, deg)
    Z[:, 1:size(X)[2]] .= X
    Z
end

function standard_transformations(X::AbstractVector{T}; deg = 2) where T <: Real
    Z = standard_transforms(reshape(X, (length(X), 1)))
end
