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
using Distributions
using BaryRational


struct TransfSpec{T}
    μ::Vector{T}
    σ::Vector{T}
    minim::Vector{T}
    ϵ::T
    deg::Int
end


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


function meanscale(X::AbstractMatrix{T}, spec::TransfSpec{T}) where T <: Real
    (X .- spec.μ') ./ spec.σ'
end


function posscale(X::AbstractMatrix{T}, spec::TransfSpec{T}) where T <: Real
    Z = (X .- spec.minim') ./ spec.σ' * 5 .+ spec.ϵ
end


"""Transform data and construct TransfSpec"""
function standard_transformations(X::AbstractMatrix{T}; deg = 2, ϵ = 1e-2) where T <: Real
    μ = mean(X, dims=1)[:]
    σ = std(X, dims=1)[:]
    minim = minimum(X, dims=1)[:]
    spec = TransfSpec(μ, σ, minim, ϵ, deg)

    return standard_transformations(X, spec), spec
end


"""Transform inputs using a pre-computed TransfSpec"""
function standard_transformations(X::AbstractMatrix{T}, spec::TransfSpec{T}) where T <: Real
    h = posscale(X, spec)
    h[h .< 1e-12] .= 1e-12 # must ensure positivity
    Z = hcat(meanscale(X, spec), sqrt.(h), log.(h))
    Z = polyexpand(Z, spec.deg)
    Z[:, 1:size(X)[2]] .= X
    Z
end


function standard_transformations(x::AbstractVector{T}, spec::TransfSpec{T}) where T <: Real
    standard_transformations(reshape(X, (1, length(x))), spec)
end


"""Return transforms that make Y roughly into standard normal, and
back. Uses (inverse) logcdf transform from Distributions package to
achieve this, along with barycentric rational interpolation, which
seems to work really well.  iqr is the interquantile range of the CDF
that we fit to data. If the function fails, use a smaller
interquantile range, like 0.98."""
function gaussianize_transforms(y::Vector{T}; iqr = .99) where T <: Real

    ys = sort(y)
    no = Normal()
    ny = length(ys)

    # Don't do transformation at the very edges, the transformations
    # and inverse transformations become unstable for outliers.
    quants = collect(0:ny-1) ./ (ny-1) .* iqr .+ (1. - iqr)/2

    idx = [1]
    y0 = ys[1]

    # use max 1e4 points. Generally more points work better for this method
    for (i,yy) ∈ enumerate(ys)
        if yy - y0 > 1e-4  * (ys[end] - ys[1]) || i == ny
            push!(idx, i)
            y0 = yy
        end
    end

    tr = FHInterp(ys[idx], quants[idx]; order = 1, grid = false)
    transf(x) = invlogcdf(no, log(tr(x)))
    tri = FHInterp(quants[idx], ys[idx]; order = 1, grid = false)
    invtransf(x) = tri(exp(logcdf(no, x)))

    # display(transf.(y))
    maxerr = maximum(abs.(y - invtransf.(transf.(y))))
    println("Max absolute error in 2-way transformation: $maxerr")

    return transf, invtransf
end
