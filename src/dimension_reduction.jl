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

export DimRedStruct, original_to_reduced, reduced_to_original
export dimreduce_basic, dimreduce_PCA, dimreduce_CCA_PCA, dimreduce_CCA_PCA_augmented
export gaussianize_transforms

using Random
using StatsBase
using LinearAlgebra
using Arpack
using Distributions
using BaryRational
# using Polynomials


struct DimRedStruct{T}
    μ::Vector{T} # mean of data before transformation
    # F contains eigenvectors and singular values of data:
    F::NamedTuple{(:vectors, :values), Tuple{Matrix{T}, Vector{T}}}

    nCCA::Int # number of CCA vectors (only used for inputs X, not Y)
end


"""Returns transformed data: Z[i,j] = projection of X[i,:] onto j'th
component in the new representation (e.g. eigenvector of the data)."""
function original_to_reduced(X::AbstractArray{T}, D::DimRedStruct{T};
                             nvecs::Int = length(D.F.values), nomean::Bool = false) where T <: Real

    nvecs = min(nvecs, length(D.F.values))
    ndata = size(X)[1]

    Z = zeros(ndata, nvecs) # Coefficients for nvecs first eigenvectors

    μ = nomean ? zero(D.μ) : D.μ

    X2 = X[:,:]
    X2 .-= D.μ'
    H = D.F.vectors ./ D.F.values'
    mul!(Z, X2, H)

    return Z
end


"""Convenience function with dummy input variable d to replicate the
call interface with D being a vector of DimRedStructs"""
function original_to_reduced(X::AbstractArray{T}, D::DimRedStruct{T}, d::Int; nvecs::Int = length(D.F.values), nomean::Bool = false) where T <: Real
    original_to_reduced(X, D; nvecs, nomean)
end


"""Convenience function that takes in a vector of
DimRedStructs and the index of dimension d"""
function original_to_reduced(X::AbstractArray{T}, D::Vector{DimRedStruct{T}}, d::Int; nvecs::Int = length(D.F.values), nomean::Bool = false) where T <: Real
    original_to_reduced(X, D[d]; nvecs, nomean)
end


"""Convenience function that takes in a vector of
DimRedStructs and returns all reduced dimensions"""
function original_to_reduced(X::AbstractArray{T}, D::Vector{DimRedStruct{T}}; nvecs::Int = length(D.F.values), nomean::Bool = false) where T <: Real
    [original_to_reduced(X, DD; nvecs, nomean) for DD ∈ D]
end


"""Reconstructs data in its original dimensions from transformed data"""
function reduced_to_original(Z::AbstractMatrix{T}, D::DimRedStruct{T};
                             nvecs = length(D.F.values), nomean::Bool = false) where T <: Real

    # println("Recovering from $nvecs PCs")
    ndata = size(Z)[1]

    X = zeros(ndata, size(D.F.vectors)[1])
    H = D.F.vectors' .* D.F.values
    mul!(X, Z, H)
    nomean || (X .+= D.μ') # add mean to all data points

    X
end


function reduced_unc_to_original(z::AbstractVector{T}, D::DimRedStruct{T};
                                 npcs = length(z)) where T <: Real

    # when using LinearAlgebra.eigen()
    # F = D.F.vectors[:, end:-1:end-npcs+1]
    # F * Diagonal(sqrt.(D.F.values[end:-1:end-npcs+1]) .* z[1:npcs]) * F'

    # when using Arnold.eigs()
    G = @view D.F.vectors[:, 1:npcs]
    G * Diagonal((D.F.values[1:npcs]) .* z[1:npcs]) * G'
end


function reduced_unc_to_original(Z::AbstractMatrix{T}, D::DimRedStruct{T};
                                 npcs = size(Z)[2]) where T <: Real

    [reduced_unc_to_original(z, D; npcs) for z ∈ eachrow(Z)]
end


"""X has data in the rows, i.e. X[3,:] is the third data vector. All
data will be dimreduced, but eigenvectors will use a limited amount of
data, just for speed if data is very high dimensional."""
function dimreduce_PCA(X::AbstractMatrix{T}; maxdata = 1000, nvecs = min(10, size(X)[2])) where T <: Real
    H = (size(X)[1] > maxdata) ? X[randperm(size(X)[1])[1:maxdata],:] : X
    C = cov(H)
    DimRedStruct(mean(H, dims = 1)[:], fasteigs(C, nvecs), 0)
end


function dimreduce_PCA(x::AbstractVector{T}; maxdata = 1000) where T <: Real
    dimreduce_PCA(reshape(x, (length(x), 1)); maxdata, nvecs = 1)
end


"""Center and scale data only."""
function dimreduce_basic(X::AbstractMatrix{T}) where T <: Real
    D = DimRedStruct(mean(X, dims = 1)[:],
                     (vectors = diagm(ones(size(X)[2])),
                      values = std(X, dims = 1)[:]), 0)
end




"""Carries out dimension reduction which is partly based on CCA,
partly PCA. We will be emulating loadings of nvecs_CCA components of Y
with CCA, and nvecs_PCA components of Y with PCA. Each emulated
loading is modeled in its own basis.

Since CCA eigenvectors are not a basis (they are not eigenvectors of
the data covariance matrix), we'll orthonormalize them. After going
through the nvecs_CCA vectors we then model some of the rest of the
variance in Y with PCA.

For now we only do CCA for the X part, meaning that nYCCA is
also the number of dimensions that we use for X.

"""
function dimreduce_CCA_PCA(X::Matrix{T}, Y::Matrix{T}; reg::T = 1e-3, maxdata::Int = 3000, nYCCA::Int = size(X)[2], nXCCA::Int = nYCCA, nYPCA::Int = 10) where T <: Real

    # No more CCA / PCA dims than there are dimensions
    nYCCA = min(nYCCA, size(Y)[2], size(X)[2])
    nYPCA = min(nYPCA, size(Y)[2] - nYCCA)

    nvecs_Y_tot = nYCCA + nYPCA

    Y_basis = zeros(size(Y)[2], nvecs_Y_tot)
    Y_eigvals = zeros(nvecs_Y_tot)

    # Downsize X and Y to make covariance computations faster
    n = min(maxdata, size(X)[1])
    s = randperm(size(X)[1])[1:n]
    X = X[s,:]
    Y = Y[s,:]

    # Center data
    μ_Y = mean(Y, dims = 1)[:]
    μ_X = mean(X, dims = 1)[:]
    X .-= μ_X'
    Y .-= μ_Y'

    # We will remove dimension by dimension from Y
    Y_tmp = Y[:,:]

    # This is returned below
    DXs = Vector{DimRedStruct{T}}() # (undef, nvecs_Y_tot)

    for j ∈ 1:nYCCA

        # Start afresh with X for each basis vector
        X_tmp = X[:,:]

        X_basis = zeros(size(X)[2], nXCCA)
        X_eigvals = zeros(nXCCA)

        # Get next Y basis vector, and the first X basis vector that
        # maximizes covariance for that Y direction.
        F_X, F_Y = CCA(X_tmp, Y_tmp; reg, maxdata, nvecs = 1)

        # Ensure orthonormality; see comment below for X_basis iteration
        Y_basis[:,j] .= GramSchmidt(F_Y.vectors[:,1], Y_basis[:,1:j-1])
        X_basis[:,1] .= F_X.vectors[:,1]

        X_tmp, vXprojs = remove_direction(X_tmp, F_X.vectors[:,1])
        Y_tmp, vYprojs = remove_direction(Y_tmp, Y_basis[:,j])

        # F_X.values and F_Y.values are not the correct values since
        # they are correlations. We are interested in the amount of
        # variance they explain of X and Y. Let's use the variances of
        # the projections instead: vXprojs and vYprojs

        X_eigvals[1] = std(vXprojs)
        Y_eigvals[j] = std(vYprojs)

        # p = scatter(vXprojs, vYprojs, label = "Y basis vec $j 1 and best corr")
        # savefig(p, "basis_$j 1.png")

        for i ∈ 2:nXCCA

            # Fit deg 4 polynomial to data and remove from projections
            # so that we don't try second time to predict Y
            # differences that we already predicted

            # FIXME UNCOMMENT THE NEXT TWO LINES TO DO THE POLYFIT;
            # HOWEVER, PROBABLY NOT THE RIGHT THING TO DO.
            # po = Polynomials.fit(vXprojs, vYprojs, 4)
            # vYprojs -= po.(vXprojs)

            F_X, _ = CCA(X_tmp, reshape(vYprojs, (n,1)); reg, maxdata, nvecs = 1)

            # Apparently because of regularization the vector
            # F_X.vectors[:,i] may in degenerate spaces (minimal cross
            # correlation between X and Y) end up being
            # non-orthogonal. For this reason we force it to be
            # orthogonal by doing Gram-Schmidt with earlier vectors

            X_basis[:,i] .= GramSchmidt(F_X.vectors[:,1], X_basis[:,1:i-1])

            for ii ∈ 1:i
                X_tmp, vXprojs = remove_direction(X_tmp, X_basis[:,ii])
            end
            X_eigvals[i] = std(vXprojs)

            # p = scatter(vXprojs, vYprojs, label = "Y basis vec $j $i and best corr")
            # savefig(p, "basis_$j $i.png")
        end

        # Check that your vectors are orthogonal:
        # display(X_basis' * X_basis)

        push!(DXs, DimRedStruct(μ_X, (vectors = X_basis, values = X_eigvals[:]), nXCCA))
    end

    # Add PCA components to our modeling
    if nYPCA > 0
        G = fasteigs(cov(Y_tmp), nYPCA)
        Y_basis[:,nYCCA + 1:end] .= G.vectors # [:,end:-1:end - nYPCA + 1]
        Y_eigvals[nYCCA + 1:end] .= G.values # [end:-1:end - nYPCA + 1]
    end

    # Alternative for reference using eigendecomposition of the full
    # matrix - fasteigs works better, however.
    # G = eigen(cov(Y_tmp))
    # Y_basis[:,nYCCA + 1:end] .= G.vectors[:,end:-1:end - nYPCA + 1]
    # Y_eigvals[nYCCA + 1:end] .= sqrt.(G.values[end:-1:end - nYPCA + 1])

    DX_basic = dimreduce_basic(X .+ μ_X')

    for l ∈ 1:nYPCA
        push!(DXs, DX_basic)
    end

    DY = DimRedStruct(μ_Y, (vectors = Y_basis, values = Y_eigvals), 0)

    return DXs, DY
end


"""This function carries out the CCA-based dimension reduction for X
and Y and augments that with PCA-based dimension reduction for Y (if
nYPCA > 0), and then still adds the centered and scaled X to the
X inputs. This may result in X that has more dimensions than the
original data; this generally increase performance of the constructed
GP emulator because the GP ends up being more expressive. If only some
of X dimensions are wanted in this augmentation, the Xdims variable
can be used to control which ones."""
function dimreduce_CCA_PCA_augmented(X::Matrix{T}, Y::Matrix{T};
                                     reg::T = 1e-3, maxdata::Int = 3000,
                                     nYCCA::Int = min(6, size(X)[2]),
                                     nXCCA::Int = nYCCA,
                                     nYPCA::Int = 10,
                                     Xdims::AbstractVector{Int} = 1:size(X)[2]) where T <: Real

    DXs_orig, DY = dimreduce_CCA_PCA(X, Y; reg, maxdata, nYCCA, nXCCA, nYPCA)
    DXbasic = dimreduce_basic(X)

    DXs = Vector{DimRedStruct{T}}(undef, 0)

    # Only add extra vectors to CCA dimensions, since for PCA
    # dimensions these would merely be doubled
    for DX in DXs_orig[1:nYCCA]
        m = DX.μ
        F = (vectors = hcat(DX.F.vectors, DXbasic.F.vectors[:,Xdims]),
             values = vcat(DX.F.values, DXbasic.F.values[Xdims]))
        push!(DXs, DimRedStruct(m, F, DX.nCCA))
    end

    for DX in DXs_orig[nYCCA+1:end]
        m = DX.μ
        F = (vectors = DX.F.vectors, values = DX.F.values)
        push!(DXs, DimRedStruct(m, F), 0)
    end

    return DXs, DY
end

