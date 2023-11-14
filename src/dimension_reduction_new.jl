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

struct ProjectionSpec
    nCCA::Int # number of CCA vecs
    nPCA::Int # total number of PCA vectors
    ndummy::Int # number of scaling-only vecs
    dummydims::AbstractVector{Int} # list of dummy dimensions
end


struct Projection{T}
    vectors::Matrix{T} # directions of transformed inputs
    values::Vector{T} # standard deviations of transformed data
    spec::ProjectionSpec # where did the vectors/values come from
end


struct GPGeometry{T}
    Xprojs::Vector{Projection{T}}
    Yproj::Projection{T}
    μX::Vector{T} # input (X) mean
    σX::Vector{T} # input standard deviation
    μY::Vector{T} # output (Y) mean
    σY::Vector{T} # output standard deviation
    YSpec::ProjectionSpec # Projection spec for Y-coordinates
    reg_CCA::T # CCA regular
end


"""Construct GPDimensionMap for diagonal univariate GPs.:

Don't change Y dimensions apart from centering and scaling (one 1-d GP
for each column in Y). Uses whatever defaults are in place for X.

julia> dimreduce(X, Y)

3 CCA and 3 PCA dimensions for Y, and use 2 CCA vectors on the input
side for each of these 6 dimensions. Augment the input side with first
five X dimensions.

julia> dimreduce(X, Y; nYCCA = 3, nYPCA = 3, nXCCA = 2, dummyXdims = 1:5)
"""
function dimreduce(X::AbstractMatrix{T}, Y::AbstractMatrix{T};
                   nYCCA::Int = 0, nYPCA::Int = 0, nXCCA::Int = 1,
                   dummyXdims::AbstractVector{Int} = 1:size(X)[2],
                   reg_CCA::T = 1e-2, maxdata::Int = 3000) where T <: Real

    # Do not use more CCA / PCA dims than there are dimensions
    nYCCA = min(nYCCA, size(Y)[2], size(X)[2])
    nYPCA = min(nYPCA, size(Y)[2] - nYCCA)
    nYCCA == 0 && (reg_CCA = 0.0)

    # If there are no CCA or PCA output vectors, we don't do any
    # transforms but model the data directly in the original
    # dimensions. If there are any CCA or PCA Y-dimensions, no dummy
    # Y-dimensions will be used.
    nYdummy = nYCCA + nYPCA == 0 ? size(Y)[2] : 0

    nX = nXCCA + length(dummyXdims) # total number of transformed inputs
    nY = nYCCA + nYPCA + nYdummy # total number of transformed outputs

    # Shrink X and Y to make covariance computations faster
    n = min(maxdata, size(X)[1])
    s = randperm(size(X)[1])[1:n]
    X = X[s,:]
    Y = Y[s,:]

    # Center and scale data
    μX = mean(X, dims = 1)[:]
    μY = mean(Y, dims = 1)[:]
    σX = std(X, dims = 1)[:]
    σY = std(Y, dims = 1)[:]

    X .= (X .- μX') ./ σX'
    Y .= (Y .- μY') ./ σY'

    # Allocate Projection objects for inputs
    Xprojs = Vector{Projection{T}}()
    for i in 1:nY
        XSpec = ProjectionSpec(nXCCA, 0, length(dummyXdims), dummyXdims)
        push!(Xprojs, Projection(zeros(size(X)[2], nX), zeros(nX), XSpec))
    end

    YSpec = ProjectionSpec(nYCCA, nYPCA, nYdummy, 1:nYdummy)
    Yproj = Projection(zeros(size(Y)[2], nY), zeros(nY), YSpec)

    for i in 1:nYCCA
        FX, FY = CCA(X, Y; reg = reg_CCA, nvecs = 1)

        # Orthogonalize Y-vector
        yvec = GramSchmidt(FY.vectors[:,1], Yproj.vectors[:,1:i-1])

        # Compute projections and remove yvec direction from Y
        Y, vYprojs = remove_direction(Y, yvec)
        
        Xprojs[i].vectors[:,1] = FX.vectors[:,1]
        Yproj.vectors[:,i] = yvec
        Xprojs[i].values[1] = FX.values[1]
        Yproj.values[i] = std(vYprojs)
    end

    # Get output PCA vectors and values
    if nYPCA > 0
        vectors, values = get_PCA_vectors(Y, nYPCA)
        Yproj.vectors[:, nYCCA+1:nYCCA+nYPCA] = vectors
        Yproj.values[nYCCA+1:nYCCA+nYPCA] = values
    end

    # Get dummy output vectors and values
    if nYdummy > 0
        Yproj.values .= 1.0 # Data was standardized earlier
        Yproj.vectors .= diagm(ones(nYdummy))
    end
    
    # Fill the rest of CCA X-dimensions and dummy X dimensions for all Y-vectors
    for i in 1:nY
        yproj_i = Y * Yproj.vectors[:,i]
        get_X_CCA_vectors!(X, yproj_i; nXCCA, reg_CCA,
                           X_basis = Xprojs[i].vectors, X_values = Xprojs[i].values)

        dummyvecs, dummyvals = get_dummy_vectors(X; dummydims = dummyXdims)
        Xprojs[i].vectors[:,nXCCA+1:end] = dummyvecs
        Xprojs[i].values[nXCCA+1:end] = dummyvals
    end

    GPGeometry(Xprojs, Yproj, μX, σX, μY, σY, YSpec, reg_CCA)
end


"""Get CCA input dimensions that correlate maximally in data with
projected output vector yproj. In case we have earlier dimensions that
the resulting vectors need to be orthogonal with, those can be given
in the X_basis and X_values optional arguments. Note, that in that
case the dimensions of these arrays should still be at least
(size(X)[2], nXCCA) and (nXCCA,)."""
function get_X_CCA_vectors!(X::AbstractMatrix{T}, yproj::AbstractVector{T};
                            nXCCA::Int = 1, reg_CCA::T = 1e-2,
                            X_basis::AbstractMatrix{T} = zeros(T, size(X)[2], nXCCA),
                            X_values::AbstractVector{T} = zeros(T, nXCCA)) where T <: Real

    for i in 1:nXCCA

        if X_values[i] != 0
            # Skip vectors that have been computed already. Inferred
            # from non-zero singular values. In order to not produce
            # the same vectors twice, we remove the corresponding
            # directions from data.
            X, vXprojs = remove_direction(X, X_basis[:,i])
            continue
        end

        F_X, _ = CCA(X, reshape(yproj, (length(yproj),1));
                     reg = reg_CCA, nvecs = 1)

        # Apparently because of regularization the CCA vectors may end
        # up being non-orthogonal. For this reason we force it to be
        # orthogonal by doing Gram-Schmidt with earlier vectors

        X_basis[:,i] .= GramSchmidt(F_X.vectors[:,1], X_basis[:,1:i-1])

        for ii ∈ 1:i
            X, vXprojs = remove_direction(X, X_basis[:,ii])
        end

        X_values[i] = std(vXprojs)
    end
end


function get_PCA_vectors(X::AbstractMatrix{T}, nPCA::Int) where T <: Real
    G = fasteigs(cov(X), nPCA)
    return (G.vectors, G.values)
end


function get_dummy_vectors(X::AbstractMatrix{T};
                           dummydims::AbstractVector{Int} = 1:size(X)[2]) where T <: Real
    values = @views std(X[:,dummydims], dims = 1)
    vectors = zeros(size(X)[2], length(dummydims))
    for (i,d) in enumerate(dummydims)
        vectors[d,i] = 1
    end
    return (vectors, values)
end
