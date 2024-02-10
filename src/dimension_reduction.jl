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
export dimreduce, reduce, reduce_X, reduce_Y, recover, recover_Y


struct ProjectionSpec
    nCCA::Int # number of CCA vecs
    nPCA::Int # total number of PCA vectors
    ndummy::Int # number of scaling-only vecs, used if nCCA == nPCA == 0
    dummydims::AbstractVector{Int} # list of dummy dimensions (relevant for X)
    sparsedims::Vector{Int} # use only vectors listed here (relevant for X)
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
    reg_CCA::T # CCA regular
    Xtransfspec::TransfSpec{T} # non-linear input transformations
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
                   Xtransf_deg::Int = 0, Xtransf_ϵ::Real = 1e-2,
                   nYCCA::Int = 0, nYPCA::Int = 0, nXCCA::Int = 1,
                   dummyXdims::Union{Bool, AbstractVector{Int}} = true,
                   reg_CCA::Real = 1e-2, reg_CCA_X::Real = reg_CCA,
                   maxdata::Int = 3000, scale_Y::Bool = true) where T <: Real

    X, Xtransf_spec = standard_transformations(X; deg = Xtransf_deg, ϵ = Xtransf_ϵ)

    # Enforce types here without demanding user to e.g. add f0's to
    # parameters:
    Xtransf_ϵ = T(Xtransf_ϵ)
    reg_CCA = T(reg_CCA)
    reg_CCA_X = T(reg_CCA_X)

    (dummyXdims == false) && (dummyXdims = 1:0)
    (dummyXdims == true) && (dummyXdims = 1:size(X)[2])

    # Do not use more CCA / PCA dims than there are dimensions
    nYCCA = min(nYCCA, size(Y)[2])
    nYPCA = min(nYPCA, size(Y)[2] - nYCCA)
    nYCCA == 0 && (reg_CCA = zero(T))

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
    σY = scale_Y ? std(Y, dims = 1)[:] : ones(T, size(Y)[2])

    X .= (X .- μX') ./ σX'
    Y .= (Y .- μY') ./ σY'

    Y_unreduced = Y[:,:]

    # Allocate Projection objects for inputs
    Xprojs = Vector{Projection{T}}()
    for i in 1:nY
        sparseXdims = collect(1:nXCCA + length(dummyXdims))
        XSpec = ProjectionSpec(nXCCA, 0, length(dummyXdims), dummyXdims, sparseXdims)
        push!(Xprojs, Projection(zeros(T, (size(X)[2], nX)), zeros(T, nX), XSpec))
    end

    sparseYdims = collect(1:nYCCA + nYPCA + nYdummy)
    YSpec = ProjectionSpec(nYCCA, nYPCA, nYdummy, 1:nYdummy, sparseYdims)
    Yproj = Projection(zeros(T, size(Y)[2], nY), zeros(T, nY), YSpec)

    for i in 1:nYCCA
        FX, FY = CCA(X, Y; reg_Y = reg_CCA, reg_X = reg_CCA_X, nvecs = 1)

        # Orthogonalize Y-vector
        yvec = GramSchmidt(FY.vectors[:,1], Yproj.vectors[:,1:i-1])

        # Compute projections and remove yvec direction from Y
        Y, vYprojs = remove_direction(Y, yvec)

        # Save relevant quantities
        Yproj.vectors[:,i] .= yvec
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
        Yproj.vectors .= diagm(ones(T, nYdummy))
    end

    # Fill the rest of CCA X-dimensions and dummy X dimensions for all Y-vectors
    for i in 1:nY
        yproj_i = @views Y_unreduced * Yproj.vectors[:,i]
        get_X_CCA_vectors!(X, yproj_i; nXCCA, reg_CCA = reg_CCA_X, reg_CCA_X,
                           X_basis = Xprojs[i].vectors, X_values = Xprojs[i].values)

        dummyvecs, dummyvals = get_dummy_vectors(X; dummydims = dummyXdims)
        Xprojs[i].vectors[:,nXCCA+1:end] = dummyvecs
        Xprojs[i].values[nXCCA+1:end] = dummyvals
    end

    GPGeometry(Xprojs, Yproj, μX, σX, μY, σY, reg_CCA, Xtransf_spec)
end


"""Get CCA input dimensions that correlate maximally in data with
projected output vector yproj. In case we have earlier dimensions that
the resulting vectors need to be orthogonal with, those can be given
in the X_basis and X_values optional arguments. Note, that in that
case the dimensions of these arrays should still be at least
(size(X)[2], nXCCA) and (nXCCA,)."""
function get_X_CCA_vectors!(X::AbstractMatrix{T}, yproj::AbstractVector{T};
                            nXCCA::Int = 1, reg_CCA::T = 1e-2, reg_CCA_X::T = reg_CCA,
                            X_basis::AbstractMatrix{T} = zeros(T, size(X)[2], nXCCA),
                            X_values::AbstractVector{T} = zeros(T, nXCCA)) where T <: Real

    vXprojs = zeros(size(X)[1], nXCCA)

    for i in 1:nXCCA

        if X_values[i] != 0
            # Skip vectors that have been computed already. Inferred
            # from non-zero singular values. In order to not produce
            # the same vectors twice, we remove the corresponding
            # directions from data.
            X, vXprojs[:,i] = remove_direction(X, X_basis[:,i])
            continue
        end

        F_X, _ = CCA(X, reshape(yproj, (length(yproj),1));
                     reg_Y = reg_CCA, reg_X = reg_CCA_X, nvecs = 1)

        X_basis[:,i] .= F_X.vectors[:,1]

        # Make the X basis vectors such that projections of inputs on
        # the dimensions are not similar

        # Recursive version
        # normalize(x) = x ./ sqrt(x' * x)
        # if i > 1
        #     p2 = X * X_basis[:,i]
        #     p2_no = normalize(p2)
        #     p2mod =  p2[:]

        #     for ii in 1:i-1
        #         p1 = @view vXprojs[:,ii]
        #         p1_no = normalize(p1)
        #         p2mod .-= (p1_no' * p2_no) * p1
        #     end
        #     X_basis[:,i] .= normalize(inv(X' * X) * X' * p2mod)
        # end

        # Just handle the previous dim
        # normalize(x) = x ./ sqrt(x' * x)
        # if i > 1
        #     p1 = @view vXprojs[:,i-1]
        #     p1_no = normalize(p1)
        #     p2 = X * X_basis[:,i]
        #     p2_no = normalize(p2)
        #     p2mod = p2 - (p1_no' * p2_no) * p1
        #     X_basis[:,i] .= normalize(inv(X' * X) * X' * p2mod)
        # end

        # Apparently because of regularization, and because of the
        # recursion aboce, the CCA vectors may end up being
        # non-orthogonal. For this reason we force it to be orthogonal
        # by doing Gram-Schmidt with earlier vectors
        X_basis[:,i] .= GramSchmidt(X_basis[:,i], X_basis[:,1:i-1])

        # Update X to be orthogonal to all basis vectors up to now
        for ii in 1:i
            X, vXprojs[:,i] = remove_direction(X, X_basis[:,ii])
        end

        # Remove from yprojs linear regression results of previous X
        # CCA vectors, to make next X CCA vectors independently
        # informative for predicting the current Y vector.
        # vXp = vXprojs[:,i]
        # A = hcat(ones(length(vXp)), vXp)
        # β = inv(A' * A) * A' * yproj
        # yproj_regr_pred = β[2] * vXp .+ β[1]
        # yproj .-= yproj_regr_pred
        # X_values[i] = std(vXp)
        X_values[i] = std(vXprojs[:,i])
    end
end


function get_PCA_vectors(X::AbstractMatrix{T}, nPCA::Int) where T <: Real
    # Check if we are centered, and if not, center data before SVD
    @views X = abs(sum(X[:,1])) > 1e-10 ? X .- mean(X, dims = 1) : X
    (U, S, Vt) = svd(X)
    nPCA = min(length(S), nPCA)
    (vecs, vals) =  (Vt[:,1:nPCA], S[1:nPCA]  ./ sqrt(size(X)[1]-1.))
    return (vecs, vals)

    # # Old method
    # @time F = fasteigs(cov(X), nPCA)
    # # display(vecs1 ./ F.vectors)
    # # display(vals1 ./ F.values)
    # return (vecs1, vals1)
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


"""Move from original coordinates to dimension-reduced (or augmented)
coordinates."""
function reduce(X::AbstractMatrix{T}, P::Projection{T}, μ::Vector{T}, σ::Vector{T}) where T <: Real
    X = X .- μ' # center
    X ./= σ' # scale
    @views H = P.vectors[:, P.spec.sparsedims] ./ P.values[P.spec.sparsedims]'
    X * H
end


"""Move from reduced (or augmented) back to original space."""
function recover(Z::AbstractMatrix{T}, P::Projection{T}, μ::Vector{T}, σ::Vector{T}) where T <: Real
    H = P.vectors' .* P.values
    (Z * H) .* σ' .+ μ'
end


"""Apply standard transformations and dimension reduction as described
in GPGeometry object in G. This function does not scale the inputs
according to learned kernel parameters"""
function reduce_X(X::AbstractMatrix{T}, G::GPGeometry{T}, i::Int) where T <: Real
    X = G.Xtransfspec.deg == 0 ? X : standard_transformations(X, G.Xtransfspec)
    reduce(X, G.Xprojs[i], G.μX,  G.σX)
end


function reduce_Y(Y::AbstractMatrix{T}, G::GPGeometry{T}) where T <: Real
    reduce(Y, G.Yproj, G.μY,  G.σY)
end


function recover_Y(Z::AbstractMatrix{T}, G::GPGeometry{T}) where T <: Real
    recover(Z, G.Yproj, G.μY,  G.σY)
end


# Not yet adapted to new dimension reduction code
# function reduced_unc_to_original(z::AbstractVector{T}, D::DimRedStruct{T};
#                                  npcs = length(z)) where T <: Real
#     # when using Arnold.eigs()
#     G = @view D.F.vectors[:, 1:npcs]
#     G * Diagonal((D.F.values[1:npcs]) .* z[1:npcs]) * G'
# end
