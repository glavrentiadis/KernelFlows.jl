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


"""Combines a vector of Projections into a single Projection. This is
a helper function for combining multiple GPGeometry objects
together."""
function combine(Ps::Vector{Projection{T}}) where T <: Real
    vectors = hcat([P.vectors for P in Ps]...)
    values = vcat([P.values for P in Ps]...)
    dummydims = vcat([P.spec.dummydims for P in Ps]...)
    ndummy = length(dummydims)

    # To get sparsedims right, we have to do compute offsets so that
    # we don't include dimensions twice, which would break reduce_Y()
    s = [0, cumsum([length(P.values) for P in Ps])...]
    sparsedims = vcat([P.spec.sparsedims .+ s[i] for (i,P) in enumerate(Ps)]...)

    nCCA = sum([P.spec.nCCA for P in Ps])
    nPCA = sum([P.spec.nPCA for P in Ps])
    spec = ProjectionSpec(nCCA, nPCA, ndummy, dummydims, sparsedims)
    Projection(vectors, values, spec)
end


"""Combine a number of GPGeometry structs into a single one. Typically
this would be used to combine GPGeometry objects that were constructed
using disjoint Ydims kwargs (no overlapping dimensions). There are no
automatic checks to ensure this disjointedness. The motivation behind
such a construct comes from when different kinds of dimension
reduction are needed for different parts of the state vector."""
function combine(Gs::Vector{GPGeometry{T}}) where T <: Real
    Yproj = combine([G.Yproj for G in Gs])
    Xprojs = vcat([G.Xprojs for G in Gs]...)

    μX = Gs[1].μX
    σX = Gs[1].σX
    μY = Gs[1].μY

    # Y data should be the same for all G in Gs, so the mean of Y
    # should also be the same, modulo random sampling
    σY = Gs[1].σY # Straight from data; not effected by Ydims.

    # NOTE! for the following, we only record one value, even though
    # many might have been used for getting the different Gs.G. These
    # are anyway recorded for diagnostic purposes only.
    reg_CCA = Gs[1].reg_CCA
    Xtransfspec = Gs[1].Xtransfspec

    GPGeometry(Xprojs, Yproj, μX, σX, μY, σY, reg_CCA, Xtransfspec)
end

"""Use different dimension reduction specifications for different
parts of the state. See docstring of combine() for further
details. Each element of D should specify a unique Ydims range."""
function dimreduce(X::AbstractMatrix{T}, Y::AbstractMatrix{T}, D::Vector) where T <: Real
    combine([dimreduce(X, Y; d...) for d in D])
end


"""Regular dimreduce(), but with kwargs given as a Dict"""
function dimreduce(X::AbstractMatrix{T}, Y::AbstractMatrix{T},
                   drargs::Dict{Symbol,H}) where {H,T<:Real}
    dimreduce(X, Y; drargs...)
end


"""Construct GPGeometry object that describes the geometry of the
multivariate GP in terms of a number of univariate GPs. Examples:

1) Don't change Y dimensions apart from centering and scaling (one 1-d
   GP for each column in Y). Uses whatever defaults are in place for
   X.

   julia> dimreduce(X, Y)

2) Use 3 CCA and 3 PCA dimensions for Y, and use 2 CCA vectors on the
   input side for each of these 6 dimensions. Augment the input side
   with first five X dimensions.

   julia> dimreduce(X, Y; nYCCA = 3, nYPCA = 3, nXCCA = 2, dummyXdims = 1:5)

3) Same as before, but use all X dims for augmentation

   julia> dimreduce(X, Y; nYCCA = 3, nYPCA = 3, nXCCA = 2, dummyXdims = true)

4) Same as above, but also scale output dimensions to unit variance
   and treat everything outside dimensions 3 to 5 as constant:

   julia> dimreduce(X, Y; nYCCA = 3, nYPCA = 3, nXCCA = 2, dummyXdims = true,
                    scale_Y = true, Ydims = 3:5)

"""
function dimreduce(X::AbstractMatrix{T}, Y::AbstractMatrix{T};
                   Xtransf_deg::Int = 0, Xtransf_ϵ::Real = 1e-2,
                   Ydims::AbstractVector{Int} = 1:size(Y)[2],
                   nYCCA::Int = 0, nYPCA::Int = 0,
                   nXPCA::Int = 0, nXCCA::Int = 1,
                   dummyXdims::Union{Bool, AbstractVector{Int}} = true,
                   reg_CCA::Real = 1e-2, reg_CCA_X::Real = reg_CCA,
                   maxdata::Int = 3000, scale_Y::Bool = false) where T <: Real

    X, Xtransf_spec = standard_transformations(X; deg = Xtransf_deg, ϵ = Xtransf_ϵ)

    # Enforce types here without demanding user to e.g. add f0's to
    # parameters:
    Xtransf_ϵ = T(Xtransf_ϵ)
    reg_CCA = T(reg_CCA)
    reg_CCA_X = T(reg_CCA_X)

    (dummyXdims == false) && (dummyXdims = 1:0)
    (dummyXdims == true) && (dummyXdims = 1:size(X)[2])

    # If no input X were given, we force dummyXdims to true, in order
    # to have _any_ dimensions to predict with. Otherwise we are not
    # learning an λs.
    if (nXPCA + nXCCA + length(dummyXdims) == 0)
        nXdummy = size(X)[2]
        println("Setting dummyXdims to 1:$nXdummy")
        dummyXdims =1:nXdummy
    end

    # Do not use more CCA / PCA dims than there are dimensions
    nY_full = length(Ydims) # only these can be non-constant
    nYCCA = min(nYCCA, nY_full)
    nYPCA = min(nYPCA, nY_full - nYCCA)
    nYCCA == 0 && (reg_CCA = zero(T))

    nXCCA = min(nXCCA, size(X)[2])
    nXPCA = min(nXPCA, size(X)[2] - nXCCA)

    # If there are no CCA or PCA output vectors, we don't do any
    # transforms but model the data directly in the original
    # dimensions. If there are any CCA or PCA Y-dimensions, no dummy
    # Y-dimensions will be used.
    nYdummy = nYCCA + nYPCA == 0 ? nY_full : 0

    nX = nXPCA + nXCCA + length(dummyXdims) # total number of transformed inputs
    nY = nYCCA + nYPCA + nYdummy # total number of transformed outputs

    # Shrink X and Y to make covariance computations faster
    n = min(maxdata, size(X)[1])
    s = randperm(size(X)[1])[1:n]
    X = X[s,:]
    Y = Y[s,:]

    # Get mean and standard deviation for centering and scaling data
    μX = mean(X, dims = 1)[:]
    μY = mean(Y, dims = 1)[:]
    σX = std(X, dims = 1)[:]

    # No special handling for YdimsC dimensions, so that σY should be
    # the same, no matter what Ydims kwarg was given.
    σY = std(Y, dims = 1)[:]

    # Don't scale dimensions with zero variance to avoid NaNs
    Xnonconstdims = (σX .!= 0.)
    Ynonconstdims = (σY .!= 0.)

    # Override Y scaling if scale_Y = false
    if !scale_Y
        σY[Ynonconstdims] .= 1.0
    end

    σY .= max.(eps(T), σY) # reduce() divides by σY
    σY .= min.(1e6, σY) # also avoid crazily large values

    X .= X .- μX'
    Y .= Y .- μY'
    X[:,Xnonconstdims] ./= σX[Xnonconstdims]'
    Y[:,Ynonconstdims] ./= σY[Ynonconstdims]'

    # Set non-selected dimensions to zero. Any predictions will then
    # be just μY. C stands for complement.
    YdimsC = setdiff(1:size(Y)[2], Ydims)
    Y[:, YdimsC] .= 0

    Y_unreduced = Y[:,:]

    # Allocate Projection objects for inputs
    Xprojs = Vector{Projection{T}}()
    for i in 1:nY
        sparseXdims = collect(1:(nXCCA + nXPCA + length(dummyXdims)))
        XSpec = ProjectionSpec(nXCCA, nXPCA, length(dummyXdims), dummyXdims, sparseXdims)
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
        Yproj.vectors .= 0.0
        for (i,yd) in enumerate(Ydims)
            Yproj.vectors[yd,i] = 1.0
        end
    end

    # Fill the rest of CCA X-dimensions and dummy X dimensions for all Y-vectors
    Threads.@threads for i in 1:nY
        yproj_i = @views Y_unreduced * Yproj.vectors[:,i]
        if nXCCA > 0
            get_X_CCA_vectors!(X, yproj_i; nXCCA, reg_CCA = reg_CCA_X, reg_CCA_X,
                               X_basis = Xprojs[i].vectors, X_values = Xprojs[i].values)
        end

        if nXPCA > 0
            r = (nXCCA+1):(nXCCA+nXPCA)
            Xiv = Xprojs[i].vectors # shorthand
            (XPCvecs, XPCvals) = get_PCA_vectors(X - (X * Xiv) * Xiv', nXPCA)
            Xprojs[i].values[r] .= XPCvals
            Xprojs[i].vectors[:,r] .= XPCvecs
        end

        dummyvecs, dummyvals = get_dummy_vectors(X; dummydims = dummyXdims)
        Xprojs[i].vectors[:,nXCCA+nXPCA+1:end] = dummyvecs
        Xprojs[i].values[nXCCA+nXPCA+1:end] = dummyvals
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

    # Old method
    # @time F = fasteigs(cov(X_orig), nPCA)
    # F = fasteigs(cov(X), nPCA)
    # display(vecs1 ./ F.vectors)
    # display(vals1 ./ F.values)
    # display(F.vectors - vecs)
    # display(F.values - vals.^2)
    # return (F.vectors, sqrt.(F.values))
end


function get_dummy_vectors(X::AbstractMatrix{T};
                           dummydims::AbstractVector{Int} = 1:size(X)[2]) where T <: Real
    values = @views std(X[:,dummydims], dims = 1)
    values[values .== 0.] .= 1. # avoid NaNs when reducing constant dimensions
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

function reduce!(X::AbstractMatrix{T}, P::Projection{T}, μ::Vector{T}, σ::Vector{T},
                 Xbuf::Matrix{T}, Hbuf::Matrix{T}, Zbuf::Matrix{T}) where T <: Real
    Xbuf .= (X .- μ') ./ σ'
    Hbuf .= P.vectors ./ P.values'
    mul!(Zbuf, Xbuf, Hbuf)
    Zbuf
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
    recover(Z, G.Yproj, G.μY, G.σY)
end


"""Function to recover just one vector / scalar. As we return just one
vector, it is a column vector. This is different from reduce_Y, where
data are in rows."""
function recover_y(z::AbstractVector{T}, G::GPGeometry{T}) where T <: Real
    recover_Y(reshape(z, (1, length(z))), G)[:]
end


# function reduce!(Z_out::AbstractMatrix{T}, X::AbstractMatrix{T}, X_tmp::AbstractMatrix{T},
#                  H_tmp::AbstractMatrix{T}, P::Projection{T}, μ::Vector{T}, σ::Vector{T}) where T <: Real
#     X_tmp .= X .- μ' # center
#     X_tmp ./= σ' # scale
#     H_tmp .= P.vectors ./ P.values'
#     mul!(Z_out, X, H_tmp)
# end


# function reduce_X!(Z_out::AbstractMatrix{T}, X::AbstractMatrix{T},
#                    X_tmp::AbstractMatrix{T}, H_tmp::AbstractMatrix{T},
#                    G::GPGeometry{T}, i::Int) where T <: Real
#     reduce!(Z_out, X, X_tmp, H_tmp, G.Xprojs[i], G.μX,  G.σX)
# end


# Not yet adapted to new dimension reduction code
# function reduced_unc_to_original(z::AbstractVector{T}, D::DimRedStruct{T};
#                                  npcs = length(z)) where T <: Real
#     # when using Arnold.eigs()
#     G = @view D.F.vectors[:, 1:npcs]
#     G * Diagonal((D.F.values[1:npcs]) .* z[1:npcs]) * G'
# end
