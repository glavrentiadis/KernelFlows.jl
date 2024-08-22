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


"""Carries out CCA for input-output pair X,Y"""
function CCA(X::Matrix{T}, Y::Matrix{T}; reg_Y::T = 1e-2, reg_X::T = reg_Y, maxdata::Int = 3000, nvecs = 50) where T <: Real

    ndata = min(maxdata, size(X)[1]) # Use max ndata data points
    ndx = size(X)[2]
    s = randperm(size(X)[1])[1:ndata]

    X = X[s,:]
    Y = Y[s,:]
    (Xvecs, Xvals) = get_PCA_vectors(X, maxdata)
    (Yvecs, Yvals) = get_PCA_vectors(Y, maxdata)

    X = X * Xvecs
    Y = Y * Yvecs

    H = hcat(X, Y)
    C = cov(H)

    Cxx = @view C[1:ndx,1:ndx]
    Cxy = @view C[ndx+1:end,1:ndx]
    Cyy = @view C[ndx+1:end,ndx+1:end]

    Cxx[diagind(Cxx)] .+= reg_X
    Cyy[diagind(Cyy)] .+= reg_Y

    CxxI = inv(Cxx)
    CyyI = inv(Cyy)
    R_X = CxxI * Cxy' * CyyI * Cxy

    F_X = fasteigs(R_X, nvecs; force_real = true)

    # No need to compute Y vectors to get 1-d subspace of 1-d space
    if size(Y)[2] > 1
        R_Y = CyyI * Cxy * CxxI * Cxy'
        F0_Y = fasteigs(R_Y, nvecs; force_real = true)
        F_Y = (vectors = Yvecs * F0_Y.vectors, values = F0_Y.values)
    else
        F_Y = (vectors = ones(T, (1,1)), values = ones(T, 1))
    end

    F_X = (vectors = Xvecs * F_X.vectors, values = F_X.values)

    F_X, F_Y
end


"""Removes direction v from X and returns flattened X and projections
along the removed direction"""
function remove_direction(X::Matrix{T}, v::Vector{T}) where T <: Real
    v ./= sqrt(sum(v.^2)) # normalize v
    vprojs = X * v
    return X -  vprojs .* v', vprojs
end


"""Removes the directions in columns of M from v and returns
normalized vector"""
function GramSchmidt(v::Vector{T}, M::Matrix{T}) where T <: Real
    projs = M' * v
    for (i,c) ∈ enumerate(eachcol(M))
        v .-= projs[i] * c
    end
    v ./ sqrt(sum(v.^2))
end


"""Returns largest nvecs eigenvectors and singular values of square
(covariance) matrix C in a NamedTuple."""
function fasteigs(C::Matrix{T}, nvecs::Int; force_real::Bool = false) where T <: Real

    d = size(C)[1] # dimension of C
    nvecs = min(nvecs, d)

    (U, S, Vt) = svd(C)
    FF = (vectors = U[:,1:nvecs], values = S[1:nvecs])
    return FF

    # Old version below, kept for reference.
    # More accurate, but slow for very large dimension
    # if (d < 500) || (d - nvecs < 2)
    #     G = eigen(C)
    #     F = (vectors = G.vectors[:,end:-1:end-nvecs+1],
    #          values = sqrt.(G.values[end:-1:end-nvecs+1]))
    # # Faster for large dimension
    # else
    #     G = Arpack.eigs(C, nev = nvecs)
    #     F = (vectors = G[2], values = sqrt.(G[1]))
    # end

    # # display(FF.vectors' ./ F.vectors')
    # # println(FF.values[1] / F.values[1])
    # # println(F.values[1])
    # # println(size(C))
    # # display(FF.values' ./ F.values')

    # if force_real
    #     return (vectors = real.(F.vectors), values = real.(F.values))
    # end

    # F
end
