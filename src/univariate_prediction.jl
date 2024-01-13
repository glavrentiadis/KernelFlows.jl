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
export predict


using Distances

"""GP prediction with Model (univariate output). By default M.λ will
be applied, namely,

   X = M.λ' * X

This can be overruled by setting apply_λ to false. Note that in this
function X needs to be given in reduced coordinates.
"""
function predict(M::GPModel{T}, X::AbstractMatrix{T};
                 apply_λ::Bool = true,
                 apply_zyinvtransf::Bool = true,
                 workbuf::Union{Nothing, Matrix{T}} = nothing,
                 outbuf::Union{Nothing, Matrix{T}} = nothing,
                 nXlinear::Int = 1) where T <: Real

    apply_λ && (X .*= M.λ')

    # Allocate if buffers not given
    (workbuf == nothing) && (workbuf = zeros(size(X)[1], length(M.h)))
    (outbuf == nothing) && (outbuf = zeros(size(X)[1]))

    cross_covariance_matrix!(M.kernel, M.θ, X, M.Z, workbuf)
    mul!(outbuf, workbuf, M.h)

    apply_zyinvtransf && (outbuf .= M.zyinvtransf.(outbuf))

    outbuf
end


"""Compute cross-covariance matrix between X1 and X2. Typically this
would be between test inputs X (X1) and training data in GPModel.Z
(X2). The covariance matrix is computed in-place in workbuf."""
function cross_covariance_matrix!(k::UnaryKernel, θ::AbstractVector{T},
                                  X1::AbstractMatrix{T}, X2::AbstractMatrix{T},
                                  workbuf::Matrix{T}) where T <: Real

    s = Euclidean()

    # This is a cheap way to compute distances
    pairwise!(s, workbuf, X1, X2, dims = 1)
    workbuf .= k.k.(workbuf, θ[1], θ[2])

    # mul! won't accept AbstractArrays, but gemm! does not mind
    @views BLAS.gemm!('N', 'T', θ[3], X1[:,1:k.nXlinear],
                      X2[:,1:k.nXlinear], one(T), workbuf)
end


function cross_covariance_matrix!(k::BinaryKernel, θ::AbstractVector{T},
                                  X1::AbstractMatrix{T}, X2::AbstractMatrix{T},
                                  workbuf::Matrix{T}) where T <: Real

    (n,m) = size(workbuf)
    @inbounds for i in 1:n
        @inbounds for j in 1:m
            workbuf[i,j] = @views k.k(X1[i,:], X2[j,:], θ)
        end
    end




end
