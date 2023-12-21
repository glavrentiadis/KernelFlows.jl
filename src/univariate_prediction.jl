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
    s = Euclidean()

    # This is a cheap way to compute distances
    pairwise!(s, workbuf, X, M.Z, dims = 1)
    workbuf .= M.kernel.(workbuf, M.θ[1], M.θ[2])
    # mul! won't accept AbstractArrays, but gemm! does not mind
    BLAS.gemm!('N', 'T', M.θ[3], (@view X[:,1:nXlinear]),
               (@view M.Z[:,1:nXlinear]), one(T), workbuf)
    mul!(outbuf, workbuf, M.h)

    apply_zyinvtransf && (outbuf .= M.zyinvtransf.(outbuf))

    outbuf
end
