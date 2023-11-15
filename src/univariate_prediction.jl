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
                 buf::Union{Nothing, Matrix{T}} = nothing,
                 nXlinear::Int = 1) where T <: Real

    apply_λ && (X .*= M.λ')

    # Allocate buffer if not given
    (buf == nothing) && (buf = zeros(size(X)[1], length(M.h)))

    s = Euclidean()
    pairwise!((a,b) -> M.kernel(s(a,b); M.θ), buf, eachrow(X), eachrow(M.Z))

    buf += @views M.θ[end - 1] * X[:,1:nXlinear] * M.Z[:,1:nXlinear]'

    ret = apply_zyinvtransf ? M.zyinvtransf.(buf * M.h) : buf * M.h
end
