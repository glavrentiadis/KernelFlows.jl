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
function predict(MVM::MVGPModel{T}, X::AbstractMatrix{T};
                 reduce_inputs::Bool = true,
                 apply_λ::Bool = true,
                 recover_outputs::Bool = true,
                 apply_zyinvtransf::Bool = true,
                 Mlist::AbstractVector{Int} = 1:length(MVM.Ms)) where T <: Real

    nte = size(X)[1]
    nzycols = length(MVM.Ms)
    ZY_pred = zeros(nte, nzycols)
    Threads.@threads :static for i ∈ Mlist
        Z = reduce_inputs ? reduce_X(X, MVM.G, i) : X
        nXlinear = nXl(MVM, i)
        ZY_pred[:,i] .= predict(MVM.Ms[i], Z; apply_λ, apply_zyinvtransf, nXlinear)
    end

    return recover_outputs ? recover_Y(ZY_pred, MVM.G) : ZY_pred
end


"""Remove points outside the training data (along any input
axis). This is useful if e.g. in a random testing data batch there are
data that end up outside the training data domain."""
function remove_extrapolations(MVM::MVGPModel{T}, X::Matrix{T}) where T <: Real

    m = X[:,1] .> Inf # don't remove anything yet
    nte = length(m)
    for (i,M) in enumerate(MVM.Ms)

        ZX = reduce_X(X, MVM, i)
        for j in 1:MVM.G.Xprojs[i].spec.nCCA
            a,b = extrema(MVM.Ms[i].Z[:,j])
            z = @views ZX[:,j]
            m .= m .|| (z .< a) .|| (z .> b)
        end
    end

    s_te = setdiff(1:nte, collect(1:nte)[m])
    X[s_te,:], s_te
end


