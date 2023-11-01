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
                 apply_DX::Bool = true,
                 apply_λ::Bool = true,
                 apply_DY::Bool = true,
                 apply_zyinvtransf::Bool = true) where T <: Real

    nte = size(X)[1]
    nzycols = length(MVM.Ms)
    ZY_pred = zeros(nte, nzycols)
    Threads.@threads for (i,M) ∈ collect(enumerate(MVM.Ms))
        ZY_pred[:,i] .= predict(MVM.Ms[i], X; apply_DX, apply_λ, apply_zyinvtransf)
    end

    return apply_DY ? reduced_to_original(ZY_pred, MVM.DY) : ZY_pred
end
