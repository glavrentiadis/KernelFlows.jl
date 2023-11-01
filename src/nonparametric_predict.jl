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
# How does nonparametric prediction look like?
# 1. Transform test inputs X_te with DX
# 2. Move X_te by iterating predicting over warpings
# 3. Predict labels by applying B_warped to moved X_te


function predict(M::NonParametricMVGPModel{T}, X::AbstractMatrix{T};
                 apply_zyinvtransf::Bool = true) where T <: Real

    # 1. Always transform X to Z by applying all DX and scaling λ
    Zs = [m.λ' * original_to_reduced(X, m.DX) for m in M.B_orig.Ms]

    ZYs = Matrix{T}[]
    Y_te_predicteds = Matrix{T}[]

    for (i,m1) in enumerate(M.warpings)
        for w ∈ M.warpings
            athath
        end
    end
    # 2.

    # DX REALLY NEEDS TO BE IDENTITY AND NOT EVEN THE DIMREDUCE_BASIC.
    #     WE NEED TO NOT DO ANY TRANSFORMATIONS _OR_ WE NEED TO FIGURE
    #     OUT DX CHAINING FOR PREDICTIONS AS WELL. PERHAPS

end


# function predict_single()
