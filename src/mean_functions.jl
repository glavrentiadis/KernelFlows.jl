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
"""Polynomial correction of mean. Sometimes, after GP has been fitted,
the prediction bias is not a random function of the inputs but rather
there is a pattern. If so, a function can be fitted to just remove
that bias. This function does that with polynomials that are functions
of one dimension only. ZX_te and ZY_te are the (dimension reduced)
inputs and outputs to the GP, and ZY_te_pred are the GP predictions
that have been obtained. DY describes the dimension reduction of Y,
and dims is a vector describing which dimension in ZX we take to have
explaining power for which dimension. For instance, if dims is [3,2],
that means that polynomials will be fitted for input-output pairs
(ZX_te[s, 3], ZY_te[s,1]) and (ZX_te[s,2], ZY_te[s,2]). npts
determines how many input space points are used for the fit, and deg
gives the degree of the polynomial.

The function returns the polynomials, corrected predictions in both
untransformed and transformed spaces, and the indexes of the data used
for the fit, and the complement.

N.B. This function assumes that all ZY columns are modeled with the
same ZX data (no different dimension reductions for different Y
dimensions). In the absence of nonlinear transformations untransformed
X should work reasonably well, or then scaled and centered inputs from
dimreduce_basic().  """

function polycorrect_mf(ZX::Matrix{T}, ZY::Matrix{T}, ZY_pred::Matrix{T},
                        DY::DimRedStruct, dims::AbstractVector{Int};
                        npts::Int = 100, deg::Int = 10) where T <: Real

    # Take points from ZX in as space-filling fashion as
    # possible. We assume first dimension is the most important
    # here. If not, change the [1] to something else.
    idx = sortperm(ZX[:,dims[1]])[KFCommon.splitrange(1, size(ZX)[1], npts - 1)]
    idxC = setdiff(1:size(ZX)[1], idx) # Complement

    # Fit polynomials for each column of ZY and appropriate X dimension
    polys = [Polynomials.fit(ZX[idx, dims[k]], ZY[idx, k] - ZY_pred[idx, k], 10) for k ∈ 1:size(ZY_pred)[2]]

    # Apply corrections
    ZY_pred_corr = 1 * ZY_pred
    for (i,c) ∈ enumerate(eachcol(ZY_pred_corr))
        c .+= polys[i].(ZX[:,dims[i]])
    end

    # Compute corrected Y transformed back into original coordinates
    Y_pred_corr = reduced_to_original(ZY_pred_corr, DY; nvecs = size(ZY)[2])

    return polys, Y_pred_corr, ZY_pred_corr, idx, idxC
end



