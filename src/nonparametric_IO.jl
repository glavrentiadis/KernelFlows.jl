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
# N.B. This file is obsolete, but it can be used as inspiration for a
# proper implementation.

function save_subdb_info(DX::DimensionReductionStruct{T}, DY::DimensionReductionStruct{T}, sols::Vector{FlowResultSGD}, ZY_tr::Matrix{T}; segments = 10, outf_basename = "subdb_output") where T <: Real
    DXμ = DX.μ
    DXvecs = DX.F.vectors
    DXvals = DX.F.values
    DYμ = DY.μ
    DYvecs = DY.F.vectors
    DYvals = DY.F.values

    ntimes = length(sols[1].u)
    segments = min(ntimes, segments)
    sidx = KFCommon.splitrange(1, ntimes, segments)
    sols = [s.u[sidx] for s ∈ sols]
    # sols_s = [s.s_values[sidx] for s ∈ sols] # not saved for now


    # For quick regression. however, then there is another bottleneck
    # which is moving the points, and which costs much more. That
    # could also be quicksolved, but would require a bit
    # thinking. That's why we comment this now out, and leave that for
    # the future.

    # h = [kernel_matrix_new(s[end], k,  θ[i]) for (i,s) ∈ enumerate(sols)]
    # H = [inv(Symmetric(kernel_matrix_new(s[end], k, θ[i]))) * ZY_tr[:,i] for (i,s) ∈ enumerate(sols)]

    # Assume kernel does not change from what was specified when running code.
    jldsave(string(outf_basename, ".jld2"); DXμ, DXvecs, DXvals, DYμ, DYvecs, DYvals, sols, ZY_tr)

end

function load_subdb_info(fname)
    f = load(fname)
    return (f["sols"], f["ZY_tr"],
            DimensionReductionStruct(f["DXμ"], (vectors = f["DXvecs"], values = f["DXvals"])),
            DimensionReductionStruct(f["DYμ"], (vectors = f["DYvecs"], values = f["DYvals"])))
            # (μ = f["DXμ"], F = (vecs = f["DXvecs"], vals = f["DXvals"])),
            # (μ = f["DYμ"], F = (vecs = f["DYvecs"], vals = f["DYvals"])))
end

function predict_NPKF_from_disk(id, X_te; σ = 1., reg = 1e-7, segments = 2)
    sols, ZY_tr, D_X, D_Y = load_subdb_info(string("sub_", id, ".jld2"))
    ZX_te = KFCommon.original_to_reduced(X_te, D_X)

    trajs = [(u = s,) for s ∈ sols]

    ZY_te_pred = Predict.NPKF_predict(trajs, ZY_tr, ZX_te;
                                      return_unc = false,
                                      σ, reg, σ_factor = 1, segments)[2][1]
    KFCommon.reduced_to_original(ZY_te_pred, D_Y)
end

# function quicksolve_SGD(f, X_te)
#     # get_subdb_id(x_te)
#     H, sols, D_X, D_Y = load_subdb_info(string("sub", id, ".jld2"))
#     ZX_te = KFCommon.original_to_reduced(X_te, f)

#     # 1. warp ZX_te based on sols. Separate function?
#     # 2. do gp prediction based on pre-computed K-1 × y_te...
# end
