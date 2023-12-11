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
export TwoLevelMVGP

struct TwoLevelMVGP{T}
    MVM1::MVGPModel{T} # First-level model
    MVM2::MVGPModel{T} # Second-level model
    projvecs::Union{Nothing, Matrix{T}} # Projection vector for augmenting inputs for 2nd level
end


"""Construct twolevel model from an existing MVGPModel. Training
inputs and outputs need to be resupplied to this function, since
reconstructing them may or may not work, depending on what dimension
reduction was used in the first place. Especially the augmented
methods that are used for the input space can cause errors."""
function TwoLevelMVGP(MVM1::MVGPModel{T}, # Upper level model
                      X_tr::Matrix{T},   # Training inputs for second-level MVM
                      Y_tr::Matrix{T};   # Training outputs for second-level MVM
                      kernel::Function = MVM1.Ms[1].kernel,
                      dimreduceargs::NamedTuple = (nYCCA = 1, nYPCA = 1, nXCCA = 1),
                      nvec_for_X_aug::Int = 0,
                      transform_zy = false) where T <: Real

    Y_tr_pred = predict(MVM1, X_tr)
    Ydiff_tr = Y_tr - Y_tr_pred # Training labels for new GP

    if nvec_for_X_aug > 0
        Y_projvecs = get_PCA_vectors(Y_tr_pred, nvec_for_X_aug)[1]
        display(Y_projvecs)
        X_tr_new = hcat(Y_tr_pred * Y_projvecs, X_tr)
    else
        Y_projvecs = nothing
        X_tr_new = X_tr
    end

    # New dimension reduction for residual model
    G = dimreduce(X_tr_new, Ydiff_tr; dimreduceargs...)

    # Return second level model
    MVM2 = MVGPModel(X_tr_new, Ydiff_tr, kernel, G; transform_zy)
    TwoLevelMVGP(MVM1, MVM2, Y_projvecs)
end


function LOO_predict_training(MVM::MVGPModel{T}) where T <: Real

    ndata = size(MVM.Ms[1].Z)[1]
    nM = length(MVM.Ms)
    buf1s = [zeros(ndata, ndata) for _ in 1:nM]
    buf2s = [zeros(ndata, ndata) for _ in 1:nM]

    ZY_pred = zeros(ndata, nM)
    Threads.@threads for (j,M) in collect(enumerate(MVM.Ms))
        Ω⁻¹ = kernel_matrix_fast(M.Z, buf1s[j], buf2s[j], M.kernel, M.θ, nXlinear = MVM.G.Xprojs[j].spec.nCCA, precision = true)
        Ω = Symmetric(buf1s[j]')[:,:]

        buf = zeros(ndata - 1, ndata - 1)
        buff = zeros(ndata, ndata)
        z = zeros(ndata)
        b = zeros(ndata)
        b2 = zeros(ndata)

        for i in 1:ndata
            (i % 100 == 0) && println(i)
            m = [1:i-1; i+1:ndata]
            buff .= Ω⁻¹
            b .= @view Ω⁻¹[:,i]
            b2 .= @view Ω[:,i]
            cc = -1. /b[i]
            BLAS.ger!(cc, b, b, buff)
            buff[:,i] .= 0.
            buff[i,:] .= 0.
            @views  BLAS.symv!('U', 1., buff, b2, 0., z)
            ZY_pred[i,j] =  z' * M.ζ

            # ZY_pred[i,j] = @time @views Ω[m,i]' * (Ω⁻¹ - Ω⁻¹[:,i] * Ω⁻¹[:,i]' / Ω⁻¹[i,i])[m,m] * M.ζ[m] # - M.ζ[i]
        end
    end

    Y_tr_pred_LOO = recover_Y(ZY_pred, MVM.G)


end


function train!(MVT::TwoLevelMVGP{T}, ρ::Function; trainingargs::NamedTuple) where T <: Real
    train!(MVT.MVM1, ρ; trainingargs...)
    train!(MVT.MVM2, ρ; trainingargs...)
end


function predict(MVT::TwoLevelMVGP{T}, X::AbstractArray{T}) where T <: Real
    Y1 = predict(MVT.MVM1, X)
    X2 = MVT.projvecs == nothing ? X : hcat(Y1 * MVT.projvecs, X)
    Y2 = predict(MVT.MVM2, X2)
    Y1, Y2
end
