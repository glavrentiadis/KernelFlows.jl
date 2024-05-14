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
    # Projection vectors for augmenting inputs with 1st level
    # predictions for 2nd level:
    projvecs::Union{Nothing, Matrix{T}}
end

struct MultilevelGP{T}
    MVMs::Vector{MVGPModel{T}}
end

function MultilevelGP(X_tr::Matrix{T}, Y_tr::Matrix{T},
                      dimreduceargs::Union{NamedTuple, Vector{<:NamedTuple}},
                      trainingargs::Union{NamedTuple, Vector{<:NamedTuple}},
                      kernels::Union{H1, Vector{H1}},
                      ρ::Function;
                      transform_zy::Union{Bool, Vector{Bool}} = false,
                      nlev::Int = 2) where {T<:Real, H1<:Function}
    # Helper function to get the arguments from vectors
    b(t::Union{T, Vector{T}}, i::Int) where T = typeof(t) <: Vector ? t[i] : t

    MVMs = MVGPModel{T}[]
    Y = Y_tr
    for i in 1:nlev
        da = b(dimreduceargs, i)
        ta = b(trainingargs, i)
        k = b(kernels, i)
        t_zy = b(transform_zy, i)
        G = dimreduce(X_tr, Y; da...)
        MVM = MVGPModel(X_tr, Y, k, G; transform_zy = t_zy)
        train!(MVM, ρ; ta...)
        push!(MVMs, MVM)
        (i < nlev) && (Y .-= LOO_predict_training(MVM))
    end

    return MultilevelGP(MVMs)
end


function predict(MLGP::MultilevelGP{T}, X::AbstractArray{T}) where T <: Real
    Y_te_pred = predict(MLGP.MVMs[1], X)
    for MVM in MLGP.MVMs[2:end]
        Y_te_pred .+= predict(MVM, X)
    end
    Y_te_pred
end


"""Construct twolevel model from an existing MVGPModel. Training
inputs and outputs need to be resupplied to this function, since
reconstructing them may or may not work, depending on what dimension
reduction was used in the first place. Especially the augmented
methods that are used for the input space can cause errors."""
function TwoLevelMVGP(MVM1::MVGPModel{T}, # Upper level model
                      X_tr::Matrix{T},   # Training inputs for second-level MVM
                      Y_tr::Matrix{T};   # Training outputs for second-level MVM
                      kernel::Symbol = :Matern32,
                      dimreduceargs::NamedTuple = (nYCCA = 3, nYPCA = 3, nXCCA = 1),
                      nvec_for_X_aug::Int = 0,
                      npredict_tr::Int = 500,
                      transform_zy = false) where T <: Real

    (Y_tr_pred_LOO, s_LOO) = LOO_predict_training(MVM1; npredict = npredict_tr)
    Ydiff_tr = Y_tr[s_LOO,:] - Y_tr_pred_LOO # Training labels for new GP

    if nvec_for_X_aug > 0
        Y_projvecs = get_PCA_vectors(Y_tr_pred_LOO, nvec_for_X_aug)[1]
        X_tr_new = hcat(Y_tr_pred_LOO * Y_projvecs, X_tr[s_LOO,:])
        # X_tr_new = Y_tr_pred_LOO * Y_projvecs # Y only
    else
        Y_projvecs = nothing
        X_tr_new = X_tr[s_LOO,:]
    end

    G = dimreduce(X_tr_new, Ydiff_tr; dimreduceargs...)

    # Return second level model
    MVM2 = MVGPModel(X_tr_new, Ydiff_tr, kernel, G; transform_zy)
    TwoLevelMVGP(MVM1, MVM2, Y_projvecs)
end


function LOO_predict_training(MVM::MVGPModel{T}; npredict::Int = length(MVM.Ms[1].ζ)) where T <: Real

    ndata = size(MVM.Ms[1].Z)[1]
    nM = length(MVM.Ms)
    nt = Threads.nthreads()
    buf1s = [zeros(ndata, ndata) for _ in 1:nM]
    buf2s = [zeros(ndata, ndata) for _ in 1:nM]

    ZY_pred = zeros(npredict, nM)
    s = randperm(ndata)[1:npredict]
    println("Predicting $npredict training data with leave-one-out")
    computed = zeros(Int, nt)
    Threads.@threads :static for (j,M) in collect(enumerate(MVM.Ms))
        kernel_matrix_fast!(M.kernel, M.θ, M.Z, buf1s[j], buf2s[j]; precision = true)
        Ω⁻¹ = buf2s[j]
        Ω = Symmetric(buf1s[j]')[:,:]

        buf = zeros(ndata - 1, ndata - 1)
        buff = zeros(ndata, ndata)
        z = zeros(ndata)
        b = zeros(ndata)
        b2 = zeros(ndata)

        for k in 1:npredict
            if k % 100 == 0
                print("\rComputed $(sum(computed)) out of $(nM * npredict) points ")
            end

            i = s[k]
            m = [1:i-1; i+1:ndata]
            buff .= Ω⁻¹
            b .= @view Ω⁻¹[:,i]
            b2 .= @view Ω[:,i]
            cc = -1. /b[i]
            BLAS.ger!(cc, b, b, buff)
            buff[:,i] .= 0.
            buff[i,:] .= 0.
            @views  BLAS.symv!('U', 1., buff, b2, 0., z)
            ZY_pred[k,j] =  z' * M.ζ
            computed[Threads.threadid()] += 1
        end
    end
    println()

    Y_tr_pred_LOO = recover_Y(ZY_pred, MVM.G)

    return Y_tr_pred_LOO, s[1:npredict]
end


function train!(MVT::TwoLevelMVGP{T}, ρ::Function; trainingargs::NamedTuple) where T <: Real
    train!(MVT.MVM1, ρ; trainingargs...)
    train!(MVT.MVM2, ρ; trainingargs...)
end


function predict(MVT::TwoLevelMVGP{T}, X::AbstractArray{T}) where T <: Real
    Y1 = predict(MVT.MVM1, X)
    X2 = MVT.projvecs == nothing ? X : hcat(Y1 * MVT.projvecs, X)
    # X2 = MVT.projvecs == nothing ? X : Y1 * MVT.projvecs # Y only
    Y2 = predict(MVT.MVM2, X2)
    Y1, Y2
end
