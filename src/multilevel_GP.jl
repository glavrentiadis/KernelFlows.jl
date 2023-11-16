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
export MVGPModel_twolevel

"""Construct twolevel model from an existing MVGPModel. Training
inputs and outputs need to be resupplied to this function, since
reconstructing them may or may not work, depending on what dimension
reduction was used in the first place. Especially the augmented
methods that are used for the input space can cause errors."""
function MVGPModel_twolevel(MVM::MVGPModel{T}, # Upper level model
                            X_tr::Matrix{T},   # Training inputs for MVM
                            Y_tr::Matrix{T};   # Training outputs for MVM
                            s_tr::Union{Nothing, AbstractVector{Int}} = nothing,
                            s_te::Union{Nothing, AbstractVector{Int}} = nothing,
                            kernel::Function = MVM.Ms[1].kernel,
                            dimreduceargs::NamedTuple = (nYCCA = 1, nYPCA = 1, nXCCA = 1)) where T <: Real

    # Divide data as given by s_tr or in half
    ntr = size(MVM.Ms[1].Z)[1]
    (s_tr == nothing) && (s_tr = randperm(ntr)[1:ntr÷2])
    (s_te == nothing) && (s_te = setdiff(1:ntr, s_tr))

    # Construct residual training data
    MVM1 = trim_MVGP_data(MVM, s_tr)
    X_tr_ste = X_tr[s_te,:] # Training inputs for new GP
    Y_tr_ste_pred = predict(MVM1, X_tr_ste)
    Ydiff_tr_ste = Y_tr[s_te,:] - Y_tr_ste_pred # Training labels for new GP

    augment_X_with_Y_residuals = false
    X_tr_new = augment_X_with_Y_residuals ? hcat(Y_tr_ste_pred, X_tr_ste) : X_tr_ste
    # New dimension reduction for residual model
    G = dimreduce(X_tr_new, Ydiff_tr_ste; dimreduceargs...)
    
    # Return second level model
    MVM2 = MVGPModel(X_tr_new, Ydiff_tr_ste, kernel, G; transform_zy = false)
end
