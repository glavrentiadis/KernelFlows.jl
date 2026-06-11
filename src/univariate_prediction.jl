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


abstract type AbstractPredictionBuffer{T} end

struct StandardPredictionBuffer{T} <: AbstractPredictionBuffer{T}
    M_cross::Matrix{T} # cross covariance matrix storage
    M_uqprod::Matrix{T} # matrix for computing uncertainties
    M_lin::Matrix{T} # linear part storage
    M_Xtr::Matrix{T} # X1 squared
    M_Xte::Matrix{T} # X2 squared
    v_ntr::Vector{T}
    v_nte::Vector{T}
    nte::Int # prediction batch size
end


struct FallbackPredictionBuffer{T} <: AbstractPredictionBuffer{T}
    M_cross::Matrix{T} # cross covariance matrix storage
end


function zero!(pb::StandardPredictionBuffer{T}) where T <: Real
    pb.v_nte .= zero(T)
    pb.v_ntr .= zero(T)
    pb.M_Xtr .= zero(T)
    pb.M_Xte .= zero(T)
    pb.M_lin .= zero(T)
    pb.M_uqprod .= zero(T)
end


"""GP prediction with Model (univariate output). By default M.λ will
be applied, namely,

   Z = M.λ' * Z

This can be overruled by setting apply_λ to false. Note that in this
function inputs (Z) need to be given in reduced coordinates.
"""
function predict(M::GPModel{T}, Z::AbstractMatrix{T}, pb::AbstractPredictionBuffer{T};
                 apply_λ::Bool = true,
                 apply_zyinvtransf::Bool = true,
                 outbuf::Union{Nothing, AbstractVector{T}} = nothing,
                 outbuf_uq::Union{Nothing, AbstractVector{T}} = nothing,
                 quantify_uncertainties::Bool = false) where T <: Real

    apply_λ && (Z .*= M.λ')

    # Allocate if buffers not given
    (outbuf == nothing) && (outbuf = zeros(T, size(X)[1]))
    ((outbuf_uq == nothing) && (quantify_uncertainties)) && (outbuf_uq = zeros(T, size(X)[1]))

    cross_covariance_matrix!(M.kernel, M.θ, Z, M.Z, pb)
    mul!(outbuf, pb.M_cross, M.h)

    # UQ portion
    if quantify_uncertainties
        h = marginal_variance(M, Z)
        uqres = uq!(M.IUM, pb, h, outbuf_uq) #pb.M_cross, h, pb.M_uqprod)
    else
        uqres = DummyUQResult()
    end

    apply_zyinvtransf && (outbuf .= M.zyinvtransf.(outbuf))

    # N.B. zyinvtransf is not applied to uqres! Use non-Gaussian UQ instead.
    return (outbuf, uqres)
end


function marginal_variance(M::GPModel{T}, Z::AbstractMatrix{T}) where T <: Real
    nte = size(Z)[1]
    h = zeros(T, size(Z)[1])
    h .= exp(T(-12)) + M.θ[4] + M.kernel.k(zero(T), M.θ[1], M.θ[2])
    for i in 1:nte
        h[i] += @views M.θ[3] * dot(Z[i,:], Z[i,:])
    end
    h
end


"""Compute cross-covariance matrix between X1 and X2. Typically this
would be between test inputs X (X1) and training data in GPModel.Z
(X2). The covariance matrix ends up in pb.M_cross."""
function cross_covariance_matrix!(k::UnaryKernel, θ::AbstractVector{T},
                                  X1::AbstractMatrix{T}, X2::AbstractMatrix{T},
                                  pb::StandardPredictionBuffer{T}) where T <: Real


    # FIXME: Use the following instead:
    # pairwise_Euclidean!(X1, X2, pb)
    # pb.M_cross .= k.k.(pb.M_cross, θ[1], θ[2])
    # pb.M_cross .+= θ[3]/T(-2) .* pb.M_lin

    s = Euclidean()
    wb = pb.M_cross # shorthand

    pairwise!(s, wb, X1, X2, dims = 1)
    wb .= @fastmath k.k.(wb, θ[1], θ[2])

    # mul! won't accept AbstractArrays, but gemm! does not mind
    @fastmath @views BLAS.gemm!('N', 'T', θ[3], X1[:,1:k.nXlinear],
                                X2[:,1:k.nXlinear], one(T), wb)
end


function cross_covariance_matrix!(k::BinaryKernel, θ::AbstractVector{T},
                                  X1::AbstractMatrix{T}, X2::AbstractMatrix{T},
                                  pb::FallbackPredictionBuffer{T}) where T <: Real

    (n,m) = size(pb.M_cross)
    @inbounds for i in 1:n
        @inbounds for j in 1:m
            pb.M_cross[i,j] = @views k.k(X1[i,:], X2[j,:], θ[1:end-1])
        end
    end
end


"""Use the regular Matern32 covariance function for prediction"""
function cross_covariance_matrix!(k::AnalyticKernel, θ::AbstractVector{T},
                                  X1::AbstractMatrix{T}, X2::AbstractMatrix{T},
                                  pb::StandardPredictionBuffer{T}) where T <: Real


    pairwise_Euclidean!(X1, X2, pb)
    pb.M_cross .= Matern32.(pb.M_cross, θ[1], θ[2])
    pb.M_cross .+= θ[3]/T(-2) .* pb.M_lin

    # Uncomment to validate implementation accuracy
    # hh = pb.M_cross[:,:]
    # k_pred = UnaryKernel(Matern32, T[], size(X1)[2])
    # cross_covariance_matrix!(k_pred, θ, X1, X2, pb.M_cross)

    # display(hh - pb.M_cross)
end


# Pairwise distances with minimal allocations to aid multi-threaded
# prediction performance
function pairwise_Euclidean!(X1::AbstractMatrix{T}, X2::AbstractMatrix{T},
                             pb::StandardPredictionBuffer{T}) where T <: Real

    pb.M_Xte .= X1.^2
    pb.M_Xtr .= X2.^2

    (nte, ntr) = size(pb.M_cross)
    nXdims = size(X1)[2]

    pb.v_nte .= zero(T)
    @inbounds for i in 1:nte
        for k in 1:nXdims
            pb.v_nte[i] += pb.M_Xte[i,k]
        end
    end

    pb.v_ntr .= zero(T)
    @inbounds for i in 1:ntr
        for k in 1:nXdims
            pb.v_ntr[i] += pb.M_Xtr[i,k]
        end
    end

    pb.M_cross .= pb.v_nte
    pb.M_cross .+= pb.v_ntr'
    BLAS.gemm!('N', 'T', T(-2), X1, X2, T(0), pb.M_lin) # -2 X1 * X2'
    pb.M_cross .+= pb.M_lin
    pb.M_cross[pb.M_cross .< zero(T)] .= zero(T)
    pb.M_cross .= sqrt.(pb.M_cross)
end
