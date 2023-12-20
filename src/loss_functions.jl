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
export ρ_LOI, ρ_KF, ρ_LOO, ρ_MLE, ρ_RMSE, ρ_abs, ρ_L2_with_unc


# using Zygote
using LinearAlgebra
using Distances
using StatsBase

"""Version of ρ, where Nc = 1 and we average over all possible Xc."""
function ρ_LOI(Xₛ::AbstractArray{T}, yₛ::AbstractVector{Float64}, k::Function, logθ::AbstractVector; nXlinear::Int = 1) where T
    n = length(yₛ)
    Ω = kernel_matrix(Xₛ, k, logθ; nXlinear)

    # For the numerator, go over all combinations of size 1 for all
    # samples in Xₛ and average. Reduces to:
    num = yₛ' * yₛ / Ω[1] / n

    return 1. - num / (yₛ' * inv(Symmetric(Ω)) * yₛ)[1]
end


function ρ_LOI_2(Xₛ::AbstractArray{T}, yₛ::AbstractVector{Float64}, buf::AbstractArray{Float64}, k::Function, logθ::Vector{T}; nXlinear::Int = 1) where T
     Ω = kernel_matrix(Xₛ, k, logθ; nXlinear)

     return (yₛ' * inv(Symmetric(Ω)) * yₛ)[1]
end


"""Maximum likelihood."""
function ρ_MLE(Xₛ::AbstractArray{T}, yₛ::AbstractVector{Float64}, k::Function, logθ::Vector{T}; nXlinear::Int = 1) where T
    n = length(yₛ)
    Ω = kernel_matrix(Xₛ, k, logθ; nXlinear)

    # Whichever is faster; should be same result
    L = cholesky(Ω).U
    LI = inv(L)
    z = LI * yₛ

    a1 = .5 * z' * z
    # a2 = .5 * (yₛ' * inv(Symmetric(Ω)) * yₛ)
    # println("$a1, $a2")

    l1 = sum(log.(diag(LI)))
    # l2 = .5 * log(det(Ω))
    # println("$l1, $l2")

    ret1 = a1 - l1
    # ret2 = a2 + l2
    # println("Should be 0: $(ret1 - ret2)")

    return ret1
end


"""Original version, converges slower but also works"""
function ρ_KF(Xf::AbstractArray{T}, yf::AbstractArray{T}, k::Function, logθ::Vector{T}; nXlinear::Int = 1) where T
    Ω = kernel_matrix(Xf, k, logθ; nXlinear)
    Nc = size(Ω)[1] ÷ 2
    yc = @view yf[1:Nc]
    Ωc = Symmetric(Ω[1:Nc, 1:Nc])

    return 1. - ((yc' * inv(Ωc) * yc)[1] / (yf' * inv(Symmetric(Ω)) * yf)[1])
end


"""Original version, with complement subbatching, slightly improves on original."""
function ρ_complement(Xf::AbstractArray{T}, yf::AbstractArray{T}, k::Function, logθ::Vector{T}; nXlinear::Int = 1) where T
    Ω = kernel_matrix(Xf, k, logθ; nXlinear)

    nchunks = 2
    sr = KFCommon.splitrange(1, length(yf), nchunks + 1)
    chunks = [sr[i]:sr[i+1]-1 for i ∈ 1:length(sr)-1]

    tot = 0.0
    N = size(Ω)[1]
    n = N ÷ 2

    term(idx) = @views yf[idx]' * inv(Symmetric(Ω[idx, idx])) * yf[idx]

    for r ∈ chunks
       tot += term(r)
    end

    return 1. - tot / term(1:N)
end


"""Leave one out cross validation"""
function ρ_LOO(X::AbstractArray{Float64}, y::AbstractVector{Float64}, k::Function, logθ::AbstractVector{T}; nXlinear::Int = 1) where T
    Ω = kernel_matrix(X, k, logθ; nXlinear)
    Ω⁻¹ = inv(Ω)
    N = length(y)
    M = N * Ω⁻¹

    for i ∈ 1:N
        M = @views M - Ω⁻¹[:,i] * Ω⁻¹[:,i]' / Ω⁻¹[i,i]
    end

    return 1.0 * N - (y' * M * y) / (y' * Ω⁻¹ * y)
end


# Minimize cross-validated RMSE directly (L2 loss).
function ρ_RMSE(X::AbstractArray{T}, y::AbstractVector{Float64}, k::Function, logθ::AbstractVector; predictonlycenter::Bool = false, nXlinear::Int = 1) where T
    Ω = kernel_matrix(X, k, logθ; nXlinear)

    Ω⁻¹ = inv(Ω)
    n = length(y)

    # With predictonlycenter, points on the edges of the minibatch are
    # not predicted. May improve performance / accuracy.

    s = predictonlycenter ? sortperm(sum(Ω, dims = 2)[1:max(2n÷3, 50)], rev=true) : 1:n
    tot = 0.

    for i in s
        m = [1:i-1; i+1:n]
        t = @views (Ω[m,i]' * (Ω⁻¹ - Ω⁻¹[:,i] * Ω⁻¹[:,i]' / Ω⁻¹[i,i])[m,m] * y[m] - y[i])^2
        tot += t
    end

    return tot / n
end


function ρ_L2_with_unc(X::AbstractArray{T}, y::AbstractVector{Float64}, k::Function, logθ::AbstractVector; nXlinear::Int = 1, predictonlycenter::Bool = true) where T <: Real
    Ω = kernel_matrix(X, k, logθ; nXlinear)
    Ω⁻¹ = inv(Ω)
    n = length(y)
    L2tot = 0.0
    vartot = 0.0

    # Predict this many points closest to the center, or everything
    s = sortperm(sum(Ω, dims = 2)[:], rev = true)
    # Unlike with ρ_RMSE, one should not leave too many points out
    # here. Otherwise the standard deviations go wrong.
    M = predictonlycenter ? 95 * n ÷ 100 : n

    for (j,i) ∈ enumerate(s[1:M])
        m = [1:i-1; i+1:n]
        A = @views Ω[m,i]' * (Ω⁻¹ - Ω⁻¹[:,i] * Ω⁻¹[:,i]' / Ω⁻¹[i,i])[m,m]
        δ = @views A * y[m] - y[i]
        σ = @views Ω[i,i] - A * Ω[m,i]
        # println("δ: $δ, σ: $σ, z-score var: $(δ^2/σ)")
        L2tot +=  δ^2
        vartot += δ^2/σ
    end
    # println((vartot/(n-1) - 1.0)^2)

    # The first term below is the average squared error, as in
    # ρ_RMSE. The second one penalizes for any departure of the
    # z-score sample variance from unity.
    return L2tot / n + (vartot/(n-1) - 1.0)^2
end


"""Same function as ρ_RMSE, but absolute error instead of squared"""
function ρ_abs(X::AbstractArray{T}, y::AbstractVector{Float64}, k::Function, logθ::AbstractVector; predictonlycenter::Bool = false, nXlinear::Int = 1) where T
    Ω = kernel_matrix(X, k, logθ; nXlinear)
    Ω⁻¹ = inv(Ω)
    N = length(y)
    M = predictonlycenter ? 3 : N
    tot = 0.

    for i ∈ 1:M
        m = [1:i-1; i+1:N]
        tot +=  abs(Ω[m,i]' * (Ω⁻¹ - Ω⁻¹[:,i] * Ω⁻¹[:,i]' / Ω⁻¹[i,i])[m,m] * y[m] - y[i])
    end

    return tot / N
end


# function ρ_RMSE_localized(X::AbstractArray{T}, y::AbstractVector{Float64}, k::Function, logθ::Vector{T}, predictonlycenter::Bool = false) where T
#     # If predictonlycenter == true, predicts only the center point,
#     # around which training data has been sampled. Make sure that
#     # minibatch_method in Parametric.jl is set to "neighborhood".

#     D = @views k.(sum((X .- X[1,:]').^2, dims = 2); logθ = logθ)[:]
#     m2 = D .> 1e-9
#     nfeas = sum(m2) # number of feasible observations

#     # sample max ncloseobs points using covariances as weights.
#     ncloseobs = 64
#     if nfeas > ncloseobs
#         m4 = sortperm(D[m2])
#         b = cumsum(D[m2][m4])
#         m3 = m4[unique([searchsortedfirst(b, rand()*b[end]) for _ ∈ 1:ncloseobs])]
#     else
#         m3 = 1:nfeas
#     end

#     Ω = @views kernel_matrix(X[m2,:][m3,:], k, logθ)
#     Ω⁻¹ = inv(Ω)

#     y2 = y[m2][m3]
#     N = length(y2)
#     M = predictonlycenter ? min(1, N) : N
#     tot = 0.0

#     if N == 0 # Not sure why in rare cases X's are NaN's and then this all fails
#         println("Zero obs found in localization radius, N = $N and logθ = $logθ")
#         println("returning $(y[1]^2)")
#         return y[1]^2
#     end

#     # Compute the RMSE
#     for i ∈ 1:M
#         m = [1:i-1; i+1:N]
#         tot +=  @views (Ω[m,i]' * (Ω⁻¹ - Ω⁻¹[:,i] * Ω⁻¹[:,i]' / Ω⁻¹[i,i])[m,m] * y2[m] - y2[i])^2
#     end

#     return tot / M
# end


# # Does not produce the right results for some reason
# function ρ_RMSE_simplified(X::AbstractArray{T}, y::AbstractVector{Float64}; σ = 1., reg = 1e-6) where T

#     Ω = kernel_matrix(X; reg, σ)
#     Ω⁻¹ = inv(Ω)
#     w = Ω⁻¹ * y

#     tot = zero(T)
#     for i ∈ 1:length(y)
#         a = Ω⁻¹[:,i]' * Ω⁻¹[:,i] / Ω⁻¹[i,i]
#         # println(tot)
#         tot += @views (Ω⁻¹[:,i]' * (w - a' * y) - y[i])^2
#     end

#     return sqrt(tot / length(y))

# end
