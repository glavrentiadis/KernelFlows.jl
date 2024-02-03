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
function ρ_LOI(X::AbstractArray{T}, y::AbstractVector{Float64}, k::Kernel, logθ::AbstractArray{T}) where T
    n = length(y)
    Ω = kernel_matrix(k, logθ, X)

    # For the numerator, go over all combinations of size 1 for all
    # samples in X and average. Reduces to:
    num = y' * y / Ω[1] / n

    return one(T) - num / (y' * inv(Symmetric(Ω)) * y)[1]
end


function ρ_LOI_2(X::AbstractArray{T}, y::AbstractVector{Float64}, buf::AbstractArray{Float64}, k::Kernel) where T
     Ω = kernel_matrix(k, logθ, X)

     return (y' * inv(Symmetric(Ω)) * y)[1]
end


"""Maximum likelihood."""
function ρ_MLE(X::AbstractArray{T}, y::AbstractVector{Float64}, k::Kernel, logθ::AbstractArray{T}) where T
    n = length(y)
    Ω = kernel_matrix(k, logθ, X)

    # Whichever is faster; should be same result
    L = cholesky(Ω).U
    LI = inv(L)
    z = LI * y

    a1 = T(.5) * z' * z
    # a2 = .5 * (y' * inv(Symmetric(Ω)) * y)
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
function ρ_KF(X::AbstractArray{T}, y::AbstractArray{T}, k::Kernel, logθ::AbstractArray{T}) where T
    Ω = kernel_matrix(k, logθ, X)
    Nc = size(Ω)[1] ÷ 2
    yc = @view y[1:Nc]
    Ωc = Symmetric(Ω[1:Nc, 1:Nc])

    return one(T) - ((yc' * inv(Ωc) * yc)[1] / (y' * inv(Symmetric(Ω)) * y)[1])
end


"""Original version, with complement subbatching, slightly improves on original."""
function ρ_complement(X::AbstractArray{T}, y::AbstractArray{T}, k::Kernel, logθ::AbstractArray{T}) where T
    Ω = kernel_matrix(k, logθ, X)

    nchunks = 2
    sr = splitrange(1, length(y), nchunks + 1)
    chunks = [sr[i]:sr[i+1]-1 for i ∈ 1:length(sr)-1]

    tot = 0.0
    N = size(Ω)[1]
    n = N ÷ 2

    term(idx) = @views y[idx]' * inv(Symmetric(Ω[idx, idx])) * y[idx]

    for r ∈ chunks
       tot += term(r)
    end

    return 1. - tot / term(1:N)
end


"""Leave one out cross validation"""
function ρ_LOO(X::AbstractArray{Float64}, y::AbstractVector{Float64}, k::Kernel, logθ::AbstractArray{T}) where T
    Ω = kernel_matrix(k, logθ, X)
    Ω⁻¹ = inv(Ω)
    N = length(y)
    M = N * Ω⁻¹

    for i ∈ 1:N
        M = @views M - Ω⁻¹[:,i] * Ω⁻¹[:,i]' / Ω⁻¹[i,i]
    end

    return one(T) * N - (y' * M * y) / (y' * Ω⁻¹ * y)
end


# Minimize cross-validated RMSE directly (L2 loss).
function ρ_RMSE(X::AbstractArray{T}, y::AbstractVector{T}, k::Kernel, logθ::AbstractArray{T}; predictonlycenter::Bool = true) where T
    Ω = kernel_matrix(k, logθ, X)

    Ω⁻¹ = inv(Ω)
    n = length(y)

    # With predictonlycenter, only κ best-informed points are
    # predicted, discarding anything on the edges. May improve
    # performance / accuracy.
    κ = max(min(n÷5, 50),4)
    s = predictonlycenter ? sortperm(sum(Ω, dims = 2)[1:κ], rev=true) : 1:n

    tot = zero(T)
    for i in s
        m = [1:i-1; i+1:n]
        t = @views (Ω[m,i]' * (Ω⁻¹ - Ω⁻¹[:,i] * Ω⁻¹[:,i]' / Ω⁻¹[i,i])[m,m] * y[m] - y[i])^2
        tot += t
    end

    return tot / n
end


function ρ_L2_with_unc(X::AbstractArray{T}, y::AbstractVector{T}, k::Kernel,
                       logθ::AbstractArray{T}; predictonlycenter::Bool = true) where T <: Real
    Ω = kernel_matrix(k, logθ, X)
    Ω⁻¹ = inv(Ω)
    n = length(y)
    L2tot = zero(T)
    vartot = zero(T)

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
    L2tot / n + (vartot/(n-1) - one(T))^2
end


"""Same function as ρ_RMSE, but absolute error instead of squared"""
function ρ_abs(X::AbstractArray{T}, y::AbstractVector{Float64}, k::Kernel, logθ::AbstractArray{T}; predictonlycenter::Bool = false) where T
    Ω = kernel_matrix(k, logθ, X)
    Ω⁻¹ = inv(Ω)
    N = length(y)
    M = predictonlycenter ? 3 : N
    tot = zero(T)

    for i ∈ 1:M
        m = [1:i-1; i+1:N]
        tot +=  abs(Ω[m,i]' * (Ω⁻¹ - Ω⁻¹[:,i] * Ω⁻¹[:,i]' / Ω⁻¹[i,i])[m,m] * y[m] - y[i])
    end

    return tot / N
end
