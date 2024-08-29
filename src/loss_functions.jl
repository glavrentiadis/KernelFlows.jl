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
function ρ_LOI(X::AbstractArray{T}, y::AbstractVector{Float64}, k::AutodiffKernel, logθ::AbstractArray{T}) where T
    n = length(y)
    Ω = kernel_matrix(k, logθ, X)

    # For the numerator, go over all combinations of size 1 for all
    # samples in X and average. Reduces to:
    num = y' * y / Ω[1] / n

    return one(T) - num / (y' * inv(Symmetric(Ω)) * y)[1]
end


function ρ_LOI_2(X::AbstractArray{T}, y::AbstractVector{Float64}, buf::AbstractArray{Float64}, k::AutodiffKernel) where T
     Ω = kernel_matrix(k, logθ, X)

     return (y' * inv(Symmetric(Ω)) * y)[1]
end


"""Maximum likelihood."""
function ρ_MLE(X::AbstractArray{T}, y::AbstractVector{Float64}, k::AutodiffKernel, logθ::AbstractArray{T}) where T
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
function ρ_KF(X::AbstractArray{T}, y::AbstractArray{T}, k::AutodiffKernel, logθ::AbstractArray{T}) where T
    Ω = kernel_matrix(k, logθ, X)
    Nc = size(Ω)[1] ÷ 2
    yc = @view y[1:Nc]
    Ωc = Symmetric(Ω[1:Nc, 1:Nc])

    return one(T) - ((yc' * inv(Ωc) * yc)[1] / (y' * inv(Symmetric(Ω)) * y)[1])
end


"""Original version, with complement subbatching, slightly improves on original."""
function ρ_complement(X::AbstractArray{T}, y::AbstractArray{T}, k::AutodiffKernel, logθ::AbstractArray{T}) where T
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


"""Full leave-one-out cross validation"""
function ρ_LOO(X::AbstractArray{Float64}, y::AbstractVector{Float64}, k::AutodiffKernel, logθ::AbstractArray{T}) where T
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
function ρ_RMSE(X::AbstractArray{T}, y::AbstractVector{T}, k::AutodiffKernel, logθ::AbstractArray{T}; predictonlycenter::Bool = true) where T
    K = kernel_matrix(k, logθ, X)

    KI = inv(K)
    n = length(y)

    κ = κ_default
    s = collect(1:κ_default)
    tot = zero(T)

    for i in s
        m = [1:i-1; i+1:n]
        t = @views (K[m,i]' * (KI - KI[:,i] * KI[:,i]' / KI[i,i])[m,m] * y[m] - y[i])^2
        tot += t
    end

    return tot / κ
end


# Optimized version of the RMSE loss. This does fewer
# calculations. However, performance benefits as of now are minor, so
# further profiling is needed.
function ρ_RMSE_optimized(X::AbstractArray{T}, y::AbstractVector{T}, k::AutodiffKernel, logθ::AbstractArray{T}) where T
    K = kernel_matrix(k, logθ, X)
    L = cholesky(K)
    KI = inv(L)
    n = length(y)

    κ = κ_default
    s = collect(1:κ_default)

    tot = zero(T)

    r = KI * y

    for i in s
        m = [1:i-1; i+1:n]
        yi = y[i]
        Ki = @view K[m,i]
        KIi = @view KI[m,i]

        # second factor, latter product:
        c1 = r[i] - KI[i,i]*yi
        c2 = c1 / KI[i,i]
        c3 = c2 * dot(KIi, Ki)

        # THESE ARE GOOD
        # println("should be zero: $(dot(Ki, r[m] - yi * KIi) - dot(Ki, KI[m,m], y[m]))")
        t = @views dot(Ki, r[m] - yi * KIi)
        t -= c3
        cc = t - yi
        tot += (t - yi)^2
    end

    # tot2 = zero(T)
    # for i in s
    #     m = [1:i-1; i+1:n]
    #     t = @views (K[m,i]' * (KI - KI[:,i] * KI[:,i]' / KI[i,i])[m,m] * y[m] - y[i])^2
    #     # println(t)
    #     tot2 += t
    # end
    # println("Correct total: $(tot2/κ)")

    return tot / κ
end


# Minimize cross-validated RMSE directly (L2 loss).
function ρ_RMSE_resampled(X::AbstractArray{T}, y::AbstractVector{T}, k::AutodiffKernel, logθ::AbstractArray{T}) where T
    Ω = kernel_matrix(k, logθ, X)
    n = length(y)

    κ = κ_default
    s = collect(1:κ_default)

    tot = zero(T)
    for ss in s
        m = [1:ss-1; ss+1:n]
        println("m size: $(size(m))")
        w = abs.([Ω[ss,1:ss-1]..., zero(T), Ω[ss,ss+1:n]...])
        println("w size: $(size(w))")
        m2 = sample_without_replacement(w, κ÷2)
        Ω⁻¹ = inv(Ω[m2,m2])
        t = @views (Ω[m2,ss]' * Ω⁻¹ * y[m2] - y[ss])^2
        tot += t
    end

    return tot / n
end


"""Analytic derivatives-version of RMSE loss. Minibatch can be given
beforehand with s_LOO. If return_components is true, the function
returns each individual predicted observation (not residual), as well
as the gradient of each observation with respect to the parameters
logα"""
function ρ_RMSE(X::AbstractArray{T}, y::AbstractVector{T}, k::AnalyticKernel,
                logα::AbstractVector{T}, workbufs::Vector{Array{T}},
                Kgrads::Vector{Matrix{T}}; predictonlycenter::Bool = true,
                s_LOO::AbstractVector{Int} = Int[],
                return_components::Bool = false) where T <: Real

    n, nXdims = size(X)
    nα = length(logα)

    # Split workbuf into the buffers that are needed below
    K, KI, αgrad, vbuf1, vbuf2, vbuf3, k_Linbuf, k_Dbuf, k_Xbuf,
        k_vbuf1, k_vbuf2, k_Mbuf1, k_Mbuf2, k_Mbuf3 = workbufs
    k_workbufs = [k_Linbuf, k_Dbuf, k_Xbuf, k_vbuf1,
                  k_vbuf2, k_Mbuf1, k_Mbuf2, k_Mbuf3]

    # Get kernel matrix and its gradient wrt log in K and Kgrad.
    k.K_and_∂K∂logα!(X, logα, K, Kgrads, k_workbufs)

    κ = κ_default # set in minibatching.jl, optimally should be passed to ρ
    s_LOO = collect(1:κ)

    # Invert K
    KI .= K
    LAPACK.potrf!('U', KI)
    LAPACK.potri!('U', KI)
    LinearAlgebra.copytri!(KI, 'U')

    ρtot = 0.0

    # When training multiple MVMs together, buffers are sized
    # according to the largest one. Hence, different number of input
    # dimensions would lead to αgrad buffer of size of the largest
    # one. We ensure correct gradient buffer here.
    (length(αgrad) > nα) && (αgrad = @views αgrad[1:nα])
    αgrad .= 0.0

    hgrad = vbuf1
    g1 = vbuf2
    g2 = vbuf3
    hIi = k_vbuf2

    extrabufs = [col for col in eachcol(k_Mbuf1)]
    hi = extrabufs[2]

    Hyi = zeros(n)
    Khi = zeros(n)

    if return_components
        allgrads = zeros(nα, length(s_LOO))
        allpreds = zeros(length(s_LOO))
    end

    for (k,i) in enumerate(s_LOO)
        yi = y[i]
        y[i] = 0.0

        # Hyi .= KI * y
        BLAS.symv!('U', one(T), KI, y, zero(T), Hyi)

        hi .= @view K[:,i]
        hIi = @view KI[:,i]

        # Khi = KI * hi
        BLAS.symv!('U', one(T), KI, hi, zero(T), Khi)

        c = 1. / KI[i,i]
        chy = dot(hIi, y)
        chh = dot(hi, hIi)

        β = dot(hi, Hyi) - c * chh * chy - yi
        ρ = β^2

        for j in 1:nα
            hgrad .= @view Kgrads[j][:,i]
            Kgrad = Kgrads[j]
            Kgrad[:,i] .= 0
            Kgrad[i,:] .= 0

            # g = inv(K[m,m]) # This works for y' KI y LOO loss
            # αgrad[j] +=  - y[m]' * (g * Kgrad[m,m] * g) * y[m]

            # This works for y' KI y LOO loss
            # g2 = KI - c * hIi * hIi'
            # g2 = g2[m,m]
            # αgrad[j] +=  - y[m]' * (g2 * Kgrad[m,m] * g2) * y[m]

            # This still works  for y' KI y LOO loss
            # αgrad[j] +=  - y' * (g2 * Kgrad * g2) * y

            # This works for y' KI y LOO loss
            # g3 = Hyi - c * chy * hIi
            # αgrad[j] += -dot(g3, Kgrad, g3)

            # # This works for h' * KI * y loss
            # g1 = Khi - c * chh * hIi
            # g2 = Hyi - c * chy * hIi
            # t1 = dot(g1, Kgrad, g2)
            # t2 = dot(hgrad, g2)
            # αgrad[j] += -t1 + t2

            # This works for RMSE loss, (h' * KI * y - yi)^2
            # @time begin
                g1 .= hIi
                g1 .*= -c * chh
                g1 .+= Khi
                g2 .= hIi
                g2 .*= -c*chy
                g2 .+= Hyi
            # end
            # @time begin
            #     g1 .= Khi - c * chh * hIi
            #     g2 .= Hyi - c * chy * hIi
            # end

            # The BLAS version here is twice as fast as it uses the
            # fact that Kgrad is symmetric
            t1 = dot(g1, BLAS.symv!('U', one(T), Kgrad, g2, zero(T), extrabufs[1]))
            # t1 = dot(g1, Kgrad, g2)

            t2 = dot(hgrad, g2)
            αgrad[j] += 2 * β * (t2 - t1)

            return_components && (allgrads[j,k] = t2 - t1)

            Kgrads[j][:,i] .= hgrad
            Kgrads[j][i,:] .= hgrad
        end
        y[i] = yi
        ρtot += ρ

        return_components && (allpreds[k] = β + yi )

    end

    return_components && (return allpreds, allgrads)

    ρtot /= κ
    αgrad ./= κ

    return ρtot, αgrad
end




function ρ_L2_with_unc(X::AbstractArray{T}, y::AbstractVector{T}, k::AutodiffKernel,
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
    # L2tot / n + (vartot/(n-1) - one(T))^2
    one(T) / L2tot / vartot

end


"""Same function as ρ_RMSE, but absolute error instead of squared"""
function ρ_abs(X::AbstractArray{T}, y::AbstractVector{Float64}, k::AutodiffKernel, logθ::AbstractArray{T}; predictonlycenter::Bool = false) where T
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


"""This is for testing purposes only - e.g. to compare analytic and
autodiff results. Modify as needed to check correctness"""
function ρ_testloss(X::AbstractArray{T}, y::AbstractVector{T},
                    k::AutodiffKernel, logθ::AbstractArray{T}) where T <: Real
    n = length(y)
    tot = 0
    s = [3,5]
    K = kernel_matrix(k, logθ, X)
    for i in s
        m = [1:i-1...,i+1:n...]
        KI = inv(K[m,m])
        ym = y[m]
        h = K[m,i]
        tot += (h' * KI * ym - y[i])^2
    end
    return tot
end
