#  Copyright 2023-2024 California Institute of Technology
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


# using IntelVectorMath

function Matern32!(D::AbstractMatrix{T}, a::T, b::T, buf::AbstractMatrix{T}) where T <: Real
    D .*= -sqrt(T(3.)) / b
    buf .= D
    # IVM.exp!(buf)  # 4x faster on Intel CPUs
    buf .= exp.(buf) # slower but works on M-series Macs
    D .-= one(T)
    D .*= -a
    D .*= buf
end


# pairwise for elements of A. In A the data is in rows.
function pw_and_linear!(A::AbstractMatrix{T}, out1::AbstractMatrix{T}, out2::AbstractMatrix{T}, γ::T) where T <: Real
    BLAS.syrk!('U', 'N', T(-2), A, zero(T), out1)
    out2 .= out1 # copy to out2
    out2 .*= -γ/T(2) # multiply out2 by -γ/2
    A .*= A # square of A
    a = sum(A, dims = 2)
    out1 .+= a
    out1 .+= a'
    # display(out1)
    # precision error may lead to tiny small values on diagonal
    out1[diagind(out1)] .= 0
    out1 .+= 1e-14 # we may get tiny negative values somewhere
    out1 .= sqrt.(out1) # Euclidean distance
end


"""Returns both the kernel matrix and its derivatives with respect to logα"""
function Matern32_αgrad!(X::AbstractMatrix{T}, logα::AbstractVector{T},
                         K::AbstractMatrix{T}, Kgrads::Vector{Matrix{T}},
                         workbufs::Vector{Array{T}}) where T <: Real

    n = size(X)[1]
    nα = length(logα)

    Linbuf, Dbuf, Xbuf, vbuf1, vbuf2, Mbuf1, Mbuf2, Mbuf3 = workbufs

    if nα < length(vbuf1)
        vbuf1 = @view vbuf1[1:nα]
        Xbuf = @view Xbuf[:,1:nα-4]
    end

    # Shorthands:
    α = vbuf1
    α .= exp.(logα)
    λ = @view α[1:nα-4]
    nλ = length(λ)
    θ = @view α[nα-3:nα]
    nθ = length(θ)

    Xbuf .= X
    Xbuf .*= λ'

    pw_and_linear!(Xbuf, Dbuf, Linbuf, α[end-1])
    Xbuf .= X # restore Xbuf to original scaled distances
    Xbuf .*= λ'
    Linbuf .= Symmetric(Linbuf)
    Dbuf .= Symmetric(Dbuf)
    Mbuf3 .= Dbuf # save the distances before computing Matern

    # Compute the Matern32 kernel, output in Dbuf. After computation,
    # the Matern32 exponent (exp(-sqrt(3)/2*d)) will be in Mbuf1
    Matern32!(Dbuf, θ[1], θ[2], Mbuf1)

    c = T(-3)*θ[1]/(θ[2]^2)
    Mbuf1 .*= c

    # Mbuf4 = K # K is overwritten at the  end so we can use it as buffer until then
    v2 = vbuf3 = similar(vbuf2)

    for i in 1:nλ
        vbuf2 .= @view Xbuf[:,i]

        # Get squared distances between vbuf elements in Mbuf2
        BLAS.syrk!('U', 'N', T(-2), vbuf2, zero(T), Mbuf2)
        @views v2 .= -.5*Mbuf2[diagind(Mbuf2)] # squared elements
        Mbuf2 .= Mbuf2 .+ v2 .+ v2' # Now we have the squared distances
        Mbuf2 .*= Mbuf1

        # Add gradient of linear kernel:
        BLAS.syrk!('U', 'N', T(2*α[end-1]), vbuf2, one(T), Mbuf2)
        # BLAS.ger!(T(2*α[end-1]), vbuf2, vbuf2, Mbuf2)
        Kgrads[i] .= Symmetric(Mbuf2)
        # LinearAlgebra.copytri!(Kgrads[i], 'U')
    end

    # RBF weight derivative. Note that multiplication of the Matern by
    # the weight θ[1] is already included in Dbuf.
    Kgrads[nλ+1] .= Dbuf

    # Derivative wrt RBF common scaling parameter ("b" in Matern32)
    Mbuf3 .*= Mbuf3 # squared pairwise distances
    Mbuf3 .*= Mbuf1 # after Matern32! Mbuf1 has exp(-sqrt(3)d/θ[2])
    Mbuf3 .*= -one(T)
    Kgrads[nλ+2] .= Mbuf3 # Copy derivative where it belongs

    # Derivative wrt linear kernel weight
    # Kgrads[nλ+3] .= α[end-1] * Xbuf * Xbuf'

    BLAS.syrk!('U', 'N', α[nλ+3], Xbuf, zero(T), Kgrads[nλ+3])
    LinearAlgebra.copytri!(Kgrads[nλ+3], 'U')

    # Derivative wrt nugget weight
    @views Kgrads[nλ+4][diagind(Kgrads[nλ+4])] .= α[end]

    debug = false
    if debug
        println("Analysis of autodiff and analytical gradients:")
        linearpart(x1, x2, λ, a) = a*((x1 .* λ)' * (x2 .* λ))
        nugget(x1, x2, α) = sum((x1-x2).^2) == 0. ? α : zero(T)
        ∇f(x1, x2, logα) = Zygote.gradient(logα ->
            Matern32(sqrt(sum(((x1 - x2) .* exp.(logα[1:length(x1)])).^2)), exp(logα[end-3]), exp(logα[end-2])) +
            linearpart(x1, x2, exp.(logα[1:end-4]), exp(logα[end-1])) +
            nugget(x1, x2, exp(logα[end])), logα)[1]

        function prints(i, j, X)
            x1 = X[i,:]
            x2 = X[j,:]
            v1 = ∇f(x1, x2, logα)
            v2 = [Kg[i,j] for Kg in Kgrads]
            println("ratio of gradients: should be one: last should be one only if i==j")
            display((v1 ./ v2)')
        end

        refgrad = zeros(n,n,nα)
        for i in 1:n
            for j in 1:n
                refgrad[i,j,:] .= ∇f(X[i,:], X[j,:], logα)
            end
        end

        jldopen("debug.jld2", "w") do file
            G = file.root_group
            G["autodiff_dKda"] = refgrad
            G["analyticdiff_dKda"] = stack(Kgrads)
        end

        prints(1, 2, X)
        prints(5, 6, X)
        prints(4, 4, X)
    end

    # Copy covariance to the proper matrix
    K .= Dbuf
    K .+= Linbuf

    K[diagind(K)] .+= α[end]

    # K seems to be correct here, as are both its components: Dbuf and Linbuf

    return K, Kgrads
end
