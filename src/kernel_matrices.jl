function sqr(x::T) where T <: Real
    sqrt(x+eps(T))
end


"""A Distances.pairwise() workalike, but works with Zygote"""
function pairwise_Euclidean(X::AbstractMatrix{T}) where T <: Real
    H = T(-2.) * X * X'
    D = T(.5) * diag(H)
    sqr.(Symmetric(H .- D .- D'))
end


"""Compute kernel matrix for unary kernels; no inversion. This
   function is autodifferentiable with Zygote and used for
   training."""
function kernel_matrix(k::UnaryKernel, logθ::AbstractVector{T}, X::AbstractArray{T}) where T <: Real

    # Linear component only for first X dimensions
    KK = @fastmath @views X[:,1:k.nXlinear] * X[:,1:k.nXlinear]'
    H1 = @fastmath pairwise_Euclidean(X)

    δ = max(exp(-15), exp(logθ[4]))
    H2 = @fastmath k.k.(H1, exp(logθ[1]), exp(logθ[2])) +
        Diagonal(δ * ones(T, size(X)[1])) + exp(logθ[3]) * KK
    H2
end


"""Compute kernel matrix for binary kernels; no inversion. This
   function is autodifferentiable with Zygote and used for
   training."""
function kernel_matrix(k::BinaryKernel, logθ::AbstractVector{T}, X::AbstractArray{T}) where T <: Real

    K = hcat([[k.k(x, y, exp.(logθ[1:end-1])) for y in eachrow(X)] for x in eachrow(X)]...)
    # Add nugget
    δ = max(exp(-15.), exp(logθ[4]))
    K += Diagonal(δ * ones(size(X)[1]))
    return K
end


"""Compute kernel matrix K for unary kernels, or if precision == true,
   its inverse. Not autodifferentiable, used for predictions"""
function kernel_matrix_fast!(k::UnaryKernel, θ::AbstractVector{T}, X::AbstractArray{T}, buf::AbstractMatrix{T}, outbuf::AbstractMatrix{T}; precision = true) where T <: Real
    s = Euclidean()

    pairwise!(s, buf, X, dims = 1)
    buf .= k.k.(buf, θ[1], θ[2])

    δ = max(exp(-15), θ[4])
    @views buf[diagind(buf)] .+= δ # max(T(exp(-15.)), θ[4])
    lf = θ[3] # linear kernel component weight

    # Linear component only sees first nXlinear dimensions of X
    XX = @views X[:,1:k.nXlinear]
    BLAS.gemm!('N', 'T', lf, XX, XX, one(T), buf)

    if precision
        # Symmetrize: MKL gives 1e-17 rounding errors -> not PD
        for i in 1:size(X)[1]
            for j in 1:i-1
                buf[i,j] = buf[j,i]
            end
        end

        L = cholesky!(buf)
        ldiv!(outbuf, L, UniformScaling(1.)(size(X)[1]))
    else
        outbuf .= buf
    end
end


function kernel_matrix_fast!(k::AnalyticKernel, θ::AbstractVector{T},
                             X::AbstractArray{T}, buf::AbstractMatrix{T}, outbuf::AbstractMatrix{T}; precision = true) where T <: Real

    # tmpbuf = zero(buf)
    # pw_and_linear!(X, outbuf, buf, θ[end-1])
    # Matern32!(outbuf, θ[1], θ[2], tmpbuf)

    # outbuf .+= buf
    # outbuf[diagind(outbuf)] .+= θ[end]

    k_pred = UnaryKernel(Matern32, T[], size(X)[2])
    kernel_matrix_fast!(k_pred, θ, X, buf, outbuf; precision)
end


"""Compute kernel matrix K for binary kernels, or if precision == true,
   its inverse. Not autodifferentiable, used for predictions"""
function kernel_matrix_fast!(k::BinaryKernel, θ::AbstractVector{T}, X::AbstractArray{T}, buf::AbstractMatrix{T}, outbuf::AbstractMatrix{T}; precision = true) where T <: Real

    n = size(X)[1]
    K = zeros(n,n)

    @inbounds for i in 1:n
        @inbounds for j in 1:i
            buf[i,j] = @views k.k(X[i,:], X[j,:], θ[1:end-1])
            buf[j,i] = buf[i,j]
        end
    end

    # Add nugget
    δ = max(exp(-15), exp(logθ[4]))
    buf[diagind(buf)] .+= δ

    if precision
        L = cholesky!(buf)
        ldiv!(outbuf, L, UniformScaling(1.)(n))
    else
        outbuf .= buf
    end

    outbuf
end
