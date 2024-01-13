function sqr(x::T) where T <: Real
    sqrt(x+1e-15)
end


"""A Distances.pairwise() workalike, but works with Zygote"""
function pairwise_Euclidean(X::AbstractMatrix{T}) where T <: Real
    H = -2. * X * X'
    D = .5 * diag(H)
    sqr.(Symmetric(H .- D .- D'))
end


"""Compute kernel matrix for unary kernels; no inversion. This
   function is autodifferentiable with Zygote and used for
   training."""
function kernel_matrix(k::UnaryKernel, logθ::AbstractVector, X::AbstractArray{T}) where T <: Real

    # Linear component only for first X dimensions
    KK = @views X[:,1:k.nXlinear] * X[:,1:k.nXlinear]'
    H1 = pairwise_Euclidean(X)
    H2 = k.k.(H1, exp(logθ[1]), exp(logθ[2])) +
        Diagonal(exp(max(logθ[4], -15.)) * ones(size(X)[1])) + exp(logθ[3]) * KK

    H2
end


"""Compute kernel matrix for binary kernels; no inversion. This
   function is autodifferentiable with Zygote and used for
   training."""
function kernel_matrix(k::BinaryKernel, logθ::AbstractVector{T}, X::AbstractArray{T}) where T <: Real

    K = hcat([[k.k(x, y, exp.(logθ[1:end-1])) for y in eachrow(X)] for x in eachrow(X)]...)
    # Add nugget
    K += Diagonal(exp(logθ[end]) * ones(size(X)[1]))
    return K
end


"""Compute kernel matrix K for unary kernels, or if precision == true,
   its inverse. Not autodifferentiable, used for predictions"""
function kernel_matrix_fast(k::UnaryKernel, θ::AbstractVector{T}, X::AbstractArray{T}, buf::AbstractMatrix{T}, outbuf::AbstractMatrix{T}; precision = true) where T <: Real
    s = Euclidean()

    pairwise!(s, buf, X, dims = 1)
    buf .= k.k.(buf, θ[1], θ[2])

    buf[diagind(buf)] .+= max(exp(-15.), θ[4])
    lf = θ[3] # linear kernel component weight

    # Linear component only sees first nXlinear dimensions of X
    XX = @views X[:,1:k.nXlinear]
    BLAS.gemm!('N', 'T', lf, XX, XX, 1., buf)

    if precision
        L = cholesky!(buf)
        ldiv!(outbuf, L, UniformScaling(1.)(size(X)[1]))
    else
        outbuf .= Symmetric(buf)
    end

    outbuf
end


"""Compute kernel matrix K for binary kernels, or if precision == true,
   its inverse. Not autodifferentiable, used for predictions"""
function kernel_matrix_fast(k::BinaryKernel, θ::AbstractVector{T}, X::AbstractArray{T}, buf::AbstractMatrix{T}, outbuf::AbstractMatrix{T}; precision = true) where T <: Real

    n = size(X)[1]
    K = zeros(n,n)

    @inbounds for i in 1:n
        @inbounds for j in 1:i
            buf[i,j] = @views k.k(X[i,:], X[j,:], θ)
            buf[j,i] = buf[i,j]
        end
    end

    # Add nugget
    buf[diagind(buf)] .+= max(exp(-15.), θ[end])

    if precision
        L = cholesky!(buf)
        ldiv!(outbuf, L, UniformScaling(1.)(n))
    else
        outbuf .= buf
    end

    outbuf
end
