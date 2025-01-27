# Path Kernel Function
#------------------------------
# using KernelFunctions: Kernel as KFKernel
# using KernelFunctions: kernel as KFkernel
# using KernelFunctions: kappa as KFkappa
# using KernelFunctions: kernelmatrix as KFkernelmatrix
# using KernelFunctions
import KernelFunctions
using FastGaussQuadrature: gausslobatto
using StaticArrays

"""
    PathKernel

Derived kernel based on integration.

# Arguments
- `n`: number of points to integrate
- `kernel`: base kernel
- `normalized`: flag for normalized variant w.r.t. path length
- `t`: gauss integration points (1st axis)
- `s`: gauss integration points (2nd axis)
- `w`: gauss weights 
"""
struct PathKernel <: KernelFunctions.Kernel
    n::Int          
    # kernel
    kernel::KernelFunctions.Kernel
    normalized::Bool  
    #2d integration parameters
    t1D::AbstractVector{<:Real}
    w1D::AbstractVector{<:Real}
    #2d integration parameters
    t2D::AbstractVector{<:Real}
    s2D::AbstractVector{<:Real}
    w2D::AbstractVector{<:Real}
    #2d difference integration parameters
    td2D::AbstractVector{<:Real}
    wd2D::AbstractVector{<:Real}

    #constructor
    function PathKernel(n::Int, 
                        kernel::KernelFunctions.Kernel, 
                        normalized::Bool)

        #compute 1D quadrature points
        t1D, w1D = gaussquad1d(n)
        #compute 2D quadrature points
        t2D, s2D, w2D = gaussquad2d(n)
        #compute 2D difference quadrature points
        td2D, wd2D = gaussdiff2d(abs.(t2D - s2D), w2D)
        
        new(n, kernel, normalized, t1D, w1D, t2D, s2D, w2D, td2D, wd2D)
    end
end

"""Constructor for Path kernel"""
function (κ::PathKernel)(x::AbstractVector{T}, y::AbstractVector{T}) where T <: Real

    #parse coordinates, assuming x = [x1; x2] and y = [y1; y2]
    n = div(length(x), 2) #dimension
    x1 = @view x[begin:n]
    x2 = @view x[(n + 1):end]
    y1 = @view y[begin:n]
    y2 = @view y[(n + 1):end]
    #path lengths
    x_len = euclidean_len(x1, x2)
    y_len = euclidean_len(y1, y2)

    #distance between points on paths
    fun_dist = getdist(x1, x2, y1, y2)

    #underling kernel function, distance metric parametrization 
    # κ_b(d) = KernelFunctions.kappa(κ.kernel, d)
    # κ_b(d) = κ.kernel(d)

    #compute distance
    d = fun_dist.(κ.t2D, κ.s2D)

    #integrate path kernel
    # K = compute_kernel_integral(κ.kernel, d, κ.w2D) 
    K = compute_kernel_integral_ts(κ.kernel, fun_dist, κ.t2D, κ.s2D, κ.w2D) 

    #normalization
    if κ.normalized
        #normalize kernel
        K /= sqrt( compute_kernel_integral_t(κ.kernel, x_len, κ.td2D, κ.wd2D) * 
                   compute_kernel_integral_t(κ.kernel, y_len, κ.td2D, κ.wd2D) )
    else
        K *= x_len * y_len
    end

    return K
end

"""kernelmatrix overload for PathKernel and matrix X and Y"""
function KernelFunctions.kernelmatrix(κ::PathKernel, X::AbstractMatrix{T}, Y::AbstractMatrix{T}) where T <: Real
    n = size(X, 2)
    m = size(Y, 2)
    K = Matrix{T}(undef, n, m)
    @inbounds for i in 1:n, j in 1:m
        K[i, j] = @views κ(X[:, i], Y[:, j])  #compute kernel for each pair (X[:, i], Y[:, j])
    end
    return K
end

"""kernelmatrix overload for PathKernel and matrix X"""
function KernelFunctions.kernelmatrix(κ::PathKernel, X::AbstractMatrix{T}) where T <: Real
    n = size(X, 2)
    K = Matrix{T}(undef, n, n)
    @inbounds for i in 1:n
        K[i, i] = @views κ(X[:, i], X[:, i])
        @inbounds for  j in (i+1):n
            K[i, j] = @views κ(X[:, i], X[:, j])  #compute kernel for each pair (X[:, i], Y[:, j])
            K[j, i] = K[i, j]
        end
    end
    return K
end

"""kernelmatrix overload for PathKernel on RowVecs X and Y"""
function KernelFunctions.kernelmatrix(κ::PathKernel, X::RowVecs, Y::RowVecs)
    #convert RowVecs X and Y to a matrices
    X_matrix = hcat(X...)
    Y_matrix = hcat(Y...)
    return KernelFunctions.kernelmatrix(κ, X_matrix, Y_matrix)
end

"""kernelmatrix overload for PathKernel on RowVecs X"""
function KernelFunctions.kernelmatrix(κ::PathKernel, X::RowVecs)
    #convert RowVecs to a matrix
    X_matrix = hcat(X...)
    #compute the kernel matrix
    return KernelFunctions.kernelmatrix(κ, X_matrix)
end

# Auxilary functions
#------------------------------
# Specialized kernel evaluations
"""
    kappa(kernel, d)    

Compute square exponential kernel function value at distance `d`.
"""
@inline function kappa(kernel::KernelFunctions.SqExponentialKernel, d::T) where T<:Real
    return exp(-(d^2))
end

"""
    kappa(kernel, d)

Compute exponential kernel function value at distance `d`.
"""
@inline function kappa(kernel::KernelFunctions.ExponentialKernel, d::T) where T<:Real
    return exp(-d )
end

"""
    kappa(kernel, d)

Compute Matern32 kernel function value at distance `d`.
"""
@inline function kappa(kernel::KernelFunctions.Matern32Kernel, d::T) where T<:Real
    sqrt3 = sqrt(3.)
    return (1. + sqrt3*d) * exp(-sqrt3*d)
end

#fallback if needed (generic)
"""
    kappa(kernel, d)

Compute generic kernel function value at distance `d`.
"""
@inline function kappa(kernel::KernelFunctions.Kernel, d::T) where T<:Real
    
    return KernelFunctions.kappa(kernel, d)
end

"""
    compute_kernel_integral(kernel, d, w)

Compute integral kernel parametrized by distance `d` and weights `w`.
"""
@inline function compute_kernel_integral_d(kernel::KernelFunctions.Kernel, 
                                           d::AbstractVector{T}, 
                                           w::AbstractVector{T}) where T <: Real

    acc = zero(T)
    @inbounds @simd for i in eachindex(d)
        acc += w[i] * kappa(kernel, d[i])
    end
    return acc
end

"""
    compute_kernel_integral(kernel, len, t, w)

Compute integral kernel parametrized by lenght `len`, integration points `t` and weights `w`.
"""
@inline function compute_kernel_integral_t(kernel::KernelFunctions.Kernel, 
                                           len::T,
                                           t::AbstractVector{T}, 
                                           w::AbstractVector{T}) where T <: Real

    acc = zero(T)
    @inbounds @simd for i in eachindex(w)
        acc += w[i] * kappa(kernel, len * t[i])
    end
    return acc
end

"""
    compute_kernel_integral(kernel, fun_dist, t, s, w)

Compute integral `kernel` function parametrized by distance function `fun_dist(t,s)` 
    integration points `t` and `s`, and weights `w`.
"""
@inline function compute_kernel_integral_ts(kernel::KernelFunctions.Kernel, 
                                            fun_dist::F,
                                            t::AbstractVector{T}, 
                                            s::AbstractVector{T}, 
                                            w::AbstractVector{T}) where {T <: Real, F <: Function}

    acc = zero(T)
    @inbounds @simd for i in eachindex(w)
        acc += w[i] * kappa(kernel, fun_dist(t[i], s[i]))
    end
    return acc
end

"""
    dotsub(x, y, u, v)

Compute the inner product ``\\langle x - y, u - v \\rangle``.
"""
@inline function dotsub(x::AbstractArray{T}, y::AbstractArray{T}, 
                        u::AbstractArray{T}, v::AbstractArray{T}) where T <: Real
    
    dot(x,u) - dot(y,u) - dot(x,v) + dot(y,v)
end

"""
    psqrt(x)

Compute `sqrt(relu(x))`.
"""
@inline psqrt(x::T) where T <: Real = sqrt(x + 1e-12) 

"""
    euclidean_dist(x1, x2)  

Compute the Euclidean length of vector with origin x1 and end x2.
"""
@inline function euclidean_len(x1::AbstractVector{T}, x2::AbstractVector{T}) where T<:Real
    acc = zero(T)
    @inbounds @simd for i in eachindex(x1, x2)
        Δ = x2[i] - x1[i]
        acc += Δ*Δ
    end
    return sqrt(acc)
end

"""
    getdist(x1, x2, y1, y2)

Return a function of `t` and `s` to compute the quantity below.
```julia
norm(x1 + t * (x2 - x1) - (y1 + s * (y2 - y1)))
```
"""
@inline function getdist(x1::AbstractArray{T}, x2::AbstractArray{T}, 
                         y1::AbstractArray{T}, y2::AbstractArray{T}) where T <: Real

    t2 = dotsub(x2, x1, x2, x1)
    s2 = dotsub(y2, y1, y2, y1)
    t1 = 2.0 * dotsub(x1, y1, x2, x1)
    s1 = 2.0 * dotsub(y1, x1, y2, y1)
    ts = 2.0 * dotsub(x2, x1, y2, y1)
    c = dotsub(x1, y1, x1, y1)

    let t2 = t2, s2 = s2, t1 = t1, s1 = s1, ts = ts, c = c
        """
            dist(t, s)

        Compute a distance quantity given `t` and `s`.
        """
        function dist(t::Real, s::Real)
            return psqrt(
                t^2 * t2 + t * t1 + s^2 * s2 + s * s1 - t * s * ts + c
            )
        end
    end
end

"""
    dist(x1, x2, y1, y2, t, s)

Compute the distance between two paths given `t` and `s`.
"""
function dist(x1::AbstractVector{T}, x2::AbstractVector{T}, 
              y1::AbstractVector{T}, y2::AbstractVector{T},
              t::AbstractVector{T}, s::AbstractVector{T}) where T <: Real 

    x = x1' .+ t .* (x2 - x1)' 
    y = y1' .+ s .* (y2 - y1)'

    return sqrt.(sum((x-y).^2; dims=2))
end

"""
    trapezoid1d(f, a, b, n)

Integrate the 1-d function `f` over [`a`, `b`] with `n` points.
"""
function trapezoid1d(f, a::T, b::T, n::Int)::T where T <:Real
    x = range(a, b; length=n)[(begin + 1):(end - 1)]
    return ((f(a) + f(b)) / 2.0 + sum(f, x; init=0.0)) / (n - 1)
end

"""
    trapezoid2d(f, a1, b1, a2, b2, n)

Integrate 2-d function `f` over [`a1`, `b1`] x [`a2`, `b2`] by ``n^2`` points.
"""
function trapezoid2d(f, a1::T, b1::T, a2::T, b2::T, n::Int)::T where T <: Real
    #x1 = range(a1, b1; length=n)[(begin + 1):(end - 1)]
    #x2 = range(a2, b2; length=n)[(begin + 1):(end - 1)]
    x1 = [a1 + i*(b1-a1)/(n-1) for i in 1:n-2]
    x2 = [a1 + i*(b1-a1)/(n-1) for i in 1:n-2]
    # corners
    value = (f(a1, a2) + f(a1, b2) + f(b1, a2) + f(b1, b2)) / 4.0
    # edges
    for x in x1
        value += (f(x, a2) + f(x, b2)) / 2.0
    end
    for y in x2
        value += (f(a1, y) + f(b1, y)) / 2.0
    end
    # interior
    for x in x1, y in x2
        value += f(x, y)
    end
    return value / (n - 1)^2
end

"""
    gaussquad1d(n_int)

Gauss quadrature 1D integration in [0,1]
"""
function gaussquad1d(n_int::Int)

    #compute quadrature points
    t, w = gausslobatto(n_int)

    #adjust for 0.0 - 1.0 domain
    t  = 0.5 .+ 0.5*t
    w = 0.5*w
    w = w / sum(w)

    return t, w
end

"""
    gaussquad2d(n_int)

Gauss quadrature 2D integration in [[0,1] x [0,1]]
"""
function gaussquad2d(n_int::Int)

    #compute 1D quadrature points
    t1D, w1D = gaussquad1d(n_int)

    #compute 2D quadrature points
    w2D  = prod.([Iterators.product(w1D,w1D)...])
    w2D  = w2D / sum(w2D)
    ts2D = [Iterators.product(t1D,t1D)...]
    t2D  = [ts[1] for ts in ts2D]
    s2D  = [ts[2] for ts in ts2D]

    return t2D, s2D, w2D
end

"""
    gaussdiff2d(diffst, w; digits)

Compute the sum of weights for gauss quadrature location `diffst` difference with `digits` significant digits.
"""
function gaussdiff2d(diffst::AbstractVector{T}, w::AbstractVector{T}; digits::Int=9) where T <: Real

    #dictionary to store rounded values and their sums
    sums_dict = Dict{T, T}()

    #iterate over the values
    for (dsti, wi) in zip(diffst, w)
        #round xi to digits decimal places
        key = round(dsti, sigdigits=digits)
        #add new weight
        sums_dict[key] = get(sums_dict, key, 0.0) + wi
    end

    #sort keys
    idx = sortperm(collect(keys(sums_dict)))

    #convert dictionary to two arrays
    return collect(keys(sums_dict))[idx], collect(values(sums_dict))[idx]
end
