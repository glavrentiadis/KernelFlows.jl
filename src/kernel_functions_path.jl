# Path Kernel Function
#------------------------------
"""
    PathKernel

Derived kernel based on integration.

# Arguments
- `n`: number of points to integrate
- `kernel`: base kernel
- `normalized`: flag for normalized variant w.r.t. path length
"""
# struct PathKernel{Kernel<:KernelFunctions.Kernel} <: KernelFunctions.Kernel
#     n::Int          
#     kernel::Kernel
#     normalized::Bool  
# end
struct PathKernel <: KernelFunctions.Kernel
    n::Int          
    kernel::KernelFunctions.Kernel
    #kernel
    normalized::Bool  
end

"""Path kernel"""
function (k::PathKernel)(x::AbstractVector{T}, y::AbstractVector{T}) where T <: Real
    
    #parse coordinates, assuming x = [x1; x2] and y = [y1; y2]
    d = div(length(x), 2) #dimension
    x1 = @view x[begin:d]
    x2 = @view x[(d + 1):end]
    y1 = @view y[begin:d]
    y2 = @view y[(d + 1):end]
    #path lenghts
    x_len = psqrt( dotsub(x2, x1, x2, x1) )
    y_len = psqrt( dotsub(y2, y1, y2, y1) )

    #distance between points on paths
    dist = getdist(x1, x2, y1, y2)

    #underling kernel function, distance metric parametrization 
    κ_b(d) = KernelFunctions.kappa(k.kernel, d)
    #κ_b = k.kernel
    #underling kernel function, along path parametrization 
    κ′_b(t::Real, s::Real) = κ_b( dist(t, s) ) 
    
    #evaluate path kernel
    K = trapezoid2d(κ′_b, 0.0, 1.0, 0.0, 1.0, k.n) * x_len * y_len
    #normalization
    if k.normalized
        K /= psqrt( trapezoid2d((t, s) -> κ_b(x_len * abs(t - s)), 0.0, 1.0, 0.0, 1.0, k.n) ) * x_len
        K /= psqrt( trapezoid2d((t, s) -> κ_b(y_len * abs(t - s)), 0.0, 1.0, 0.0, 1.0, k.n) ) * y_len
        K = min(K, 1.)
    end

    return K
end


# Auxilary functions
#------------------------------
"""
    dotsub(x, y, u, v)

Compute the inner product ``\\langle x - y, u - v \\rangle``.
"""
function dotsub(x::AbstractArray{T}, y::AbstractArray{T}, 
                u::AbstractArray{T}, v::AbstractArray{T}) where T <: Real
    value = 0.0
    for i in eachindex(x)
        value += (x[i] - y[i]) * (u[i] - v[i])
    end
    return value
end

"""
    psqrt(x)

Compute `sqrt(relu(x))`.
"""
psqrt(x::Real) = sqrt((x > 0.0) ? x : 0.0)

"""
    getdist(x1, x2, y1, y2)

Return a function of `t` and `s` to compute the quantity below.
```julia
norm(x1 + t * (x2 - x1) - (y1 + s * (y2 - y1)))
```
"""
function getdist(x1::AbstractArray{T}, x2::AbstractArray{T}, 
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