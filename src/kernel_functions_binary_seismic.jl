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
# Author: Grigorios Lavrentiadis, glavrent@caltech.edu
#

using KernelFunctions: KernelFunctions as Kernels, ExponentialKernel
using Statistics: mean

"""
    PathKernel

Derived kernel based on integration.

# Arguments
- `n`: number of points to integrate
- `kernel`: base kernel
"""
struct PathKernel{Kernel<:Kernels.Kernel} <: Kernels.Kernel
    n::Int
    kernel::Kernel
end


"""Path kernel"""
function (k::PathKernel)(x₁::AbstractVector{T}, x₂::AbstractVector{T},
                          θ::AbstractVector{T}) where T <: Real
    #hyperparameters
    ω = θ[1]
    λ = θ[2]

    #coordinates
    x₁ₗ = x₁[1:2]
    x₂ₗ = x₂[1:2]
    x₁ₛ = x₁[3:4]
    x₂ₛ = x₂[3:4]

    #paths
    dist = getdist(x₁, x₂, 1/λ)
    f(t, s) = Kernels.kappa(k.kernel, dist(t, s))

    return ω^2 * trapezoid2d(f, 0.0, 1.0, 0.0, 1.0, k.n)
end

#
#------------------------------
"""Binary between event aleatory variability kernel function"""
function binary_event_aleat(eqid₁::Union{AbstractVector{T}, T}, eqid₂::Union{AbstractVector{T}, T},
                            θ::Union{AbstractVector{T}, T}) where T <: Real
        
    #hyperparameters
    τ₀ = θ[1] #source parametes
    
    #evaluate kernel function components
    Kₐ = binary_group(eqid₁, eqid₂, τ₀; tol=1e-6) #aleatory kernel

    return Kₐ
end

#Individual Non-ergodic Kernels
#------------------------------
"""Binary source kernel function"""
function binary_source(x₁::AbstractVector{T}, x₂::AbstractVector{T},
                    θ::AbstractVector{T}) where T <: Real
    
    #coordinates
    x₁ₗ = x₁ #1st event
    x₂ₗ = x₂ #2nd event

    #hyperparameters
    θₗ = θ[1:2] #source parametes
    
    #evaluate kernel function components
    Kₗ = binary_exp(x₁ₗ,x₂ₗ,θₗ) #source kernel

    return Kₗ
end

"""Binary path site kernel function"""
function binary_site(x₁::AbstractVector{T}, x₂::AbstractVector{T},
                    θ::AbstractVector{T}) where T <: Real
    
    #coordinates
    x₁ₛ = x₁ #1st site
    x₂ₛ = x₂ #2nd site

    #hyperparameters
    θₛ = θ[1:2] #site parameters
    
    #evaluate kernel function components
    Kₛ = binary_exp(x₁ₛ,x₂ₛ,θₛ) #site kernel 

    return Kₛ
end

"""Binary path kernel function"""
function binary_path(x₁::AbstractVector{T}, x₂::AbstractVector{T},
                     θ::AbstractVector{T}; 
                     n_integ_pt=10, flag_normalize=true) where T <: Real
    
    #coordinates
    x₁ₗ = x₁[1:2] #1st event
    x₂ₗ = x₂[1:2] #2nd event
    x₁ₛ = x₁[3:4] #1st site
    x₂ₛ = x₂[3:4] #2nd site

    #hyperparameters
    θₚ = θ[1:2] #path parameters

    #set up path kernel
    # - number of integration points
    # - underling covariance function
    κₚ = PathKernel(n_integ_pt,ExponentialKernel())
    
    #define kernel function components
    Kₚ₁ = κₚ(x₁,x₂,θₚ) #path kernel (x1,x2)
    Kₚ₂ = κₚ(x₂,x₁,θₚ) #path kernel (x1,x2)
    Kₚ = mean([Kₚ₁,Kₚ₂]) #path kernel (average)
    
    if flag_normalize
        Kₚ /= sqrt(κₚ(x₁,x₁,[1.,θₚ[2]]))
        Kₚ /= sqrt(κₚ(x₂,x₂,[1.,θₚ[2]]))
    end

    return Kₚ
end

# Composite Non-ergodic Kernels
#------------------------------
"""Binary source & site kernel function"""
function binary_sourcesite(x₁::AbstractVector{T}, x₂::AbstractVector{T},
                           θ::AbstractVector{T}) where T <: Real
    
    #coordinates
    x₁ₗ = x₁[1:2] #1st event
    x₂ₗ = x₂[1:2] #2nd event
    x₁ₛ = x₁[3:4] #1st site
    x₂ₛ = x₂[3:4] #2nd site

    println(θ)
    #hyperparameters
    θₗ = θ[1:2] #source parametes
    θₛ = θ[3:4] #site parameters
    
    #define kernel function components
    Kₗ = binary_exp(x₁ₗ,x₂ₗ,θₗ) #source kernel
    Kₛ = binary_exp(x₁ₛ,x₂ₛ,θₛ) #site kernel 

    #total kernel
    Kₜ = Kₗ + Kₛ

    return Kₜ
end

"""Binary path & site kernel function"""
function binary_pathsite(x₁::AbstractVector{T}, x₂::AbstractVector{T},
                         θ::AbstractVector{T}) where T <: Real
    
    #coordinates
    x₁ₗ = x₁[1:2] #1st event
    x₂ₗ = x₂[1:2] #2nd event
    x₁ₛ = x₁[3:4] #1st site
    x₂ₛ = x₂[3:4] #2nd site

    #hyperparameters
    θₛ = θ[1:2] #site parameters
    θₚ = θ[3:4] #path parameters
    
    #define kernel function components
    Kₛ = binary_site(x₁ₛ,x₂ₛ,θₛ)         #site kernel 
    Kₚ = binary_path(x₁[1:4],x₂[1:4],θₚ) #path kernel (x1,x2)
    
    #total kernel
    Kₜ = Kₛ + Kₚ
    return Kₜ
end

"""Binary full kernel function"""
function binary_full(x₁::AbstractVector{T}, x₂::AbstractVector{T},
                    θ::AbstractVector{T}) where T <: Real
    
    #coordinates
    x₁ₗ = x₁[1:2] #1st event
    x₂ₗ = x₂[1:2] #2nd event
    x₁ₛ = x₁[3:4] #1st site
    x₂ₛ = x₂[3:4] #2nd site

    #hyperparameters
    θₗ = θ[1:2] #source parametes
    θₛ = θ[3:4] #site parameters
    θₚ = θ[5:6] #path parameters

    #define kernel function components
    Kₗ = binary_source(x₁ₗ,x₂ₗ,θₗ)       #source kernel
    Kₛ = binary_site(x₁ₛ,x₂ₛ,θₛ)         #site kernel 
    Kₚ = binary_path(x₁[1:4],x₂[1:4],θₚ) #path kernel (x1,x2)

    #total kernel
    Kₜ = Kₗ + Kₛ + Kₚ
    # println("end")

    return Kₜ
end

# Non-ergodic Kernels with Aleatory Variability
#------------------------------
"""Binary non-ergodic site and event aleatory kernel function"""
function binary_site_aleat(x₁::AbstractVector{T}, x₂::AbstractVector{T},
                                 θ::AbstractVector{T}) where T <: Real
    
    #coordinates
    x₁ₛ    = x₁[1:2] #1st site location
    x₂ₛ    = x₂[1:2] #2nd site location
    eqid₁  = x₁[3]   #1st event id
    eqid₂  = x₂[3]   #2nd event id

    #hyperparameters
    θₙ = θ[1:2] #non-ergodic parameters
    θₐ = θ[3]   #aleatory parameters

    #define kernel function components
    Kₙ = binary_site(x₁ₛ, x₂ₛ, θₙ)             #non-ergodic (site)
    Kₐ = binary_event_aleat(eqid₁, eqid₂, θₐ)  #aleatory
    
    #total kernel
    Kₜ = Kₙ + Kₐ
    return Kₜ
end

"""Binary non-ergodic path and event aleatory kernel function"""
function binary_path_aleat(x₁::AbstractVector{T}, x₂::AbstractVector{T},
                           θ::AbstractVector{T}) where T <: Real
    
    #coordinates
    x₁ₗₛ   = x₁[1:4] #1st event & site location
    x₂ₗₛ   = x₂[1:4] #2nd event & site location
    eqid₁  = x₁[5]    #1st event id
    eqid₂  = x₂[5]    #2nd event id

    #hyperparameters
    θₙ = θ[1:4] #non-ergodic parameters
    θₐ = θ[5]   #aleatory parameters

    #define kernel function components
    Kₙ = binary_path(x₁ₗₛ, x₂ₗₛ, θₙ)           #non-ergodic (path)
    Kₐ = binary_event_aleat(eqid₁, eqid₂, θₐ)  #aleatory
    
    #total kernel
    Kₜ = Kₙ + Kₐ
    return Kₜ
end

"""Binary non-ergodic source & site and event aleatory kernel function"""
function binary_sourcesite_aleat(x₁::AbstractVector{T}, x₂::AbstractVector{T},
                                 θ::AbstractVector{T}) where T <: Real
    
    #coordinates
    x₁ₗₛ   = x₁[1:4] #1st event & site location
    x₂ₗₛ   = x₂[1:4] #2nd event & site location
    eqid₁  = x₁[5]    #1st event id
    eqid₂  = x₂[5]    #2nd event id

    #hyperparameters
    θₙ = θ[1:4] #non-ergodic parameters
    θₐ = θ[5]   #aleatory parameters

    #define kernel function components
    Kₙ = binary_sourcesite(x₁ₗₛ, x₂ₗₛ, θₙ)     #non-ergodic (source and site)
    Kₐ = binary_event_aleat(eqid₁, eqid₂, θₐ)  #aleatory
    
    #total kernel
    Kₜ = Kₙ + Kₐ
    return Kₜ
end

"""Binary non-ergodic path & site and event aleatory kernel function"""
function binary_pathsite_aleat(x₁::AbstractVector{T}, x₂::AbstractVector{T},
                               θ::AbstractVector{T}) where T <: Real
    
    #coordinates
    x₁ₗₛ   = x₁[1:4] #1st event & site location
    x₂ₗₛ   = x₂[1:4] #2nd event & site location
    eqid₁  = x₁[5]    #1st event id
    eqid₂  = x₂[5]    #2nd event id

    #hyperparameters
    θₙ = θ[1:4] #non-ergodic parameters
    θₐ = θ[5]   #aleatory parameters

    #define kernel function components
    Kₙ = binary_pathsite(x₁ₗₛ, x₂ₗₛ, θₙ)       #non-ergodic (path and site)
    Kₐ = binary_event_aleat(eqid₁, eqid₂, θₐ)  #aleatory
    
    #total kernel
    Kₜ = Kₙ + Kₐ
    return Kₜ
end

"""Binary full non-ergodic and event aleatory kernel function"""
function binary_full_aleat(x₁::AbstractVector{T}, x₂::AbstractVector{T},
                               θ::AbstractVector{T}) where T <: Real

    #coordinates
    x₁ₗₛ   = x₁[1:4] #1st event & site location
    x₂ₗₛ   = x₂[1:4] #2nd event & site location
    eqid₁  = x₁[5]    #1st event id
    eqid₂  = x₂[5]    #2nd event id

    #hyperparameters
    θₙ = θ[1:6] #non-ergodic parameters
    θₐ = θ[7]   #aleatory parameters

    #define kernel function components
    Kₙ = binary_full(x₁ₗₛ, x₂ₗₛ, θₙ)           #non-ergodic (source, path and site)
    Kₐ = binary_event_aleat(eqid₁, eqid₂, θₐ)  #aleatory
    
    #total kernel
    Kₜ = Kₙ + Kₐ
    return Kₜ
end

#Auxilary functions
#------------------------------
"""
    dotsub(x, y, u, v)

Compute the inner product ``\\langle x - y, u - v \\rangle``.
"""
function dotsub(x, y, u, v)
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
psqrt(x) = sqrt((x > 0.0) ? x : 0.0)


"""
    getdist(x, y, ell)

Return a function of `t` and `s` to compute the quantity below.
```julia
norm(x1 + t * (x2 - x1) - (y1 + s * (y2 - y1)))
```
"""
function getdist(x, y, ell)
    # assume x = [x1; x2] and y = [y1; y2]
    d = div(length(x), 2)
    x1 = @view x[begin:d]
    x2 = @view x[(d + 1):end]
    y1 = @view y[begin:d]
    y2 = @view y[(d + 1):end]
    t2 = dotsub(x2, x1, x2, x1)
    s2 = dotsub(y2, y1, y2, y1)
    t1 = 2.0 * dotsub(x1, y1, x2, x1)
    s1 = 2.0 * dotsub(y1, x1, y2, y1)
    ts = 2.0 * dotsub(x2, x1, y2, y1)
    c = dotsub(x1, y1, x1, y1)

    let ell = ell, t2 = t2, s2 = s2, t1 = t1, s1 = s1, ts = ts, c = c
        """
            dist(t, s)

        Compute a distance quantity given `t` and `s`.
        """
        function dist(t, s)
            return psqrt(
                t^2 * t2 + t * t1 + s^2 * s2 + s * s1 - t * s * ts + c
            ) / ell
        end
    end
end

"""
    trapezoid2d(f, a1, b1, a2, b2, n)

Integrate 2-d function `f` over [`a1`, `b1`] x [`a2`, `b2`] by ``n^2`` points.
"""
function trapezoid2d(f, a1, b1, a2, b2, n)
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
