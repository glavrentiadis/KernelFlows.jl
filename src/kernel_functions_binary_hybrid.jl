
function hybrid_kernel!(K::AbstractMatrix{T}, 
                        X₁::AbstractVector{T}, X₂::AbstractVector{T},
                        θ::T) where T <: Real

    θ² = θ^2

    for j₁ in findall(>(1e-9), X₁)
        for j₂ in eachindex(X₂)
            K[j₁,j₂] *=  X₂[j₂] >(1e-9) ? θ² : θ    
            println("X₁: ", X₁[j₁], " X₂: ", X₂[j₂])
        end
    end

end

"""
    Binary source, path & site Matern kernel function for hybrid dataset
"""
function sourcepathsite_matern_hybrid_binary(X₁::AbstractMatrix{T}, X₂::AbstractMatrix{T},
                                             idₕ₁::AbstractVector{T}, idₕ₂::AbstractVector{T}, 
                                             θ::AbstractVector{T}, θₕ::AbstractArray{T}) where T <: Real
    
    #hyperparameters
    θₗ = @views θ[1:2] #source parametes
    θₚ = @views θ[3:4] #path parametes
    θₛ = @views θ[5:6] #site parameters

    #coordinate dimension size
    d = div(size(X₁)[2], 2) 
    println("dims X1: ", size(X₁))
    #evaluate total kernel
    Kₗ  = @views source_matern_binary(X₁[:,1:d],     X₂[:,1:d],       θₗ)
    Kₚ += @views path_matern_binary(X₁,              X₂,              θₚ)
    Kₛ += @views site_matern_binary(X₁[:,(d+1):end], X₂[:,(d+1):end], θₛ)
    hybrid_kernel!(Kₗ, idₕ₁, idₕ₂, θₕ[1])
    hybrid_kernel!(Kₚ, idₕ₁, idₕ₂, θₕ[2])
    hybrid_kernel!(Kₛ, idₕ₁, idₕ₂, θₕ[3])

    return Kₗ + Kₚ + Kₛ
end



"""
    Binary source, path and site Matern kernel function with between event aleatory variability
    for hybrid dataset
"""
function sourcepathsite_matern_aleat_hybrid_binary(X₁::AbstractMatrix{T}, X₂::AbstractMatrix{T},
                                                   θ::AbstractVector{T}) where T <: Real
    
    #hyperparameters
    println("length θ: ", length(θ))
    θₙ = @view θ[1:6] #non-ergodic parametes
    θₕ = @view θ[7:9] #hybrid parameters
    θₐ = @view θ[10]  #aleatory parameters
    
    #evaluate total kernel
    Kₜ  = @views aleat_bevent_binary(X₁[:,1], X₂[:,1], θₐ)
    Kₜ += @views sourcepathsite_matern_hybrid_binary(X₁[:,3:end], X₂[:,3:end], X₁[:,2], X₂[:,2], θₙ, θₕ)
    
    return Kₜ
end