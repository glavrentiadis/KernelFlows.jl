struct StandardUQModel{T} <: IntegratedUQModel
    LI::UpperTriangular{T} # Cholesky factor of inverse kernel matrix
end


struct StandardUQResult{T} <: AbstractUQResult
    stds::Matrix{T} # Posterior standard deviations for test data
    Yproj::Projection{T}
    σY::Vector{T}
    # G::GPGeometry{T}
end


function uqresult(::StandardUQModel{T}, stds::Matrix{T}, Yproj::Projection{T}, σY::Vector{T}) where T <: Real
    StandardUQResult(stds, Yproj, σY)
end


function recover_covariance(z_std::Vector{T}, Yproj::Projection{T}, σY::Vector{T}) where T <: Real
    nzy = length(z_std)
    zvar = @views z_std .* Yproj.values
    zvar .^= 2
    # for undoing initial scaling
    Yvecs = @views Diagonal(σY) * Yproj.vectors[:,1:nzy]
    Yvecs * Diagonal(zvar) * Yvecs'
end


function recover_covariance(UQR::StandardUQResult{T}, i::Int) where T <: Real
    recover_covariance(UQR.stds[i,:], UQR.Yproj, UQR.σY)
end


function update_UQModel!(UQM::StandardUQModel, buf::Matrix{T}, args...; kwargs...) where T <: Real
    UQM.LI .= UpperTriangular(buf) \ I
end


"""Get posterior standard deviations for test data in Z, given in
reduced coordinates. Use the multivariate version of predict() to get
the transformations right.

Inputs:
        M_cross: cross covariance matrix between inputs and outputs
        h: marginal variances k(z*, z*) for test inputs z*.
"""
function uq!(UQM::StandardUQModel, pb::StandardPredictionBuffer{T}, h::Vector{T}, out::AbstractVector{T}) where T <: Real
    H = eltype(UQM.LI)
    # When doing potrf!('U', C) and then LI = UpperTriangular(C) \ I,
    # the reconstruction of C inverse is Cinv ≈ LI * LI'. Therefore
    # there is no transpose in the last term below.
    @time mul!(pb.M_uqprod, pb.M_cross, T.(UQM.LI))
    pb.M_uqprod .^= 2
    out .= @time -sum(pb.M_uqprod, dims = 2)
    out .+= h
end


function sample_uqmodel(UQM::StandardUQModel, X::Matrix{T}) where T <: Real
    error("Sampling not implemented")
end

C_rank(IUM::StandardUQModel) = size(IUM.LI)[1] # for PredictionBuffer allocations
