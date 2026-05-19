abstract type AbstractUncertaintyModel end
abstract type AbstractScalarUncertaintyModel end


# No uncertainties
struct DummyUncertaintyModel <: AbstractUncertaintyModel
end


# Dummy model with no computation, for testing only.
function predict_variance(UQM::DummyUncertaintyModel,
                          M_cross::Matrix;
                          diagonals::Bool = true
                          )

    (nte, ntr) = size(M_cross)
    return diagonals ? zeros(nte) : zeros(nte,nte)

end


# Standard conditional variance model
struct ScalarFullRankUncertaintyModel{T} <: AbstractUncertaintyModel
    LI::UpperTriangular{T} # Cholesky factor of inverse kernel matrix
end


function ScalarFullRankUncertaintyModel(M::GPModel{T}; H::Type = Float32
                                             ) where T <: Real
    ntr = length(M.ζ)
    buf = zeros(H, ntr, ntr)
    buf2 = zeros(H, ntr, ntr)
    kernel_matrix_fast!(M.kernel, H.(M.θ), H.(M.Z), buf; precision = false)
    LI = UpperTriangular(diagm(zeros(H, ntr)))
    unit = Diagonal(ones(H, ntr))
    LI .= unit \ UpperTriangular(buf)
    UQM = ScalarFullRankUncertaintyModel(LI)
    #ldiv!(buf2, buf, UQM.LI)
end


function predict_variance(UQM::ScalarFullRankUncertaintyModel{T},
                          M_cross::Matrix{T};
                          # set to false to return full uncertainties
                          diagonals::Bool = true
                          ) where T <: Real

    # Note: We don't compute covariances between test points by
    # default. The reason is, that it's perhaps not interesting, and
    # makes computation much more expensive.
    (nte, ntr) = size(M_cross)
    A = UQM.LI * M_cross
    ret = diagonals ? sum(A.^2, dims = 1)[:] : A' * A
    ret
end


# Low-rank variant of the conditional variance model. Covariance is
# estimated as P'P + D'D, where D is diagonal and P is low-rank. The
# rationale for this is, that we don't want to store the full-rank
# precision matrices, but can still match the marginal uncertainties
# exactly.
struct ScalarLowRankUncertaintyModel{T}
    P::Matrix{T} # matrix of r leading eigenvectors scaled by singular vectors
    D::Diagonal{T} # square roots of residual variances to make model
end


function ScalarLowRankUncertaintyModel(M::GPModel{T}, r::Int; H::Type = Float32
                                 ) where T <: Real
    ntr = length(M.ζ)
    buf = zeros(H, ntr, ntr)
    kernel_matrix_fast!(M.kernel, H.(M.θ), H.(M.Z), buf; precision = false)
    F = eigen(buf)
    P = (Diagonal(1.0 ./ sqrt.(F.values[1:r])) * F.vectors[:,1:r]')[:,:]
    d = sum(P.^2, dims = 1)[:]

    dfull = sum((Diagonal(1.0 ./ sqrt.(F.values)) * F.vectors').^2, dims = 1)
    D = Diagonal(sqrt.(dfull - d))

    UQM = ScalarLowRankUncertaintyModel(P, D)
end


function predict_variance(UQM::ScalarLowRankUncertaintyModel{T},
                          M_cross::Matrix{T};
                          diagonals::Bool = true) where T <: Real

    (nte, ntr) = size(M_cross)
    A = UQM.P * M_cross
    ret1 = diagonals ? sum(A.^2, dims = 1)[:] : A' * A
    A = UQM.D * M_cross
    ret2 = diagonals ? sum(A.^2, dims = 1)[:] : A' * A
    ret1 + ret2
end


struct MultivariateUncertaintyModel <: AbstractUncertaintyModel
    UQMs::Vector{AbstractScalarUncertaintyModel}
end


function predict_covariance(MVUM::MultivariateUncertaintyModel, X_te::Matrix{T}) where {T<:Real}
    (nte,nx) = size(X_te)
    nz = length(MVUM)

    # All variances in the reduced form are here, row by row.
    var_te = zeros(T, (nte, nz))
    for i in 1:nz
    end

end
