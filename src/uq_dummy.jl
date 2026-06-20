# No uncertainties
# Dummy model with no computation. This is the default integrated UQ model.

struct DummyUQModel <: IntegratedUQModel end
struct DummyUQResult <: AbstractUQResult end


function update_UQModel!(UQM::DummyUQModel, args...; kwargs...)
    UQM
end


function uq!(UQM::DummyUQModel, args...; kwargs...)
    DummyUQResult()
end


function sample_uqmodel(UQM::DummyUQModel, args...; kwargs...)
    nothing
end


C_rank(IUM::DummyUQModel) = 0 # for getting PredictionBuffer sizes


recover_covariance(UQR::DummyUQResult, args...) = nothing
