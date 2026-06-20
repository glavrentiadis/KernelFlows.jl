abstract type AbstractUQResult end
abstract type AbstractUQModel end


struct MVUQResult
    Ms::Vector{AbstractUQResult}
    G::GPGeometry
end


include("uq_standard.jl")
include("uq_dummy.jl")
include("uq_nongaussian.jl")
# include("uq_lowrank.jl")]
