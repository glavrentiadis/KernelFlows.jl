# Prediction with Intel GPUs on oneAPI

# Intel
using oneAPI
GPUMatrix = oneMatrix
GPUVector = oneVector
GPUzeros = oneAPI.zeros
GPUgemm! = oneMKL.gemm!
GPUgemv! = oneMKL.gemv!
GPUsqrt = sqrt

include("predict_generic.jl")
