# Prediction with CPUs using standard LinearAlgebra backend

# CPU
using LinearAlgebra
GPUMatrix = Matrix
GPUVector = Vector
GPUzeros = zeros
GPUgemm! = BLAS.gemm!
GPUgemv! = BLAS.gemv!
GPUsqrt = sqrt
 

include("predict_generic.jl")
