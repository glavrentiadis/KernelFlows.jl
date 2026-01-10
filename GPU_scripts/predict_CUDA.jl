# Prediction with NVIDIA GPUs using CUDA

# NVIDIA
using CUDA
GPUMatrix = CuMatrix
GPUVector = CuVector
GPUzeros = CUDA.zeros
GPUgemm! = CUDA.CUBLAS.gemm!
GPUgemv! = CUDA.CUBLAS.gemv!
GPUsqrt = CUDA.sqrt

include("predict_generic.jl")

## Uncomment for some basic timing
# Y_gpu = GPUpredict(MVM, X_te[1:1000,:]) # for compilation
# CUDA.@time Y_gpu_all = GPUpredict(MVM, X_te) #[1:100000,:]) # for timing
# CUDA.@profile refl = GPUpredict(MVM, X_te[1:500000,:])
# GC.gc()
# CUDA.reclaim()
