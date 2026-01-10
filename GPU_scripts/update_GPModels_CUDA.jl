
export CUDA_update_all_K


function pw_and_linear_d!(A::AbstractMatrix{T}, out1::AbstractMatrix{T}, out2::AbstractMatrix{T}, vbuf::AbstractVector{T}, vbuf2::AbstractVector{T}, γ::T) where T <: Real
    CUBLAS.syrk!('U', 'N', T(-2), A, zero(T), out1)
    out2 .= out1 # copy to out2
    out2 .*= -γ/T(2) # multiply out2 by -γ/2
    A .*= A # square of A
    a = vbuf
    a .= zero(T) # 10*eps(T)
    vbuf2 .= T(1)

    # This does a sum over columns to a and does not allocate as vbuf2
    # comes from outside

    CUBLAS.gemv!('N', one(T), A, vbuf2, one(T), a)
    out1 .+= a
    out1 .+= a'
    out1 .= sqrt.(abs.(out1)) # Euclidean distance
end


function update_all_K(Ms::Vector{GPModel{H}}) where H <: Real
    T = Float64

    n, nλ = size(Ms[1].Z)

    buf1_GPU = CUDA.zeros(T, n, n)
    buf2_GPU = CUDA.zeros(T, n, n)
    buf3_GPU = CUDA.zeros(T, n, n)
    v1_n_GPU = CUDA.zeros(T, n)
    v2_nλ_GPU = CUDA.zeros(T, nλ)
    Z_GPU = CUDA.zeros(T, n, nλ)
    ntot = length(Ms)

    for (i,M) in enumerate(Ms)
        print("\rUpdating GPModel $i/$ntot...  ")
        θ = T.(M.θ)
        copy!(Z_GPU, T.(M.Z))
        pw_and_linear_d!(Z_GPU, buf1_GPU, buf2_GPU, v1_n_GPU, v2_nλ_GPU, θ[3])
        KernelFlows.Matern32!(buf1_GPU, θ[1], θ[2], buf3_GPU)
        buf1_GPU .+= buf2_GPU
        # buf1_GPU[1:n+1:n^2] .+= max(θ[4], T(2f-4))
        buf1_GPU[1:n+1:n^2] .+= θ[4]

        # This type of regularization works better for Float32 or
        # smaller. Fundamentally we should just continue increasing
        # reg until we get success here.
        
        # buf1_GPU[1:n+1:n^2] .*= 1.0001

        (out1, info) = CUDA.CUSOLVER.potrf!('U', buf1_GPU)
        (info != 0) && println("nonzero info from potrf!: $info")
        copy!(v1_n_GPU, M.ζ)
        CUDA.CUSOLVER.potrs!('U', buf1_GPU, v1_n_GPU)
        copy!(M.h, v1_n_GPU)
    end

    # Force freeing of arrays. You can accomplish this also by doing
    # GC.gc(true); CUDA.reclaim() *after* you return from the
    # function, but better force this to be automatic. Otherwise CUDA
    # only releases memory if it is needed again. This approach won't
    # work with persistent sessions on a shared machine.
    CUDA.unsafe_free!(v1_n_GPU)
    CUDA.unsafe_free!(v2_nλ_GPU)
    CUDA.unsafe_free!(Z_GPU)
    CUDA.unsafe_free!(buf1_GPU)
    CUDA.unsafe_free!(buf2_GPU)
    CUDA.unsafe_free!(buf3_GPU)
    CUDA.reclaim()

    nothing
end
