# TODO: reorganize for-loops

# To use prediction capabilities on GPUs, just include() the desired
# predict_BACKEND.jl, where BACKEND can be oneAPI (Intel), CUDA
# (NVIDIA), or CPU. To add others, just create another file with the
# correct translation.


struct GPUPredictionBuffer{T} # <: AbstractPredictionBuffer{T}
    M_cross::GPUMatrix{T} # cross covariance matrix storage
    # M_lin::GPUMatrix{T} # linear part storage
    M_Ztr::GPUMatrix{T} # X1 squared
    M_Zte::GPUMatrix{T} # X2 squared
    v_ntr::GPUVector{T}
    v_nte::GPUVector{T}
    v_nZdim::GPUVector{T}
    nte::Int # prediction batch size
end


function GPUPredictionBuffer(k::Union{KernelFlows.AnalyticKernel{T}, KernelFlows.UnaryKernel{T}},
                              ntr::Int, nte::Int, nZdim::Int) where T <: Real

    GPUPredictionBuffer(GPUzeros(T, (nte, ntr)), # GPUzeros(T, (nte, ntr)),
                             GPUzeros(T, (ntr, nZdim)), GPUzeros(T, (nte, nZdim)),
                             GPUzeros(T, ntr), GPUzeros(T, nte), GPUzeros(T, nZdim), nte)
end


function GPUreduce_X(X_GPU::GPUMatrix{T}, # Inputs on GPU
                      V::GPUMatrix{T}, # Projection vectors (Xproj[i].vectors ./ Xproj[i].values')
                      μ::Vector{T}, # G.μX
                      σ::Vector{T}, # G.σX
                      λ::GPUVector{T}, # M.λ
                      Zte_out::GPUMatrix{T}) where T <: Real


    # display(Matrix(X_GPU))
    # display(Matrix(V))
    # display(Vector(λ))
    # display(Matrix(Zte_out))

    v_nX = GPUVector(T.(μ))
    X_GPU .-= v_nX' # center
    copyto!(v_nX, σ)
    X_GPU ./= v_nX' # scale

    GPUgemm!('N', 'N', one(T), X_GPU, V, zero(T), Zte_out)

    Zte_out .*= λ'

    # display(Matrix(Zte_out))

    # Undo X_GPU scaling so that it can be reused # Note that this
    # could also be done by changing arguments μ and σ in the for-loop
    # that calls this, which would be cheaper...
    copyto!(v_nX, σ)
    X_GPU .*= v_nX' # scale
    copyto!(v_nX, μ)
    X_GPU .+= v_nX' # center
end


function pairwise_Euclidean!(Z1::GPUMatrix{T}, Z2::GPUMatrix{T},
                             pb::GPUPredictionBuffer{T}) where T <: Real

    #display(Matrix(Z2))
    #display(Matrix(pb.M_cross))

    pb.M_Zte .= Z1.^2
    # display(Matrix(pb.M_Zte))
    # display(Matrix(Z1))
    # gagga
    pb.M_Ztr .= Z2.^2

    (nte, ntr) = size(pb.M_cross)
    nXdims = size(Z1)[2]

    pb.v_nte .= sum(pb.M_Zte, dims = 2)
    pb.v_ntr .= sum(pb.M_Ztr, dims = 2)

    # display(Vector(pb.v_nte)')
    # display(Vector(pb.v_ntr)')
    pb.v_ntr .+= T(1e-14)

    pb.M_cross .= pb.v_nte .+ pb.v_ntr'
    # pb.M_cross .+= pb.v_ntr' #  .+ T(1e-14)

    GPUgemm!('N', 'T', T(-2), Z1, Z2, T(1), pb.M_cross) # -2 Z1 * X2'

    # display(Matrix(Z1))
    # display(Matrix(Z2))
    # display(Matrix(Z1) * Matrix(Z2)')
    # display(Matrix(pb.M_cross))

    pb.M_cross .= GPUsqrt.(pb.M_cross) # .+ pb.M_lin .+ 1e-14)
    # pb.M_cross .*= CUDA.rsqrt.(pb.M_cross) # .+ pb.M_lin .+ 1e-14)
end


function GPUpredict(MVM::MVGPModel{T}, X_te::Matrix{T}; f_inv::Function = identity) where T <: Real

    ntr, nZdim = size(MVM.Ms[1].Z)
    nλ = nZdim
    nte_full, nXdim = size(X_te)
    nWdim = length(MVM.Ms)
# println("1")
    # # pb.nte will be nte_batch, not nte_full.
    nte_batch = nte_full ÷ 6
    pb = GPUPredictionBuffer(MVM.Ms[1].kernel, ntr, nte_batch, nZdim)
# println("2")

    Xbatch_gpu = GPUzeros(Float32, (pb.nte, nXdim))
# println("3")

    μ = MVM.G.μX
    σ = MVM.G.σX
# println("4")

    λ_buf = GPUzeros(Float32, nλ)
    all_λ = GPUMatrix(Float32.(hcat([M.λ for M in MVM.Ms]...)))
    all_h = GPUMatrix(Float32.(hcat([M.h for M in MVM.Ms]...)))
# println("5")

    # These should really go to predictionbuffer
    Zte_buf = similar(pb.M_Zte)
    Ztr_buf = similar(pb.M_Ztr)
# println("6")

    Z_full_out = GPUzeros(Float32, (nte_full, nWdim))
# println("7")

    # Indices where to copy batch data for GPUreduce_X()
    CI_dest = CartesianIndices((1:nte_batch, 1:nXdim))
# println("8")

    for i in 1:(nte_full ÷ pb.nte)
        # # println("New i = $i")
        CI_src = CartesianIndices(((i-1)*nte_batch+1:i*nte_batch, 1:nXdim))
        copyto!(Xbatch_gpu, CI_dest, X_te, CI_src)
# println("00A")

        for j in 1:nWdim
            # # println("start")
            V = GPUMatrix(Float32.(MVM.G.Xprojs[j].vectors ./ MVM.G.Xprojs[j].values'))
            # λ = cu(MVM.Ms[j].λ)
            copyto!(λ_buf, 1, all_λ, (j-1)*nλ + 1, nλ)
            copyto!(Ztr_buf, MVM.Ms[j].Z) # No multiplying with λ
            θ = MVM.Ms[j].θ
# println("a")

            GPUreduce_X(Xbatch_gpu, V, μ, σ, λ_buf, Zte_buf)
            pairwise_Euclidean!(Zte_buf, Ztr_buf, pb)
# println("b")

            pb.M_cross .= KernelFlows.Matern32.(pb.M_cross, θ[1], θ[2])
            # pb.M_cross .+= θ[3]/(-2f0) .* pb.M_lin
# println("c")

            # Cross covariance times M.h
            # copyto!(pb.v_ntr, MVM.Ms[j].h)
            copyto!(pb.v_ntr, 1, all_h, (j-1)*ntr + 1, ntr)
# println("d")

            GPUgemv!('N', 1f0, pb.M_cross, pb.v_ntr, 0f0, pb.v_nte)
# println("e")

            # Add linear component by (Zte * (Ztr * M.h))
            GPUgemv!('T', θ[3], Ztr_buf, pb.v_ntr, 0f0, pb.v_nZdim)
            GPUgemv!('N', 1f0, Zte_buf, pb.v_nZdim, 1f0, pb.v_nte)
# println("f")

            CI_Zout = CartesianIndices(((i-1)*nte_batch+1:i*nte_batch, j:j))
            Z_full_out[CI_Zout] .= pb.v_nte
        end
    end
# println("GGG")

    # Drop unnecessary buffers here
    pb = nothing
    # GC.gc()
    # CUDA.reclaim()

    H = GPUMatrix(Float32.(MVM.G.Yproj.vectors' .* MVM.G.Yproj.values))
    Y_te_pred_gpu = Z_full_out * H .* GPUVector(Float32.(MVM.G.σY))' .+ GPUVector(Float32.(MVM.G.μY))'
    Y_te_pred_gpu .= f_inv.(Y_te_pred_gpu)
    Y_te_pred_cpu = Matrix(Y_te_pred_gpu)
    Y_te_pred_gpu = nothing

    # Release buffers
    # CUDA.unsafe_free!(Y_te_pred_gpu)
    # CUDA.unsafe_free!(H)
    # GC.gc()
    # CUDA.reclaim()

    return Y_te_pred_cpu
end
