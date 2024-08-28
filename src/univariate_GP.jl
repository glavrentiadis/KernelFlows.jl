#  Copyright 2023 California Institute of Technology
#
#  Licensed under the Apache License, Version 2.0 (the \"License\");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an \"AS IS\" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# Author: Jouni Susiluoto, jouni.i.susiluoto@jpl.nasa.gov
#
export GPModel, update_GPModel!


"""Univariate GP model struct. This struct is sufficient for
prediction of scalar zy at a new x, so that different zys can be
combined to construct the original y."""
struct GPModel{T}
    ζ::Vector{T}        # ζ = zytransf.(zy_tr), needed for updating h
    h::Vector{T}        # K⁻¹ ζ, where K is the kernel matrix
    Z::Matrix{T}        # transformed X_tr *after* applying λ-scaling
    λ::Vector{T}        # X_tr scaling factors for each input dimension
    θ::Vector{T}        # kernel parameters for spherical + linear kernel
    kernel::Kernel      # kernel struct including the kernel function
    zytransf::Function    # nonlinear 1-d output transformations
    zyinvtransf::Function # inverse 1-d output transformations
    ρ_values::Vector{T} # loss function values from latest training
    λ_training::Vector{Vector{T}} # scaling factors from training
    θ_training::Vector{Vector{T}} # kernel parameters from training
end


# These need struct GPModel => include only after its definition
include("univariate_training.jl")
include("univariate_prediction.jl")


function GPModel(ZX_tr::Matrix{T}, # inputs after reduce()
                 zy_tr::Vector{T}, # outputs after reduce()
                 kernel::Kernel;
                 λ::Union{Nothing, Vector{T}} = nothing,
                 θ::Union{Nothing, Vector{T}} = nothing,
                 transform_zy::Bool = false) where T <: Real

    # Gaussianize 1-d labels if requested
    transform_zy || (zytransf = zyinvtransf = identity)
    transform_zy && ((zytransf, zyinvtransf) = gaussianize_transforms(zy_tr))
    ζ = zytransf.(zy_tr)

    ntr, nλ = size(ZX_tr)

    h = zeros(T, ntr)
    # Initialize so, that if ZX_tr were Gaussian, points are on
    # average 1 std away from each other.
    λ == nothing && (λ =  sqrt(T(2)) * ones(T, nλ) ./ nλ)
    θ == nothing && (θ = kernel.θ_start[:])

    @assert length(λ) == nλ "Invalid λ length"
    nθ = length(θ)
    typeof(kernel) == UnaryKernel && (@assert nθ == 4 "Invalid θ length")

    Z = ZX_tr .* λ'

    return GPModel(ζ, h, Z, λ, θ, kernel, zytransf, zyinvtransf, T[], Vector{T}[], Vector{T}[])
end


default_λ(M::GPModel) = length(M.λ_training) == 0 ? M.λ : M.λ_training[end]
default_θ(M::GPModel) = length(M.θ_training) == 0 ? M.θ : M.θ_training[end]

"""Updates 1-d GPmodel M after new parameters newλ and/or newθ have
   been obtained by training the model. Even if these parameters are
   not given, M.h is recomputed, unless skip_K_update == true (it is
   not computed when GPModel is initialized)."""
function update_GPModel!(M::GPModel{T};
                         newλ::Vector{T} = default_λ(M),
                         newθ::Vector{T} = default_θ(M),
                         buf1::Union{Nothing, Matrix{T2}} = nothing,
                         buf2::Union{Nothing, Matrix{T2}} = nothing,
                         skip_K_update::Bool = false) where {T<:Real,T2<:Real}

    # Update M.Z, M.λ, and M.θ, if requested
    if newλ != nothing
        newλ[newλ .< 1e-9] .= 1e-9
        newλ[newλ .> 1e2] .= 1e2
        l = newλ ./ M.λ
        M.Z .*= l'
        M.λ .= newλ
    end

    if newθ != nothing
        M.θ .= newθ
    end

    if !skip_K_update
        # Allocate buffers
        ntr = length(M.ζ)
        H = Float64
        (buf1 == nothing) && (buf1 = zeros(H, (ntr, ntr)))
        (buf2 == nothing) && (buf2 = zeros(H, (ntr, ntr)))

        # If buf1 of type T2 != T was given, we compute Cholesky with
        # that type. Using e.g. Float32 instead of Float64 is faster,
        # but less stable.
        H = eltype(buf1)
        h = H.(M.ζ) # need type converion for potrs!

        kernel_matrix_fast!(M.kernel, H.(M.θ), H.(M.Z), buf1, buf2; precision = false)
        LAPACK.potrf!('U', buf2) # cholesky
        LAPACK.potrs!('U', buf2, h) # solve h <- inv(K) * h
        M.h .= h # copy K⁻¹ζ where it belongs
    end

    M
end


"""Updates a vector of GPModels by calling update_GPModel!() for each
of them. The parameters are supplied in matrices. By default, the
inverse covariance is recomputed using whatever parameter values are
in the variables M.λ and M.θ"""
function update_GPModel!(Ms::Vector{GPModel{T}}; skip_K_update::Bool = false) where T <: Real

    nM = length(Ms)
    ndata = length(Ms[1].ζ)
    parallel = ndata < 10000

    txt1 = parallel ? "in parallel" : "serially"
    txt2 = skip_K_update ? "Skip updating" : "Also updates"
    println("Updating $(length(Ms)) GPs in $txt1. $txt2 K⁻¹y in M.h.")

    # Helper function for getting buffers
    u(nd, nt) = [zeros(T, (nd, nd)) for _ in 1:nt]

    print("\rCompleted 0/$nM tasks...")
    if parallel
        nt = Threads.nthreads()
        buf1s = u(ndata, nt)
        buf2s = u(ndata, nt)
        computed = zeros(Int, nt)

        Threads.@threads :static for (i,M) ∈ collect(enumerate(Ms))
            tid = Threads.threadid()
            update_GPModel!(M; buf1 = buf1s[tid], buf2 = buf2s[tid], skip_K_update)
            computed[Threads.threadid()] += 1
            print("\rCompleted $(sum(computed))/$nM tasks...")
        end
    else
        buf1, buf2 = u(ndata, 2)
        for (i,M) ∈ collect(enumerate(Ms))
            update_GPModel!(M; skip_K_update, buf1, buf2)
            print("\rCompleted $i/$nM tasks...")
    end
    end
    println("done!")
    Ms
end



"""Convenience function to obtain logα from a GPModel{T}"""
function get_logα(M::GPModel{T}) where T <: Real
    log.(vcat(M.λ, M.θ))
end


"""Any kernel-specific code that's needed for sparsifying inputs can
be implemented here. The default function does nothing"""
function sparsify_inputs_hook(::Kernel, M::GPModel, newdims::Vector{Int}) end


"""Update kernel.nXlinear when sparsifying GPModels with UnaryKernel
kernels."""
function sparsify_inputs_hook(newkernel::UnaryKernel, M::GPModel, newdims::Vector{Int})
    max_nXlinear = M.kernel.nXlinear
    newkernel.nXlinear = sum(newdims .< max_nXlinear)
end


function sparsify_inputs(M::GPModel{T}, nleave::Int) where T <: Real
    ζ = M.ζ[:]
    h = similar(M.h)
    newdims = sort(sortperm(var(M.Z, dims = 1)[:], rev = true)[1:nleave])
    Z = M.Z[:, newdims]
    λ = M.λ[newdims]
    θ = M.θ[:]

    newkernel = deepcopy(M.kernel)
    sparsify_inputs_hook(newkernel, M, newdims)

    M = GPModel(ζ, h, Z, λ, θ, newkernel, M.zytransf, M.zyinvtransf, T[], Vector{T}[], Vector{T}[])

    update_GPModel!(M)
    return M, newdims
end
