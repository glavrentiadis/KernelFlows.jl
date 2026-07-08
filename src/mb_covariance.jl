
# FIXME Covariance minibatching needs a way to figure out a way to pass the
# pointer to θ, or maybe we need to do something else like make
# constructors take the GPModel objects instead of X etc? That might
# be cleaner in any case.


# mutable struct CovarianceMinibatch <: AbstractMinibatch
#     const X::AbstractArray # pointer to input data
#     const epoch_length::Int # recompute covariances and draw samples every this many steps
#     const κ::Int # number of centers
#     const n # total size of the minibatch
#     const niter::Int # number of iterations
#     epoch_s::Vector{Vector{Int}} # minibatches for current epoch
#     all_centers::Vector{Vector{Int}} # center points for the whole training
#     i::Int
# end


# function CovarianceMinibatch(X::AbstractArray; # pointer
#                              θ::AbstractArray; # pointer
#                              n::Int = n_default,
#                              κ::Int = κ_default,
#                              niter::Int = 1000,
#                              epoch_length::Int = 50)


#     ndata = size(X)[1]
#     all_centers = get_random_partitions(ndata, κ, niter) # get random centers
#     CovarianceMinibatch(X, min(niter, epoch_length), κ, n,
#                         niter, Vector{Int}[], all_centers, 1)
# end


# function minibatch(B::CovarianceMinibatch, λ::Vector{T}) where T <: Real
#     k = (B.i - 1) % B.epoch_length
#     (k == 0) && update_samples(B, λ)
#     s = B.epoch_s[k+1]
#     B.i += 1
#     s
# end


# function update_samples(B::CovarianceMinibatch, λ::Vector{T}) where T <: Real
# #  tree = KDTree(B.X' .* λ, leafsize = 10)
#     ndata = size(B.X)[1]
#     epoch_s = Vector{Int}[]
#     epoch_end = min(B.i + B.epoch_length - 1, B.niter) # don't go past niter
#     for c in B.all_centers[B.i:epoch_end]
#         # Get neighborhoods. B.κ first elements are the centers.
#         s_nbs = @views unique(vcat(c, knn(tree, B.X[c,:]' .* λ, B.nnb+1)[1]...))
#         m = ndata - length(s_nbs) # number of points to sample global data from
#         nglobal = B.n - length(s_nbs)
#         r1 = randperm(m)[1:nglobal] # global data indexes

#         r2 = setdiff(1:ndata, s_nbs)[r1]
#         push!(epoch_s, vcat(s_nbs, r2))
#     end
#     B.epoch_s = epoch_s
# end

