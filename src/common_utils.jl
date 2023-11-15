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
export runningmedian, RMSE, splitrange, renormalize_columns, rebalance_data, get_random_partitions, kernel_matrix, kernel_matrix_fast, deciles

using LinearAlgebra
using Random
using Statistics

using Distances


runningmedian(x, n) = [median(x[i:i+n]) for i ∈ 1:length(x)-n]
RMSE(Y_true, Y_pred) = sqrt(sum((Y_true - Y_pred).^2)/size(Y_pred)[1])


"""Compute kernel matrix K, or if precision == true, its inverse."""
function kernel_matrix_fast(X::AbstractArray{T}, buf1::AbstractMatrix{T}, buf2::AbstractMatrix{T}, k::Function, θ::AbstractVector{T}; precision = true, nXlinear::Int = 1) where T <: Real
    s = Euclidean()

    pairwise!((a,b) -> k(s(a,b); θ = θ[end-3:end-2]), buf1, eachrow(X), eachrow(X), symmetric = true)
    buf1[diagind(buf1)] .+= max(exp(-15.), θ[end])
    lf = θ[end-1] # linear kernel component weight

    # Linear component only sees first nXlinear dimensions of X
    XXt = X[:,1:nXlinear] * X[:,1:nXlinear]'

    # Linear component with full input array X; for swapping you'll
    # also need to change GP_predict()
    # XXt = X * X'

    return precision ? ldiv!(buf2, cholesky!(buf1 + lf * XXt), UniformScaling(1.)(size(X)[1])) : Symmetric(buf1) + lf * XXt
end


function sqr(x::T) where T <: Real
    # iszero(x) ? zero(T) : sqrt(x)
    sqrt(x+1e-15)
end


"""A Distances.pairwise() workalike, but works with Zygote"""
function pairwise_Euclidean(X::AbstractMatrix{T}) where T <: Real
    H = -2. * X * X'
    D = .5 * diag(H)
    sqr.(Symmetric(H .- D .- D'))
end


"""Compute kernel matrix; no inversion. This function is
   autodifferentiable with Zygote."""
function kernel_matrix(X::AbstractArray{T}, k::Function, logθ::AbstractVector; nXlinear::Int = 1) where T <: Real
    # Notice that regularization apparently does not affect the flow;
    # however, when predicting it does affect the RMSE very much in Predict.jl.
    # N.B. You could also to regularization ala KRR w/ delta × RKHS

    # Linear component only for first X dimension
    KK = (X[:,1:nXlinear] * X[:,1:nXlinear]')
    H1 = pairwise_Euclidean(X)
    H2 = k.(H1; θ = exp.(logθ[1:end-2])) +
      Diagonal(exp(max(logθ[end], -15.)) * ones(size(X)[1])) + exp(logθ[end-1]) * KK

    # Linear component for all dimensions. (prediction code assumes
    # first component and similar comment swapping is needed in
    # GP_predict to change behavior.)
    # KK = X * X'
    # H2 = k.(H1; θ = exp.(logθ[1:end-2])) +
    # Diagonal(exp(max(logθ[end], -15.)) * ones(size(X)[1])) + exp(logθ[end-1]) * X * X'

    H2
end


"Splits integer range to as equal portions as possible, with number of
points given by nodes. Start and stop are always included."
function splitrange(start::Int, stop::Int, nodes::Int)
    n = stop - start
    r = nodes >= stop - start ? Vector(start:stop) : (n .* Vector(0:nodes) .÷ nodes) .+ start

end


function deciles(y::Vector{T}) where T <: Real
    cs = sort(y)
    n = length(y)
    decile_idx = Int.(range(0, n, length=11))
    decile_idx[1] = 1
    limits = cs[decile_idx]

    indexes = [collect(1:n)[(y .>= limits[i]) .&& (y .< limits[i+1])] for i ∈ 1:10]
end


"""Makes sure all training data are equally often sampled when not all
data are always observed"""
function get_random_partitions(N::Int, n::Int, niter::Int)
    k = N ÷ n # shorthand
    m = round(Int, (niter / k) + 1) # how many times partitioning N needs to be done
    R = [randperm(N) for _ ∈ 1:m]
    samples = [reshape(r[1:k*n], (k, n)) for r ∈ R]
    vcat(samples...)[1:niter,:]
end


"""Cut off norms of colums in V at the q'th quantile of all norms in
   V. Mean length of renormalized vectors is given by parameter
   new_scale, in case it is not set to zero. """
function renormalize_columns(V::AbstractMatrix{T}; q::T = 1., new_scale::T = 0.) where T <: Real
    norms = @view sqrt.(sum(V.^2, dims=2) .+ 1e-12)[:]
    a = quantile(norms, q)
    norms_new = norms[:]
    norms_new[norms_new .> a] .= a
    V_new = V ./ norms .* norms_new

    if new_scale != 0.
        V_new = new_scale * V_new ./ mean(sqrt.(sum(V_new.^2, dims=2)))
    end

    return V_new
end


function rebalance_data(ZX_tr, nleave, kernel, logλ; nolinear = true)
    npars = size(ZX_tr)[2]
    logθ = logλ[npars+1:end]
    nolinear && (logθ[end - 1] = -100.)
    pw = abs.(LossFunctions.kernel_matrix(ZX_tr .* logλ[1:npars]', kernel, logθ))

    v = sum(pw, dims = 2)[:]
    ndata = size(ZX_tr)[1]
    drop = zeros(Int, ndata - nleave)
    for i ∈ 1:ndata - nleave
        l = argmax(v)
        v .-= @view pw[l,:]
        v[l] = -Inf
        drop[i] = l
    end

    return setdiff(1:ndata, drop)
end


function rebalance_next()

    # Placeholder idea: remove training data points for which
    # predictions are best

end


# function rebalance_data(X, nleave; k = d -> exp(-d'*d), buf_rebalance = nothing)
#     pw = exp.(-100*pairwise(SqEuclidean(), X'))
#     v = sum(pw, dims = 2)[:]
#     ndata = size(X)[1]
#     drop = zeros(Int, ndata - nleave)
#     for i ∈ 1:ndata - nleave
#         s = sortperm(v)
#         l = s[end]
#         v .-= pw[l,:]
#         v[l] = -1
#         pw[l,:] .= 0
#         pw[:,l] .= 0
#         drop[i] = l
#     end
#     return setdiff(1:ndata, drop)
# end


# THIS FUNCTION IS BROKEN, DOES NOT WORK AS EXPECTED DUE TO BUG
# SOMEWHERE, EVEN THOUGH IT IS FASTER THAN NAIVE IMPLEMENTATION
# """Optimal resampling of a training set so that mutual correlations
#    among inputs are minimized. Returns the index set for inputs left in.
#    This is the pizza algorithm."""
# function rebalance_data_(X, nleave; k = d -> exp(-d'*d), buf_rebalance = nothing) # d1 = 1, d2 = 2)
#     (ndata, dims) = size(X)
#     nremove = ndata - nleave
#     # println(nremove)

#     buf = buf_rebalance == nothing ?  zeros(1, ndata) : buf_rebalance

#     buf2 = zeros(size(X)[2])
#     buf3 = similar(buf2)

#     function eucl!(x, y, buf1, buf2)
#         buf1 .= x
#         buf2 .= y
#         buf2 .*= -2.0
#         buf2 .+= buf1
#         buf1 .*= buf2
#         buf2 .= y
#         buf2 .*= buf2
#         buf1 .+= buf2
#         sum(buf1)
#     end

#     s(a,b) = eucl!(a, b, buf2, buf3)
#     erX = eachrow(X)

#     dbuf = zeros(1)
#     function kexp(d,dbuf)
#         dbuf[1] = -d
#         dbuf[1] *= d
#         dbuf[1] = exp(-dbuf[1])
#     end

#     v = zeros(ndata)

#     for (i,r) ∈ enumerate(erX)
#         pairwise!((a,b) -> kexp(s(a,b), dbuf), buf, (r,), erX)
#         v[i] = sum(buf)
#     end

#     # v = [@time sum(vval(i)) for i ∈ 1:ndata]
#     # vvalmax(i) = begin vv = vval(i); vv[i] = 0; maximum(vv) end
#     # v = [vvalmax(i) for i ∈ 1:ndata]
#     a = zeros(Int, nremove)

#     f(a,b) = kexp(s(a,b), dbuf)

#     for i ∈ 1:nremove
#         # println(i)
#         j = argmax(v)
#         Xj = @views eachcol(X[j,:])
#         pairwise!(f, buf, Xj, erX)
#         v .-= @view buf[:] # @view vval(j)[:]
#         a[i] = j
#         v[j] = 0
#     end
#     # println(size(unique(a)))
#     b = setdiff(1:ndata, a)

#     # dim1 = 1
#     # dim2 = 2
#     # p = scatter(X[a,dim1], X[a,dim2], alpha = 0.1, label = "removed")
#     # scatter!(p, X[b,dim1], X[b,dim2], label = "not removed")
#     # savefig(string("test_rebalance", rand(1:30000), ".png"))
#     # b
# end


# function rebalance_Xy(X::AbstractMatrix{T}, y::AbstractVector{T}, nleave::Int, k::Function, logdimscales::Vector{T}, logkpars::Vector{T}; buf_rebalance::Matrix{T} = nothing, workbuf = nothing) where T <: Real
#     k2(d) = k(d; logθ = logkpars)
#     @time wb = workbuf == nothing ? similar(X) : workbuf
#     @time wb .= X .* exp.(logdimscales)'
#     s_reb = @time KFCommon.rebalance_data(wb, nleave; k = k2, buf_rebalance)
#     @views X[s_reb,:], y[s_reb], s_reb
# end
