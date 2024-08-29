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
export runningmedian, RMSE, splitrange, renormalize_columns, rebalance_data, get_random_partitions, kernel_matrix, kernel_matrix_fast, deciles, split_data

using LinearAlgebra
using Random
using Statistics

using Distances


runningmedian(x, n) = [median(x[i:i+n]) for i ∈ 1:length(x)-n]
RMSE(Y_true, Y_pred) = sqrt(sum((Y_true - Y_pred).^2)/size(Y_pred)[1])


"Splits integer range to as equal portions as possible, with number of
points given by nodes. Start and stop are always included."
function splitrange(start::Int, stop::Int, nodes::Int)
    n = stop - start
    r = nodes >= stop - start ? Vector(start:stop) : (n .* Vector(0:nodes) .÷ nodes) .+ start
end


"""Split inputs X and outputs Y randomly into training and
testing. The number of points in testing is given by kwarg nte. Random
seed can be fixed for reproducibility. By default the very edges of
inputs values go to the training set, in order to avoid extrapolation
and maximize coverage."""
function split_data(X::Matrix{T}, Y::Matrix{T};
                    ntr::Int = -1, nte::Int = 500,
                    seed::UInt = rand(UInt),
                    edges_to_training::Int = 2) where T <: Real
    s_tr, s_te = randomsplit(X, nte; seed, edges_to_training, ntr)
    X[s_tr,:], Y[s_tr,:], X[s_te,:], Y[s_te,:]
end


function split_data(X::Matrix{T}, Y_all::Vector{Matrix{T}};
                    ntr::Int = -1, nte::Int = 500,
                    seed::UInt = rand(UInt),
                    edges_to_training::Int = 2) where T <: Real
    s_tr, s_te = randomsplit(X, nte; seed, edges_to_training, ntr)
    X[s_tr,:], [Y[s_tr,:] for Y in Y_all], X[s_te,:], [Y[s_te,:] for Y in Y_all]
end


function split_data(Zs::Vector{Matrix{T}};
                    ntr::Int = -1, nte::Int = 500,
                    seed::UInt = rand(UInt),
                    edges_to_training::Int = 2) where T <: Real
    s_tr, s_te = randomsplit(Zs[1], nte; seed, edges_to_training, ntr)
    [[Z[s_tr, :] for Z in Zs]..., [Z[s_te, :] for Z in Zs]...]
end


function randomsplit(nfull::Int, npart::Int; seed::UInt = rand(UInt))
    Random.seed!(seed)
    s = randperm(nfull)
    s_te = s[1:npart]
    s_tr = s[npart+1:end]
    return s_tr, s_te
end


function randomsplit(X::AbstractArray, nte::Int;
                     ntr::Int = -1, seed::UInt = rand(UInt),
                     edges_to_training::Int = 2)
    n = size(X)[1]

    starts = [sortperm(c)[1:edges_to_training] for c in eachcol(X)]
    ends = [sortperm(c, rev = true)[1:edges_to_training] for c in eachcol(X)]
    s_edges = unique(vcat(starts..., ends...))
    s_remain = setdiff(1:n, s_edges)
    n_remain = length(s_remain)

    (s_tr, s_te) = randomsplit(n_remain, nte; seed)
    s_tr = [s_edges..., s_remain[s_tr]...]
    s_tr = ntr == -1 ? s_tr : s_tr[1:ntr]
    return (s_tr, s_remain[s_te])
end


function deciles(y::Vector{T}) where T <: Real
    cs = sort(y)
    n = length(y)
    decile_idx = Int.(range(0, n, length=11))
    decile_idx[1] = 1
    limits = cs[decile_idx]

    indexes = [collect(1:n)[(y .>= limits[i]) .&& (y .< limits[i+1])] for i ∈ 1:10]
end


"""Randomly split ndata data points into n subsets. This makes sure
all training data are equally often sampled in SGD. Returns a vector
of vectors, each one of which contains indices for minibatches for
each iteration"""
function get_random_partitions(ndata::Int, n::Int, niter::Int)
    k = ndata ÷ n # shorthand
    m = round(Int, (niter / k) + 1) # how many times data needs to be partitioned
    R = [randperm(ndata) for _ ∈ 1:m]
    samples = [reshape(r[1:k*n], (k, n)) for r ∈ R]
    [collect(r) for r in eachrow(vcat(samples...)[1:niter,:])]
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


function print_parameters(Ms::Vector{GPModel{T}}) where T <: Real
    println("Log scaling factors (λ):")
    display(round.(log.(vcat([M.λ' for M in Ms]...)), sigdigits = 2))
    println("\nLog kernel parameters (θ):")
    display(round.(log.(vcat([M.θ' for M in Ms]...)), sigdigits = 2))
end


function rebalance_data(X::AbstractMatrix{T}, nleave::Int, MVM::MVGPModel{T};
                        ydims::AbstractVector{Int} = 1:length(MVM.Ms),
                        nXlinear::Union{Int, Nothing} = nothing) where T <: Real
    ndata = size(X)[1]
    pw = zeros(ndata, ndata)
    for ydim in ydims
        nXlinear == nothing ? nXl(MVM, ydim) : nXlinear
        Z = reduce_X(X, MVM.G, ydim)
        pw .+= MVM.G.Yproj.values[ydim] * abs.(kernel_matrix(Z, MVM.Ms[ydim].kernel, log.(MVM.Ms[ydim].θ), nXlinear = 0))
    end

    v = sum(pw, dims = 2)[:]
    ndata = size(X)[1]
    drop = zeros(Int, ndata - nleave)
    for i ∈ 1:ndata - nleave
        l = argmax(v)
        v .-= @view pw[l,:]
        v[l] = -Inf
        drop[i] = l
    end

    # Return the optimized trimmed training data
    return setdiff(1:ndata, drop)
end


function rebalance_next()

    # Placeholder idea: remove training data points for which
    # predictions are best

end



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


# function rebalance_Xy(X::AbstractMatrix{T}, y::AbstractVector{T}, nleave::Int, k::Kernel, logdimscales::Vector{T}, logkpars::Vector{T}; buf_rebalance::Matrix{T} = nothing, workbuf = nothing) where T <: Real
#     k2(d) = k(d; logθ = logkpars)
#     @time wb = workbuf == nothing ? similar(X) : workbuf
#     @time wb .= X .* exp.(logdimscales)'
#     s_reb = @time KFCommon.rebalance_data(wb, nleave; k = k2, buf_rebalance)
#     @views X[s_reb,:], y[s_reb], s_reb
# end
