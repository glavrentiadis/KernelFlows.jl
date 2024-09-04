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


# This file contains the different minibatching methods available in
# KernelFlows.jl. It works the same way as the optimizers.jl class,
# defining a mutable struct for each minibatching method.

using NearestNeighbors


abstract type AbstractMinibatch end


const n_default = 64
const κ_default = 5


mutable struct RandomPartitions <: AbstractMinibatch
    const ndata::Int
    const n::Int # size of minibatch
    const κ::Int # number of points to predict
    const niter::Int # number of iterations
    const all_s::Vector{Vector{Int}} # minibatch
    i::Int
end


function RandomPartitions(X::AbstractArray;
                          n::Int = n_default, niter::Int = 1000, κ::Int = κ_default)
    ndata = size(X)[1]
    all_s = get_random_partitions(ndata, n, niter)
    RandomPartitions(ndata, n, κ, niter, all_s, 1)
end


function minibatch(B::RandomPartitions, λ::Vector{T}) where T <: Real
    s = B.all_s[B.i]
    B.i = B.i + 1
    s
end


mutable struct MulticenterMinibatch <: AbstractMinibatch
    const X::AbstractArray # pointer to input data
    const epoch_length::Int # recompute KDTree every this many steps
    const κ::Int # number of centers
    const nnb::Int # number of neighboring points for each center
    const n # total size of the minibatch
    const niter::Int # number of iterations
    epoch_s::Vector{Vector{Int}} # minibatches for current epoch
    all_centers::Vector{Vector{Int}} # center points for the whole training
    i::Int
end


function MulticenterMinibatch(X::AbstractArray;
                              n::Int = n_default, niter::Int = 1000,
                              κ::Int = κ_default,
                              epoch_length::Int = 500,
                              nnb::Int = min(3*n÷κ÷4, 2*size(X)[2]))

    ndata = size(X)[1]
    all_centers = get_random_partitions(ndata, κ, niter)
    MulticenterMinibatch(X, min(niter, epoch_length), κ, nnb, n,
                         niter, Vector{Int}[], all_centers, 1)
end


function minibatch(B::MulticenterMinibatch, λ::Vector{T}) where T <: Real
    k = (B.i - 1) % B.epoch_length
    (k == 0) && update_samples(B, λ)
    s = B.epoch_s[k+1]
    B.i += 1
    s
end


function update_samples(B::MulticenterMinibatch, λ::Vector{T}) where T <: Real
    tree = KDTree(B.X' .* λ, leafsize = 10)
    ndata = size(B.X)[1]
    epoch_s = Vector{Int}[]
    epoch_end = min(B.i + B.epoch_length - 1, B.niter) # don't go past niter
    for c in B.all_centers[B.i:epoch_end]
        # Get neighborhoods. B.κ first elements are the centers.
        s_nbs = @views unique(vcat(c, knn(tree, B.X[c,:]', B.nnb+1)[1]...))
        m = ndata - length(s_nbs) # number of points to sample global data from
        nglobal = B.n - length(s_nbs)
        r1 = randperm(m)[1:nglobal] # global data indexes

        r2 = setdiff(1:ndata, s_nbs)[r1]
        push!(epoch_s, vcat(s_nbs, r2))
    end
    B.epoch_s = epoch_s
end


"""Test and plot results to verify that minibatching works as intended."""
function test_Multicenter()
    X = rand(1000,2)
    κ = 3
    B = KernelFlows.MulticenterMinibatch(X; n = 100, κ, niter = 100, nnb = 6)
    s = KernelFlows.minibatch(B, ones(2))
    Plots.scatter(X[s[κ+1:end],1], X[s[κ+1:end],2], label = "Minibatch / others")
    Plots.scatter!(X[s[1:κ],1], X[s[1:κ],2], label = "Minibatch / centers")
    sdiff = setdiff(1:1000,s)
    Plots.scatter!(X[sdiff,1], X[sdiff,2], label = "Data not in minibatch", alpha = 0.1)
end



function get_minibatcher(mbalg::Symbol, X::AbstractArray{T};
                       mbargs::Dict{Symbol,H} = Dict{Symbol,Any}()) where {T<:Real,H<:Any}
    mbalg == :multicenter && (return MulticenterMinibatch(X; mbargs...))
    mbalg == :randompartitions && (return RandomPartitions(X; mbargs...))

end
