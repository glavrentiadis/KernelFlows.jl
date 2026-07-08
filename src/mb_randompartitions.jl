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


"""Test and plot results to verify that minibatching works as intended."""
function test_RandomPartitions(; p = Plots.plot())
    cs = palette(:tab10)
    ms = 7

    X = rand(1000,2)
    λ = rand(2)
    κ = 3
    B = RandomPartitions(X .* λ'; n = 64, niter = 1000, κ)
    s0 = minibatch(B, ones(2))
    s1 = s0[1:59]
    s2 = s0[60:64]
    sdiff = setdiff(1:1000,vcat(s1, s2))
    # s2 = minibatch(B, ones(2))
    # s3 = minibatch(B, ones(2))
    Plots.scatter!(p, X[s1,1], X[s1,2], label = "Predictor points", alpha = .5, color = cs[1], ms = ms)
    Plots.scatter!(p, X[s2,1], X[s2,2], label = "Predicted points", alpha = 1.0, color = cs[4], z_order = 2, ms = ms)
    Plots.scatter!(p, X[sdiff,1], X[sdiff,2], label = "Not in minibatch", alpha = 0.2, color = :gray, ms = ms)
    # Plots.scatter!(p, X[s3,1], X[s3,2], label = "Third minibatch", alpha = 0.7)
    # sdiff = setdiff(1:1000,vcat(s1, s2, s3))
    Plots.plot!(p, xticks = [], yticks = [])

end
