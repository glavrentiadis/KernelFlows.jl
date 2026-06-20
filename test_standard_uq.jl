using KernelFlows

function get_data(n::Int)
    f(x) = [x[1] + sin(x[2]), x[3] - x[2], x[1]*x[3]]
    X = 2π*(rand(n, 3) .- .5)
    Y = hcat([f(x) for x in eachrow(X)]...)'[:,:]
    Y .= 0
    Y += randn(size(Y)) .* [.1, 2., 5.]' # add noise
    return X, Y
end

X, Y = get_data(5000)
X_tr, Y_tr, X_te, Y_te = split_data(X, Y; nte = 500)
G = dimreduce(X_tr, Y_tr, nYCCA = 0, nYPCA = 0, nXCCA = 0, nXPCA = 0,
              reg_CCA = 1e-1, reg_CCA_X = 1e0, maxdata = 3000,
              scale_Y = false, dummyXdims = false)

uqmodel = :standard
MVM = MVGPModel(X_tr, Y_tr, :zerokernel, G;
                transform_zy = false, uqmodel)

optargs = Dict(:ϵ => 1e-2) # see optimizers.jl for details
mbargs = Dict(:niter => 2000, :n => 160) # minibatching.jl

train!(MVM; ρ = ρ_MLE, optalg = :AMSGrad, optargs, mbalg = :randompartitions, mbargs)

plot_training(MVM)

(Y_te_pred, UQR) = predict(MVM, X_te; quantify_uncertainties = true)

using Plots
using LinearAlgebra
p11 = Plots.scatter(Y_te_pred, Y_te, xlabel = "Predicted", ylabel = "True", aspect_ratio = :equal, xlim = (-12,12), ylim = (-12,12))
Plots.plot!(p11, [-12,12], [-12,12], color = :red)

println("Should be around [.1, 2, 5]:")
C_post = KernelFlows.recover_covariance(UQR, 5)
println(sqrt.(diag(C_post))')
