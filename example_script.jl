# Make sure to add this locally by pressing ] in julia terminal and
# then writing add /path/to/KernelFlows.jl.

# Also, you should add "using Revise" to your startup.jl, which in
# Linux is at ~/.julia/startup.jl. You need to import the KernelFlows
# module with

using KernelFlows

# Then you need to load data into rows. Convention: inputs are X,
# outputs are Y. _tr is training, _te is testing. Use NPZ for reading
# nympy files, and DelimitedFiles for ascii data, etc. Look up online
# for instructions. I usually write a file for reproducibility, with
# get_data() function. Something like

X, Y = get_data()

# Split it randomly to training and testing (if you did not do that
# already) by using funtion split_data(). The nte is the number of
# data that you leave for testing, so X_te and Y_te will be of size
# (nte, nXdims) and (nte, nYdims). You can optionally specify also
# ntr, in case you don't want to use all data for training. Splitting
# multiple data sets is also possible - see split_data() in
# common_utils.jl for usage.

X_tr, Y_tr, X_te, Y_te = split_data(X, Y; nte = 500)

# Now, you need to do some dimension reduction. The number of
# Y-dimensions is your number of scalar GPs. If you only have one, you
# still can do X-space dimension reduction / rotation with CCA, and
# you would specify nYPCA = 1, nYCCA = 0 below. Typically the labels would
# be higher-dimensional, and you might want to specify more than one Y
# dimension. Each one of these becomes another scalar GP.

G = dimreduce(X_tr, Y_tr, nYCCA = 1, nYPCA = 2, nXCCA = 2,
              reg_CCA = 1e-1, reg_CCA_X = 1e0, maxdata = 3000,
              scale_Y = false, dummyXdims = true)

# Now you have an nXdims + 2-dimensional input space for prediction.
# Next, build the multivariate GP model (MVM). This is the interface
# also for constructing univariate models. The defaults here should
# work fine.

MVM = MVGPModel(X_tr, Y_tr, :Matern32_analytic, G; transform_zy = false)

# We next need to learn the model (learn parameters). The default loss
# is the L2 loss (ρ_RMSE), but there are many more available such as
# the Kernel Flows loss, ρ_KF.

# The quick & dirty way of training the model uses the defaults, while
# overriding just the central parameters: number of iterations,
# minibatch size, and learning rate. The call to traing looks like:

train!(MVM; niter = 500, n = 128, ϵ = 1e-3)

# This is enough in most settings. For more flexibility, there are two
# different ways to construct minibatches (multi-center and random
# partitions), as well as two different optimizers (SGD, AMSGrad)
# available. For the minibatching object what matters is the minibatch
# type and size and number of iterations. For the optimizer what
# matters most is the learning rate ϵ. We set those with

optargs = Dict(:ϵ => 1e-3) # see optimizers.jl for details
mbargs = Dict(:niter => 700, :n => 64, :epoch_length => 500) # minibatching.jl

# and the function call to train the becomes

train!(MVM; ρ = ρ_RMSE, optalg = :AMSGrad, optargs, mbalg = :multicenter, mbargs)

# Notice that in Julia you do not need to write optargs = optargs etc
# - it is found automatically if a variable with that name is found in
# scope.

# Sometimes one wants to train models without inverting the kernel
# matrix at the end, as that may be costly if there is a large number
# of training data (over 10k). This is useful when one wants to do
# repeated training while e.g. changing parameters from training to
# training. To skip the inversion, just add update_K = false to
# the train! kwargs:

train!(MVM; ρ = ρ_RMSE, optalg = :AMSGrad, optargs,
       mbalg = :multicenter, mbargs, update_K = false)

# You can look at how the scaling factors (λ), kernel parameters (θ),
# and loss function values changed during training, with

plot_training(MVM)

# Predicting is simple: With test data in X_te, one would predict with

Y_te_pred = predict(MVM, X_te)

# The results can be plotted using e.g. the Plots package. For scalar
# data a good starting point is looking at the 1-1 plot by something
# like

scatter(Y_te_pred, Y_te)

# The scatter() function is available from both Makie and Plots
# packages.

# There are some (somewhat immature) standard plotting functions
# available in the parametric_plots.jl file. Some of the more useful
# ones are matrixplot_preds, and plot_training.
