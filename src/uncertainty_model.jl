# Recover Y_tr from a MVGPModel object
function recover_Y_tr(MVM::MVGPModel{T}) where T <: Real
    all_ζ = hcat([M.ζ for M in MVM.Ms]...)
    all_ζ * (MVM.G.Yproj.vectors .* MVM.G.Yproj.values')' .* MVM.G.σY' .+ MVM.G.μY'
end


struct KernelFlowsUncertaintyModel{T}
    MVMD::MVGPModel{T} # GP model for variance prediction
    PLMDY::Vector{KernelFlows.PiecewiseLinearMap{T}} # De-Gaussianization of predicted data
    GD::KernelFlows.GPGeometry{T} # GPGeometry to reconstruct non-orthogonal data
    P_nugget::Matrix{T} # nugget for the model
end


function predict_training_kfold(MVM::MVGPModel{T}, X_tr::Matrix{T}, Y_tr::Matrix{T}; k::Int = 2) where T <: Real
    # First let's construct training data with k-fold cross validation
    # Y_tr = recover_Y_tr(MVM) # perhaps works but untested
    ntr = size(Y_tr)[1]
    s = randperm(ntr)

    # Indexes of data for k-fold XV
    kchunks = [s[chunk] for chunk in KernelFlows.ranges(1, ntr, k)]
    Y_tr_pred = similar(Y_tr)

    for i in 1:k
        GC.gc()
        str = setdiff(1:ntr, kchunks[i])
        ste = kchunks[i]
        X_tr_k = X_tr[str,:]
        X_te_k = @views X_tr[ste,:]
        Y_tr_k = Y_tr[str,:]
        MVMk = MVGPModel(X_tr_k, Y_tr_k, :Matern32, MVM.G)
        allpars = get_parameters(MVM)
        set_parameters!(MVMk, allpars; update_K = true)
        Y_tr_pred[ste,:] .= KernelFlows.predict(MVMk, X_te_k)
    end

    Y_tr_pred
end


function construct_uncertainty_model(MVM::MVGPModel{T}, X_tr::Matrix{T}, Y_tr::Matrix{T};
                                     k::Int = 2,
                                     GD_kwargs::Dict,
                                     GD2_kwargs::Dict) where T <: Real

    Y_tr_pred = predict_training_kfold(MVM, X_tr, Y_tr; k)

    # MVMk = MVGPModel(X_tr, Y_tr, :Matern32, MVM.G)
    # allpars = get_parameters(MVM)
    # allpars[:,end] .= 1e-3
    # Y_tr_pred .= predict(MVMk, X_tr)

    # Training data for uncertainty model. D stands for "difference"
    YD_tr = Y_tr_pred - Y_tr
    XD_tr = X_tr # Could also be Y_tr_pred

    GD = dimreduce(XD_tr, YD_tr; GD_kwargs...)
    ZDY_tr = reduce_Y(YD_tr, GD) # Dimension-reduced (Z) training label residuals

    YD_nullspaceproj = YD_tr - recover_Y(ZDY_tr, GD) # Complement of ZDY_tr

    # C_nugget is the covariance of output data not explicitly modeled.
    C_nugget = cov(YD_nullspaceproj)
    F = eigen(C_nugget)
    mpos = F.values .> 0 # mask for positive eigenvalues

    # FIXME WRONG NAME!!! This is the square root of the covariance,
    # for drawing from the noise term
    P_nugget = sqrt.(F.values[mpos])' .* F.vectors[:, mpos]

    # ZDY columns are independent. These are now the
    # variances. Columns are χ²-distributed
    ZDY2_tr = ZDY_tr.^2

    # Enforce positivity by going to log space
    ZDY2L_tr = log.(ZDY2_tr)

    GD2 = dimreduce(XD_tr, ZDY2L_tr; GD2_kwargs...)

    # This is the model for log variances
    # MVMD = MVGPModel(XD_tr, ZDY2L_tr, :spherical_sqexp, GD2);
    MVMD = MVGPModel(XD_tr, ZDY2L_tr, :Matern32, GD2);
    allpars = get_parameters(MVMD)

    # Set large nugget as we must not interpolate exactly, just find
    # approximate conditional mean. Similarly, set linear part to
    # (effectively) zero as we are not interested in fitting a linear
    # model to variances. FIXME! make these tunable.
    allpars[end,:] .= T(1f-1) # nugget
    allpars[end-1,:] .= T(1f-9) # linear

    set_parameters!(MVMD, allpars; update_K = false)

    # @time train!(MVMD; ρ = KernelFlows.ρ_RMSE_no_LOO, niter = 20000, n = 80, ϵ = 1f-3,
    @time train!(MVMD; ρ = KernelFlows.ρ_RMSE, niter = 50000, n = 80, ϵ = 1f-2,
                 optalg = :SGD, mbalg = :multicenter, mbargs = Dict(:nnb => 10),
                 # optalg = :AMSGrad, mbalg = :randompartitions,
                 quiet = true, update_K = true)

    # We don't want to get the exact residual variances for the
    # posterior mean prediction, rather a smooth function, as the
    # exact values are just draws from some distribution. That's why
    # we do the k-fold prediction here as well.
    ZDY2L_tr_pred = predict_training_kfold(MVMD, XD_tr, ZDY2L_tr; k)

    ZDYscales_tr = exp.(5f-1 * ZDY2L_tr_pred) # 0.5 is the square root

    # Absolute residuals with unit standard deviation. The
    # standardization is needed, as we are making the assumption that
    # data are draws from a fixed distribution whose variance changes
    # based on the inputs. With this scaling we standardize the
    # residuals so that they can be pooled to describe this reference
    # distribution.
    ZDYnor_tr = ZDY_tr ./ ZDYscales_tr

    # FIXME: Parameterize how many bins are used for the PiecewiseLinearMaps
    PLMDY =  KernelFlows.PiecewiseLinearMaps(ZDYnor_tr; n = 300, mapping=:gaussian)

    UQM = KernelFlowsUncertaintyModel(MVMD, PLMDY, GD, P_nugget)

end


# Convenience function to return standard deviations for X from a
# KernelFlowsUncertaintyModel object, in the MVMD-transformed space
function ZDstd(UQM::KernelFlowsUncertaintyModel{T}, X::Matrix{T}) where T <: Real
    ZDstd = exp.(T(.5) * KernelFlows.predict(UQM.MVMD, X))
    # recover_Y(ZDvar, UQM.GD)
end


function sample_uqmodel(UQM::KernelFlowsUncertaintyModel{T}, x_te::Vector{T}; ndraws::Int = 30) where T <: Real
    # Sampling amounts to just drawing from a Gaussian, getting the
    # standardized non-Gaussian residuals with PLMDY, scaling those with
    # MVMD-predicted stds, and then project back

    nxz = length(x_te)
    x_te = reshape(x_te, (1, nxz))[:,:]
    # zDYG2_te_pred = exp.(predict(UQM.MVMD, x_te))

    # for i in 1:length(zDYG2_te_pred)
    #     zDYG2_te_pred[i] = clamp(zDYG2_te_pred[i], extrema(UQM.MVMD.Ms[i].ζ)...)
    # end

    # The above quantities zDYG2_te_pred are now predictions of
    # variance in the Gaussianized orthogonal space (for one
    # point). We need to transform that back with the
    # de-Gaussianization transform.

    zDYscales = exp.(5f-1 * KernelFlows.predict(UQM.MVMD, x_te)) # Get standard deviations for draws

    nZY = length(zDYscales)
    normals = randn(T, (ndraws, nZY)) # draw random normals

    # FIXME make this match the quantiles that are omitted in
    # constructing the residuals

    # IS THIS UQM.PLMDY.values[[2,end-1]]?

    normals = clamp.(normals, T(-3), T(3))

    # zDYG2_te_pred = clamp.(zDYG2_te_pred, 0, 100)
    # zDYG_te_pred = sqrt.(zDYG2_te_pred) # variance -> std
    # display(zDYG_te_pred)


    # zDYG_draws = normals .* zDYG_te_pred # draws from Gaussianized distribution

    # println("\n\nSTART")
    # display(zDYG_te_pred)

    # de-Gaussianized draws
    # zDY_draws = KernelFlows.tr_inv(UQM.PLMDY, zDYG_draws)
    zDY_draws = KernelFlows.tr_inv(UQM.PLMDY, normals)
    zDY_draws .*= zDYscales

    # H = GD.Yproj.vectors[:,1:nZY] .* GD.Yproj.values[1:nZY]'
    # zDY_draws * H'

    zDY_draws = clamp.(zDY_draws, -10,10) # This is ad hoc, to avoid unstable tail behavior. Not sure if it's even needed...

    uncs = recover_Y(zDY_draws, UQM.GD)

    nY = size(UQM.P_nugget)[2]

    # println(size(uncs))
    uncs += (UQM.P_nugget * randn(T, (nY, ndraws)))'

    -uncs
    # FIXME NUGGET STILL MISSING

    # Eigendecomposition is PDPᵀ, so we need square root. Compare to
    # drawing with Cholesky
    # H .*= sqrt.(azDY)'
    # H .*= GD.Yproj.values[1:naz]'
end
