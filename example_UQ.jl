include("example_script.jl")

# For the residual model that is used under k-fold cross validation to
# create training data for the log-variance model
GD_kwargs = Dict(:nYCCA => 2, :nYPCA => 1, :nXCCA => 1, :nXPCA => 1,
                 :reg_CCA => 1e-1, :reg_CCA_X => 1e-1,
                 :maxdata => 3000, :scale_Y => true, :dummyXdims => true)

# TRY ALSO WITH RANK-DEFICIENT GD AND CHECK THAT L_nugget has
# reasonable numbers in it, not just epsilons. Also, L_nugget does not
# need to be full rank, only rank full - r


# For the log-variance model.
GD2_kwargs = Dict(:nYCCA => 2, :nYPCA => 1, :nXCCA => 1, :nXPCA => 1,
                  :reg_CCA => 1e-1, :reg_CCA_X => 1e-1,
                  :maxdata => 3000, :scale_Y => true, :dummyXdims => true)

UQM = KernelFlows.construct_uncertainty_model(MVM, X_tr, Y_tr; k = 2, GD_kwargs, GD2_kwargs)

ndraws = 10
UQR = KernelFlows.quantify_uncertainties(UQM, X_te[1:ndraws,:])

nplotx = ndraws # how many predictions to plot
draws = KernelFlows.sample_uqmodel(UQR, ndraws)


(nte_tot, nY) = size(Y_te)
(_, nX) = size(X_te)
p_nGUQ = plot(layout = (nY,nX), size = (1200,800), dpi = 150)
for k in 1:nY
    for j in 1:nX
        for i in 1:ndraws
            lab = i == 1 ? "samples (diff: should span zero)" : false
            scatter!(p[k,j], X_te[1:nplotx,j], Y_te_pred[1:nplotx,k] - Y_te[1:nplotx,k] + draws[1:nplotx,k,i], color = "gray", alpha = 0.6, label = lab, xlabel = "Input $j", ylabel = "Output dim $k")
        end
    end
end
p
