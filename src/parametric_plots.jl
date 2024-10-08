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

using Plots
using Makie

function quantileplot!(p::Plots.Subplot, Y_te::AbstractMatrix{T}, Y_te_pred::AbstractMatrix{T};
                       x::AbstractArray = 1:size(Y_te)[2], μ = zeros(T, size(Y_te)[2])) where T <: Real
    Y_res = Y_te - Y_te_pred
    qs = [.005, .025, .05, .25, .5, .75, .95, .975, .995]
    colors = ["gray", "green", "blue", "red", "red", "blue", "green", "gray"]
    labels = [nothing, nothing, nothing, nothing, "50%", "90%", "95%", "99%"]
    quantiles = hcat([quantile(y, qs) for y ∈ eachcol(Y_res)]...) .+ μ'

    for (i,c) ∈ enumerate(colors[1:4])
        Plots.plot!(p, x, quantiles[i,:], fillrange = quantiles[i+1,:],
              color = c, alpha = .2, label = labels[i])
        Plots.plot!(p, x, quantiles[9-i+1,:], fillrange = quantiles[9-i,:],
              color = c, alpha = .2, label = labels[i])
    end
    Plots.plot!(p, xlims = extrema(x))

    p
end

function quantileplot!(ax::Makie.Axis, Y_te::AbstractMatrix{T}, Y_te_pred::AbstractMatrix{T};
                       x::AbstractVector = 1:size(Y_te)[2], μ = zeros(T, size(Y_te)[2])) where T <: Real
    Y_res = Y_te - Y_te_pred
    qs = [.005, .025, .05, .25, .5, .75, .95, .975, .995]
    colors = ["gray", "green", "blue", "red", "red", "blue", "green", "gray"]
    labels = [nothing, nothing, nothing, nothing, "50%", "90%", "95%", "99%"]
    quantiles = hcat([quantile(y, qs) for y ∈ eachcol(Y_res)]...) .+ μ'

    lineidxs = [1,2,3,4,6,7,8,9]

    for (i,c) ∈ enumerate(colors)
        lines!(ax, x, quantiles[lineidxs[i],:], color=c)
        band!(ax, x, quantiles[i,:], quantiles[i+1,:], color=c, alpha=.2, label = labels[i])
    end
    Makie.xlims!(ax, extrema(x))
end

function plot_training(MVM::MVGPModel; p = nothing, title = "Training results")

    nY = length(MVM.Ms)
    nYCCA = MVM.G.Yproj.spec.nCCA
    nYPCA = MVM.G.Yproj.spec.nPCA

    r = length(MVM.Ms[1].ρ_values)
    m = r > 1000 ? splitrange(1, r, 1000) : 1:r

    xlabels1 = ["CC $i (w=$(round(MVM.G.Yproj.values[i], sigdigits=2)))" for i in 1:nYCCA]
    xlabels2 = ["PC $i (w=$(round(MVM.G.Yproj.values[i+nYCCA], sigdigits=2)))" for i in 1:nYPCA]
    xl = [xlabels1..., xlabels2...]

    p == nothing && (p = Plots.plot(layout = grid(3, nY, heights = [0.4, 0.4, 0.2]),
                                    size = (2400, 800), plot_title = title)) # , link = :both)

    for i in 1:3nY
        # no y tick labels for columns >1
        (i-1)%nY > 0 && Plots.plot!(p[i], yformatter = _ -> "",
                                    right_margin = -4mm, left_margin = -4mm)
        i > nY && Plots.plot!(p[i], top_margin = -6mm)
        i < nY && Plots.plot!(p[i], bottom_margin = -6mm)
    end

    for i in 1:nY
        M = MVM.Ms[i]
        λs = log.(hcat(M.λ_training...)[:,m])
        nXCCA = MVM.G.Xprojs[i].spec.nCCA # number of X CCA vectors for this Y dim
        Plots.plot!(p[i], λs[1:nXCCA,:]', legend = false, xformatter = _ -> "")
        Plots.plot!(p[i], λs[nXCCA+1:end,:]', legend = false, color = "gray", alpha = .2)
        Plots.plot!(p[i+nY], log.(hcat(M.θ_training...)[:,m]'),
                    legend = false, xformatter = _ -> "", top_margin = 0mm)

        xlab = length(xl) > 0 ? xl[i] : ""
        Plots.plot!(p[i+2nY], log.(M.ρ_values[m]), legend = false,
                        xlabel = xlab, top_margin = 0mm)
    end

    niter_tot = length(MVM.Ms[1].ρ_values)
    Plots.plot!(p[1], ylabel = "log(λ)", right_margin = -4mm)
    Plots.plot!(p[nY+1], ylabel = "log(θ)", right_margin = -4mm)
    Plots.plot!(p[2nY+1], ylabel = "log(ρ)", bottom_margin = 12mm, right_margin = -4mm)
    # Plots.plot!(p, xticks = [1, niter_tot÷2, niter_tot] , xrotation = 70)
    Plots.plot!(p[1], left_margin = 12mm, top_margin = 6mm)
    p
end


"""Convenience function for subplots of matrixplot_preds"""
function pl!(p, x::Vector{T}, y::Vector{T}, y_pred::Vector{T};
             diff = false) where T <: Real
    diff && return Makie.scatter!(p, x, y - y_pred, strokewidth = 1)
    Makie.scatter!(p, x, y, label = "truth", strokewidth = 1)
    Makie.scatter!(p, x, y_pred, label = "predicted", strokewidth = 1)

    return p
end

using Makie

function matrixplot_preds(MVM::MVGPModel{T}, X_te::AbstractMatrix{T}, Y_te::AbstractMatrix{T};
                          diff = false, origspace = false, plot_dummyXdims::Bool = true,
                          Y_te_pred::Union{Nothing, AbstractMatrix{T}} = nothing,
                          Xtransfs = ones(Int, size(X_te)[2]), nYdims::Int = 0, nXdims::Int = 0, offset::Int = 0) where T <: Real

    # These are the same as in VSWIREmulator.jl
    xt = [identity, log, cosd, x -> cosd(x-90), x -> log(180-x), exp, sqrt, x -> sign(x) * x^2]
    X_te = X_te[:,:]
    for (i,t) in enumerate(Xtransfs)
        X_te[:,i] .= xt[t].(X_te[:,i])
    end

    ZY_te_pred = (Y_te_pred == nothing) ?  predict(MVM, X_te; recover_outputs = false) : reduce_Y(Y_te_pred, MVM.G)

    ZY_te = reduce_Y(Y_te, MVM.G)

    nY = size(ZY_te_pred)[2]
    nX = plot_dummyXdims ? length(MVM.G.Xprojs[1].spec.sparsedims) : MVM.G.Xprojs[1].spec.nCCA + MVM.G.Xprojs[1].spec.nPCA
    nX = origspace ? size(X_te)[2] : nX

    nX = nXdims == 0 ? nX : min(nX, nXdims)
    nY = nYdims == 0 ? nY : min(nY, nYdims)

    f = Figure(size = (3200, 3200))
    axes = [[Axis(f[i,j]) for i in 1:nY] for j in 1:nX]
    # p = plot(layout = (nY, nX), size = (3200,2000), top_margin = -6mm)

    for i in offset + 1:offset + nY
        k = i - offset
        M = MVM.Ms[i]
        ZX_te = origspace ? X_te : reduce_X(X_te, MVM.G, i)
        for j in 1:nX
            print("\r$i, $j")
            !diff && Makie.scatter!(axes[j][k], M.Z[:,j] / M.λ[j], M.zyinvtransf.(M.ζ), color = :gray, legend = false, alpha=.3, strokewidth = 1)
            pl!(axes[j][k], ZX_te[:,j], ZY_te[:,k], ZY_te_pred[:,k]; diff)
            if i - offset < nY
                hidexdecorations!(axes[j][k], grid = false)
                linkxaxes!(axes[j][k], axes[j][nY])
            end
            if (j > 1) hideydecorations!(axes[j][k], grid = false)
                linkyaxes!(axes[j][k], axes[1][k])
            end
        end
    end

    t = diff ? "Prediction errors for test data" : "Predictions vs. truth"
    f
end


function plot_11(Y_te, Y_te_pred1, Y_te_pred2)
    nvecs = size(Y_te)[2]
    p = plot(layout = nvecs, size = (1920,1200))
    for i in 1:nvecs
        scatter!(p[i], Y_te[:,i], Y_te_pred1[:,i])
        scatter!(p[i], Y_te[:,i], Y_te_pred2[:,i])
    end

    for i in 1:nvecs
        plot!(p[i], [-2,8], [-2,8], color="red")
    end
    p
end


function plot_11(MVT::TwoLevelMVGP{T}, X_te::AbstractArray{T}, Y_te::AbstractArray{T}) where T <: Real
    f = Figure(size = (3200,3200))
    Y_te_pred1, Y_te_pred2 = predict(MVT, X_te)

    Y_te_pred = Y_te_pred1 + Y_te_pred2 #  .* MVT.MVM2.G.Yproj.values'

    ncols = 4
    for (i,y) in enumerate(eachcol(Y_te))
        ax = Axis(f[(i-1)÷ncols+1,i%ncols+1])

        Makie.scatter!(ax, y, Y_te_pred1[:,i], label = "One level", strokewidth = 1)
        Makie.scatter!(ax, y, Y_te_pred[:,i], label = "Two levels", strokewidth = 1)
        Makie.lines!(ax, [-2,6], [-2,6], color = :red)
    end
    f
end


function plot_error_contribs(ZY_te, ZY_te_pred, G, title)
    npcs = size(ZY_te_pred)[2]
    p = plot()
    data = (sum((abs.(ZY_te_pred - ZY_te[:,1:npcs])), dims = 1) .* G.Yproj.values[1:npcs]')[:]
    data ./= sum(data)
    scatter!(p, data, label = "Errors")

    plot!(p, title = title, ylabel = "Error fraction", xlabel = "Principal component (output space)", xticks = npcs)
    # savefig("error_contributions_$(title).pdf")
end
