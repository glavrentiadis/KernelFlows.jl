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

function quantileplot(Y_te::AbstractMatrix{T}, Y_te_pred::AbstractMatrix{T}, wl, label) where T <: Real
    Y_res = Y_te - Y_te_pred
    qs = [.005, .025, .05, .25, .5, .75, .975, .995]
    colors = ["gray", "green", "blue", "red", "blue", "green", "gray"]
    labels = [nothing, nothing, nothing, "50%", "90%", "95%", "99%"]
    quantiles = hcat([quantile(y, qs) for y ∈ eachcol(Y_res)]...)
    p = plot()

    for (i,c) ∈ enumerate(colors)
        plot!(p, wl, quantiles[i,:], fillrange = quantiles[i+1,:], color = c, alpha = .2, label = labels[i])
    end

    plot!(p, title = label * ": error quantiles")
    plot!(p, ylabel = "Prediction error")
    plot!(p, xlabel = "Wavelength (nm)")
    p
end

"""Convenience function for subplots of matrixplot_preds"""
function pl!(p, x::Vector{T}, y::Vector{T}, y_pred::Vector{T};
             diff = false) where T <: Real
    diff && return scatter!(p, x, y - y_pred, label = nothing, markerstrokewidth = 0.1)
    scatter!(p, x, y, label = "truth", markerstrokewidth = 0.1)
    scatter!(p, x, y_pred, label = "predicted", markerstrokewidth = 0.1)
    return p
end



function matrixplot_preds(X_te::AbstractMatrix{T}, DXs::Vector{DimRedStruct{T}},
                          ZY_te::AbstractMatrix{T}, ZY_te_pred::AbstractMatrix{T};
                          diff = false, origspace = false) where T <: Real


    npcs = size(ZY_te_pred)[2]

    npars = [size(DX.F.vectors)[2] for DX in DXs]
    npars = origspace ? [size(X_te)[2] for _ in DXs] : npars
    npars_max = maximum(npars)

    p = plot(layout = (npcs,npars_max), size = (3000,1500), top_margin = -6mm)

    for i ∈ 1:npcs
        ZX_te = origspace ? X_te : original_to_reduced(X_te, DXs[i])
        for j ∈ 1:npars[i]
            println(j)
            pl!(p[i,j], ZX_te[:,j], ZY_te[:,i], ZY_te_pred[:,i]; diff)
            # Formatting below
            # i < npcs && plot!(p[i,j], xformatter = x -> "")
            j > 1 && plot!(p[i,j], yformatter = x -> "")
            i == npcs && plot!(bottom_margin = 3mm)
        end
        for j ∈ npars[i]+1:npars_max
            plot!(p[i,j], axis=false)
        end

        plot!(p[i,1], ylabel = "PC $i", left_margin = 12mm)
    end

    # for i ∈ 1:npars
    #     plot!(p[npcs,i], xlabel = parnames[i])
    # end
    plot!(p, margin = 0mm, legend = false)
    plot!(p[1], legend = true)

    t = diff ? "Prediction errors for test data" : "Predictions vs. truth"
    plot!(p, plot_title = t, titlefontsize = 24)
    # savefig("matrixplot_predictions_" * fname * ".pdf")
    # savefig("matrixplot.pdf")
    p
end

"""DX here is just a single DimRedStruct for X, as opposed to a vector"""
function matrixplot_preds(X_te::Matrix{T}, DX::DimRedStruct{T},
                          ZY_te::Matrix{T}, ZY_te_pred::Matrix{T};
                          diff = false, origspace = false) where T <: Real
    DXs = [DX for i ∈ 1:size(ZY_te_pred)[2]]
    matrixplot_preds(X_te, DXs, ZY_te, ZY_te_pred; diff, origspace)
end


function matrixplot_predictions(X_te, ZY_te, ZY_te_pred, fname, Yvarname, lossname; diff = false, npcs = size(ZY_te_pred)[2], npars = size(X_te)[2], fsize = (2000,2000), parnames = 1:size(X_te)[2])

    function pl!(p, indX, indY)
        if diff
            scatter!(p, X_te[:,indX], ZY_te[:,indY] - ZY_te_pred[:,indY], label = nothing, markerstrokewidth = 0.1)
        else
            scatter!(p, X_te[:,indX], ZY_te[:,indY], label = "truth", markerstrokewidth = 0.1)
            scatter!(p, X_te[:,indX], ZY_te_pred[:,indY], label = "predicted", markerstrokewidth = 0.1)
        end
        p
    end

    p = plot(layout = (npcs,npars), size = fsize, top_margin = -6mm)

    for i ∈ 1:npcs
        for j ∈ 1:npars
            pl!(p[i,j], j, i)
        end
        plot!(p[i,1], ylabel = "PC $i", left_margin = 12mm)
    end

    for i ∈ 1:npars
        plot!(p[npcs,i], xlabel = parnames[i])
    end
    plot!(p, margin = 0mm)

    for i ∈ 1:npcs
        for j ∈ 1:npars
            i < npcs && plot!(p[i,j], xformatter = x -> "")
            j > 1 && plot!(p[i,j], yformatter = x -> "")
            i == npcs && plot!(bottom_margin = 3mm)
        end
    end
    t = diff ? Yvarname * " / " * lossname * ": prediction errors for test data" : "Predictions vs. truth"
    plot!(p, plot_title = t, titlefontsize = 24)
    savefig("matrixplot_predictions_" * fname * ".pdf")
end

function plot_error_contribs(ZY_te, ZY_te_pred, DY, title)
    npcs = size(ZY_te_pred)[2]
    p = plot()
    data = (sum((abs.(ZY_te_pred - ZY_te[:,1:npcs])), dims = 1) .* sqrt.(DY.F.values[1:npcs]'))[:]
    data ./= sum(data)
    scatter!(p, data, label = "Errors")

    plot!(p, title = title, ylabel = "Error fraction", xlabel = "Principal component (output space)", xticks = npcs)
    # savefig("error_contributions_$(title).pdf")
end
