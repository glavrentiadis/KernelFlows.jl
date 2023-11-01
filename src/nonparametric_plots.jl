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
using Plots; gr() # Use Plots with GR backend
using Measures


function plot_label_comparison(Y_te, Y_te_predicted; fname = "label_comparison.png")
    p = plot(layout = (2,1), dpi = 300)
    plot!(p[1], [minimum(Y_te), maximum(Y_te)], [minimum(Y_te), maximum(Y_te)], label = "1-1 line")
    scatter!(p[1], Y_te, Y_te_predicted, xlabel = "True labels", ylabel = "Predicted labels", label = "Data")
    s = sortperm(Y_te[:])
    scatter!(p[2], Y_te[s], (Y_te_predicted - Y_te)[s], xlabel = "True labels", ylabel = "Prediction error")
    plot!(p[1], foreground_color_legend = nothing, background_color_legend = nothing, legend = :bottomright)
    plot!(p[2], foreground_color_legend = nothing, background_color_legend = nothing, legend = false)
    plot!(p, plot_title = "Test data prediction quality")
    savefig(fname)
end


function plot_NPKF_losses(all_ρ_values::Vector{Vector{T}};
                          pcs = 1:length(all_ρ_values), namesuffix = nothing, avgperiod = 20) where T <: Real

    npcs = length(pcs)
    plotscale = max(2, sqrt(length(pcs))) * 400 / 16
    p = plot(layout=(npcs,1), size=(9, 5*npcs) .* plotscale)

    for (i, pc) ∈ enumerate(pcs)
        ρ_values = all_ρ_values[pc]
        plot!(p[i], ρ_values, title = string("ρ: PC ", pc), legend = false)
        if avgperiod > 1
            ρavgd = [sum(ρ_values[max(j - avgperiod + 1, 1):j]) / min(j, avgperiod)
                     for j ∈ 1:length(ρ_values)]
            plot!(p[i], ρavgd, title = string("ρ: PC ", pc), legend = false, color = "red")
        end
    end

    if namesuffix != nothing
        savefig(string("loss_values_", namesuffix, ".pdf"))
    end

    return p
end

function plot_warping_step!(X, y;
                           i::Union{Int, Nothing} = nothing, # step index
                           t::Union{Float64, Nothing} = nothing, # time (for DE methods)
                           σ::Union{Float64, Nothing} = nothing, # length scale parameter
                           ρ::Union{Float64, Nothing} = nothing, # loss function value
                           p = nothing,
                           plotkwargs = ()) # extra keyword arguments to pass on to all subplots

    # construct title
    titletext = []
    push!(titletext, i == nothing ? "" : "i: $i")
    push!(titletext, t == nothing ? "" : "t: $(round(t, sigdigits = 3))")
    push!(titletext, ρ == nothing ? "" : "ρ: $(round(ρ, sigdigits = 3))")
    push!(titletext, σ == nothing ? "" : "σ: $(round(σ, sigdigits = 3))")
    titletext = join(titletext[titletext .!= ""], ", ")

    plim = 1.2 * maximum(abs.(X))

    basekwargs = (color = :isoluminant_cgo_70_c39_n256,
              xlim = (-plim, plim), ylim = (-plim, plim), colorbar = false,
              aspect_ratio = :equal)

    p = p == nothing ? plot(; basekwargs...,) : p

    # plot_warping_step!(p, X, y; kwargs = basekwargs)
    scatter!(p, X[:,1], X[:,2]; marker_z = y, aspect = :square, basekwargs..., plotkwargs...)
    if titletext != ""
        plot!(p, title = titletext)
    end

    p
end



function plot_warping(sols, X_idxs, y, plotiters; Y_pc = 1, D_X = nothing, markershape = :circle)
    p = plot(layout = length(plotiters), # will be e.g. (3,3) for 9
             aspect_ratio = :equal, # equal scaling for subplots
             dpi = 300,
             size = (1200,1200), # canvas size
             margin = -1mm,
             foreground_color_border = "white")

    for (i, it) ∈ enumerate(plotiters)
        X = sols[Y_pc].u[it][:,X_idxs]
        X = D_X == nothing ? X : KFCommon.reduced_to_original(X, D_X)

        plim = 1.1 * maximum(abs.(X))

        t = hasproperty(sols[Y_pc], :t_values) ? sols[Y_pc].t_values[it] : nothing
        ρ = hasproperty(sols[Y_pc], :ρ_values) ? sols[Y_pc].ρ_values[it] : nothing

        plotkwargs = (legend = false, ticks = false, markershape =
        markershape, markersize = 10, markeralpha = 0.8)

        plot_warping_step!(X, y; p = p[i], t = t, i = it, ρ = ρ, plotkwargs)
    end
    p
end

function plot_warping_testing(p, ZX_trajs, X_idxs, y, plotiters; Y_pc = 1, D_X = nothing, markershape = :star4)
    for (i, it) ∈ enumerate(plotiters)
        X = ZX_trajs[Y_pc][i][:,X_idxs]
        X = D_X == nothing ? X : KFCommon.reduced_to_original(X, D_X)
        plotkwargs = (legend = false, ticks = false, markershape = markershape,
                      markersize = 10, markerstrokecolor = "black", markeralpha = 0.8)
        plot_warping_step!(X, y; p = p[i], plotkwargs )
    end
    p
end

function plot_training_and_testing(X_tr::Matrix{T}, y_tr::Vector{T}, X_te::Matrix{T}, y_te::Vector{T}, y_te_pred::Vector{T}; inds = (1,2), kws = ()) where T <: Real
    kws = (color = :vik, aspect = :equal, dpi = 300, markerstrokecolor = :white, kws...)
    p = scatter(X_tr[:,1], X_tr[:,2], marker_z = y_tr; marker = :hexagon, label = "Training data", markersize = 8, kws...)
    scatter!(p, X_te[:,1], X_te[:,2], marker_z = y_te; marker = :square, label = "True labels", markersize = 6, kws...)
    scatter!(p, X_te[:,1], X_te[:,2], marker_z = y_te_pred; marker = :circle, label = "Predicted labels", markersize = 4, kws...)
end

function plot_training_and_testing_1d(x_tr::Vector{T}, y_tr::Vector{T}, x_te::Vector{T}, y_te::Vector{T}, y_te_predicted::Vector{T}) where T <: Real
    scatter(x_te, y_te, dpi = 300, label = "Test data, true labels",
            foreground_color_legend = nothing,
            background_color_legend = nothing)
    scatter!(x_te, y_te_predicted[:], label = "Test data, predicted labels")
    scatter!(x_tr, y_tr[:], dpi = 300, label = "Training data")

end






# function plot_warping()
#     kwargs = (xlim = nothing, ylim = nothing)
# end

# function plot_final_state(loss_q = 0.99)
# end






# OLD FUNCTIONS BELOW

# function plot_ρ(ρs::Vector{T}) where T <: Real
#     plot(ρs, label="ρ as function of iteration")
#     savefig("loss_values.pdf")
# end

# function plot_flowstep(X::Array{T}, sf::Vector{Int64}, ∂ρ∂X::Matrix{T},
#                        y::Vector{T}, ρ::T, suffix_index::Int64;
#                        plot_arrows::Bool=false, fdgrad::Union{Array{T}, Nothing}=nothing,
#                        save_figure::Bool = true, fignameprefix::String="kf_step") where T <: Real

#     # We assume here that the labels are -1 and +1 50%-50%.

#     N = size(X)[1]
#     n = N ÷ 2
#     Nf = size(sf)[1]

#     sfC = setdiff(1:N, sf)
#     sc = sf[1:(Nf÷2)]
#     scatter(X[1,sf], X[2,sf], marker_z=y[sf], markershape=:square, # aspect_ratio=:equal,
#             xlim=(-4,4), ylim=(-3,6), figsize=(4, 4), dpi=300, label="Xf (batch)")

#     scatter!(X[1,sfC], X[2,sfC], marker_z=y[sfC], markershape=:star5, label="XfC (complement of Xf)")


#     if plot_arrows
#         # scale of the quivers, so longest is 3. NOTE! This is negative,
#         # since we are minimizing the loss function. That's why there is a
#         # minus sign when the X values are incremented.
#         sca = -3. / sqrt(maximum(sum(∂ρ∂X.^2; dims=2)))
#         quiver!(X[1,:], X[2,:], quiver=(sca*∂ρ∂X[1,:], sca*∂ρ∂X[2,:]), color="black", alpha=0.2)

#         if fdgrad != nothing
#             quiver!(X[1,:], X[2,:], quiver=(sca*fdgrad[1,:], sca*fdgrad[2,:]), color="green")
#         end
#     end

#     s = string("ρ(X,y) = ", round(ρ, digits=3))
#     title!(s)

#     png(string(fignameprefix, lpad(suffix_index, 6, "0"), ".png"))
# end
