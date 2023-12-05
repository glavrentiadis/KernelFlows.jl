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


using JLD2


"""Saves Projection as a group in a JLD2 file."""
function save_Projection(P::Projection, G::JLD2.Group)
    G["vectors"] = P.vectors
    G["values"] = P.values
    G["nCCA"] = P.spec.nCCA
    G["nPCA"] = P.spec.nPCA
    G["ndummy"] = P.spec.ndummy
    G["dummydims"] = collect(P.spec.dummydims)
end


function load_Projection(G::JLD2.Group)
    spec = ProjectionSpec(G["nCCA"], G["nPCA"], G["ndummy"], G["dummydims"])
    Projection(G["vectors"], G["values"], spec)
end


"""Saves a GPModel as a group in a JLD2 file"""
function save_GPModel(M::GPModel{T}, G::JLD2.Group) where T <: Real
    G["zeta"] = M.ζ
    G["h"] = M.h
    G["Z"] = M.Z
    G["lambda"] = M.λ
    G["theta"] = M.θ
    G["rho_values"] = M.ρ_values
    G["lambda_training"] = M.λ_training
    G["theta_training"] = M.θ_training
    G["kernel"] = string(M.kernel)
end


function load_GPModel(G::JLD2.Group)

    kerneltable = Dict("Matern32" => Matern32,
                       "Matern52" => Matern52,
                       "spherical_exp" => spherical_exp,
                       "spherical_sqexp" => spherical_sqexp)

    ζ = G["zeta"]
    h = G["h"]
    Z = G["Z"]
    λ = G["lambda"]
    θ = G["theta"]
    ρ_values = G["rho_values"]
    kernel = kerneltable[G["kernel"]]
    λ_training = G["lambda_training"]
    θ_training = G["theta_training"]

    GPModel(ζ, h, Z, λ, θ, kernel, identity, identity, ρ_values, λ_training, θ_training)
end


function save_GPGeometry(geom::GPGeometry{T}, G::JLD2.Group) where T <: Real

    for (i,Xp) in enumerate(geom.Xprojs)
        g = JLD2.Group(G, "Xproj" * string(i))
        save_Projection(Xp, g)
    end

    g = JLD2.Group(G, "Yproj")
    save_Projection(geom.Yproj, g)

    G["Xmean"] = geom.μX
    G["Xstd"] = geom.σX
    G["Ymean"] = geom.μY
    G["Ystd"] = geom.σY
    G["reg_CCA"] = geom.reg_CCA

    G
end


function load_GPGeometry(G::JLD2.Group)

    σX = G["Xstd"]
    μX = G["Xmean"]
    σY = G["Ystd"]
    μY = G["Ymean"]
    reg_CCA = G["reg_CCA"]
    Yproj = load_Projection(G["Yproj"])
    Xprojs = Vector{Projection{eltype(Yproj.values)}}()

    for i in 1:length(Yproj.values)
        push!(Xprojs, load_Projection(G["Xproj" * string(i)]))
    end

    GPGeometry(Xprojs, Yproj, μX, σX, μY, σY, reg_CCA)
end


"""Function to save an MVGPModel to a file or a new group in a file."""
function save_MVGPModel(MVM::MVGPModel, fname::String; grpname::String = "")
    jldopen(fname, "a+") do file
        rootgrp = (grpname == "") ? file.root_group : JLD2.Group(file, grpname)
        for (i,M) in enumerate(MVM.Ms)
            (M.zytransf != identity) &&
                (println("WARNING! zytransf are not saved correctly for now!"))
            G = JLD2.Group(rootgrp, "M" * string(i))
            save_GPModel(M, G)
        end

        G = JLD2.Group(rootgrp, "G")
        save_GPGeometry(MVM.G, G)
    end
end


"""Function to load an MVGPModel object from file"""
function load_MVGPModel(fname::String; grpname::Union{Nothing, String} = nothing)

    local MVM

    jldopen(fname, "r") do file
        grp = (grpname == nothing) ? file.root_group : file[grpname]
        MVM = load_MVGPModel(grp)
    end

    MVM
end


function load_MVGPModel(G::JLD2.Group)

    geom = load_GPGeometry(G["G"])
    Ms = Vector{GPModel{eltype(geom.μX)}}()
    nM = length(keys(G)) - 1 # number of GPModels in MVGPModel

    for i in 1:nM
        push!(Ms, load_GPModel(G["M" * string(i)]))
    end

    MVGPModel(Ms, geom)
end
