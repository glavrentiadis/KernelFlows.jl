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


# """Function to load a GPModel object from file"""
# function GPload(fname::String)
# end


# """Function to save a GPModel to a file"""
# function GPsave(M::GPModel, fname::String)
# end


"""Function to save an MVGPModel to a file"""
function GPsave(fname::String, MVM::MVGPModel)
    jldopen(fname, "a+") do file
        for (i,M) in enumerate(MVM.Ms)
            G = JLD2.Group(file, "M" * string(i))
            G["zeta"] = M.ζ
            G["h"] = M.h
            G["Z"] = M.Z
            G["DXmu"] = M.DX.μ
            G["DXF"] = M.DX.F
            G["lambda"] = M.λ
            G["theta"] = M.θ
            G["kernel"] = string(M.kernel)
        end

        file["DYmu"] = MVM.DY.μ
        file["DYF"] = MVM.DY.F
    end
end


"""Function to load an MVGPModel object from file"""
function GPload(fname::String)

    kerneltable = Dict("Matern32" => Matern32,
                       "Matern52" => Matern52,
                       "spherical_exp" => spherical_exp,
                       "spherical_sqexp" => spherical_sqexp)

    # Make Ms and DY available outside do-loop scope
    local Ms
    local DY

    jldopen(fname, "r") do file
        DY = DimRedStruct(file["DYmu"], file["DYF"])
        Ms = Vector{GPModel{eltype(DY.μ)}}()
        nM = length(keys(file)) - 2 # number of GPModels in MVGPModel
        for i in 1:nM
            G = file["M" * string(i)]
            ζ = G["zeta"]
            h = G["h"]
            Z = G["Z"]
            DX = DimRedStruct(G["DXmu"], G["DXF"])
            λ = G["lambda"]
            θ = G["theta"]
            kernel = kerneltranslations[G["kernel"]]
            push!(Ms, GPModel(ζ, h, Z, DX, λ, θ, kernel, identity, identity))
        end

    end
    MVGPModel(Ms, DY)

end
