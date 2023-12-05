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
function train!(MVM::MVGPModel{T}, ρ::Function;
                ϵ::T = .05, niter::Int = 500, n::Int = 32,
                ngridrounds::Int = 6, navg::Union{Nothing, Int} = nothing,
                ζcomps::AbstractVector{Int} = 1:length(MVM.Ms),
                skip_K_update::Bool = false, quiet::Bool = false) where
                T <: Real

    if !quiet
        println("Initial scaling factors (log):")
        display(log.(vcat([M.λ' for M in MVM.Ms]...)))

        println("Initial kernel parameters (log):")
        display(log.(vcat([M.θ' for M in MVM.Ms]...)))
    end

    Threads.@threads for k ∈ ζcomps
        nXlinear = MVM.G.Xprojs[k].spec.nCCA
        train!(MVM.Ms[k], ρ; ϵ, niter, n, ngridrounds, navg,
               skip_K_update = true, quiet, nXlinear)
    end

    if !quiet
        println("Final scaling factors (log):")
        display(log.(vcat([M.λ' for M in MVM.Ms]...)))

        println("Final kernel parameters (log):")
        display(log.(vcat([M.θ' for M in MVM.Ms]...)))
    end


    # Update the MVGPModels sequentially, if requested.
    skip_K_update || update_MVGPModel!(MVM)

    MVM
end
