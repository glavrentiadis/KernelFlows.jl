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


# This file contains the different minibatching methods available in
# KernelFlows.jl. It works the same way as the optimizers.jl class,
# defining a mutable struct for each minibatching method.

abstract type AbstractMinibatch end


const n_default = 64
const κ_default = 3


include("mb_covariance.jl")
include("mb_multicenter.jl")
include("mb_randompartitions.jl")



function get_minibatcher(mbalg::Symbol, X::AbstractArray{T};
                       mbargs::Dict{Symbol,H} = Dict{Symbol,Any}()) where {T<:Real,H<:Any}
    mbalg == :multicenter && (return MulticenterMinibatch(X; mbargs...))
    mbalg == :randompartitions && (return RandomPartitions(X; mbargs...))

end
