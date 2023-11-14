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
module KernelFlows

# Export- and using statements are at the top of the files below. Each
# of these files - minus common_utils.jl - describes a well-defined
# functionality that can be used either standalone or with just a few
# dependencies. The motivation for the code structure is to make it
# easy to separate these files into their own modules / packages.
include("kernel_functions.jl")
include("loss_functions.jl")

include("dimension_reduction.jl")
include("cca.jl")

include("parametric_models.jl")
include("nonparametric_models.jl")

include("common_utils.jl")
include("standard_transformations.jl")

end # module KernelFlows
