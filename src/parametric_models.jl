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
export train!
export predict!
export MVGPModel_twolevel
export polycorrect_mf
export quantileplot, matrixplot_preds, matrixplot_predictions, plot_error_contribs


using Zygote
using Random
using Plots
using Statistics
using Measures
using Polynomials


include("univariate_GP.jl")
include("multivariate_GP.jl")
include("multilevel_GP.jl")
include("mean_functions.jl")

include("parametric_IO.jl")
include("parametric_plots.jl")

