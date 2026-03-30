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


function PredictionBuffer(k::Union{AnalyticKernel{T}, UnaryKernel{T}},
                          ntr::Int, nte::Int, ndim::Int) where T <: Real

    StandardPredictionBuffer(zeros(T, (nte, ntr)), zeros(T, (nte, ntr)),
                             zeros(T, (ntr, ndim)), zeros(T, (nte, ndim)),
                             zeros(T, ntr), zeros(T, nte), nte)
end


function PredictionBuffer(k::Kernel, ntr::Int, nte::Int, ndim::Int)
    FallbackPredictionBuffer(zeros(T, nte, ntr))
end


function predict(MVM::MVGPModel{T}, X::AbstractMatrix{T};
                 reduce_inputs::Bool = true,
                 apply_λ::Bool = true,
                 recover_outputs::Bool = true,
                 apply_zyinvtransf::Bool = true,
                 Mlist::AbstractVector{Int} = 1:length(MVM.Ms)) where T <: Real

    G = MVM.G # shorthand
    (nte, nXdims) = size(X)
    nXdims = size(G.Xprojs[1].vectors)[1] # needed for Xtransf_deg > 0
    nzycols = length(MVM.Ms)

    ZY_pred = zeros(T, (nte, nzycols))

    nt = Threads.nthreads(:default)
    ntr = length(MVM.Ms[1].h)

    # Predictive performance varies a lot according to maxalloc both
    # from processor to another and from application to another.
    # 2^26 seems to work fastest for 9900X, but 2^25 is safer for
    # systems with less cache
    maxalloc = 2^25

    chunksize = maxalloc ÷ 2 ÷ sizeof(T) ÷ ntr ÷ nt
    chunksize = min(chunksize, nte)

    nchunks = nte ÷ chunksize + 1
    kernel = MVM.Ms[1].kernel

    (chunksize < nte) && (println("chunk size for prediction: $chunksize"))

    ck = collect(0:chunksize:nchunks*chunksize)
    ck[end] = nte
    batches = [c1+1:c2 for (c1,c2) in zip(ck[1:end-1], ck[2:end])]

    # batches = ranges(1, nte, nchunks) # indexes for each batch predictions

    # ranges() gives batch sizes which are not exactly of size chunksize
    chunksize = maximum(length.(batches))

    tasks = collect(Iterators.product(Mlist, batches))[:]

    nzxcols = length(G.Xprojs[1].values)
    bufs = [PredictionBuffer(kernel, ntr, chunksize, nzxcols) for _ in 1:nt]

    if reduce_inputs
        Zbufs = [zeros(T, (chunksize, nzxcols)) for _ in 1:nt]
        Xbufs = [zeros(T, (chunksize, nXdims)) for _ in 1:nt]
        Hbufs = [zeros(T, (nXdims, nzxcols)) for _ in 1:nt]
    end

    tasks_done = 0
    ntasks = length(tasks)

    Threads.@threads for (i,batch_I) in tasks
        tid = Threads.threadid() % nt + 1
        bs = length(batch_I) # batch size

        if bs == bufs[tid].nte
            @views X_unreduced = G.Xtransfspec.deg == 0 ? X[batch_I,:] :
                standard_transformations(X[batch_I,:], G.Xtransfspec)

            Z = @views reduce!(X_unreduced, G.Xprojs[i], G.μX, G.σX,
                               Xbufs[tid], Hbufs[tid], Zbufs[tid]);
            pb = bufs[tid]
        else
            Z = reduce_inputs ? (@views reduce_X(X[batch_I, :], G, i)) : (@views X[batch_I, :])
            pb = PredictionBuffer(kernel, ntr, bs, nzxcols)
        end

        # Do the prediction in-place directly to outbuf
        @views predict(MVM.Ms[i], Z, pb; apply_λ, apply_zyinvtransf,
                       outbuf = ZY_pred[batch_I,i])

        tasks_done += 1
        if tid == 1
            tdpct = round((100. * tasks_done / ntasks); sigdigits = 3)
            print("$(tdpct)% of prediction chunks done\r")
        end

    end

    return recover_outputs ? recover_Y(ZY_pred, G) : ZY_pred
end


"""Remove points outside the training data (along any input
axis). This is useful if e.g. in a random testing data batch there are
data that end up outside the training data domain."""
function remove_extrapolations(MVM::MVGPModel{T}, X::Matrix{T}) where T <: Real

    m = X[:,1] .> Inf # don't remove anything yet
    nte = length(m)
    for (i,M) in enumerate(MVM.Ms)

        ZX = reduce_X(X, MVM, i)
        for j in 1:MVM.G.Xprojs[i].spec.nCCA
            a,b = extrema(MVM.Ms[i].Z[:,j])
            z = @views ZX[:,j]
            m .= m .|| (z .< a) .|| (z .> b)
        end
    end

    s_te = setdiff(1:nte, collect(1:nte)[m])
    X[s_te,:], s_te
end
