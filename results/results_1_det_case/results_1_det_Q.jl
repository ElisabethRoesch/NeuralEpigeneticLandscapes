# This constructs Q(X) based on a trained neuralODE (loaded from BSON files)
using DifferentialEquations, Plots, DiffEqFlux, Flux, KernelDensity
using BSON: @load , @save

#########################
# Loading gradients...
# Note: BSON load in function doesn't work.
dudts = []
alphas = Array(range(-3, stop = 3, length = 5))
for ind_alpha in 1:length(alphas)
        path = string("results/results_1_det_case/models/minimizer_",ind_alpha,".bson")
        @load path min_res
        push!(dudts, min_res)
end
#########################
# Constructing Q(X)...
#########################
u0s = Array(range(-3, stop = 3, length = 5))
datasize = 30
tspan = (0.0f0, 50f0)
tsteps = range(tspan[1], tspan[2], length = datasize)
predictions = []
qs = []
for ind_alpha in 1:length(alphas)
        prediction = Float32[]
        dudt = FastChain(FastDense(1, 50, tanh),
                  FastDense(50, 50, tanh),
                  FastDense(50, 1))
        model = NeuralODE(dudt, tspan, Tsit5(), saveat = tsteps)
        for u0 in u0s
                sol = Array(model([u0], dudts[ind_alpha]))
                append!(prediction, sol)
        end
        kdx_1_N200 = kde(prediction, bandwidth = 0.05)
        q_x_shifted = -log.(pdf(kdx_1_N200,-5.0:0.01:5.0).+1)
        push!(predictions, prediction)
        push!(qs, q_x_shifted)
end
#########################
# Visualise raw data and Q(X)...
#########################
plts_predictions = []
for ind_alpha in 1:length(alphas)
        p = scatter(predictions[ind_alpha], label = "", grid = "off")
        push!(plts_predictions, p)
end
plts_qs = []
for ind_alpha in 1:length(alphas)
        p = plot(-qs[ind_alpha], 1:length(qs[ind_alpha]), label = "", grid = "off")
        push!(plts_qs, p)
end
all_plts =[plts_predictions..., plts_qs...]
plot(all_plts..., layout = (2,5), size = (2000, 600))
savefig("results/results_1_det_case/plots/raw_q.pdf")
