using Plots, Statistics
using Flux, DiffEqFlux, StochasticDiffEq, DiffEqBase.EnsembleAnalysis, BSON

u0 = Float32[2.; 0.]
datasize = 30
tspan = (0.0f0, 1.0f0)
tsteps = range(tspan[1], tspan[2], length = datasize)

function trueSDEfunc(du, u, p, t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u.^3)'true_A)'
end

mp = Float32[0.2, 0.2]
function true_noise_func(du, u, p, t)
    du .= mp.*u
end

prob_truesde = SDEProblem(trueSDEfunc, true_noise_func, u0, tspan)

# Take a typical sample from the mean
ensemble_prob = EnsembleProblem(prob_truesde)
ensemble_sol = solve(ensemble_prob, SOSRI(), trajectories = 10000)
ensemble_sum = EnsembleSummary(ensemble_sol)

sde_data, sde_data_vars = Array.(timeseries_point_meanvar(ensemble_sol, tsteps))

drift_dudt = FastChain((x, p) -> x.^3,
                       FastDense(2, 50, tanh),
                       FastDense(50, 2))
diffusion_dudt = FastChain(FastDense(2, 2))

neuralsde = NeuralDSDE(drift_dudt, diffusion_dudt, tspan, SOSRI(),
                       saveat = tsteps, reltol = 1e-1, abstol = 1e-1)


 # Get the prediction using the correct initial condition
prediction0 = neuralsde(u0)

drift_(u, p, t) = drift_dudt(u, p[1:neuralsde.len])
diffusion_(u, p, t) = diffusion_dudt(u, p[(neuralsde.len+1):end])

prob_neuralsde = SDEProblem(drift_, diffusion_, u0,(0.0f0, 1.2f0), neuralsde.p)

ensemble_nprob = EnsembleProblem(prob_neuralsde)
ensemble_nsol = solve(ensemble_nprob, SOSRI(), trajectories = 100,
                      saveat = tsteps)
ensemble_nsum = EnsembleSummary(ensemble_nsol)


# scatter(tsteps, sde_data[1,:], label = "data")
# scatter!(tsteps, prediction0[1,:], label = "prediction")

function predict_neuralsde(p)
  return Array(neuralsde(u0, p))
end

function loss_neuralsde(p; n = 100)
  samples = [predict_neuralsde(p) for i in 1:n]
  means = reshape(mean.([[samples[i][j] for i in 1:length(samples)]
                                        for j in 1:length(samples[1])]),
                      size(samples[1])...)
  vars = reshape(var.([[samples[i][j] for i in 1:length(samples)]
                                      for j in 1:length(samples[1])]),
                      size(samples[1])...)
  loss = sum(abs2, sde_data - means) + sum(abs2, sde_data_vars - vars)
  return loss, means, vars
end


list_plots = []
iter = 0

# Callback function to observe training
callback = function (p, loss, means, vars; doplot = false)
  global list_plots, iter

  if iter == 0
    list_plots = []
  end
  iter += 1

  # loss against current data
  display(loss)

  # plot current prediction against data
  plt = scatter(tsteps, sde_data[1,:], yerror = sde_data_vars[1,:],
                ylim = (-4.0, 8.0), label = "data")
  scatter!(plt, tsteps, means[1,:], ribbon = vars[1,:], label = "prediction")
  push!(list_plots, plt)

  if doplot
    display(plt)
  end
  return false
end


opt = ADAM(0.025)

result = DiffEqFlux.sciml_train((p) -> loss_neuralsde(p, n = 2),
                                 neuralsde.p, opt,
                                 cb = callback, maxiters = 10)
BSON.@save "models/test_diffeqflux_neuralsde_example/drift_dudt1.bson" drift_dudt
BSON.@save "models/test_diffeqflux_neuralsde_example/diffusion_dudt1.bson" diffusion_dudt

# samples1 = [predict_neuralsde(result1.minimizer) for i in 1:1000]
# means1 = reshape(mean.([[samples1[i][j] for i in 1:length(samples1)]
#                                       for j in 1:length(samples1[1])]),
#                     size(samples1[1])...)
# vars1 = reshape(var.([[samples1[i][j] for i in 1:length(samples1)]
#                                     for j in 1:length(samples1[1])]),
#                     size(samples1[1])...)
# plt1 = plot(ensemble_nsum, title = "Neural SDE: Before Training")
# scatter!(plt1, tsteps, sde_data', lw = 3)
# plt2 = scatter(tsteps, sde_data', yerror = sde_data_vars',
#                label = "", title = "Neural SDE: After Training",
#                xlabel = "Time")
# plot!(plt2, tsteps, means1', lw = 8, ribbon = vars1', label = "")
# plt = plot(plt1, plt2, layout = (2, 1))
# savefig(plt, "figures/test_diffeqflux_neuralsde_example/NN_sde_combined1.pdf")



# this takes long, careful
# result2 = DiffEqFlux.sciml_train((p) -> loss_neuralsde(p, n = 100),
#                                   result1.minimizer, opt,
#                                   cb = callback, maxiters = 100)
# BSON.@save "models/test_diffeqflux_neuralsde_example/drift_dudt2.bson" drift_dudt
# BSON.@save "models/test_diffeqflux_neuralsde_example/diffusion_dudt2.bson" diffusion_dudt
# samples2 = [predict_neuralsde(result2.minimizer) for i in 1:1000]
# means2 = reshape(mean.([[samples2[i][j] for i in 1:length(samples2)]
#                                       for j in 1:length(samples2[1])]),
#                     size(samples2[1])...)
# vars2 = reshape(var.([[samples2[i][j] for i in 1:length(samples2)]
#                                     for j in 1:length(samples2[1])]),
#                     size(samples2[1])...)
# plt3 = scatter(tsteps, sde_data', yerror = sde_data_vars',
#                label = "", title = "Neural SDE: After Training",
#                xlabel = "Time")
# plot!(plt3, tsteps, means2', lw = 8, ribbon = vars2', label = "")
# plt = plot(plt1, plt3, layout = (2, 1))
# savefig(plt, "figures/test_diffeqflux_neuralsde_example/NN_sde_combined2.pdf")
