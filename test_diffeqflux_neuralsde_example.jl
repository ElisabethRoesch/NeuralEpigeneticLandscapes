using Plots, Statistics
using Flux, DiffEqFlux
using StochasticDiffEq, DiffEqBase.EnsembleAnalysis

# dudt(u, p, t) = model(u)
# g(u, p, t) = model2(u)
# prob = SDEProblem(dudt, g, x, tspan, nothing)
#
# dudt!(u, h, p, t) = model([u; h(t - p.tau)])
# prob = DDEProblem(dudt_, u0, h, tspan, nothing)



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
### FAILS HERE #### added ", num_monte = 10"
ensemble_sol = solve(ensemble_prob, SOSRI(), trajectories = 10, num_monte = 10)
ensemble_sum = EnsembleSummary(ensemble_sol)

### FAILS HERE #### modified to [1] and [2] added ".u"
sde_data_var = timeseries_point_meanvar(ensemble_sol, tsteps)
sde_data, sde_data_vars = Array.(sde_data_var[1].u), Array.(sde_data_var[2].u)
aa=timeseries_point_meanvar(ensemble_sol, tsteps)

drift_dudt = Chain((x, p) -> x.^3,
                       Dense(2, 50, tanh),
                       Dense(50, 2))
diffusion_dudt = Chain(Dense(2, 2))

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

plt1 = plot(ensemble_nsum, title = "Neural SDE: Before Training")
scatter!(plt1, tsteps, sde_data', lw = 3)

scatter(tsteps, sde_data[1,:], label = "data")
scatter!(tsteps, prediction0[1,:], label = "prediction")

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

# First round of training with n = 10
result1 = DiffEqFlux.sciml_train((p) -> loss_neuralsde(p, n = 10),
                                 neuralsde.p, opt,
                                 cb = callback, maxiters = 100)

result2 = DiffEqFlux.sciml_train((p) -> loss_neuralsde(p, n = 100),
                                  result1.minimizer, opt,
                                  cb = callback, maxiters = 100)

ad
samples = [predict_neuralsde(result2.minimizer) for i in 1:1000]
means = reshape(mean.([[samples[i][j] for i in 1:length(samples)]
                                      for j in 1:length(samples[1])]),
                    size(samples[1])...)
vars = reshape(var.([[samples[i][j] for i in 1:length(samples)]
                                    for j in 1:length(samples[1])]),
                    size(samples[1])...)

plt2 = scatter(tsteps, sde_data', yerror = sde_data_vars',
               label = "data", title = "Neural SDE: After Training",
               xlabel = "Time")
plot!(plt2, tsteps, means', lw = 8, ribbon = vars', label = "prediction")

plt = plot(plt1, plt2, layout = (2, 1))
savefig(plt, "NN_sde_combined.png"); nothing # sde
