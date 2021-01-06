using DiffEqFlux, StochasticDiffEq, Flux, Optim, Plots, DiffEqBase.EnsembleAnalysis, Statistics

u0 = Float32[2.0]
datasize = 30
tspan = (0.0f0, 1.5f0)
tsteps = range(tspan[1], tspan[2], length = datasize)

function trueSDEfunc(dx, x, p, t)
    alpha=5
    dx[1] =alpha*x[1]-x[1]*x[1]*x[1]
end

mp = Float32[0.2]
function true_noise_func(dx, x, p, t)
    dx .= mp.*x
end
prob_truesde = SDEProblem(trueSDEfunc, true_noise_func, u0, tspan)
ensemble_prob = EnsembleProblem(prob_truesde)
ensemble_sol = solve(ensemble_prob, SOSRI(), trajectories = 10000)
ensemble_sum = EnsembleSummary(ensemble_sol)

sde_data, sde_data_vars = Array.(timeseries_point_meanvar(ensemble_sol, tsteps))

drift_dudt = FastChain(FastDense(1, 50, tanh),
                       FastDense(50, 50, tanh),
                       FastDense(50, 1))
diffusion_dudt = FastChain(FastDense(1, 1))

neuralsde = NeuralDSDE(drift_dudt, diffusion_dudt, tspan, SOSRI(),
                       saveat = tsteps, reltol = 1e-1, abstol = 1e-1)

function predict_neuralsde(p)
  Array(neuralsde(u0, p))
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

iter = 0

callback = function (p, loss, means, vars; doplot = true)
  global list_plots, iter

  if iter == 0
    list_plots = []
  end
  iter += 1

  # loss against current data
  display(loss)

  # plot current prediction against data
  plt = scatter(tsteps, sde_data[1,:], yerror = sde_data_vars[1,:],
                ylim = (1.0, 3.0),
                label = "data")
  scatter!(plt, tsteps, means[1,:], ribbon = vars[1,:], label = "prediction")
  push!(list_plots, plt)

  if doplot
    display(plt)
  end
  return false
end


opt = ADAM(0.025)

result = DiffEqFlux.sciml_train((p) -> loss_neuralsde(p, n = 10),
                                 neuralsde.p, opt,
                                 cb = callback, maxiters = 200)
