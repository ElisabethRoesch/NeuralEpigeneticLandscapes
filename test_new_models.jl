using Plots, Statistics
using Flux, DiffEqFlux, StochasticDiffEq, DiffEqBase.EnsembleAnalysis, BSON
using BSON: @load


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


@load "models/test_diffeqflux_neuralsde_example/diffusion_dudt2.bson" diffusion_dudt
@load "models/test_diffeqflux_neuralsde_example/drift_dudt2.bson" drift_dudt


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

plt1 = plot(ensemble_nsum, title = "Neural SDE: After Training")
scatter!(plt1, tsteps, sde_data', lw = 3)
