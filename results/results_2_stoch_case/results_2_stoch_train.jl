using DiffEqFlux, StochasticDiffEq, Flux, Optim, Plots, DiffEqBase.EnsembleAnalysis, Statistics
println("Pkgs loaded.")
u0s = Array(range(-3, stop = 3, length = 4))
datasize = 15
tspan = (0.0f0, 1.5f0)
tsteps = range(tspan[1], tspan[2], length = datasize)

function run_super_pf_sde_one(u0)
    function trueSDEfunc(dx, x, p, t)
        alpha=5
        dx[1] =alpha*x[1]-x[1]*x[1]*x[1]
    end
    mp = Float32[0.05]
    function true_noise_func(dx, x, p, t)
        dx .= mp.*x
    end
    prob_truesde = SDEProblem(trueSDEfunc, true_noise_func, u0, tspan)
    ensemble_prob = EnsembleProblem(prob_truesde)
    ensemble_sol = solve(ensemble_prob, SOSRI(), trajectories = 10)
    ensemble_sum = EnsembleSummary(ensemble_sol)
    sde_data, sde_data_vars = Array.(timeseries_point_meanvar(ensemble_sol, tsteps))
    return sde_data, sde_data_vars
end

function run_super_pf_sde_multi(u0s)
    sde_datae = []
    sde_datae_vars = []
    for u0 in u0s
        temp_sde_data, temp_sde_vars = run_super_pf_sde_one([u0])
        push!(sde_datae, temp_sde_data)
        push!(sde_datae_vars, temp_sde_vars)
    end
    return sde_datae, sde_datae_vars
end

sde_datae, sde_datae_vars = run_super_pf_sde_multi(u0s)

drift_dudt = FastChain(FastDense(1, 50, tanh),
                       FastDense(50, 50, tanh),
                       FastDense(50, 1))
diffusion_dudt = FastChain(FastDense(1, 1))

neuralsde = NeuralDSDE(drift_dudt, diffusion_dudt, tspan, SOSRI(),
                       saveat = tsteps, reltol = 1e-1, abstol = 1e-1)

function predict_neuralsde(p)
  Array(neuralsde(u0, p))
end

# loss function. Caution, the return is only for first training pair (where x = u0[1])
function loss_neuralsde(p; n = 100)

    counter = 1
    samples_first = [Array(neuralsde([u0s[counter]], p)) for i in 1:n]
    means_first = reshape(mean.([[samples_first[i][j] for i in 1:length(samples_first)] for j in 1:length(samples_first[1])]), size(samples_first[1])...)
    vars_first = reshape(var.([[samples_first[i][j] for i in 1:length(samples_first)] for j in 1:length(samples_first[1])]), size(samples_first[1])...)
    loss =  sum(abs2, sde_datae[counter] - means_first) + sum(abs2, sde_datae_vars[counter] - vars_first)
    for i in 2:length(u0s)
        counter = counter + 1
        temp_u0 = u0s[counter]
        samples = [Array(neuralsde([temp_u0], p)) for i in 1:n]
        means = reshape(mean.([[samples[i][j] for i in 1:length(samples)]
                                                for j in 1:length(samples[1])]),
                              size(samples[1])...)
        vars = reshape(var.([[samples[i][j] for i in 1:length(samples)]
                                              for j in 1:length(samples[1])]),
                              size(samples[1])...)
        s = sum(abs2, sde_datae[counter] - means) #+ sum(abs2, sde_datae_vars[counter] - vars)
        loss = loss + s
    end
    return loss,  means_first, vars_first
end

iter = 0

callback = function (p, loss, means, vars; doplot = false)
    global list_plots, iter

    if iter == 0
        list_plots = []
    end
    iter += 1
    display(loss)
    plt = scatter(tsteps, sde_datae[1][1,:], yerror = sde_datae_vars[1][1,:],
                ylim = (-3.0, -1.0),
                label = "data")
        scatter!(plt, tsteps, means[1,:], ribbon = vars[1,:], label = "prediction")
        push!(list_plots, plt)
    if doplot
        display(plt)
    end
    return false
end


opt = ADAM(0.05)

result = DiffEqFlux.sciml_train((p) -> loss_neuralsde(p, n = 2),
                                 neuralsde.p, opt,
                                 cb = callback, maxiters = 100)

min_res = result.minimizer
using BSON
ind_alpha=1
BSON.@save string("results/results_2_stoch_case/models/minimizer_",ind_alpha,".bson") min_res

function plot_sdes(p = min_res, n = 10)
    drift_dudt_re = FastChain(FastDense(1, 50, tanh),
                           FastDense(50, 50, tanh),
                           FastDense(50, 1))
    diffusion_dudt_re = FastChain(FastDense(1, 1))

    neuralsde _re = NeuralDSDE(drift_dudt_re, diffusion_dudt_re, tspan, SOSRI(), saveat = tsteps, reltol = 1e-1, abstol = 1e-1)
    plt = plot()
    for i in 1:length(u0s)
         u0 = u0s[i]
        scatter!(tsteps, sde_datae[i][1,:], yerror = sde_datae_vars[i][1,:], label = "data", color = "blue")
        samples_temp = [Array(neuralsde([u0], min_res)) for i in 1:n]
        means_temp = reshape(mean.([[samples_temp[i][j] for i in 1:length(samples_temp)] for j in 1:length(samples_temp[1])]), size(samples_temp[1])...)
        vars_temp = reshape(var.([[samples_temp[i][j] for i in 1:length(samples_temp)] for j in 1:length(samples_temp[1])]), size(samples_temp[1])...)
        plot!(plt, tsteps, means_temp[1,:], label = "prediction", color = "red")
    end
    return plt
end


plt = plot_sdes()
