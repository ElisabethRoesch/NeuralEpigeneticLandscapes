# This is a modified copy of ODE_multi_IC_timecourse.jl from 1_state_system folder. Here added: multi alpha
using DifferentialEquations, Plots, DiffEqFlux, Flux
using BSON: @load , @save
#########################
# Training data:
# Case 1: Training data is ODE solution without any technical noise.
#########################
u0s = Array(range(-3, stop = 3, length = 5))
alphas = Array(range(-3, stop = 3, length = 5))
datasize = 30
tspan = (0.0f0, 1.5f0)
tsteps = range(tspan[1], tspan[2], length = datasize)

function run_pfsuper_one_u0(u0, alpha)
    x0 = [u0]
    p = [alpha]
    function pfsuper(dx, x, p, t)
        dx[1] =p[1]*x[1]-x[1]*x[1]*x[1]
    end
    prob = ODEProblem(pfsuper, x0 ,tspan, p)
    obs = Array(solve(prob, Tsit5(), saveat = tsteps))
    return obs
end
function run_pfsuper_multi_u0(u0s, alpha)
    multi_u0s_obs = []
    for u0 in u0s
        push!(multi_u0s_obs, run_pfsuper_one_u0(u0, alpha))
    end
    return multi_u0s_obs
end
function run_pfsuper_multi_u0_multi_alpha(u0s, alphas)
    multi_alphas_obs = []
    for alpha in alphas
        push!(multi_alphas_obs, run_pfsuper_multi_u0(u0s, alpha))
    end
    return multi_alphas_obs
end
ode_data = run_pfsuper_multi_u0_multi_alpha(u0s, alphas)
plts = []
col_train = "#a8224a"
for idx_alpha in 1:length(alphas)
    plt = plot()
    for idx_u0 in 1:length(u0s)
        scatter!(tsteps, ode_data[idx_alpha][idx_u0][1,:], label = "", grid = "off", color = col_train)
    end
    push!(plts, plt)
end
plot(plts..., layout = (1, length(alphas)), size = (1000, 300))
savefig("results/results_1_det_case/plots/train.pdf")
#########################
# Defining and training deterministic models: One neural ODE per alpha.
#########################
col_predict = "#22a880"
models = []
function def_and_train_model(ode_data_one_alpha, ind_alpha)
    dudt = FastChain(FastDense(1, 50, tanh),
                      FastDense(50, 50, tanh),
                      FastDense(50, 1))
    prob_neuralode = NeuralODE(dudt, tspan, Tsit5(), saveat = tsteps)
    function loss_neuralode(p)
        loss = 0.
        counter = 0
        for u0 in u0s
            counter = counter + 1
            s = sum(abs2, ode_data_one_alpha[counter] .- Array(prob_neuralode([u0], p)))
            loss = loss + s
        end
        return loss, [Array(prob_neuralode([u0s[i]], p)) for i in 1:length(u0s)]
    end

    callback = function (p, l, pred; doplot = false)
      display(l)
      # plot current prediction against data
      plt = scatter(tsteps, ode_data_one_alpha[1][1,:], label = "data 1", color = col_train)
      scatter!(tsteps, ode_data_one_alpha[2][1,:], label = "data 2", color = col_train)
      scatter!(tsteps, ode_data_one_alpha[3][1,:], label = "data 3", color = col_train)
      scatter!(tsteps, ode_data_one_alpha[4][1,:], label = "data 4", color = col_train)
      scatter!(tsteps, ode_data_one_alpha[5][1,:], label = "data 5", color = col_train)
      scatter!(plt, tsteps, pred[1][1,:], label = "prediction 1", color = col_predict)
      scatter!(plt, tsteps, pred[2][1,:], label = "prediction 2", color = col_predict)
      scatter!(plt, tsteps, pred[3][1,:], label = "prediction 3", color = col_predict)
      scatter!(plt, tsteps, pred[4][1,:], label = "prediction 4", color = col_predict)
      scatter!(plt, tsteps, pred[5][1,:], label = "prediction 5", color = col_predict)
      if doplot
        display(plot(plt))
      end
      return false
    end
    result_neuralode = DiffEqFlux.sciml_train(loss_neuralode, prob_neuralode.p,
                                              ADAM(0.05), cb = callback,
                                              maxiters = 100)
    @save string("results/results_1_det_case/models/dudt_",ind_alpha,".bson") dudt
    min_res = result_neuralode.minimizer
    @save string("results/results_1_det_case/models/minimizer_",ind_alpha,".bson") min_res
    return result_neuralode
end
function def_and_train_models(ode_data)
    models = []
    for ind_alpha in 1:length(ode_data)
        push!(models, def_and_train_model(ode_data[ind_alpha], ind_alpha))
    end
    return models
end
models = def_and_train_models(ode_data)
#########################
# Testing deterministic models
#########################
function test_model(model, alpha)
    test_u0s = Array(range(-3, stop = 3, length = 12))
    ode_data_tests = run_pfsuper_multi_u0(test_u0s, alpha)
    preds_tests = []
    for u0 in test_u0s
        dudt = FastChain(FastDense(1, 50, tanh),
                          FastDense(50, 50, tanh),
                          FastDense(50, 1))
        prob_neuralode = NeuralODE(dudt, tspan, Tsit5(), saveat = tsteps)
        preds_test = prob_neuralode([u0], model.minimizer)
        push!(preds_tests, preds_test)
    end
    plt = plot(grid = "off")
    for i in 1:length(test_u0s)
        scatter!(tsteps, ode_data_tests[i][1,:], color = col_train, label = "")
        scatter!(tsteps, preds_tests[i][1,:], color = col_predict, label = "")
    end
    return plt
end

function test_models(models, alphas)
    plts = []
    for idx_alphas in 1:length(alphas)
        plt = test_model(models[idx_alphas], alphas[idx_alphas])
        push!(plts, plt)
    end
    return plts
end

test_pts = test_models(models, alphas)
plot(test_pts..., layout = (1, length(alphas)), size = (1000, 300))
savefig("results/results_1_det_case/plots/test.pdf")
