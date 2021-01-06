using DiffEqFlux, OrdinaryDiffEq, Flux, Optim, Plots, KernelDensity

u0 = Float32[2.0]
datasize = 30
tspan = (0.0f0, 1.5f0)
tsteps = range(tspan[1], tspan[2], length = datasize)
min_lim = 1.
max_lim = 3.
lims = Array(range(min_lim, step = 0.04, stop = max_lim))

function trueODEfunc(dx, x, p, t)
    alpha = 5
    dx[1] = alpha*x[1]-x[1]*x[1]*x[1]
end

prob_trueode = ODEProblem(trueODEfunc, u0, tspan)
ode_data = Array(solve(prob_trueode, Tsit5(), saveat = tsteps))
kde_data = [count(x->lims[ind_lim]<=x<lims[ind_lim+1], ode_data[1, :]) for ind_lim in 1:length(lims)-1]
plot(kde_data, Array(range(1, step = 1, stop = length(kde_data))))
dudt2 = FastChain(FastDense(1, 50, tanh),
                  FastDense(50, 50, tanh),
                  FastDense(50, 1))
prob_neuralode = NeuralODE(dudt2, tspan, Tsit5(), saveat = tsteps)

function predict_neuralode(p)
  Array(prob_neuralode(u0, p))
end

predict_neuralode(prob_neuralode.p)

function loss_neuralode(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, kde_data .- [count(x->lims[ind_lim]<=x<lims[ind_lim+1], pred[1,:]) for ind_lim in 1:length(lims)-1])
    return loss, pred
end

callback = function (p, l, pred; doplot = true)
    display(l)
    plt = scatter(tsteps, ode_data[1,:], ylim = (min_lim, max_lim), label = "data")
    scatter!(plt, tsteps, pred[1,:], label = "prediction")
    if doplot
        plt2 = plot(kde_data, Array(range(1, step = 1, stop = length(kde_data))), label = "data")
        plot!([count(x->lims[ind_lim]<=x<lims[ind_lim+1], pred[1,:]) for ind_lim in 1:length(lims)-1], Array(range(1, step = 1, stop = length(kde_data))), label = "prediction")
        display(plot(plt, plt2))
    end
    return false
end

result_neuralode = DiffEqFlux.sciml_train(loss_neuralode, prob_neuralode.p,
                                          ADAM(0.05), cb = callback,
                                          maxiters = 30)

result_neuralode2 = DiffEqFlux.sciml_train(loss_neuralode,
                                           result_neuralode.minimizer,
                                           LBFGS(),
                                           cb = callback,
                                           allow_f_increases = false)
