using DiffEqFlux, OrdinaryDiffEq, Flux, Optim, Plots, KernelDensity

u0 = Float32[2.0]
datasize = 30
tspan = (0.0f0, 1.5f0)
tsteps = range(tspan[1], tspan[2], length = datasize)

function trueODEfunc(dx, x, p, t)
    alpha = 5
    dx[1] = alpha*x[1]-x[1]*x[1]*x[1]
end

prob_trueode = ODEProblem(trueODEfunc, u0, tspan)
ode_data = Array(solve(prob_trueode, Tsit5(), saveat = tsteps))
kde_data = kde(ode_data[1, :], bandwidth = 0.05, boundary = (-5,5)).density
dudt2 = FastChain(FastDense(1, 50, tanh),
                  FastDense(1, 50, tanh),
                  FastDense(50, 1))
prob_neuralode = NeuralODE(dudt2, tspan, Tsit5(), saveat = tsteps)

function predict_neuralode(p)
  Array(prob_neuralode(u0, p))
end

function loss_neuralode(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, kde_data .- kde(pred[1,:], bandwidth = 0.05, boundary = (-5,5)).density)
    return loss, pred
end

callback = function (p, l, pred; doplot = true)
  display(l)
  plt = scatter(tsteps, ode_data[1,:], label = "data")
  scatter!(plt, tsteps, pred[1,:], label = "prediction")
  if doplot
    display(plot(plt))
  end
  return false
end

result_neuralode = DiffEqFlux.sciml_train(loss_neuralode, prob_neuralode.p,
                                          ADAM(0.05), cb = callback,
                                          maxiters = 300)

result_neuralode2 = DiffEqFlux.sciml_train(loss_neuralode,
                                           result_neuralode.minimizer,
                                           LBFGS(),
                                           cb = callback,
                                           allow_f_increases = false)



using KernelDensity, DifferentialEquations, DiffEqFlux, Plots, Flux

datasize = 35
alpha, tspan, solver = 5.0, (0, 20.0), Tsit5()
t = range(tspan[1], tspan[2], length = datasize)

function super_pf_single_u0(u0)
    x0 = [u0]
    function pfsuper(dx, x, p, t)
        dx[1] =alpha*x[1]-x[1]*x[1]*x[1]
    end
    prob = ODEProblem(pfsuper, x0 ,tspan)
    obs = Array(solve(prob, solver,saveat=t))
    kde_d = kde(obs[1, :], bandwidth = 0.05, boundary = (-5,5)).density
    return kde_d
end

train_u0 = -3.
ode_data = super_pf_single_u0(train_u0)
plot(ode_data, range(1, step = 1, stop = length(ode_data)), grid = "off")

dudt = FastChain(FastDense(1, 50, tanh),
                  FastDense(50, 50, tanh),
                  FastDense(50, 1))
n_ode = NeuralODE(dudt, tspan, Tsit5(), saveat = t)
opt1 = Descent(0.005)

function predict_neuralode(u0, p)
  return n_ode(u0, p)
end

function loss_fct(p)
    d_1 = ode_data
    pred = predict_neuralode([train_u0], p)
    d_2_raw = reshape(hcat(pred.u...), length(pred))
    d_2 = kde(d_2_raw, bandwidth = 0.05, boundary = (-5,5)).density
    println("ode solution kde density: ", typeof(d_1))
    println("prediction kde density: ", typeof(d_2))
    loss = sum(abs2, ode_data .- pred)
    return loss
end

cb1 = function (p, l, pred)
    println("Loss: ", l, "\n")
end

loss_fct(n_ode.p)
result = DiffEqFlux.sciml_train(loss_fct, n_ode.p, opt1, cb = cb1, maxiters = 100)
