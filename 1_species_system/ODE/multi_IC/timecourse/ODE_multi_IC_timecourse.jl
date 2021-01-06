using DiffEqFlux, OrdinaryDiffEq, Flux, Optim, Plots

u0s = [[-3.],[ -1.], [0.], [1.0], [3.0]]
alpha_bifur = 5.
datasize = 30
tspan = (0.0f0, 1.5f0)
tsteps = range(tspan[1], tspan[2], length = datasize)

function run_pfsuper_one_u0(u0)
    x0 = u0
    function pfsuper(dx, x, p, t)

        dx[1] =alpha_bifur*x[1]-x[1]*x[1]*x[1]
    end
    prob = ODEProblem(pfsuper, x0 ,tspan)
    obs = Array(solve(prob, Tsit5(), saveat = tsteps))
    return obs
end

function run_pfsuper_multi_u0(u0s)
    obs =[]
    for i in u0s
        push!(obs,run_pfsuper_one_u0(i))
    end
    obs
end

ode_data = run_pfsuper_multi_u0(u0s)

dudt2 = FastChain(FastDense(1, 50, tanh),
                  FastDense(50, 50, tanh),
                  FastDense(50, 1))
prob_neuralode = NeuralODE(dudt2, tspan, Tsit5(), saveat = tsteps)


function loss_neuralode(p)
    loss = 0.
    counter = 0
    for u0 in u0s
        counter = counter + 1
        s = sum(abs2, ode_data[counter] .- Array(prob_neuralode(u0, p)))
        loss = loss + s
    end
    return loss, [Array(prob_neuralode(u0s[1], p)),
                Array(prob_neuralode(u0s[2], p)),
                Array(prob_neuralode(u0s[3], p)),
                Array(prob_neuralode(u0s[4], p)),
                Array(prob_neuralode(u0s[5], p))]
end

callback = function (p, l, pred; doplot = true)
  display(l)
  # plot current prediction against data
  plt = scatter(tsteps, ode_data[1][1,:], label = "data 1", color = "blue")
  scatter!(tsteps, ode_data[2][1,:], label = "data 2", color = "blue")
  scatter!(tsteps, ode_data[3][1,:], label = "data 3", color = "blue")
  scatter!(tsteps, ode_data[4][1,:], label = "data 4", color = "blue")
  scatter!(tsteps, ode_data[5][1,:], label = "data 5", color = "blue")
  scatter!(plt, tsteps, pred[1][1,:], label = "prediction 1", color = "red")
  scatter!(plt, tsteps, pred[2][1,:], label = "prediction 2", color = "red")
  scatter!(plt, tsteps, pred[3][1,:], label = "prediction 3", color = "red")
  scatter!(plt, tsteps, pred[4][1,:], label = "prediction 4", color = "red")
  scatter!(plt, tsteps, pred[5][1,:], label = "prediction 5", color = "red")


  if doplot
    display(plot(plt))
  end
  return false
end



result_neuralode = DiffEqFlux.sciml_train(loss_neuralode, prob_neuralode.p,
                                          ADAM(0.05), cb = callback,
                                          maxiters = 500)

result_neuralode2 = DiffEqFlux.sciml_train(loss_neuralode,
                                           result_neuralode.minimizer,
                                           LBFGS(),
                                           cb = callback,
                                           allow_f_increases = false)






# not sure whats happening.
test_u0s = [[-3.],[-2.5],[-2.],[-1.5],[-1.],[-0.5],[0.],[0.5],[1.],[1.5],[2.],[2.5],[-3.]]
ode_data_tests = run_pfsuper_multi_u0(test_u0s)
preds_tests = []
for u0 in test_u0s
    preds_test = prob_neuralode(u0, result_neuralode2.minimizer)
    push!(preds_tests, preds_test)
end
function plot_test()
    plt = plot(grid = "off")
    for i in 1:length(test_u0s)
        scatter!(tsteps, ode_data_tests[i][1,:], label = "data $i", color = "blue")
        scatter!(tsteps, preds_tests[i][1,:], label = "prediction $i", color = "red")
    end
    display(plot(plt))
end
plot_test()
