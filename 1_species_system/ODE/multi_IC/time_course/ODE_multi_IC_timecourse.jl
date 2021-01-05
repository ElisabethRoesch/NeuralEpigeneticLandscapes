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

dudt2 = FastChain((x, p) -> x.^3,
                  FastDense(1, 50, tanh),
                  FastDense(50, 1))
prob_neuralode = NeuralODE(dudt2, tspan, Tsit5(), saveat = tsteps)

function test()
    a = "slurm-22260172."
    return a, "he", ["s","s"]
end

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
                                          maxiters = 300)

result_neuralode2 = DiffEqFlux.sciml_train(loss_neuralode,
                                           result_neuralode.minimizer,
                                           LBFGS(),
                                           cb = callback,
                                           allow_f_increases = false)






# not sure whats happening. 
test_u0s = [-3.,-2.5,-2.,-1.5,-1.,-0.5,0.,0.5,1.,1.5,2.,2.5,-3.]
preds = []
for i in test_u0s
    pred = prob_neuralode([i])
    push!(preds, pred[1,:])
end
plot(Array(range(1,stop = datasize)),preds[1])
    plot!(Array(range(1,stop = datasize)),preds[2])
    plot!(Array(range(1,stop = datasize)),preds[3])
    plot!(Array(range(1,stop = datasize)),preds[4])
    plot!(Array(range(1,stop = datasize)),preds[5])
    plot!(Array(range(1,stop = datasize)),preds[6])
    plot!(Array(range(1,stop = datasize)),preds[7])
    plot!(Array(range(1,stop = datasize)),preds[8])
    plot!(Array(range(1,stop = datasize)),preds[9])
    plot!(Array(range(1,stop = datasize)),preds[10])
    plot!(Array(range(1,stop = datasize)),preds[11])
    plot!(Array(range(1,stop = datasize)),preds[12])
    plot!(Array(range(1,stop = datasize)),preds[13])

derivs = []
for i in test_u0s
    d = dudt2([i],  prob_neuralode.p)
    push!(derivs,d[1])
end
a = test_u0s.+ derivs
plot([1,2],[test_u0s[1], a[1]], label = "", color = "blue", grid =:off)
    plot!([1,2], [test_u0s[2], a[2]], label = "", color = "blue")
    plot!([1,2], [test_u0s[3], a[3]], label = "", color = "blue")
    plot!([1,2], [test_u0s[4], a[4]], label = "", color = "blue")
    plot!([1,2], [test_u0s[5], a[5]], label = "", color = "blue")
    plot!([1,2], [test_u0s[6], a[6]], label = "", color = "blue")
    plot!([1,2], [test_u0s[7], a[7]], label = "", color = "blue")
    plot!([1,2], [test_u0s[8], a[8]], label = "", color = "blue")
    plot!([1,2], [test_u0s[9], a[9]], label = "", color = "blue")
    plot!([1,2], [test_u0s[10], a[10]], label = "", color = "blue")
    plot!([1,2], [test_u0s[11], a[11]], label = "", color = "blue")
    plot!([1,2], [test_u0s[12], a[12]], label = "", color = "blue")
    plot!([1,2], [test_u0s[13], a[13]], label = "", color = "blue")
    hline!([-sqrt(alpha_bifur), 0, sqrt(alpha_bifur)], label = "",color = "red")
savefig("alpha_bifur_ks_5.pdf")
@save "pitchfork_bifur_alpha_bifur_ks_5.bson" dudt





function kolmogorov_smirnov_distance(data1, data2)
            ecdf_func_1 = StatsBase.ecdf(data1)
            ecdf_func_2 = StatsBase.ecdf(data2)
            max = maximum([data1;data2])
            intervals = max/999
            ecdf_vals_1 = Array{Float64,1}(undef, 1000)
            for i in 1:1000
                ecdf_vals_1[i] = ecdf_func_1(intervals*(i-1))
            end
            ecdf_vals_2 = Array{Float64,1}(undef, 1000)
            for i in 1:1000
                ecdf_vals_2[i] = ecdf_func_2(intervals*(i-1))
            end
            dist = maximum(abs.(ecdf_vals_1-ecdf_vals_2))
            return dist
end




kde([1,1,2,3,4,5,5.])


kde([1,1,2,3,4,5,5.], boundary = (1,5.))
