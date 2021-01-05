using Distances, KernelDensity, DifferentialEquations, DiffEqFlux, Plots, Flux

datasize = 35
alpha, tspan, solver = 5.0, (0, 20.0), Tsit5()
t = range(tspan[1], tspan[2], length = datasize)

function run_pfsuper_one_u0(u0)
    x0 = [u0]
    function pfsuper(dx, x, p, t)
        dx[1] =alpha*x[1]-x[1]*x[1]*x[1]
    end
    prob = ODEProblem(pfsuper, x0 ,tspan)
    obs = Array(solve(prob, solver,saveat=t))
    min_val = min(obs...)
    max_val = max(obs...)
    div = 5
    # pdf_100 = pdf(kde(obs[1, :], bandwidth = 0.05), range(min_val - div, stop = max_val + div, length = 100))
    kdx_1_N200 = kde(obs[1, :], bandwidth = 0.05)
    potential = -log.(pdf(kdx_1_N200,-10.0:0.01:10.0).+1)
    return potential
end

function run_pfsuper_multi_u0(u0s)
    obs =[]
    for i in u0s
        push!(obs,run_pfsuper_one_u0(i))
    end
    obs
end

train_u0s = [-3., -1., 0., 1.0, 3.0]
ode_data = run_pfsuper_multi_u0(train_u0s)
plot(ode_data[1], range(1, step = 1, stop = 2001), grid = "off", ylim = (-250, 2250), legend = :topleft)
plot!(ode_data[2], range(1, step = 1, stop = 2001))
plot!(ode_data[3], range(1, step = 1, stop = 2001))
plot!(ode_data[4], range(1, step = 1, stop = 2001))
plot!(ode_data[5], range(1, step = 1, stop = 2001))

dudt = Chain(Dense(1,15,tanh),
       Dense(15,15,tanh),
       Dense(15,1))
ps = Flux.params(dudt)
n_ode = NeuralODE(dudt, tspan, Tsit5(), saveat = t)
n_epochs = 200
data1 = Iterators.repeated((), n_epochs)
#opt1 = ADAM(0.0001)
opt1 = Descent(0.005)
#L2_loss_fct() = sum(abs2,ode_data .- n_ode(u0))+sum(abs2,ode_data2 .- n_ode(u02))

function loss_fct()
    sum = 0.0
    counter = 0
    for i in train_u0s
        counter = counter+1
        d_1 = ode_data[counter[1]]
        temp_pred = n_ode([i])
        d_2 = reshape(hcat(temp_pred.u...), length(temp_pred))
        # min_val = min(d_2...)
        # max_val = max(d_2...)
        div = 5
        # pdf_100 = pdf(kde(d_2, bandwidth = 0.05), range(min_val - div, stop = max_val + div, length = 100))
        kdx_1_N200 = kde(d_2, bandwidth = 0.05)
        println("len: ", length(kdx_1_N200.x))
        println("x min: ", min(kdx_1_N200.x...))
        println("x max: ", max(kdx_1_N200.density...))
        println("p min: ", min(kdx_1_N200.density...))
        println("p max: ", max(kdx_1_N200.density...))
        potential = -log.(pdf(kdx_1_N200,-10.0:0.01:10.0).+1)
        s = euclidean(d_1, potential)
        sum = sum + s
    end
    return sum
end
loss_fct()
cb1 = function ()
    println("Loss: ", loss_fct(), ("\n"))
end

# train n_ode with collocation method
@time Flux.train!(loss_fct, ps, data1, opt1, cb = cb1)

test_u0s = [-3.,-2.5,-2.,-1.5,-1.,-0.5,0.,0.5,1.,1.5,2.,2.5,-3.]
preds = []
for i in test_u0s
    pred = n_ode([i])
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
    d = dudt([i])
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
    hline!([-sqrt(alpha), 0, sqrt(alpha)], label = "",color = "red")
savefig("alpha_ks_5.pdf")
@save "pitchfork_bifur_alpha_ks_5.bson" dudt





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
