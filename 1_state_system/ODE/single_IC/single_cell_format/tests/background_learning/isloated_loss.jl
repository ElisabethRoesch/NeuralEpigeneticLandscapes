using Zygote, Plots

counter_of_observed_data = [1,2,3]

function counter_loss(prediction)
    lims = Array(range(1, step = 1, stop = 4))
    counter_of_prediction = [count(x -> lims[ind_lim] <= x < lims[ind_lim + 1], prediction) for ind_lim in 1:length(lims) - 1]
    println(" ")
    println("pred: ", prediction)
    println("counter_of_prediction: ", counter_of_prediction)
    loss = sum(abs2, counter_of_prediction .- counter_of_observed_data)
    println("loss: ", loss)
    return loss
end

temp1_prediction = [1, 2, 4]
temp2_prediction = [1, 2, 5]

gradient(counter_loss, temp1_prediction)
gradient(counter_loss, temp2_prediction)


dim = 10
vals = Array{Float32,2}(undef,dim,dim)
for i in 1:dim
    for j in 1:dim
    vals[i,j] = counter_loss([1, j, i])
    end
end

contourf(vals)
print(vals)
plt = plot(vals[1,:])
for i in 2:dim
    plot!(vals[i,:])
end
plot(plt)
