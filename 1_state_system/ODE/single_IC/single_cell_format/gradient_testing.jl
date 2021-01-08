# Opjective: finding diffable loss function
using Zygote, KernelDensity

# dummy input
pred = [1,2,3]
obs = pred

# throws matable error
function tester_kde(pred)
    kde_a = kde(pred).density
    kde_b = kde(obs).density
    loss = sum(abs2, kde_a.-kde_b)
    return loss
end
Zygote.gradient(tester_kde, pred)

# throws matable error
function tester_save_in_array(pred)
    lims = [1,2,3,4]
    a = []
    for ind_lim in 1:length(lims)-1
        counter = 0
        for x in pred
            if lims[ind_lim]<=x<lims[ind_lim+1]
                counter = counter + 1
            end
        end
        push!(a, counter)
    end
    loss = sum(abs2, a.-[1,3,3.])
    return loss
end
Zygote.gradient(tester_save_in_array, pred)

# empty gradient
function tester_for_directly_in_array(pred)
    lims = [1,2,3,4]
    a =[count(x->lims[ind_lim]<=x<lims[ind_lim+1], pred) for ind_lim in 1:length(lims)-1]
    loss = sum(abs2, a.-[1,3,3.])
    return loss
end
Zygote.gradient(tester_for_directly_in_array, pred)


# empty gradient
function tester_dict(pred)
           lims = [1,2,3,4]
           a = Dict()
           for ind_lim in 1:length(lims)-1
               counter = 0
               for x_ind in 1:length(pred)
                   if lims[ind_lim]<=pred[x_ind]<lims[ind_lim+1]
                       counter = counter + 1
                   end
               end
               a[ind_lim]=counter
           end
           loss = sum(abs2, [a[ind_lim] for ind_lim in 1:length(lims)-1].-[1,3,3.])
           return loss
end
Zygote.gradient(tester_dict, pred)
