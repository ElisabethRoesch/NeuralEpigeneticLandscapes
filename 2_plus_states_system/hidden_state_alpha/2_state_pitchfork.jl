
using DifferentialEquations, Plots, KernelDensity

function postprocessing(ensemble_sol, number_trajects, datasize)
    end_indx = rand(1:datasize, number_trajects)
    end_points = [ensemble_sol.u[i][:,end_indx[i]] for i in 1:number_trajects]
    return end_points
end

function get_sols(number_trajects)
    u0 = Float32[0., -5.]
    datasize = 30
    tspan = (0.0, 30.0)
    tsteps = range(tspan[1], tspan[2], length = datasize)
    alpha  = 0.5
    function trueSDEfunc(du, u, p, t)
        du[1] = u[2]*u[1]-u[1]^3
        du[2] = alpha
        return du
    end
    mp = Float32[0.1, 0.1]
    function true_noise_func(du, u, p, t)
        du .= mp#*.u this is for multiplicative noise case.
    end
    prob_truesde = SDEProblem(trueSDEfunc, true_noise_func, u0, tspan)
    ensemble_prob = EnsembleProblem(prob_truesde)
    ensemble_sol = solve(ensemble_prob, SOSRI(), saveat = tsteps, trajectories = number_trajects)
    ensemble_sum = EnsembleSummary(ensemble_sol)
    pp = postprocessing(ensemble_sol, number_trajects, datasize)
    return ensemble_sol, ensemble_sum, pp
end


a, b, c = get_sols(10000)
cc = hcat(c...)'
ccc = kde(cc).density
contour(ccc)

savefig("2d_kde_addi_noise.pdf")
