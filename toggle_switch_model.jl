using Plots, DifferentialEquations

# Gardner switch (BMC Syst Biol 2016 Leon)
# u is the concentration of repressor 1
# v the concentration of repressor 2
# α1 and α2 denote the effective rates of synthesis of repressors 1 and 2
# β and γ are the cooperativity of repression of promoter 1 and of repressor 2

# derivative as function
function dudt(states, ps, t)
    alpha1, beta, alpha2, gamma = ps
    u, v = states
    du = alpha1/(1+v^beta)-u
    dv = alpha2/(1+u^gamma)-v
    return [du, dv]
end
# test setting
u0=[3., 2.9]
ps = [100., 2, 100., 2] # from Supplementary material (BMC Syst Biol 2016 Leon)
tspan = (0.0, 100.0)

function get_prob_sol(u0temp)
    probtemp =ODEProblem(dudt, u0temp, tspan, ps)
    soltemp = Array(solve(probtemp))
    atemp = round(soltemp[1,end])
    btemp = round(soltemp[2,end])
    return soltemp, atemp, btemp
end

function get_sols(u0_s, v0_s)
    as=[]
    bs=[]
    cs=[]
    for u in range_u0
        for v in range_v0
            u0temp = [u, v]
            a,b,c = get_prob_sol(u0temp)
            push!(as,a)
            push!(bs,b)
            push!(cs,c)
        end
    end
    l1 = length(u0_s)
    l2 = length(v0_s)
    as = reshape(as, l1, l2)
    bs = reshape(bs, l1, l2)
    cs =reshape(cs, l1, l2)
    return as, bs, cs
end

function visu(range_u0, range_v0, as, bs, cs)
    function plot_helper_bs(x,y)
        return bs[x,y]
    end
    function plot_helper_cs(x,y)
        return cs[x,y]
    end
    ind_1 = Array(range(1,length(range_u0), step =1))
    ind_2 = Array(range(1,length(range_v0), step =1))
    p_temp_b = contourf(ind_1, ind_2, plot_helper_bs, xlabel = "Init U", ylabel = "Init V", title = "Fixed point U", size = (600,500))
    p_temp_c = contourf(ind_1, ind_2, plot_helper_cs, xlabel = "Init U", ylabel = "Init V", title = "Fixed point V", size = (600,500))
    return p_temp_b, p_temp_c
end



range_u0 = Array(range(0, 10, step = 1))
range_v0 = Array(range(0, 10, step = 1))
as, bs, cs = get_sols(range_u0, range_v0)
p1, p2 = visu(range_u0, range_v0, as, bs, cs)
sum_plt = plot(p1, p2, size = (1200,600))


# savefig("figures/ts_bistable_first_gamma_beta_smaller.pdf")

plot(as[5,10][1,:], xlabel = "Time", ylabel ="Species", label = "U", title = "U is \"ON\" (IC: U=9, V=4).")
    plot!(as[5,10][2,:], label ="V")
savefig("figures/ts_bistable_U_on.pdf")

plot(as[10,5][1,:], xlabel = "Time", ylabel ="Species", label = "U", title = "V is \"ON\" (IC: U=4, V=9).")
    plot!(as[10,5][2,:], label = "V")
savefig("figures/ts_bistable_V_on.pdf")



# ich glaube 1d ana macht mehr sind da immer eins an udn eins aus
