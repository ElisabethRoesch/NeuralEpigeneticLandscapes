
using ModelingToolkit, DiffEqFlux, Flux, Plots, Test, Optim, CUDA, NeuralPDE, GalacticOptim

cb = function (p,l)
    println("Current loss is: $l")
    return false
end

@parameters t, x, θ
@variables u(..)
@derivatives Dt'~t
@derivatives Dx'~x
@derivatives Dxx''~x

#2D PDE
eq  = Dt(u(t,x,θ)) + u(t,x,θ)*Dx(u(t,x,θ)) - (0.01/pi)*Dxx(u(t,x,θ)) ~ 0

# Initial and boundary conditions
bcs = [u(0,x,θ) ~ -sin(pi*x),
       u(t,-1,θ) ~ 0.,
       u(t,1,θ) ~ 0.]

# Space and time domains
domains = [t ∈ IntervalDomain(0.0,1.0),
           x ∈ IntervalDomain(-1.0,1.0)]
# Discretization
dx = 0.1

# Neural network
chain = FastChain(FastDense(2,16,Flux.σ),FastDense(16,16,Flux.σ),FastDense(16,1))

strategy = GridTraining()
discretization = PhysicsInformedNN(dx,chain,strategy=strategy)

indvars = [t,x]
depvars = [u]
dim = length(domains)

expr_pde_loss_function = build_loss_function(eq,indvars,depvars)
expr_bc_loss_functions = [build_loss_function(bc,indvars,depvars) for bc in bcs]
train_sets = generate_training_sets(domains,dx,bcs,indvars,depvars)

train_domain_set, train_bound_set, train_set= train_sets

phi = discretization.phi
autodiff = discretization.autodiff
derivative = discretization.derivative
initθ = discretization.initθ

pde_loss_function = get_loss_function(eval(expr_pde_loss_function),
                                      train_domain_set,
                                      phi,
                                      derivative,
                                      strategy)
bc_loss_function = get_loss_function(eval.(expr_bc_loss_functions),
                                     train_bound_set,
                                     phi,
                                     derivative,
                                     strategy)

function loss_function(θ,p)
    return pde_loss_function(θ) + bc_loss_function(θ)
end

f = OptimizationFunction(loss_function, initθ, GalacticOptim.AutoZygote())

prob = GalacticOptim.OptimizationProblem(f, initθ)

# optimizer
opt = Optim.BFGS()
res = GalacticOptim.solve(prob, opt; cb = cb, maxiters=2000)
