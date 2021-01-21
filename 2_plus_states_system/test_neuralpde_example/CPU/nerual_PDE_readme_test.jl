# Running
using Flux
using DiffEqFlux
using ModelingToolkit
using DiffEqBase
using Test, NeuralPDE
using GalacticOptim
using Optim

# using Quadrature,Cubature, Cuba
# using QuasiMonteCarlo

using Random
Random.seed!(100)

cb = function (p,l)
    println("Current loss is: $l")
    return false
end

## Example 1, 1D ode
@parameters θ
@variables u(..)
@derivatives Dθ'~θ

# 1D ODE
eq = Dθ(u(θ)) ~ θ^3 + 2*θ + (θ^2)*((1+3*(θ^2))/(1+θ+(θ^3))) - u(θ)*(θ + ((1+3*(θ^2))/(1+θ+θ^3)))

# Initial and boundary conditions
bcs = [u(0.) ~ 1.0]

# Space and time domains
domains = [θ ∈ IntervalDomain(0.0,1.0)]
# Discretization
dt = 0.1
# Neural network
chain = FastChain(FastDense(1,12,Flux.σ),FastDense(12,1))

strategy = NeuralPDE.GridTraining(dt)

discretization = NeuralPDE.PhysicsInformedNN(chain,
                                             strategy;
                                             init_params = nothing,
                                             phi = nothing,
                                             autodiff=false,
                                             derivative = nothing,
                                             )

pde_system = PDESystem(eq,bcs,domains,[θ],[u])

prob = NeuralPDE.discretize(pde_system,discretization)

res = GalacticOptim.solve(prob, ADAM(0.1); cb = cb, maxiters=1000)
phi = discretization.phi
