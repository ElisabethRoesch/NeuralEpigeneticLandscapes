
using Flux
println("NNPDE_tests")
using DiffEqFlux
println("Starting Soon!")
using ModelingToolkit
using DiffEqBase
using Test, NeuralPDE
println("Starting Soon!")
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
"""
LoadError: MethodError: no method matching iterate(::GridTraining)
Closest candidates are:
  iterate(::DataStructures.SparseIntSet, ::Any...) at /Users/eroesch/.julia/packages/DataStructures/5hvIb/src/sparse_int_set.jl:147
  iterate(::Combinatorics.FixedPartitions) at /Users/eroesch/.julia/packages/Combinatorics/Udg6X/src/partitions.jl:112
  iterate(::Combinatorics.FixedPartitions, ::Array{Int64,1}) at /Users/eroesch/.julia/packages/Combinatorics/Udg6X/src/partitions.jl:112
  ...
Stacktrace:
 [1] discretize(::PDESystem, ::PhysicsInformedNN{FastChain{Tuple{FastDense{typeof(σ),DiffEqFlux.var"#initial_params#73"{typeof(Flux.glorot_uniform),typeof(Flux.zeros),Int64,Int64}},FastDense{typeof(identity),DiffEqFlux.var"#initial_params#73"{typeof(Flux.glorot_uniform),typeof(Flux.zeros),Int64,Int64}}}},GridTraining,NeuralPDE.var"#200#202"{FastChain{Tuple{FastDense{typeof(σ),DiffEqFlux.var"#initial_params#73"{typeof(Flux.glorot_uniform),typeof(Flux.zeros),Int64,Int64}},FastDense{typeof(identity),DiffEqFlux.var"#initial_params#73"{typeof(Flux.glorot_uniform),typeof(Flux.zeros),Int64,Int64}}}}},NeuralPDE.var"#206#207"{Float32},GridTraining,Base.Iterators.Pairs{Symbol,Any,NTuple{4,Symbol},NamedTuple{(:init_params, :phi, :autodiff, :derivative),Tuple{Nothing,Nothing,Bool,Nothing}}}}) at /Users/eroesch/.julia/packages/NeuralPDE/nfwQb/src/pinns_pde_solve.jl:639
 [2] top-level scope at /Users/eroesch/github/NeuralEpigeneticLandscapes/2_plus_states_system/test_neuralpde_example/nerual_PDE_readme_test.jl:54
 [3] include_string(::Function, ::Module, ::String, ::String) at ./loading.jl:1088
in expression starting at /Users/eroesch/github/NeuralEpigeneticLandscapes/2_plus_states_system/test_neuralpde_example/nerual_PDE_readme_test.jl:54
"""
