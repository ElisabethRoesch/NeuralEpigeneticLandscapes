# using
# eq def
# repre fuc (grid)
# repre deriv (diffs)
# solve system


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
ERROR: LoadError: MethodError: no method matching iterate(::FastChain{Tuple{FastDense{typeof(σ),DiffEqFlux.var"#initial_params#73"{typeof(Flux.glorot_uniform),typeof(Flux.zeros),Int64,Int64}},FastDense{typeof(σ),DiffEqFlux.var"#initial_params#73"{typeof(Flux.glorot_uniform),typeof(Flux.zeros),Int64,Int64}},FastDense{typeof(identity),DiffEqFlux.var"#initial_params#73"{typeof(Flux.glorot_uniform),typeof(Flux.zeros),Int64,Int64}}}})
Closest candidates are:
  iterate(::DataStructures.SparseIntSet, ::Any...) at /Users/eroesch/.julia/packages/DataStructures/5hvIb/src/sparse_int_set.jl:147
  iterate(::Combinatorics.FixedPartitions) at /Users/eroesch/.julia/packages/Combinatorics/Udg6X/src/partitions.jl:112
  iterate(::Combinatorics.FixedPartitions, ::Array{Int64,1}) at /Users/eroesch/.julia/packages/Combinatorics/Udg6X/src/partitions.jl:112
  ...
Stacktrace:
 [1] discretize(::PDESystem, ::PhysicsInformedNN{Float64,FastChain{Tuple{FastDense{typeof(σ),DiffEqFlux.var"#initial_params#73"{typeof(Flux.glorot_uniform),typeof(Flux.zeros),Int64,Int64}},FastDense{typeof(σ),DiffEqFlux.var"#initial_params#73"{typeof(Flux.glorot_uniform),typeof(Flux.zeros),Int64,Int64}},FastDense{typeof(identity),DiffEqFlux.var"#initial_params#73"{typeof(Flux.glorot_uniform),typeof(Flux.zeros),Int64,Int64}}}},NeuralPDE.var"#201#203"{Flux.var"#34#36"{Float64}},NeuralPDE.var"#206#207"{Float32},GridTraining,Base.Iterators.Pairs{Union{},Union{},Tuple{},NamedTuple{(),Tuple{}}}}) at /Users/eroesch/.julia/packages/NeuralPDE/nfwQb/src/pinns_pde_solve.jl:639
 [2] top-level scope at /Users/eroesch/github/NeuralEpigeneticLandscapes/2_plus_states_system/test_neuralpde_example/PINN_combi.jl:45
 [3] include_string(::Function, ::Module, ::String, ::String) at ./loading.jl:1088
in expression starting at /Users/eroesch/github/NeuralEpigeneticLandscapes/2_plus_states_system/test_neuralpde_example/PINN_combi.jl:45


"""
