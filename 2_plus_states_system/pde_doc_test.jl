using DifferentialEquations

f(x) = sin(2π.*x[:,1]).*cos(2π.*x[:,2])
sig(x) = .01 #Additive noise
dx = 1//2^(5)
mesh = notime_squaremesh([0 1 0 1],dx,:dirichlet)
prob = PoissonProblem(f,mesh,σ=sig)
sol = solve(prob)





f(t,x,u)  = ones(size(x,1)) - .5u
u0_func(x) = zeros(size(x,1))
σ(t,x,u) = 1u.^2
tspan = (0.0,5.0)
dx = 1//2^(3)
dt = 1//2^(11)
mesh = parabolic_squaremesh([0 1 0 1],dx,dt,tspan,:neumann)
u0 = u0_func(mesh.node)
prob = HeatProblem(u0,f,mesh,σ=σ)
