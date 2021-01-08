function mutate_a(a)
    a_mut = Array{Float32,1}(undef, 3)
    @time a_mut .= a
    return a_mut
end

function copy_a(a)
    @time a_copy = copy(a)
    return a_copy
end

function assign_a(a)
    @time a_ass = a
    return a_ass
end

function modify(x)
    x[1]=0
    return nothing
end

a = [1,2,3]


a_mutate = mutate_a(a)
a_assign = assign_a(a)
a_copy = copy_a(a)

println("a: ", a)
println("a_mutate: ", a_mutate)
println("a_assign: ", a_assign)
println("a_copy: ", a_copy)


modify(a)
