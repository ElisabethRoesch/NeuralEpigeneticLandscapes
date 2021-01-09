function mutate_a(a) # mutates the objects which a are pointing to. returns array.
        a .= ones(length(a))
        # println("\t a: ", a, " within mutate call")
end


function mutate_a!(a) # mutates the objects which a are pointing too. returns nothing.
        a .=  ones(length(a))
        # println("\t a: ", a, " within assign call")
        return nothing
end

function assign_a(a) # Creates a binding seeable in the scope of this function which assigns the variable "a" to 3 zeros. returns array.
        a =  ones(length(a))
        # println("\t a: ", a, " within assign call")
end

function assign_a2(a) # Creates a binding seeable in the scope of this function which assigns the variable "a" to 3 zeros. returns nothing.
        a =  ones(length(a))
        # println("\t a: ", a, " within assign call")
        return nothing
end


println("\n")

a = [1,2,3]
@time a_mutated = mutate_a(a)
println("a: ", a, " after mutate call")
a = [1,2,3]
@time a_mutated_ex = mutate_a!(a)
println("a: ", a, " after mutate! call ")
a = [1,2,3]
@time a_assigned = assign_a(a)
println("a: ", a, " after assign call")
a = [1,2,3]
@time a_assigned2 = assign_a2(a)
println("a: ", a, " after assign2 call")


println("\n")
println("a: ", a)
println("a_mutated: ", a_mutated)
println("a_mutated_ex: ", a_mutated_ex)
println("a_assigned: ", a_assigned)
println("a_assigned2: ", a_assigned2)
