function tests(x)
    y= Int(round(x, digits = 0))
    return y
end

tests(2.1)
gradient(tests,2)


xes = range(1, step =0.2, stop = 10)
plot(xes, tests.(xes))
