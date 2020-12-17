#load packages
using DataFrames, CSV, Plots, KernelDensity, Distributions
#data
data = CSV.read("RNA_count_all22.csv", DataFrame, header = true) #read
#names of genes in the file
genes = names(data[:,(4:99)])

#plot layout
p = plot(grid = "off", layout = (8,12), ylabel = "Normlised cell count", xlabel = "Gene expression level")

#Option 1
plots = []
for i in 1:length(genes)
    data_raw = data[(data[:,3] .== 0.0), genes[i]]#names of genes in the file
    max_val = max(data_raw...)
    min_val = min(data_raw...)
    if min_val<max_val
        #println("min_val: ", min_val, " max_val: ", max_val)
        data_kde = kde(data_raw, boundary =(min(data_raw...), max(data_raw...)))
        #plot distributions for e.g. betaglobin
        p = plot(grid = "off", ylabel = "Normlised cell count", xlabel = "Gene expression level")
        histogram!(data_raw, normed = true, bins = 60, color = "#ffbb66", label = "Counts")
        plot!(data_kde.x, data_kde.density, color = "#3a8381", label = "KDE")
        push!(plots, p)
    else
        print("min_val!<max_val at: ",i)
    end

end
plot(plots[1:4]...)
savefig("option_1.pdf")

#Option 2
plots = []
for i in 1:length(genes)
    data_raw = data[(data[:,3] .== 0.0), genes[i]]#names of genes in the file
    max_val = max(data_raw...)
    min_val = min(data_raw...)
    if min_val<max_val
        #println("min_val: ", min_val, " max_val: ", max_val)
        data_kde = kde(data_raw, boundary =(min(data_raw...), max(data_raw...)))
        #plot distributions for e.g. betaglobin
        p = plot(grid = "off", ylabel = "#Cells", xlabel = "Expr.")
        histogram!(data_raw, normed = true, bins = 60, color = "#ffbb66", label = "Counts")
        plot!(data_kde.x, data_kde.density, color = "#3a8381", label = "KDE")
        push!(plots, p)
    else
        print("min_val!<max_val at: ",i)
    end

end
plot(plots[1:9]...)
savefig("option_2.pdf")


#Option 3
plots = []
for i in 1:length(genes)
    data_raw = data[(data[:,3] .== 0.0), genes[i]]#names of genes in the file
    max_val = max(data_raw...)
    min_val = min(data_raw...)
    if min_val<max_val
        #println("min_val: ", min_val, " max_val: ", max_val)
        data_kde = kde(data_raw, boundary =(min(data_raw...), max(data_raw...)))
        #plot distributions for e.g. betaglobin
        p = plot(grid = "off", ylabel = "", xlabel = "", axis= nothing)
        histogram!(data_raw, normed = true, bins = 60, linecolor= "#ffbb66", color = "#ffbb66", label = "")
        plot!(data_kde.x, data_kde.density, color = "#3a8381", label = "")
        push!(plots, p)
    else
        print("min_val!<max_val at: ",i)
    end

end
plot(plots[1:25]...)
savefig("option_3.pdf")

#Option 4
plots = []
for i in 1:length(genes)
    data_raw = data[(data[:,3] .== 0.0), genes[i]]#names of genes in the file
    max_val = max(data_raw...)
    min_val = min(data_raw...)
    if min_val<max_val
        #println("min_val: ", min_val, " max_val: ", max_val)
        data_kde = kde(data_raw, boundary =(min(data_raw...), max(data_raw...)))
        #plot distributions for e.g. betaglobin
        p = plot(grid = "off", ylabel = "", xlabel = "", axis= nothing)
        histogram!(data_raw, normed = true, bins = 60, linecolor= "#ffbb66", color = "#ffbb66", label = "")
        plot!(data_kde.x, data_kde.density, color = "#3a8381", label = "")
        push!(plots, p)
    else
        print("min_val!<max_val at: ",i)
    end

end
plot(plots...)
savefig("option_4.pdf")
