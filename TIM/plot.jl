using DelimitedFiles
using Plots
using Measures
using LaTeXStrings
using Images
using Statistics
# Parameters
#x = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.4]
#x = [0.4 1.2, 1.6, 2.0, 2.4]
#x=[2.4, 1.6, 1.2]

# Start empty plot with larger fonts and a box frame
plot(xlabel=L"\beta", ylabel=L"f",
     xguidefont=font(20), yguidefont=font(14),   # axis labels bigger
     xtickfont=font(12), ytickfont=font(12),
     legend=true,
     # tick labels bigger
     #legendfont=font(9),                    # legend text bigger
     #legend=:topleft ,
     #(0.1,0.95), 
     # legend position
     framestyle=:box,
     guidefontsize = 14,
     thickness_scaling =2.4, 
     grid=false,                             # full box frame
     size=(1200, 800))

# Loop over datasets
x=[0, 2]
for val in x
    filename = "data_$(val).dat"
    data = readdlm(filename)
    #println(val," ", mean(data[:,2]))
    plot!((data[:,1]), data[:,2],
          marker=:circle, markersize=2,
          markerstrokewidth=0,
          lw=1,
          label = L"h= %$val")   # γ in math, value in text
end

#Save + display
savefig("data_TIM.png")
display(current())

