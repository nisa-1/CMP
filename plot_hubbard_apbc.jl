using DelimitedFiles
using Plots
using LaTeXStrings

data = readdlm(joinpath(@__DIR__, "data_hubbard_apbc.dat"))

plot(xlabel=L"\mu", ylabel=L"\ln Z / V",
     xguidefont=font(20), yguidefont=font(14),
     xtickfont=font(12), ytickfont=font(12),
     legend=:topleft,
     framestyle=:box,
     guidefontsize=14,
     thickness_scaling=2.4,
     grid=false,
     size=(1200, 800))

plot!(data[:, 1], data[:, 2],
      marker=:circle, markersize=4,
      markerstrokewidth=0,
      lw=2,
      label=L"U/t=4,\ \epsilon=10^{-4},\ D=48")

savefig(joinpath(@__DIR__, "data_hubbard_apbc.png"))
display(current())
