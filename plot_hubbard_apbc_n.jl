using DelimitedFiles
using Plots
using LaTeXStrings

data = readdlm(joinpath(@__DIR__, "data_density_U4.0_t0.dat"))

plot(xlabel=L"\mu", ylabel=L"<n>",
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
      label=L"U=4,t=0,\ \epsilon=10^{-4},\ D=48")

savefig(joinpath(@__DIR__, "data_hubbard_apbc_n1.png"))
display(current())
