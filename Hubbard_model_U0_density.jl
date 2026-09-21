using TNRKit
using TensorKit
using LinearAlgebra
using DelimitedFiles

function R_tensor()
    R = zeros(ComplexF64, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2)
    for (i_u, i_d, j_u, j_d, iτ_u, iτ_d, pi_u, pi_d, pj_u, pj_d, piτ_u, piτ_d) in Iterators.product([0:1 for _ in 1:12]...)
        R[i_u+1, i_d+1, j_u+1, j_d+1, iτ_u+1, iτ_d+1, pi_u+1, pi_d+1, pj_u+1, pj_d+1, piτ_u+1, piτ_d+1] =
            (i_u   *    iτ_u) +
           ( i_d   *  ( iτ_u + pj_u  + piτ_u + pi_u  + j_u + iτ_d) )+
           ( j_u   *  ( iτ_u + pj_u  + piτ_u + pi_u) )+
            (j_d   *  ( iτ_u + pj_u  + piτ_u + pi_u  + iτ_d + pj_d + piτ_d + pi_d)) +
           ( iτ_d  *  ( pj_u + piτ_u + pi_u)) +
           ( piτ_d *  ( pj_u + piτ_u + pi_u  + pj_d)) +
            (piτ_u *    pj_u) +
            (pj_d  *  ( pj_u + pi_u) )+
            (pi_d  *  pi_u)

    end
    return R
end
function Hubbard_tensor_12_leg(μ::Number, ε::Number, t1::Number, U::Number; T::Type{<:Complex} = ComplexF64)


    δ(x, y) = ==(x, y)
    t = zeros(T, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2)
    R = R_tensor()
    V = Vect[FermionParity](0 => 1, 1 => 1)

    for (i_u, i_d, j_u, j_d, iτ_u, iτ_d, pi_u, pi_d, pj_u, pj_d, piτ_u, piτ_d) in Iterators.product([0:1 for _ in 1:12]...)
        r = R[i_u+1, i_d+1, j_u+1, j_d+1, iτ_u+1, iτ_d+1, pi_u+1, pi_d+1, pj_u+1, pj_d+1, piτ_u+1, piτ_d+1]
            t[i_u+1, i_d+1, j_u+1, j_d+1, iτ_u+1, iτ_d+1, pi_u+1, pi_d+1, pj_u+1, pj_d+1, piτ_u+1, piτ_d+1] =

          (  (-1)^(i_u+i_d)*(sqrt(t1*ε))^( i_u + i_d + j_u + j_d + pi_u + pi_d + pj_u + pj_d) )*
          (
                            (  δ( 1, iτ_d + i_d + pj_d) * δ(1, piτ_d + pi_d + j_d) * δ(1, iτ_u + i_u + pj_u) * δ(1, piτ_u + pi_u + j_u)
            - ((μ*ε+1) *       δ( 0, iτ_d + i_d + pj_d) * δ(0, piτ_d + pi_d + j_d) * δ(1, iτ_u + i_u + pj_u) * δ(1, piτ_u + pi_u + j_u))
            - ((μ*ε+1) *       δ( 1, iτ_d + i_d + pj_d) * δ(1, piτ_d + pi_d + j_d) * δ(0, iτ_u + i_u + pj_u) * δ(0, piτ_u + pi_u + j_u))
            - ( (U*ε-(μ*ε+1)^2)*δ( 0, iτ_d + i_d + pj_d) * δ(0, piτ_d + pi_d + j_d) * δ(0, iτ_u + i_u + pj_u) * δ(0, piτ_u + pi_u + j_u)) ))*((-1)^r)

    end
    return TensorMap(t, V ⊗ V ⊗ V ⊗ V ⊗ V ⊗ V ← V ⊗ V ⊗ V ⊗ V ⊗ V ⊗ V)
end

function Hubbard_tensor(μ::Number, ε::Number, t1::Number, U::Number; kwargs...)
    return Hubbard_tensor(FermionParity, μ, ε, t1, U; kwargs...)
end

function Hubbard_tensor(
    ::Type{FermionParity},
    μ::Number, ε::Number, t1::Number, U::Number;
    T::Type{<:Complex} = ComplexF64,
)

    T_unfused = Hubbard_tensor_12_leg(μ, ε, t1, U; T=T)

    V  = Vect[FermionParity](0 => 1, 1 => 1)
    W4 = fuse(V, V, V, V)
    W2 = fuse(V, V)

    U4   = isometry(W4, V ⊗ V ⊗ V ⊗ V)
    U2   = isometry(W2, V ⊗ V)
    U4dg = adjoint(U4)
    U2dg = adjoint(U2)


    @tensor Tf[-1 -2; -3 -4] :=
    T_unfused[1 2 3 4 5 6; 7 8 9 10 11 12] *
    U4[-1; 1 2 3 4] * U2[-2; 5 6] *
    U2dg[11 12; -3] * U4dg[7 8 9 10; -4]
    E   = isometry(W4, W2)
    Edg = adjoint(E)

    @tensor T_equal[-1 -2; -3 -4] :=
    Tf[-1 1; 2 -4] *
    E[-2; 1] *
    Edg[2; -3]

    return T_equal
end

##############################################################################
# mu sweep at U=0 (free fermions, t=1): two-stage HOTRG (pure-tau HOTRG_Y
# warmup, then combined x/y HOTRG), APBC in tau / PBC in sigma via
# TNRKit.finalize_apbc!, following Akiyama & Kuramashi, arXiv:2105.00372.
#
# Electron density <n>(mu) = (1/V) d lnZ/dmu = df/dmu, since f is already
# defined here as lnZ/V_phys -- computed by central-differencing the f(mu)
# values on the mu grid (Eq. 7 of the paper).
##############################################################################

μvals = Float64[]
fvals = Float64[]

N_y  = 28   # pure-τ HOTRG_Y warmup steps
N_xy = 15   # combined x/y HOTRG steps
ε    = 1e-4
d    = 48
U    = 4.0
t1   = 0

# mu grid: coarse away from the free-fermion band edges (mu = +-2t = +-2),
# finer between 1<=|mu|<=2, following the paper's note (Sec III, below
# Eq. 7) that finer resolution there is needed to resolve the complicated
# mu dependence of <n> near the (U,t)=(0,1) band edge.
μgrid = sort(unique(vcat(
    -4:0.5:-2.0,
    -2.0:0.1:-1.0,
    -0.5:0.5:0.5,
    1.0:0.1:2.0,
    2.0:0.5:4.0,
)))

apbc_Finalizer = Finalizer(TNRKit.finalize_apbc!, Float64)

# physical volume V = Nσ·β for the final lattice, used to turn the
# telescoped ln Z into the density plotted in Akiyama & Kuramashi (2021),
# Fig 1 ("thermodynamic potential ln Z/V")
Nσ_final = 2.0^N_xy
Nτ_final = 2.0^(N_y + N_xy)
V_phys   = Nσ_final * ε * Nτ_final

for μ in μgrid
    T1 = Hubbard_tensor(μ, ε, t1, U)
    scheme_y = HOTRG_Y(T1)
    data_y = run!(
        scheme_y,
        truncrank(d),
        maxiter(N_y),
        apbc_Finalizer; finalize_beginning=false,
        verbosity=0
    )

    scheme = HOTRG(scheme_y.T)
    data = run!(
        scheme,
        truncrank(d),
        maxiter(N_xy),
        apbc_Finalizer; finalize_beginning=false,
        verbosity=0
    )

    # ln Z, telescoped continuously across the stage-1 -> stage-2 boundary
    # (Z_i = n_i * Z_{i-1}^growth, growth=2 during the y-only stage and 4
    # during the combined stage). Summing two independent free_energy(...)
    # calls is wrong here because each call resets its own initial_size=1
    # convention, even though stage 2 actually starts from the
    # already-coarse-grained block that stage 1 produced.
    lnZ = 0.0
    for n in data_y
        lnZ = log(n) + 2 * lnZ
    end
    for n in data
        lnZ = log(n) + 4 * lnZ
    end
    f = lnZ / V_phys

    @show f, μ
    push!(μvals, μ)
    push!(fvals, f)
end

println("\nμ, f:")
for (μi, fi) in zip(μvals, fvals)
    println(μi, "  ", fi)
end

# electron density via central differences of f(μ) on the (non-uniform)
# swept grid: <n> = df/dμ ≈ [f(μ_{i+1}) - f(μ_{i-1})] / (μ_{i+1} - μ_{i-1})
nvals = Float64[]
for i in 2:(length(μvals) - 1)
    push!(nvals, (fvals[i+1] - fvals[i-1]) / (μvals[i+1] - μvals[i-1]))
end

println("\nμ, <n>:")
for (μi, ni) in zip(μvals[2:end-1], nvals)
    println(μi, "  ", ni)
end

f_datfile = joinpath(@__DIR__, "data_hubbard_U$(U)_t$(t1).dat")
n_datfile = joinpath(@__DIR__, "data_density_U$(U)_t$(t1).dat")
writedlm(f_datfile, hcat(μvals, fvals))
writedlm(n_datfile, hcat(μvals[2:end-1], nvals))
println("\nwrote ", f_datfile)
println("wrote ", n_datfile)
