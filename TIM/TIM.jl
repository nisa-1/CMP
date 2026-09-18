using TNRKit
using TensorKit
using LinearAlgebra

 
# Hubbard_tensor_12_leg(μ, ε, t1, U; T=T)
μvals  = Float64[]
Nvals  = Float64[]
norms  = Float64[]
Hvalues =[2.0]
for β in 0:0.05:1.5
    #Hubbard_tensor(μ, ε, t1, U; T=T)
    N1 = 12
    ε  = 1e-4
    L  = 2^N1
    V  = L*β
    d  = 24
    Jx = 0.1
    Jy = 0.3
    #δμ = 1e-4
    T1 = classical_ising(Trivial, β;
                     h=2.0,
                     Jx=1 ,
                     Jy=1 )
                     
                     
    ###########################################
    #=
    HOTRG_Y: modified function which has function:

    function step!(scheme::HOTRG, trunc::MatrixAlgebraKit.TruncationStrategy)
    Ux, = _get_hotrg_xproj(scheme.T, scheme.T, trunc)
    scheme.T = _step_hotrg_y(scheme.T, scheme.T, Ux)
    Uy, = _get_hotrg_yproj(scheme.T, scheme.T, trunc)
    scheme.T = _step_hotrg_x(scheme.T, scheme.T, Uy)
    return scheme
    
    =# 
    ###################################################
    
    
    
    scheme_y = HOTRG_Y(T1)
    data_y=run!(scheme_y, truncrank(d), maxiter(15) ;verbosity=0)
    T_after_y = scheme_y.T
    


  
   
    
    scheme=HOTRG(T_after_y)
    data=run!(scheme, truncrank(d),maxiter(35) ;verbosity=2)
    
    f = free_energy(data_y,  β; scalefactor=2.0)+free_energy(data, 1; scalefactor=4.0)
    @show  β, f
    push!(μvals,  β)
    push!(Nvals, f)

end
    

for ( β, f) in zip(μvals, Nvals)
    println("$β"," ", "$f")
end


