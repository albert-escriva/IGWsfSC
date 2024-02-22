##################################################################
######################### IGWsfSC code (main-file) v_1_0#############
##################################################################
# Numerical code written in "Julia" languaje for the numerical computation of the Induced Gravitational Waves in the presence of 
# Crosovers beyond Standard Model.
# Numerical code written by Albert Escrivà , to obtain some of the results of Arxiv:2311.17760 (see Fig.4 and appendix-I for details)
# In case of using the code or taking profit, please consider to cite this github repostory and Arxiv:2311.17760.
# If you have questions, please send a message to Albert Escriva -- // escriva.manas.albert.y0@a.mail.nagoya-u.ac.jp // berto3493@gmail.com //
# See the notes pdf file for some more extra details and 
# check Arxiv:2311.17760  and references rethein for details about the formalism and numerical procedure.

########################################
##We import the necessary numerical packages. The user needs to install all the packages before can proceed with the calculation
begin
using Plots
using DifferentialEquations
using SpecialFunctions
using BenchmarkTools
using CubicSplines
using Base.Threads
using LinearAlgebra
using QuadGK
using Printf
using Mmap
using JLD2
using FileIO
end
########################################

##############################
##############################
##############################
##Parametters to set up the resolution of the ODE solver, absolute and relative tolernce
const re_tol_wcs = 1e-16
const abs_tol_wcs = 1e-16
const max_iter = 10 ^ 9 #maximum iterations allowed for the ODE solvers
const re_tol_g = 1e-11
const abs_tol_g = 1e-11
##############################
##############################
############################## parametters for the numerical convolution
const lim_superior_convolution = 100. #superior limit of the convolution
const lim_inferior_convolution = 0.01 #inferior limit of the convolution
const points_integration = 300  #number of points for the numerical convolution
##############################
##############################
### upper and lower values for the range of k-values considered, in terms of k_star (computed below), which is 
### the k-mode once it reenters the horizon at the scale of the minimum of the sound speed.
const lim_inferior_k = 0.01
const lim_superior_k = 1000.
##############################
##############################

##number of points for the desrivd spectrum of induced GWs (number of points)
const points_integration_full_gws = 30


##convinient notation x= k * eta
const x_ini_general = 0.001 #we set the initial value
const x_final = 1000. #value for the modes being well within the horizon

###values for radiation fluid (w and cs)
const w_rad = 1. / 3.;
const cs_rad = w_rad;
const a_0 = 1.0  #initial scale factor (gauge).


##let's now define the parametters for the equation of state. The values csmin = 0.1 and sigma = 3.0 are given as an example for the calculation
############################## accordng to the template of Eq.(1) and Eq.(3) in the work Arxiv:2311.17760 we have basically two parametters: cs_min and sigma
const rhoc =  1.0; ##energy density location of the minimum of c_s (it is arbitrary, just shift w and cs)
const csmin = 0.1 ; #minimum value of the sound speed
const sigma = 3.0; #value of the width of the crossover
##############################


#limit values for the energy density
const res_eq_state_diff = 10 ^ (-5.)
const factor_min_rho = 10^(-30.) 
const factor_max_rho = 10^(30.)

###############################################################
################ background radiation dynamics ################
############################################################### 
### equation of state
function w_rho(rhorho::Float64)
    return w_rad - ( sigma / rhorho ) * (w_rad - csmin) * sqrt( pi / 2.) * exp( sigma ^2 / 2. )  * erfc( (sigma ^2 - log(rhorho) ) / (sqrt(2.) * sigma))                        
end
## sound speed
function cs_rho(rhorho::Float64)
    return w_rad - (w_rad - csmin) * exp( -((log(rhorho) )^2) / (2. * sigma ^2) )            
end
## scale factor
function a_rad(etaeta::Vector{Float64})
    return a_0 .* (etaeta ./ eta_0)
end
## hubble factor
function H_rad(etaeta::Vector{Float64})
    return 1. ./ etaeta
end

function find_radiation_max(resol_eq_state::Float64)
    array_rhos = 10 .^ range(log10(rhoc), stop=log10((10 ^ 15) * rhoc), length=500) 
    for rho_item in array_rhos
        if abs(w_rho(rho_item)-w_rad)<resol_eq_state && abs(cs_rho(rho_item)-w_rad)<resol_eq_state
            return rho_item
        
        end
    end
end
function find_radiation_min(resol_eq_state::Float64)
    array_rhos =  10 .^ range(log10(rhoc), stop=log10((10 ^ (-15)) * rhoc), length=500) 
    for rho_item in array_rhos
        #println(rho_item, w_rho(rho_item))
        if abs(w_rho(rho_item)-w_rad)<resol_eq_state && abs(cs_rho(rho_item)-w_rad)<resol_eq_state
            return rho_item
        
        end
    end
end

const value_rho_min = find_radiation_min(res_eq_state_diff) 
const value_rho_max = find_radiation_max(res_eq_state_diff)
const rho_0_rad_min =  factor_min_rho * value_rho_min
const rho_0_rad_max =  factor_max_rho * value_rho_max
const k_cc = ((8. * pi *a_0^2 / 3.)^ (0.5) ) * ((rho_0_rad_max *  rhoc) ^ (0.25))

const eta_0_rad_min = sqrt( 3. / (8. * pi  * rho_0_rad_max * a_0^2))
const eta_0_rad_max = (3. * eta_0_rad_min / (8. * pi * rho_0_rad_min * a_0^2 )) ^ (0.25)
const eta_0_rad_min_rad = eta_0_rad_min * ( rho_0_rad_max / value_rho_max) ^ (0.25)


##solving the e.o.m for the background dynamics in log scale
function df_ln(du, u, p, t)
    du[1] = -exp(t) * sqrt(24. * pi) * (1. + w_rho(u[1])) * u[2] * u[1]^(3/2)
    du[2] = exp(t) * sqrt(8. * pi * u[1] / 3.) * u[2]^2
end


## find the horizon crosing value using the background solution
function find_horizon_crosing(kkk::Float64)
    puntos_crosing = 10000
    grid_crosing = 10. .^ range(log10(0.1 * (1. / kkk)), stop=log10(10. * (1. / kkk)), length =  puntos_crosing)
    for i=1:puntos_crosing
        #println(spline_H(grid_crosing[i])-kkk)
        if  spline_H(grid_crosing[i])-kkk  < 0 
            return grid_crosing[i]
            break
        end
    end
end


###let's solve in log-time steeep
u0_ln = [rho_0_rad_max, a_0]
tspan_ln = (log(eta_0_rad_min), log(eta_0_rad_max))
prob_ln = ODEProblem(df_ln, u0_ln, tspan_ln)
sol_ln = solve(prob_ln ,reltol=re_tol_wcs, abstol=abs_tol_wcs,force_dtmin=true,maxiters=max_iter )
time_array = exp.(sol_ln.t)
rho_sol = sol_ln[1,:]
a_sol = sol_ln[2,:]
prob_ln = nothing
sol_ln = nothing
GC.gc() # clean the memory delating the solver 



# vectorized solution
H_sol = a_sol .* sqrt.(8. .* pi .* rho_sol ./ 3.)
ww_sol = w_rad .- ( sigma ./ rho_sol ) .* sqrt.( pi / 2.) .* (w_rad .- csmin) .* exp.( sigma ^2 / 2. )  .* erfc.( (sigma ^2 .- log.(rho_sol) ) / (sqrt.(2.) .* sigma))  
cs_sol = w_rad .- (w_rad .- csmin) .* exp.( .-((log.(rho_sol) ) .^2 ) / (2. .* sigma .^2) )        


## cubic splines of the background solution, to be used later
spline_H = CubicSpline(time_array,H_sol)
spline_w = CubicSpline(time_array,ww_sol)
spline_cs = CubicSpline(time_array,cs_sol)
spline_a = CubicSpline(time_array,a_sol)
spline_rho = CubicSpline(time_array,rho_sol)


##########################################################################################
##### we writte the equation of state and the background dynamics in a data file #########
##########################################################################################
filename_data_eq_state = @sprintf("eq_state_julia_csmin=%.3e_sigms=%.3e_rhoc=%.3e.dat",csmin,sigma,rhoc) 
open(filename_data_eq_state, "w") do f  ##we open the file to writte the data in julia
    for rr = 1:lastindex(ww_sol)
        println(  f, time_array[rr], " ", ww_sol[rr] , " " , cs_sol[rr]  , " " ,  H_sol[rr], " " , a_sol[rr], " ", rho_sol[rr])
        flush(f)

end
end

###############################################################
###############################################################
###############################################################
###############################################################
###############################################################
###############################################################


### we find and compute k_star:
const min_cs, index_min_cs = findmin(cs_sol) ##finding the minimum of the sound speed
const eta_k_star = time_array[index_min_cs] 
const k_star = spline_H(eta_k_star) #k_star
const eta_ini_min = x_ini_general / (k_star * lim_superior_k * lim_superior_convolution) # minimum value of eta for the largest wave-mode of the convolution
const eta_eval_max = x_final / (k_star * lim_inferior_k * lim_inferior_convolution) # maximum value of eta for the minimum wave-mode of the convolution




###########################################################
###########################################################
## ode for the green function and bardeen potential ######
###########################################################
############# analytical solution for radiation domination ###
function phi_rad(etaeta,kk)
    xx = etaeta .* kk
    return (9. ./ (xx .^2)) .* (sin.(xx ./ sqrt(3.)) ./ (xx ./ sqrt(3.)) .- cos.(xx ./ sqrt(3.)) )
end
function der_phi_rad(etaeta,kk)
    xx = etaeta .* kk
    return (3. ./ (kk .^3 .* etaeta .^4)) .* (9. .* xx .* cos.(xx ./ sqrt.(3.)) .+ sqrt(3.) .* (-9. .+ xx .^2) .* sin.(xx ./ sqrt(3.)))
end
################################## two independent solutions of the green function
function g1_rad(etaeta,kk)
    return sin.(etaeta .* kk)
end
function g1_der_rad(etaeta,kk)
    return kk .* cos.(etaeta .* kk)
end
function g2_rad(etaeta,kk)
    return cos.(etaeta .* kk)
end
function g2_der_rad(etaeta,kk)
    return -kk .* sin.(etaeta .* kk)
end
##################################

###let's solve the bardeen equation
function bardeen(du, u, p, t)
    Hv = spline_H(t)
    cscs = spline_cs(t)
    du[1] = u[2]
    du[2] = -3. * u[2] * Hv * (1. + cscs)  -  u[1] * ( (p[1] ^ 2) * cscs + 3. * (Hv ^2) *(cscs-spline_w(t)) ) 
end

## let's solve the homogeneus green function
function green_homogeneus(du, u, p, t)
    Hv =  spline_H(t)
    kk =  p[1]
    du[1] = u[2]
    du[2] = -u[1] * ( (kk ^2) - (Hv ^2) * 0.5 * (1. - 3. * spline_w(t))) 
    nothing
end

##cumulative trapezoidal function (not used in the code actually)
function cumtrapz_cumulative(X::T, Y::T) where {T <: AbstractVector}
  @assert length(X) == length(Y)
  out = similar(X)
  out[1] = 0
  for i in 2:length(X)
    out[i] = out[i-1] + 0.5*(X[i] - X[i-1])*(Y[i] + Y[i-1])
  end
  return out
end

## integration using trapezoidal rule
function cumtrapz(x::Vector{T}, y::Vector{T}) where {T<:Float64}
    local len = length(y)
    if (len != length(x))
        error("Vectors must be of same length")
    end
    r = 0.0
    for i in 2:len
        r += (x[i] - x[i-1]) * (y[i] + y[i-1])
    end
    return r/2.0
end

## to find the time of horizon croosing
function find_horizon_crosing(kkk::Float64)
    puntos_crosing = 10000
    grid_crosing = 10. .^ range(log10(0.1 * (1. / kkk)), stop=log10(10. * (1. / kkk)), length =  puntos_crosing)
    for i=1:puntos_crosing
        #println(spline_H(grid_crosing[i])-kkk)
        if  spline_H(grid_crosing[i])-kkk  < 0 
            #println("ij")
            #cosa = grid_crosing[i]
            #println("crosing encontrado", " ", spline_H(cosa) / kkk)
            return grid_crosing[i]
            break
            
        end
    end
end
#############################
#############################
#############################



#### main function to compute the density of GWs
function density_GWs_mode(kk_mode::Float64, iteracion::Int64)



# we set the initial conformal time
    eta_inicial_integracion = x_ini_general / kk_mode 
#this condition ensures that we start the computation fulfill the initial conditions in the radiation dominated era
    if  eta_inicial_integracion > eta_0_rad_min_rad
        eta_inicial_integracion = eta_0_rad_min_rad
    end
    ##final eta value for modes well inside the horizon
    eta_eval = x_final / kk_mode ##we put eta star, and the corresponding k-mode
    tspan = (eta_inicial_integracion, eta_eval) #range of integration

    ##initial conditions for the two independent homogeneus green functions
    initial_g1 = g1_rad(eta_inicial_integracion,kk_mode)
    initial_der_g1 =  g1_der_rad(eta_inicial_integracion,kk_mode)
    initial_g2 = g2_rad(eta_inicial_integracion,kk_mode)
    initial_der_g2 =  g2_der_rad(eta_inicial_integracion,kk_mode)
        
    g1_initial = [initial_g1, initial_der_g1]
    g2_initial = [initial_g2, initial_der_g2]

    ##solving the first solution of the green function
    sol_g1_prob = ODEProblem( green_homogeneus , g1_initial , tspan , kk_mode )
    sol_g1 = solve(sol_g1_prob,reltol=re_tol_g,abstol=abs_tol_g,force_dtmin=true,maxiters=max_iter)
    time_arrayg = sol_g1.t ###it gives us the time we use to evaluate the solution in the other computation
    gg1_sol = sol_g1[1,:]
    lenght_time_array = length(time_arrayg)
    sol_g1 =nothing

    ##solving the second solution of the green function
    sol_g2_prob = ODEProblem(green_homogeneus , g2_initial , tspan , kk_mode  )
    sol_g2 = solve(sol_g2_prob, saveat=time_arrayg  ,   reltol=re_tol_g , abstol=abs_tol_g,force_dtmin=true,maxiters=max_iter)
    gg2_sol = sol_g2[1,:]
    sol_g2 = nothing    


    ##we can also make a cubicspline and make the integration
    #Notice taht the average should be computed with the averaged over a oscillation period, but it can be aproximated (for the cases tested) very well by considering A
    #larger period of integration, since the period of the oscilations is much smaller than the interval of integration. This makes the computation simplier.
    spline_g1 = CubicSpline(time_arrayg   ,  gg1_sol .^2  )
    spline_g2 = CubicSpline(time_arrayg   ,   gg2_sol .^2  )
    spline_g1g2 = CubicSpline(time_arrayg ,  gg1_sol .* gg2_sol )
    averagedg1 = (1. / (eta_eval-0.1 * eta_eval)) .* quadgk(x -> spline_g1(x), 0.1 * eta_eval, time_arrayg[end])[1]
    averagedg2 = (1. / (eta_eval-0.1 * eta_eval)) .* quadgk(x -> spline_g2(x), 0.1 * eta_eval, time_arrayg[end])[1]
    averagedg1g2 = (1. / (eta_eval-0.1 * eta_eval)) .* quadgk(x -> spline_g1g2(x), 0.1 * eta_eval, time_arrayg[end])[1]

    #averagedg1 = (1. / (eta_eval-eta_inicial_integracion)) .* quadgk(x -> spline_g1(x), time_arrayg[1], time_arrayg[end])[1]
    #averagedg2 = (1. / (eta_eval-eta_inicial_integracion)) .* quadgk(x -> spline_g2(x), time_arrayg[1], time_arrayg[end])[1]
    #averagedg1g2 = (1. / (eta_eval-eta_inicial_integracion)) .* quadgk(x -> spline_g1g2(x), time_arrayg[1], time_arrayg[end])[1]


    ###averaged quantities for the green function using trapezoidal rule. basically gives very similar result than using the 
    ###cubicspline, but is a bit faster
    #averagedg1 = (1. / (eta_eval-eta_inicial_integracion)) .* (cumtrapz(time_arrayg , gg1_sol .^2))
    #averagedg2 = (1. / (eta_eval-eta_inicial_integracion)) .* (cumtrapz(time_arrayg , gg2_sol .^2))
    #averagedg1g2 = (1. / (eta_eval-eta_inicial_integracion)) .* (cumtrapz(time_arrayg , gg1_sol .* gg2_sol ))  


    ### evaluate the quantities at the spline
    ww_vv = spline_w(time_arrayg)
    HH_vv = spline_H(time_arrayg)
    a_vv = spline_a(time_arrayg)
    # logarithmic grid spacing for the convolution over k1,k2
    grid_k = exp(1.0) .^ range(log(lim_inferior_convolution * kk_mode), stop=log(lim_superior_convolution * kk_mode), length = points_integration)
    

########## temporal files to save the solution of the Bardeen equation. Notice that the files will be generated
    filename__maping_bardeen_eq_phi = @sprintf("mmap_phi_%d_csmin=%.3e_sigms=%.3e.bin",iteracion,csmin,sigma) 
    filename__maping_bardeen_eq_derphi = @sprintf("mmap_derphi_%d_csmin=%.3e_sigms=%.3e.bin",iteracion,csmin,sigma) 
    temp_file_phi = open(filename__maping_bardeen_eq_phi, "w+") ###file of the eqaution of state, to be shared in all process
    temp_file_derphi = open(filename__maping_bardeen_eq_derphi, "w+") ###file of the eqaution of state, to be shared in all process
###################let's create the matrix to store the values of the kernel matrix





    #loop over the k1 grid of the convolution
    for kk1 in grid_k

        tspan_k1 = (eta_inicial_integracion, eta_eval)
        y0_k1 = [phi_rad(eta_inicial_integracion, kk1), der_phi_rad(eta_inicial_integracion, kk1)]
        prob_phi_1 = ODEProblem( bardeen , y0_k1 , tspan_k1 , kk1 )
        sol_phi_1 = solve(prob_phi_1  ,   reltol=re_tol_g , abstol=abs_tol_g , force_dtmin=true,maxiters=max_iter ,saveat = time_arrayg)

        ##we save the solution into the files
        write(temp_file_phi, sol_phi_1[1,:])
        write(temp_file_derphi, sol_phi_1[2,:])

        sol_phi_1 = nothing
        prob_phi_1 = nothing

        GC.gc()

    end
    close(temp_file_phi)
    close(temp_file_derphi)
    #let's close the files


    #now let's open again :)
    s_phi = open(filename__maping_bardeen_eq_phi)   # default is read-only
    s_derphi = open(filename__maping_bardeen_eq_derphi)   # default is read-only
    #we make a memory mapped, this allows us to save a  lot of memory.
    matrix_phi  = mmap(s_phi  ,  Matrix{Float64}  , ( lenght_time_array, points_integration))
    matrix_derphi  = mmap(s_derphi  ,  Matrix{Float64}  , ( lenght_time_array, points_integration))

############### now let's define the averaged_kernel_matrix, the most important quantity for this computation.
    kernel_matrix = zeros( points_integration , points_integration )

    #let's do the loop over the values k1,k2 to perform the numerical convolution
for l1 = 1:points_integration
        k1 = grid_k[l1]
        phi_v1 = @view matrix_phi[:,l1] #we take a slice of the matrix, but we don't store the solution, we just make a copy
        phi_der_v1 = @view matrix_derphi[:,l1]
        for l2 = 1:points_integration
            k2 = grid_k[l2]
            if  kk_mode <= k1 +k2 && abs(k1-k2) <= kk_mode
                #this is the constraint over the combinations of momentum modes (k1,k2). Notice that if the condition is not satisfied, 
                #the corresponding component of the kernel matrix will have a 0., so no contribution.

                phi_v2 = @view matrix_phi[:,l2]
                phi_der_v2 = @view matrix_derphi[:,l2]

                #we descompose the kernel into two parts
                kernel1_averaged = a_vv .* gg1_sol .* ( 2. .* phi_v1 .* phi_v2 .+ ( (4. ./ (3. .* (1. .+ ww_vv)))  .* (phi_v1 .+ phi_der_v1 ./ HH_vv) .* (phi_v2 .+ phi_der_v2 ./ HH_vv) ) )
                kernel2_averaged = a_vv .* gg2_sol .* ( 2. .* phi_v1 .* phi_v2 .+ ( (4. ./ (3. .* (1. .+ ww_vv)))  .* (phi_v1 .+ phi_der_v1 ./ HH_vv) .* (phi_v2 .+ phi_der_v2 ./ HH_vv) ) )
                
                #integration using the trapezoidal rule
                averaged_kernel_I1 = cumtrapz(time_arrayg , kernel1_averaged )        
                averaged_kernel_I2= cumtrapz(time_arrayg , kernel2_averaged ) 

                ##cubicspline and integration of the two kernel contributions. 
                #it can become slowly due to the large numbers of combinations in the loop. more efficient use the trapezoidal aroximation and without a significant difference in the results
                #spline_kernel_I1 = CubicSpline(time_arrayg   ,  kernel1_averaged  )
                #spline_kernel_I2 = CubicSpline(time_arrayg   ,  kernel2_averaged  )
                #averaged_kernel_I1 = quadgk(x -> spline_kernel_I1(x), time_arrayg[1], time_arrayg[end])[1]
                #averaged_kernel_I2 = quadgk(x -> spline_kernel_I2(x), time_arrayg[1], time_arrayg[end])[1]



                averaged_kernel_total = (kk_mode ^2) * ( (averaged_kernel_I1 ^2) * averagedg2 + (averaged_kernel_I2 ^2) * averagedg1 - 2. * averaged_kernel_I1 *averaged_kernel_I2 * averagedg1g2) 

                averaged_kernel_I1= nothing
                averaged_kernel_I2= nothing
                    
                kernel_matrix[l1,l2] = averaged_kernel_total ##matrix components of the averaged kernel square

            end
        end

    end
    #we exit the loop
    #######################
    #######################


    acHc = spline_H(x_final / kk_mode) * spline_a(x_final / kk_mode) #array to save the data 
    afHf =  spline_H(0.1 * eta_0_rad_max) * spline_a(0.1 * eta_0_rad_max)  #array to save the data 


    ##closing and cleaning the files if nedeed
    matrix_phi= nothing
    matrix_derphi = nothing
    GC.gc() #garbage collection
    

    #this delates the files for paralel mapping. The user can avoid the command and preserve those files
    rm(filename__maping_bardeen_eq_phi) 
    rm(filename__maping_bardeen_eq_derphi) 

    #######################
    #######################
#we return the kernel matrix, together with other useful quantities
return kernel_matrix , grid_k , acHc  , afHf , spline_a(eta_eval)  ,spline_H(eta_eval)

end

##################################
##################################
##################################
###we start our numerical computation
##################################
##################################

println("Welcome to the IGWsfSC code, We start the computation")

#let's define some Vectors
array_acHc_factor = zeros(points_integration_full_gws)
array_afHf_factor = zeros(points_integration_full_gws)
array_af_factor = zeros(points_integration_full_gws)
array_Hf_factor = zeros(points_integration_full_gws)
array_eta_crosing = zeros(points_integration_full_gws)

####we create the file where to save the kernel arrays and other quantities
jldopen("kernel_matrices_FILE2.jld2", "w") do file_spectrum_gws
##let's define three groups to endorse our data
##1) we endorse the data of the kernel group
kernel_group = JLD2.Group(file_spectrum_gws, "kernel_matrix_iteracion_group") #definir el grupo de matrices
#2) we create a group to endorse the data of each vector k from the convolution
vector_k_iter_group = JLD2.Group(file_spectrum_gws, "vector_k_iteracion_group")
#3) we create a group to endorse the data related to the global quantities for each iteration
vector_ktotal_gws_group = JLD2.Group(file_spectrum_gws, "vector_k_total_gws_quantities_group")
########



## paralelized loop over the different k-values
grid_k_modes_evaluation = 10. .^ range( log10(lim_inferior_k * k_star )  , stop=log10(lim_superior_k * k_star )  , length = points_integration_full_gws)
vector_ktotal_gws_group[string("vector_full_range_k_gws")] = grid_k_modes_evaluation ##we writte the vector of k values from the convolution
    ######
@time @threads for rr = 1:points_integration_full_gws
    ######
    kernel_matrix_iter, grid_k_iter , acHc_iter  , afHf_iter  , af_iter , Hf_iter =  density_GWs_mode(grid_k_modes_evaluation[rr] , rr) 
    ######
    vector_k_iter_group[string("vector_k_convolution_iteracion_",rr)] = grid_k_iter ##we writte the vector of k values from the convolution
    kernel_group[string("kernel_matrix_iteracion_",rr)] = kernel_matrix_iter ##we write the kernel matrix
    array_acHc_factor[rr] =  acHc_iter
    array_afHf_factor[rr] =  afHf_iter 
    ######
    array_af_factor[rr] =  af_iter
    array_Hf_factor[rr] =  Hf_iter 
    ######
    # time of horizon croosing for each mode k1
    eta_crosing = find_horizon_crosing(grid_k_modes_evaluation[rr])
    array_eta_crosing[rr] = eta_crosing
    println("number of points to compute: ", points_integration_full_gws , " we have finished iteration: " , rr)



end

vector_ktotal_gws_group[string("quantities_flrw_acHc")] = array_acHc_factor
vector_ktotal_gws_group[string("quantities_flrw_afHf")] = array_afHf_factor
vector_ktotal_gws_group[string("quantities_flrw_af")] = array_af_factor
vector_ktotal_gws_group[string("quantities_flrw_Hf")] = array_Hf_factor
vector_ktotal_gws_group[string("quantities_eta_crosing")] = array_eta_crosing
vector_ktotal_gws_group[string("k_star_value")] = k_star
println("computation finished")


close(file_spectrum_gws)
end
