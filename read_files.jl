###############################################################################
######################### IGWsfSC code (2-file) v_1_0#############
###############################################################################
# Numerical code written in "Julia" language for the numerical computation of the induced gravitational waves.
# Numerical code written by Albert Escrivà , to obtain some of the results of Arxiv:2311.17760.
# In case of using the code, please consider citing Arxiv:2311.17760 and this GitHub repository.
# If you have questions, please send a message to // escriva.manas.albert.y0@a.mail.nagoya-u.ac.jp // berto3493@gmail.com //
# This code make a simple post-procesing of the quantities obtained from the main code "IGWsfSC" code.  The aim here is to compute the
# spectrum of the induced gravitational waves for each mode, once the modes are well inside the horizon (see Eq.(10) of Arxiv:2311.17760).
########################################

##We import the necessary numerical packages
begin
using Plots
using BenchmarkTools
using LaTeXStrings
using LinearAlgebra
using Printf
using Mmap
using JLD2
using FileIO
end
###let's solve the backgroud dynamics
# Trapezoidal matrix to make the numerical convolution
##################################
function trapezoidal_matrix(MM_points::Int64)
    MM_trapezoid = ones(MM_points , MM_points)
    MM_trapezoid[1 , MM_points] = 0.25
    MM_trapezoid[MM_points , 1] = 0.25
    MM_trapezoid[1 , 1] = 0.25
    MM_trapezoid[MM_points , MM_points] = 0.25
    for ii=2:MM_points-1
        MM_trapezoid[1,ii]= 0.5
        MM_trapezoid[ii,1]= 0.5
        MM_trapezoid[MM_points,ii]= 0.5
        MM_trapezoid[ii,MM_points]= 0.5
    end
    return MM_trapezoid
end
##################################


##function of the power spectrum. For the case considered in Arxiv:2311.17760 we consider flat spectrum. 
## Notice that we can generalize to any PS since we store numerically the kernel I(k,k1,k2) evaluated at sub-horizon scales.
function power_spectrum(kkk::Float64)
    A = 1.0 #amplitud power spetrum
    return A
end





##################################
## We open the file where we have stored the kernel matrix for a specific realization of the crossover template. See the main file "CIGWs_code_v_1_0.jl" 
## to check the string variables associated with the computations
jldopen("kernel_matrices_FILE2.jld2", "r") do file
    
    vector_k_array = file["vector_k_total_gws_quantities_group"][string("vector_full_range_k_gws")] # vector with the k values
    
    ##some background quantities
    array_acHc_factor = file["vector_k_total_gws_quantities_group"][string("quantities_flrw_acHc")]
    array_afHf_factor = file["vector_k_total_gws_quantities_group"][string("quantities_flrw_afHf")]
    array_af_factor = file["vector_k_total_gws_quantities_group"][string("quantities_flrw_af")]
    array_Hf_factor = file["vector_k_total_gws_quantities_group"][string("quantities_flrw_Hf")]
    kstar = file["vector_k_total_gws_quantities_group"][string("k_star_value")]    

    ####
    NN_k_points = length(vector_k_array) #number of points of the spectrum


    spectrum_IGWS_array = zeros(NN_k_points)
    for i=1:NN_k_points

        ##we take some values from each k-mode
        kkk = vector_k_array[i]
        acHc_factorr = array_acHc_factor[i]
        afHf_factorr = array_afHf_factor[i]
        af_factor = array_af_factor[i]
        Hf_factor = array_Hf_factor[i]

        ##### #open the categories stored from the "jld2" file
        vector_kj_iteration = file["vector_k_iteracion_group"][string("vector_k_convolution_iteracion_",i)]
        matrix_kernel = file["kernel_matrix_iteracion_group"][string("kernel_matrix_iteracion_",i)]
        suma_averaged_PS = 0.

        Njk = length(vector_kj_iteration) #number of k-modes from the convolution
        MM_trapezoid = trapezoidal_matrix(Njk) #set up the trapezoidal matrix

        kk_mode = vector_k_array[i] ##k-mode
        spacing = log(vector_kj_iteration[2]) - log(vector_kj_iteration[1]) #spacing in log-grid

        suma_averaged_PS= 0.
        #loop to do the convolution over k1,k2 modes
        for p1=1:Njk
            k1 = vector_kj_iteration[p1]
            for p2=1:Njk
                k2 = vector_kj_iteration[p2]
                factor_ks = ((k1 ^2 - (kk_mode ^2 - k2 ^2 + k1 ^2) ^2 / (4. * kk_mode ^2) ) ^2 ) / (k1 * k2 * kk_mode ^2)
                avegared_PS = (64. / ( 81. * af_factor^2)) * matrix_kernel[p1,p2] * (spacing ^2) * factor_ks * MM_trapezoid[p1,p2] * power_spectrum(k1) *power_spectrum(k2)
                suma_averaged_PS = avegared_PS + suma_averaged_PS
            end
        end
        grav_waves = (1. / 24.) * (  ( (acHc_factorr / afHf_factorr) ^2. ) ) * (   (kk_mode / ( Hf_factor ) ) ^2 ) * suma_averaged_PS 
        spectrum_IGWS_array[i] = grav_waves #we store the values
    println("corresponding induced grav. wave value for given mode k iteration::"," ",i, " ," ,grav_waves )
    end
    #let's make a plot of the final result and save it
    Omega_gws_plot = plot( vector_k_array ./ kstar, spectrum_IGWS_array , xaxis=:log ,xlab = L"k/k_{\star}",  legend = false,ylab = L"\Omega_{\rm GW}(k, \eta_{\rm sh})/(\Omega_{r,0} h^2)", yaxis=:log, ls=:dot,seriestype=:scatter, linewidth=3 , ylims=(0.01,2.))
    savefig(Omega_gws_plot, "Omega_gws_plot.pdf") 



    ### we also store the solution
    filename_data_gws = @sprintf("gws_data_result.dat") 
    open(filename_data_gws, "w") do f  ##we open the file to writte the data in julia
        for qq = 1:NN_k_points
            println(  f, vector_k_array[qq] / kstar, " ", kstar , " " , spectrum_IGWS_array[qq]  )
            flush(f)

end
end



end



