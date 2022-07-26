#
#
#  Both vertical and horizontal symmetric 
#   for tensor network renormalization group
#
#
using TensorOperations
using KrylovKit
using LinearAlgebra
using Printf
using LsqFit

if Sys.islinux()
    include("./tnr_idr_symmetric_func.jl")
    include("/home/wangwei/Dropbox/Code/WESTLAKE/TensorNetwork/VERSION1/2DClassical/partition.jl")
elseif Sys.isunix()
    include("./tnr_idr_symmetric_func.jl")
    include("/Users/wangweilan/Dropbox/Code/WESTLAKE/TensorNetwork/VERSION1/2DClassical/partition.jl")
end


FreeEnergyRelativeError =[]


# load parameter 
Tc = 2/log(1+sqrt(2))
#chimax = 10
J1 = 1.0


FE_exact =[]
FreeEnergy =[]
Magnetization =[]
Second_Magnetization =[]
Third_Magnetization =[]
Fourth_Magnetization =[]
Binder =[]
g4 =[]

Fidelity_Matrix =[]
#for relT in 0.96:0.01:1.03
for chimax in 32:2:32
    #local Temperature
    relT = 1.0
    println("Relative Temperature $relT, chimax $chimax")
    Temperature = relT*Tc

    # Result 
    Result = Dict{String,Any}()
    push!(Result,"logAnorm"=>0.0)
    push!(Result,"FreeEnergy"=>[])
    push!(Result,"Magnetization"=>[])
    push!(Result,"Second_Magnetization"=>[])
    push!(Result,"Third_Magnetization"=>[])
    push!(Result,"Fourth_Magnetization"=>[])
    push!(Result,"g4"=>[])

    feexact = ComputeFreeEnergy(1/Temperature)
    append!(FE_exact,feexact)


    #---------------------------------initial A Tensor----------------------------
    beta = 1/Temperature
    Jtemp = zeros(2,2,2,2); Jtemp[1,1,1,1]=1.0; Jtemp[2,2,2,2]=1.0;
    Jtemp_impurity = zeros(2,2,2,2); Jtemp_impurity[1,1,1,1] = -1.0; Jtemp_impurity[2,2,2,2] = 1.0

    Ltemp = [exp(J1*beta) exp(-J1*beta); exp(-J1*beta) exp(J1*beta)];
    Etemp = sqrt(Ltemp)

    @tensor A[:] := Jtemp[1,2,3,4]*Etemp[1,-1]*Etemp[2,-2]*Etemp[3,-3]*Etemp[4,-4]  
    @tensor A_single[:] := Jtemp_impurity[1,2,3,4]*Etemp[1,-1]*Etemp[2,-2]*Etemp[3,-3]*Etemp[4,-4]  
    #
    @tensor A_double[:] := Jtemp[1,2,3,4]*Etemp[1,-1]*Etemp[2,-2]*Etemp[3,-3]*Etemp[4,-4]
    @tensor A_triple[:] := Jtemp_impurity[1,2,3,4]*Etemp[1,-1]*Etemp[2,-2]*Etemp[3,-3]*Etemp[4,-4]  
    @tensor A_quadruple[:] := Jtemp[1,2,3,4]*Etemp[1,-1]*Etemp[2,-2]*Etemp[3,-3]*Etemp[4,-4]

    #=
    A,A_single,A_double,A_triple,A_quadruple = HOTRG_initial(A,A_single,A_double,A_triple,A_quadruple)
    A,A_single,A_double,A_triple,A_quadruple = HOTRG_initial(permutedims(A,[4,1,2,3]),permutedims(A_single,[4,1,2,3]),
                            permutedims(A_double,[4,1,2,3]),permutedims(A_triple,[4,1,2,3]),permutedims(A_quadruple,[4,1,2,3]))

    A = permutedims(A,[2,3,4,1]);
    A_single = permutedims(A_single,[2,3,4,1]);
    A_double = permutedims(A_double,[2,3,4,1]);
    A_triple = permutedims(A_triple,[2,3,4,1]);
    A_quadruple = permutedims(A_quadruple,[2,3,4,1]);
    =#





    Result["logAnorm"] = log(maximum(A))+4*Result["logAnorm"];
    A_single = A_single/maximum(A)
    A_double = A_double/maximum(A)
    A_triple = A_triple/maximum(A)
    A_quadruple = A_quadruple/maximum(A)
    A = A/maximum(A)

    @tensor Z[:] := A[1,2,1,2]
    fe = -1*Temperature*(log(Z[1])+Result["logAnorm"])/4;
    append!(Result["FreeEnergy"],fe)




    for nStep in 1:20
        println("============================ RG step $nStep =========================================")
        println("size(A) : ",size(A))
        #global A,A_single,A_double,A_triple,A_quadruple
        local fe,Z
        sizeA = size(A)

        # initial isometry and implicit disentangler
        v = compute_isometry_symmetric(A,chimax)
        #v_single = compute_isometry_symmetric(A,chimax,Aimp = A_single)
        
        #=
        v_double = compute_isometry_symmetric(A,chimax,Aimp = A_double)
        v_triple = compute_isometry_symmetric(A,chimax,Aimp = A_triple)
        v_quadruple = compute_isometry_symmetric(A,chimax,Aimp = A_quadruple)
        =#


        s = deepcopy(v)
        #s_single = deepcopy(v_single)

        #=
        s_double = deepcopy(v_double)
        s_triple = deepcopy(v_triple)
        s_quadruple = deepcopy(v_quadruple)
        =#

        @tensor phiphi[:] := A[1,2,3,7]*A[1,2,3,8]*A[6,5,4,7]*A[6,5,4,8]
        dis = reshape(Matrix(1.0I,sizeA[1]^2,sizeA[1]^2),sizeA[1],sizeA[1],sizeA[1],sizeA[1])
        println("Initial Fidelity")
        error = compute_fidelity(A,v,s,dis,phiphi,printdetail=true)
        @. model(x,p) = p[1].*x.^2 + p[2].*x + p[3]
        UpdateIter = 1000
        for k in 1:UpdateIter
            #println("=======================UpdateIter $k ==================================")
            if abs(error) < 1.0e-14
                break 
            end 





            #
            #s = update_s(A,)
            stemp = update_s(A,v,s,dis,k)
            k < 1000 ? alpha = 0.1 : alpha = 0.01
            k < 1000 ? iterrange = [1.0,5.0,7.0,10.0] : iterrange = [91.0,95.0,97.0,100.0]
            fidelity_array = Array{Float64}(undef,size(iterrange))
            itemp = 1 
            for i in iterrange#[0.0,3.0,5.0,10.0]#0:1:10
                snew = alpha*i*s + (1.0 - alpha*i)*stemp
                fidelity = compute_fidelity(A,v,snew,dis,phiphi,printdetail=false)
                fidelity_array[itemp] = fidelity
                itemp += 1
            end              
            L = curve_fit(model,iterrange,fidelity_array,[0.0,0.0,0.0])
            itemp = -L.param[2]/(2*L.param[1])
            #println("itemp $itemp")
            s = alpha*itemp*s + (1.0-alpha*itemp)*stemp
            if k %100 == 0
                fidelity = compute_fidelity(A,v,s,dis,phiphi,printdetail=true)
            end
            #

            #; Newton gradient 
            #
            #=
            s_k = deepcopy(s)            
            for l in 1:1 

                grad_k = compute_gradient(A,v,s_k,dis)
                sdirt_k = compute_direction(A,v,s_k,dis,grad_k)
                alpha = 1.0;itemp = 0
                fidelity_initial =  compute_fidelity(A,v,s_k,dis,phiphi,printdetail=false)
                fidelity_array = Array{Float64}(undef,30)
                alpha_array = Array{Float64}(undef,30)
                for i in 1:30 
                    #println("alpha: $alpha itemp : $itemp")
                    snew_k = s_k + alpha*sdirt_k
                    fidelity = compute_fidelity(A,v,snew_k,dis,phiphi,printdetail=false)
                    #=
                    if abs(fidelity) < abs(fidelity_initial) 
                        s = s+alpha*sdirt
                        break 
                    end
                    =#
                    fidelity_array[i] = fidelity
                    alpha_array[i] = alpha
                    alpha = alpha/2
                    itemp+=1
                end
                #
                value,index = findmin(fidelity_array)
                s_k = s_k+alpha_array[index]*sdirt_k
                fidelity_initial =  compute_fidelity(A,v,s_k,dis,phiphi,printdetail=true)
            end
            s = deepcopy(s_k)
            =#


            #; conjugate gradient
            # initial g_k and d_k and s_k
            #=
            @time begin
            @time @tensor AAvvdis[:] := A[5,4,-1,7]*A[3,2,-2,7]*v[4,6,-3]*v[2,1,-4]*dis[5,3,6,1]
            for m in 1:1
            s_k = deepcopy(s)
            grad_k = compute_gradient(A,v,s_k,dis,AAvvdis)
            d_k = -1*grad_k
            d_k_1 = deepcopy(d_k);
            beta_k_1 = 0;
            for l in 1:50
                #println("======================== l = $l ============================= ")
                #@time begin 
                ############################################################
                # compute alpha_k 

                #! compute denominator of alpha_k 
                #=
                @time @tensor AAs[:] := A[-2,-1,3,-4]*A[2,1,3,-5]*s_k[1,2,-3]
                @time @tensor AAss[:] := A[2,1,5,-1]*A[3,4,5,-2]*s_k[1,2,6]*s_k[4,3,6]            
                @time @tensor AAvx[:] := A[2,1,5,-1]*A[3,4,5,-2]*v[1,2,6]*d_k[4,3,6]
                @time @tensor denotr1[:] := AAs[-1,-2,-3,1,2]*AAvx[1,2] 
                @time @tensor denotr2[:] := AAs[-1,-2,-3,1,2]*AAvx[2,1]
                @time @tensor denotr3[:] := A[-2,-1,4,5]*d_k[1,2,-3]*AAss[5,3]*A[2,1,4,3]
                @time @tensor denotr4[:] := AAvvdis[6,3,-3,4]*A[-2,-1,6,5]*A[1,2,3,5]*d_k[2,1,4]
                =#

                @tensor Ad_k[:] := A[1,2,-1,-2]*d_k[2,1,-3]
                @tensor As_k[:] := A[1,2,-1,-2]*s_k[2,1,-3]
                @tensor As_kAd_k[:] := Ad_k[1,-1,2]*As_k[1,-2,2]
                @tensor denotr1[:] := As_kAd_k[1,2]*As_kAd_k[1,2]
                @tensor denotr2[:] := As_kAd_k[1,2]*As_kAd_k[2,1]                    
                @tensor denotr3[:] := As_k[1,5,2]*As_k[1,6,2]*Ad_k[4,6,3]*Ad_k[4,5,3] 
                @tensor denotr4[:] := AAvvdis[8,3,7,4]*A[6,5,8,9]*A[2,1,3,9]*d_k[1,2,4]*d_k[5,6,7] 
                Denominator = 4*(denotr1[1]+denotr2[1]+denotr3[1]-denotr4[1])
                ############################################################
                alpha_k = -1*dot(grad_k,d_k)/Denominator

                # update s_k with momentum
                
                s_k = s_k - alpha_k*grad_k +alpha_k*beta_k_1*d_k_1
               
                # update grad_k
                grad_k = compute_gradient(A,v,s_k,dis,AAvvdis)

                # compute beta_k
                @tensor Agrad_k[:] := A[1,2,-1,-2]*grad_k[2,1,-3]
                @tensor As_kAgrad_k[:] := Agrad_k[2,-1,1]*As_k[2,-2,1]
                @tensor denotr1[:] := As_kAgrad_k[1,2]*As_kAd_k[1,2]
                @tensor denotr2[:] := As_kAgrad_k[1,2]*As_kAd_k[2,1]
                @tensor denotr3[:] := As_k[1,5,2]*As_k[1,6,2]*Ad_k[4,6,3]*Agrad_k[4,5,3]
                @tensor denotr4[:] := AAvvdis[8,3,7,4]*A[6,5,8,9]*A[2,1,3,9]*d_k[1,2,4]*grad_k[5,6,7] 
                Nominator = 4*(denotr1[1]+denotr2[1]+denotr3[1]-denotr4[1])
                beta_k = Nominator/Denominator


                # update d_k_1 and beta_k_1 for later iterations
                beta_k_1 = beta_k; d_k_1 = d_k
                d_k = -1*grad_k + beta_k*d_k
                end
                s = deepcopy(s_k)
            end
            end
            #end
            if k%1 == 0
                @time fidelity_initial =  compute_fidelity(A,v,s,dis,phiphi,printdetail=true)
            end
            =#

            #=
            #Fletcher-Reeves Method
            s_k = deepcopy(s)
            grad_k = compute_gradient(A,v,s,dis)
            d_k = -1*grad_k
            d_k_1 = deepcopy(d_k);
            beta_k_1 = 0;
            for l in 1:20

                ############################################################
                # compute alpha_k
                for j in 1:10


                end
                alpha_k = -1*dot(grad_k,d_k)/Denominator[1]

                # update s_k with momentum
                s_k = s_k -alpha_k*grad_k +alpha_k*beta_k_1*d_k_1
               
                
                # update grad_k
                grad_k = compute_gradient(A,v,s_k,dis)

                # compute beta_k
                @tensor Nominator[:] := denotor[1,2,3]*grad_k[1,2,3]
                beta_k = Nominator[1]/Denominator[1]

                # update d_k_1 and beta_k_1 for later iterations
                beta_k_1 = beta_k; d_k_1 = d_k
                d_k = -1*grad_k + beta_k*d_k
                
            end
            fidelity_initial =  compute_fidelity(A,v,s_k,dis,phiphi,printdetail=true)
            s = deepcopy(s_k)
            end
            =#


            #println("dis entangler upudate")
            dis = update_dis(A,v,s,dis)
            #dis = update_dis_impurity(A,A_single,v,v_single,s,s_single,dis)
            v = update_v(A,v,s,dis)




            #=
            v_single = update_v_impurity(A,A_single,v,v_single,s,s_single,dis)
            s_single = update_s_impurity(A,A_single,v,v_single,s,s_single,dis)
            #
            v_double = update_v_impurity(A,A_double,v,v_double,s,s_double,dis)
            s_double = update_s_impurity(A,A_double,v,v_double,s,s_double,dis)

            v_triple = update_v_impurity(A,A_triple,v,v_triple,s,s_triple,dis)
            s_triple = update_s_impurity(A,A_triple,v,v_triple,s,s_triple,dis)

            v_quadruple = update_v_impurity(A,A_quadruple,v,v_quadruple,s,s_quadruple,dis)
            s_quadruple = update_s_impurity(A,A_quadruple,v,v_quadruple,s,s_quadruple,dis)
            =#

            #=
            if k%100==0
                error = compute_fidelity(A,v,s,dis,phiphi,printdetail=true)
                if abs.(error) .< 1.0e-14
                    break
                end
            end
            =#
        end

        # renormalize the tensor 
        y,w,sAsA = compute_projector_tnr_symmetric(A,v,s,dis,chimax)
        @tensor A_new[:] := A[13,12,16,17]*A[15,14,16,18]*A[1,2,5,7]*A[3,4,5,6]*s[12,13,24]*v[8,10,24]*s[14,15,22]*v[19,21,22]*s[2,1,11]*v[8,9,11]*
                                s[4,3,23]*v[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3]
        #=
        A_single_new = renormalize_fragment_tnr_one_impurity(A,A_single,v,v_single,s,s_single,y,w)
        #                   
        A_double_new_1 = renormalize_fragment_tnr_one_impurity(A,A_double,v,v_double,s,s_double,y,w)
        A_double_new_2 = renormalize_fragment_tnr_two_impurity(A,A_single,A_single,v,v_single,v_single,s,s_single,s_single,y,w)


        A_triple_new_1 = renormalize_fragment_tnr_one_impurity(A,A_triple,v,v_triple,s,s_triple,y,w)
        A_triple_new_2 = renormalize_fragment_tnr_two_impurity(A,A_single,A_double,v,v_single,v_double,s,s_single,s_double,y,w)
        A_triple_new_3 = renormalize_fragment_tnr_two_impurity(A,A_double,A_single,v,v_double,v_single,s,s_double,s_single,y,w)


        A_quadruple_new_1 = renormalize_fragment_tnr_one_impurity(A,A_quadruple,v,v_quadruple,s,s_quadruple,y,w)
        A_quadruple_new_2 = renormalize_fragment_tnr_two_impurity(A,A_triple,A_single,v,v_triple,v_single,s,s_triple,s_single,y,w)
        A_quadruple_new_3 = renormalize_fragment_tnr_two_impurity(A,A_single,A_triple,v,v_single,v_triple,s,s_single,s_triple,y,w)
        A_quadruple_new_4 = renormalize_fragment_tnr_two_impurity(A,A_double,A_double,v,v_double,v_double,s,s_double,s_double,y,w)

        A_double = deepcopy(A_double_new_1+A_double_new_2*2)
        A_triple = deepcopy(A_triple_new_1 + A_triple_new_2*3+A_triple_new_3*3)
        A_quadruple = deepcopy(A_quadruple_new_1+4*A_quadruple_new_2+4*A_quadruple_new_3+6*A_quadruple_new_4)
        
        
        A_double = A_double/maximum(A)/16
        A_triple = A_triple/maximum(A)/40
        A_quadruple = A_quadruple/maximum(A)/88
        =#
        A = deepcopy(A_new)
        #A_single = deepcopy(A_single_new)

        #A_single = A_single/maximum(A)/4
        Result["logAnorm"] = log(maximum(A))+4*Result["logAnorm"];
        A = A/maximum(A)


        @tensor Z[:] := A[1,2,1,2]

        #=
        #@tensor Z_single[:] := A_single[1,2,1,2]   
        @tensor Z_double[:] := A_double[1,2,1,2]
        @tensor Z_triple[:] := A_triple[1,2,1,2]
        @tensor Z_quadruple[:] := A_quadruple[1,2,1,2]
        =#

        fe = -1*Temperature*(log(Z[1])+Result["logAnorm"])/4^(nStep+1);
        append!(Result["FreeEnergy"],fe)
        #append!(Result["Magnetization"],Z_single[1]/Z[1])

        #=
        append!(Result["Second_Magnetization"],Z_double[1]/Z[1])
        append!(Result["Third_Magnetization"],Z_triple[1]/Z[1])
        append!(Result["Fourth_Magnetization"],Z_quadruple[1]/Z[1])
        append!(Result["g4"],(Z_quadruple[1]*Z[1]-3*Z_double[1]^2)/Z_double[1]^2)
        println("RG step: $nStep : ",fe," Magnetization: ",Z_single[1]/Z[1])
        =#


    end

    #
    append!(FreeEnergy,Result["FreeEnergy"][end])
    #append!(Magnetization,Result["Magnetization"][end])

    #=
    append!(Second_Magnetization,Result["Second_Magnetization"][end])
    append!(Third_Magnetization,Result["Third_Magnetization"][end])
    append!(Fourth_Magnetization,Result["Fourth_Magnetization"][end])
    append!(Binder,Result["Fourth_Magnetization"][end]*Result["Magnetization"][end]/Result["Second_Magnetization"][end]^2)
    append!(g4,Result["g4"][end])
    =#

end

x = 1