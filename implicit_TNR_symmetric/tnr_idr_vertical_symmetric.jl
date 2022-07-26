using TensorOperations
using KrylovKit
using LinearAlgebra
using Printf
using JLD2
include("./tnr_idr_vertical_symmetric_func.jl")
include("/home/wangwei/Dropbox/Code/WESTLAKE/TensorNetwork/VERSION1/2DClassical/partition.jl")



FreeEnergyRelativeError =[]


# load parameter 
Tc = 2/log(1+sqrt(2))
chimax = 6
J1 = 1.0


FE_exact =[]
FreeEnergy =[]
Magnetization =[]
Second_Magnetization =[]
Third_Magnetization =[]
Fourth_Magnetization =[]
Binder =[]
g4 =[]
Reflection =[]
#for relT in 0.96:0.01:1.02
for chimax in 6:1:12
    #chimax = 10
    chiV = 2*chimax
    #chiV = chimax
    chiS = chimax
    chidis = chimax
    #local Temperature
    #chimax = 16
    relT = 1.0
    println("Relative Temperature $relT")
    Temperature = relT*Tc

    # Result 
    global Result = Dict{String,Any}()
    push!(Result,"logAnorm"=>0.0)
    push!(Result,"FreeEnergy"=>[])
    push!(Result,"Magnetization"=>[])
    push!(Result,"Second_Magnetization"=>[])
    push!(Result,"Third_Magnetization"=>[])
    push!(Result,"Fourth_Magnetization"=>[])
    push!(Result,"g4"=>[])
    push!(Result,"Amatrix"=>[])
    push!(Result,"Binder"=>[])
    global feexact = ComputeFreeEnergy(1/Temperature)
    append!(FE_exact,feexact)


    #---------------------------------initial A Tensor----------------------------
    beta = 1/Temperature
    Jtemp = zeros(2,2,2,2); Jtemp[1,1,1,1]=1.0; Jtemp[2,2,2,2]=1.0;
    Jtemp_impurity = zeros(2,2,2,2); Jtemp_impurity[1,1,1,1] = -1.0; Jtemp_impurity[2,2,2,2] = 1.0

    Ltemp = [exp(J1*beta) exp(-J1*beta); exp(-J1*beta) exp(J1*beta)];
    Etemp = sqrt(Ltemp)

    @tensor A[:] := Jtemp[1,2,3,4]*Etemp[1,-1]*Etemp[2,-2]*Etemp[3,-3]*Etemp[4,-4]  
    #@tensor A[:] := A[-1,-3,2,1]*A[-2,1,4,-7]*A[-5,-4,2,3]*A[-6,3,4,-8]
    #A = reshape(A,4,4,4,4)    
    #
    @tensor A_single[:] := Jtemp_impurity[1,2,3,4]*Etemp[1,-1]*Etemp[2,-2]*Etemp[3,-3]*Etemp[4,-4]  
    @tensor A_double[:] := Jtemp[1,2,3,4]*Etemp[1,-1]*Etemp[2,-2]*Etemp[3,-3]*Etemp[4,-4]
    @tensor A_triple[:] := Jtemp_impurity[1,2,3,4]*Etemp[1,-1]*Etemp[2,-2]*Etemp[3,-3]*Etemp[4,-4]  
    @tensor A_quadruple[:] := Jtemp[1,2,3,4]*Etemp[1,-1]*Etemp[2,-2]*Etemp[3,-3]*Etemp[4,-4]

    #
    A,A_single,A_double,A_triple,A_quadruple = HOTRG_initial(A,A_single,A_double,A_triple,A_quadruple)
    A,A_single,A_double,A_triple,A_quadruple = HOTRG_initial(permutedims(A,[4,1,2,3]),permutedims(A_single,[4,1,2,3]),
                            permutedims(A_double,[4,1,2,3]),permutedims(A_triple,[4,1,2,3]),permutedims(A_quadruple,[4,1,2,3]))
    A = permutedims(A,[2,3,4,1]);
    A_single = permutedims(A_single,[2,3,4,1]);
    A_double = permutedims(A_double,[2,3,4,1]);
    A_triple = permutedims(A_triple,[2,3,4,1]);
    A_quadruple = permutedims(A_quadruple,[2,3,4,1]);
    #





    A_single = A_single/maximum(A)
    A_double = A_double/maximum(A)
    A_triple = A_triple/maximum(A)
    A_quadruple = A_quadruple/maximum(A)
    #
    Result["logAnorm"] = log(maximum(A))+4*Result["logAnorm"];
    A = A/maximum(A)

    @tensor Z[:] := A[1,2,1,2]
    fe = -1*Temperature*(log(Z[1])+Result["logAnorm"])/4;
    append!(Result["FreeEnergy"],fe)

    
    
    
    UpdateIter = 1200
    for nStep in 1:25
        println("============================ RG step $nStep =========================================")
        println(size(A))
        #global A,A_single,A_double,A_triple,A_quadruple
        #global A
        local fe,Z
        local y,w
        sizeA = size(A)

        # initial isometry and implicit disentangler
        vL0,vR0 = compute_isometry_vertical_symmetric(A,chiV) 
        println(size(vL0)," ",size(vR0))
        chikeptL = min(chiS,size(vL0,3))
        chikeptR = min(chiS,size(vR0,3))
        #println("chikept: ",chikeptL)
        sL = deepcopy(vL0[:,:,1:chikeptL]);sR = deepcopy(vR0[:,:,1:chikeptR]);
        vL = deepcopy(vL0[:,:,1:chikeptL]);vR = deepcopy(vR0[:,:,1:chikeptR]);
        println(size(vL)," ",size(vR))
        chitemp = min(chidis,size(vL,2))
        #vL = vL[:,1:chitemp,:]
        #vR = vR[:,1:chitemp,:]
        println(size(A))
        #=
        println(size(A_single))
        vL_single0,vR_single0 = compute_isometry_vertical_symmetric(A,chiV,Aimp = A_single)
        vL_single0 = deepcopy(vL);vR_single0 = deepcopy(vR)
        vL_single = deepcopy(vL_single0);vR_single = deepcopy(vR_single0)
        sL_single = deepcopy(vL_single0);sR_single = deepcopy(vR_single0)
        =#
        #=
        vL_double,vR_double= compute_isometry_vertical_symmetric(A,chimax,Aimp = A_double)
        vL_triple,vR_triple = compute_isometry_vertical_symmetric(A,chimax,Aimp = A_triple)
        vL_quadruple,vR_quadruple = compute_isometry_vertical_symmetric(A,chimax,Aimp = A_quadruple)
        =#


        #=
        sL_double = deepcopy(vL_double);sR_double = deepcopy(vR_double)
        sL_triple = deepcopy(vL_triple);sR_triple = deepcopy(vR_triple)
        sL_quadruple = deepcopy(vL_quadruple);sR_quadruple = deepcopy(vR_quadruple)
        =#         

        #
        dis = reshape(Matrix(1.0I,sizeA[1]^2,sizeA[1]^2),sizeA[1],sizeA[1],sizeA[1],sizeA[1])
        #dis = reshape(Matrix(1.0I,sizeA[1]^2,chitemp^2),sizeA[1],sizeA[1],chitemp,chitemp)
        println("Initial Fidelity")
        error = compute_fidelity_vertical_symmetric(A,vL,vR,sL,sR,dis,printdetail=true)
        #error = compute_fidelity_left_impurity_vertical_symmetric(A,A_single,vL,vR,vL_single,sL,sR,sL_single,dis,printdetail=true)   
        #error = compute_fidelity_right_impurity_vertical_symmetric(A,A_single,vL,vR,vR_single,sL,sR,sR_single,dis,printdetail=true)  
        
        @tensor A4[:] := A[2,1,-1,5]*A[3,5,-2,4]*A[2,1,-3,6]*A[3,6,-4,4]
        @tensor A4vLvR[:] := A4[-1,-2,6,7]*vL0[1,2,-3]*vR0[3,4,-4]*A[2,1,6,5]*A[4,5,7,3]
        #@tensor A4vLimpvR[:] := A4[-1,-2,6,7]*vL_single0[1,2,-3]*vR0[3,4,-4]*A_single[2,1,6,5]*A[4,5,7,3]
        #@tensor A4vLvRimp[:] := A4[-1,-2,6,7]*vL0[1,2,-3]*vR_single0[3,4,-4]*A[2,1,6,5]*A_single[4,5,7,3]

        for k in 1:UpdateIter

            if k%1000==0 || k==1
                println("======== iteration $k")
                error = compute_fidelity_vertical_symmetric(A,vL,vR,sL,sR,dis,printdetail=true)
                #compute_fidelity_left_impurity_vertical_symmetric(A,A_single,vL,vR,vL_single,sL,sR,sL_single,dis,printdetail=true)
                #compute_fidelity_right_impurity_vertical_symmetric(A,A_single,vL,vR,vR_single,sL,sR,sR_single,dis,printdetail=true)
        
                if abs.(error) < 1.0e-14
                    break
                end 
            end
            
            #
            sL = update_sL(A,vL,vR,sL,sR,dis)
            sR = update_sR(A,vL,vR,sL,sR,dis)
            if k>0
                vL = update_vL(A,vL,vR,sL,sR,dis)
                vR = update_vR(A,vL,vR,sL,sR,dis)
                dis = update_dis(A,vL,vR,sL,sR,dis)
            end
            #
            #=
            sL = update_sL_vertical_symmetric_large(A,vL0,vR0,A4vLvR,A4,vL,vR,sL,sR,dis)
            sR = update_sR_vertical_symmetric_large(A,vL0,vR0,A4vLvR,A4,vL,vR,sL,sR,dis)
            vL = update_vL_vertical_symmetric_large(A,vL0,vR0,A4vLvR,A4,vL,vR,sL,sR,dis)
            vR = update_vR_vertical_symmetric_large(A,vL0,vR0,A4vLvR,A4,vL,vR,sL,sR,dis)
            =#
            #dis = update_dis_vertical_symmetric_large(A,vL0,vR0,A4vLvR,A4,vL,vR,sL,sR,dis)


            
            #=
            sL = update_sL_full(A,A_single,vL,vR,vL_single,vR_single,sL,sR,sL_single,sR_single,dis)
            vL = update_vL_full(A,A_single,vL,vR,vL_single,vR_single,sL,sR,sL_single,sR_single,dis)
            sR = update_sR_full(A,A_single,vL,vR,vL_single,vR_single,sL,sR,sL_single,sR_single,dis)
            vR = update_vR_full(A,A_single,vL,vR,vL_single,vR_single,sL,sR,sL_single,sR_single,dis)
            dis = update_dis_full(A,A_single,vL,vR,vL_single,vR_single,sL,sR,sL_single,sR_single,dis)
            =#
            #vL_single = deepcopy(vL);
            #sL_single = deepcopy(sL);
            #vR_single = deepcopy(vR)
            #sR_single = deepcopy(sR)
            #
            
            #=
            if k > 0
                sL_single = update_sL_vertical_symmetric_impurity_large(A,A_single,vL0,vR0,vL_single0,vR_single0,A4vLvR,A4vLimpvR,A4vLvRimp,A4,vL,vR,vL_single,vR_single,
                                                    sL,sR,sL_single,sR_single,dis)
                sR_single = update_sR_vertical_symmetric_impurity_large(A,A_single,vL0,vR0,vL_single0,vR_single0,A4vLvR,A4vLimpvR,A4vLvRimp,A4,vL,vR,vL_single,vR_single,
                                                    sL,sR,sL_single,sR_single,dis)
                vL_single = update_vL_vertical_symmetric_impurity_large(A,A_single,vL0,vR0,vL_single0,vR_single0,A4vLvR,A4vLimpvR,A4vLvRimp,A4,vL,vR,vL_single,vR_single,
                                                    sL,sR,sL_single,sR_single,dis)
                vR_single = update_vR_vertical_symmetric_impurity_large(A,A_single,vL0,vR0,vL_single0,vR_single0,A4vLvR,A4vLimpvR,A4vLvRimp,A4,vL,vR,vL_single,vR_single,
                                                    sL,sR,sL_single,sR_single,dis)
             

                #=
                vL_single = update_vL_impurity(A,A_single,vL,vR,vL_single,sL,sR,sL_single,dis)
                sR_single = update_sR_impurity(A,A_single,vL,vR,vR_single,sL,sR,sR_single,dis)
                vR_single = update_vR_impurity(A,A_single,vL,vR,vR_single,sL,sR,sR_single,dis)
                =#
            end
            =#

        end


        #
        # renormalize the tensor 
        y,w = compute_projector_tnr_vertical_symmetric(A,vL,vR,sL,sR,dis,chimax)        
        @tensor A_new[:] := A[13,12,16,17]*A[15,14,16,18]*A[1,7,5,2]*A[3,6,5,4]*sL[12,13,24]*vL[8,10,24]*sL[14,15,22]*vL[19,21,22]*sR[2,1,11]*vR[8,9,11]*
                                sR[4,3,23]*vR[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3]
        #=
        #    single impurity
        A_single_new = renormalize_fragment_tnr_one_impurity_vertical_symmetric(A,A_single,vL,vR,vL_single,vR_single,sL,sR,sL_single,sR_single,y,w)
        #
        #   double  impurity
        A_double_new_1 = renormalize_fragment_tnr_one_impurity_vertical_symmetric(A,A_double,vL,vR,vL_double,vR_double,sL,sR,sL_double,sR_double,y,w)
        A_double_new_2 = renormalize_fragment_tnr_two_impurity_vertical_symmetric(A,A_single,A_single,vL,vR,vL_single,vR_single,vL_single,vR_single,
                                                                sL,sR,sL_single,sR_single,sL_single,sR_single,y,w)
        #   triple impurity
        A_triple_new_1 = renormalize_fragment_tnr_one_impurity_vertical_symmetric(A,A_triple,vL,vR,vL_triple,vR_triple,sL,sR,sL_triple,sR_triple,y,w)
        A_triple_new_2 = renormalize_fragment_tnr_two_impurity_vertical_symmetric(A,A_single,A_double,vL,vR,vL_single,vR_single,vL_double,vR_double,
                                                                sL,sR,sL_single,sR_single,sL_double,sR_double,y,w)
        A_triple_new_3 = renormalize_fragment_tnr_two_impurity_vertical_symmetric(A,A_double,A_single,vL,vR,vL_double,vR_double,vL_single,vR_single,
                                                                sL,sR,sL_double,sR_double,sL_single,sR_single,y,w)
        A_triple_new_4 = renormalize_fragment_tnr_three_impurity_vertical_symmetric(A,A_single,A_single,A_single,vL,vR,vL_single,vR_single,vL_single,vR_single,
                                                            vL_single,vR_single,sL,sR,sL_single,sR_single,sL_single,sR_single,sL_single,sR_single,y,w)
        
        #  quadruple impurity
        A_quadruple_new_1 = renormalize_fragment_tnr_one_impurity_vertical_symmetric(A,A_quadruple,vL,vR,vL_quadruple,vR_quadruple,sL,sR,sL_quadruple,sR_quadruple,y,w)
        A_quadruple_new_2 = renormalize_fragment_tnr_two_impurity_vertical_symmetric(A,A_triple,A_single,vL,vR,vL_triple,vR_triple,vL_single,vR_single,
                                                sL,sR,sL_triple,sR_triple,sL_single,sR_single,y,w)
        A_quadruple_new_3 = renormalize_fragment_tnr_two_impurity_vertical_symmetric(A,A_single,A_triple,vL,vR,vL_single,vR_single,vL_triple,vR_triple,
                                                sL,sR,sL_single,sR_single,sL_triple,sR_triple,y,w)
        A_quadruple_new_4 = renormalize_fragment_tnr_two_impurity_vertical_symmetric(A,A_double,A_double,vL,vR,vL_double,vR_double,vL_double,vR_double,
                                                sL,sR,sL_double,sR_double,sL_double,sR_double,y,w)
        A_quadruple_new_5 = renormalize_fragment_tnr_three_impurity_vertical_symmetric(A,A_single,A_single,A_double,vL,vR,vL_single,vR_single,vL_single,vR_single,vL_double,vR_double,
                                                sL,sR,sL_single,sR_single,sL_single,sR_single,sL_double,sR_double,y,w)
        A_quadruple_new_6 = renormalize_fragment_tnr_three_impurity_vertical_symmetric(A,A_single,A_double,A_single,vL,vR,vL_single,vR_single,vL_double,vR_double,vL_single,vR_single,
                                                sL,sR,sL_single,sR_single,sL_double,sR_double,sL_single,sR_single,y,w)
        A_quadruple_new_7 = renormalize_fragment_tnr_three_impurity_vertical_symmetric(A,A_double,A_single,A_single,vL,vR,vL_double,vR_double,vL_single,vR_single,vL_single,vR_single,
                                                sL,sR,sL_double,sR_double,sL_single,sR_single,sL_single,sR_single,y,w)
        A_quadruple_new_8 = renormalize_fragment_tnr_four_impurity_vertical_symmetric(A,A_single,A_single,A_single,A_single,vL,vR,vL_single,vR_single,vL_single,vR_single,vL_single,vR_single,vL_single,vR_single,
                                                sL,sR,sL_single,sR_single,sL_single,sR_single,sL_single,sR_single,sL_single,sR_single,y,w)
        =#
        
        ########################################################################################################
        #
        #   alternative rg method (test whether they will remove short range entanglements or not)
        #
        ########################################################################################################
        #=
        y,w = compute_projector_tnr_vertical_symmetric_alterate_rg(A,vL,vR,sL,sR,dis,chimax)
        @tensor A_new[:] :=  A[13,12,17,16]*A[14,16,18,15]*A[5,4,6,9]*A[8,9,10,7]*sL[12,13,23]*vL[2,3,23]*sR[15,14,22]*vR[21,19,22]*
                            sL[4,5,11]*vL[1,3,11]*sR[7,8,24]*vR[20,19,24]*y[2,1,-2]*y[21,20,-4]*w[17,18,-3]*w[6,10,-1]
        =#


        A = deepcopy(A_new)

        #=
        #A_single = deepcopy(A_single_new)
        #A_single = A_single/maximum(A)/(4*1)
        A_double = deepcopy(A_double_new_1+A_double_new_2*2)
        A_triple = deepcopy(A_triple_new_1 + (A_triple_new_2+A_triple_new_3)*3+A_triple_new_4*6)
        A_quadruple = deepcopy(A_quadruple_new_1+(A_quadruple_new_2+A_quadruple_new_3)*4+6*A_quadruple_new_4+
                        (A_quadruple_new_5+A_quadruple_new_6+A_quadruple_new_7)*12+A_quadruple_new_8*24)
        =#
        #
        #=
        A_double = A_double/maximum(A)/(4*1+6*2)
        A_triple = A_triple/maximum(A)/(4*1+12*3+4*6)
        A_quadruple = A_quadruple/maximum(A)/(4*1+12*4+6*6+12*12+1*24)
        =#
        #=
        println("check gauge:")
        x = compute_gauge(A,printdetail=true)
        @tensor Atemp[:] := A_single[-1,1,-3,2]*x[-2,1]*x[-4,2]
        println(" is impurity symmetric? : ",norm(Atemp - permutedims(A_single,[3,2,1,4]))/norm(Atemp))
        append!(Reflection,norm(Atemp - permutedims(A_single,[3,2,1,4]))/norm(Atemp))
        =#
        #=
        @tensor Atemp[:] := A_double[-1,1,-3,2]*x[-2,1]*x[-4,2]
        println(" is impurity symmetric? : ",norm(Atemp - permutedims(A_double,[3,2,1,4]))/norm(Atemp))
        @tensor Atemp[:] := A_triple[-1,1,-3,2]*x[-2,1]*x[-4,2]
        println(" is impurity symmetric? : ",norm(Atemp - permutedims(A_triple,[3,2,1,4]))/norm(Atemp))
        @tensor Atemp[:] := A_quadruple[-1,1,-3,2]*x[-2,1]*x[-4,2]
        println(" is impurity symmetric? : ",norm(Atemp - permutedims(A_quadruple,[3,2,1,4]))/norm(Atemp))
        =#

        Result["logAnorm"] = log(maximum(A))+4*Result["logAnorm"];
        A = A/maximum(A)
        append!(Result["Amatrix"],[A])

        @tensor Z[:] := A[1,2,1,2]
        fe = -1*Temperature*(log(Z[1])+Result["logAnorm"])/4^(nStep+1);
        append!(Result["FreeEnergy"],fe)
        #=
        @tensor Z_single[:] := A_single[1,2,1,2]   
        append!(Result["Magnetization"],Z_single[1]/Z[1])
        =#
        #=
        @tensor Z_double[:] := A_double[1,2,1,2]            
        @tensor Z_triple[:] := A_triple[1,2,1,2]
        @tensor Z_quadruple[:] := A_quadruple[1,2,1,2]
        append!(Result["Second_Magnetization"],Z_double[1]/Z[1])
        append!(Result["Third_Magnetization"],Z_triple[1]/Z[1])
        append!(Result["Fourth_Magnetization"],Z_quadruple[1]/Z[1])
        append!(Result["Binder"],Z_quadruple[1]*Z[1]/Z_double[1]^2)
        append!(Result["g4"],(Z_quadruple[1]*Z[1]-3*Z_double[1]^2)/Z_double[1]^2)
        =#
        #println("RG step: $nStep : ",fe," Magnetization: ",Z_single[1]/Z[1])
        #


    end

    #
    append!(FreeEnergy,Result["FreeEnergy"][25])
    #append!(Magnetization,Result["Magnetization"][end])
    #=
    append!(Second_Magnetization,Result["Second_Magnetization"][end])
    append!(Third_Magnetization,Result["Third_Magnetization"][end])
    append!(Fourth_Magnetization,Result["Fourth_Magnetization"][end])
    append!(Binder,Result["Fourth_Magnetization"][end]*Result["Magnetization"][end]/Result["Second_Magnetization"][end]^2)
    append!(g4,Result["g4"][end])
    =#
    #@save "Square_Ising_implicitTNR_vertical_symmetric_chimax_$(chimax).jld2" Result
    #@save "Square_Ising_0.9998Tc_implicitTNR_vertical_symmetric_chimax_$(chimax).jld2" Result


end

x = 1