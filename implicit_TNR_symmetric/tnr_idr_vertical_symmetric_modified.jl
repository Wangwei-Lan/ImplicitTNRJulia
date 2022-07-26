using TensorOperations
using KrylovKit
using LinearAlgebra
using Printf
using JLD2
include("./tnr_idr_vertical_symmetric_modified_func.jl")
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
for chimax in 8:2:24
    #chimax = 20
    chidis = 
    
    chimaxtemp = chimax^2
    #local Temperature
    #chimax = 16
    relT = 0.90
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

    PL,PR,S = TEFT(A,chimax)
    @tensor test[:] := PL[-2,1]*PR[-1,1]



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

    
    
    
    UpdateIter = 0
    for nStep in 1:40
    #nStep = 1
        println("============================ RG step $nStep =========================================")
        println(size(A))
        #global A,A_single,A_double,A_triple,A_quadruple
        #global A
        local fe,Z
        local y,w
        sizeA = size(A)

        # initial isometry and implicit disentangler
        wL,wR,vL,vR = compute_isometry_vertical_symmetric_modified(A,chimaxtemp,chimax,RGstep=nStep)
        #wL_single,wR_single,vL_single,vR_single = compute_isometry_vertical_symmetric_modified(A,chimaxtemp,chimax,Aimp = A_single,RGstep=nStep)

        sL = deepcopy(vL);sR = deepcopy(vR);
        #sL_single = deepcopy(vL_single);sR_single = deepcopy(vR_single)
       

        dis = reshape(Matrix(1.0I,sizeA[1]^2,sizeA[1]^2),sizeA[1],sizeA[1],sizeA[1],sizeA[1])
        println("Initial Fidelity")
        @tensor sLtemp[:] := wL[-1,-2,1]*sL[1,-3];@tensor sRtemp[:] := wR[-1,-2,1]*sR[1,-3];
        @tensor vLtemp[:] := wL[-1,-2,1]*vL[1,-3];@tensor vRtemp[:] := wR[-1,-2,1]*vR[1,-3];
        error = compute_fidelity_vertical_symmetric_modified(A,vLtemp,vRtemp,sLtemp,sRtemp,dis,printdetail=true)
        #error = compute_fidelity_left_impurity_vertical_symmetric(A,A_single,vL,vR,vL_single,sL,sR,sL_single,dis,printdetail=true)
        #error = compute_fidelity_right_impurity_vertical_symmetric(A,A_single,vL,vR,vR_single,sL,sR,sR_single,dis,printdetail=true)  


        #
        for k in 1:UpdateIter

            if k%100==0 || k==100
                println("======== iteration $k")
                @tensor sLtemp[:] := wL[-1,-2,1]*sL[1,-3];@tensor sRtemp[:] := wR[-1,-2,1]*sR[1,-3];
                @tensor vLtemp[:] := wL[-1,-2,1]*vL[1,-3];@tensor vRtemp[:] := wR[-1,-2,1]*vR[1,-3];
                error = compute_fidelity_vertical_symmetric_modified(A,vLtemp,vRtemp,sLtemp,sRtemp,dis,printdetail=true)
                #compute_fidelity_left_impurity_vertical_symmetric(A,A_single,vL,vR,vL_single,sL,sR,sL_single,dis,printdetail=true)
                #compute_fidelity_right_impurity_vertical_symmetric(A,A_single,vL,vR,vR_single,sL,sR,sR_single,dis,printdetail=true)
        
                if abs.(error) < 1.0e-15
                    break
                end 
            end
            
            if k <0  #&& nStep <= 0       
                sL = update_sL_cg(A,wL,wR,vL,vR,sL,sR,dis)
            else
                sL = update_sL(A,wL,wR,vL,vR,sL,sR,dis)
            end

            if k <0 #&& nStep <= 0
                sR = update_sR_cg(A,wL,wR,vL,vR,sL,sR,dis)
            else
                sR = update_sR(A,wL,wR,vL,vR,sL,sR,dis)
            end

            if k > 0
                vL = update_vL(A,wL,wR,vL,vR,sL,sR,dis)
                vR = update_vR(A,wL,wR,vL,vR,sL,sR,dis)
                dis = update_dis(A,wL,wR,vL,vR,sL,sR,dis)
            end
            #
            
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
                sL_single = update_sL_impurity(A,A_single,wL,wR,wL_single,wR_single,vL,vR,vL_single,vR_single,sL,sR,sL_single,sR_single,dis)
                vL_single = update_vL_impurity(A,A_single,wL,wR,wL_single,wR_single,vL,vR,vL_single,vR_single,sL,sR,sL_single,sR_single,dis)
                sR_single = update_sR_impurity(A,A_single,wL,wR,wL_single,wR_single,vL,vR,vL_single,vR_single,sL,sR,sL_single,sR_single,dis)
                vR_single = update_vR_impurity(A,A_single,wL,wR,wL_single,wR_single,vL,vR,vL_single,vR_single,sL,sR,sL_single,sR_single,dis)
            end
            =#

        end
        #


        # renormalize the tensor 
        y,w = compute_projector_tnr_vertical_symmetric_modified(A,wL,wR,vL,vR,sL,sR,dis,chimax)
        @tensor sL[:] := wL[-1,-2,1]*sL[1,-3];@tensor sR[:] := wR[-1,-2,1]*sR[1,-3]
        @tensor vL[:] := wL[-1,-2,1]*vL[1,-3];@tensor vR[:] := wR[-1,-2,1]*vR[1,-3]
        @tensor A_new[:] := A[13,12,16,17]*A[15,14,16,18]*A[1,7,5,2]*A[3,6,5,4]*sL[12,13,24]*vL[8,10,24]*sL[14,15,22]*vL[19,21,22]*sR[2,1,11]*vR[8,9,11]*
                                sR[4,3,23]*vR[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3]
        #    single impurity
        #=
        @tensor sL_single[:] := wL_single[-1,-2,1]*sL_single[1,-3];@tensor sR_single[:] := wR_single[-1,-2,1]*sR_single[1,-3];
        @tensor vL_single[:] := wL_single[-1,-2,1]*vL_single[1,-3];@tensor vR_single[:] := wR_single[-1,-2,1]*vR_single[1,-3];
        A_single_new = renormalize_fragment_tnr_one_impurity_vertical_symmetric(A,A_single,vL,vR,vL_single,vR_single,sL,sR,sL_single,sR_single,y,w)
        =#
        #=
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
        
        #
        A = deepcopy(A_new)
        #A_single = deepcopy(A_single_new)
        #=
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
        
        #println("check gauge:")
        #x = compute_gauge(A,printdetail=true)
        #@tensor Atemp[:] := A_single[-1,1,-3,2]*x[-2,1]*x[-4,2]
        #println(" is impurity symmetric? : ",norm(Atemp - permutedims(A_single,[3,2,1,4]))/norm(Atemp))
        #append!(Reflection,norm(Atemp - permutedims(A_single,[3,2,1,4]))/norm(Atemp))
        #=
        @tensor Atemp[:] := A_double[-1,1,-3,2]*x[-2,1]*x[-4,2]
        println(" is impurity symmetric? : ",norm(Atemp - permutedims(A_double,[3,2,1,4]))/norm(Atemp))
        @tensor Atemp[:] := A_triple[-1,1,-3,2]*x[-2,1]*x[-4,2]
        println(" is impurity symmetric? : ",norm(Atemp - permutedims(A_triple,[3,2,1,4]))/norm(Atemp))
        @tensor Atemp[:] := A_quadruple[-1,1,-3,2]*x[-2,1]*x[-4,2]
        println(" is impurity symmetric? : ",norm(Atemp - permutedims(A_quadruple,[3,2,1,4]))/norm(Atemp))
        =#
        
        #
        Result["logAnorm"] = log(maximum(A))+4*Result["logAnorm"];
        A_single = A_single/maximum(A)/(4*1)
        A = A/maximum(A)
        append!(Result["Amatrix"],[A])

        @tensor Z[:] := A[1,2,1,2]
        fe = -1*Temperature*(log(Z[1])+Result["logAnorm"])/4^(nStep+1);
        append!(Result["FreeEnergy"],fe)
        #@tensor Z_single[:] := A_single[1,2,1,2]   
        #append!(Result["Magnetization"],Z_single[1]/Z[1])
        #
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
    @save "Square_Ising_0.9994Tc_implicitTNR_vertical_symmetric_chimax_$(chimax)_pinv_1.0e-9_1.jld2" Result


end
