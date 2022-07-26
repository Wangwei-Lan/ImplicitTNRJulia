#
#
# TNR with Z2 symetry + horizontal symmetry
# without single isometries in RG steps; large environment
#
#
using TensorOperations
using KrylovKit
using LinearAlgebra
using Printf
using JLD2

include("./Z2_symmetry_func.jl")
include("/home/wangwei/Dropbox/Code/WESTLAKE/TensorNetwork/VERSION1/2DClassical/partition.jl")



FreeEnergyRelativeError =[]


# load parameter 
Tc = 2/log(1+sqrt(2))
chimax = 10
J1 = 1.0


FE_exact =[]
FreeEnergy =[]
freeenergy =[]
Magnetization =[]
Second_Magnetization =[]
Third_Magnetization =[]
Fourth_Magnetization =[]
Binder =[]
g4 =[]
#for relT in 0.9998:0.00001:1.0002
for chimax in 3:1:16
    #local Temperature
    #chimax = 10
    chiw_max = 2*chimax
    chidis_max = chimax
    chis_max = chimax
    #relT = 1.0#
    relT = 1.0
    println("===================================Relative Temperature $relT, chimax = $chimax=====================================")
    Temperature = relT*Tc

    location = 20
    position = "rightdown"
    # Result 
    Result = Dict{String,Any}()
    push!(Result,"logAnorm"=>0.0)
    push!(Result,"FreeEnergy"=>[])
    push!(Result,"Magnetization"=>[])
    push!(Result,"Second_Magnetization"=>[])
    push!(Result,"Third_Magnetization"=>[])
    push!(Result,"Fourth_Magnetization"=>[])
    push!(Result,"Binder"=>[])
    push!(Result,"g4"=>[])
    push!(Result,"Amatrix"=>[])
    feexact = ComputeFreeEnergy(1/Temperature)
    append!(FE_exact,feexact)


    #---------------------------------initial A Tensor----------------------------
    beta = 1/Temperature
    Jtemp = zeros(2,2,2,2); Jtemp[1,1,1,1]=1.0; Jtemp[2,2,2,2]=1.0;
    Jtemp_impurity = zeros(2,2,2,2); Jtemp_impurity[1,1,1,1] = 1.0; Jtemp_impurity[2,2,2,2] = 1.0
    Jtemp_impurity_1 = zeros(2,2,2,2); Jtemp_impurity_1[1,1,1,1] = -1.0; Jtemp_impurity_1[2,2,2,2] = 1.0
    Jtemp_impurity_2 = zeros(2,2,2,2); Jtemp_impurity_2[1,1,1,1] = -1.0; Jtemp_impurity_2[2,2,2,2] = 1.0

    Ltemp = [exp(J1*beta) exp(-J1*beta); exp(-J1*beta) exp(J1*beta)];
    Etemp = sqrt(Ltemp)
    Dtemp = zeros(2,2,2);Dtemp[1,1,1] = 1.0; Dtemp[2,2,2] = 1.0



    #
    # Binder ration for magnetization
    #@tensor A[:] := Dtemp[1,-1,2]*Dtemp[4,-4,3]*Dtemp[8,-3,5]*Dtemp[6,-2,7]*Ltemp[2,4]*Ltemp[3,8]*Ltemp[5,6]*Ltemp[7,1]
    
    #
    @tensor A[:] := Jtemp[1,2,3,4]*Etemp[1,-1]*Etemp[2,-2]*Etemp[3,-3]*Etemp[4,-4]  
    @tensor A_single_1[:] := Jtemp_impurity_1[1,2,3,4]*Etemp[1,-1]*Etemp[2,-2]*Etemp[3,-3]*Etemp[4,-4];  
    @tensor A_single_2[:] := Jtemp_impurity_2[1,2,3,4]*Etemp[1,-1]*Etemp[2,-2]*Etemp[3,-3]*Etemp[4,-4];  
    @tensor A_double[:] := Jtemp[1,2,3,4]*Etemp[1,-1]*Etemp[2,-2]*Etemp[3,-3]*Etemp[4,-4]
    @tensor A_triple[:] := Jtemp_impurity[1,2,3,4]*Etemp[1,-1]*Etemp[2,-2]*Etemp[3,-3]*Etemp[4,-4]  
    @tensor A_quadruple[:] := Jtemp[1,2,3,4]*Etemp[1,-1]*Etemp[2,-2]*Etemp[3,-3]*Etemp[4,-4]
    #


    Sx = [0 1;1 0]
    # initial tensor  
    H = 1/sqrt(2)*[1 1;1 -1]
    @tensor A[:] := A[1,2,3,4]*H[1,-1]*H[2,-2]*H[3,-3]*H[4,-4]
    @tensor A_single[:] := A_single_1[1,2,3,4]*H[1,-1]*H[2,-2]*H[3,-3]*H[4,-4]
   

    ###########################################################
    @tensor A_single[:] := A_single[-1,-3,2,1]*A[-2,-7,4,1]*A[-5,-4,2,3]*A[-6,-8,4,3]+
                            A[-1,-3,2,1]*A_single[-2,-7,4,1]*A[-5,-4,2,3]*A[-6,-8,4,3]+
                            A[-1,-3,2,1]*A[-2,-7,4,1]*A_single[-5,-4,2,3]*A[-6,-8,4,3]+
                            A[-1,-3,2,1]*A[-2,-7,4,1]*A[-5,-4,2,3]*A_single[-6,-8,4,3]
    @tensor A[:] := A[-1,-3,2,1]*A[-2,-7,4,1]*A[-5,-4,2,3]*A[-6,-8,4,3]
    A = reshape(A,4,4,4,4)
    A_single = reshape(A_single,4,4,4,4)
    A_single = A_single/4
    
    ################################### initial quantum number 
    Qnum = Array{Array{Float64}}(undef,4)
    Qnum[1] = Float64[0,1,1,0]
    Qnum[2] = Float64[0,1,1,0]
    Qnum[3] = Float64[0,1,1,0]
    Qnum[4] = Float64[0,1,1,0]

    Fidelity=[]
    for nStep in 1:20
        #nStep = 1
        print("RG $nStep: ")
        #global Qnum,A,A_single,Sx

        ########################################### initial wL 
        # environment for wL
        #@tensor  Env_wL[:] := A[-2,-1,5,4]*A[3,4,1,2]*A[-4,-3,5,6]*A[3,6,1,2];
        @tensor Env_wL[:] := A[-2,-1,13,14]*A[8,14,9,7]*A[2,1,13,5]*A[3,5,9,4]*
                                A[-4,-3,11,12]*A[8,12,10,7]*A[2,1,11,6]*A[3,6,10,4]
        sizeEnv_wL = size(Env_wL);
        
        # quantum number for wL
        qnumtemp = Merge(Qnum[2],Qnum[1]);
        # compute wL for truncation
        wL,qnum_new = compute_isometry_Z2(reshape(Env_wL,prod(sizeEnv_wL[1:2]),prod(sizeEnv_wL[3:4])),qnumtemp,qnumtemp,chiw_max);
        wL = reshape(wL,sizeEnv_wL[1],sizeEnv_wL[2],size(wL,2));
        Qnum_wL = Array{Array{Float64}}(undef,3);
        Qnum_wL[1] = Qnum[2];Qnum_wL[2] = Qnum[1];Qnum_wL[3] = qnum_new;


        ########################################### initial wR 
        # environment for wR
        #@tensor  Env_wR[:] := A[3,1,2,4]*A[-2,4,6,-1]*A[3,1,2,5]*A[-4,5,6,-3];
        @tensor Env_wR[:] := A[7,8,9,11]*A[-2,11,12,-1]*A[2,1,9,5]*A[3,5,12,4]*
                                A[2,1,10,6]*A[3,6,14,4]*A[7,8,10,13]*A[-4,13,14,-3]
        sizeEnv_wR = size(Env_wR);
        
        # quantum number for wR
        qnumtemp = Merge(Qnum[4],Qnum[1]);
        # compute wR for truncation
        wR,qnum_new = compute_isometry_Z2(reshape(Env_wR,prod(sizeEnv_wR[1:2]),prod(sizeEnv_wR[3:4])),qnumtemp,qnumtemp,chiw_max);
        wR = reshape(wR,sizeEnv_wR[1],sizeEnv_wR[2],size(wR,2));
        Qnum_wR = Array{Array{Float64,1},1}(undef,3);
        Qnum_wR[1] = Qnum[4];Qnum_wR[2] = Qnum[1];Qnum_wR[3] = qnum_new

        #########################################################################################
        #########################################################################################
        #########################################################################################
        # introduce vL,vR,sL and sR
        @tensor Env_vL[:] := Env_wL[1,2,3,4]*wL[1,2,-1]*wL[3,4,-2]
        vL,qnum_vL = compute_isometry_Z2(Env_vL,Qnum_wL[3],Qnum_wL[3],chis_max)
        Qnum_vL_temp = Array{Array{Float64,1},1}(undef,2)
        Qnum_vL_temp[1] = Qnum_wL[3];Qnum_vL_temp[2] = qnum_vL

        @tensor Env_vR[:] := Env_wR[1,2,3,4]*wR[1,2,-1]*wR[3,4,-2] 
        vR,qnum_vR = compute_isometry_Z2(Env_vR,Qnum_wR[3],Qnum_wR[3],chis_max)
        Qnum_vR_temp = Array{Array{Float64,1},1}(undef,2)
        Qnum_vR_temp[1] = Qnum_wR[3];Qnum_vR_temp[2] = qnum_vR


        #########################################################################################
        #########################################################################################
        #########################################################################################
        sL = deepcopy(vL);Qnum_sL = deepcopy(Qnum_vL_temp)
        sR = deepcopy(vR);Qnum_sR = deepcopy(Qnum_vR_temp)

        @tensor vL[:] := wL[-1,-2,1]*vL[1,-3]
        @tensor vR[:] := wR[-1,-2,1]*vR[1,-3]

        Qnum_vL = deepcopy(Qnum_wL);Qnum_vL[3] = Qnum_vL_temp[2]
        Qnum_vR = deepcopy(Qnum_wR);Qnum_vR[3] = Qnum_vR_temp[2]
        @tensor vA[:] := A[8,7,9,12]*A[10,12,13,11]*A[2,1,9,5]*A[3,5,13,4]*A[2,1,16,6]*A[3,6,20,4]*A[15,14,16,19]*
                        A[17,19,20,18]*wL[7,8,-1]*wL[14,15,-2]*wR[11,10,-3]*wR[18,17,-4]

        # initial disentangler
        sizeA = size(A)
        dis = zeros(sizeA[1]*sizeA[1],size(Qnum_wL[2],1)*size(Qnum_wR[2],1))
        qnum_in = Merge(Qnum[1],Qnum[1]);qnum_out = Merge(Qnum_wL[2],Qnum_wR[2])
        odd_in = findall(x->x==1,qnum_in);odd_out = findall(x->x==1,qnum_out)
        even_in = findall(x->x==0,qnum_in);even_out = findall(x->x==0,qnum_out)
        dis[odd_in,odd_out] = Matrix(1.0I,size(odd_in,1),size(odd_out,1))
        dis[even_in,even_out] = Matrix(1.0I,size(even_in,1),size(even_out,1))
        dis = reshape(dis,sizeA[1],sizeA[1],size(Qnum_wL[2],1),size(Qnum_wR[2],1))
        #
        #dis = reshape(Matrix(1.0I,sizeA[1]^2,sizeA[1]^2),sizeA[1],sizeA[1],sizeA[1],sizeA[1])
        Qnum_dis = Array{Array{Float64,1},1}(undef,4)                       # create disentangler qnumtum number
        Qnum_dis[1] = Qnum[1];Qnum_dis[2] = Qnum[1]
        Qnum_dis[3] = Qnum_wL[2];Qnum_dis[4] = Qnum_wR[2]


        println("chidis: ===============================: ",size(dis))
        println("chiA: ===============================: ",size(A))
        print("Initial truncation error: ")
        fidelity_initial = compute_fidelity_modified_large_env(A,wL,wR,vL,vR,sL,sR,dis,printdetail=true); tol = fidelity_initial/100
        print("\n")


        Env_c = deepcopy(dis)
        dis_c = deepcopy(dis)
        vL_c = deepcopy(vL); sL_c = deepcopy(sL)
        vR_c = deepcopy(vR); sR_c = deepcopy(sR)

        dis_initial = deepcopy(dis)
        vL_initial = deepcopy(vL); sL_initial = deepcopy(sL)
        vR_initial = deepcopy(vR); sR_initial = deepcopy(sR)

        UpdateLoop = 0
        for j in 1:UpdateLoop



            ################################## break if the fidelity is less than 1.0e-14
            tol = 1.0e-10
            beta = 0
            ##############################################################################
            ################################################ update implicit disentangler            
            # update sL
            @tensor P[:] := vA[-1,7,4,5]*wL[9,8,7]*wR[1,3,5]*vL[9,10,-2]*vR[1,2,6]*sR[4,6]*dis[8,3,10,2]
            @tensor B[:] := vA[-1,-2,1,2]*sR[1,3]*sR[2,3]
            sizeP = size(P);sizeB = size(B)
            P = reshape(P,sizeP[1],sizeP[2]);B = reshape(B,sizeB[1],sizeB[2])
            P_Qnum_in = Qnum_sL[1];P_Qnum_out = Qnum_sL[2]
            B_Qnum_in = Qnum_sL[1];B_Qnum_out = Qnum_sL[1]
            sL_temp = update_sM_Z2(P,B,P_Qnum_in,P_Qnum_out,B_Qnum_in,B_Qnum_out;tol=tol)
            sL_temp = reshape(sL_temp,sizeP...)
            sL = 0.01*beta*sL + (1-0.01*beta)*sL_temp


            # update sR
            @tensor P[:] := vA[4,6,-1,7]*wL[1,2,6]*wR[9,8,7]*vL[1,3,5]*vR[9,10,-2]*sL[4,5]*dis[2,8,3,10]
            @tensor B[:] := vA[1,3,-1,-2]*sL[1,2]*sL[3,2]
            sizeP = size(P);sizeB = size(B)
            P = reshape(P,sizeP[1],sizeP[2]);B = reshape(B,sizeB[1],sizeB[2])
            P_Qnum_in = Qnum_sR[1];P_Qnum_out = Qnum_sR[2]
            B_Qnum_in = Qnum_sR[1];B_Qnum_out = Qnum_sR[1]
            sR_temp = update_sM_Z2(P,B,P_Qnum_in,P_Qnum_out,B_Qnum_in,B_Qnum_out;tol=tol)
            sR_temp = reshape(sR_temp,sizeP...)
            sR = 0.01*beta*sR + (1-0.01*beta)*sR_temp



            ############################################################################################
            ############################################## update isometries            
            # update vL 
            @tensor Env_vL[:] := vA[9,8,4,6]*wL[-1,7,8]*wR[1,3,6]*vR[1,2,5]*sL[9,-3]*sR[4,5]*dis[7,3,-2,2]
            sizeEnv_vL = size(Env_vL)            
            Env_vL = reshape(Env_vL,prod(sizeEnv_vL[1:2]),sizeEnv_vL[3])
            Qnum_in = Merge(Qnum_vL[1],Qnum_vL[2]);Qnum_out = Qnum_vL[3]
            vL = update_vM_Z2(Env_vL,Qnum_in,Qnum_out)
            vL = reshape(vL,sizeEnv_vL)


            # update vR
            @tensor Env_vR[:] := vA[4,6,9,8]*wL[1,2,6]*wR[-1,7,8]*vL[1,3,5]*sL[4,5]*sR[9,-3]*dis[2,7,3,-2]
            sizeEnv_vR = size(Env_vR)
            Env_vR = reshape(Env_vR,prod(sizeEnv_vR[1:2]),sizeEnv_vR[3])
            Qnum_in = Merge(Qnum_vR[1],Qnum_vR[2]);Qnum_out = Qnum_vR[3]
            vR = update_vM_Z2(Env_vR,Qnum_in,Qnum_out)
            vR = reshape(vR,sizeEnv_vR)


            ############################################################################################
            ############################################################################################
            ############################################################################################


            # update dis
            @tensor Env_dis[:] := vA[6,7,2,3]*wL[5,-1,7]*wR[1,-2,3]*vL[5,-3,8]*vR[1,-4,4]*sL[6,8]*sR[2,4]
            sizeEnv_dis = size(Env_dis)
            Env_dis = reshape(Env_dis,prod(sizeEnv_dis[1:2]),prod(sizeEnv_dis[3:4]))
            Qnum_in = Merge(Qnum_dis[1],Qnum_dis[2]);Qnum_out = Merge(Qnum_dis[3],Qnum_dis[4])
            dis = update_vM_Z2(Env_dis,Qnum_in,Qnum_out)
            dis = reshape(dis,sizeEnv_dis)
           

            ##########################################   keep initial if updated fidelity is larger than initial
            #
            if j == UpdateLoop && nStep >= 1
                fidelity = compute_fidelity_modified_large_env(A,wL,wR,vL,vR,sL,sR,dis,printdetail=true)   
                print("\n")
                if abs(fidelity_initial) < abs(fidelity)
                println("=========================================================")
                println("                  isometries from initial ")
                println("=========================================================")
                dis = deepcopy(dis_initial)
                vL = deepcopy(vL_initial); sL = deepcopy(sL_initial)
                vR = deepcopy(vR_initial); sR = deepcopy(sR_initial)
                break       
                end
            end
            #

            ##########################     print fidelity for every 100 iterations
            if j%100 ==0
                #println("dif : ", norm(dis-dis_c)/norm(dis)," ",norm(vL-vL_c)/norm(vL)," ",norm(vR-vR_c)/norm(vR))
                #println("dif : ", norm(sL-sL_c)/norm(sL)," ",norm(sR-sR_c)/norm(sR))        
                #println("Env dif",norm(Env_dis-Env_c)/norm(Env_c))    
                print("Iteration : $j ")
                tol = compute_fidelity_modified_large_env(A,wL,wR,vL,vR,sL,sR,dis,printdetail=true) / 100  
                print("\n")



                @tensor Env_mL[:] := A[-2,-1,13,14]*A_single[8,14,9,7]*A[2,1,13,5]*A[3,5,9,4]*A[2,1,11,6]*A[3,6,10,4]*A[-4,-3,11,12]*A_single[8,12,10,7]
                sizeEnv_mL = size(Env_mL);
                qnumtemp = Merge(Qnum[2],Qnum[1]);
                mL,qnum_new = compute_isometry_Z2(reshape(Env_mL,prod(sizeEnv_mL[1:2]),prod(sizeEnv_mL[3:4])),qnumtemp,qnumtemp,chiw_max);
                mL = reshape(mL,sizeEnv_mL[1],sizeEnv_mL[2],size(mL,2));
                Qnum_mL = Array{Array{Float64}}(undef,3);
                Qnum_mL[1] = Qnum[2];Qnum_mL[2] = Qnum[1];Qnum_mL[3] = qnum_new;
                @tensor phiphi[:] := Env_mL[1,2,1,2]
                @tensor psiphi[:] := Env_mL[1,2,3,4]*mL[1,2,5]*mL[3,4,5]
                println("Truncation mL: ",(phiphi[1]-psiphi[1])/phiphi[1])


                @tensor Env_mR[:] := A[7,8,9,11]*A_single[-2,11,12,-1]*A[2,1,9,5]*A[3,5,12,4]*A[2,1,10,6]*A[3,6,14,4]*A[7,8,10,13]*A_single[-4,13,14,-3]
                sizeEnv_mR = size(Env_mR);
                # quantum number for wR
                qnumtemp = Merge(Qnum[4],Qnum[1]);
                # compute wR for truncation
                mR,qnum_new = compute_isometry_Z2(reshape(Env_mR,prod(sizeEnv_mR[1:2]),prod(sizeEnv_mR[3:4])),qnumtemp,qnumtemp,chiw_max);
                mR = reshape(mR,sizeEnv_mR[1],sizeEnv_mR[2],size(mR,2));
                Qnum_mR = Array{Array{Float64,1},1}(undef,3);
                Qnum_mR[1] = Qnum[4];Qnum_mR[2] = Qnum[1];Qnum_mR[3] = qnum_new
                @tensor phiphi[:] := Env_mR[1,2,1,2]
                @tensor psiphi[:] := Env_mR[1,2,3,4]*mR[1,2,5]*mR[3,4,5]
                println("Truncation mR: ",(phiphi[1]-psiphi[1])/phiphi[1])

                @tensor sL_temp[:] := wL[-1,-2,1]*sL[1,-3]
                @tensor phiphi[:] := A[8,7,12,11]*A_single[10,11,13,9]*A[2,1,12,5]*A[3,5,13,4]*A[2,1,20,6]*A[3,6,21,4]*A[15,14,20,18]*A_single[17,18,21,16]*
                                mL[7,8,19]*mL[14,15,19]*mR[9,10,22]*mR[16,17,22]
                @tensor psiphi[:] :=  A[15,14,16,19]*A_single[17,19,20,18]*A[2,1,16,5]*A[3,5,20,4]*A[2,1,9,6]*A[3,6,13,4]*A[8,7,9,12]*A_single[10,12,13,11]*
                        mL[14,15,27]*mL[21,22,27]*mR[18,17,33]*mR[29,30,33]*mL[7,8,28]*mL[24,25,28]*mR[11,10,32]*mR[29,31,32]*sL_temp[21,22,23]*vL[24,26,23]*dis[25,31,26,30]
                
                @tensor psipsi[:] :=  A[5,4,15,8]*A_single[6,8,16,7]*A[10,9,15,13]*A[11,13,16,12]*A[10,9,26,14]*A[11,14,27,12]*A[21,20,26,24]*A_single[22,24,27,23]*
                                mR[7,6,28]*mR[23,22,28]*mL[4,5,3]*mL[1,2,3]*mL[20,21,19]*mL[17,18,19]*sL_temp[1,2,25]*sL_temp[17,18,25]

                println("Fidelity is : ",(1-psiphi[1]^2/phiphi[1]/psipsi[1]))

            end


            ##################################  break if dis and sL and wL converge
            if abs(norm(dis-dis_c)/norm(dis)) <1.0e-10 && abs(norm(sL-sL_c)/norm(sL)) < 1.0e-10
               break
            end           

            Env_c = deepcopy(Env_dis)
            dis_c = deepcopy(dis)
            vL_c = deepcopy(vL); sL_c = deepcopy(sL)
            vR_c = deepcopy(vR); sR_c = deepcopy(sR)
            vL_single_c = deepcopy(vL); sL_single_c = deepcopy(sL)
            vR_single_c = deepcopy(vR); sR_single_c = deepcopy(sR)
         
        end


        ########### compute y,w for renormalization

        # compute sL and sR to be compatible with previous works
        @tensor sL[:] := wL[-1,-2,1]*sL[1,-3]
        @tensor sR[:] := wR[-1,-2,1]*sR[1,-3]

        y,w,Qnum_new = compute_projector_TNR_Z2(A,Qnum,vL,Qnum_vL,vR,Qnum_vR,sL,sR,chimax)
        y = reshape(y,sizeA[2],sizeA[2],size(y,2))
        w = reshape(w,size(vR,2),size(vL,2),size(w,2))
       
        
        
        ######   create new local and local impurity tensor 
        @tensor A_new[:] := A[13,12,16,17]*A[15,14,16,18]*A[1,7,5,2]*A[3,6,5,4]*sL[12,13,24]*vL[8,10,24]*sL[14,15,22]*vL[19,21,22]*sR[2,1,11]*vR[8,9,11]*
                sR[4,3,23]*vR[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3]

        @tensor A_single_new[:] := A_single[19,18,15,16]*A[13,12,15,14]*A[1,3,7,2]*A[4,6,7,5]*sL[12,13,20]*vL[9,10,20]*
                sR[2,1,21]*vR[18,17,21]*sR[5,4,11]*vR[9,8,11]*y[6,3,-2]*y[14,16,-4]*w[8,10,-3]*w[17,19,-1]+
                A[13,12,15,14]*A_single[19,18,15,16]*A[1,3,7,2]*A[4,6,7,5]*sL[12,13,21]*vL[9,10,21]*sR[2,1,11]*vR[9,8,11]*
                sR[5,4,20]*vR[18,17,20]*y[6,3,-2]*y[16,14,-4]*w[17,19,-3]*w[8,10,-1]+
                A[2,1,6,3]*A[5,4,6,7]*A_single[18,15,16,19]*A[12,14,16,13]*sL[1,2,21]*vL[19,17,21]*sL[4,5,11]*vL[9,10,11]*
                sR[13,12,20]*vR[9,8,20]*y[14,15,-2]*y[7,3,-4]*w[18,17,-1]*w[8,10,-3]+
                A[2,1,6,3]*A[5,4,6,7]*A[12,14,16,13]*A_single[18,15,16,19]*sL[1,2,11]*vL[10,8,11]*sL[4,5,21]*vL[19,17,21]*
                sR[13,12,20]*vR[10,9,20]*y[15,14,-2]*y[7,3,-4]*w[18,17,-3]*w[9,8,-1]
        A_single_new = A_single_new/4


        ###### update new tensor         
        Qnum = Qnum_new    
        A = deepcopy(A_new)
        A_single = deepcopy(A_single_new)
        check_gauge(A_single,Qnum)


        Result["logAnorm"] = log(maximum(A))+4*Result["logAnorm"];
        A_single = A_single/maximum(A)
        A = A/maximum(A)
        append!(Result["Amatrix"],[A])



        ##### compute free energy
        @tensor Z[:] := A[1,2,1,2]
        fe = -1*Temperature*(log(Z[1])+Result["logAnorm"])/4^(nStep+1);
        append!(Result["FreeEnergy"],fe)


        ####  if  A has one index with dimension 1, break
        check = prod(size(A) .-1)
        if check ==0 
            break
        end


        ######  compute Magnetization
        @tensor A_temp[:] := A[-1,1,-2,1]
        Rlt_temp = eigsolve(A_temp)

        @tensor A_single_temp[:] := A_single[-1,1,-2,1]
        Rlt_single_temp = eigsolve(A_single_temp)
        append!(Result["Magnetization"],(Rlt_single_temp[1]/Rlt_temp[1][1])[1])
    
    end

    append!(FreeEnergy,Result["FreeEnergy"][end])
    append!(freeenergy,[Result["FreeEnergy"]])
    append!(Magnetization,Result["Magnetization"][end])

end
x = 1


