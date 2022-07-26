#
#
# TNR with Z2 symetry + horizontal symmetry
#
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
for chimax in 6:1:6
    #local Temperature
    #chimax = 8
    chiw_max = 2*chimax
    chidis_max = chimax
    chis_max = chimax
    #relT = 1.0#
    relT = 0.9994
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
   
    #
    Qnum = Array{Array{Float64}}(undef,4)
    Qnum[1] = Float64[0,1]
    Qnum[2] = Float64[0,1]
    Qnum[3] = Float64[0,1]
    Qnum[4] = Float64[0,1]
    #


    #
    @tensor A_single[:] := A_single[-1,-3,2,1]*A[-2,-7,4,1]*A[-5,-4,2,3]*A[-6,-8,4,3]+
                            A[-1,-3,2,1]*A_single[-2,-7,4,1]*A[-5,-4,2,3]*A[-6,-8,4,3]+
                            A[-1,-3,2,1]*A[-2,-7,4,1]*A_single[-5,-4,2,3]*A[-6,-8,4,3]+
                            A[-1,-3,2,1]*A[-2,-7,4,1]*A[-5,-4,2,3]*A_single[-6,-8,4,3]
    @tensor A[:] := A[-1,-3,2,1]*A[-2,-7,4,1]*A[-5,-4,2,3]*A[-6,-8,4,3]
    A = reshape(A,4,4,4,4)
    A_single = reshape(A_single,4,4,4,4)
    A_single = A_single/4
    #
    # initial quantum number 
    Qnum = Array{Array{Float64}}(undef,4)
    Qnum[1] = Float64[0,1,1,0]
    Qnum[2] = Float64[0,1,1,0]
    Qnum[3] = Float64[0,1,1,0]
    Qnum[4] = Float64[0,1,1,0]
    #

    Fidelity=[]
    for nStep in 1:20
    #nStep = 1
        print("RG $nStep: ")
        #global Qnum,A,A_single,Sx

        ########################################### initial wL 
        # environment for vL
        @tensor  Env_wL[:] := A[-2,-1,5,4]*A[3,4,1,2]*A[-4,-3,5,6]*A[3,6,1,2];
        sizeEnv_wL = size(Env_wL);
        Env_wL = reshape(Env_wL,prod(sizeEnv_wL[1:2]),prod(sizeEnv_wL[3:4]));
        
        # quantum number for vL
        qnumtemp = merge(Qnum[2],Qnum[1]);
        # compute vL for truncation
        wL,qnum_new = compute_isometry_Z2(Env_wL,qnumtemp,qnumtemp,chiw_max);
        wL = reshape(wL,sizeEnv_wL[1],sizeEnv_wL[2],size(wL,2));
        Qnum_wL = Array{Array{Float64}}(undef,3);
        Qnum_wL[1] = Qnum[2];Qnum_wL[2] = Qnum[1];Qnum_wL[3] = qnum_new;


        #
        ########################################### initial wL_single 
        # environment for vL_single
        @tensor  Env_wL_single[:] := A_single[-2,-1,5,4]*A[3,4,1,2]*A_single[-4,-3,5,6]*A[3,6,1,2];
        sizeEnv_wL_single = size(Env_wL_single);
        Env_wL_single = reshape(Env_wL_single,prod(sizeEnv_wL_single[1:2]),prod(sizeEnv_wL_single[3:4]));
        
        # quantum number for vL_single
        qnumtemp = merge(Qnum[2],Qnum[1]);

        # compute vL_single for truncation
        wL_single,qnum_new = compute_isometry_Z2(Env_wL_single,qnumtemp,qnumtemp,chiw_max);
        wL_single = reshape(wL_single,sizeEnv_wL_single[1],sizeEnv_wL_single[2],size(wL_single,2));
        Qnum_wL_single = Array{Array{Float64}}(undef,3);
        Qnum_wL_single[1] = Qnum[2];Qnum_wL_single[2] = Qnum[1];Qnum_wL_single[3] = qnum_new;
        #


        ########################################### initial wR 
        # environment for v
        @tensor  Env_wR[:] := A[3,1,2,4]*A[-2,4,6,-1]*A[3,1,2,5]*A[-4,5,6,-3];
        sizeEnv_wR = size(Env_wR);
        Env_wR = reshape(Env_wR,prod(sizeEnv_wR[1:2]),prod(sizeEnv_wR[3:4]));
        
        # quantum number for v
        qnumtemp = merge(Qnum[4],Qnum[1]);

        # compute v for truncation
        wR,qnum_new = compute_isometry_Z2(Env_wR,qnumtemp,qnumtemp,chiw_max);
        wR = reshape(wR,sizeEnv_wR[1],sizeEnv_wR[2],size(wR,2));
        Qnum_wR = Array{Array{Float64,1},1}(undef,3);
        Qnum_wR[1] = Qnum[4];Qnum_wR[2] = Qnum[1];Qnum_wR[3] = qnum_new


        ######################################## initial wR_single 
        # environment for vR_single
        @tensor  Env_wR_single[:] := A[3,1,2,4]*A_single[-2,4,6,-1]*A[3,1,2,5]*A_single[-4,5,6,-3];
        sizeEnv_wR_single = size(Env_wR_single);
        Env_wR_single = reshape(Env_wR_single,prod(sizeEnv_wR_single[1:2]),prod(sizeEnv_wR_single[3:4]));
        
        # quantum number for v
        qnumtemp = merge(Qnum[4],Qnum[1]);

        # compute v for truncation
        wR_single,qnum_new = compute_isometry_Z2(Env_wR_single,qnumtemp,qnumtemp,chiw_max);
        wR_single = reshape(wR_single,sizeEnv_wR_single[1],sizeEnv_wR_single[2],size(wR_single,2));
        Qnum_wR_single = Array{Array{Float64,1},1}(undef,3);
        Qnum_wR_single[1] = Qnum[4];Qnum_wR_single[2] = Qnum[1];Qnum_wR_single[3] = qnum_new
        #


        #########################################################################################
        #########################################################################################
        #########################################################################################
        # introduce vL,vR,sL and sR 

        @tensor Env_vL[:] := A[5,4,9,6]*A[3,6,1,2]*A[8,7,9,10]*A[3,10,1,2]*wL[4,5,-1]*wL[7,8,-2]
        vL,qnum_vL = compute_isometry_Z2(Env_vL,Qnum_wL[3],Qnum_wL[3],chis_max)
        Qnum_vL_temp = Array{Array{Float64,1},1}(undef,2)
        Qnum_vL_temp[1] = Qnum_wL[3];Qnum_vL_temp[2] = qnum_vL

        @tensor Env_vR[:] := A[1,2,3,6]*A[1,2,3,9]*A[4,6,10,5]*A[7,9,10,8]*wR[5,4,-1]*wR[8,7,-2]
        vR,qnum_vR = compute_isometry_Z2(Env_vR,Qnum_wR[3],Qnum_wR[3],chis_max)
        Qnum_vR_temp = Array{Array{Float64,1},1}(undef,2)
        Qnum_vR_temp[1] = Qnum_wR[3];Qnum_vR_temp[2] = qnum_vR


        @tensor Env_vL_single[:] := A_single[5,4,9,6]*A[3,6,1,2]*A_single[8,7,9,10]*A[3,10,1,2]*wL_single[4,5,-1]*wL_single[7,8,-2]
        vL_single,qnum_vL_single = compute_isometry_Z2(Env_vL_single,Qnum_wL_single[3],Qnum_wL_single[3],chiv_max)
        Qnum_vL_single_temp = Array{Array{Float64,1},1}(undef,2)
        Qnum_vL_single_temp[1] = Qnum_wL_single[3];Qnum_vL_single_temp[2] = qnum_vL_single

        @tensor Env_vR_single[:] := A[1,2,3,6]*A[1,2,3,9]*A_single[4,6,10,5]*A_single[7,9,10,8]*wR_single[5,4,-1]*wR_single[8,7,-2]
        vR_single,qnum_vR_single = compute_isometry_Z2(Env_vR_single,Qnum_wR_single[3],Qnum_wR_single[3],chiv_max)
        Qnum_vR_single_temp = Array{Array{Float64,1},1}(undef,2)
        Qnum_vR_single_temp[1] = Qnum_wR_single[3];Qnum_vR_single_temp[2] = qnum_vR_single
        #
        #########################################################################################
        #########################################################################################
        #########################################################################################


        sL = deepcopy(vL);Qnum_sL = deepcopy(Qnum_vL_temp)
        sR = deepcopy(vR);Qnum_sR = deepcopy(Qnum_vR_temp)
        sL_single = deepcopy(vL_single); Qnum_sL_single = deepcopy(Qnum_vL_single_temp)
        sR_single = deepcopy(vR_single); Qnum_sR_single = deepcopy(Qnum_vR_single_temp)

        @tensor vL[:] := wL[-1,-2,1]*vL[1,-3]
        @tensor vR[:] := wR[-1,-2,1]*vR[1,-3]
        @tensor vL_single[:] := wL_single[-1,-2,1]*vL_single[1,-3]
        @tensor vR_single[:] := wR_single[-1,-2,1]*vR_single[1,-3]

        Qnum_vL = deepcopy(Qnum_wL);Qnum_vL[3] = Qnum_vL_temp[2]
        Qnum_vR = deepcopy(Qnum_wR);Qnum_vR[3] = Qnum_vR_temp[2]
        Qnum_vL_single = deepcopy(Qnum_wL_single); Qnum_vL_single[3] = Qnum_vL_single_temp[2]
        Qnum_vR_single = deepcopy(Qnum_wR_single); Qnum_vR_single[3] = Qnum_vR_single_temp[2]

        @tensor vA[:] := A[2,1,11,5]*A[3,5,12,4]*A[7,6,11,10]*A[8,10,12,9]*wL[1,2,-1]*wL[6,7,-2]*wR[4,3,-3]*wR[9,8,-4]
        @tensor vA_single_L[:] := A_single[2,1,11,5]*A[3,5,12,4]*A_single[7,6,11,10]*A[8,10,12,9]*wL_single[1,2,-1]*wL_single[6,7,-2]*wR[4,3,-3]*wR[9,8,-4]
        @tensor vA_single_R[:] := A[2,1,11,5]*A_single[3,5,12,4]*A[7,6,11,10]*A_single[8,10,12,9]*wL[1,2,-1]*wL[6,7,-2]*wR_single[4,3,-3]*wR_single[9,8,-4]
        
        # initial disentangler
        sizeA = size(A)
        #
        dis = zeros(sizeA[1]*sizeA[1],size(Qnum_wL[2],1)*size(Qnum_wR[2],1))
        qnum_in = merge(Qnum[1],Qnum[1]);qnum_out = merge(Qnum_wL[2],Qnum_wR[2])
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
        println("Initial truncation error ")
        fidelity_initial = compute_fidelity_modified(A,wL,wR,vL,vR,sL,sR,dis,printdetail=true)  
        compute_fidelity_impurity_modified(A,A_single,wL,wR,wL_single,wR_single,vL,vR,vL_single,vR_single,sL,sR,sL_single,sR_single,dis;printdetail=true)

        Env_c = deepcopy(dis)
        dis_c = deepcopy(dis)
        vL_c = deepcopy(vL); sL_c = deepcopy(sL)
        vR_c = deepcopy(vR); sR_c = deepcopy(sR)
        vL_single_c = deepcopy(vL_single); sL_single_c = deepcopy(sL_single)
        vR_single_c = deepcopy(vR_single); sR_single_c = deepcopy(sR_single)


        dis_initial = deepcopy(dis)
        vL_initial = deepcopy(vL); sL_initial = deepcopy(sL)
        vR_initial = deepcopy(vR); sR_initial = deepcopy(sR)
        vL_single_initial = deepcopy(vL_single); sL_single_initial = deepcopy(sL_single)
        vR_single_initial = deepcopy(vR_single); sR_single_initial = deepcopy(sR_single)

        UpdateLoop = 1000
        for j in 1:UpdateLoop

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
            sL_temp = update_sM_Z2(P,B,P_Qnum_in,P_Qnum_out,B_Qnum_in,B_Qnum_out)
            sL_temp = reshape(sL_temp,sizeP...)
            sL = 0.01*beta*sL + (1-0.01*beta)*sL_temp


            # update sR 
            @tensor P[:] := vA[4,6,-1,7]*wL[1,2,6]*wR[9,8,7]*vL[1,3,5]*vR[9,10,-2]*sL[4,5]*dis[2,8,3,10]
            @tensor B[:] := vA[1,3,-1,-2]*sL[1,2]*sL[3,2]
            sizeP = size(P);sizeB = size(B)
            P = reshape(P,sizeP[1],sizeP[2]);B = reshape(B,sizeB[1],sizeB[2])
            P_Qnum_in = Qnum_sR[1];P_Qnum_out = Qnum_sR[2]
            B_Qnum_in = Qnum_sR[1];B_Qnum_out = Qnum_sR[1]
            sR_temp = update_sM_Z2(P,B,P_Qnum_in,P_Qnum_out,B_Qnum_in,B_Qnum_out)
            sR_temp = reshape(sR_temp,sizeP...)
            sR = 0.01*beta*sR + (1-0.01*beta)*sR_temp




            ###########################################################################################
            ###########################################################################################
            # update sL_single
            @tensor P_single[:] := vA_single_L[-1,7,4,5]*wL_single[9,8,7]*wR[1,3,5]*vL_single[9,10,-2]*vR[1,2,6]*sR[4,6]*dis[8,3,10,2]
            @tensor B_single[:] := vA_single_L[-1,-2,1,2]*sR[1,3]*sR[2,3]
            sizeP = size(P_single);sizeB = size(B_single)
            P_single = reshape(P_single,sizeP[1],sizeP[2]);B_single = reshape(B_single,sizeB[1],sizeB[2])
            P_Qnum_in = Qnum_sL_single[1];P_Qnum_out = Qnum_sL_single[2]
            B_Qnum_in = Qnum_sL_single[1];B_Qnum_out = Qnum_sL_single[1]

            sL_single_temp = update_sM_Z2(P_single,B_single,P_Qnum_in,P_Qnum_out,B_Qnum_in,B_Qnum_out)
            sL_single_temp = reshape(sL_single_temp,sizeP...)
            sL_single = 0.01*beta*sL_single + (1-0.01*beta)*sL_single_temp
 

            # update sR_single 
            @tensor P_single[:] := vA_single_R[4,6,-1,7]*wL[1,2,6]*wR_single[9,8,7]*vL[1,3,5]*vR_single[9,10,-2]*sL[4,5]*dis[2,8,3,10]
            @tensor B_single[:] := vA_single_R[1,3,-1,-2]*sL[1,2]*sL[3,2]
            sizeP = size(P_single);sizeB = size(B_single)
            P_single = reshape(P_single,sizeP[1],sizeP[2]);B_single = reshape(B_single,sizeB[1],sizeB[2])
            P_Qnum_in = Qnum_sR_single[1];P_Qnum_out = Qnum_sR_single[2]
            B_Qnum_in = Qnum_sR_single[1];B_Qnum_out = Qnum_sR_single[1]

            sR_single_temp = update_sM_Z2(P_single,B_single,P_Qnum_in,P_Qnum_out,B_Qnum_in,B_Qnum_out)
            sR_single_temp = reshape(sR_single_temp,sizeP...)
            sR_single = 0.01*beta*sR_single + (1-0.01*beta)*sR_single_temp
 

            ############################################################################################
            ############################################## update isometries            
            # update vL 
            @tensor Env_vL_1[:] := vA[9,8,4,6]*wL[-1,7,8]*wR[1,3,6]*vR[1,2,5]*sL[9,-3]*sR[4,5]*dis[7,3,-2,2]
            @tensor Env_vL_2[:] := vA_single_R[9,8,4,6]*wL[-1,7,8]*wR_single[1,3,6]*vR_single[1,2,5]*sL[9,-3]*sR_single[4,5]*dis[7,3,-2,2]
            Env_vL = Env_vL_1 + Env_vL_2
            sizeEnv_vL = size(Env_vL)            
            Env_vL = reshape(Env_vL,prod(sizeEnv_vL[1:2]),sizeEnv_vL[3])
            Qnum_in = merge(Qnum_vL[1],Qnum_vL[2]);Qnum_out = Qnum_vL[3]
            vL = update_vM_Z2(Env_vL,Qnum_in,Qnum_out)
            vL = reshape(vL,sizeEnv_vL)

            # update vR
            @tensor Env_vR_1[:] := vA[4,6,9,8]*wL[1,2,6]*wR[-1,7,8]*vL[1,3,5]*sL[4,5]*sR[9,-3]*dis[2,7,3,-2]
            @tensor Env_vR_2[:] := vA_single_L[4,6,9,8]*wL_single[1,2,6]*wR[-1,7,8]*vL_single[1,3,5]*sL_single[4,5]*sR[9,-3]*dis[2,7,3,-2]
            Env_vR = Env_vR_1 + Env_vR_2
            sizeEnv_vR = size(Env_vR)
            Env_vR = reshape(Env_vR,prod(sizeEnv_vR[1:2]),sizeEnv_vR[3])
            Qnum_in = merge(Qnum_vR[1],Qnum_vR[2]);Qnum_out = Qnum_vR[3]
            vR = update_vM_Z2(Env_vR,Qnum_in,Qnum_out)
            vR = reshape(vR,sizeEnv_vR)



            ############################################################################################
            ############################################################################################
            # update vL_single
            @tensor Env_vL_single[:] := vA_single_L[9,8,4,6]*wL_single[-1,7,8]*wR[1,3,6]*vR[1,2,5]*sL_single[9,-3]*sR[4,5]*dis[7,3,-2,2]
            sizeEnv_vL_single = size(Env_vL_single)            
            Env_vL_single = reshape(Env_vL_single,prod(sizeEnv_vL_single[1:2]),sizeEnv_vL_single[3])
            Qnum_in = merge(Qnum_vL_single[1],Qnum_vL_single[2]);Qnum_out = Qnum_vL_single[3]
            vL_single = update_vM_Z2(Env_vL_single,Qnum_in,Qnum_out)
            vL_single = reshape(vL_single,sizeEnv_vL_single)


            # update vR_single
            @tensor Env_vR_single[:] := vA_single_R[4,6,9,8]*wL[1,2,6]*wR_single[-1,7,8]*vL[1,3,5]*sL[4,5]*sR_single[9,-3]*dis[2,7,3,-2]
            sizeEnv_vR_single = size(Env_vR_single)
            Env_vR_single = reshape(Env_vR_single,prod(sizeEnv_vR_single[1:2]),sizeEnv_vR_single[3])
            Qnum_in = merge(Qnum_vR_single[1],Qnum_vR_single[2]);Qnum_out = Qnum_vR_single[3]
            vR_single = update_vM_Z2(Env_vR_single,Qnum_in,Qnum_out)
            vR_single = reshape(vR_single,sizeEnv_vR_single)



            # update dis
            @tensor Env_dis_1[:] := vA[6,7,2,3]*wL[5,-1,7]*wR[1,-2,3]*vL[5,-3,8]*vR[1,-4,4]*sL[6,8]*sR[2,4]
            @tensor Env_dis_2[:] := vA_single_L[6,7,2,3]*wL_single[5,-1,7]*wR[1,-2,3]*vL_single[5,-3,8]*vR[1,-4,4]*sL_single[6,8]*sR[2,4]
            @tensor Env_dis_3[:] := vA_single_R[6,7,2,3]*wL[5,-1,7]*wR_single[1,-2,3]*vL[5,-3,8]*vR_single[1,-4,4]*sL[6,8]*sR_single[2,4]
            alpha = 1.0; gamma = 1.0
            Env_dis = gamma*Env_dis_1+alpha*Env_dis_2+alpha*Env_dis_3
            sizeEnv_dis = size(Env_dis)
            Env_dis = reshape(Env_dis,prod(sizeEnv_dis[1:2]),prod(sizeEnv_dis[3:4]))
            Qnum_in = merge(Qnum_dis[1],Qnum_dis[2]);Qnum_out = merge(Qnum_dis[3],Qnum_dis[4])
            dis = update_vM_Z2(Env_dis,Qnum_in,Qnum_out)
            dis = reshape(dis,sizeEnv_dis)

   
            fidelity = compute_fidelity_modified(A,wL,wR,vL,vR,sL,sR,dis,printdetail=false)
            if abs(fidelity) < 1.0e-14
                break
            end


            #
            if abs(fidelity_initial) < abs(fidelity) && j == UpdateLoop
                dis = deepcopy(dis_initial)
                vL = deepcopy(vL_initial); sL = deepcopy(sL_initial)
                vR = deepcopy(vR_initial); sR = deepcopy(sR_initial)
                vL_single = deepcopy(vL_single_initial); sL_single = deepcopy(sL_single_initial)
                vR_single = deepcopy(vR_single_initial); sR_single = deepcopy(sR_single_initial)             
                break
            end
            #

            if j%100 ==0
                println("dif : ", norm(dis-dis_c)/norm(dis)," ",norm(vL-vL_c)/norm(vL)," ",norm(vR-vR_c)/norm(vR))
                println("dif : ", norm(sL-sL_c)/norm(sL)," ",norm(sR-sR_c)/norm(sR))        
                println("Env dif",norm(Env_dis-Env_c)/norm(Env_c))    
                compute_fidelity_modified(A,wL,wR,vL,vR,sL,sR,dis,printdetail=true)   
                compute_fidelity_impurity_modified(A,A_single,wL,wR,wL_single,wR_single,vL,vR,vL_single,vR_single,sL,sR,sL_single,sR_single,dis;printdetail=true)
            end
            
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
        @tensor sL_single[:] := wL_single[-1,-2,1]*sL_single[1,-3]        
        @tensor sR_single[:] := wR_single[-1,-2,1]*sR_single[1,-3]

        y,w,Qnum_new = compute_projector_TNR_Z2(A,Qnum,vL,Qnum_vL,vR,Qnum_vR,sL,sR,chimax)
        y = reshape(y,sizeA[2],sizeA[2],size(y,2))
        w = reshape(w,size(vR,2),size(vL,2),size(w,2))
       
        
        
        ######   create new local and local impurity tensor 
        @tensor A_new[:] := A[13,12,16,17]*A[15,14,16,18]*A[1,7,5,2]*A[3,6,5,4]*sL[12,13,24]*vL[8,10,24]*sL[14,15,22]*vL[19,21,22]*sR[2,1,11]*vR[8,9,11]*
                sR[4,3,23]*vR[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3]

        #
        @tensor A_single_new[:] := A_single[13,12,16,17]*A[15,14,16,18]*A[1,7,5,2]*A[3,6,5,4]*sL_single[12,13,24]*vL_single[8,10,24]*sL[14,15,22]*vL[19,21,22]*sR[2,1,11]*vR[8,9,11]*
                sR[4,3,23]*vR[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3]+
                A[13,12,16,17]*A_single[15,14,16,18]*A[1,7,5,2]*A[3,6,5,4]*sL[12,13,24]*vL[8,10,24]*sL_single[14,15,22]*vL_single[19,21,22]*sR[2,1,11]*vR[8,9,11]*
                sR[4,3,23]*vR[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3]+
                A[13,12,16,17]*A[15,14,16,18]*A_single[1,7,5,2]*A[3,6,5,4]*sL[12,13,24]*vL[8,10,24]*sL[14,15,22]*vL[19,21,22]*sR_single[2,1,11]*vR_single[8,9,11]*
                sR[4,3,23]*vR[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3]+
                A[13,12,16,17]*A[15,14,16,18]*A[1,7,5,2]*A_single[3,6,5,4]*sL[12,13,24]*vL[8,10,24]*sL[14,15,22]*vL[19,21,22]*sR[2,1,11]*vR[8,9,11]*
                sR_single[4,3,23]*vR_single[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3]
        A_single_new = A_single_new/4
        


        ###### update new tensor         
        Qnum = Qnum_new    
        A = deepcopy(A_new)
        A_single = deepcopy(A_single_new)

        Result["logAnorm"] = log(maximum(A))+4*Result["logAnorm"];
        A_single = A_single/maximum(A)
        A = A/maximum(A)
        append!(Result["Amatrix"],[A])



        ##### compute free energy
        @tensor Z[:] := A[1,2,1,2]
        fe = -1*Temperature*(log(Z[1])+Result["logAnorm"])/4^(nStep+1);
        append!(Result["FreeEnergy"],fe)


        ####  if  A has one index with dimension 1, break
        print(" size(A): ",size(A),"\n")
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
        println(" Magnetization: ",(abs.((Rlt_single_temp[1]/Rlt_temp[1][1])[1]) -M)/M)
    
    end

    append!(FreeEnergy,Result["FreeEnergy"][end])
    append!(freeenergy,[Result["FreeEnergy"]])
    append!(Magnetization,Result["Magnetization"][end])

end
x = 1


