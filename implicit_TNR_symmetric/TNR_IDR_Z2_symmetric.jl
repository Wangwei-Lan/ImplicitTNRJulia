#
#
# TNR with Z2 symetry + horizontal symmetry
#  NOTE: looks like larger environment works worse than smaller ones.??? Why
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
for chimax in 3:1:7
    #local Temperature
    #chimax = 7
    chiw_max = 3*chimax
    chiv_max = chimax
    chidis_max = chimax
    #relT = 1.0#
    relT = 0.9998
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
    # initial quantum number 
    Qnum = Array{Array{Float64}}(undef,4)
    Qnum[1] = Float64[0,1]
    Qnum[2] = Float64[0,1]
    Qnum[3] = Float64[0,1]
    Qnum[4] = Float64[0,1]



    for nStep in 1:20
    #nStep = 1
        print("RG $nStep:  \n")
        #global Qnum,A,A_single,Sx

        ########################################### initial vL 
        # environment for vL
        @tensor  Env_vL[:] := A[-2,-1,5,4]*A[3,4,1,2]*A[-4,-3,5,6]*A[3,6,1,2];
        #@tensor Env_vL[:] := A[-2,-1,13,14]*A[8,14,9,7]*A[2,1,13,5]*A[3,5,9,4]*
        #        A[-4,-3,11,12]*A[8,12,10,7]*A[2,1,11,6]*A[3,6,10,4]


        sizeEnv_vL = size(Env_vL);
        Env_vL = reshape(Env_vL,prod(sizeEnv_vL[1:2]),prod(sizeEnv_vL[3:4]));
        
        # quantum number for vL
        qnumtemp = Merge(Qnum[2],Qnum[1]);
        # compute vL for truncation
        vL,qnum_vL = compute_isometry_Z2(Env_vL,qnumtemp,qnumtemp,chiv_max);
        wL,qnum_wL = compute_isometry_Z2(Env_vL,qnumtemp,qnumtemp,chiw_max);
        vL = reshape(vL,sizeEnv_vL[1],sizeEnv_vL[2],size(vL,2));
        wL = reshape(wL,sizeEnv_vL[1],sizeEnv_vL[2],size(wL,2));
        sL = deepcopy(vL);
        Qnum_sL = Array{Array{Float64}}(undef,3);
        Qnum_sL[1] = Qnum[2];Qnum_sL[2] = Qnum[1];Qnum_sL[3] = qnum_vL;


        #
        ########################################### initial vL_single 
        # environment for vL_single
        @tensor  Env_vL_single[:] := A_single[-2,-1,5,4]*A[3,4,1,2]*A_single[-4,-3,5,6]*A[3,6,1,2];
        sizeEnv_vL_single = size(Env_vL_single);
        Env_vL_single = reshape(Env_vL_single,prod(sizeEnv_vL_single[1:2]),prod(sizeEnv_vL_single[3:4]));
        
        # quantum number for vL_single
        qnumtemp = Merge(Qnum[2],Qnum[1]);

        # compute vL_single for truncation
        vL_single,qnum_new = compute_isometry_Z2(Env_vL_single,qnumtemp,qnumtemp,chiv_max);
        vL_single = reshape(vL_single,sizeEnv_vL_single[1],sizeEnv_vL_single[2],size(vL_single,2));
        sL_single = deepcopy(vL_single)
        Qnum_sL_single = Array{Array{Float64}}(undef,3);
        Qnum_sL_single[1] = Qnum[2];Qnum_sL_single[2] = Qnum[1];Qnum_sL_single[3] = qnum_new;
        Qnum_vL_single = deepcopy(Qnum_sL_single);
        
        #=
        #@tensor vL_single[:] := vL_single[-1,1,-3]*wL[1,-2]
        Qnum_vL_single[2] = qnum_wL
        =#





        ########################################### initial vR 
        # environment for v
        @tensor  Env_vR[:] := A[3,1,2,4]*A[-2,4,6,-1]*A[3,1,2,5]*A[-4,5,6,-3];
        #@tensor Env_vR[:] := A[7,8,9,11]*A[-2,11,12,-1]*A[2,1,9,5]*A[3,5,12,4]*
        #                A[2,1,10,6]*A[3,6,14,4]*A[7,8,10,13]*A[-4,13,14,-3]
        sizeEnv_vR = size(Env_vR);
        Env_vR = reshape(Env_vR,prod(sizeEnv_vR[1:2]),prod(sizeEnv_vR[3:4]));
        
        # quantum number for v
        qnumtemp = Merge(Qnum[4],Qnum[1]);

        # compute v for truncation
        vR,qnum_vR = compute_isometry_Z2(Env_vR,qnumtemp,qnumtemp,chiv_max);
        wR,qnum_wR = compute_isometry_Z2(Env_vR,qnumtemp,qnumtemp,chiw_max);
        vR = reshape(vR,sizeEnv_vR[1],sizeEnv_vR[2],size(vR,2));
        wR = reshape(wR,sizeEnv_vR[1],sizeEnv_vR[2],size(wR,2));
        sR = deepcopy(vR);
        Qnum_sR = Array{Array{Float64,1},1}(undef,3);
        Qnum_sR[1] = Qnum[4];Qnum_sR[2] = Qnum[1];Qnum_sR[3] = qnum_vR

        Qnum_vL = deepcopy(Qnum_sL)
        Qnum_vR = deepcopy(Qnum_sR)


        #@tensor qdA_w[:] := A[7,6,-1,10]*A[8,10,-2,9]*A[7,6,12,11]*A[8,11,13,9]*A[2,1,12,5]*A[4,5,13,3]*wL[1,2,-3]*wR[3,4,-4]
        #@tensor qdA[:] := A[2,1,-1,5]*A[3,5,-2,4]*A[2,1,-3,6]*A[3,6,-4,4]

        #
        # environment for vR_single
        @tensor  Env_vR_single[:] := A[3,1,2,4]*A_single[-2,4,6,-1]*A[3,1,2,5]*A_single[-4,5,6,-3];
        sizeEnv_vR_single = size(Env_vR_single);
        Env_vR_single = reshape(Env_vR_single,prod(sizeEnv_vR_single[1:2]),prod(sizeEnv_vR_single[3:4]));
        
        # quantum number for v
        qnumtemp = Merge(Qnum[4],Qnum[1]);

        # compute v for truncation
        vR_single,qnum_new = compute_isometry_Z2(Env_vR_single,qnumtemp,qnumtemp,chiv_max);
        vR_single = reshape(vR_single,sizeEnv_vR_single[1],sizeEnv_vR_single[2],size(vR_single,2));
        sR_single = deepcopy(vR_single);
        Qnum_sR_single = Array{Array{Float64,1},1}(undef,3);
        Qnum_sR_single[1] = Qnum[4];Qnum_sR_single[2] = Qnum[1];Qnum_sR_single[3] = qnum_new
        Qnum_vR_single = deepcopy(Qnum_sR_single)                           # create vR_single Qnumtum Number
        
        #=
        @tensor vR_single[:] := vR_single[-1,1,-3]*wR[1,-2]             
        Qnum_vR_single[2] = qnum_wR
        =#


        # initial disentangler
        sizeA = size(A)

        #=
        dis = zeros(sizeA[1]*sizeA[1],size(qnum_vL,1)*size(qnum_vR,1))
        qnum_in = Merge(Qnum[1],Qnum[1]);qnum_out = Merge(qnum_vL,qnum_vR)
        odd_in = findall(x->x==1,qnum_in);odd_out = findall(x->x==1,qnum_out)
        even_in = findall(x->x==0,qnum_in);even_out = findall(x->x==0,qnum_out)
        dis[odd_in,odd_out] = Matrix(1.0I,size(odd_in,1),size(odd_out,1))
        dis[even_in,even_out] = Matrix(1.0I,size(even_in,1),size(even_out,1))
        dis = reshape(dis,sizeA[1],sizeA[1],size(qnum_vL,1),size(qnum_vR,1))
        =#
        dis = reshape(Matrix(1.0I,sizeA[1]^2,sizeA[1]^2),sizeA[1],sizeA[1],sizeA[1],sizeA[1])
        Qnum_dis = Array{Array{Float64,1},1}(undef,4)                       # create disentangler qnumtum number
        Qnum_dis[1] = Qnum[1];Qnum_dis[2] = Qnum[1]
        Qnum_dis[3] = Qnum[1];Qnum_dis[4] = Qnum[1]
        print(" Initial Truncation Error  : ")
        compute_fidelity(A,vL,vR,sL,sR,dis,printdetail=true)   
        print("\n")
        dis_c = deepcopy(dis);Env_c = deepcopy(dis)
        vL_c = deepcopy(vL); sL_c = deepcopy(sL)
        vR_c = deepcopy(vR); sR_c = deepcopy(sR)
        vL_single_c = deepcopy(vL); sL_single_c = deepcopy(sL)
        vR_single_c = deepcopy(vR); sR_single_c = deepcopy(sR)
        
        UpdateLoop = 1500
        for j in 1:UpdateLoop

            ################################################ update implicit disentangler            
            # update sL
            @tensor P_1[:] := A[-2,-1,12,13]*A[1,13,4,2]*A[10,9,12,11]*A[6,11,4,5]*vL[9,8,-3]*vR[5,7,3]*sR[2,1,3]*dis[10,6,8,7]
            @tensor B_1[:] := A[-2,-1,8,7]*A[1,7,5,2]*A[-4,-3,8,9]*A[4,9,5,3]*sR[2,1,6]*sR[3,4,6]
            @tensor P_2[:] := A[-2,-1,12,13]*A_single[1,13,4,2]*A[10,9,12,11]*A_single[6,11,4,5]*vL[9,8,-3]*vR_single[5,7,3]*sR_single[2,1,3]*dis[10,6,8,7]
            @tensor B_2[:] := A[-2,-1,8,7]*A_single[1,7,5,2]*A[-4,-3,8,9]*A_single[4,9,5,3]*sR_single[2,1,6]*sR_single[3,4,6]
            #=
            @tensor P_2[:] := A_single[-2,-1,12,13]*A[1,13,4,2]*A_single[10,9,12,11]*A[6,11,4,5]*vL[9,8,-3]*vR[5,7,3]*sR[2,1,3]*dis[10,6,8,7]
            @tensor B_2[:] := A_single[-2,-1,8,7]*A[1,7,5,2]*A_single[-4,-3,8,9]*A[4,9,5,3]*sR[2,1,6]*sR[3,4,6]
            @tensor P_3[:] := A[-2,-1,12,13]*A_single[1,13,4,2]*A[10,9,12,11]*A_single[6,11,4,5]*vL[9,8,-3]*vR[5,7,3]*sR[2,1,3]*dis[10,6,8,7]
            @tensor B_3[:] := A[-2,-1,8,7]*A_single[1,7,5,2]*A[-4,-3,8,9]*A_single[4,9,5,3]*sR[2,1,6]*sR[3,4,6]
            =#

            #=    larger environment
            @tensor P[:] := A[-2,-1,14,13]*A[9,13,11,10]*qdA_w[14,11,7,8]*wL[4,5,7]*wR[1,2,8]*vL[4,6,-3]*sR[10,9,12]*vR[1,3,12]*dis[5,2,6,3]
            @tensor B[:] := qdA[10,6,8,7]*A[-2,-1,10,11]*A[3,11,6,4]*A[-4,-3,8,9]*A[2,9,7,1]*sR[4,3,5]*sR[1,2,5]
            =#
            P = P_1 + P_2 #+ P_3
            B = B_1 + B_2 #+ B_3
            sizeP = size(P);sizeB = size(B)
            P = reshape(P,prod(sizeP[1:2]),sizeP[3])
            B = reshape(B,prod(sizeB[1:2]),prod(sizeB[3:4]))
            P_Qnum_in = Merge(Qnum[2],Qnum[1])
            P_Qnum_out = Qnum_vL[3]
            B_Qnum_in = Merge(Qnum[2],Qnum[1])
            B_Qnum_out =  Merge(Qnum[2],Qnum[1])
            #=
            sL = reshape(sL,prod(sizeP[1:2]),sizeP[3])
            sLnew,info1,info2 = linear_update(P,B,sL,P_Qnum_in,P_Qnum_out,B_Qnum_in,B_Qnum_out)
            sL = reshape(sLnew,sizeP...)
            =#
            #=
            sL = update_sM_Z2(P,B,P_Qnum_in,P_Qnum_out,B_Qnum_in,B_Qnum_out;tol=1.0e-8)
            sL = reshape(sL,sizeP...)
            =#
            # update sR 
            @tensor P_1[:] := A[2,1,5,12]*A[-2,12,13,-1]*A[6,4,5,11]*A[10,11,13,9]*vL[4,7,3]*vR[9,8,-3]*sL[1,2,3]*dis[6,10,7,8]
            @tensor B_1[:] := A[2,1,6,7]*A[4,3,6,8]*A[-2,7,9,-1]*A[-4,8,9,-3]*sL[1,2,5]*sL[3,4,5]
            @tensor P_2[:] := A_single[2,1,5,12]*A[-2,12,13,-1]*A_single[6,4,5,11]*A[10,11,13,9]*vL_single[4,7,3]*vR[9,8,-3]*sL_single[1,2,3]*dis[6,10,7,8]
            @tensor B_2[:] := A_single[2,1,6,7]*A_single[4,3,6,8]*A[-2,7,9,-1]*A[-4,8,9,-3]*sL_single[1,2,5]*sL_single[3,4,5]
            #=
            @tensor P_2[:] := A[2,1,5,12]*A_single[-2,12,13,-1]*A[6,4,5,11]*A_single[10,11,13,9]*vL[4,7,3]*vR[9,8,-3]*sL[1,2,3]*dis[6,10,7,8]
            @tensor B_2[:] := A[2,1,6,7]*A[4,3,6,8]*A_single[-2,7,9,-1]*A_single[-4,8,9,-3]*sL[1,2,5]*sL[3,4,5]
            @tensor P_3[:] := A_single[2,1,5,12]*A[-2,12,13,-1]*A_single[6,4,5,11]*A[10,11,13,9]*vL[4,7,3]*vR[9,8,-3]*sL[1,2,3]*dis[6,10,7,8]
            @tensor B_3[:] := A_single[2,1,6,7]*A_single[4,3,6,8]*A[-2,7,9,-1]*A[-4,8,9,-3]*sL[1,2,5]*sL[3,4,5]
            =#
            P = P_1 + P_2 #+ P_3
            B = B_1 + B_2 #+ B_3
            #= larger environment
            @tensor P[:] := qdA_w[11,14,7,8]*A[10,9,11,13]*A[-2,13,14,-1]*wL[1,2,7]*wR[4,6,8]*sL[9,10,12]*vL[1,3,12]*vR[4,5,-3]*dis[2,6,3,5]
            @tensor B[:] := qdA[6,9,7,11]*A[1,2,6,8]*A[-2,8,9,-1]*A[4,3,7,10]*A[-4,10,11,-3]*sL[2,1,5]*sL[3,4,5]
            =#            
            sizeP = size(P);sizeB = size(B)
            P = reshape(P,prod(sizeP[1:2]),sizeP[3])
            B = reshape(B,prod(sizeB[1:2]),prod(sizeB[3:4]))
            P_Qnum_in = Merge(Qnum[4],Qnum[1])
            P_Qnum_out = Qnum_vR[3]
            B_Qnum_in = Merge(Qnum[4],Qnum[1])
            B_Qnum_out =  Merge(Qnum[4],Qnum[1])
            #=
            sR = reshape(sR,prod(sizeP[1:2]),sizeP[3])
            sRnew,info1,info2 = linear_update(P,B,sR,P_Qnum_in,P_Qnum_out,B_Qnum_in,B_Qnum_out)
            sR = reshape(sRnew,sizeP...)
            =#
            #
            sR = update_sM_Z2(P,B,P_Qnum_in,P_Qnum_out,B_Qnum_in,B_Qnum_out;tol=1.0e-8)
            sR = reshape(sR,sizeP...)
            #

            ############################################## update isometries            
            # update dis
            @tensor Env_dis_1[:] := A[1,2,5,11]*A[6,11,9,7]*A[-1,4,5,12]*A[-2,12,9,10]*vL[4,-3,3]*vR[10,-4,8]*sL[2,1,3]*sR[7,6,8]
            @tensor Env_dis_2[:] := A_single[1,2,5,11]*A[6,11,9,7]*A_single[-1,4,5,12]*A[-2,12,9,10]*vL_single[4,-3,3]*vR[10,-4,8]*sL_single[2,1,3]*sR[7,6,8]
            @tensor Env_dis_3[:] := A[1,2,5,11]*A_single[6,11,9,7]*A[-1,4,5,12]*A_single[-2,12,9,10]*vL[4,-3,3]*vR_single[10,-4,8]*sL[2,1,3]*sR_single[7,6,8]
            #@tensor Env_dis_2[:] := A_single[1,2,5,11]*A[6,11,9,7]*A_single[-1,4,5,12]*A[-2,12,9,10]*vL[4,-3,3]*vR[10,-4,8]*sL[2,1,3]*sR[7,6,8]
            #@tensor Env_dis_3[:] := A[1,2,5,11]*A_single[6,11,9,7]*A[-1,4,5,12]*A_single[-2,12,9,10]*vL[4,-3,3]*vR[10,-4,8]*sL[2,1,3]*sR[7,6,8]
            #@tensor Env_dis_2[:] := A_single[1,2,5,11]*A[6,11,9,7]*A_single[-1,4,5,12]*A[-2,12,9,10]*vL_single[4,-3,3]*vR[10,-4,8]*sL_single[2,1,3]*sR[7,6,8]
            #@tensor Env_dis_3[:] := A[1,2,5,11]*A_single[6,11,9,7]*A[-1,4,5,12]*A_single[-2,12,9,10]*vL[4,-3,3]*vR_single[10,-4,8]*sL[2,1,3]*sR_single[7,6,8]

            # larger environment
            #@tensor Env_dis_1[:] := qdA_w[6,7,12,10]*A[2,1,6,5]*A[3,5,7,4]*wL[11,-1,12]*wR[8,-2,10]*sL[1,2,13]*vL[11,-3,13]*sR[4,3,9]*vR[8,-4,9]
            alpha = 1.0
            Env_dis = Env_dis_1+alpha*Env_dis_2+alpha*Env_dis_3
            sizeEnv_dis = size(Env_dis)
            Env_dis = reshape(Env_dis,prod(sizeEnv_dis[1:2]),prod(sizeEnv_dis[3:4]))
            Qnum_in = Merge(Qnum_dis[1],Qnum_dis[2])
            Qnum_out = Merge(Qnum_dis[3],Qnum_dis[4])
            dis = update_vM_Z2(Env_dis,Qnum_in,Qnum_out)
            dis = reshape(dis,sizeEnv_dis)

            # update vL 
            @tensor Env_vL_1[:] := A[8,9,11,10]*A[1,10,4,2]*A[13,-1,11,12]*A[6,12,4,5]*vR[5,7,3]*sL[9,8,-3]*sR[2,1,3]*dis[13,6,-2,7]
            @tensor Env_vL_2[:] := A[8,9,11,10]*A_single[1,10,4,2]*A[13,-1,11,12]*A_single[6,12,4,5]*vR_single[5,7,3]*sL[9,8,-3]*sR_single[2,1,3]*dis[13,6,-2,7]
            #@tensor Env_vL_2[:] := A_single[8,9,11,10]*A[1,10,4,2]*A_single[13,-1,11,12]*A[6,12,4,5]*vR[5,7,3]*sL[9,8,-3]*sR[2,1,3]*dis[13,6,-2,7]
            #@tensor Env_vL_3[:] := A[8,9,11,10]*A_single[1,10,4,2]*A[13,-1,11,12]*A_single[6,12,4,5]*vR[5,7,3]*sL[9,8,-3]*sR[2,1,3]*dis[13,6,-2,7]
            # larger environment
            #@tensor Env_vL[:] := qdA_w[6,7,14,12]*wL[-1,13,14]*wR[8,10,12]*A[4,3,6,5]*A[1,5,7,2]*sL[3,4,-3]*sR[2,1,11]*vR[8,9,11]*dis[13,10,-2,9]
            Env_vL = Env_vL_1 + Env_vL_2 #+ Env_vL_3
            sizeEnv_vL = size(Env_vL)            
            Env_vL = reshape(Env_vL,prod(sizeEnv_vL[1:2]),sizeEnv_vL[3])
            Qnum_in = Merge(Qnum_vL[1],Qnum_vL[2])
            Qnum_out = Qnum_vL[3]
            vL = update_vM_Z2(Env_vL,Qnum_in,Qnum_out)
            vL = reshape(vL,sizeEnv_vL)


            # update vR
            @tensor Env_vR_1[:] := A[2,1,5,10]*A[8,10,12,9]*A[7,4,5,11]*A[13,11,12,-1]*vL[4,6,3]*sL[1,2,3]*sR[9,8,-3]*dis[7,13,6,-2]
            @tensor Env_vR_2[:] := A_single[2,1,5,10]*A[8,10,12,9]*A_single[7,4,5,11]*A[13,11,12,-1]*vL_single[4,6,3]*sL_single[1,2,3]*sR[9,8,-3]*dis[7,13,6,-2]
            #@tensor Env_vR_2[:] := A_single[2,1,5,10]*A[8,10,12,9]*A_single[7,4,5,11]*A[13,11,12,-1]*vL[4,6,3]*sL[1,2,3]*sR[9,8,-3]*dis[7,13,6,-2]
            #@tensor Env_vR_3[:] := A[2,1,5,10]*A_single[8,10,12,9]*A[7,4,5,11]*A_single[13,11,12,-1]*vL[4,6,3]*sL[1,2,3]*sR[9,8,-3]*dis[7,13,6,-2]
            # larger environment
            #@tensor Env_vR[:] := qdA_w[6,7,12,14]*A[2,1,6,5]*A[3,5,7,4]*wL[8,9,12]*wR[-1,13,14]*sL[1,2,11]*vL[8,10,11]*sR[4,3,-3]*dis[9,13,10,-2]
            
            Env_vR = Env_vR_1 + Env_vR_2 #+ Env_vR_3
            sizeEnv_vR = size(Env_vR)
            Env_vR = reshape(Env_vR,prod(sizeEnv_vR[1:2]),sizeEnv_vR[3])
            Qnum_in = Merge(Qnum_vR[1],Qnum_vR[2])
            Qnum_out = Qnum_vR[3]
            vR = update_vM_Z2(Env_vR,Qnum_in,Qnum_out)
            vR = reshape(vR,sizeEnv_vR)


            #
            ###########################################################################################
            # update sL_single
            #@tensor P_single[:] := A_single[-2,-1,12,13]*A[1,13,4,2]*A_single[10,9,12,11]*A[6,11,4,5]*vL_single[9,8,-3]*vR[5,7,3]*sR[2,1,3]*dis[10,6,8,7]
            #@tensor B_single[:] := A_single[-2,-1,8,7]*A[1,7,5,2]*A_single[-4,-3,8,9]*A[4,9,5,3]*sR[2,1,6]*sR[3,4,6]
            @tensor P_single[:] := A_single[-2,-1,12,13]*A[1,13,4,2]*A_single[10,9,12,11]*A[6,11,4,5]*vL_single[9,8,-3]*vR[5,7,3]*sR[2,1,3]*dis[10,6,8,7]
            @tensor B_single[:] := A_single[-2,-1,8,7]*A[1,7,5,2]*A_single[-4,-3,8,9]*A[4,9,5,3]*sR[2,1,6]*sR[3,4,6]
            sizeP = size(P_single);sizeB = size(B_single)
            P_single = reshape(P_single,prod(sizeP[1:2]),sizeP[3])
            B_single = reshape(B_single,prod(sizeB[1:2]),prod(sizeB[3:4]))
            P_Qnum_in = Merge(Qnum_sL_single[1],Qnum_sL_single[2])
            P_Qnum_out = Qnum_sL_single[3]
            B_Qnum_in = Merge(Qnum_sL_single[1],Qnum_sL_single[2])
            B_Qnum_out =  Merge(Qnum_sL_single[1],Qnum_sL_single[2])

            #sL_single = update_sM_Z2(P_single,B_single,P_Qnum_in,P_Qnum_out,B_Qnum_in,B_Qnum_out)
            #=
            sL_single = reshape(sL_single,prod(sizeP[1:2]),sizeP[3])
            sL_single,info1,info2 = linear_update(P_single,B_single,sL_single,P_Qnum_in,P_Qnum_out,B_Qnum_in,B_Qnum_out)
            sL_single = reshape(sL_single,sizeP...)
            =#
            sL_single = update_sM_Z2(P_single,B_single,P_Qnum_in,P_Qnum_out,B_Qnum_in,B_Qnum_out;tol=1.0e-8)
            sL_single = reshape(sL_single,sizeP...)

            # update sR 
            @tensor P_single[:] := A[2,1,5,12]*A_single[-2,12,13,-1]*A[6,4,5,11]*A_single[10,11,13,9]*vL[4,7,3]*vR_single[9,8,-3]*sL[1,2,3]*dis[6,10,7,8]
            @tensor B_single[:] := A[2,1,6,7]*A[4,3,6,8]*A_single[-2,7,9,-1]*A_single[-4,8,9,-3]*sL[1,2,5]*sL[3,4,5]
            sizeP = size(P_single);sizeB = size(B_single)
            P_single = reshape(P_single,prod(sizeP[1:2]),sizeP[3])
            B_single = reshape(B_single,prod(sizeB[1:2]),prod(sizeB[3:4]))
            P_Qnum_in = Merge(Qnum_sR_single[1],Qnum_sR_single[2])
            P_Qnum_out = Qnum_sR_single[3]
            B_Qnum_in = Merge(Qnum_sR_single[1],Qnum_sR_single[2])
            B_Qnum_out =  Merge(Qnum_sR_single[1],Qnum_sR_single[2])
            #=
            sR_single = reshape(sR_single,prod(sizeP[1:2]),sizeP[3])
            sR_single,info1,info2 = linear_update(P_single,B_single,sR_single,P_Qnum_in,P_Qnum_out,B_Qnum_in,B_Qnum_out)   
            sR_single = reshape(sR_single,sizeP...)
            =#
            #
            sR_single = update_sM_Z2(P_single,B_single,P_Qnum_in,P_Qnum_out,B_Qnum_in,B_Qnum_out;tol=1.0e-8)
            sR_single = reshape(sR_single,sizeP...)
            #


            ############################################################################################
            # update vL_single
            @tensor Env_vL_single[:] := A_single[8,9,11,10]*A[1,10,4,2]*A_single[13,-1,11,12]*A[6,12,4,5]*vR[5,7,3]*sL_single[9,8,-3]*sR[2,1,3]*dis[13,6,-2,7]
            sizeEnv_vL_single = size(Env_vL_single)            
            Env_vL_single = reshape(Env_vL_single,prod(sizeEnv_vL_single[1:2]),sizeEnv_vL_single[3])
            Qnum_in = Merge(Qnum_vL_single[1],Qnum_vL_single[2])
            Qnum_out = Qnum_vL_single[3]
            vL_single = update_vM_Z2(Env_vL_single,Qnum_in,Qnum_out)
            vL_single = reshape(vL_single,sizeEnv_vL_single)



            # update vR
            @tensor Env_vR_single[:] := A[2,1,5,10]*A_single[8,10,12,9]*A[7,4,5,11]*A_single[13,11,12,-1]*vL[4,6,3]*sL[1,2,3]*sR_single[9,8,-3]*dis[7,13,6,-2]
            sizeEnv_vR_single = size(Env_vR_single)
            Env_vR_single = reshape(Env_vR_single,prod(sizeEnv_vR_single[1:2]),sizeEnv_vR_single[3])
            Qnum_in = Merge(Qnum_vR_single[1],Qnum_vR_single[2])
            Qnum_out = Qnum_vR_single[3]
            vR_single = update_vM_Z2(Env_vR_single,Qnum_in,Qnum_out)
            vR_single = reshape(vR_single,sizeEnv_vR_single)
            #
            if j%100 ==0
                #println("dif : ", norm(dis-dis_c)/norm(dis)," ",norm(vL-vL_c)/norm(vL)," ",norm(vR-vR_c)/norm(vR))
                #println("dif : ", norm(sL-sL_c)/norm(sL)," ",norm(sR-sR_c)/norm(sR))        
                #println("Env dif",norm(Env_dis-Env_c)/norm(Env_c))    
                print(" Iteration: $j : ")
                compute_fidelity(A,vL,vR,sL,sR,dis,printdetail=true)   
                print("\n")
            end
            
            if abs(norm(dis-dis_c)/norm(dis)) <1.0e-10 && abs(norm(sL-sL_c)/norm(sL)) < 1.0e-10
                compute_fidelity(A,vL,vR,sL,sR,dis,printdetail=true)   
                break
            end

            Env_c = deepcopy(Env_dis)
            dis_c = deepcopy(dis)
            vL_c = deepcopy(vL); sL_c = deepcopy(sL)
            vR_c = deepcopy(vR); sR_c = deepcopy(sR)
            vL_single_c = deepcopy(vL); sL_single_c = deepcopy(sL)
            vR_single_c = deepcopy(vR); sR_single_c = deepcopy(sR)
         
        end


        # compute y,w for renormalization
        y,w,Qnum_new = compute_projector_TNR_Z2(A,Qnum,vL,Qnum_vL,vR,Qnum_vR,sL,sR,chimax)
        y = reshape(y,sizeA[2],sizeA[2],size(y,2))
        w = reshape(w,size(vR,2),size(vL,2),size(w,2))
       
        
        @tensor A_new[:] := A[13,12,16,17]*A[15,14,16,18]*A[1,7,5,2]*A[3,6,5,4]*sL[12,13,24]*vL[8,10,24]*sL[14,15,22]*vL[19,21,22]*sR[2,1,11]*vR[8,9,11]*
                sR[4,3,23]*vR[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3]
        #
        #vL_single = deepcopy(vL);sL_single = deepcopy(sL);vR_single = deepcopy(vR);sR_single = deepcopy(sR)
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
        #

        #=
        # renormalize the network
       @tensor A_single_new[:] := A_single[13,12,16,17]*A[15,14,16,18]*A[1,2,5,7]*A[3,4,5,6]*v_single[12,13,24]*v_single[8,10,24]*v[14,15,22]*v[19,21,22]*v[2,1,11]*v[8,9,11]*
                    v[4,3,23]*v[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3]+
                    A[13,12,16,17]*A_single[15,14,16,18]*A[1,2,5,7]*A[3,4,5,6]*v[12,13,24]*v[8,10,24]*v_single[14,15,22]*v_single[19,21,22]*v[2,1,11]*v[8,9,11]*
                    v[4,3,23]*v[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3]+
                    A[13,12,16,17]*A[15,14,16,18]*A_single[1,2,5,7]*A[3,4,5,6]*v[12,13,24]*v[8,10,24]*v[14,15,22]*v[19,21,22]*v_single[2,1,11]*v_single[8,9,11]*
                    v[4,3,23]*v[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3]+
                    A[13,12,16,17]*A[15,14,16,18]*A[1,2,5,7]*A_single[3,4,5,6]*v[12,13,24]*v[8,10,24]*v[14,15,22]*v[19,21,22]*v[2,1,11]*v[8,9,11]*
                    v_single[4,3,23]*v_single[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3]
        A_single_new = A_single_new/4
        #@tensor Sx_new[:] := Sx[1,2]*y[2,3,-2]*y[1,3,-1]
        =#
        Qnum = Qnum_new    
        A = deepcopy(A_new)
        maximumA = maximum(A)
        Result["logAnorm"] = log(maximum(A))+4*Result["logAnorm"];
        A = A/maximumA
        append!(Result["Amatrix"],[A])
        @tensor Z[:] := A[1,2,1,2]
        fe = -1*Temperature*(log(Z[1])+Result["logAnorm"])/4^(nStep+1);
        append!(Result["FreeEnergy"],fe)

        check = prod(size(A) .-1)
        if check ==0 
            break
        end

        #
        @tensor A_temp[:] := A[-1,1,-2,1]
        Rlt_temp = eigsolve(A_temp)
        A_single = deepcopy(A_single_new)
        A_single = A_single/maximumA
        @tensor Z_single[:] := A_single[1,2,1,2]
        @tensor A_single_temp[:] := A_single[-1,1,-2,1]
        Rlt_single_temp = eigsolve(A_single_temp)
        append!(Result["Magnetization"],(Rlt_single_temp[1]/Rlt_temp[1][1])[1])
        println(" Magnetization: ",(abs.((Rlt_single_temp[1]/Rlt_temp[1][1])[1]) -M)/M)
        #

    end

    append!(FreeEnergy,Result["FreeEnergy"][end])
    append!(freeenergy,[Result["FreeEnergy"]])
    append!(Magnetization,Result["Magnetization"][end])

end
x = 1


