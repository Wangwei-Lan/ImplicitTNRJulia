#
#
# TNR with Z2 symetry + horizontal symmetry
# normal environment; 
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
for chimax in 4:2:20
    #local Temperature
    #chimax = 15
    chiw_max = 2*chimax
    chidis_max = chimax
    chis_max = chimax
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
   
    #=
    Qnum = Array{Array{Float64}}(undef,4)
    Qnum[1] = Float64[0,1]
    Qnum[2] = Float64[0,1]
    Qnum[3] = Float64[0,1]
    Qnum[4] = Float64[0,1]
    =#


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

        ########################################### initial w 
        #@tensor Env_w[:] := A[-2,-1,5,4]*A[3,2,1,4]*A[-4,-3,5,6]*A[3,2,1,6]
        @tensor Env_w[:] := A[-2,-1,13,14]*A[8,7,9,14]*A[2,1,13,5]*A[3,4,9,5]*
                            A[-4,-3,11,12]*A[8,7,10,12]*A[2,1,11,6]*A[3,4,10,6]
        sizeEnv_w = size(Env_w)
        # quantum number for v
        qnumtemp = merge(Qnum[2],Qnum[1]);
        # compute w for truncation
        w,qnum_new = compute_isometry_Z2(reshape(Env_w,prod(sizeEnv_w[1:2]),prod(sizeEnv_w[3:4])),qnumtemp,qnumtemp,chiw_max);
        w = reshape(w,sizeEnv_w[1],sizeEnv_w[2],size(w,2));
        Qnum_w = Array{Array{Float64}}(undef,3);
        Qnum_w[1] = Qnum[2];Qnum_w[2] = Qnum[1];Qnum_w[3] = qnum_new;
       

        #=
        ########################################### initial w_single 
        # environment for v_single
        @tensor  Env_w_single[:] := A_single[-2,-1,5,4]*A[3,2,1,4]*A_single[-4,-3,5,6]*A[3,2,1,6];
        sizeEnv_w_single = size(Env_w_single);
        
        # quantum number for v_single
        qnumtemp = merge(Qnum[2],Qnum[1]);

        # compute v_single for truncation
        w_single,qnum_new = compute_isometry_Z2(reshape(Env_w_single,prod(sizeEnv_w_single[1:2]),prod(sizeEnv_w_single[3:4])),
                                qnumtemp,qnumtemp,chiw_max);
        w_single = reshape(w_single,sizeEnv_w_single[1],sizeEnv_w_single[2],size(w_single,2));
        Qnum_w_single = Array{Array{Float64}}(undef,3);
        Qnum_w_single[1] = Qnum[2];Qnum_w_single[2] = Qnum[1];Qnum_w_single[3] = qnum_new;
        =#
        #########################################################################################
        #########################################################################################
        #########################################################################################
        # introduce vL,vR,sL and sR 

        @tensor Env_v[:] := Env_w[1,2,3,4]*w[1,2,-1]*w[3,4,-2]
        v,qnum_v = compute_isometry_Z2(Env_v,Qnum_w[3],Qnum_w[3],chis_max)
        Qnum_v_temp = Array{Array{Float64,1},1}(undef,2)
        Qnum_v_temp[1] = Qnum_w[3];Qnum_v_temp[2] = qnum_v

        #=
        @tensor Env_v_single[:] := Env_w_single[1,2,3,4]*w_single[1,2,-1]*w_single[3,4,-2]
        v_single,qnum_v_single = compute_isometry_Z2(Env_v_single,Qnum_w_single[3],Qnum_w_single[3],chiw_max)
        Qnum_v_single_temp = Array{Array{Float64,1},1}(undef,2)
        Qnum_v_single_temp[1] = Qnum_w_single[3];Qnum_v_single_temp[2] = qnum_v_single
        =#

        #########################################################################################
        #########################################################################################
        #########################################################################################
        s = deepcopy(v);Qnum_s = deepcopy(Qnum_v_temp)
        #s_single = deepcopy(v_single); Qnum_s_single = deepcopy(Qnum_v_single_temp)

        @tensor v[:] := w[-1,-2,1]*v[1,-3]
        #@tensor v_single[:] := w_single[-1,-2,1]*v_single[1,-3]

        Qnum_v = deepcopy(Qnum_w);Qnum_v[3] = Qnum_v_temp[2]
        #Qnum_v_single = deepcopy(Qnum_w_single); Qnum_v_single[3] = Qnum_v_single_temp[2]

        @tensor vA[:] := A[2,1,11,5]*A[3,4,12,5]*A[7,6,11,10]*A[8,9,12,10]*w[1,2,-1]*w[6,7,-2]*w[4,3,-3]*w[9,8,-4]
        #@tensor vA_single_L[:] := A_single[2,1,11,5]*A[3,4,12,5]*A_single[7,6,11,10]*A[8,9,12,10]*w_single[1,2,-1]*w_single[6,7,-2]*w[4,3,-3]*w[9,8,-4]
        #@tensor vA_single_R[:] := A[2,1,11,5]*A_single[3,4,12,5]*A[7,6,11,10]*A_single[8,9,12,10]*w[1,2,-1]*w[6,7,-2]*w_single[4,3,-3]*w_single[9,8,-4]
        
        # initial disentangler
        sizeA = size(A)
        #
        dis = zeros(sizeA[1]*sizeA[1],size(Qnum_w[2],1)*size(Qnum_w[2],1))
        qnum_in = merge(Qnum[1],Qnum[1]);qnum_out = merge(Qnum_w[2],Qnum_w[2])
        odd_in = findall(x->x==1,qnum_in);odd_out = findall(x->x==1,qnum_out)
        even_in = findall(x->x==0,qnum_in);even_out = findall(x->x==0,qnum_out)
        dis[odd_in,odd_out] = Matrix(1.0I,size(odd_in,1),size(odd_out,1))
        dis[even_in,even_out] = Matrix(1.0I,size(even_in,1),size(even_out,1))
        dis = reshape(dis,sizeA[1],sizeA[1],size(Qnum_w[2],1),size(Qnum_w[2],1))
        #
        #dis = reshape(Matrix(1.0I,sizeA[1]^2,sizeA[1]^2),sizeA[1],sizeA[1],sizeA[1],sizeA[1])
        Qnum_dis = Array{Array{Float64,1},1}(undef,4)                       # create disentangler qnumtum number
        Qnum_dis[1] = Qnum[1];Qnum_dis[2] = Qnum[1]
        Qnum_dis[3] = Qnum_w[2];Qnum_dis[4] = Qnum_w[2]


        println("chidis: ===============================: ",size(dis))
        println("chiA: ===============================: ",size(A))
        println("Initial truncation error ")
        fidelity_initial = compute_fidelity_modified(A,w,w,v,v,s,s,dis,printdetail=true); tol = fidelity_initial/100
        #compute_fidelity_impurity_modified(A,A_single,w,w,w_single,w_single,v,v,v_single,v_single,s,s,s_single,s_single,dis;printdetail=true)

        Env_c = deepcopy(dis)
        dis_c = deepcopy(dis)
        v_c = deepcopy(v); s_c = deepcopy(s)
        #v_single_c = deepcopy(v_single); s_single_c = deepcopy(s_single)


        dis_initial = deepcopy(dis)
        v_initial = deepcopy(v); s_initial = deepcopy(s)
        #v_single_initial = deepcopy(v_single); s_single_initial = deepcopy(s_single)

        UpdateLoop = 0
        for j in 1:UpdateLoop



            ################################## break if the fidelity is less than 1.0e-14
            #fidelity = compute_fidelity_modified(A,wL,wR,vL,vR,sL,sR,dis,printdetail=false)
            #=
            if abs(fidelity_initial) < 1.0e-14
                break
            end
            =#
            tol = 1.0e-6
            beta = 0
            #=
            print(" sL before update:  ")
            f1 = compute_fidelity_modified(A,wL,wR,vL,vR,sL,sR,dis,printdetail=false)   
            f2,f3 = compute_fidelity_impurity_modified(A,A_single,wL,wR,wL_single,wR_single,vL,vR,vL_single,vR_single,sL,sR,sL_single,sR_single,dis;printdetail=false)
            print(f1+f2+f3)
            =#
            ##############################################################################
            ################################################ update implicit disentangler            
            # update sL
            @tensor P_1[:] := vA[-1,7,4,5]*w[9,8,7]*w[1,3,5]*v[9,10,-2]*v[1,2,6]*s[4,6]*dis[8,3,10,2]
            @tensor B_1[:] := vA[-1,-2,1,2]*s[1,3]*s[2,3]
            @tensor P_2[:] := vA_single_R[-1,7,4,5]*w[9,8,7]*w_single[1,3,5]*v[9,10,-2]*v_single[1,2,6]*s_single[4,6]*dis[8,3,10,2]
            @tensor B_2[:] := vA_single_R[-1,-2,1,2]*s_single[1,3]*s_single[2,3]
            P = P_1 + P_2;B = B_1 + B_2
            sizeP = size(P);sizeB = size(B)
            P = reshape(P,sizeP[1],sizeP[2]);B = reshape(B,sizeB[1],sizeB[2])
            P_Qnum_in = Qnum_s[1];P_Qnum_out = Qnum_s[2]
            B_Qnum_in = Qnum_s[1];B_Qnum_out = Qnum_s[1]
            s_temp = update_sM_Z2(P,B,P_Qnum_in,P_Qnum_out,B_Qnum_in,B_Qnum_out;tol=tol)
            s_temp = reshape(s_temp,sizeP...)
            s = 0.01*beta*s + (1-0.01*beta)*s_temp


            ###########################################################################################
            ###########################################################################################
            # update sL_single
            @tensor P_single[:] := vA_single_L[-1,7,4,5]*w_single[9,8,7]*w[1,3,5]*v_single[9,10,-2]*v[1,2,6]*s[4,6]*dis[8,3,10,2]
            @tensor B_single[:] := vA_single_L[-1,-2,1,2]*s[1,3]*s[2,3]
            sizeP = size(P_single);sizeB = size(B_single)
            P_single = reshape(P_single,sizeP[1],sizeP[2]);B_single = reshape(B_single,sizeB[1],sizeB[2])
            P_Qnum_in = Qnum_s_single[1];P_Qnum_out = Qnum_s_single[2]
            B_Qnum_in = Qnum_s_single[1];B_Qnum_out = Qnum_s_single[1]

            s_single_temp = update_sM_Z2(P_single,B_single,P_Qnum_in,P_Qnum_out,B_Qnum_in,B_Qnum_out;tol=tol)
            s_single_temp = reshape(sL_single_temp,sizeP...)
            s_single = 0.01*beta*s_single + (1-0.01*beta)*s_single_temp


            ############################################################################################
            ############################################## update isometries            
            # update vL 
            @tensor Env_v_1[:] := vA[9,8,4,6]*w[-1,7,8]*w[1,3,6]*v[1,2,5]*s[9,-3]*s[4,5]*dis[7,3,-2,2]
            @tensor Env_v_2[:] := vA_single_R[9,8,4,6]*w[-1,7,8]*w_single[1,3,6]*v_single[1,2,5]*s[9,-3]*s_single[4,5]*dis[7,3,-2,2]
            Env_v = Env_v_1 + Env_v_2
            sizeEnv_v = size(Env_v)            
            Env_v = reshape(Env_v,prod(sizeEnv_v[1:2]),sizeEnv_v[3])
            Qnum_in = merge(Qnum_v[1],Qnum_v[2]);Qnum_out = Qnum_v[3]
            v = update_vM_Z2(Env_v,Qnum_in,Qnum_out)
            v = reshape(v,sizeEnv_v)


            ############################################################################################
            ############################################################################################
            ############################################################################################
            # update vL_single
            @tensor Env_v_single[:] := vA_single_L[9,8,4,6]*w_single[-1,7,8]*w[1,3,6]*v[1,2,5]*s_single[9,-3]*s[4,5]*dis[7,3,-2,2]
            sizeEnv_v_single = size(Env_v_single)            
            Env_v_single = reshape(Env_v_single,prod(sizeEnv_v_single[1:2]),sizeEnv_v_single[3])
            Qnum_in = merge(Qnum_v_single[1],Qnum_v_single[2]);Qnum_out = Qnum_v_single[3]
            v_single = update_vM_Z2(Env_v_single,Qnum_in,Qnum_out)
            v_single = reshape(v_single,sizeEnv_v_single)


            # update dis
            @tensor Env_dis_1[:] := vA[6,7,2,3]*w[5,-1,7]*w[1,-2,3]*v[5,-3,8]*v[1,-4,4]*s[6,8]*s[2,4]
            @tensor Env_dis_2[:] := vA_single_L[6,7,2,3]*w_single[5,-1,7]*w[1,-2,3]*v_single[5,-3,8]*v[1,-4,4]*s_single[6,8]*s[2,4]
            @tensor Env_dis_3[:] := vA_single_R[6,7,2,3]*w[5,-1,7]*w_single[1,-2,3]*v[5,-3,8]*v_single[1,-4,4]*s[6,8]*s_single[2,4]
            alpha = 1.0; gamma = 1.0
            Env_dis = gamma*Env_dis_1+alpha*Env_dis_2+alpha*Env_dis_3
            sizeEnv_dis = size(Env_dis)
            Env_dis = reshape(Env_dis,prod(sizeEnv_dis[1:2]),prod(sizeEnv_dis[3:4]))
            Qnum_in = merge(Qnum_dis[1],Qnum_dis[2]);Qnum_out = merge(Qnum_dis[3],Qnum_dis[4])
            dis = update_vM_Z2(Env_dis,Qnum_in,Qnum_out)
            dis = reshape(dis,sizeEnv_dis)
           
            
            
            
            ##########################################   keep initial if updated fidelity is larger than initial
            #=
            if j == UpdateLoop && nStep >= 1
                fidelity = compute_fidelity_modified(A,wL,wR,vL,vR,sL,sR,dis,printdetail=true)   
                compute_fidelity_impurity_modified(A,A_single,wL,wR,wL_single,wR_single,vL,vR,vL_single,vR_single,sL,sR,sL_single,sR_single,dis;printdetail=true)
                if abs(fidelity_initial) < abs(fidelity)
                println("=========================================================")
                println("                  isometries from initial ")
                println("=========================================================")
                dis = deepcopy(dis_initial)
                vL = deepcopy(vL_initial); sL = deepcopy(sL_initial)
                vR = deepcopy(vR_initial); sR = deepcopy(sR_initial)
                vL_single = deepcopy(vL_single_initial); sL_single = deepcopy(sL_single_initial)
                vR_single = deepcopy(vR_single_initial); sR_single = deepcopy(sR_single_initial)      
                break       
                end
            end
            =#

            ##########################     print fidelity for every 100 iterations
            if j%100 ==0
                #println("dif : ", norm(dis-dis_c)/norm(dis)," ",norm(vL-vL_c)/norm(vL)," ",norm(vR-vR_c)/norm(vR))
                #println("dif : ", norm(sL-sL_c)/norm(sL)," ",norm(sR-sR_c)/norm(sR))        
                #println("Env dif",norm(Env_dis-Env_c)/norm(Env_c))    
                print("Iteration : $j ")
                tol = compute_fidelity_modified(A,w,w,v,v,s,s,dis,printdetail=true) / 100  
                compute_fidelity_impurity_modified(A,A_single,w,w,w_single,w_single,v,v,v_single,v_single,s,s,s_single,s_single,dis;printdetail=true)
            end


            ##################################  break if dis and sL and wL converge
            if abs(norm(dis-dis_c)/norm(dis)) <1.0e-10 && abs(norm(s-s_c)/norm(s)) < 1.0e-10
               break
            end           

            Env_c = deepcopy(Env_dis)
            dis_c = deepcopy(dis)
            v_c = deepcopy(v); s_c = deepcopy(s)
            v_single_c = deepcopy(v); s_single_c = deepcopy(s)
         
        end


        ########### compute y,w for renormalization

        # compute sL and sR to be compatible with previous works
        @tensor s[:] := w[-1,-2,1]*s[1,-3]
        #@tensor s_single[:] := w_single[-1,-2,1]*s_single[1,-3]        


        y,w,Qnum_new = compute_projector_TNR_Z2_modified(A,Qnum,v,Qnum_v,s,chimax)
        y = reshape(y,sizeA[2],sizeA[2],size(y,2))
        w = reshape(w,size(v,2),size(v,2),size(w,2))
       
        
        
        ######   create new local and local impurity tensor 
        @tensor A_new[:] := A[13,12,16,17]*A[15,14,16,18]*A[1,2,5,7]*A[3,4,5,6]*s[12,13,24]*v[8,10,24]*s[14,15,22]*v[19,21,22]*s[2,1,11]*v[8,9,11]*
                s[4,3,23]*v[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3]

        #=
        @tensor A_single_new[:] := 
                A_single[13,12,16,17]*A[15,14,16,18]*A[1,2,5,7]*A[3,4,5,6]*s_single[12,13,24]*v_single[8,10,24]*s[14,15,22]*v[19,21,22]*s[2,1,11]*v[8,9,11]*
                s[4,3,23]*v[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3]+
                A[13,12,16,17]*A_single[15,14,16,18]*A[1,2,5,7]*A[3,4,5,6]*s[12,13,24]*v[8,10,24]*s_single[14,15,22]*v_single[19,21,22]*s[2,1,11]*v[8,9,11]*
                s[4,3,23]*v[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3]+
                A[13,12,16,17]*A[15,14,16,18]*A_single[1,2,5,7]*A[3,4,5,6]*s[12,13,24]*v[8,10,24]*s[14,15,22]*v[19,21,22]*s_single[2,1,11]*v_single[8,9,11]*
                s[4,3,23]*v[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3]+
                A[13,12,16,17]*A[15,14,16,18]*A[1,2,5,7]*A_single[3,4,5,6]*s[12,13,24]*v[8,10,24]*s[14,15,22]*v[19,21,22]*s[2,1,11]*v[8,9,11]*
                s_single[4,3,23]*v_single[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3]
        A_single_new = A_single_new/4
        =#

        @tensor A_single_new[:] := A_single[19,18,15,16]*A[13,12,15,14]*A[1,2,7,3]*A[4,5,7,6]*s[12,13,20]*v[9,10,20]*
                        s[2,1,21]*v[18,17,21]*s[5,4,11]*v[9,8,11]*y[6,3,-2]*y[14,16,-4]*w[8,10,-3]*w[17,19,-1]+
                        A[13,12,15,14]*A_single[19,18,15,16]*A[1,2,7,3]*A[4,5,7,6]*s[12,13,21]*v[9,10,21]*s[2,1,11]*v[9,8,11]*
                        s[5,4,20]*v[18,17,20]*y[6,3,-2]*y[16,14,-4]*w[17,19,-3]*w[8,10,-1]+
                        A[2,1,6,3]*A[5,4,6,7]*A_single[18,19,16,15]*A[12,13,16,14]*s[1,2,21]*v[19,17,21]*s[4,5,11]*v[9,10,11]*
                        s[13,12,20]*v[9,8,20]*y[14,15,-2]*y[7,3,-4]*w[18,17,-1]*w[8,10,-3]+
                        A[2,1,6,3]*A[5,4,6,7]*A[12,13,16,14]*A_single[18,19,16,15]*s[1,2,11]*v[10,8,11]*s[4,5,21]*v[19,17,21]*
                        s[13,12,20]*v[10,9,20]*y[15,14,-2]*y[7,3,-4]*w[18,17,-3]*w[9,8,-1]
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


