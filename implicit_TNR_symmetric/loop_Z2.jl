#
#              
#       Z2 symmetric loop TNR 
#      remove short range entanlement in two loops
#      use implicit disentanglers instead of purly loop disentanglers
#!     The difference between implicit disentanglers and loop disentanglers are: implicit disentanglers
#! does not make modifications of the origional local tensors 
#      //don't include any spacial symmetry; 
#      TODO: include spacial symmetry: vertical or horizontal  
#!     The code need to be both vertical and horizontal symmetric; 
#!     It is required due to projectors to renormalize the network need to be the same 
#!     on both sites of the impurity, it will work only if the networks has both spacial symmetries 
#
#
#
using TensorOperations
using KrylovKit
using LinearAlgebra
using Printf
using JLD2
if Sys.isunix()
    include("./Z2_symmetry_func.jl")
    include("./loop_Z2_func.jl")
    include("/Users/wangweilan/Dropbox/Code/WESTLAKE/TensorNetwork/VERSION1/2DClassical/partition.jl")
elseif Sys.islinux()
    include("./Z2_symmetry_func.jl")
    include("./loop_Z2_func.jl")
    include("/home/wangwei/Dropbox/Code/WESTLAKE/TensorNetwork/VERSION1/2DClassical/partition.jl")
end

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
    #! Binder ration for magnetization
    #! A_single means one single impurity 
    #! A_doube means two impurities, others are relattively the same 
    #
    @tensor A[:] := Jtemp[1,2,3,4]*Etemp[1,-1]*Etemp[2,-2]*Etemp[3,-3]*Etemp[4,-4]  
    @tensor A_single_1[:] := Jtemp_impurity_1[1,2,3,4]*Etemp[1,-1]*Etemp[2,-2]*Etemp[3,-3]*Etemp[4,-4];  
    @tensor A_single_2[:] := Jtemp_impurity_2[1,2,3,4]*Etemp[1,-1]*Etemp[2,-2]*Etemp[3,-3]*Etemp[4,-4];  
    @tensor A_double[:] := Jtemp[1,2,3,4]*Etemp[1,-1]*Etemp[2,-2]*Etemp[3,-3]*Etemp[4,-4]
    @tensor A_triple[:] := Jtemp_impurity[1,2,3,4]*Etemp[1,-1]*Etemp[2,-2]*Etemp[3,-3]*Etemp[4,-4]  
    @tensor A_quadruple[:] := Jtemp[1,2,3,4]*Etemp[1,-1]*Etemp[2,-2]*Etemp[3,-3]*Etemp[4,-4]

    # initial tensor  
    #! change gauge to implement Z2 symmetry
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
        print("RG $nStep:  \n")
        #global Qnum,A,A_single,Sx

        ########################################### initial vL 
        #? compute implicit s and v
        @tensor Env[:] := A[-2,11,12,-1]*A[3,5,12,4]*A[7,11,9,8]*A[2,5,9,1]*
                        A[3,6,14,4]*A[2,6,10,1]*A[-4,13,14,-3]*A[7,13,10,8]
        sizeEnv = size(Env); qnum_temp = Merge(Qnum[4],Qnum[1])
        s,qnum_s = compute_isometry_Z2(Env,qnum_temp,qnum_temp,chis_max)
        v = deepcopy(s);                                        #! initial v and s are the same 
        Qnum_s = Array{Array{Float64}}(undef,3)                 #! quantum number for s 
        Qnum_s[1] = Qnum[4];Qnum_s[2] = Qnum[1];Qnum_s= qnum_s  #! quantum number for v are the same as s   
        Qnum_v = deepcopy(Qnum_s)


        UpdateLoop = 1500
        for j in 1:UpdateLoop

            #? remove short range entanglement for sector 1
            @tensor P_1[:] := A[15,16,19,18]*A[21,23,19,20]*A[10,9,11,18]*A[4,5,11,20]*
                            A[-2,23,24,-1]*A[2,5,12,1]*A[12,16,24,13]*A[1,9,12,6]*
                            s[1,2,3]*s[6,7,8]*s[13,12,14]*v[22,21,-3]*v[22,4,3]*v[17,10,8]*v[17,15,14] 
            @tensor B_1[:] := A[25,26,-3,24]*A[10,13,14,9]*A[5,6,14,4]*A[3,6,15,2]*A[12,13,15,11]*
                            A[22,26,-1,23]*s[24,25,28]*s[9,10,17]*s[4,5,8]*s[4,5,8]*s[2,3,7]*
                            s[11,12,18]*s[23,22,27]*v[20,19,28]*v[20,16,17]*v[30,1,8]*v[30,29,-4]*
                            v[31,1,7]*v[31,29,-2]*v[21,16,18]*v[21,19,27]
            sizeP = size(P); sizeB = size(B)
            #? quantum numbers for P_1 and B_1 
            P_Qnum_in = Merge(Qnum_s[1],Qnum_s[2]); P_Qnum_out = Qnum_s[3]
            B_Qnum_in = Merge(); B_Qnum_out = Merge()

            #? remove short range entanglement for sector 2
            @tensor P_2[:] := A[20,19,16,17]*A[11,19,10,9]*A[20,21,24,22]*A[11,21,4,5]*
                            A[-2,25,24,-1]*A[1,25,4,2]*A[16,18,14,13]*A[10,18,6,7]*
                            s[2,1,3]*s[7,6,8]*s[13,14,15]*v[22,23,-3]*v[5,12,3]*v[9,12,8]*v[22,23,-3]
            @tensor B_2[:] := A[22,24,23,21]*A[12,24,16,13]*A[1,-3,5,2]*A[4,-1,5,3]*A[14,25,16,15]*
                            A[20,25,23,19]*s[21,22,30]*s[13,12,18]*s[2,1,7]*s[3,4,8]*s[15,14,17]*
                            s[19,20,31]*v[26,29,30]*v[9,10,18]*v[6,10,7]*v[27,29,-4]*v[6,11,8]*
                            v[9,11,17]*v[26,28,31]*v[27,28,-2]
            #? quantum numbers for P_2 and B_2
            sizeP = size(P); sizeB = size(B)            
            P_Qnum_in = Merge();P_Qnum_out = Merge()
        end



        #=
        ##################   permutedims to obtain local matrix A
        Alocal = Array{Array{Float64}}(undef,4)
        Alocal[1] = permutedims(A,[2,1,4,3]);Alocal[2] = permutedims(A,[1,4,3,2])
        Alocal[3] = permutedims(A,[4,3,2,1]);Alocal[4] = permutedims(A,[3,2,1,4])

        #################
        Slocal = Array{Array{Float64}}(undef,8)                             #! local S matrix 
        Qnum_Slocal = Array{Array{Array{Float64,1}}}(undef,8)               #! quantum numbers for local S matrix 

        #? initial Slocal[1] and Slocal[2]
        @tensor Env[:] := A[-2,-1,13,14]*A[8,14,9,7]*A[13,1,2,5]*A[9,5,3,4]*A[11,1,2,6]*A[10,6,3,4]*A[-4,-3,11,12]*A[8,12,10,7]
        Snew,qnum_S = initial_S(Env,Qnum[2],Qnum[1],chis_max)
        Slocal[1] = deepcopy(Snew);Slocal[2] = deepcopy(Snew) 
        #? determine quantum numbers for Slocal[1] and Slocal[2] 
        Qnum_Slocal[1]  = Array{Array{Float64,1}}(undef,3)
        Qnum_Slocal[1][1] = Qnum[2]; Qnum_Slocal[1][2] = Qnum[1]; Qnum_Slocal[1][3] = qnum_S
        Qnum_Sloocal[2] = deepcopy(Qnum_Slocal[1])

        #? initial Slocal[3] and Slocal[4]
        @tensor Env[:] := A[2,1,11,6]*A[3,6,10,4]*A[11,-3,-4,12]*A[10,12,8,7]*A[2,1,13,5]*A[3,5,9,4]*A[13,-1,-2,14]*A[9,14,8,7]
        Snew,qnum_S = initial_S(Env,Qnum[2],Qnum[4],chis_max) 
        Slocal[3] = deepcopy(Snew); Slocal[4] = deepcopy(Snew)
        #? determine quantum numbers for Slodal[3] and Slocal[4]
        Qnum_Slocal[3] = Array{Array{Float64,1}}(undef,3)
        Qnum_Slocal[3][1] = Qnum[2]; Qnum_Slocal[3][2] = Qnum[3] ; Qnum_Slocal[3][3] = qnum_S
        Qnum_Slocal[4] = deepcopy(Qnum_Slocal[3])

        #? initial Slocal[5] and Slocal[6]
        @tensor Env[:] := A[2,1,10,6]*A[3,6,14,4]*A[10,8,7,13]*A[14,13,-4,-3]*A[2,1,9,5]*A[3,5,12,4]*A[9,8,7,11]*A[12,11,-2,-1]
        Snew,qnum_S = initial_S(Env,Qnum[4],Qnum[3],chis_max)
        Slocal[5] = deepcopy(Snew); Slocal[6] = deepcopy(Snew)
        #? initial qnumtum numbers for Slocal[5] and Slocal[6]
        Qnum_Slocal[5] = Array{Array{Float64,1}}(undef,3)
        Qnum_Slocal[5][1] = Qnum[4]; Qnum_Slocal[5][2] = Qnum[3]; Qnum_Slocal[5][3] = qnum_S
        Qnum_Slocal[6] = deepcopy(Qnum_Slocal[5])
       
        #? initial Slocal[7] and Slocal[8] 
        @tensor Env[:] := A[7,8,9,11]*A[-2,11,12,-1]*A[9,1,2,5]*A[12,5,3,4]*A[10,1,2,6]*A[14,6,3,4]*A[7,8,10,13]*A[-4,13,14,-3]
        Snew,qnum_S = initial_S(Env,Qnum[4],Qnum[1],chis_max) 
        Slocal[7] = deepcopy(Snew); Slocal[8] = deepcopy(Snew) 
        #? initial quantum numbers for Slocal[7] and Slocal[8]
        Qnum_Slocal[7] = Array{Array{Float64,1}}(undef,3)
        Qnum_Slocal[7][1] = Qnum[4];Qnum_Slocal[7][2] = Qnum[1];Qnum_Slocal[7][3] = qnum_S
        Qnum_Slocal[8] = deepcopy(Qnum_Slocal[7])


        #;#################
        #;#################
        #? initial reduced density matrix for sector 1  
        rho_psiphi_R_1 = Array{Array{Float64}}(undef,4)               #! density matrix for psiphi from Right; need initial
        rho_psipsi_R_1 = Array{Array{Float64}}(undef,4)               #! density matrix for psipsi from Right; need initial 
        rho_psiphi_L_1 = Array{Array{Float64}}(undef,4)               #! density matrix for psiphi from Left; update on the fly
        rho_psipsi_L_1 = Array{Array{Float64}}(undef,4)               #! density matrix for psipsi from Left; update on the fly

        #? initial reduced density matrix for sector 2
        rho_psiphi_R_2 = Array{Array{Float64}}(undef,4)               #! density matrix for psiphi from Right; need initial
        rho_psipsi_R_2 = Array{Array{Float64}}(undef,4)               #! density matrix for psipsi from Right; need initial 
        rho_psiphi_L_2 = Array{Array{Float64}}(undef,4)               #! density matrix for psiphi from Left; update on the fly
        rho_psipsi_L_2 = Array{Array{Float64}}(undef,4)               #! density matrix for psipsi from Left; update on the fly

        rho_psiphi_R_1[end] = reshape(Matrix(1.0I,size(Alocal[4],4)^2,size(Alocal[4],4)^2),size(Alocal[4],4),size(Alocal[4],4),size(Alocal[4],4),size(Alocal[4],4))
        rho_psipsi_R_1[end] = reshape(Matrix(1.0I,size(Alocal[4],4)^2,size(Alocal[4],4)^2),size(Alocal[4],4),size(Alocal[4],4),size(Alocal[4],4),size(Alocal[4],4))
        rho_psiphi_L_1[1] = reshape(Matrix(1.0I,size(Alocal[4],4)^2,size(Alocal[4],4)^2),size(Alocal[4],4),size(Alocal[4],4),size(Alocal[4],4),size(Alocal[4],4))
        rho_psipsi_L_1[1] = reshape(Matrix(1.0I,size(Alocal[4],4)^2,size(Alocal[4],4)^2),size(Alocal[4],4),size(Alocal[4],4),size(Alocal[4],4),size(Alocal[4],4))

        rho_psiphi_R_2[end] = reshape(Matrix(1.0I,size(Alocal[4],4)^2,size(Alocal[4],4)^2),size(Alocal[4],4),size(Alocal[4],4),size(Alocal[4],4),size(Alocal[4],4))
        rho_psipsi_R_2[end] = reshape(Matrix(1.0I,size(Alocal[4],4)^2,size(Alocal[4],4)^2),size(Alocal[4],4),size(Alocal[4],4),size(Alocal[4],4),size(Alocal[4],4))
        rho_psiphi_L_2[1] = reshape(Matrix(1.0I,size(Alocal[4],4)^2,size(Alocal[4],4)^2),size(Alocal[4],4),size(Alocal[4],4),size(Alocal[4],4),size(Alocal[4],4))
        rho_psipsi_L_2[1] = reshape(Matrix(1.0I,size(Alocal[4],4)^2,size(Alocal[4],4)^2),size(Alocal[4],4),size(Alocal[4],4),size(Alocal[4],4),size(Alocal[4],4))




        #? compute reduced density matrix from right to left
        for j in 3:-1:1
            if j%2 == 1
                @tensor rho_psiphi_R_1[j][:] := rho_psiphi_R_1[j+1][6,7,-3,-4]*Alocal[j+1][-1,4,5,6]*Alocal[j+1][-2,4,2,1]*
                                Slocal[2*j-1][1,2,3]*Slocal[2*j][7,5,3]
                @tensor rho_psipsi_R_1[j][:] := rho_psipsi_R_1[j+1][2,3,-3,-4]*Alocal[j+1][-1,8,6,7]*Alocal[j+1][-2,8,5,4]*
                                Slocal[2*j-1][4,5,10]*Slocal[2*j-1][7,6,9]*Slocal[2*j][3,1,10]*Slocal[2*j][2,1,9]
            elseif j%2 == 0 
                @tensor rho_psiphi_R_1[j][:] := rho_psiphi_R_1[j+1][6,7,-3,-4]*Alocal[j+1][-1,4,5,6]*Alocal[j+1][1,2,5,7]*
                                Slocal[2*j-1][-2,4,3]*Slocal[2*j][1,2,3]
                @tensor rho_psipsi_R_1[j][:] := rho_phiphi_R[j+1][6,7,-3,-4]*Alocal[j+1][2,1,5,6]*Alocal[j+1][3,4,5,7]*
                                Slocal[2*j-1][-1,8,9]*Slocal[2*j-1][-2,8,10]*Slocal[2*j][2,1,9]*Slocal[2*j][3,4,10]
            end
        end




            for j in 1:UpdateLoop

                #! here we introduce concept site and block; block corresponding to local tensor A at different position 
                #! sites (even and odd) means two sites that are connect to local tensor A; 
                #! for example: block 1, site 1 corresponding to S1; block 1, site 2 corresponding to S2;
                
                #! we also have two sections: corresponding to two loops within which the loop entanglement should be removed
                #! In section 1, Slocal start with S1 and ends at S8; The local tensor Alocal start from A1 to A4
                #! In section 2, Slocal start with S2 (to S1) and ends at S7;
                for k in 1:4 
                    if k % 2 == 1 
                        #! odd block; (A1 and A3)
                        #; section 1 
                        #? odd site at odd block 
                        @tensor PS_1[:] := rho_psiphi_L_1[k][2,1,3,-1]*rho_psiphi_R_1[k][4,8,2,1]*Alocal[k][3,-2,7,4]*
                                        Alocal[k][5,6,7,8]*Slocal[2*k][5,6,-3]                    
                        @tensor BS_1[:] := rho_psiphi_L_1[k][2,1,-1,-2]*rho_psiphi_R_1[k][8,9,2,1]*Alocal[k][3,4,7,8]*
                                        Alocal[k][5,6,7,9]*Slocal[2*k][3,4,-3]*Slocal[2*k][5,6,-4]   


                        #; section 2  
                        @tensor PS_2[:] :=  
                        @tensor BS_2[:] := 
                        #? even site at odd block  
                        @tensor PS_1[:] := rho_psiphi_L_1[k][1,2,3,5]*rho_psiphi_R_1[k][4,8,1,2]*Alocal[k][3,6,7,4]*
                                        Alocal[k][-1,-2,7,8]*Slocal[2*k-1][5,6,-3]
                        @tensor BS_1[:] := rho_psiphi_L_1[k][5,4,2,3]*rho_psiphi_R_1[k][-3,-4,5,4]*Slocal[2*k-1][2,1,-1]*
                                        Slocal[2*k-1][3,1,-2]
                    elseif k%2 == 0 
                        #! even block; (A2 and A4)
                        #; section 1 
                        #? odd sites at even block 


                        #? even sites at even block 



                    end

                end


                #@tensor P_S1_1[:] := A1*A2*A3*Alocal[4]*A1*A2*A3*Alocal[4]*S1*S2*S3*S4*S5*S6*S7*S8
                #@tensor P_S1_2[:] := A1*A2*A3*Alocal[4]*A1*A2*A3*Alocal[4]*S1*S2*S3*S4*S5*S6*S7*S8

            #@tensor B_S1_1[:] :=  A1*A2*A3*Alocal[4]*A1*A2*A3*Alocal[4]*S1*S2*S3*S4*S5*S6*S7*S8
            #@tensor B_S1_2[:] :=  A1*A2*A3*Alocal[4]*A1*A2*A3*Alocal[4]*S1*S2*S3*S4*S5*S6*S7*S8
            #Env_S1 = Env_S1_1 + Env_S1_2

            



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
    =#
    append!(FreeEnergy,Result["FreeEnergy"][end])
    append!(freeenergy,[Result["FreeEnergy"]])
    append!(Magnetization,Result["Magnetization"][end])

end
x = 1


