#
#              
#       non Z2 symmetric loop TNR 
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
using LsqFit
if Sys.isunix()
    include("loop_func.jl")
    include("/Users/wangweilan/Dropbox/Code/WESTLAKE/TensorNetwork/VERSION1/2DClassical/partition.jl")
elseif Sys.islinux()
    include("loop_func.jl")
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
Fidelity = []
#for relT in 0.9998:0.00001:1.0002
#for chimax in 4:2:24
    #local Temperature
    chimax = 20
    chiw_max = 3*chimax
    chiv_max = chimax
    chis_max = chimax
    chidis_max = chimax
    relT = 1.0#
    #relT = 0.9998
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

    for nStep in 1:25
        print("RG $nStep:  \n")
        global A

        ########################################### initial vL 
        #? compute implicit s and v
        #@tensor Env[:] := A[-2,11,12,-1]*A[3,5,12,4]*A[7,11,9,8]*A[2,5,9,1]*
        #                A[3,6,14,4]*A[2,6,10,1]*A[-4,13,14,-3]*A[7,13,10,8]
        @tensor Env[:] := A[-2,4,6,-1]*A[-4,5,6,-3]*A[1,4,3,2]*A[1,5,3,2]
        sizeEnv = size(Env)
        Env = reshape(Env,prod(sizeEnv[1:2]),prod(sizeEnv[3:4]))
        v = compute_isometry(Env,chis_max)
        v = reshape(v,sizeEnv[1],sizeEnv[2],size(v,2))
        s = deepcopy(v)




        UpdateLoop = 1000
        @. model(x,p) = p[1].*x.^2 + p[2].*x + p[3]
        @tensor phiphi_1[:] := A[3,4,5,13]*A[10,9,11,13]*A[2,1,5,14]*A[7,8,11,14]*A[2,1,6,15]*
                A[7,8,12,15]*A[3,4,6,16]*A[10,9,12,16]
        @tensor phiphi_2[:] := A[9,13,7,8]*A[14,13,12,11]*A[9,5,2,1]*A[14,5,3,4]*
                A[10,6,2,1]*A[16,6,3,4]*A[10,15,7,8]*A[16,15,12,11]

        #; alternative way to do update : theories are the same, but faster by reusing contracted 
        #; subnetworks 

        @tensor sA[:] := A[2,-2,-1,1]*s[1,2,-3];
        @tensor sAv[:] := sA[-1,-2,1]*v[-3,-4,1];
        @tensor sAvA_1[:] := sAv[-4,2,-3,1]*A[1,2,-1,-2];
        @tensor sAvsAv_1[:] := sAv[-3,2,-4,1]*sAv[-1,2,-2,1]
        @tensor sAvA_2[:] := sAv[1,-3,2,-4]*A[-1,-2,1,2] 
        @tensor sAvsAv_2[:] := sAv[1,-2,2,-1]*sAv[1,-4,2,-3]

        fidelity_initial = compute_fidelity_1(A,s,v,phiphi_1,phiphi_2,sA,sAv,sAvA_1,sAvsAv_1,sAvA_2,sAvsAv_2,printdetail=true)
        print("\n")
        for j in 1:UpdateLoop

            if j %100 ==0 
                print(" Update Iteration $j, ")
                fidelity_initial = compute_fidelity_1(A,s,v,phiphi_1,phiphi_2,sA,sAv,sAvA_1,sAvsAv_1,sAvA_2,sAvsAv_2,printdetail=true)
                print(" \n")
                #fidelity_check(A,s,v,chimax,printdetail=true)
                #print(" \n")
            end
            if abs(fidelity_initial) < 1.0e-14
                break
            end
            #; update s 
            #! sA = s*A; sAv = sA*v ; sAvA = sAv*A; sAvsAv = sAv*sAv 
            @tensor sA[:] = A[2,-2,-1,1]*s[1,2,-3];
            @tensor sAv[:] = sA[-1,-2,1]*v[-3,-4,1];
            @tensor sAvA_1[:] = sAv[-4,2,-3,1]*A[1,2,-1,-2];
            @tensor sAvsAv_1[:] = sAv[-3,2,-4,1]*sAv[-1,2,-2,1]
            @tensor sAvA_2[:] = sAv[1,-3,2,-4]*A[-1,-2,1,2] 
            @tensor sAvsAv_2[:] = sAv[1,-2,2,-1]*sAv[1,-4,2,-3]
            #
            @tensor P_1[:] := sAvA_1[1,6,8,2]*sAvA_1[1,4,3,2]*sAvA_1[5,4,3,10]*A[7,9,5,6]*
                            A[-2,9,10,-1]*v[8,7,-3]
            @tensor B_1[:] := sAvsAv_1[3,5,4,7]*sAvsAv_1[3,2,4,1]*sAvsAv_1[-3,2,-1,1]*v[7,6,-2]*v[5,6,-4]

            @tensor P_2[:] := sAvA_2[1,6,10,2]*sAvA_2[1,4,3,2]*sAvA_2[5,4,3,8]*A[-2,10,9,-1]*
                            A[5,6,9,7]*v[7,8,-3]
            @tensor B_2[:] := sAvsAv_2[2,-1,1,-3]*sAvsAv_2[1,4,2,3]*sAvsAv_2[7,3,6,4]*v[5,6,-4]*v[5,7,-2]
            stemp = update_s(P_1,B_1,P_2,B_2,A,s,chis_max,j)                
            
            #compute_fidelity_1(A,s,v,phiphi_1,phiphi_2,printdetail=true)
            #@time begin 
            if j <= 1000 
                fidelity_array= Array{Float64}(undef,4)
                itemp_s = 1
                for i in [1.0,3.0,5.0,7.0]
                    snew = 0.1*i*s + (1-0.1*i)*stemp
                    fidelity = compute_fidelity_1(A,snew,v,phiphi_1,phiphi_2,sA,sAv,sAvA_1,sAvsAv_1,sAvA_2,sAvsAv_2,printdetail=false)
                    fidelity_array[itemp_s] = fidelity
                    itemp_s += 1
                end
                L = curve_fit(model,[1.0,3.0,5.0,7.0],fidelity_array,[0.0,0.0,0.0])
                itemp = -L.param[2]/(2*L.param[1])
            else
                itemp = 5.0
            end
            s = 0.1*itemp*s + (1-0.1*itemp)*stemp
            #end  
            #
            #; update v 
            @tensor sA[:] = A[2,-2,-1,1]*s[1,2,-3]
            @tensor sAv[:] = sA[-1,-2,1]*v[-3,-4,1]
            @tensor sAvA_1[:] = A[1,2,-1,-2]*sAv[-4,2,-3,1]
            @tensor sAvsAv_1[:] = sAv[-1,2,-2,1]*sAv[-3,2,-4,1]
            @tensor sAvA_2[:] = sAv[1,-3,2,-4]*A[-1,-2,1,2] 
            @tensor sAvsAv_2[:] = sAv[1,-2,2,-1]*sAv[1,-4,2,-3]


            @tensor P_1[:] := sAvA_1[1,6,-1,2]*sAvA_1[1,4,3,2]*sAvA_1[5,4,3,10]*A[-2,9,5,6]*sA[10,9,-3] 
            @tensor B_1[:] := sAvsAv_1[4,-3,3,-1]*sAvsAv_1[3,2,4,1]*sAvsAv_1[6,2,7,1]*sA[7,5,-4]*sA[6,5,-2]

            @tensor P_2[:] := sAvA_2[1,6,8,2]*sAvA_2[1,4,3,2]*sAvA_2[5,4,3,-2]*A[5,6,7,-1]*sA[7,8,-3]
            @tensor B_2[:] := sAvsAv_2[2,7,1,6]*sAvsAv_2[1,4,2,3]*sAvsAv_2[-3,4,-1,3]*sA[5,7,-2]*sA[5,6,-4]
            vtemp = update_v(P_1,B_1,P_2,B_2,v,chis_max,j)                
            
            if j <= 1000 
                fidelity_array = Array{Float64}(undef,4)
                itemp_v = 1
                for i in [1.0,3.0,5.0,7.0]
                    vnew = 0.1*i*v + (1-0.1*i)*vtemp
                    fidelity = compute_fidelity_1(A,s,vnew,phiphi_1,phiphi_2,sA,sAv,sAvA_1,sAvsAv_1,sAvA_2,sAvsAv_2,printdetail=false)
                    fidelity_array[itemp_v] = fidelity
                    append!(Fidelity,fidelity)
                    itemp_v +=1
                end
                L = curve_fit(model,[1.0,3.0,5.0,7.0],fidelity_array,[0.0,0.0,0.0])
                itemp = -L.param[2]/(2*L.param[1])
            else
                itemp = 5.0
            end
            v = 0.1*itemp*v + (1-0.1*itemp)*vtemp
        end 

            fidelity = compute_fidelity_1(A,s,v,phiphi_1,phiphi_2,sA,sAv,sAvA_1,sAvsAv_1,sAvA_2,sAvsAv_2,printdetail=true)
            print("\n")

        #=
        sizev = size(v)
        v = reshape(v,prod(sizev[1:2]),sizev[3])
        Rlt = qr(v)
        v = reshape(Rlt.Q[:,1:sizev[3]],sizev...)
        @tensor s[:] := s[-1,-2,1]*deepcopy(Rlt.R)[-3,1]
        =#
        #
        y,w = compute_projectors(A,s,v,chimax)
        @tensor A_new[:] := A[13,17,16,12]*A[15,18,16,14]*A[1,7,5,2]*A[3,6,5,4]*s[12,13,24]*v[8,10,24]*s[14,15,22]*v[19,21,22]*s[2,1,11]*v[8,9,11]*
                s[4,3,23]*v[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3]
        #
        #=
        y,w = compute_projector_vertical(A,s,v,chimax)
        @tensor A_new[:] := A[5,9,6,4]*A[8,9,10,7]*A[13,16,17,12]*A[14,16,18,15]*
                    s[4,5,11]*s[7,8,24]*s[12,13,23]*s[15,14,22]*v[1,3,11]*v[20,19,24]*
                    v[2,3,23]*v[21,19,22]*y[2,1,-2]*y[21,20,-4]*w[6,10,-1]*w[17,18,-3]
        =#
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

        #=
        @tensor A_temp[:] := A[-1,1,-2,1]
        Rlt_temp = eigsolve(A_temp)
        A_single = deepcopy(A_single_new)
        A_single = A_single/maximumA
        @tensor Z_single[:] := A_single[1,2,1,2]
        @tensor A_single_temp[:] := A_single[-1,1,-2,1]
        Rlt_single_temp = eigsolve(A_single_temp)
        append!(Result["Magnetization"],(Rlt_single_temp[1]/Rlt_temp[1][1])[1])
        println(" Magnetization: ",(abs.((Rlt_single_temp[1]/Rlt_temp[1][1])[1]) -M)/M)
        =#

    end
    append!(FreeEnergy,Result["FreeEnergy"][end])
    #append!(freeenergy,[Result["FreeEnergy"]])
    #append!(Magnetization,Result["Magnetization"][end])

#end
x = 1


