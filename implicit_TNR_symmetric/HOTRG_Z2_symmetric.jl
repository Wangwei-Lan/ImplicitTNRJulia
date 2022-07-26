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
chimax = 4
J1 = 1.0


FE_exact =[]
FreeEnergy =[]
Magnetization =[]
Second_Magnetization =[]
Third_Magnetization =[]
Fourth_Magnetization =[]
Binder =[]
g4 =[]
#for relT in 0.9998:0.00001:1.0002
for chimax in 2:1:14
    #local Temperature
    #chimax = 10
    relT = 0.9994
    println("=========================Relative Temperature $relT, chimax = $chimax ===============================")
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


    #
    # Binder ration for magnetization
    @tensor A[:] := Jtemp[1,2,3,4]*Etemp[1,-1]*Etemp[2,-2]*Etemp[3,-3]*Etemp[4,-4]  
    @tensor A_single_1[:] := Jtemp_impurity_1[1,2,3,4]*Etemp[1,-1]*Etemp[2,-2]*Etemp[3,-3]*Etemp[4,-4];  
    @tensor A_single_2[:] := Jtemp_impurity_2[1,2,3,4]*Etemp[1,-1]*Etemp[2,-2]*Etemp[3,-3]*Etemp[4,-4];  
    @tensor A_double[:] := Jtemp[1,2,3,4]*Etemp[1,-1]*Etemp[2,-2]*Etemp[3,-3]*Etemp[4,-4]
    @tensor A_triple[:] := Jtemp_impurity[1,2,3,4]*Etemp[1,-1]*Etemp[2,-2]*Etemp[3,-3]*Etemp[4,-4]  
    @tensor A_quadruple[:] := Jtemp[1,2,3,4]*Etemp[1,-1]*Etemp[2,-2]*Etemp[3,-3]*Etemp[4,-4]



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



    for nStep in 1:40
        print("RG $nStep : ")
        #global Qnum,A,A_single




        # environment for projector w_v (vertical)
        @tensor Env_v[:] := A[1,-1,5,2]*A[3,-2,5,4]*A[1,-3,6,2]*A[3,-4,6,4]
        sizeEnv_v = size(Env_v)
        Env_v = reshape(Env_v,prod(sizeEnv_v[1:2]),prod(sizeEnv_v[3:4]))
        qnumtemp = merge(Qnum[2],Qnum[2])

        # compute w_v 
        w_v,qnum_new = compute_isometry_Z2(Env_v,qnumtemp,qnumtemp,chimax)
        w_v = reshape(w_v,sizeEnv_v[1],sizeEnv_v[2],size(w_v,2))

        # update local tensor A
        @tensor A_new[:] := A[-1,1,3,4]*A[-3,2,3,5]*w_v[1,2,-2]*w_v[4,5,-4]     
        @tensor A_single_new[:] :=  A_single[-1,1,3,4]*A[-3,2,3,5]*w_v[1,2,-2]*w_v[4,5,-4]     
        A = deepcopy(A_new)
        A_single = deepcopy(A_single_new)
        Qnum[2] = qnum_new
        Qnum[4] = qnum_new

        # environment for projector w_h (horizontal)
        @tensor Env_h[:] := A[-1,1,2,5]*A[-2,4,3,5]*A[-3,1,2,6]*A[-4,4,3,6]
        sizeEnv_h = size(Env_h)
        Env_h = reshape(Env_h,prod(sizeEnv_h[1:2]),prod(sizeEnv_h[3:4]))
        qnumtemp = merge(Qnum[1],Qnum[1])

        # compute w_h
        w_h,qnum_new = compute_isometry_Z2(Env_h,qnumtemp,qnumtemp,chimax)
        w_h = reshape(w_h,sizeEnv_h[1],sizeEnv_h[2],size(w_h,2))

        # update local tensor A
        @tensor A_new[:] := A[1,-2,4,2]*A[3,-4,5,2]*w_h[1,3,-1]*w_h[4,5,-3]
        @tensor A_single_new[:] := A_single[1,-2,4,2]*A[3,-4,5,2]*w_h[1,3,-1]*w_h[4,5,-3]
        A = deepcopy(A_new)
        A_single = deepcopy(A_single_new)
        Qnum[1] = qnum_new
        Qnum[3] = qnum_new
        

        Result["logAnorm"] = log(maximum(A))+4*Result["logAnorm"];
        A_single = A_single/maximum(A)
        A = A/maximum(A)
        append!(Result["Amatrix"],[A])

        @tensor Z[:] := A[1,2,1,2]
        @tensor Z_single[:] := A_single[1,2,1,2]
        fe = -1*Temperature*(log(Z[1])+Result["logAnorm"])/4^(nStep+1);
        println("Z_single: ",Z_single[1])
        append!(Result["FreeEnergy"],fe)
        #append!(Result["Magnetization"],Z_single[1]/Z[1])

        @tensor A_temp[:] := A[-1,1,-2,1]
        Rlt_temp = eigsolve(A_temp)

        @tensor A_single_temp[:] := A_single[-1,1,-2,1]
        Rlt_single_temp = eigsolve(A_single_temp)
        append!(Result["Magnetization"],(Rlt_single_temp[1]/Rlt_temp[1][1])[1])
    



    end

    append!(FreeEnergy,Result["FreeEnergy"][end])
    append!(Magnetization,Result["Magnetization"][end])





end
x = 1


