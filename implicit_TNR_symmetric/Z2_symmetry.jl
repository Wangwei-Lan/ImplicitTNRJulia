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
#for chimax in 2:2:24
    #local Temperature
    chimax = 10
    relT = 0.9994
    println("Relative Temperature $relT")
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



    for nStep in 1:20
        println("==================== RG $nStep ==========================================")
        global Qnum,A,A_single

        # environment for v
        @tensor  Env[:] := A[-2,-1,5,4]*A[3,2,1,4]*A[-4,-3,5,6]*A[3,2,1,6]
        sizeEnv = size(Env)
        Env = reshape(Env,prod(sizeEnv[1:2]),prod(sizeEnv[3:4]))
        
        # quantum number for v
        qnumtemp = merge(Qnum[2],Qnum[1])

        # compute v for truncation
        v,qnum_new = compute_isometry_Z2(Env,qnumtemp,qnumtemp,chimax)
        v = reshape(v,sizeEnv[1],sizeEnv[2],size(v,2))


        # environment for v_single
        @tensor  Env_single[:] := A_single[-2,-1,5,4]*A[3,2,1,4]*A_single[-4,-3,5,6]*A[3,2,1,6]
        sizeEnv_single = size(Env_single)
        Env_single = reshape(Env_single,prod(sizeEnv_single[1:2]),prod(sizeEnv_single[3:4]))
 
        # quantum number for v
        qnumtemp = merge(Qnum[2],Qnum[1])

        # compute v for truncation
        v_single,qnum_new = compute_isometry_Z2(Env_single,qnumtemp,qnumtemp,chimax)
        v_single = reshape(v_single,sizeEnv_single[1],sizeEnv_single[2],size(v_single,2))


        # compute y,w for renormalization
        y,w,Qnum_new = compute_projector_Z2(A,Qnum,v,chimax)


        # renormalize the network
        @tensor A_new[:] := A[13,12,16,17]*A[15,14,16,18]*A[1,2,5,7]*A[3,4,5,6]*v[12,13,24]*v[8,10,24]*v[14,15,22]*v[19,21,22]*v[2,1,11]*v[8,9,11]*
                    v[4,3,23]*v[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3]
        @tensor A_single_new[:] := A_single[13,12,16,17]*A[15,14,16,18]*A[1,2,5,7]*A[3,4,5,6]*v_single[12,13,24]*v_single[8,10,24]*v[14,15,22]*v[19,21,22]*v[2,1,11]*v[8,9,11]*
                v[4,3,23]*v[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3]

        Qnum = Qnum_new    
        A = deepcopy(A_new)
        A_single = deepcopy(A_single_new)

        Result["logAnorm"] = log(maximum(A))+4*Result["logAnorm"];
        A_single = A_single/maximum(A)
        A = A/maximum(A)
        append!(Result["Amatrix"],[A])

        @tensor Z[:] := A[1,2,1,2]
        @tensor Z_single[:] := A_single[1,2,1,2]
        fe = -1*Temperature*(log(Z[1])+Result["logAnorm"])/4^(nStep+1);
        append!(Result["FreeEnergy"],fe)
        append!(Result["Magnetization"],Z_single[1]/Z[1])


    end

    append!(FreeEnergy,Result["FreeEnergy"][end])
    append!(Magnetization,Result["Magnetization"])

#end
x = 1


