using TensorOperations
using KrylovKit
using LinearAlgebra
using Printf
using JLD2
include("./non_symmetric_func.jl")
include("/home/wangwei/Dropbox/Code/WESTLAKE/TensorNetwork/VERSION1/2DClassical/partition.jl")



FreeEnergyRelativeError =[]


# load parameter 
Tc = 2/log(1+sqrt(2))
chimax = 36
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
#for chimax in 6:3:36
    #local Temperature
    chimax = 20
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
    Jtemp_impurity = zeros(2,2,2,2); Jtemp_impurity[1,1,1,1] = -1.0; Jtemp_impurity[2,2,2,2]
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



    A_single = deepcopy(A_single_1)
    #=
    @tensor A_single[:] :=  A_single[-1,-3,2,1]*A[-2,-7,4,1]*A[-5,-4,2,3]*A[-6,-8,4,3]+
                            A[-1,-3,2,1]*A_single[-2,-7,4,1]*A[-5,-4,2,3]*A[-6,-8,4,3]+
                            A[-1,-3,2,1]*A[-2,-7,4,1]*A_single[-5,-4,2,3]*A[-6,-8,4,3]+
                            A[-1,-3,2,1]*A[-2,-7,4,1]*A[-5,-4,2,3]*A_single[-6,-8,4,3]
    A_single = A_single/4
    #
    #@tensor A_single[:] := A_single_1[-1,-3,2,1]*A_single_2[-2,-7,4,1]*A[-5,-4,2,3]*A[-6,-8,4,3]
    @tensor A[:] := A[-1,-3,2,1]*A[-2,-7,4,1]*A[-5,-4,2,3]*A[-6,-8,4,3]
    A_single = reshape(A_single,4,4,4,4)
    A = reshape(A,4,4,4,4)
    =#
    @tensor Z[:] := A[1,2,1,2]
    fe = -1*Temperature*(log(Z[1])+Result["logAnorm"])/4;
    append!(Result["FreeEnergy"],fe)




    for nStep in 1:40
        println("============================ RG step $nStep =========================================")
        global A,A_single,A_double,A_triple,A_quadruple
        local fe,Z

        wLup,wRup = compute_isometry_up(A,chimax)
        wLdn,wRdn = compute_isometry_dn(A,chimax)
        wLup_single,wRup_single = compute_isometry_up_impurity(A,A_single,chimax)
        wLdn_single,wRdn_single = compute_isometry_dn_impurity(A,A_single,chimax)

        y,w = compute_projector(A,wLup,wRup,wLdn,wRdn,chimax)


        @tensor A_new[:] :=  A[13,12,16,17]*A[16,14,15,18]*A[1,7,5,2]*A[5,6,3,4]*wLup[12,13,24]*wLup[8,10,24]*wLdn[14,15,22]*wLdn[19,21,22]*wRup[2,1,11]*wRup[8,9,11]*
                                wRdn[4,3,23]*wRdn[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3]
        #
        @tensor A_single_new[:] :=  A_single[13,12,16,17]*A[16,14,15,18]*A[1,7,5,2]*A[5,6,3,4]*wLup[12,13,24]*wLup[8,10,24]*wLdn[14,15,22]*wLdn[19,21,22]*wRup[2,1,11]*wRup[8,9,11]*
                                wRdn[4,3,23]*wRdn[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3]+
                                A[13,12,16,17]*A_single[16,14,15,18]*A[1,7,5,2]*A[5,6,3,4]*wLup[12,13,24]*wLup[8,10,24]*wLdn[14,15,22]*wLdn[19,21,22]*wRup[2,1,11]*wRup[8,9,11]*
                                wRdn[4,3,23]*wRdn[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3]+
                                A[13,12,16,17]*A[16,14,15,18]*A_single[1,7,5,2]*A[5,6,3,4]*wLup[12,13,24]*wLup[8,10,24]*wLdn[14,15,22]*wLdn[19,21,22]*wRup[2,1,11]*wRup[8,9,11]*
                                wRdn[4,3,23]*wRdn[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3]+
                                A[13,12,16,17]*A[16,14,15,18]*A[1,7,5,2]*A_single[5,6,3,4]*wLup[12,13,24]*wLup[8,10,24]*wLdn[14,15,22]*wLdn[19,21,22]*wRup[2,1,11]*wRup[8,9,11]*
                                wRdn[4,3,23]*wRdn[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3]
        #
        #=
        @tensor A_single_new[:] :=  A_single[13,12,16,17]*A[16,14,15,18]*A[1,7,5,2]*A[5,6,3,4]*wLup_single[12,13,24]*wLup_single[8,10,24]*wLdn[14,15,22]*wLdn[19,21,22]*wRup[2,1,11]*wRup[8,9,11]*
                                wRdn[4,3,23]*wRdn[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3]
        =#
        #=
        @tensor A_single_new[:] :=  A_single[13,12,16,17]*A[16,14,15,18]*A[1,7,5,2]*A[5,6,3,4]*wLup_single[12,13,24]*wLup_single[8,10,24]*wLdn[14,15,22]*wLdn[19,21,22]*wRup[2,1,11]*wRup[8,9,11]*
                                wRdn[4,3,23]*wRdn[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3]+
                                A[13,12,16,17]*A_single[16,14,15,18]*A[1,7,5,2]*A[5,6,3,4]*wLup[12,13,24]*wLup[8,10,24]*wLdn_single[14,15,22]*wLdn_single[19,21,22]*wRup[2,1,11]*wRup[8,9,11]*
                                wRdn[4,3,23]*wRdn[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3]+
                                A[13,12,16,17]*A[16,14,15,18]*A_single[1,7,5,2]*A[5,6,3,4]*wLup[12,13,24]*wLup[8,10,24]*wLdn[14,15,22]*wLdn[19,21,22]*wRup_single[2,1,11]*wRup_single[8,9,11]*
                                wRdn[4,3,23]*wRdn[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3]+
                                A[13,12,16,17]*A[16,14,15,18]*A[1,7,5,2]*A_single[5,6,3,4]*wLup[12,13,24]*wLup[8,10,24]*wLdn[14,15,22]*wLdn[19,21,22]*wRup[2,1,11]*wRup[8,9,11]*
                                wRdn_single[4,3,23]*wRdn_single[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3]
        =#
        A = deepcopy(A_new)
        A_single = deepcopy(A_single_new)

        Result["logAnorm"] = log(maximum(A))+4*Result["logAnorm"];

        A_single = A_single/maximum(A)/4
        A = A/maximum(A)
        append!(Result["Amatrix"],[A])

        @tensor Z[:] := A[1,2,1,2]
        @tensor Z_single[:] := A_single[1,2,1,2]   

        fe = -1*Temperature*(log(Z[1])+Result["logAnorm"])/4^(nStep+1);
        append!(Result["FreeEnergy"],fe)
        append!(Result["Magnetization"],Z_single[1]/Z[1])
        println("RG step: $nStep : ",fe," Magnetization: ",Z_single[1]/Z[1])

    end


    append!(FreeEnergy,Result["FreeEnergy"][25])
    append!(Magnetization,Result["Magnetization"][end])

    #@save "Square_Ising_0.9998Tc_HOTRG_symmetric_chimax_$(chimax).jld2" Result
    #@save "Square_Ising_HOTRG_symmetric_chimax_$(chimax).jld2" Result
#end

x = 1