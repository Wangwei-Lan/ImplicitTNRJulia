#
#
#      PTTRG; both vertical and horizontal symmetric 
#      higher order moments calculations are included
#
#

using TensorOperations
using KrylovKit
using LinearAlgebra
using Printf
using JLD2
include("./tnr_idr_symmetric_func.jl")
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
for chimax in 8:2:40
    #local Temperature
    #chimax = 20
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



    #=
    A,A_single,A_double,A_triple,A_quadruple = HOTRG_initial(A,A_single,A_double,A_triple,A_quadruple)
    A,A_single,A_double,A_triple,A_quadruple = HOTRG_initial(permutedims(A,[4,1,2,3]),permutedims(A_single,[4,1,2,3]),
                            permutedims(A_double,[4,1,2,3]),permutedims(A_triple,[4,1,2,3]),permutedims(A_quadruple,[4,1,2,3]))
    A = permutedims(A,[2,3,4,1]);
    A_single = permutedims(A_single,[2,3,4,1]);
    A_double = permutedims(A_double,[2,3,4,1]);
    A_triple = permutedims(A_triple,[2,3,4,1]);
    A_quadruple = permutedims(A_quadruple,[2,3,4,1]);
    Result["logAnorm"] = log(maximum(A))+4*Result["logAnorm"];
    A_single = A_single/maximum(A)
    A_double = A_double/maximum(A)
    A_triple = A_triple/maximum(A)
    A_quadruple = A_quadruple/maximum(A)
    A = A/maximum(A)
    =#

    #
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
    println("Initial Reflection symmetry: ")
    x = compute_gauge(A,printdetail=true)
    @tensor Atemp[:] := A[-1,1,-3,2]*x[-2,1]*x[-4,2]
    println("reflection symmetry: ",norm(Atemp-permutedims(A,[3,2,1,4])))
    z = compute_gauge(permutedims(A,[2,3,4,1]),printdetail=true)
    @tensor Atemp[:] := A[1,-2,2,-4]*z[-1,1]*z[-3,2]
    println("reflection symmetry: ",norm(Atemp-permutedims(A,[1,4,3,2])))

    @tensor Atemp[:] := A[1,2,3,4]*x[-2,2]*x[-4,4]*z[-1,1]*z[-3,3]
    println("reflection symmetry: ",norm(Atemp-permutedims(A,[3,4,1,2])))

    @tensor Atemp[:] := A_single[1,-2,2,-4]*z[-1,1]*z[-3,2]
    println("impurity reflection symmetry: ",norm(Atemp-permutedims(A_single,[1,4,3,2]))/norm(Atemp))
    @tensor Atemp[:] := A_single[-1,1,-3,2]*x[-2,1]*x[-4,2]
    println("impurity reflection symmetry: ",norm(Atemp-permutedims(A_single,[3,2,1,4]))/norm(Atemp))
    
    
    @tensor Atemp[:] := A_single[1,2,3,4]*x[-2,2]*x[-4,4]*z[-1,1]*z[-3,3]
    println("reflection symmetry: ",norm(Atemp-permutedims(A_single,[3,4,1,2])))


    @tensor Z[:] := A[1,2,1,2]
    fe = -1*Temperature*(log(Z[1])+Result["logAnorm"])/4;
    append!(Result["FreeEnergy"],fe)




    for nStep in 1:40
        println("============================ RG step $nStep =========================================")
        #global A,A_single,A_double,A_triple,A_quadruple
        local fe,Z

        #v = compute_isometry_symmetric(A,chimax)
        #v_single = deepcopy(v)        
        #v_single = compute_isometry_symmetric(A,chimax,Aimp = A_single)
        
        #v = compute_isometry_symmetric_single_tensor(A,chimax)
        #v_single = deepcopy(v)
        #v_single = compute_isometry_symmetric_single_tensor(A_single,chimax)

        v = compute_isometry_symmetric_two_tensor(A,chimax)
        v_single = deepcopy(v)
        v_single = compute_isometry_symmetric_two_tensor(A,chimax,Aimp=A_single)

        #=
        v_double = compute_isometry_symmetric(A,chimax,Aimp = A_double)
        v_triple = compute_isometry_symmetric(A,chimax,Aimp = A_triple)
        v_quadruple = compute_isometry_symmetric(A,chimax,Aimp = A_quadruple)
        v_single = deepcopy(v)
        =#
        
        
        y,w,vAvA = compute_projector_symmetric(A,v,chimax)
        @tensor A_new[:] := A[13,12,16,17]*A[15,14,16,18]*A[1,2,5,7]*A[3,4,5,6]*v[12,13,24]*v[8,10,24]*v[14,15,22]*v[19,21,22]*v[2,1,11]*v[8,9,11]*
                                v[4,3,23]*v[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3]
        #
        #
        x = compute_gauge(A,printdetail=true)
        @tensor Atemp[:] := A[-1,1,-3,2]*x[-2,1]*x[-4,2]
        println("reflection symmetry: ",norm(Atemp-permutedims(A,[3,2,1,4])))        
        z = compute_gauge(permutedims(A,[2,3,4,1]),printdetail=true)
        @tensor Atemp[:] := A[1,-2,2,-4]*z[-1,1]*z[-3,2]
        println("reflection symmetry: ",norm(Atemp-permutedims(A,[1,4,3,2])))
        @tensor Atemp[:] := A[1,2,3,4]*x[-2,2]*x[-4,4]*z[-1,1]*z[-3,3]
        println("reflection symmetry: ",norm(Atemp-permutedims(A,[3,4,1,2])))


        #
        @tensor Atemp[:] := A_single[-1,1,-3,2]*x[-2,1]*x[-4,2]
        println("impurity reflection symmetry: ",norm(abs.(Atemp)-permutedims(abs.(A_single),[3,2,1,4]))/norm(Atemp))
        @tensor Atemp[:] := A_single[1,-2,2,-4]*z[-1,1]*z[-3,2]
        println("impurity reflection symmetry: ",norm(abs.(Atemp)-permutedims(abs.(A_single),[1,4,3,2]))/norm(Atemp))
        @tensor Atemp[:] := A_single[1,2,3,4]*x[-2,2]*x[-4,4]*z[-1,1]*z[-3,3]
        println("reflection symmetry: ",norm(Atemp-permutedims(A_single,[3,4,1,2])))
      

        #
        if nStep <= 40
            A_single_new = renormalize_fragment_one_impurity(A,A_single,v,v_single,y,w)
            A_single_new = A_single_new/4
        elseif nStep >= 40
            @tensor A_single_new[:] := A_single[13,12,16,17]*A[15,14,16,18]*A[1,2,5,7]*A[3,4,5,6]*v_single[12,13,24]*v_single[8,10,24]*v[14,15,22]*v[19,21,22]*v[2,1,11]*v[8,9,11]*
                            v[4,3,23]*v[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3]
        else
            println("This is final step")
            @tensor A_single[:] := A_single[1,2,3,4]*x[-2,2]*x[-4,4]*z[-1,1]*z[-3,3]
            @tensor A_single_new[:] := A[13,12,16,17]*A[15,14,16,18]*A[1,2,5,7]*A_single[5,6,3,4]*v[12,13,24]*v[8,10,24]*v[14,15,22]*v[19,21,22]*v[2,1,11]*v[8,9,11]*
                 v_single[4,3,23]*v_single[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3]
        end
        #

        #=
        impurity = true
        println(impurity)
            if position =="leftup" && nStep == location 
                @tensor A_single[:] := A_single[1,-2,2,-4]*z[-1,1]*z[-3,2]
                if impurity == true
                    v_single = compute_isometry_symmetric_two_tensor_impurity(A,A_single,position="leftup")
                end
                @tensor A_single_new[:] := A[13,12,16,17]*A[15,14,16,18]*A_single[1,7,5,2]*A[3,4,5,6]*v[12,13,24]*v[8,10,24]*v[14,15,22]*v[19,21,22]*v_single[2,1,11]*v_single[8,9,11]*
                                v[4,3,23]*v[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3]
            elseif position =="rightdown" && nStep == location 
                @tensor A_single[:] := A_single[-1,1,-3,2]*x[-2,1]*x[-4,2]
                if impurity == true
                    v_single = compute_isometry_symmetric_two_tensor_impurity(A,A_single,position="rightdown")
                end
                
                @tensor A_single_new[:] := A[13,12,16,17]*A_single[16,14,15,18]*A[1,2,5,7]*A[3,4,5,6]*v[12,13,24]*v[8,10,24]*v_single[14,15,22]*v_single[19,21,22]*v[2,1,11]*v[8,9,11]*
                            v[4,3,23]*v[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3]
            elseif position == "rightup" && nStep == location
                if impurity == true
                    v_single = compute_isometry_symmetric_two_tensor_impurity(A,A_single,position="rightup")
                end
                @tensor A_single_new[:] := A_single[13,12,16,17]*A[15,14,16,18]*A[1,2,5,7]*A[3,4,5,6]*v_single[12,13,24]*v_single[8,10,24]*v[14,15,22]*v[19,21,22]*v[2,1,11]*v[8,9,11]*
                        v[4,3,23]*v[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3]
            elseif position == "leftdown" && nStep == location
                println("HERE!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                @tensor A_single[:] := A_single[1,2,3,4]*x[-2,2]*x[-4,4]*z[-1,1]*z[-3,3]
                
                if impurity == true
                    v_single = compute_isometry_symmetric_two_tensor_impurity(A,A_single,position="leftdown")
                end
                
                @tensor A_single_new[:] := A[13,12,16,17]*A[15,14,16,18]*A[1,2,5,7]*A_single[5,6,3,4]*v[12,13,24]*v[8,10,24]*v[14,15,22]*v[19,21,22]*v[2,1,11]*v[8,9,11]*
                            v_single[4,3,23]*v_single[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3]
            #
            elseif  nStep%2 == 0 
                if impurity == true
                    v_single = compute_isometry_symmetric_two_tensor_impurity(A,A_single,position="rightup")
                end
                @tensor A_single_new[:] := A_single[13,12,16,17]*A[15,14,16,18]*A[1,2,5,7]*A[3,4,5,6]*v_single[12,13,24]*v_single[8,10,24]*v[14,15,22]*v[19,21,22]*v[2,1,11]*v[8,9,11]*
                        v[4,3,23]*v[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3]

            elseif  nStep%2 == 1
                @tensor A_single[:] := A_single[1,2,3,4]*x[-2,2]*x[-4,4]*z[-1,1]*z[-3,3]
                
                if impurity == true
                    v_single = compute_isometry_symmetric_two_tensor_impurity(A,A_single,position="leftdown")
                end
                
                @tensor A_single_new[:] := A[13,12,16,17]*A[15,14,16,18]*A[1,2,5,7]*A_single[5,6,3,4]*v[12,13,24]*v[8,10,24]*v[14,15,22]*v[19,21,22]*v[2,1,11]*v[8,9,11]*
                            v_single[4,3,23]*v_single[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3]
            #
            #=
            elseif nStep%2 ==0
                
                @tensor A_single[:] := A_single[1,-2,2,-4]*z[-1,1]*z[-3,2]
                
                if impurity == true
                    v_single = compute_isometry_symmetric_two_tensor_impurity(A,A_single,position="leftup")
                end
                @tensor A_single_new[:] := A[13,12,16,17]*A[15,14,16,18]*A_single[1,7,5,2]*A[3,4,5,6]*v[12,13,24]*v[8,10,24]*v[14,15,22]*v[19,21,22]*v_single[2,1,11]*v_single[8,9,11]*
                                v[4,3,23]*v[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3]
            elseif nStep%2 ==1
                
                @tensor A_single[:] := A_single[-1,1,-3,2]*x[-2,1]*x[-4,2]
                
                if impurity == true
                    v_single = compute_isometry_symmetric_two_tensor_impurity(A,A_single,position="rightdown")
                end
                
                @tensor A_single_new[:] := A[13,12,16,17]*A_single[16,14,15,18]*A[1,2,5,7]*A[3,4,5,6]*v[12,13,24]*v[8,10,24]*v_single[14,15,22]*v_single[19,21,22]*v[2,1,11]*v[8,9,11]*
                            v[4,3,23]*v[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3]
            =#
            end
            =#


        #=
        A_double_new_1 = renormalize_fragment_one_impurity(A,A_double,v,v_double,y,w)
        A_double_new_2 = renormalize_fragment_two_impurity(A,A_single,A_single,v,v_single,v_single,y,w)        
        A_triple_new_1 = renormalize_fragment_one_impurity(A,A_triple,v,v_triple,y,w)
        A_triple_new_2 = renormalize_fragment_two_impurity(A,A_double,A_single,v,v_double,v_single,y,w)
        A_triple_new_3 = renormalize_fragment_two_impurity(A,A_single,A_double,v,v_single,v_double,y,w)
        A_triple_new_4 = renormalize_fragment_three_impurity(A,A_single,A_single,A_single,v,v_single,v_single,v_single,y,w)

        A_quadruple_new_1 = renormalize_fragment_one_impurity(A,A_quadruple,v,v_quadruple,y,w)
        A_quadruple_new_2 = renormalize_fragment_two_impurity(A,A_triple,A_single,v,v_triple,v_single,y,w)
        A_quadruple_new_3 = renormalize_fragment_two_impurity(A,A_single,A_triple,v,v_single,v_triple,y,w)
        A_quadruple_new_4 = renormalize_fragment_two_impurity(A,A_double,A_double,v,v_double,v_double,y,w)
        A_quadruple_new_5 = renormalize_fragment_three_impurity(A,A_single,A_single,A_double,v,v_single,v_single,v_double,y,w)
        A_quadruple_new_6 = renormalize_fragment_three_impurity(A,A_single,A_double,A_single,v,v_single,v_double,v_single,y,w)
        A_quadruple_new_7 = renormalize_fragment_three_impurity(A,A_double,A_single,A_single,v,v_double,v_single,v_single,y,w)
        A_quadruple_new_8 = renormalize_fragment_four_impurity(A,A_single,A_single,A_single,A_single,v,v_single,v_single,v_single,v_single,y,w)
        =#

        A = deepcopy(A_new)
        A_single = deepcopy(A_single_new)
        #=
        A_double = deepcopy(A_double_new_1+A_double_new_2*2)
        A_triple = deepcopy(A_triple_new_1 + (A_triple_new_2+A_triple_new_3)*3+A_triple_new_4*6)
        A_quadruple = deepcopy(A_quadruple_new_1+(A_quadruple_new_2+A_quadruple_new_3)*4+6*A_quadruple_new_4+
                        (A_quadruple_new_5+A_quadruple_new_6+A_quadruple_new_7)*12+A_quadruple_new_8*24)
        =#
        Result["logAnorm"] = log(maximum(A))+4*Result["logAnorm"];
        #=
        A_double = A_double/maximum(A)/(4*1+6*2)
        A_triple = A_triple/maximum(A)/(4*1+12*3+4*6)
        A_quadruple = A_quadruple/maximum(A)/(4*1+12*4+6*6+12*12+1*24)
        =#
        #A_single = A_single/maximum(A)/(4*1)
        A_single = A_single/maximum(A)
        A = A/maximum(A)
        append!(Result["Amatrix"],[A])

        @tensor Z[:] := A[1,2,1,2]
        @tensor Z_single[:] := A_single[1,2,1,2]   
        #=
        @tensor Z_double[:] := A_double[1,2,1,2]
        @tensor Z_triple[:] := A_triple[1,2,1,2]
        @tensor Z_quadruple[:] := A_quadruple[1,2,1,2]
        =#
        fe = -1*Temperature*(log(Z[1])+Result["logAnorm"])/4^(nStep+1);
        append!(Result["FreeEnergy"],fe)
        append!(Result["Magnetization"],Z_single[1]/Z[1])
        
        #=
        append!(Result["Second_Magnetization"],Z_double[1]/Z[1])
        append!(Result["Third_Magnetization"],Z_triple[1]/Z[1])
        append!(Result["Fourth_Magnetization"],Z_quadruple[1]/Z[1])
        append!(Result["g4"],(Z_quadruple[1]*Z[1]-3*Z_double[1]^2)/Z_double[1]^2)
        append!(Result["Binder"],Z_quadruple[1]*Z[1]/Z_double[1]^2)
        =#
        #=
        @tensor A_temp[:] := A[-1,1,-2,1]
        @tensor A_single_temp[:] := A_single[-1,1,-2,1]
        Rlt_temp = eigsolve(A_temp)
        @tensor A_single_temp[:] := A_single[-1,1,-2,1]
        Rlt_single_temp = eigsolve(A_single_temp)
        append!(Result["Magnetization"],(Rlt_single_temp[1]/Rlt_temp[1][1])[1])
        println(" Magnetization: ",(abs.((Rlt_single_temp[1]/Rlt_temp[1][1])[1]) -M)/M)
        =#
        
        #println("RG step: $nStep : ",fe," Magnetization: ",Z_single[1]/Z[1])

    end


    append!(FreeEnergy,Result["FreeEnergy"][20])
    append!(Magnetization,Result["Magnetization"][end])
    #=
    append!(Second_Magnetization,Result["Second_Magnetization"][end])
    append!(Third_Magnetization,Result["Third_Magnetization"][end])
    append!(Fourth_Magnetization,Result["Fourth_Magnetization"][end])
    append!(Binder,Result["Binder"][end])
    append!(g4,Result["g4"][end])
    =#
    #@save "Square_Ising_0.9998Tc_HOTRG_symmetric_chimax_$(chimax).jld2" Result
    #@save "Square_Ising_HOTRG_symmetric_chimax_$(chimax).jld2" Result
end

x = 1