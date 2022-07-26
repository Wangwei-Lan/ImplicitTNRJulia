"""
	Use Implicit Disentangler to optimize Tensor Network Renormalization. Based on arXiv:1707.05770


	Author: Wangwei Lan
"""

using TensorOperations
using LinearAlgebra
using Printf
using JLD2
#using Revise

include("./ImplicitTNR_func.jl")
include("/home/wangwei/Dropbox/Code/WESTLAKE/TensorNetwork/VERSION1/2DClassical/partition.jl")


# Define relative Temperature for 2D classical Ising model
relT = 1.0
println("Relative Temperature $relT")


# define temperature
Temperature = 2/log(1+sqrt(2))
Temperature = relT*Temperature


#!
#---------------------------------initial A Tensor----------------------------
#!
E0=-0.0                              # used to limit maximum number
A0 = zeros(2,2,2,2)
A0 = A0 .+ exp(-(0-E0)/Temperature)
A0[1,1,1,1]=A0[2,2,2,2]= exp(-(4-E0)/Temperature)
A0[1,2,1,2]=A0[2,1,2,1]= exp(-(-4-E0)/Temperature)
    


#=
#
#! Alternative way to construct local tensor 
#
betaval = 1/Temperature;
Jtemp = zeros(2,2,2); Jtemp[1,1,1]=1; Jtemp[2,2,2]=1;
Etemp = [exp(betaval) exp(-betaval);exp(-betaval) exp(betaval)];

@tensor Ainit[:]:= Jtemp[-1,8,1]*Jtemp[-2,2,3]*Jtemp[-3,4,5]*Jtemp[-4,6,7]*Etemp[1,2]*Etemp[3,4]*Etemp[5,6]*Etemp[7,8];
Xloc = (1/sqrt(2))*[1 1;1 -1];
@tensor Ainit[:] := Ainit[1,2,3,4]*Xloc[1,-1]*Xloc[2,-2]*Xloc[3,-3]*Xloc[4,-4];
=#


#!
#! Construct new local tensor from 2x2 Unit Cell
#!
@tensor A[:] := A0[-1,-3,2,1]*A0[-5,-4,2,3]*A0[-2,1,4,-7]*A0[-6,3,4,-8];
A = reshape(A,4,4,4,4)



#!
#!	loop over maximum bond dimension
#!
for chimax in 26:2:26


    #!	
    #!
    #!	Output results into Dict Result
    #!
    #!
    Result = Dict{String,Any}()
    push!(Result,"logTanorm"=>0.0)
    push!(Result,"logTbnorm"=>0.0)
    push!(Result,"sLMatrix"=>[])
    push!(Result,"sRMatrix"=>[])
    push!(Result,"wLMatrix"=>[])
    push!(Result,"wRMatrix"=>[])
    push!(Result,"disMatrix"=>[])
    push!(Result,"wMatrix"=>[])
    push!(Result,"yMatrix"=>[])



    #!
    #!  copy local tensor to A,B sublattice
    #!
    Ta = deepcopy(A);Tb = deepcopy(Ta);
    # log(maximum(Ta)) are stored to calculate free energy
    Result["logTanorm"] = log(maximum(Ta));
    Result["logTbnorm"] = log(maximum(Tb));
    Ta = Ta/maximum(Ta);
    Tb = Tb/maximum(Tb);
    #

	
    # Exact Free Energy Density for Infinite systems 
    feexact = ComputeFreeEnergy(1/Temperature)

    # free energy density at first layer
    @tensor Z[:] := Ta[1,8,2,6]*Tb[3,6,4,8]*Tb[4,5,3,7]*Ta[2,7,1,5]
    fe = -1*Temperature*(log(Z[1])+2*Result["logTanorm"]+2*Result["logTbnorm"])/(4^2*2)
    println("Free Energy: ",fe)


    push!(Result,"FreeEnergyExact"=>feexact)
    push!(Result,"FreeEnergy"=>[])
    push!(Result,"FreeEnergyErr"=>[])
    append!(Result["FreeEnergy"],fe)
    append!(Result["FreeEnergyErr"],(fe-feexact)/feexact)



    RGstep = 15
    UpdateLoop = 2000
    for k in 1:RGstep
	
	#
        #global Ta,Tb
        println("RGstep $k")

	#!
	#! define bond dimension for each tensor 
	#!
        chiTa1 = size(Ta,1);chiTa2 = size(Ta,2)
        chiTb4 = size(Tb,4);chiTb1 = size(Tb,1)
        chiTb2 = size(Tb,2)
        chikept = min(chiTa1*chiTa2,chimax)



        # initial wL,sL & wR,sR
        @tensor temp[:] := Ta[-2,-1,13,14]*Ta[2,1,13,5]*Ta[2,1,11,6]*Ta[-4,-3,11,12]*Tb[8,14,9,7]*Tb[3,5,9,4]*Tb[3,6,10,4]*Tb[8,12,10,7]
        temp = reshape(temp,chiTa1*chiTa2,chiTa1*chiTa2)
        Rlt = svd(temp)
        sL = deepcopy(reshape(Rlt.U[:,1:chikept],chiTa2,chiTa1,chikept))
        wL = deepcopy(sL)

        @tensor temp[:] := Ta[7,8,9,11]*Ta[2,1,9,5]*Ta[2,1,10,6]*Ta[7,8,10,13]*Tb[-2,11,12,-1]*Tb[3,5,12,4]*Tb[3,6,14,4]*Tb[-4,13,14,-3]
        temp = reshape(temp,chiTb1*chiTb2,chiTb1*chiTb2)
        Rlt = svd(temp)
        sR = deepcopy(reshape(Rlt.U[:,1:chikept],chiTb4,chiTb1,chikept))
        wR = deepcopy(sR)
        dis = reshape(eye(chiTa1*chiTb1),chiTa1,chiTb1,chiTa1,chiTb1)
        
	#!
	#! Update isometries and disentanglers
	#!
        for j in 1:UpdateLoop

            # global wL,wR,sL,sR,dis
            if j %100 ==0 || j == 1
                print("update loop : $j ")
                err = compute_fidelity(Ta,Tb,wL,wR,sL,sR,dis,printdetail=true)
                if abs.(err) <1.0e-14
                    break
                end 
            end
            
            sL = update_sL(Ta,Tb,wL,wR,sL,sR,dis)
            sR = update_sR(Ta,Tb,wL,wR,sL,sR,dis)
	    # S is the singular values, used to check convergence in some cases 
            wL,S = update_wL(Ta,Tb,wL,wR,sL,sR,dis)
            wR,S = update_wR(Ta,Tb,wL,wR,sL,sR,dis)
            dis,S = update_dis(Ta,Tb,wL,wR,sL,sR,dis)
            #
        end
        #

        #!
	#!   update tensor y 
	#!
        @tensor sT[:] := Ta[7,6,10,11]*Ta[9,8,10,12]*Tb[1,11,5,2]*Tb[4,12,5,3]*sL[6,7,-2]*sL[8,9,-1]*sR[2,1,-3]*sR[3,4,-4]
        @tensor wLwR[:] := wL[1,4,-2]*wL[2,4,-3]*wR[1,3,-1]*wR[2,3,-4]

        #@tensor Env_y[:] := sT[5,3,1,2]*sT[6,4,1,2]*sT[-1,-2,7,8]*sT[-3,-4,9,10]*wLwR[7,3,4,9]*wLwR[8,5,6,10]
        @tensor Env_y[:] := sT[5,3,1,2]*sT[6,4,1,2]*wLwR[12,3,4,19]*wLwR[13,5,6,20]*Tb[7,-2,11,8]*Tb[9,-1,11,10]*
                        sR[8,7,12]*sR[10,9,13]*Tb[15,-4,18,14]*Tb[17,-3,18,16]*sR[14,15,19]*sR[16,17,20]
        sizeEnv_y = size(Env_y)
        println(norm(Env_y-permutedims(Env_y,[2,1,4,3])))
        println(norm(Env_y-permutedims(Env_y,[3,4,1,2])))
        Env_y  = reshape(Env_y,prod(sizeEnv_y[1:2]),prod(sizeEnv_y[3:end]))
        Rlt = svd(Env_y)
        chitemp = sum(Rlt.S/maximum(Rlt.S) .> 1.0e-12)
        #chikept = min(chimax,chiTb2^2,chitemp)
        chikept = min(chimax,chiTb2^2)
        y = reshape(Rlt.U[:,1:chikept],chiTb2,chiTb2,chikept)
        println("chikept: ",chikept)
        println(norm(tr(Env_y)-tr(Env_y*Rlt.U[:,1:chikept]*(Rlt.U[:,1:chikept])'))/norm(tr(Env_y)))


    
	#!
	#! Update tensor w
	#!
        @tensor Env_w[:] := sT[2,1,14,7]*sT[11,15,10,9]*sT[2,1,17,8]*sT[12,18,10,9]*
                    wR[13,-1,14]*wR[3,5,7]*wR[4,5,8]*wR[16,-3,17]*wL[13,-2,15]*wL[3,6,11]*wL[4,6,12]*wL[16,-4,18]
        sizeEnv_w = size(Env_w)
        Env_w = reshape(Env_w,prod(sizeEnv_w[1:2]),prod(sizeEnv_w[3:end]))
        Rlt = svd(Env_w)
        chitemp = sum(Rlt.S/maximum(Rlt.S) .> 1.0e-12)
        #chikept = min(chimax,chiTb1*chiTa1,chitemp)
        chikept = min(chimax,chiTb1*chiTa1)
        w = reshape(Rlt.U[:,1:chikept],chiTb1,chiTa1,chikept)
        println("chikept: ",chikept)
        println(norm(tr(Env_w)-tr(Env_w*Rlt.U[:,1:chikept]*(Rlt.U[:,1:chikept])'))/norm(tr(Env_w)))



        append!(Result["sLMatrix"],[sL])
        append!(Result["sRMatrix"],[sR])
        append!(Result["wLMatrix"],[wL])
        append!(Result["wRMatrix"],[wR])
        append!(Result["disMatrix"],[dis])
        append!(Result["wMatrix"],[w])
        append!(Result["yMatrix"],[y])


	#!
	#!  Coarsegrain the tensors 
	#!
        @tensor Ta[:] := Ta[13,12,16,17]*Ta[15,14,16,18]*Tb[1,7,5,2]*Tb[3,6,5,4]*sL[12,13,24]*sL[14,15,22]*sR[2,1,11]*
                            sR[4,3,23]*wL[8,10,24]*wL[19,21,22]*wR[8,9,11]*wR[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3] 
        Tb = deepcopy(Ta)
        Result["logTanorm"] = log(maximum(Ta))+4*Result["logTanorm"];
        Result["logTbnorm"] = log(maximum(Tb))+4*Result["logTbnorm"];
        Ta = Ta/maximum(Ta);
        Tb = Tb/maximum(Tb);
    
	#!
	#! Calculate free energy at each coarsegraining level
	#!
        @tensor Z[:] := Ta[1,2,1,2]
        fe = -1*Temperature*(log(Z[1])+Result["logTanorm"])/(4^k*4^2*2)
        #append!(FEerror,(fe-feexact)/feexact)



	#!
	#! Alterantive way to calculate free energy at each coarsegraining level
	#!
        @tensor Z[:] := Ta[1,8,2,6]*Tb[3,6,4,8]*Ta[4,5,3,7]*Tb[2,7,1,5]
        fe = -1*Temperature*(log(Z[1])+2*Result["logTanorm"]+2*Result["logTbnorm"])/(4^k*4^2*2)
        println("Free Energy: ",fe)
        append!(Result["FreeEnergy"],fe)
        append!(Result["FreeEnergyErr"],(fe-feexact)/feexact)
        println("Free Energy Error : ",(fe-feexact)/feexact)
        #


        #=
        @tensor Env_y[:] :=  Ta[2,1,5,11]*Ta[4,3,5,12]*Ta[6,7,10,11]*Ta[8,9,10,12]*Tb[25,-2,29,26]*Tb[27,-1,29,28]*Tb[33,-4,36,32]*Tb[35,-3,36,34]*
                            sL[1,2,14]*sL[3,4,17]*sL[7,6,15]*sL[9,8,18]*sR[26,25,30]*sR[28,27,31]*sR[32,33,38]*sR[34,35,37]*wL[20,13,14]*
                            wL[21,13,15]*wL[23,16,17]*wL[24,16,18]*wR[20,19,30]*wR[23,22,31]*wR[21,19,38]*wR[24,22,37]
        sizeEnv_y = size(Env_y)
        Env_y  = reshape(Env_y,prod(sizeEnv_y[1:2]),prod(sizeEnv_y[3:end]))
        Rlt = svd(Env_y)
        chitemp = sum(Rlt.S .> 1.0e-14)
        chikept = min(chimax,chiTb2^2,chitemp)
        #chikept = min(chimax,chiTb2^2)
        println(norm(tr(Env_y)-tr(Env_y*Rlt.U[:,1:chikept]*(Rlt.U[:,1:chikept])'))/norm(tr(Env_y)))
        y = reshape(Rlt.U[:,1:chikept],chiTb2,chiTb2,chikept)


        @tensor Env_w[:] :=  Ta[28,27,32,31]*Ta[13,12,32,16]*Ta[15,14,33,16]*Ta[30,29,33,31]*Tb[20,24,25,21]*Tb[5,9,25,6]*Tb[7,9,26,8]*Tb[22,24,26,23]*
                            sL[27,28,36]*sL[12,13,17]*sL[14,15,18]*sL[29,30,39]*sR[21,20,35]*sR[6,5,10]*sR[8,7,11]*sR[23,22,38]*wL[34,-2,36]*wL[3,2,17]*
                            wL[4,2,18]*wL[37,-4,39]*wR[34,-1,35]*wR[3,1,10]*wR[4,1,11]*wR[37,-3,38]
        sizeEnv_w = size(Env_w)
        Env_w = reshape(Env_w,prod(sizeEnv_w[1:2]),prod(sizeEnv_w[3:end]))
        Rlt = svd(Env_w)
        chitemp = sum(Rlt.S .> 1.0e-14)
        chikept = min(chimax,chiTb1*chiTa1,chitemp)
        #chikept = min(chimax,chiTb1*chiTa1)
        w = reshape(Rlt.U[:,1:chikept],chiTb1,chiTa1,chikept)
        println(norm(tr(Env_w)-tr(Env_w*Rlt.U[:,1:chikept]*(Rlt.U[:,1:chikept])'))/norm(tr(Env_w)))



        @tensor Ta[:] := Ta[13,12,16,17]*Ta[15,14,16,18]*Tb[1,7,5,2]*Tb[3,6,5,4]*sL[12,13,24]*sL[14,15,22]*sR[2,1,11]*
                            sR[4,3,23]*wL[8,10,24]*wL[19,21,22]*wR[8,9,11]*wR[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3] 
        Tb = deepcopy(Ta)
        Result["logTanorm"] = log(maximum(Ta))+4*Result["logTanorm"];
        Result["logTbnorm"] = log(maximum(Tb))+4*Result["logTbnorm"];
        Ta = Ta/maximum(Ta);
        Tb = Tb/maximum(Tb);
	=#    


    
    end


    @save "TNR_IDR_pinv_chimax_$(chimax).jld2" Result
    println(Result["FreeEnergyErr"][end])
    append!(FEerror,Result["FreeEnergyErr"][end])


end
