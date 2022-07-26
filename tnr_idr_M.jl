using TensorOperations
using LinearAlgebra
using Printf
using JLD2
#using Revise

include("./tnr_idr_func.jl")
include("/home/wangwei/Dropbox/Code/WESTLAKE/TensorNetwork/VERSION1/2DClassical/partition.jl")



FreeEnergyRelativeError =[]



#for Temperature in 2.1:0.01:2.5
#for relT in 0.90:0.02:0.98
relT = 1.0
println("Relative Temperature $relT")
# load parameter 
Temperature = 2/log(1+sqrt(2))
#relT = 0.90
Temperature = relT*Temperature
#Temperature = 2.1
#Temperature = Temp
#---------------------------------initial A Tensor----------------------------
E0=-0.0                              # used to limit maximum number
A0 = zeros(2,2,2,2)
A0 = A0 .+ exp(-(0-E0)/Temperature)
A0[1,1,1,1]=A0[2,2,2,2]= exp(-(4-E0)/Temperature)
A0[1,2,1,2]=A0[2,1,2,1]= exp(-(-4-E0)/Temperature)

#=
betaval = 1/Temperature;
Jtemp = zeros(2,2,2); Jtemp[1,1,1]=1; Jtemp[2,2,2]=1;
Etemp = [exp(betaval) exp(-betaval);exp(-betaval) exp(betaval)];

@tensor Ainit[:]:= Jtemp[-1,8,1]*Jtemp[-2,2,3]*Jtemp[-3,4,5]*Jtemp[-4,6,7]*Etemp[1,2]*Etemp[3,4]*Etemp[5,6]*Etemp[7,8];
Xloc = (1/sqrt(2))*[1 1;1 -1];
@tensor Ainit[:] := Ainit[1,2,3,4]*Xloc[1,-1]*Xloc[2,-2]*Xloc[3,-3]*Xloc[4,-4];
=#


#
@tensor A[:] := A0[-1,-3,2,1]*A0[-5,-4,2,3]*A0[-2,1,4,-7]*A0[-6,3,4,-8];
#@tensor A[:] := A0[-2,-3,1,2]*A0[-1,2,3,-7]*A0[1,-4,-6,4]*A0[3,4,-5,-8];
#@tensor A[:] := A0[-1,-3,2,1]*A0[-2,1,4,-7]*A0[2,-4,-5,3]*A0[4,3,-6,-8]
A = reshape(A,4,4,4,4)


@tensor A[:] := A[-1,-3,2,1]*A[-5,-4,2,3]*A[-2,1,4,-7]*A[-6,3,4,-8];
#@tensor A[:] := A[-2,-3,1,2]*A[-1,2,3,-7]*A[1,-4,-6,4]*A[3,4,-5,-8];
#@tensor A[:] := A[-1,-3,2,1]*A[-2,1,4,-7]*A[2,-4,-5,3]*A[4,3,-6,-8]
A = reshape(A,16,16,16,16);





FEerror =[]
#


for chimax in 10:10:100

    #chimax = 24
    println("chimax $chimax: ")
    #chiM = 20
    chiM = chimax
    println("chiM: ",chiM)
    Result = Dict{String,Any}()
    push!(Result,"logTanorm"=>0.0)
    push!(Result,"logTbnorm"=>0.0)
    #

    Ta = deepcopy(A);
    Tb = deepcopy(Ta);
    Result["logTanorm"] = log(maximum(Ta));
    Result["logTbnorm"] = log(maximum(Tb));
    Ta = Ta/maximum(Ta);
    Tb = Tb/maximum(Tb);
    #


    feexact = ComputeFreeEnergy(1/Temperature)
    @tensor Z[:] := Ta[1,2,1,2]
    fe = -1*Temperature*(log(Z[1])+Result["logTanorm"])/(4^2*2)
    #append!(FEerror,(fe-feexact)/feexact)


    @tensor Z[:] := Ta[1,8,2,6]*Tb[3,6,4,8]*Tb[4,5,3,7]*Ta[2,7,1,5]
    fe = -1*Temperature*(log(Z[1])+2*Result["logTanorm"]+2*Result["logTbnorm"])/(4^3*2)
    println("Free Energy: ",fe)


    push!(Result,"FreeEnergyExact"=>feexact)
    push!(Result,"FreeEnergy"=>[])
    push!(Result,"FreeEnergyErr"=>[])
    append!(Result["FreeEnergy"],fe)
    append!(Result["FreeEnergyErr"],(fe-feexact)/feexact)



    RGstep = 25
    UpdateLoop = 500
    for k in 1:RGstep

        #global Ta,Tb
        println("RGstep $k")
        println(size(Ta))
        chiTa1 = size(Ta,1);chiTa2 = size(Ta,2)
        chiTb4 = size(Tb,4);chiTb1 = size(Tb,1)
        chiTb2 = size(Tb,2)
        chikept = min(chiTa1*chiTa2,chimax)



        # initial
        @tensor temp[:] := Ta[-2,-1,13,14]*Ta[2,1,13,5]*Ta[2,1,11,6]*Ta[-4,-3,11,12]*Tb[8,14,9,7]*Tb[3,5,9,4]*Tb[3,6,10,4]*Tb[8,12,10,7]
        temp = reshape(temp,chiTa1*chiTa2,chiTa1*chiTa2)
        Rlt = svd(temp)
        chiMkept = min(chiTa2*chiTa1,chiM)
        qL = deepcopy(reshape(Rlt.U[:,1:chiMkept],chiTa2,chiTa1,chiMkept))
        wL = deepcopy(reshape(Rlt.U[:,1:chikept],chiTa2,chiTa1,chikept))
        sL = Matrix(1.0I,chiMkept,chikept)


        @tensor temp[:] := Ta[7,8,9,11]*Ta[2,1,9,5]*Ta[2,1,10,6]*Ta[7,8,10,13]*Tb[-2,11,12,-1]*Tb[3,5,12,4]*Tb[3,6,14,4]*Tb[-4,13,14,-3]
        temp = reshape(temp,chiTb1*chiTb2,chiTb1*chiTb2)
        Rlt = svd(temp)
        chiMkept = min(chiM,chiTb4*chiTb1)
        qR = deepcopy(reshape(Rlt.U[:,1:chiMkept],chiTb4,chiTb1,chiMkept))
        wR = deepcopy(reshape(Rlt.U[:,1:chikept],chiTb4,chiTb1,chikept))
        sR = Matrix(1.0I,chiMkept,chikept)

        dis = reshape(eye(chiTa1*chiTb1),chiTa1,chiTb1,chiTa1,chiTb1)
        #=
        for j in 1:UpdateLoop
            # global wL,wR,sL,sR,dis
            if j %100 ==0 || j == 1
                print("update loop : $j ")
                err = compute_fidelity_M(Ta,Tb,wL,wR,sL,sR,qL,qR,dis,printdetail=true)
                if abs.(err) <1.0e-14
                    break
                end 
            end
            
            sL = update_sLM(Ta,Tb,wL,wR,sL,sR,qL,qR,dis)
            sR = update_sRM(Ta,Tb,wL,wR,sL,sR,qL,qR,dis)
            if j > 50    
                wL = update_wLM(Ta,Tb,wL,wR,sL,sR,qL,qR,dis)
                wR = update_wRM(Ta,Tb,wL,wR,sL,sR,qL,qR,dis)
                dis = update_disM(Ta,Tb,wL,wR,sL,sR,qL,qR,dis)
            end
            
        end
        =#


        @tensor sT[:] := Ta[5,4,7,15]*Ta[2,1,7,16]*Tb[11,15,14,12]*Tb[8,16,14,9]*qL[4,5,6]*qL[1,2,3]*qR[12,11,13]*qR[9,8,10]*
                        sL[6,-2]*sL[3,-1]*sR[13,-3]*sR[10,-4]
        @tensor wLwR[:] := wL[1,4,-2]*wL[2,4,-3]*wR[1,3,-1]*wR[2,3,-4]


        @tensor Env_y[:] :=  sT[5,3,1,2]*sT[6,4,1,2]*wLwR[14,3,4,24]*wLwR[15,5,6,23]*Tb[7,-2,13,8]*Tb[10,-1,13,11]*qR[8,7,9]*
                        qR[11,10,12]*sR[9,14]*sR[12,15]*Tb[17,-4,22,16]*Tb[20,-3,22,19]*qR[16,17,18]*qR[19,20,21]*sR[18,24]*sR[21,23]
        sizeEnv_y = size(Env_y)
        println(norm(Env_y-permutedims(Env_y,[2,1,4,3])))
        println(norm(Env_y-permutedims(Env_y,[3,4,1,2])))
        Env_y  = reshape(Env_y,prod(sizeEnv_y[1:2]),prod(sizeEnv_y[3:end]))
        Rlt = svd(Env_y)
        chitemp = sum(Rlt.S/maximum(Rlt.S) .> 1.0e-12)
        chikept = min(chimax,chiTb2^2,chitemp)
        y = reshape(Rlt.U[:,1:chikept],chiTb2,chiTb2,chikept)
        println("chikept: ",chikept)
        println(norm(tr(Env_y)-tr(Env_y*Rlt.U[:,1:chikept]*(Rlt.U[:,1:chikept])'))/norm(tr(Env_y)))



        @tensor Env_w[:] := sT[2,1,14,7]*sT[11,15,10,9]*sT[2,1,17,8]*sT[12,18,10,9]*
                    wR[13,-1,14]*wR[3,5,7]*wR[4,5,8]*wR[16,-3,17]*wL[13,-2,15]*wL[3,6,11]*wL[4,6,12]*wL[16,-4,18]
        sizeEnv_w = size(Env_w)
        Env_w = reshape(Env_w,prod(sizeEnv_w[1:2]),prod(sizeEnv_w[3:end]))
        Rlt = svd(Env_w)
        chitemp = sum(Rlt.S/maximum(Rlt.S) .> 1.0e-12)
        chikept = min(chimax,chiTb1*chiTa1,chitemp)
        w = reshape(Rlt.U[:,1:chikept],chiTb1,chiTa1,chikept)
        println("chikept: ",chikept)
        println(norm(tr(Env_w)-tr(Env_w*Rlt.U[:,1:chikept]*(Rlt.U[:,1:chikept])'))/norm(tr(Env_w)))





        @tensor Ta[:] := Ta[15,14,20,21]*Ta[18,17,20,22]*Tb[1,8,7,2]*Tb[5,9,7,4]*qL[14,15,16]*qL[17,18,19]*sL[16,28]*sL[19,26]*
                    qR[2,1,3]*qR[4,5,6]*sR[3,13]*sR[6,27]*wR[10,11,13]*wL[10,12,28]*wR[23,24,27]*wL[23,25,26]*y[9,8,-2]*y[22,21,-4]*
                    w[24,25,-3]*w[11,12,-1]
        Tb = deepcopy(Ta)
        Result["logTanorm"] = log(maximum(Ta))+4*Result["logTanorm"];
        Result["logTbnorm"] = log(maximum(Tb))+4*Result["logTbnorm"];
        Ta = Ta/maximum(Ta);
        Tb = Tb/maximum(Tb);
    


        @tensor Z[:] := Ta[1,8,2,6]*Tb[3,6,4,8]*Ta[4,5,3,7]*Tb[2,7,1,5]
        fe = -1*Temperature*(log(Z[1])+2*Result["logTanorm"]+2*Result["logTbnorm"])/(4^k*4^3*2)
        println("Free Energy: ",fe)
        append!(Result["FreeEnergy"],fe)
        append!(Result["FreeEnergyErr"],(fe-feexact)/feexact)
        println("Free Energy: ",(fe-feexact)/feexact)
        #
    
    end

    #@save "HOTRG_TRG_chimax_$(chimax)_chiM_$(chiM)_result.jld2" Result

    println(Result["FreeEnergyErr"][end])
    append!(FEerror,Result["FreeEnergyErr"][end])


end
