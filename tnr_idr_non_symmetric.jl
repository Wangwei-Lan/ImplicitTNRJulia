using TensorOperations
using LinearAlgebra
using Printf
using JLD2
#using Revise

include("./tnr_idr_func.jl")
include("/home/wangwei/Dropbox/Code/WESTLAKE/TensorNetwork/VERSION1/2DClassical/partition.jl")
include("../J1J2/j1j2.jl")
FreeEnergyRelativeError =[]



relT = 1.0
println("Relative Temperature $relT")
# load parameter 
Temperature = 2/log(1+sqrt(2))
#relT = 0.90
Temperature = relT*Temperature



FEerror =[]
FE_exact =[]
FE_TNR =[]
Mag_TNR =[]

Mag = 0.0;
J1 = -1.0; J2 = 0.0;

#for Temperature in 2.20:0.001:2.30
    
    Temperature = 2/log(1+sqrt(2))
    relT = 0.995
    Temperature = relT*Temperature
    MagMatrix =[]
    println("======================================Temperature is $Temperature ============================================")    
    Result = Dict{String,Any}()
    chimax = 12
    
    feexact = ComputeFreeEnergy(1/Temperature)
    append!(FE_exact,feexact)
    push!(Result,"logTanorm"=>0.0)
    push!(Result,"logTbnorm"=>0.0)
    push!(Result,"sLMatrix"=>[])
    push!(Result,"sRMatrix"=>[])
    push!(Result,"wLMatrix"=>[])
    push!(Result,"wRMatrix"=>[])
    push!(Result,"disMatrix"=>[])
    push!(Result,"wMatrix"=>[])
    push!(Result,"yMatrix"=>[])
    push!(Result,"FreeEnergyExact"=>feexact)
    push!(Result,"FreeEnergy"=>[])
    push!(Result,"FreeEnergyErr"=>[])
    push!(Result,"Amatrix"=>[])


    # initial the local tensor
    #=
    betaval = 1/Temperature;
    Jtemp = zeros(2,2,2); Jtemp[1,1,1]=1; Jtemp[2,2,2]=1;
    Etemp = [exp(betaval) exp(-betaval);exp(-betaval) exp(betaval)];
    
    @tensor Ainit[:]:= Jtemp[-1,8,1]*Jtemp[-2,2,3]*Jtemp[-3,4,5]*Jtemp[-4,6,7]*Etemp[1,2]*Etemp[3,4]*Etemp[5,6]*Etemp[7,8];
    Xloc = (1/sqrt(2))*[1 1;1 -1];
    @tensor A0[:] := Ainit[1,2,3,4]*Xloc[1,-1]*Xloc[2,-2]*Xloc[3,-3]*Xloc[4,-4];

    sx =[0.0 1.0; 1.0 0.0]
    sz =[1.0 0.0; 0.0 -1.0]
    A = deepcopy(A0)
    #@tensor Apos[:] := A0[-1,-2,1,-4]*sz[1,-3]
    @tensor Apos[:] := A0[-1,1,-3,-4]*sx[1,-2]
    =#


    #
    E0=-0.0                              # used to limit maximum number
    A0 = zeros(2,2,2,2)
    A0 = A0 .+ exp(-(0-E0)/Temperature)
    A0[1,1,1,1]=A0[2,2,2,2]= exp(-(4-E0)/Temperature)
    A0[1,2,1,2]=A0[2,1,2,1]= exp(-(-4-E0)/Temperature)
    delta = zeros(2,2)
    delta[1,1] = -1.0; delta[2,2]=1.0
    sx =[0.0 1.0; 1.0 0.0]

    #@tensor Apos0[:] := A0[-1,-2,-3,1]*delta[1,-4]
    #@tensor Apos0[:] := A0[1,-2,-3,-4]*delta[1,-1]
    #@tensor Apos0[:] := A0[-1,-2,1,-4]*delta[1,-3]
    @tensor Apos0[:] := A0[1,-2,-3,-4]*delta[1,-1]
    A = deepcopy(A0)
    Apos = deepcopy(Apos0)

    #
    @tensor Apos[:] := Apos[-1,-3,2,1]*A[-2,1,4,-7]*A[2,-4,-5,3]*A[4,3,-6,-8]+
                    A[-1,-3,2,1]*Apos[-2,1,4,-7]*A[2,-4,-5,3]*A[4,3,-6,-8]+
                    A[-1,-3,2,1]*A[-2,1,4,-7]*Apos[2,-4,-5,3]*A[4,3,-6,-8]+
                    A[-1,-3,2,1]*A[-2,1,4,-7]*A[2,-4,-5,3]*Apos[4,3,-6,-8];
    @tensor A[:] := A[-1,-3,2,1]*A[-2,1,4,-7]*A[2,-4,-5,3]*A[4,3,-6,-8];
    A = reshape(A,4,4,4,4);
    Apos =reshape(Apos/4,4,4,4,4);
    #

    #=
    A0,Apos0 = ConstructHamiltonian(J1,J2,Temperature,Mag);
    @tensor A[:] := A0[-1,-2,1,-5]*A0[1,-3,-4,-6];
    #@tensor Apos[:] := Apos0[1,-3,-4,-6]*A0[-1,-2,1,-5];
    #@tensor Apos[:] := A0[1,-3,-4,-6]*Apos0[-1,-2,1,-5];
    @tensor Apos[:] := Apos0[1,-3,-4,-6]*A0[-1,-2,1,-5]+A0[1,-3,-4,-6]*Apos0[-1,-2,1,-5];
    
    
    A = reshape(A,4,4,4,4);
    Apos = reshape(Apos/2,4,4,4,4);

    @tensor Apos[:] := Apos[-1,-3,2,1]*A[-2,1,4,-7]*A[2,-4,-5,3]*A[4,3,-6,-8]+
                        A[-1,-3,2,1]*Apos[-2,1,4,-7]*A[2,-4,-5,3]*A[4,3,-6,-8]+
                        A[-1,-3,2,1]*A[-2,1,4,-7]*Apos[2,-4,-5,3]*A[4,3,-6,-8]+
                        A[-1,-3,2,1]*A[-2,1,4,-7]*A[2,-4,-5,3]*Apos[4,3,-6,-8];
    @tensor A[:] := A[-1,-3,2,1]*A[-2,1,4,-7]*A[2,-4,-5,3]*A[4,3,-6,-8];
    A = reshape(A,16,16,16,16);
    Apos =reshape(Apos/4,16,16,16,16);
    =#    

    #=
    beta = 1/Temperature
    Jtemp = zeros(2,2,2,2); Jtemp[1,1,1,1]=1.0; Jtemp[2,2,2,2]=1.0;
    Jtemp_impurity = zeros(2,2,2,2); Jtemp_impurity[1,1,1,1] = -1.0; Jtemp_impurity[2,2,2,2] = 1.0

    Ltemp = [exp(-J1*beta) exp(J1*beta); exp(J1*beta) exp(-J1*beta)];
    Etemp = sqrt(Ltemp)

    @tensor A0[:] := Jtemp[1,2,3,4]*Etemp[1,-1]*Etemp[2,-2]*Etemp[3,-3]*Etemp[4,-4]  
    @tensor Apos0[:] := Jtemp_impurity[1,2,3,4]*Etemp[1,-1]*Etemp[2,-2]*Etemp[3,-3]*Etemp[4,-4]  
    A = deepcopy(real(A0))
    Apos = deepcopy(real(Apos0))
    #Apos = deepcopy(imag(Apos0))
    
    #
    @tensor Apos[:] := Apos[-1,-3,2,1]*A[-2,1,4,-7]*A[2,-4,-5,3]*A[4,3,-6,-8]+
                    A[-1,-3,2,1]*Apos[-2,1,4,-7]*A[2,-4,-5,3]*A[4,3,-6,-8]+
                    A[-1,-3,2,1]*A[-2,1,4,-7]*Apos[2,-4,-5,3]*A[4,3,-6,-8]+
                    A[-1,-3,2,1]*A[-2,1,4,-7]*A[2,-4,-5,3]*Apos[4,3,-6,-8];
    @tensor A[:] := A[-1,-3,2,1]*A[-2,1,4,-7]*A[2,-4,-5,3]*A[4,3,-6,-8];
    A = reshape(A,4,4,4,4);
    Apos =reshape(Apos/4,4,4,4,4);
    #
    @tensor Apos[:] := Apos[-1,-3,2,1]*A[-2,1,4,-7]*A[2,-4,-5,3]*A[4,3,-6,-8]
    @tensor A[:] := A[-1,-3,2,1]*A[-2,1,4,-7]*A[2,-4,-5,3]*A[4,3,-6,-8];
    A = reshape(A,16,16,16,16);
    Apos =reshape(Apos/4,16,16,16,16);
    =#


    Ta = deepcopy(A);
    Tb = deepcopy(A);
    Result["logTanorm"] = log(maximum(Ta))+4*Result["logTanorm"];
    Result["logTbnorm"] = log(maximum(Tb))+4*Result["logTbnorm"];
    Apos = Apos/maximum(Ta);
    Ta = Ta/maximum(Ta);
    Tb = Tb/maximum(Tb);


    #for chimax in 10:2:10
    
        #@tensor Z[:] := Ta[1,5,2,6]*Ta[3,6,4,5]*Ta[2,7,1,8]*Ta[4,8,3,7];
        #fe = -1*Temperature*(log(Z[1])+2*Result["logTanorm"]+2*Result["logTbnorm"])/16


        @tensor Z[:] := Ta[1,2,1,2]
        fe = -1*Temperature*(log(Z[1])+Result["logTanorm"])/4;
        println("Free Energy: ",fe);
        append!(Result["FreeEnergy"],fe);
        append!(Result["FreeEnergyErr"],(fe-feexact)/feexact);

        RGstep = 30
        UpdateLoop = 3000   
        for k in 1:RGstep


            global Ta,Tb,Apos,Z,fe
            global Mag_pos1,Mag_pos2,Mag_pos3,Mag_pos4
            println("RGstep $k")
            if k == 1
                #wLup,wRup,sLup,sRup,wLdn,wRdn,sLdn,sRdn,dis = isometry_initial(Ta,Tb,chimax)
                wLup,wRup,sLup,sRup,wLdn,wRdn,sLdn,sRdn,dis = isometry_initial(Ta,Tb,chimax,rg=true)
                wLup_imp,wRup_imp,sLup_imp,sRup_imp,wLdn_imp,wRdn_imp,sLdn_imp,sRdn_imp,dis_imp=isometry_impurity_initial(Ta,Tb,Apos,chimax;rg = true)
            else
                #wLup,wRup,sLup,sRup,wLdn,wRdn,sLdn,sRdn,dis = isometry_initial(Ta,Tb,chimax)
                wLup,wRup,sLup,sRup,wLdn,wRdn,sLdn,sRdn,dis = isometry_initial(Ta,Tb,chimax,rg=true)
                wLup_imp,wRup_imp,sLup_imp,sRup_imp,wLdn_imp,wRdn_imp,sLdn_imp,sRdn_imp,dis_imp=isometry_impurity_initial(Ta,Tb,Apos,chimax;rg = true)
            end
            println("size wLup:  ",size(wLup))
            println(" Initial Truncation Error")
            err = compute_fidelity_non_symmetric(Ta,Tb,wLup,wRup,sLup,sRup,wLdn,wRdn,sLdn,sRdn,dis;printdetail=true)
            compute_fidelity_non_symmetric_impurity(Ta,Tb,Apos,wLup,wRup,sLup,sRup,wLdn,wRdn,sLdn,sRdn,dis;printdetail=true)


            for j in 1:UpdateLoop
                if j %100 ==0 || j == 1
                    println("update loop : $j ")
                    err = compute_fidelity_non_symmetric(Ta,Tb,wLup,wRup,sLup,sRup,wLdn,wRdn,sLdn,sRdn,dis;printdetail=true)
                    compute_fidelity_non_symmetric_impurity(Ta,Tb,Apos,wLup,wRup,sLup,sRup,wLdn,wRdn,sLdn,sRdn,dis;printdetail=true)
                    if abs.(err) <1.0e-15
                        break
                    end 
                end        

                #if j > 2800
                #    sLup = update_sLup_non_symmetric_cg(Ta,Tb,wLup,wRup,sLup,sRup,dis)
                #else
                #sLup = update_sLup_non_symmetric(Ta,Tb,wLup,wRup,sLup,sRup,dis)
                #sRup = update_sRup_non_symmetric(Ta,Tb,wLup,wRup,sLup,sRup,dis)
                #end
                
                sLup = update_sLup_non_symmetric_impurity(Ta,Tb,Apos,wLup,wRup,sLup,sRup,wLup_imp,wRup_imp,sLup_imp,sRup_imp,dis)
                sRup,sRup_imp = update_sRup_non_symmetric_impurity(Ta,Tb,Apos,wLup,wRup,sLup,sRup,wLup_imp,wRup_imp,sLup_imp,sRup_imp,dis)

                sLdn = update_sLdn_non_symmetric(Ta,Tb,wLdn,wRdn,sLdn,sRdn,dis)
                sRdn = update_sRdn_non_symmetric(Ta,Tb,wLdn,wRdn,sLdn,sRdn,dis)

                #if j > 10
                    #wLup,S = update_wLup_non_symmetric(Ta,Tb,wLup,wRup,sLup,sRup,dis)
                    #wRup,S = update_wRup_non_symmetric(Ta,Tb,wLup,wRup,sLup,sRup,dis)
                    
                    wLup,S = update_wLup_non_symmetric_impurity(Ta,Tb,Apos,wLup,wRup,sLup,sRup,wLup_imp,wRup_imp,sLup_imp,sRup_imp,dis)
                    wRup,wRup_imp = update_wRup_non_symmetric_impurity(Ta,Tb,Apos,wLup,wRup,sLup,sRup,wLup_imp,wRup_imp,sLup_imp,sRup_imp,dis)

                    wLdn,S = update_wLdn_non_symmetric(Ta,Tb,wLdn,wRdn,sLdn,sRdn,dis)
                    wRdn,S = update_wRdn_non_symmetric(Ta,Tb,wLdn,wRdn,sLdn,sRdn,dis)
                    #dis,S =  update_dis_non_symmetric(Ta,Tb,wLup,wRup,sLup,sRup,wLdn,wRdn,sLdn,sRdn,dis)

                    #
                    wLup_imp = deepcopy(wLup)
                    #wRup_imp = deepcopy(wRup)
                    wLdn_imp = deepcopy(wLdn)
                    wRdn_imp = deepcopy(wRdn)
        
                    sLup_imp = deepcopy(sLup)
                    #sRup_imp = deepcopy(sRup)            
                    sLdn_imp = deepcopy(sLdn)
                    sRdn_imp = deepcopy(sRdn)
                    #
                    dis,S = update_dis_non_symmetric_impurity(Ta,Tb,Apos,wLup,wRup,sLup,sRup,wLdn,wRdn,sLdn,sRdn,
                                wLup_imp,wRup_imp,sLup_imp,sRup_imp,wLdn_imp,wRdn_imp,sLdn_imp,sRdn_imp,dis)
                #end
            end


            #
            wLup_imp = deepcopy(wLup)
            #wRup_imp = deepcopy(wRup)
            wLdn_imp = deepcopy(wLdn)
            wRdn_imp = deepcopy(wRdn)

            sLup_imp = deepcopy(sLup)
            #sRup_imp = deepcopy(sRup)            
            sLdn_imp = deepcopy(sLdn)
            sRdn_imp = deepcopy(sRdn)
            #
            #Ta_new,Apos_new,y,w = renormalize(Ta,Tb,Apos,wLup,wRup,sLup,sRup,wLdn,wRdn,sLdn,sRdn,dis,chimax,rg=true)
            #Ta_new,Apos_new,y,w = renormalize(Ta,Tb,Apos,wLup,wRup,sLup,sRup,wLdn,wRdn,sLdn,sRdn,dis,chimax)
            
            Ta_new,Apos_new,y,w =renormalize_impurity(Ta,Tb,Apos,wLup,wRup,sLup,sRup,wLdn,wRdn,sLdn,sRdn,dis,
                        wLup_imp,wRup_imp,sLup_imp,sRup_imp,wLdn_imp,wRdn_imp,sLdn_imp,sRdn_imp,dis_imp,chimax;rg=true)
            append!(Result["Amatrix"],[Ta_new])
            append!(Result["yMatrix"],[y])
            append!(Result["wMatrix"],[w])

            Ta = deepcopy(Ta_new)
            Tb = deepcopy(Ta_new)
            Apos = deepcopy(Apos_new)            
            Result["logTanorm"] = log(maximum(Ta))+4*Result["logTanorm"];
            Result["logTbnorm"] = log(maximum(Tb))+4*Result["logTbnorm"];
            Apos = Apos/maximum(Ta)
            Ta = Ta/maximum(Ta)
            Tb = Tb/maximum(Tb)


            #@tensor Z[:] := Ta[1,5,2,6]*Ta[3,6,4,5]*Ta[2,7,1,8]*Ta[4,8,3,7]
            #fe = -1*Temperature*(log(Z[1])+2*Result["logTanorm"]+2*Result["logTbnorm"])/(16*4^k)
            @tensor Z[:] := Ta[1,2,1,2]
            fe = -1*Temperature*(log(Z[1])+Result["logTanorm"])/(4*4^k)
            println(Z[1])
        
            println("Free Energy: ",fe)
            append!(Result["FreeEnergy"],fe)
            append!(Result["FreeEnergyErr"],(fe-feexact)/feexact)

            #
            @tensor Z[:] := Ta[1,5,2,6]*Ta[3,6,4,5]*Ta[2,7,1,8]*Ta[4,8,3,7]
            @tensor Mag_pos1[:] := Apos[1,5,2,6]*Ta[3,6,4,5]*Ta[2,7,1,8]*Ta[4,8,3,7]
            @tensor Mag_pos2[:] := Ta[1,5,2,6]*Apos[3,6,4,5]*Ta[2,7,1,8]*Ta[4,8,3,7]
            @tensor Mag_pos3[:] := Ta[1,5,2,6]*Ta[3,6,4,5]*Apos[2,7,1,8]*Ta[4,8,3,7]
            @tensor Mag_pos4[:] := Ta[1,5,2,6]*Ta[3,6,4,5]*Ta[2,7,1,8]*Apos[4,8,3,7]
            #
            #@tensor Mag_pos1[:] := Apos[1,2,1,2]
            println("Magnetism: ",Mag_pos1[1]/Z[1])
            append!(MagMatrix,(Mag_pos1[1]+Mag_pos2[1]+Mag_pos3[1]+Mag_pos4[1])/Z[1]/4)
            #append!(MagMatrix,(Mag_pos1[1])/Z[1])
        end

        @tensor Z[:] := Ta[1,5,2,6]*Ta[3,6,4,5]*Ta[2,7,1,8]*Ta[4,8,3,7]
        @tensor Mag_pos1[:] := Apos[1,5,2,6]*Ta[3,6,4,5]*Ta[2,7,1,8]*Ta[4,8,3,7]
        @tensor Mag_pos2[:] := Ta[1,5,2,6]*Apos[3,6,4,5]*Ta[2,7,1,8]*Ta[4,8,3,7]
        @tensor Mag_pos3[:] := Ta[1,5,2,6]*Ta[3,6,4,5]*Apos[2,7,1,8]*Ta[4,8,3,7]
        @tensor Mag_pos4[:] := Ta[1,5,2,6]*Ta[3,6,4,5]*Ta[2,7,1,8]*Apos[4,8,3,7]

        append!(FE_TNR,Result["FreeEnergy"])
        append!(FEerror,Result["FreeEnergyErr"])
        append!(Mag_TNR,MagMatrix)



    #end

#end




x = 1