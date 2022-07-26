


function update_tensor(Env,index)

    sizeEnv = size(Env)
    Env = reshape(Env,prod(sizeEnv[1:index]),prod(sizeEnv[index+1:end]))
    Env = Env/maximum(Env)
    Rlt = svd(Env)
    tensor = reshape(Rlt.U*Rlt.V',sizeEnv)
    return tensor,Rlt.S
end



function idr_update(P,B,index)

    sizeP = size(P);sizeB = size(B)
    P = reshape(P,prod(sizeP[1:index]),prod(sizeP[index+1:end]))
    B = reshape(B,prod(sizeB[1:index]),prod(sizeB[index+1:end]))
    B = (B+B')*0.5;
    P = P/maximum(B);B = B/maximum(B)
    tensor = reshape(pinv(B,rtol=5.0e-11)*P,sizeP)
    #tensor = reshape(pinv(B)*P,sizeP)
    return tensor
end


function compute_isometry_vertical_symmetric_modified(A,chimaxtemp,chimax;Aimp=0,RGstep=0)
    
    if Aimp ==0
        @tensor Env_left[:] := A[-2,-1,13,14]*A[8,14,9,7]*A[2,1,13,5]*A[3,5,9,4]*A[2,1,11,6]*A[3,6,10,4]*A[-4,-3,11,12]*A[8,12,10,7]
        @tensor Env_right[:] := A[7,8,9,11]*A[-2,11,12,-1]*A[2,1,9,5]*A[3,5,12,4]*A[2,1,10,6]*A[3,6,14,4]*A[7,8,10,13]*A[-4,13,14,-3]
    else
        @tensor Env_left[:] := Aimp[-2,-1,13,14]*A[8,14,9,7]*A[2,1,13,5]*A[3,5,9,4]*A[2,1,11,6]*A[3,6,10,4]*Aimp[-4,-3,11,12]*A[8,12,10,7]
        @tensor Env_right[:] := A[7,8,9,11]*Aimp[-2,11,12,-1]*A[2,1,9,5]*A[3,5,12,4]*A[2,1,10,6]*A[3,6,14,4]*A[7,8,10,13]*Aimp[-4,13,14,-3]
    end
    
    sizeA = size(A)
    LeftTemp = reshape(Env_left,prod(sizeA[1:2]),prod(sizeA[1:2]))
    Rlt = svd(LeftTemp)
    chitemp = sum(Rlt.S/maximum(Rlt.S) .> 1.0e-15)
    #println(Rlt.S/maximum(Rlt.S))
    println("chitemp: ",chitemp)
    #
    if RGstep <= 2
        chikept = min(chimaxtemp,prod(sizeA[1:2]))
        #chikept = prod(sizeA[1:2])
    else
        #chikept = min(chitemp,chimaxtemp,prod(sizeA[1:2]))
        chikept = min(chimaxtemp,prod(sizeA[1:2]))
    end
    #
    #chikept = min(chimaxtemp,prod(sizeA[1:2]))
    wL = reshape(Rlt.U[:,1:chikept],sizeA[2],sizeA[1],chikept)
    

    RightTemp = reshape(Env_right,prod(sizeA[3:4]),prod(sizeA[3:4]))
    Rlt = svd(RightTemp)
    chitemp = sum(Rlt.S/maximum(Rlt.S) .> 1.0e-15)
    #println(Rlt.S/maximum(Rlt.S))
    println("chitemp: ",chitemp)
    #
    if RGstep <= 2
        chikept = min(chimaxtemp,sizeA[4]*sizeA[1])
        #chikept = sizeA[4]*sizeA[1]
    else
        #chikept = min(chitemp,chimaxtemp,sizeA[4]*sizeA[1])
        chikept = min(chimaxtemp,sizeA[4]*sizeA[1])
    end
    #
    #chikept = min(chimaxtemp,sizeA[4]*sizeA[1])
    wR = reshape(Rlt.U[:,1:chikept],sizeA[4],sizeA[1],chikept)

    println(size(wL))
    println(size(wR))
    #wL = reshape(Matrix(1.0I,sizeA[2]*sizeA[1],sizeA[2]*sizeA[1]),sizeA[2],sizeA[1],sizeA[2]*sizeA[1])
    #wR = reshape(Matrix(1.0I,sizeA[4]*sizeA[3],sizeA[4]*sizeA[3]),sizeA[4],sizeA[3],sizeA[4]*sizeA[3])

    if Aimp ==0 
        @tensor Env_left_vL[:] :=  Env_left[1,2,3,4]*wL[1,2,-1]*wL[3,4,-2]
        @tensor Env_right_vR[:] := Env_right[1,2,3,4]*wR[1,2,-1]*wR[3,4,-2]
    else
        @tensor Env_left_vL[:] :=  Env_left[1,2,3,4]*wL[1,2,-1]*wL[3,4,-2]
        @tensor Env_right_vR[:] := Env_right[1,2,3,4]*wR[1,2,-1]*wR[3,4,-2]
    end


    Rlt = svd(Env_left_vL)
    chitemp = sum(Rlt.S/maximum(Rlt.S) .> 1.0e-14)
    if RGstep <= 2
        chikept = min(chimax,size(Env_left_vL,1))
    else
        chikept = min(chitemp,chimax,size(Env_left_vL,1))
    end
    vL = reshape(Rlt.U[:,1:chikept],size(Env_left_vL,1),chikept)
    
    Rlt = svd(Env_right_vR)
    chitemp = sum(Rlt.S/maximum(Rlt.S) .> 1.0e-14)
    if RGstep <=2 
        chikept = min(chimax,size(Env_right_vR,1))
    else    
        chikept = min(chitemp,chimax,size(Env_right_vR,1))
    end
    vR = reshape(Rlt.U[:,1:chikept],size(Env_right_vR,1),chikept)
    
    println(size(vL))
    println(size(vR))
    
    return wL,wR,vL,vR
end


function  compute_projector_tnr_vertical_symmetric_modified(A,wL,wR,vL,vR,sL,sR,dis,chimax)

    @tensor sLAsLA[:] := A[3,2,7,-3]*A[6,5,7,-4]*wL[2,3,1]*wL[5,6,4]*sL[1,-1]*sL[4,-2]
    @tensor sRAsRA[:] := A[2,-3,7,3]*A[6,-4,7,5]*wR[3,2,1]*wR[5,6,4]*sR[1,-1]*sR[4,-2]
    @tensor v4[:] := wL[3,7,2]*wL[6,7,4]*wR[3,8,1]*wR[6,8,5]*vL[2,-2]*vL[4,-4]*vR[1,-1]*vR[5,-3]

    @tensor Envy[:] := sRAsRA[7,8,-2,-1]*sLAsLA[3,5,1,2]*sLAsLA[4,6,1,2]*sRAsRA[9,10,-4,-3]*v4[7,3,9,4]*v4[8,5,10,6]
    sizeEnvy = size(Envy)
    Envy = reshape(Envy,prod(sizeEnvy[1:2]),prod(sizeEnvy[1:2]))
    Rlt = svd(Envy)

    chitemp = sum(Rlt.S/maximum(Rlt.S) .> 1.0e-14)
    chikept = min(chitemp,chimax,prod(sizeEnvy[1:2]))
    #chikept = min(chimax,prod(sizeEnvy[1:2]))
    println("truncation error: ",(tr(Envy)-tr(Envy*Rlt.U[:,1:chikept]*Rlt.U'[1:chikept,:]))/tr(Envy))
    y = reshape(Rlt.U[:,1:chikept],sizeEnvy[1],sizeEnvy[2],chikept)


    @tensor vLtemp[:] := wL[-1,-2,1]*vL[1,-3]
    @tensor vRtemp[:] := wR[-1,-2,1]*vR[1,-3]
    @tensor Envw[:] := sLAsLA[1,2,3,4]*sRAsRA[25,12,3,4]*sLAsLA[26,19,17,18]*sRAsRA[14,13,17,18]*
                    sLAsLA[1,2,6,5]*sRAsRA[22,11,6,5]*sLAsLA[23,20,16,15]*sRAsRA[14,13,16,15]*
                    vRtemp[24,-1,25]*vLtemp[24,-2,26]*vRtemp[9,7,12]*vLtemp[9,8,19]*vRtemp[10,7,11]*
                    vLtemp[10,8,20]*vRtemp[21,-3,22]*vLtemp[21,-4,23]
    sizeEnvw = size(Envw)
    #println(size(Envw))
    Envw = reshape(Envw,prod(sizeEnvw[1:2]),prod(sizeEnvw[1:2]))
    Rlt = svd(Envw)
    
    
    chitemp = sum(Rlt.S/maximum(Rlt.S) .> 1.0e-14)
    chikept = min(chitemp,chimax,prod(sizeEnvw[1:2]))
    #chikept = min(chimax,prod(sizeEnvw[1:2]))
    println("truncation error: ",(tr(Envw)-tr(Envw*Rlt.U[:,1:chikept]*Rlt.U'[1:chikept,:]))/tr(Envw))
    w = reshape(Rlt.U[:,1:chikept],sizeEnvw[1],sizeEnvw[2],chikept)

    return y,w

end



function  HOTRG_initial(A,A_single,A_double,A_triple,A_quadruple)

    sizeA = size(A)

    @tensor A_quadruple[:] := (A_quadruple[-1,-2,1,-5]*A[-4,-3,1,-6]+A[-1,-2,1,-5]*A_quadruple[-4,-3,1,-6]+
                                4*A_triple[-1,-2,1,-5]*A_single[-4,-3,1,-6]+4*A_single[-1,-2,1,-5]*A_triple[-4,-3,1,-6]+
                                6*A_double[-1,-2,1,-5]*A_double[-4,-3,1,-6])*0.0625
    @tensor A_triple[:] := (A_triple[-1,-2,1,-5]*A[-4,-3,1,-6]+A[-1,-2,1,-5]*A_triple[-4,-3,1,-6]+
                                3*A_double[-1,-2,1,-5]*A_single[-4,-3,1,-6]+3*A_single[-1,-2,1,-5]*A_double[-4,-3,1,-6])*0.125
    @tensor A_double[:] := (A_double[-1,-2,1,-5]*A[-4,-3,1,-6]+A[-1,-2,1,-5]*A_double[-4,-3,1,-6]+
                                2*A_single[-1,-2,1,-5]*A_single[-4,-3,1,-6])*0.25
    @tensor A_single[:] := (A_single[-1,-2,1,-5]*A[-4,-3,1,-6] + A[-1,-2,1,-5]*A_single[-4,-3,1,-6])*0.5
    @tensor A[:] := A[-1,-2,1,-5]*A[-4,-3,1,-6]

    A_quadruple = reshape(A_quadruple,sizeA[1],sizeA[2]^2,sizeA[1],sizeA[2]^2)
    A_triple= reshape(A_triple,sizeA[1],sizeA[2]^2,sizeA[1],sizeA[2]^2)
    A_double = reshape(A_double,sizeA[1],sizeA[2]^2,sizeA[1],sizeA[2]^2)
    A_single = reshape(A_single,sizeA[1],sizeA[2]^2,sizeA[1],sizeA[2]^2)
    A = reshape(A,sizeA[1],sizeA[2]^2,sizeA[1],sizeA[2]^2)

    return A,A_single,A_double,A_triple,A_quadruple

end




function renormalize_fragment_tnr_one_impurity_vertical_symmetric(A,A_imp,vL,vR,vL_imp,vR_imp,sL,sR,sL_imp,sR_imp,y,w)

    #
    @tensor A_single_new[:] := A_imp[13,12,16,17]*A[15,14,16,18]*A[1,7,5,2]*A[3,6,5,4]*sL_imp[12,13,24]*vL_imp[8,10,24]*sL[14,15,22]*vL[19,21,22]*sR[2,1,11]*vR[8,9,11]*
                            sR[4,3,23]*vR[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3]+
                            A[13,12,16,17]*A_imp[15,14,16,18]*A[1,7,5,2]*A[3,6,5,4]*sL[12,13,24]*vL[8,10,24]*sL_imp[14,15,22]*vL_imp[19,21,22]*sR[2,1,11]*vR[8,9,11]*
                            sR[4,3,23]*vR[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3]+
                            A[13,12,16,17]*A[15,14,16,18]*A_imp[1,7,5,2]*A[3,6,5,4]*sL[12,13,24]*vL[8,10,24]*sL[14,15,22]*vL[19,21,22]*sR_imp[2,1,11]*vR_imp[8,9,11]*
                            sR[4,3,23]*vR[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3]+
                            A[13,12,16,17]*A[15,14,16,18]*A[1,7,5,2]*A_imp[3,6,5,4]*sL[12,13,24]*vL[8,10,24]*sL[14,15,22]*vL[19,21,22]*sR[2,1,11]*vR[8,9,11]*
                            sR_imp[4,3,23]*vR_imp[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3]
    #
    return A_single_new
end



function compute_fidelity_vertical_symmetric_modified(A,vL,vR,sL,sR,dis;printdetail=false)

    @tensor phiphi[:] := A[1,2,3,7]*A[1,2,3,8]*A[6,7,4,5]*A[6,8,4,5]
    @tensor psiphi[:] := A[2,1,5,12]*A[10,12,13,11]*A[6,4,5,8]*A[9,8,13,15]*vL[4,7,3]*vR[15,14,16]*
                            sL[1,2,3]*sR[11,10,16]*dis[6,9,7,14]
    @tensor psipsi[:] := A[4,3,6,13]*A[9,13,11,10]*A[2,1,6,14]*A[8,14,11,7]*sL[3,4,5]*sL[1,2,5]*sR[10,9,12]*sR[7,8,12]

    if printdetail ==true
        @printf "normal phiphi: %.16e   psipsi: %.16e   psiphi: %.16e: err: %.16e  \n" phiphi[1] psipsi[1] psiphi[1] (psipsi[1]+phiphi[1]-2*psiphi[1])/phiphi[1]
    end
    return (psipsi[1]+phiphi[1]-2*psiphi[1])/phiphi[1]
end



function update_vL(A,wL,wR,vL,vR,sL,sR,dis)
    @tensor Env_vL[:] := A[16,15,17,18]*A[2,18,6,3]*A[12,11,17,13]*A[9,13,6,7]*wL[15,16,14]*wR[3,2,1]*wL[11,10,-1]*
                        wR[7,8,4]*sL[14,-2]*sR[1,5]*vR[4,5]*dis[12,9,10,8]
    vL,S = update_tensor(Env_vL,1)
    return vL
end
function update_vL_impurity(A,Aimp,wL,wR,wLimp,wRimp,vL,vR,vLimp,vRimp,sL,sR,sLimp,sRimp,dis)
    @tensor Env_vL[:] := Aimp[16,15,17,18]*A[2,18,6,3]*Aimp[12,11,17,13]*A[9,13,6,7]*wLimp[15,16,14]*wR[3,2,1]*wLimp[11,10,-1]*
                        wR[7,8,4]*sLimp[14,-2]*sR[1,5]*vR[4,5]*dis[12,9,10,8]
    vLimp,S = update_tensor(Env_vL,1)
    return vLimp
end


function update_vR(A,wL,wR,vL,vR,sL,sR,dis)

    @tensor Env_vR[:] := A[3,2,7,17]*A[15,17,18,16]*A[9,6,7,13]*A[12,13,18,11]*wL[2,3,1]*wR[16,15,14]*wL[6,8,4]*
                        wR[11,10,-1]*sL[1,5]*sR[14,-2]*vL[4,5]*dis[9,12,8,10]
    vR,S = update_tensor(Env_vR,1)
    return vR
end
function update_vR_impurity(A,Aimp,wL,wR,wLimp,wRimp,vL,vR,vLimp,vRimp,sL,sR,sLimp,sRimp,dis)
    @tensor Env_vR[:] := A[3,2,7,17]*Aimp[15,17,18,16]*A[9,6,7,13]*Aimp[12,13,18,11]*wL[2,3,1]*wRimp[16,15,14]*wL[6,8,4]*
                        wRimp[11,10,-1]*sL[1,5]*sRimp[14,-2]*vL[4,5]*dis[9,12,8,10]
    vRimp,S = update_tensor(Env_vR,1)
    return vRimp
end




function update_dis(A,wL,wR,vL,vR,sL,sR,dis)

    @tensor Env_dis[:] := A[3,2,7,15]*A[9,15,13,10]*A[-1,6,7,16]*A[-2,16,13,14]*wL[2,3,1]*wR[10,9,8]*wL[6,-3,4]*
                        wR[14,-4,11]*sL[1,5]*sR[8,12]*vL[4,5]*vR[11,12]
    dis,S = update_tensor(Env_dis,2)
    return dis
end





function update_sL(A,wL,wR,vL,vR,sL,sR,dis)
    @tensor P[:] := A[16,15,17,18]*A[2,18,6,3]*A[12,13,17,14]*A[9,14,6,7]*wL[15,16,-1]*wR[3,2,1]*wL[13,11,10]*
                        wR[7,8,4]*sR[1,5]*vL[10,-2]*vR[4,5]*dis[12,9,11,8]
    @tensor B[:] := A[10,9,14,11]*A[2,11,7,3]*A[13,12,14,15]*A[6,15,7,5]*wL[9,10,-1]*wR[3,2,1]*wL[12,13,-2]*
                        wR[5,6,4]*sR[1,8]*sR[4,8]
    sL = idr_update(P,B,1)
    return sL
end
function update_sL_impurity(A,Aimp,wL,wR,wLimp,wRimp,vL,vR,vLimp,vRimp,sL,sR,sLimp,sRimp,dis)
    @tensor P[:] := Aimp[16,15,17,18]*A[2,18,6,3]*Aimp[12,13,17,14]*A[9,14,6,7]*wLimp[15,16,-1]*wR[3,2,1]*wLimp[13,11,10]*
                        wR[7,8,4]*sR[1,5]*vLimp[10,-2]*vR[4,5]*dis[12,9,11,8]
    @tensor B[:] := Aimp[10,9,14,11]*A[2,11,7,3]*Aimp[13,12,14,15]*A[6,15,7,5]*wLimp[9,10,-1]*wR[3,2,1]*wLimp[12,13,-2]*
                        wR[5,6,4]*sR[1,8]*sR[4,8]
    sLimp = idr_update(P,B,1)
    return sLimp
end
function update_sL_cg(A,wL,wR,vL,vR,sL,sR,dis)

    mu = 0.02
    d_km1 = zeros(size(sL)...)
    grad_km1 = zeros(size(sL)...)
    sL_temp = deepcopy(sL)

    @tensor grad_1[:] := A[16,15,17,18]*A[2,18,6,3]*A[12,13,17,14]*A[9,14,6,7]*wL[15,16,-1]*wL[13,11,10]*wR[3,2,1]*wR[7,8,4]*
                        sR[1,5]*vL[10,-2]*vR[4,5]*dis[12,9,11,8]
    @tensor Env_sLsL[:] := A[10,9,14,11]*A[2,11,7,3]*A[13,12,14,15]*A[6,15,7,5]*wL[9,10,-1]*wL[12,13,-2]*wR[3,2,1]*wR[5,6,4]*
                        sR[1,8]*sR[4,8]
    
    @tensor f0_1[:] := grad_1[1,2]*sL[1,2] 
    @tensor f0_2[:] := Env_sLsL[1,2]*sL[2,3]*sL[1,3]
    F0 = f0_2[1] - 2*f0_1[1]

    # F0 
    #F0 = compute_f(A,B,wL,wR,sL,sR,dis)
    # grad_1 = <psi|phi> without sL 
    # AsR = <psi|psi> without sL,sL^\dagger
    # grad_2 = <psi|psi> without sL

    for k in 1:200
        sL_k = deepcopy(sL_temp)
        # compute gradient
        @tensor grad_2[:] := Env_sLsL[-1,1]*sL_k[1,-2]
        grad_k = -2*grad_1+2*grad_2

        # compute conjugate gradient
        if k ==1 
            d_k = -1*grad_k
        else
            beta_k =  dot(grad_k,grad_k)/dot(grad_km1,grad_km1)
            d_k = -1*grad_k+beta_k*d_km1
        end

        # update for next iteration
        grad_km1 = grad_k
        d_km1 = d_k
        
        alpha0 = -1*dot(grad_k,d_k)/dot(d_k,d_k)
        alpha_k = alpha0
        # determine alpha // line search
        for j in 1:50 
            F_hat = F0+mu*alpha_k*dot(grad_k,d_k)        
            sL_temp = sL_k+alpha_k*d_k

            @tensor f_alpha_1[:] := grad_1[1,2]*sL_temp[1,2] 
            @tensor f_alpha_2[:] := Env_sLsL[1,2]*sL_temp[2,3]*sL_temp[1,3]
            F_alpha = f_alpha_2[1] - 2*f_alpha_1[1]

            if F_alpha >= F_hat
                alpha_k = alpha_k*0.8^j
            else
                #sL_temp = sL+alpha_k*d_k
                break
            end
        end

        sL_temp = sL_k+alpha_k*d_k
    end

    return sL_temp
end

function update_sL_cg_impurity(A,Aimp,wL,wR,wLimp,wRimp,vL,vR,vLimp,vRimp,sL,sR,sLimp,sRimp,dis)

    mu = 0.02
    d_km1 = zeros(size(sL)...)
    grad_km1 = zeros(size(sL)...)
    sL_temp = deepcopy(sL)

    @tensor grad_1[:] := A[16,15,17,18]*A[2,18,6,3]*A[12,13,17,14]*A[9,14,6,7]*wL[15,16,-1]*wL[13,11,10]*wR[3,2,1]*wR[7,8,4]*
                        sR[1,5]*vL[10,-2]*vR[4,5]*dis[12,9,11,8]
    @tensor Env_sLsL[:] := A[10,9,14,11]*A[2,11,7,3]*A[13,12,14,15]*A[6,15,7,5]*wL[9,10,-1]*wL[12,13,-2]*wR[3,2,1]*wR[5,6,4]*
                        sR[1,8]*sR[4,8]
    
    @tensor f0_1[:] := grad_1[1,2]*sL[1,2] 
    @tensor f0_2[:] := Env_sLsL[1,2]*sL[2,3]*sL[1,3]
    F0 = f0_2[1] - 2*f0_1[1]

    # F0 
    #F0 = compute_f(A,B,wL,wR,sL,sR,dis)
    # grad_1 = <psi|phi> without sL 
    # AsR = <psi|psi> without sL,sL^\dagger
    # grad_2 = <psi|psi> without sL

    for k in 1:200
        sL_k = deepcopy(sL_temp)
        # compute gradient
        @tensor grad_2[:] := Env_sLsL[-1,1]*sL_k[1,-2]
        grad_k = -2*grad_1+2*grad_2

        # compute conjugate gradient
        if k ==1 
            d_k = -1*grad_k
        else
            beta_k =  dot(grad_k,grad_k)/dot(grad_km1,grad_km1)
            d_k = -1*grad_k+beta_k*d_km1
        end

        # update for next iteration
        grad_km1 = grad_k
        d_km1 = d_k
        
        alpha0 = -1*dot(grad_k,d_k)/dot(d_k,d_k)
        alpha_k = alpha0
        # determine alpha // line search
        for j in 1:50 
            F_hat = F0+mu*alpha_k*dot(grad_k,d_k)        
            sL_temp = sL_k+alpha_k*d_k

            @tensor f_alpha_1[:] := grad_1[1,2]*sL_temp[1,2] 
            @tensor f_alpha_2[:] := Env_sLsL[1,2]*sL_temp[2,3]*sL_temp[1,3]
            F_alpha = f_alpha_2[1] - 2*f_alpha_1[1]

            if F_alpha >= F_hat
                alpha_k = alpha_k*0.8^j
            else
                #sL_temp = sL+alpha_k*d_k
                break
            end
        end

        sL_temp = sL_k+alpha_k*d_k
    end

    return sL_temp





end






function update_sR(A,wL,wR,vL,vR,sL,sR,dis)
    @tensor P[:] := A[3,2,7,17]*A[15,17,18,16]*A[8,6,7,14]*A[12,14,18,13]*wL[2,3,1]*wR[16,15,-1]*wL[6,9,4]*
                        wR[13,11,10]*sL[1,5]*vL[4,5]*vR[10,-2]*dis[8,12,9,11]
    @tensor B[:] := A[3,2,8,11]*A[9,11,15,10]*A[6,5,8,14]*A[12,14,15,13]*wL[2,3,1]*wR[10,9,-1]*wL[5,6,4]*
                        wR[13,12,-2]*sL[1,7]*sL[4,7]
    sR = idr_update(P,B,1)
    return sR
end
function update_sR_impurity(A,Aimp,wL,wR,wLimp,wRimp,vL,vR,vLimp,vRimp,sL,sR,sLimp,sRimp,dis)

    @tensor P[:] := A[3,2,7,17]*Aimp[15,17,18,16]*A[8,6,7,14]*Aimp[12,14,18,13]*wL[2,3,1]*wRimp[16,15,-1]*wL[6,9,4]*
                        wRimp[13,11,10]*sL[1,5]*vL[4,5]*vRimp[10,-2]*dis[8,12,9,11]
    @tensor B[:] := A[3,2,8,11]*Aimp[9,11,15,10]*A[6,5,8,14]*Aimp[12,14,15,13]*wL[2,3,1]*wRimp[10,9,-1]*wL[5,6,4]*
                        wRimp[13,12,-2]*sL[1,7]*sL[4,7]
    sRimp = idr_update(P,B,1)
    return sRimp
end
function update_sR_cg(A,wL,wR,vL,vR,sL,sR,dis)

    mu = 0.02
    d_km1 = zeros(size(sR)...)
    grad_km1 = zeros(size(sR)...)
    sR_temp = deepcopy(sR)

    @tensor grad_1[:] := A[3,2,7,17]*A[8,6,7,14]*A[15,17,18,16]*A[12,14,18,13]*wL[2,3,1]*wL[6,9,4]*wR[16,15,-1]*wR[13,11,10]*
                        sL[1,5]*vL[4,5]*vR[10,-2]*dis[8,12,9,11]
    @tensor Env_sRsR[:] := A[3,2,8,11]*A[6,5,8,14]*A[9,11,15,10]*A[12,14,15,13]*wL[2,3,1]*wL[5,6,4]*wR[10,9,-1]*wR[13,12,-2]*
                        sL[1,7]*sL[4,7]

    @tensor f0_1[:] := grad_1[1,2]*sR[1,2] 
    @tensor f0_2[:] := Env_sRsR[1,2]*sR[2,3]*sR[1,3]
    F0 = f0_2[1] - 2*f0_1[1]

    # F0 
    #F0 = compute_f(A,B,wL,wR,sL,sR,dis)
    # grad_1 = <psi|phi> without sL 
    # AsR = <psi|psi> without sL,sL^\dagger
    # grad_2 = <psi|psi> without sL

    for k in 1:200

        sR_k = deepcopy(sR_temp)

        # compute gradient
        @tensor grad_2[:] := Env_sRsR[-1,1]*sR_k[1,-2]
        grad_k = -2*grad_1+2*grad_2

        # compute conjugate gradient
        if k ==1 
            d_k = -1*grad_k
        else
            beta_k =  dot(grad_k,grad_k)/dot(grad_km1,grad_km1)
            d_k = -1*grad_k+beta_k*d_km1
        end

        # update for next iteration
        grad_km1 = grad_k
        d_km1 = d_k
        
        s_k = -1*dot(grad_k,d_k)/dot(d_k,d_k)
        alpha_k = s_k
        # determine alpha // line search
        for j in 1:50 
            F_hat = F0+mu*alpha_k*dot(grad_k,d_k)        
            sR_temp = sR_k+alpha_k*d_k

            @tensor f_alpha_1[:] := grad_1[1,2]*sR_temp[1,2] 
            @tensor f_alpha_2[:] := Env_sRsR[1,2]*sR_temp[2,3]*sR_temp[1,3]
            F_alpha = f_alpha_2[1] - 2*f_alpha_1[1]

            if F_alpha >= F_hat
                alpha_k = alpha_k*0.8^j
            else
                #sL_temp = sL+alpha_k*d_k
                break
            end
        end
        sR_temp = sR_k+alpha_k*d_k
    end

    return sR_temp
end

function update_sR_cg_impurity(A,Aimp,wL,wR,wLimp,wRimp,vL,vR,vLimp,vRimp,sL,sR,sLimp,sRimp,dis)

    mu = 0.02
    d_km1 = zeros(size(sR)...)
    grad_km1 = zeros(size(sR)...)
    sR_temp = deepcopy(sR)

    @tensor grad_1[:] := A[3,2,7,17]*A[8,6,7,14]*A[15,17,18,16]*A[12,14,18,13]*wL[2,3,1]*wL[6,9,4]*wR[16,15,-1]*wR[13,11,10]*
                        sL[1,5]*vL[4,5]*vR[10,-2]*dis[8,12,9,11]
    @tensor Env_sRsR[:] := A[3,2,8,11]*A[6,5,8,14]*A[9,11,15,10]*A[12,14,15,13]*wL[2,3,1]*wL[5,6,4]*wR[10,9,-1]*wR[13,12,-2]*
                        sL[1,7]*sL[4,7]

    @tensor f0_1[:] := grad_1[1,2]*sR[1,2] 
    @tensor f0_2[:] := Env_sRsR[1,2]*sR[2,3]*sR[1,3]
    F0 = f0_2[1] - 2*f0_1[1]

    # F0 
    #F0 = compute_f(A,B,wL,wR,sL,sR,dis)
    # grad_1 = <psi|phi> without sL 
    # AsR = <psi|psi> without sL,sL^\dagger
    # grad_2 = <psi|psi> without sL

    for k in 1:200

        sR_k = deepcopy(sR_temp)

        # compute gradient
        @tensor grad_2[:] := Env_sRsR[-1,1]*sR_k[1,-2]
        grad_k = -2*grad_1+2*grad_2

        # compute conjugate gradient
        if k ==1 
            d_k = -1*grad_k
        else
            beta_k =  dot(grad_k,grad_k)/dot(grad_km1,grad_km1)
            d_k = -1*grad_k+beta_k*d_km1
        end

        # update for next iteration
        grad_km1 = grad_k
        d_km1 = d_k
        
        s_k = -1*dot(grad_k,d_k)/dot(d_k,d_k)
        alpha_k = s_k
        # determine alpha // line search
        for j in 1:50 
            F_hat = F0+mu*alpha_k*dot(grad_k,d_k)        
            sR_temp = sR_k+alpha_k*d_k

            @tensor f_alpha_1[:] := grad_1[1,2]*sR_temp[1,2] 
            @tensor f_alpha_2[:] := Env_sRsR[1,2]*sR_temp[2,3]*sR_temp[1,3]
            F_alpha = f_alpha_2[1] - 2*f_alpha_1[1]

            if F_alpha >= F_hat
                alpha_k = alpha_k*0.8^j
            else
                #sL_temp = sL+alpha_k*d_k
                break
            end
        end
        sR_temp = sR_k+alpha_k*d_k
    end

    return sR_temp


end



function update_TEFT_clockwise(A,L)

    @tensor Temp[:] := A[-1,1,-3,-4]*L[-2,1]
    Temp = permutedims(Temp,[4,1,2,3])
    sizeTemp = size(Temp)
    Atemp = reshape(Temp,prod(sizeTemp[1:3]),sizeTemp[4])
    Rlt = qr(Atemp)
    L = Rlt.R

    return L 
end


function update_TEFT_anticlockwise(A,R)

    @tensor Temp[:] := A[-1,-2,-3,1]*R[-4,1]
    Temp = permutedims(Temp,[3,4,1,2])
    sizeTemp = size(Temp)
    Atemp = reshape(Temp,prod(sizeTemp[1]),prod(sizeTemp[2:4]))
    Rlt = lq(Atemp)
    R = Rlt.L

    return R
end




function TEFT(A,chimax)

    sizeA = size(A)
    L = Matrix(1.0I,sizeA[4],sizeA[4])
    R = Matrix(1.0I,sizeA[4],sizeA[4])

    for j in 1:100
        L = update_TEFT_clockwise(A,L)
        L = update_TEFT_clockwise(permutedims(A,[4,3,2,1]),L)
        L = update_TEFT_clockwise(permutedims(A,[1,4,3,2]),L)
        L = update_TEFT_clockwise(permutedims(A,[2,3,4,1]),L)
        L = L/maximum(abs.(L))
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
        R = update_TEFT_anticlockwise(A,R)
        R = update_TEFT_anticlockwise(permutedims(A,[2,1,4,3]),R)
        R = update_TEFT_anticlockwise(permutedims(A,[1,4,3,2]),R)
        R = update_TEFT_anticlockwise(permutedims(A,[4,1,2,3]),R)
        R = R/maximum(abs.(R))
    end
    Rlt = svd(L*R')
    #chitemp = sum(Rlt.S/maximum(Rlt.S) .> 1.0e-14)
    #chikept = min(chitemp,chimax)
    chikept = sizeA[4]
    PR = R'*Rlt.V[:,1:chikept]*diagm(Rlt.S[1:chikept])^(-0.5)   
    PL = L'*Rlt.U[:,1:chikept]*diagm(Rlt.S[1:chikept])^(-0.5)

    #PR = R'*Rlt.V*pinv(diagm(sqrt.(Rlt.S)),rtol=1.0e-12)   
    #PL = L'*Rlt.U*pinv(diagm(sqrt.(Rlt.S)),rtol=1.0e-12)
    
    return PL,PR,Rlt.S
end




function compute_gauge(A;printdetail=false)

    sizeA = size(A);
    x = Matrix(1.0I,sizeA[2],sizeA[2])
    #x = rand(sizeA[2],sizeA[2])
    @tensor psiphi[:] := A[4,1,5,2]*A[5,3,4,6]*x[3,1]*x[6,2]
    @tensor psipsi[:] := A[1,2,3,4]*A[1,2,3,4]
    @printf "psipsi: %.16e   psiphi: %.16e: err: %.16e  \n" psipsi[1] psiphi[1] (psipsi[1]+psipsi[1]-2*psiphi[1])/psipsi[1] 

    for j in 1:200
        #=
        @tensor Env_1[:] := A[2,1,4,-2]*A[4,3,2,-1]*x[3,1]
        @tensor Env_2[:] := A[2,-2,4,1]*A[4,-1,2,3]*x[3,1]
        =#
        @tensor Env_1[:] := A[2,-2,3,1]*A[3,-1,2,4]*x[4,1]
        #@tensor Env_2[:] := A[2,-2,4,1]*A[2,-1,4,3]*x[3,1]
        x,S = update_tensor(Env_1,1)

        if j %100 ==0 && printdetail == true
            #@tensor psiphi[:] := A[3,1,5,2]*A[5,4,3,6]*x[6,2]*x[4,1]
            @tensor psiphi[:] := A[4,1,5,2]*A[5,3,4,6]*x[3,1]*x[6,2]
            @tensor psipsi[:] := A[1,2,3,4]*A[1,2,3,4]
            @printf "psipsi: %.16e   psiphi: %.16e: err: %.16e  \n" psipsi[1] psiphi[1] (psipsi[1]+psipsi[1]-2*psiphi[1])/psipsi[1] 
        end
    end

    return x

end
