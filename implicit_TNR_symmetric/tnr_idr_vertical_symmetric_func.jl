  

function update_tensor(Env,index)

    sizeEnv = size(Env)
    Env = reshape(Env,prod(sizeEnv[1:index]),prod(sizeEnv[index+1:end]))
    Rlt = svd(Env)
    tensor = reshape(Rlt.U*Rlt.V',sizeEnv)
    return tensor,Rlt.S
end


function idr_update(P,B,index)

    sizeP = size(P);sizeB = size(B)
    P = reshape(P,prod(sizeP[1:index]),prod(sizeP[index+1:end]))
    B = reshape(B,prod(sizeB[1:index]),prod(sizeB[index+1:end]))
    B = (B+B')*0.5 
    P = P/maximum(B); B = B/maximum(B)
    tensor = reshape(pinv(B,rtol=1.0e-10)*P,sizeP)
    #tensor = reshape(pinv(B)*P,sizeP)
    return tensor
end


  





function update_sL_full(A,A_imp,vL,vR,vL_imp,vR_imp,sL,sR,sL_imp,sR_imp,dis)
    @tensor P_1[:] := A[-2,-1,12,13]*A[1,13,4,2]*A[10,9,12,11]*A[6,11,4,5]*vL[9,8,-3]*vR[5,7,3]*sR[2,1,3]*dis[10,6,8,7]
    @tensor B_1[:] := A[-2,-1,8,7]*A[1,7,5,2]*A[-4,-3,8,9]*A[4,9,5,3]*sR[2,1,6]*sR[3,4,6]
    
    @tensor P_2[:] := A[-2,-1,12,13]*A_imp[1,13,4,2]*A[10,9,12,11]*A_imp[6,11,4,5]*vL[9,8,-3]*vR_imp[5,7,3]*sR_imp[2,1,3]*dis[10,6,8,7]
    @tensor B_2[:] := A[-2,-1,8,7]*A_imp[1,7,5,2]*A[-4,-3,8,9]*A_imp[4,9,5,3]*sR_imp[2,1,6]*sR_imp[3,4,6]

    sL = idr_update(P_1+P_2,B_1+B_2,2) 
    return sL
end
function update_sL(A,vL,vR,sL,sR,dis)
    @tensor P[:] := A[-2,-1,12,13]*A[1,13,4,2]*A[10,9,12,11]*A[6,11,4,5]*vL[9,8,-3]*vR[5,7,3]*sR[2,1,3]*dis[10,6,8,7]
    @tensor B[:] := A[-2,-1,8,7]*A[1,7,5,2]*A[-4,-3,8,9]*A[4,9,5,3]*sR[2,1,6]*sR[3,4,6]
    sL = idr_update(P,B,2) 
    return sL
end
function update_sL_impurity(A,A_imp,vL,vR,vL_imp,sL,sR,sL_imp,dis)
    @tensor P_imp[:] := A_imp[-2,-1,12,13]*A[1,13,4,2]*A_imp[10,9,12,11]*A[6,11,4,5]*vL_imp[9,8,-3]*vR[5,7,3]*sR[2,1,3]*dis[10,6,8,7]
    @tensor B_imp[:] := A_imp[-2,-1,8,7]*A[1,7,5,2]*A_imp[-4,-3,8,9]*A[4,9,5,3]*sR[2,1,6]*sR[3,4,6]
    sL_imp = idr_update(P_imp,B_imp,2) 
    return sL_imp
end



function update_sL_cg()


    mu = 0.01
    alpha = 1.0

    d_km1 = zeros(size(sL)...)
    grad_km1 = zeros(size(sL)...)
    sL_temp = deepcopy(sL)

    @tensor grad_1[:] := A[-2,-1,12,13]*A[8,10,12,9]*B[1,13,4,2]*B[7,9,4,5]*sR[2,1,3]*wL[10,11,-3]*wR[5,6,3]*dis[8,7,11,6] 
    @tensor Env_sRsR[:] := A[-2,-1,8,7]*A[-4,-3,8,9]*B[1,7,5,2]*B[4,9,5,3]*sR[2,1,6]*sR[3,4,6]
    
    @tensor f0_1[:] := grad_1[1,2,3]*sL[1,2,3] 
    @tensor f0_2[:] := AsR[3,4,1,2]*sL[1,2,5]*sL[3,4,5]
    F0 = f0_2[1] - 2*f0_1[1]


    # F0 
    #F0 = compute_f(A,B,wL,wR,sL,sR,dis)
    # grad_1 = <psi|phi> without sL 
    # AsR = <psi|psi> without sL,sL^\dagger
    # grad_2 = <psi|psi> without sL

    for k in 1:200
        #print(k)
        # compute gradient
        grad_k = compute_sL_gradient(sL_temp,AsR,grad_1)
        
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
    
        # determine alpha
        for j in 1:50 
            F_hat = F0+mu*alpha*dot(grad_k,d_k)        
            sL_temp = sL+alpha*d_k
            F_alpha = compute_f_sL(sL_temp,AsR,grad_1)
            if F_alpha >= F_hat
                alpha = alpha*0.5^j
            else
                sL_temp = sL+alpha*d_k
                break
            end
        end

        sL_temp = sL+alpha*d_k
    end


end




function update_sR_full(A,A_imp,vL,vR,vL_imp,vR_imp,sL,sR,sL_imp,sR_imp,dis)
    @tensor P_1[:] := A[2,1,5,12]*A[-2,12,13,-1]*A[6,4,5,11]*A[10,11,13,9]*vL[4,7,3]*vR[9,8,-3]*sL[1,2,3]*dis[6,10,7,8]
    @tensor B_1[:] := A[2,1,6,7]*A[4,3,6,8]*A[-2,7,9,-1]*A[-4,8,9,-3]*sL[1,2,5]*sL[3,4,5]

    @tensor P_2[:] := A_imp[2,1,5,12]*A[-2,12,13,-1]*A_imp[6,4,5,11]*A[10,11,13,9]*vL_imp[4,7,3]*vR[9,8,-3]*sL_imp[1,2,3]*dis[6,10,7,8]
    @tensor B_2[:] := A_imp[2,1,6,7]*A_imp[4,3,6,8]*A[-2,7,9,-1]*A[-4,8,9,-3]*sL_imp[1,2,5]*sL_imp[3,4,5]

    sR = idr_update(P_1+P_2,B_1+B_2,2)
    return sR
end
function update_sR(A,vL,vR,sL,sR,dis)
    @tensor P[:] := A[2,1,5,12]*A[-2,12,13,-1]*A[6,4,5,11]*A[10,11,13,9]*vL[4,7,3]*vR[9,8,-3]*sL[1,2,3]*dis[6,10,7,8]
    @tensor B[:] := A[2,1,6,7]*A[4,3,6,8]*A[-2,7,9,-1]*A[-4,8,9,-3]*sL[1,2,5]*sL[3,4,5]
    sR = idr_update(P,B,2)
    return sR
end
function update_sR_impurity(A,A_imp,vL,vR,vR_imp,sL,sR,sR_imp,dis)
    @tensor P_imp[:] := A[2,1,5,12]*A_imp[-2,12,13,-1]*A[6,4,5,11]*A_imp[10,11,13,9]*vL[4,7,3]*vR_imp[9,8,-3]*sL[1,2,3]*dis[6,10,7,8]
    @tensor B_imp[:] := A[2,1,6,7]*A[4,3,6,8]*A_imp[-2,7,9,-1]*A_imp[-4,8,9,-3]*sL[1,2,5]*sL[3,4,5]
    sR_imp = idr_update(P_imp,B_imp,2)
    return sR_imp
end
function update_sR_cg()


end





function update_vL_full(A,A_imp,vL,vR,vL_imp,vR_imp,sL,sR,sL_imp,sR_imp,dis)
    @tensor Env_vL_1[:] := A[8,9,11,10]*A[1,10,4,2]*A[13,-1,11,12]*A[6,12,4,5]*vR[5,7,3]*sL[9,8,-3]*sR[2,1,3]*dis[13,6,-2,7]
    #@tensor Env_vL_2[:] := A_imp[8,9,11,10]*A[1,10,4,2]*A_imp[13,-1,11,12]*A[6,12,4,5]*vR[5,7,3]*sL_imp[9,8,-3]*sR[2,1,3]*dis[13,6,-2,7]
    @tensor Env_vL_2[:] := A[8,9,11,10]*A_imp[1,10,4,2]*A[13,-1,11,12]*A_imp[6,12,4,5]*vR_imp[5,7,3]*sL[9,8,-3]*sR_imp[2,1,3]*dis[13,6,-2,7]
    vL,S = update_tensor(Env_vL_1+Env_vL_2,2)
    return vL
end
function update_vL(A,vL,vR,sL,sR,dis)
    @tensor Env_vL[:] := A[8,9,11,10]*A[1,10,4,2]*A[13,-1,11,12]*A[6,12,4,5]*vR[5,7,3]*sL[9,8,-3]*sR[2,1,3]*dis[13,6,-2,7]
    vL,S = update_tensor(Env_vL,2)
    return vL
end
function update_vL_impurity(A,A_imp,vL,vR,vL_imp,sL,sR,sL_imp,dis;)
    @tensor Env_vL_imp[:] := A_imp[8,9,11,10]*A[1,10,4,2]*A_imp[13,-1,11,12]*A[6,12,4,5]*
                    vR[5,7,3]*sL_imp[9,8,-3]*sR[2,1,3]*dis[13,6,-2,7]
    vL_imp,S = update_tensor(Env_vL_imp,2)
    return vL_imp
end



function update_vR_full(A,A_imp,vL,vR,vL_imp,vR_imp,sL,sR,sL_imp,sR_imp,dis)
    @tensor Env_vR_1[:] := A[2,1,5,10]*A[8,10,12,9]*A[7,4,5,11]*A[13,11,12,-1]*vL[4,6,3]*sL[1,2,3]*sR[9,8,-3]*dis[7,13,6,-2]
    @tensor Env_vR_2[:] := A_imp[2,1,5,10]*A[8,10,12,9]*A_imp[7,4,5,11]*A[13,11,12,-1]*vL_imp[4,6,3]*sL_imp[1,2,3]*sR[9,8,-3]*dis[7,13,6,-2]
    #@tensor Env_vR_3[:] := A[2,1,5,10]*A_imp[8,10,12,9]*A[7,4,5,11]*A_imp[13,11,12,-1]*vL[4,6,3]*sL[1,2,3]*sR_imp[9,8,-3]*dis[7,13,6,-2]
    vR,S = update_tensor(Env_vR_1+Env_vR_2,2)
    return vR
end
function update_vR(A,vL,vR,sL,sR,dis)
    @tensor Env_vR[:] := A[2,1,5,10]*A[8,10,12,9]*A[7,4,5,11]*A[13,11,12,-1]*vL[4,6,3]*
                    sL[1,2,3]*sR[9,8,-3]*dis[7,13,6,-2]
    vR,S = update_tensor(Env_vR,2)
    return vR
end
function update_vR_impurity(A,A_imp,vL,vR,vR_imp,sL,sR,sR_imp,dis)
    
    @tensor Env_vR_imp[:] := A[2,1,5,10]*A_imp[8,10,12,9]*A[7,4,5,11]*A_imp[13,11,12,-1]*vL[4,6,3]*
                    sL[1,2,3]*sR_imp[9,8,-3]*dis[7,13,6,-2]
    vR_imp,S = update_tensor(Env_vR_imp,2)
    return vR_imp
end



function update_dis(A,vL,vR,sL,sR,dis)
    @tensor Env_dis[:] := A[1,2,5,11]*A[6,11,9,7]*A[-1,4,5,12]*A[-2,12,9,10]*vL[4,-3,3]*vR[10,-4,8]*sL[2,1,3]*sR[7,6,8]
    dis,S = update_tensor(Env_dis,2)
    return dis
end
function update_dis_full(A,A_imp,vL,vR,vL_imp,vR_imp,sL,sR,sL_imp,sR_imp,dis)
    @tensor Env_dis_1[:] := A[1,2,5,11]*A[6,11,9,7]*A[-1,4,5,12]*A[-2,12,9,10]*vL[4,-3,3]*vR[10,-4,8]*sL[2,1,3]*sR[7,6,8]
    @tensor Env_dis_2[:] := A_imp[1,2,5,11]*A[6,11,9,7]*A_imp[-1,4,5,12]*A[-2,12,9,10]*vL_imp[4,-3,3]*vR[10,-4,8]*sL_imp[2,1,3]*sR[7,6,8]
    @tensor Env_dis_3[:] := A[1,2,5,11]*A_imp[6,11,9,7]*A[-1,4,5,12]*A_imp[-2,12,9,10]*vL[4,-3,3]*vR_imp[10,-4,8]*sL[2,1,3]*sR_imp[7,6,8]

    dis,S = update_tensor(Env_dis_1+Env_dis_2+Env_dis_3,2)
    return dis
end

function compute_isometry_vertical_symmetric(A,chimax;Aimp=0)

    
    if Aimp ==0
        @tensor Env_left[:] := A[-2,-1,13,14]*A[8,14,9,7]*A[2,1,13,5]*A[3,5,9,4]*A[2,1,11,6]*A[3,6,10,4]*A[-4,-3,11,12]*A[8,12,10,7]
        @tensor Env_right[:] := A[7,8,9,11]*A[-2,11,12,-1]*A[2,1,9,5]*A[3,5,12,4]*A[2,1,10,6]*A[3,6,14,4]*A[7,8,10,13]*A[-4,13,14,-3]
    else
        @tensor Env_left[:] := Aimp[-2,-1,13,14]*A[8,14,9,7]*A[2,1,13,5]*A[3,5,9,4]*A[2,1,11,6]*A[3,6,10,4]*Aimp[-4,-3,11,12]*A[8,12,10,7]
        @tensor Env_right[:] := A[7,8,9,11]*Aimp[-2,11,12,-1]*A[2,1,9,5]*A[3,5,12,4]*A[2,1,10,6]*A[3,6,14,4]*A[7,8,10,13]*Aimp[-4,13,14,-3]
    end
    
    sizeA = size(A)
    Env_left = reshape(Env_left,prod(sizeA[1:2]),prod(sizeA[1:2]))
    Rlt = svd(Env_left)
    chitemp = sum(Rlt.S/maximum(Rlt.S) .> 1.0e-14)
    println("chitemp: ",chitemp)
    chikept = min(chitemp,chimax,prod(sizeA[1:2]))
    vL = reshape(Rlt.U[:,1:chikept],sizeA[2],sizeA[1],chikept)
    

    Env_right = reshape(Env_right,prod(sizeA[3:4]),prod(sizeA[3:4]))
    Rlt = svd(Env_right)
    chitemp = sum(Rlt.S/maximum(Rlt.S) .> 1.0e-14)
    println("chitemp: ",chitemp)
    chikept = min(chitemp,chimax,prod(sizeA[3:4]))
    vR = reshape(Rlt.U[:,1:chikept],sizeA[4],sizeA[3],chikept)

    @tensor phiphi[:] := Env_left[1,1]
    @tensor psiphi[:] := A[8,7,9,12]*A[10,12,13,11]*A[2,1,9,5]*A[3,5,13,4]*A[2,1,17,6]*A[3,6,21,4]*
                        A[15,14,17,20]*A[18,20,21,19]*vL[7,8,16]*vL[14,15,16]*vR[11,10,22]*vR[19,18,22]
    println("Truncation Error: ",(psiphi[1]-phiphi[1])/phiphi[1] )

    return vL,vR
end



function compute_fidelity_vertical_symmetric(A,vL,vR,sL,sR,dis;printdetail=false)

    @tensor phiphi[:] := A[1,2,3,7]*A[1,2,3,8]*A[6,7,4,5]*A[6,8,4,5]
    @tensor psiphi[:] := A[2,1,5,12]*A[10,12,13,11]*A[6,4,5,8]*A[9,8,13,15]*vL[4,7,3]*vR[15,14,16]*
                            sL[1,2,3]*sR[11,10,16]*dis[6,9,7,14]
    @tensor psipsi[:] := A[4,3,6,13]*A[9,13,11,10]*A[2,1,6,14]*A[8,14,11,7]*sL[3,4,5]*sL[1,2,5]*sR[10,9,12]*sR[7,8,12]

    if printdetail ==true
        @printf "normal phiphi: %.16e   psipsi: %.16e   psiphi: %.16e: err: %.16e  \n" phiphi[1] psipsi[1] psiphi[1] (psipsi[1]+phiphi[1]-2*psiphi[1])/phiphi[1]
    end
    return (psipsi[1]+phiphi[1]-2*psiphi[1])/phiphi[1]
end

function compute_fidelity_vertical_symmetric_large(A)



end


function compute_fidelity_left_impurity_vertical_symmetric(A,A_imp,vL,vR,vL_imp,sL,sR,sL_imp,dis;printdetail=false)

    @tensor phiphi[:] := A_imp[1,2,3,7]*A_imp[1,2,3,8]*A[6,7,4,5]*A[6,8,4,5]
    @tensor psiphi[:] := A_imp[2,1,5,12]*A[10,12,13,11]*A_imp[6,4,5,8]*A[9,8,13,15]*vL_imp[4,7,3]*vR[15,14,16]*
                            sL_imp[1,2,3]*sR[11,10,16]*dis[6,9,7,14]
    @tensor psipsi[:] := A_imp[4,3,6,13]*A[9,13,11,10]*A_imp[2,1,6,14]*A[8,14,11,7]*sL_imp[3,4,5]*sL_imp[1,2,5]*sR[10,9,12]*sR[7,8,12]

    if printdetail ==true
        @printf "left   phiphi: %.16e   psipsi: %.16e   psiphi: %.16e: err: %.16e  \n" phiphi[1] psipsi[1] psiphi[1] 1-psiphi[1]^2/psipsi[1]/phiphi[1] 
    end
    return 1-psiphi[1]^2/psipsi[1]/phiphi[1] 
end



function compute_fidelity_right_impurity_vertical_symmetric(A,A_imp,vL,vR,vR_imp,sL,sR,sR_imp,dis;printdetail=false)

    @tensor phiphi[:] := A[1,2,3,7]*A[1,2,3,8]*A_imp[6,7,4,5]*A_imp[6,8,4,5]
    @tensor psiphi[:] := A[2,1,5,12]*A_imp[10,12,13,11]*A[6,4,5,8]*A_imp[9,8,13,15]*vL[4,7,3]*vR_imp[15,14,16]*
                            sL[1,2,3]*sR_imp[11,10,16]*dis[6,9,7,14]
    @tensor psipsi[:] := A[4,3,6,13]*A_imp[9,13,11,10]*A[2,1,6,14]*A_imp[8,14,11,7]*sL[3,4,5]*sL[1,2,5]*sR_imp[10,9,12]*sR_imp[7,8,12]

    if printdetail ==true
        @printf "right  phiphi: %.16e   psipsi: %.16e   psiphi: %.16e: err: %.16e  \n" phiphi[1] psipsi[1] psiphi[1] 1-psiphi[1]^2/psipsi[1]/phiphi[1] 
    end
    return 1-psiphi[1]^2/psipsi[1]/phiphi[1] 
end




function compute_projector_tnr_vertical_symmetric(A,vL,vR,sL,sR,dis,chimax)

    @tensor sLAsLA[:] := sL[1,2,-1]*A[2,1,5,-3]*sL[3,4,-2]*A[4,3,5,-4]
    @tensor sRAsRA[:] := sR[2,1,-1]*A[1,-3,5,2]*sR[3,4,-2]*A[4,-4,5,3]
    @tensor v4[:] := vL[1,4,-2]*vR[1,3,-1]*vL[2,4,-4]*vR[2,3,-3]
    
    @tensor Envy[:] := sRAsRA[7,8,-2,-1]*sLAsLA[3,5,1,2]*sLAsLA[4,6,1,2]*sRAsRA[9,10,-4,-3]*v4[7,3,9,4]*v4[8,5,10,6]
    sizeEnvy = size(Envy)
    Envy = reshape(Envy,prod(sizeEnvy[1:2]),prod(sizeEnvy[1:2]))
    Rlt = svd(Envy)

    chitemp = sum(Rlt.S/maximum(Rlt.S) .> 1.0e-14)
    chikept = min(chitemp,chimax,prod(sizeEnvy[1:2]))
    #chikept = min(chimax,prod(sizeEnvy[1:2]))
    println("truncation error: ",(tr(Envy)-tr(Envy*Rlt.U[:,1:chikept]*Rlt.U'[1:chikept,:]))/tr(Envy))
    y = reshape(Rlt.U[:,1:chikept],sizeEnvy[1],sizeEnvy[2],chikept)



    @tensor Envw[:] := sLAsLA[1,2,3,4]*sRAsRA[25,12,3,4]*sLAsLA[26,19,17,18]*sRAsRA[14,13,17,18]*
                    sLAsLA[1,2,6,5]*sRAsRA[22,11,6,5]*sLAsLA[23,20,16,15]*sRAsRA[14,13,16,15]*
                    vR[24,-1,25]*vL[24,-2,26]*vR[9,7,12]*vL[9,8,19]*vR[10,7,11]*vL[10,8,20]*vR[21,-3,22]*vL[21,-4,23]
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




function renormalize_fragment_tnr_two_impurity_vertical_symmetric(A,A_imp1,A_imp2,vL,vR,vL_imp1,vR_imp1,vL_imp2,vR_imp2,
                            sL,sR,sL_imp1,sR_imp1,sL_imp2,sR_imp2,y,w)

    @tensor A_new[:] := A_imp1[13,12,16,17]*A_imp2[15,14,16,18]*A[1,7,5,2]*A[3,6,5,4]*sL_imp1[12,13,24]*vL_imp1[8,10,24]*sL_imp2[14,15,22]*vL_imp2[19,21,22]*sR[2,1,11]*vR[8,9,11]*
    sR[4,3,23]*vR[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3]+
    A_imp1[13,12,16,17]*A[15,14,16,18]*A_imp2[1,7,5,2]*A[3,6,5,4]*sL_imp1[12,13,24]*vL_imp1[8,10,24]*sL[14,15,22]*vL[19,21,22]*sR_imp2[2,1,11]*vR_imp2[8,9,11]*
    sR[4,3,23]*vR[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3]+
    A_imp1[13,12,16,17]*A[15,14,16,18]*A[1,7,5,2]*A_imp2[3,6,5,4]*sL_imp1[12,13,24]*vL_imp1[8,10,24]*sL[14,15,22]*vL[19,21,22]*sR[2,1,11]*vR[8,9,11]*
    sR_imp2[4,3,23]*vR_imp2[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3]+
    A[13,12,16,17]*A_imp1[15,14,16,18]*A_imp2[1,7,5,2]*A[3,6,5,4]*sL[12,13,24]*vL[8,10,24]*sL_imp1[14,15,22]*vL_imp1[19,21,22]*sR_imp2[2,1,11]*vR_imp2[8,9,11]*
    sR[4,3,23]*vR[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3]+
    A[13,12,16,17]*A_imp1[15,14,16,18]*A[1,7,5,2]*A_imp2[3,6,5,4]*sL[12,13,24]*vL[8,10,24]*sL_imp1[14,15,22]*vL_imp1[19,21,22]*sR[2,1,11]*vR[8,9,11]*
    sR_imp2[4,3,23]*vR_imp2[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3]+
    A[13,12,16,17]*A[15,14,16,18]*A_imp1[1,7,5,2]*A_imp2[3,6,5,4]*sL[12,13,24]*vL[8,10,24]*sL[14,15,22]*vL[19,21,22]*sR_imp1[2,1,11]*vR_imp1[8,9,11]*
    sR_imp2[4,3,23]*vR_imp2[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3]

    return A_new
end




function renormalize_fragment_tnr_three_impurity_vertical_symmetric(A,A_imp1,A_imp2,A_imp3,vL,vR,vL_imp1,vR_imp1,vL_imp2,vR_imp2,vL_imp3,vR_imp3,
    sL,sR,sL_imp1,sR_imp1,sL_imp2,sR_imp2,sL_imp3,sR_imp3,y,w)

    @tensor A_new[:] := A_imp1[13,12,16,17]*A_imp2[15,14,16,18]*A_imp3[1,7,5,2]*A[3,6,5,4]*sL_imp1[12,13,24]*vL_imp1[8,10,24]*sL_imp2[14,15,22]*vL_imp2[19,21,22]*sR_imp3[2,1,11]*vR_imp3[8,9,11]*
    sR[4,3,23]*vR[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3]+
    A_imp1[13,12,16,17]*A_imp2[15,14,16,18]*A[1,7,5,2]*A_imp3[3,6,5,4]*sL_imp1[12,13,24]*vL_imp1[8,10,24]*sL_imp2[14,15,22]*vL_imp2[19,21,22]*sR[2,1,11]*vR[8,9,11]*
    sR_imp3[4,3,23]*vR_imp3[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3]+
    A_imp1[13,12,16,17]*A[15,14,16,18]*A_imp2[1,7,5,2]*A_imp3[3,6,5,4]*sL_imp1[12,13,24]*vL_imp1[8,10,24]*sL[14,15,22]*vL[19,21,22]*sR_imp2[2,1,11]*vR_imp2[8,9,11]*
    sR_imp3[4,3,23]*vR_imp3[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3]+
    A[13,12,16,17]*A_imp1[15,14,16,18]*A_imp2[1,7,5,2]*A_imp3[3,6,5,4]*sL[12,13,24]*vL[8,10,24]*sL_imp1[14,15,22]*vL_imp1[19,21,22]*sR_imp2[2,1,11]*vR_imp2[8,9,11]*
    sR_imp3[4,3,23]*vR_imp3[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3]


    return A_new
end




function renormalize_fragment_tnr_four_impurity_vertical_symmetric(A,A_imp1,A_imp2,A_imp3,A_imp4,vL,vR,vL_imp1,vR_imp1,vL_imp2,vR_imp2,vL_imp3,vR_imp3,vL_imp4,vR_imp4,
    sL,sR,sL_imp1,sR_imp1,sL_imp2,sR_imp2,sL_imp3,sR_imp3,sL_imp4,sR_imp4,y,w)

    @tensor A_new[:] := A_imp1[13,12,16,17]*A_imp2[15,14,16,18]*A_imp3[1,7,5,2]*A_imp4[3,6,5,4]*sL_imp1[12,13,24]*vL_imp1[8,10,24]*sL_imp2[14,15,22]*vL_imp2[19,21,22]*sR_imp3[2,1,11]*vR_imp3[8,9,11]*
    sR_imp4[4,3,23]*vR_imp4[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3]
    return A_new
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



function update_vL_vertical_symmetric_large(A,vL0,vR0,A4vLvR,A4,vL,vR,sL,sR,dis)

    @tensor Env_vL[:] := A[2,1,6,5]*A[4,5,7,3]*A4vLvR[6,7,14,10]*vL0[-1,13,14]*vR0[8,12,10]*sL[1,2,-3]*sR[3,4,9]*vR[8,11,9]*dis[13,12,-2,11]
    vL,S = update_tensor(Env_vL,2)
    return vL
end

function update_vL_vertical_symmetric_impurity_large(A,Aimp,vL0,vR0,vL0imp,vR0imp,A4vLvR,A4vLimpvR,A4vLvRimp,A4,vL,vR,vLimp,vRimp,sL,sR,sLimp,sRimp,dis)

    @tensor Env_vL[:] := Aimp[2,1,6,5]*A[4,5,7,3]*A4vLimpvR[6,7,14,10]*vL0imp[-1,13,14]*vR0[8,12,10]*sLimp[1,2,-3]*sR[3,4,9]*vR[8,11,9]*dis[13,12,-2,11]
    vLimp,S = update_tensor(Env_vL,2)
    return vLimp
end


function update_vR_vertical_symmetric_large(A,vL0,vR0,A4vLvR,A4,vL,vR,sL,sR,dis)
    @tensor Env_vR[:] := A[2,1,6,5]*A[4,5,7,3]*A4vLvR[6,7,10,13]*vL0[8,11,10]*vR0[-1,14,13]*sL[1,2,9]*sR[3,4,-3]*vL[8,12,9]*dis[11,14,12,-2]
    vR,S = update_tensor(Env_vR,2)
    return vR
end
function update_vR_vertical_symmetric_impurity_large(A,Aimp,vL0,vR0,vL0imp,vR0imp,A4vLvR,A4vLimpvR,A4vLvRimp,A4,vL,vR,vLimp,vRimp,sL,sR,sLimp,sRimp,dis)
    @tensor Env_vR[:] := A[2,1,6,5]*Aimp[4,5,7,3]*A4vLvRimp[6,7,10,13]*vL0[8,11,10]*vR0imp[-1,14,13]*sL[1,2,9]*sRimp[3,4,-3]*vL[8,12,9]*dis[11,14,12,-2]
    vRimp,S = update_tensor(Env_vR,2)
    return vRimp
end



function update_sL_vertical_symmetric_large(A,vL0,vR0,A4vLvR,A4,vL,vR,sL,sR,dis)

    @tensor P[:] := A[-2,-1,13,14]*A[7,14,12,8]*A4vLvR[13,12,10,11]*vL0[1,2,10]*vR0[4,6,11]*sR[8,7,9]*vL[1,3,-3]*vR[4,5,9]*dis[2,6,3,5]
    @tensor B[:] := A[-2,-1,10,11]*A[1,11,6,2]*A4[10,6,8,7]*A[-4,-3,8,9]*A[4,9,7,3]*sR[2,1,5]*sR[3,4,5]

    sL = idr_update(P,B,2)
    return sL
end
function update_sL_vertical_symmetric_impurity_large(A,Aimp,vL0,vR0,vL0imp,vR0imp,A4vLvR,A4vLimpvR,A4vLvRimp,A4,vL,vR,vLimp,vRimp,sL,sR,sLimp,sRimp,dis)

    @tensor P[:] := Aimp[-2,-1,13,14]*A[7,14,12,8]*A4vLimpvR[13,12,10,11]*vL0imp[1,2,10]*vR0[4,6,11]*sR[8,7,9]*vLimp[1,3,-3]*vR[4,5,9]*dis[2,6,3,5]
    @tensor B[:] := Aimp[-2,-1,10,11]*A[1,11,6,2]*A4[10,6,8,7]*Aimp[-4,-3,8,9]*A[4,9,7,3]*sR[2,1,5]*sR[3,4,5]
    sL = idr_update(P,B,2)
    return sL

end



function update_sR_vertical_symmetric_large(A,vL0,vR0,A4vLvR,A4,vL,vR,sL,sR,dis)
    @tensor P[:] := A[8,7,12,13]*A[-2,13,14,-1]*A4vLvR[12,14,10,11]*vL0[4,5,10]*vR0[1,3,11]*sL[7,8,9]*vL[4,6,9]*vR[1,2,-3]*dis[5,3,6,2]
    @tensor B[:] := A[2,1,6,8]*A[-2,8,9,-1]*A4[6,9,7,11]*A[4,3,7,10]*A[-4,10,11,-3]*sL[1,2,5]*sL[3,4,5]
    sR = idr_update(P,B,2)
    return sR
end
function update_sR_vertical_symmetric_impurity_large(A,Aimp,vL0,vR0,vL0imp,vR0imp,A4vLvR,A4vLimpvR,A4vLvRimp,A4,vL,vR,vLimp,vRimp,sL,sR,sLimp,sRimp,dis)
    @tensor P[:] := A[8,7,12,13]*Aimp[-2,13,14,-1]*A4vLvRimp[12,14,10,11]*vL0[4,5,10]*vR0imp[1,3,11]*sL[7,8,9]*vL[4,6,9]*vRimp[1,2,-3]*dis[5,3,6,2]
    @tensor B[:] := A[2,1,6,8]*Aimp[-2,8,9,-1]*A4[6,9,7,11]*A[4,3,7,10]*Aimp[-4,10,11,-3]*sL[1,2,5]*sL[3,4,5]
    sR = idr_update(P,B,2)
    return sR
end



function update_dis_vertical_symmetric_large(A,vL0,vR0,A4vLvR,A4,vL,vR,sL,sR,dis)

    @tensor Env_dis[:] := A[2,1,6,5]*A[4,5,7,3]*A4vLvR[6,7,10,12]*vL0[8,-1,10]*vR0[11,-2,12]*sL[1,2,9]*sR[3,4,13]*vL[8,-3,9]*vR[11,-4,13]
    dis,S = update_tensor(Env_dis,2)
    return dis
end




function compute_projector_tnr_vertical_symmetric_alterate_rg(A,vL,vR,sL,sR,dis,chimax)



    @tensor L[:] := vL[-4,1,-2]*vL[-3,1,-1]
    @tensor R[:] := vR[-4,1,-2]*vR[-3,1,-1]
    @tensor sA[:] := A[2,1,-1,5]*A[4,5,-2,3]*sL[1,2,-3]*sR[3,4,-4]
    
    @tensor Envy[:] := L[27,28,-2,-1]*sA[20,19,27,21]*R[21,25,15,16]*sA[23,24,28,25]*
                        L[11,12,15,16]*sA[3,4,11,5]*R[5,9,1,2]*sA[8,7,12,9]*L[29,30,-4,-3]*
                        sA[20,19,29,22]*R[22,26,17,18]*sA[23,24,30,26]*L[13,14,17,18]*sA[3,4,13,6]*
                        R[6,10,1,2]*sA[8,7,14,10]
    #
    sizeEnvy = size(Envy)
    Envy = reshape(Envy,prod(sizeEnvy[1:2]),prod(sizeEnvy[1:2]))
    Rlt = svd(Envy)

    chitemp = sum(Rlt.S/maximum(Rlt.S) .> 1.0e-14)
    chikept = min(chitemp,chimax,prod(sizeEnvy[1:2]))
    println("truncation error: ",(tr(Envy)-tr(Envy*Rlt.U[:,1:chikept]*Rlt.U'[1:chikept,:]))/tr(Envy))
    y = reshape(Rlt.U[:,1:chikept],sizeEnvy[1],sizeEnvy[2],chikept)


    @tensor Envw[:] := L[19,9,3,4]*R[20,17,13,14]*sA[-1,-2,19,20]*sA[7,8,9,17]*R[1,2,3,4]*L[12,11,13,14]*
                        L[21,10,6,5]*R[22,18,16,15]*sA[7,8,10,18]*sA[-3,-4,21,22]*R[1,2,6,5]*L[12,11,16,15]
    #

    sizeEnvw = size(Envw)
    Envw = reshape(Envw,prod(sizeEnvw[1:2]),prod(sizeEnvw[1:2]))
    Rlt = svd(Envw)
    chitemp = sum(Rlt.S/maximum(Rlt.S) .> 1.0e-14)
    chikept = min(chitemp,chimax,prod(sizeEnvw[1:2]))
    println("truncation error: ",(tr(Envw)-tr(Envw*Rlt.U[:,1:chikept]*Rlt.U'[1:chikept,:]))/tr(Envw))
    w = reshape(Rlt.U[:,1:chikept],sizeEnvw[1],sizeEnvw[2],chikept)

    return y,w

end
