include("./update_sL.jl")
include("./update_sR.jl")
include("./update_wL.jl")
include("./update_wR.jl")
include("./update_dis.jl")


eye(n) = Matrix(1.0I,n,n)



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
    B = (B+B')*0.5 #+ 1.0e-13*Matrix(1.0I,size(B,1),size(B,1))*norm(B)
    tensor = reshape(pinv(B,rtol=5.0e-13)*P,sizeP)
    #tensor = reshape(pinv(B)*P,sizeP)
    return tensor
end



function update_sL(A,B,wL,wR,sL,sR,dis)

    @tensor P[:] := A[-2,-1,12,13]*A[8,10,12,9]*B[1,13,4,2]*B[7,9,4,5]*sR[2,1,3]*wL[10,11,-3]*wR[5,6,3]*dis[8,7,11,6] 
    @tensor B[:] := A[-2,-1,8,7]*A[-4,-3,8,9]*B[1,7,5,2]*B[3,9,5,4]*sR[2,1,6]*sR[4,3,6] 
    sL = idr_update(P,B,2)
    return sL
end

function update_sLM(A,B,wL,wR,sL,sR,qL,qR,dis)

    @tensor P[:] := A[10,9,12,11]*A[14,15,12,13]*B[1,11,5,2]*B[8,13,5,6]*sR[3,4]*qL[9,10,-1]*qR[2,1,3]*wL[15,16,-2]*wR[6,7,4]*dis[14,8,16,7]
    @tensor B[:] := A[13,12,14,15]*A[10,9,14,11]*B[1,15,7,2]*B[5,11,7,4]*qL[12,13,-1]*qL[9,10,-2]*qR[2,1,3]*qR[4,5,6]*sR[3,8]*sR[6,8] 

    sL = idr_update(P,B,1)
    return sL
end


function update_sRM(A,B,wL,wR,sL,sR,qL,qR,dis)

    @tensor P[:] := A[2,1,6,11]*A[7,5,6,12]*B[9,11,13,10]*B[14,12,13,16]*sL[3,4]*qL[1,2,3]*qR[10,9,-1]*wL[5,8,4]*wR[16,15,-2]*dis[7,14,8,15]
    @tensor B[:] := A[2,1,8,11]*A[5,4,8,14]*B[9,11,15,10]*B[12,14,15,13]*qL[1,2,3]*qL[4,5,6]*qR[10,9,-1]*qR[13,12,-2]*sL[3,7]*sL[6,7]
    sR = idr_update(P,B,1)
    return sR
end



function update_sR(A,B,wL,wR,sL,sR,dis)

    @tensor P[:] := A[2,1,5,12]*A[6,4,5,8]*B[-2,12,13,-1]*B[9,8,13,11]*sL[1,2,3]*wL[4,7,3]*wR[11,10,-3]*dis[6,9,7,10] 
    @tensor B[:] := A[2,1,6,7]*A[4,3,6,8]*B[-2,7,9,-1]*B[-4,8,9,-3]*sL[1,2,5]*sL[3,4,5]
    sR = idr_update(P,B,2)
    return sR
end


function update_wL(A,B,wL,wR,sL,sR,dis)

    @tensor Env_wL[:] := A[12,11,13,14]*A[8,-1,13,9]*B[1,14,4,2]*B[7,9,4,5]*sL[11,12,-3]*sR[2,1,3]*wR[5,6,3]*dis[8,7,-2,6] 
    wL,S = update_tensor(Env_wL,2)

    return wL,S 
end


function update_wR(A,B,wL,wR,sL,sR,dis)


    @tensor Env_wR[:] := A[1,2,5,12]*A[7,4,5,8]*B[10,12,13,11]*B[9,8,13,-1]*sL[2,1,3]*sR[11,10,-3]*wL[4,6,3]*dis[7,9,6,-2] 
    wR,S = update_tensor(Env_wR,2)
    return wR,S
end


function update_dis(A,B,wL,wR,sL,sR,dis)

    @tensor Env_dis[:] := A[1,2,5,11]*A[-1,4,5,12]*B[6,11,9,7]*B[-2,12,9,10]*sL[2,1,3]*sR[7,6,8]*wL[4,-3,3]*wR[10,-4,8] 
    dis,S = update_tensor(Env_dis,2)

    return dis,S
end

function update_wLM(A,B,wL,wR,sL,sR,qL,qR,dis)

    @tensor Env_wL[:] := A[10,9,13,12]*A[15,-1,13,14]*B[1,12,5,2]*B[7,14,5,6]*qL[9,10,11]*qR[2,1,3]*wR[6,8,4]*sL[11,-3]*sR[3,4]*dis[15,7,-2,8]
    wL,S = update_tensor(Env_wL,2)
    return wL
end



function update_wRM(A,B,wL,wR,sL,sR,qL,qR,dis)

    @tensor Env_wR[:] := A[2,1,6,12]*A[7,5,6,13]*B[9,12,14,10]*B[15,13,14,-1]*qL[1,2,3]*qR[10,9,11]*wL[5,8,4]*sL[3,4]*sR[11,-3]*dis[7,15,8,-2]
    wR,S = update_tensor(Env_wR,2)
    return wR
end

function update_disM(A,B,wL,wR,sL,sR,qL,qR,dis)

    @tensor Env_dis[:] := A[10,9,14,15]*A[-1,13,14,16]*B[1,15,5,2]*B[-2,16,5,6]*qL[9,10,11]*qR[2,1,3]*wL[13,-3,12]*wR[6,-4,4]*sL[11,12]*sR[3,4]
    dis,S = update_tensor(Env_dis,2)
    return dis
end


function compute_fidelity_M(A,B,wL,wR,sL,sR,qL,qR,dis;printdetail=false)


    @tensor phiphi[:] := A[1,2,3,7]*B[6,7,4,5]*A[1,2,3,8]*B[6,8,4,5]
    @tensor psiphi[:] := A[10,9,14,17]*A[15,13,14,18]*B[1,17,5,2]*B[8,18,5,6]*wL[13,16,12]*wR[6,7,4]*sL[11,12]*sR[3,4]*qL[9,10,11]*qR[2,1,3]*dis[15,8,16,7]
    @tensor psipsi[:] := A[9,10,16,17]*A[13,12,16,18]*B[1,17,7,2]*B[5,18,7,4]*qL[10,9,11]*qR[2,1,3]*qL[12,13,14]*qR[4,5,6]*sL[11,15]*sR[3,8]*sL[14,15]*sR[6,8] 

    if printdetail ==true
        @printf "phiphi: %.16e   psipsi: %.16e   psiphi: %.16e: err: %.16e  \n" phiphi[1] psipsi[1] psiphi[1] 1-psiphi[1]^2/psipsi[1]/phiphi[1] 
    end
    return 1-psiphi[1]^2/psipsi[1]/phiphi[1] 


end



function compute_fidelity(A,B,wL,wR,sL,sR,dis;printdetail=false)


    @tensor phiphi[:] := A[1,2,3,7]*B[6,7,4,5]*A[1,2,3,8]*B[6,8,4,5]
    @tensor psiphi[:] := A[2,1,5,12]*B[10,12,13,11]*A[6,4,5,8]*B[9,8,13,15]*wL[4,7,3]*wR[15,14,16]*sL[1,2,3]*sR[11,10,16]*dis[6,9,7,14]
    @tensor psipsi[:] := A[2,1,6,13]*B[7,13,11,8]*A[4,3,6,14]*B[9,14,11,10]*sL[1,2,5]*sL[3,4,5]*sR[8,7,12]*sR[10,9,12]

    if printdetail ==true
        @printf "phiphi: %.16e   psipsi: %.16e   psiphi: %.16e: err: %.16e  \n" phiphi[1] psipsi[1] psiphi[1] 1-psiphi[1]^2/psipsi[1]/phiphi[1] 
    end
    return 1-psiphi[1]^2/psipsi[1]/phiphi[1] 


end



function compute_sL_gradient(sL,AsR,grad_1)

    @tensor grad_2[:] := sL[1,2,-3]*AsR[-1,-2,1,2]
    grad = grad_2-grad_1
    return grad
end


function compute_sR_gradient(sR,AsL,grad_1)

    @tensor grad_2[:] := sR[1,2,-3]*AsL[-1,-2,1,2]
    grad = grad_2-grad_1
    return grad
end



# f = <\psi|\psi> -2<\psi|phi>
function compute_f_sL(sL,AsR,grad_1)
    @tensor psiphi[:] := grad_1[1,2,3]*sL[1,2,3]
    @tensor psipsi[:] := AsR[3,4,1,2]*sL[1,2,5]*sL[3,4,5]
    return psipsi[1]-2*psiphi[1]    
end

function compute_f_sR(sR,AsL,grad_1)
    @tensor psiphi[:] := grad_1[1,2,3]*sR[1,2,3]
    @tensor psipsi[:] := AsL[3,4,1,2]*sR[1,2,5]*sR[3,4,5]
    return psipsi[1]-2*psiphi[1]    
end


function update_sL_cg(A,B,wL,wR,sL,sR,dis)

    mu = 0.01
    alpha = 1.0

    d_km1 = zeros(size(sL)...)
    grad_km1 = zeros(size(sL)...)
    sL_temp = deepcopy(sL)


    # F0 
    #F0 = compute_f(A,B,wL,wR,sL,sR,dis)
    # grad_1 = <psi|phi> without sL 
    # AsR = <psi|psi> without sL,sL^\dagger
    # grad_2 = <psi|psi> without sL
    @tensor grad_1[:] := A[-2,-1,12,13]*A[8,10,12,9]*B[1,13,4,2]*B[7,9,4,5]*sR[2,1,3]*wL[10,11,-3]*wR[5,6,3]*dis[8,7,11,6] 
    @tensor AsR[:] := A[-2,-1,8,7]*A[-4,-3,8,9]*B[1,7,5,2]*B[4,9,5,3]*sR[2,1,6]*sR[3,4,6]
    
    @tensor f0_1[:] := grad_1[1,2,3]*sL[1,2,3] 
    @tensor f0_2[:] := AsR[3,4,1,2]*sL[1,2,5]*sL[3,4,5]
    F0 = f0_2[1] - 2*f0_1[1]

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

    return sL_temp

end



function update_sR_cg(A,B,wL,wR,sL,sR,dis)


    mu = 0.01
    alpha = 1.0


    d_km1 = zeros(size(sR)...)
    grad_km1 = zeros(size(sR)...)
    sR_temp = deepcopy(sR)


    @tensor grad_1[:] := A[2,1,5,12]*A[6,4,5,8]*B[-2,12,13,-1]*B[9,8,13,11]*sL[1,2,3]*wL[4,7,3]*wR[11,10,-3]*dis[6,9,7,10] 
    #@tensor grad_2[:] := A[2,1,6,10]*A[4,3,6,9]*B[-2,10,11,-1]*B[7,9,11,8]*sL[1,2,5]*sL[3,4,5]*sR[8,7,-3]
    @tensor AsL[:] := A[1,2,6,7]*A[4,3,6,8]*B[-2,7,9,-1]*B[-4,8,9,-3]*sL[2,1,5]*sL[3,4,5]

    @tensor f0_1[:] := grad_1[1,2,3]*sR[1,2,3] 
    @tensor f0_2[:] := AsL[3,4,1,2]*sR[1,2,5]*sR[3,4,5]

    F0 = f0_2[1] -2*f0_1[1]
    for k in 1:200
        grad_k = compute_sR_gradient(sR_temp,AsL,grad_1)
        if k ==1 
            d_k = -1*grad_k
        else
            beta_k =  dot(grad_k,grad_k)/dot(grad_km1,grad_km1)
            d_k = -1*grad_k+beta_k*d_km1
        end
        grad_km1 = grad_k
        d_km1 = d_k
        
        for j in 1:50 
            F_hat = F0+mu*alpha*dot(grad_k,d_k)        
            sR_temp = sR+alpha*d_k
            F_alpha = compute_f_sR(sR_temp,AsL,grad_1)
            if F_alpha >= F_hat
                alpha = alpha*0.5^j
            else
                sR_temp = sR+alpha*d_k
                break
            end
        end

    end

    return sR_temp


end


#



function compute_fidelity_non_symmetric(Ta,Tb,wLup,wRup,sLup,sRup,wLdn,wRdn,sLdn,sRdn,dis;printdetail=false)


    @tensor phiphi_up[:] := Ta[1,2,3,7]*Tb[6,7,4,5]*Ta[1,2,3,8]*Tb[6,8,4,5]
    @tensor psiphi_up[:] := Ta[2,1,5,12]*Tb[10,12,13,11]*Ta[6,4,5,8]*Tb[9,8,13,15]*wLup[4,7,3]*wRup[15,14,16]*sLup[1,2,3]*sRup[11,10,16]*dis[6,9,7,14]
    @tensor psipsi_up[:] := Ta[2,1,6,13]*Tb[7,13,11,8]*Ta[4,3,6,14]*Tb[9,14,11,10]*sLup[1,2,5]*sLup[3,4,5]*sRup[8,7,12]*sRup[10,9,12]
 
    @tensor phiphi_dn[:] := Tb[1,2,3,7]*Ta[6,7,4,5]*Tb[1,2,3,8]*Ta[6,8,4,5]
    @tensor psiphi_dn[:] := Tb[5,1,2,14]*Tb[5,4,7,13]*Ta[11,14,9,8]*Ta[11,13,16,12]*wLdn[4,6,3]*wRdn[12,15,10]*sLdn[1,2,3]*sRdn[8,9,10]*dis[7,16,6,15]
    @tensor psipsi_dn[:] := Tb[6,1,2,14]*Tb[6,3,4,13]*Ta[11,14,8,7]*Ta[11,13,9,10]*sLdn[1,2,5]*sLdn[3,4,5]*sRdn[7,8,12]*sRdn[10,9,12]


    if printdetail ==true
        @printf "up part phiphi: %.16e   psipsi: %.16e   psiphi: %.16e: err: %.16e  \n" phiphi_up[1] psipsi_up[1] psiphi_up[1] 1-psiphi_up[1]^2/psipsi_up[1]/phiphi_up[1] 
        @printf "dn part phiphi: %.16e   psipsi: %.16e   psiphi: %.16e: err: %.16e  \n" phiphi_dn[1] psipsi_dn[1] psiphi_dn[1] 1-psiphi_dn[1]^2/psipsi_dn[1]/phiphi_dn[1] 
        
    end
    return 1-psiphi_up[1]^2/psipsi_up[1]/phiphi_up[1] +1-psiphi_dn[1]^2/psipsi_dn[1]/phiphi_dn[1] 



end



function compute_fidelity_non_symmetric_impurity(Ta,Tb,Apos,wLup,wRup,sLup,sRup,wLdn,wRdn,sLdn,sRdn,dis;printdetail=false)

    @tensor phiphi_up[:] := Apos[1,2,3,7]*Ta[6,7,4,5]*Apos[1,2,3,8]*Ta[6,8,4,5]
    @tensor psiphi_up[:] := Apos[2,1,5,12]*Ta[10,12,13,11]*Apos[6,4,5,8]*Ta[9,8,13,15]*wLup[4,7,3]*wRup[15,14,16]*sLup[1,2,3]*sRup[11,10,16]*dis[6,9,7,14]
    @tensor psipsi_up[:] := Apos[2,1,6,13]*Ta[7,13,11,8]*Apos[4,3,6,14]*Ta[9,14,11,10]*sLup[1,2,5]*sLup[3,4,5]*sRup[8,7,12]*sRup[10,9,12]
    
    #
    @tensor phiphi_dn[:] := Apos[1,2,3,7]*Ta[6,7,4,5]*Apos[1,2,3,8]*Ta[6,8,4,5]
    @tensor psiphi_dn[:] := Apos[5,1,2,14]*Apos[5,4,7,13]*Ta[11,14,9,8]*Ta[11,13,16,12]*wLdn[4,6,3]*wRdn[12,15,10]*sLdn[1,2,3]*sRdn[8,9,10]*dis[7,16,6,15]
    @tensor psipsi_dn[:] := Apos[6,1,2,14]*Apos[6,3,4,13]*Ta[11,14,8,7]*Ta[11,13,9,10]*sLdn[1,2,5]*sLdn[3,4,5]*sRdn[7,8,12]*sRdn[10,9,12]
    #
    #=
    @tensor phiphi_dn[:] := Ta[1,2,3,7]*Apos[6,7,4,5]*Ta[1,2,3,8]*Apos[6,8,4,5]
    @tensor psiphi_dn[:] := Ta[5,1,2,14]*Ta[5,4,7,13]*Apos[11,14,9,8]*Apos[11,13,16,12]*wLdn[4,6,3]*wRdn[12,15,10]*sLdn[1,2,3]*sRdn[8,9,10]*dis[7,16,6,15]
    @tensor psipsi_dn[:] := Ta[6,1,2,14]*Ta[6,3,4,13]*Apos[11,14,8,7]*Apos[11,13,9,10]*sLdn[1,2,5]*sLdn[3,4,5]*sRdn[7,8,12]*sRdn[10,9,12]
    =#

    if printdetail ==true
        println("Impurity fidelity")
        @printf "up part phiphi: %.16e   psipsi: %.16e   psiphi: %.16e: err: %.16e  \n" phiphi_up[1] psipsi_up[1] psiphi_up[1] 1-psiphi_up[1]^2/psipsi_up[1]/phiphi_up[1] 
        @printf "dn part phiphi: %.16e   psipsi: %.16e   psiphi: %.16e: err: %.16e  \n" phiphi_dn[1] psipsi_dn[1] psiphi_dn[1] 1-psiphi_dn[1]^2/psipsi_dn[1]/phiphi_dn[1] 
        
    end
    return 1-psiphi_up[1]^2/psipsi_up[1]/phiphi_up[1] +1-psiphi_dn[1]^2/psipsi_dn[1]/phiphi_dn[1] 
end


function isometry_initial(Ta,Tb,chimax;rg = false)

      
    chiTa1 = size(Ta,1);chiTa2 = size(Ta,2)
    chiTb4 = size(Tb,4);chiTb1 = size(Tb,1)
    chiTb2 = size(Tb,2)
    # initial
    @tensor temp[:] := Ta[-2,-1,13,14]*Tb[13,1,2,5]*Tb[11,1,2,6]*Ta[-4,-3,11,12]*Tb[8,14,9,7]*Ta[9,5,3,4]*Ta[10,6,3,4]*Tb[8,12,10,7]
    temp = reshape(temp,chiTa1*chiTa2,chiTa1*chiTa2)
    Rlt = svd(temp)

    #println(Rlt.S/maximum(Rlt.S))
    chitemp = sum(Rlt.S/maximum(Rlt.S) .> 1.0e-14)
    if rg == false
        chikept = min(chiTa1*chiTa2,chimax,chitemp)
    else
        chikept = min(chiTa1*chiTa2,chimax)
    end
    sLup = deepcopy(reshape(Rlt.U[:,1:chikept],chiTa2,chiTa1,chikept))
    wLup = deepcopy(sLup)    

    #=
    # identity if no truncation needed
    if chiTa1*chiTa2 == size(wLup,3)
        sLup = reshape(eye(size(sLup,3)),size(sLup)...)
        wLup = reshape(eye(size(wLup,3)),size(wLup)...)
    end
    =#

    @tensor temp[:] := Ta[7,8,9,11]*Tb[9,1,2,5]*Tb[10,1,2,6]*Ta[7,8,10,13]*Tb[-2,11,12,-1]*Ta[12,5,3,4]*Ta[14,6,3,4]*Tb[-4,13,14,-3]
    temp = reshape(temp,chiTb1*chiTb4,chiTb1*chiTb4)
    Rlt = svd(temp)

    #println(Rlt.S/maximum(Rlt.S))
    chitemp = sum(Rlt.S/maximum(Rlt.S) .> 1.0e-14)
    if rg == false
        chikept = min(chiTb1*chiTb4,chimax,chitemp)
    else
        chikept = min(chiTb1*chiTb4,chimax)
    end
    sRup = deepcopy(reshape(Rlt.U[:,1:chikept],chiTb4,chiTb1,chikept))
    wRup = deepcopy(sRup)
    dis = reshape(eye(chiTa1*chiTb1),chiTa1,chiTb1,chiTa1,chiTb1)

    #=
    # identity if no truncation needed
    if chiTb1*chiTb4 == size(wRup,3)
        sRup = reshape(eye(size(sRup,3)),size(sRup)...)
        wRup = reshape(eye(size(wRup,3)),size(wRup)...)
    end
    =#


    chiTa3 = size(Ta,3); chiTa2 = size(Ta,2) 
    chiTb3 = size(Tb,3); chiTb4 = size(Tb,4)
    @tensor temp[:] := Ta[8,7,13,10]*Tb[13,-1,-2,14]*Ta[8,7,11,9]*Tb[11,-3,-4,12]*Tb[1,10,6,2]*Ta[6,14,4,3]*Tb[1,9,5,2]*Ta[5,12,4,3]
    temp = reshape(temp,chiTa3*chiTa2,chiTa3*chiTb2)
    Rlt = svd(temp)
    chitemp = sum(Rlt.S/maximum(Rlt.S) .> 1.0e-14)
    if rg == false
        chikept = min(chiTa2*chiTa3,chimax,chitemp)
    else
        chikept = min(chiTa2*chiTa3,chimax)
    end
    sLdn = reshape(Rlt.U[:,1:chikept],chiTa2,chiTa3,chikept)
    wLdn = deepcopy(sLdn)
    
    #=
    # identity if no truncation needed
    if chiTa2*chiTa3 == size(wLdn,3)
        sLdn = reshape(eye(size(wLdn,3)),size(wLdn)...)
        wLdn = reshape(eye(size(wLdn,3)),size(wLdn)...)
    end
    =#

    @tensor temp[:] := Ta[2,1,6,10]*Tb[6,4,3,13]*Ta[2,1,5,9]*Tb[5,4,3,11]*Tb[7,10,14,8]*Ta[14,13,-2,-1]*Tb[7,9,12,8]*Ta[12,11,-4,-3]
    temp = reshape(temp,chiTb3*chiTb4,chiTb3*chiTb4)
    Rlt = svd(temp)
    chitemp = sum(Rlt.S/maximum(Rlt.S) .> 1.0e-14)
    if rg == false
        chikept = min(chiTb3*chiTb4,chimax,chitemp)
    else
        chikept = min(chiTb3*chiTb4,chimax)
    end

    sRdn = reshape(Rlt.U[:,1:chikept],chiTb4,chiTb3,chikept)
    wRdn = deepcopy(sRdn)
    #=
    # identity if no truncation needed
    if chiTa3*chiTb4 == size(wRdn,3)
        sRdn = reshape(eye(size(wRdn,3)),size(wRdn)...)
        wRdn = reshape(eye(size(wRdn,3)),size(wRdn)...)
    end
    =#
    return wLup,wRup,sLup,sRup,wLdn,wRdn,sLdn,sRdn,dis
end



function isometry_impurity_initial(Ta,Tb,Apos,chimax;rg = false)

      
    chiTa1 = size(Ta,1);chiTa2 = size(Ta,2)
    chiTb4 = size(Tb,4);chiTb1 = size(Tb,1)
    chiTb2 = size(Tb,2)
    # initial
    @tensor temp1[:] := Apos[-2,-1,13,14]*Tb[13,1,2,5]*Tb[11,1,2,6]*Apos[-4,-3,11,12]*Tb[8,14,9,7]*Ta[9,5,3,4]*Ta[10,6,3,4]*Tb[8,12,10,7]
    #@tensor temp2[:] := Ta[-2,-1,13,14]*Apos[13,1,2,5]*Apos[11,1,2,6]*Ta[-4,-3,11,12]*Tb[8,14,9,7]*Ta[9,5,3,4]*Ta[10,6,3,4]*Tb[8,12,10,7]
    #@tensor temp3[:] := Ta[-2,-1,13,14]*Tb[13,1,2,5]*Tb[11,1,2,6]*Ta[-4,-3,11,12]*Apos[8,14,9,7]*Ta[9,5,3,4]*Ta[10,6,3,4]*Apos[8,12,10,7]
    #@tensor temp4[:] := Ta[-2,-1,13,14]*Tb[13,1,2,5]*Tb[11,1,2,6]*Ta[-4,-3,11,12]*Tb[8,14,9,7]*Apos[9,5,3,4]*Apos[10,6,3,4]*Tb[8,12,10,7]
    temp = temp1#+temp2+temp3+temp4
    temp = reshape(temp,chiTa1*chiTa2,chiTa1*chiTa2)
    Rlt = svd(temp)

    #println(Rlt.S/maximum(Rlt.S))
    chitemp = sum(Rlt.S/maximum(Rlt.S) .> 1.0e-14)
    if rg == false
        chikept = min(chiTa1*chiTa2,chimax,chitemp)
    else
        chikept = min(chiTa1*chiTa2,chimax)
    end
    sLup = deepcopy(reshape(Rlt.U[:,1:chikept],chiTa2,chiTa1,chikept))
    wLup = deepcopy(sLup)    

    #=
    # identity if no truncation needed
    if chiTa1*chiTa2 == size(wLup,3)
        sLup = reshape(eye(size(sLup,3)),size(sLup)...)
        wLup = reshape(eye(size(wLup,3)),size(wLup)...)
    end
    =#

    #@tensor temp1[:] := Apos[7,8,9,11]*Tb[9,1,2,5]*Tb[10,1,2,6]*Apos[7,8,10,13]*Tb[-2,11,12,-1]*Ta[12,5,3,4]*Ta[14,6,3,4]*Tb[-4,13,14,-3]
    #@tensor temp2[:] := Ta[7,8,9,11]*Apos[9,1,2,5]*Apos[10,1,2,6]*Ta[7,8,10,13]*Tb[-2,11,12,-1]*Ta[12,5,3,4]*Ta[14,6,3,4]*Tb[-4,13,14,-3]
    @tensor temp3[:] := Ta[7,8,9,11]*Tb[9,1,2,5]*Tb[10,1,2,6]*Ta[7,8,10,13]*Apos[-2,11,12,-1]*Ta[12,5,3,4]*Ta[14,6,3,4]*Apos[-4,13,14,-3]
    #@tensor temp4[:] := Ta[7,8,9,11]*Tb[9,1,2,5]*Tb[10,1,2,6]*Ta[7,8,10,13]*Tb[-2,11,12,-1]*Apos[12,5,3,4]*Apos[14,6,3,4]*Tb[-4,13,14,-3]
    temp = temp3#+temp2+temp3+temp4
    temp = reshape(temp,chiTb1*chiTb4,chiTb1*chiTb4)
    Rlt = svd(temp)

    #println(Rlt.S/maximum(Rlt.S))
    chitemp = sum(Rlt.S/maximum(Rlt.S) .> 1.0e-14)
    if rg == false
        chikept = min(chiTb1*chiTb4,chimax,chitemp)
    else
        chikept = min(chiTb1*chiTb4,chimax)
    end
    sRup = deepcopy(reshape(Rlt.U[:,1:chikept],chiTb4,chiTb1,chikept))
    wRup = deepcopy(sRup)
    dis = reshape(eye(chiTa1*chiTb1),chiTa1,chiTb1,chiTa1,chiTb1)

    #=
    # identity if no truncation needed
    if chiTb1*chiTb4 == size(wRup,3)
        sRup = reshape(eye(size(sRup,3)),size(sRup)...)
        wRup = reshape(eye(size(wRup,3)),size(wRup)...)
    end
    =#


    chiTa3 = size(Ta,3); chiTa2 = size(Ta,2) 
    chiTb3 = size(Tb,3); chiTb4 = size(Tb,4)
    #@tensor temp1[:] := Apos[8,7,13,10]*Tb[13,-1,-2,14]*Apos[8,7,11,9]*Tb[11,-3,-4,12]*Tb[1,10,6,2]*Ta[6,14,4,3]*Tb[1,9,5,2]*Ta[5,12,4,3]
    @tensor temp2[:] := Ta[8,7,13,10]*Apos[13,-1,-2,14]*Ta[8,7,11,9]*Apos[11,-3,-4,12]*Tb[1,10,6,2]*Ta[6,14,4,3]*Tb[1,9,5,2]*Ta[5,12,4,3]
    #@tensor temp3[:] := Ta[8,7,13,10]*Tb[13,-1,-2,14]*Ta[8,7,11,9]*Tb[11,-3,-4,12]*Apos[1,10,6,2]*Ta[6,14,4,3]*Apos[1,9,5,2]*Ta[5,12,4,3]
    #@tensor temp4[:] := Ta[8,7,13,10]*Tb[13,-1,-2,14]*Ta[8,7,11,9]*Tb[11,-3,-4,12]*Tb[1,10,6,2]*Apos[6,14,4,3]*Tb[1,9,5,2]*Apos[5,12,4,3]
    temp = temp2#+temp2+temp3+temp4
    temp = reshape(temp,chiTa3*chiTa2,chiTa3*chiTb2)
    Rlt = svd(temp)
    chitemp = sum(Rlt.S/maximum(Rlt.S) .> 1.0e-14)
    if rg == false
        chikept = min(chiTa2*chiTa3,chimax,chitemp)
    else
        chikept = min(chiTa2*chiTa3,chimax)
    end
    sLdn = reshape(Rlt.U[:,1:chikept],chiTa2,chiTa3,chikept)
    wLdn = deepcopy(sLdn)
    
    #=
    # identity if no truncation needed
    if chiTa2*chiTa3 == size(wLdn,3)
        sLdn = reshape(eye(size(wLdn,3)),size(wLdn)...)
        wLdn = reshape(eye(size(wLdn,3)),size(wLdn)...)
    end
    =#

    #@tensor temp1[:] := Apos[2,1,6,10]*Tb[6,4,3,13]*Apos[2,1,5,9]*Tb[5,4,3,11]*Tb[7,10,14,8]*Ta[14,13,-2,-1]*Tb[7,9,12,8]*Ta[12,11,-4,-3]
    #@tensor temp2[:] := Ta[2,1,6,10]*Apos[6,4,3,13]*Ta[2,1,5,9]*Apos[5,4,3,11]*Tb[7,10,14,8]*Ta[14,13,-2,-1]*Tb[7,9,12,8]*Ta[12,11,-4,-3]
    #@tensor temp3[:] := Ta[2,1,6,10]*Tb[6,4,3,13]*Ta[2,1,5,9]*Tb[5,4,3,11]*Apos[7,10,14,8]*Ta[14,13,-2,-1]*Apos[7,9,12,8]*Ta[12,11,-4,-3]
    @tensor temp4[:] := Ta[2,1,6,10]*Tb[6,4,3,13]*Ta[2,1,5,9]*Tb[5,4,3,11]*Tb[7,10,14,8]*Apos[14,13,-2,-1]*Tb[7,9,12,8]*Apos[12,11,-4,-3]
    temp = temp4#+temp2+temp3+temp4
    temp = reshape(temp,chiTb3*chiTb4,chiTb3*chiTb4)
    Rlt = svd(temp)
    chitemp = sum(Rlt.S/maximum(Rlt.S) .> 1.0e-14)
    if rg == false
        chikept = min(chiTb3*chiTb4,chimax,chitemp)
    else
        chikept = min(chiTb3*chiTb4,chimax)
    end

    sRdn = reshape(Rlt.U[:,1:chikept],chiTb4,chiTb3,chikept)
    wRdn = deepcopy(sRdn)
    #=
    # identity if no truncation needed
    if chiTa3*chiTb4 == size(wRdn,3)
        sRdn = reshape(eye(size(wRdn,3)),size(wRdn)...)
        wRdn = reshape(eye(size(wRdn,3)),size(wRdn)...)
    end
    =#
    return wLup,wRup,sLup,sRup,wLdn,wRdn,sLdn,sRdn,dis

end





function renormalize(Ta,Tb,Apos,wLup,wRup,sLup,sRup,wLdn,wRdn,sLdn,sRdn,dis,chimax;rg=false)


    @tensor sT_left[:] := Ta[2,1,5,-3]*Ta[5,3,4,-4]*sLup[1,2,-1]*sLdn[3,4,-2]
    @tensor sT_right[:] := Tb[1,-3,5,2]*Tb[5,-4,4,3]*sRup[2,1,-1]*sRdn[3,4,-2]
    @tensor wLwR_up[:] := wLup[1,4,-2]*wRup[1,3,-1]*wLup[2,4,-4]*wRup[2,3,-3] 
    @tensor wLwR_dn[:] := wLdn[1,3,-2]*wRdn[1,4,-1]*wLdn[2,3,-4]*wRdn[2,4,-3]


    # compute isometry y
    @tensor Env_y[:] := sT_left[3,5,1,2]*sT_left[4,6,1,2]*sT_right[7,8,-2,-1]*sT_right[9,10,-4,-3]*wLwR_up[7,3,9,4]*wLwR_dn[8,5,10,6]
    sizeEnv_y = size(Env_y)
    Env_y = reshape(Env_y,prod(sizeEnv_y[1:2]),prod(sizeEnv_y[1:2]))
    Rlt = svd(Env_y)

    chitemp = sum(Rlt.S/maximum(Rlt.S) .> 1.0e-14)
    if rg == false    
        chikept = min(prod(sizeEnv_y[1:2]),chimax,chitemp)
    else
        chikept = min(prod(sizeEnv_y[1:2]),chimax)
    end
    y = reshape(Rlt.U[:,1:chikept],sizeEnv_y[1],sizeEnv_y[2],chikept)
    println("chikept: ",chikept)
    println(norm(tr(Env_y)-tr(Env_y*Rlt.U[:,1:chikept]*(Rlt.U[:,1:chikept])'))/norm(tr(Env_y)))


    # compute isometry w 
    @tensor Env_w[:] := sT_left[1,2,3,4]*sT_right[25,12,3,4]*sT_left[26,19,17,18]*sT_right[14,13,17,18]*wRup[24,-1,25]*wLup[24,-2,26]*wRdn[9,7,12]*wLdn[9,8,19]*
                        sT_left[1,2,6,5]*sT_right[22,11,6,5]*sT_left[23,20,16,15]*sT_right[14,13,16,15]*wRup[21,-3,22]*wLup[21,-4,23]*wRdn[10,7,11]*wLdn[10,8,20]
    sizeEnv_w = size(Env_w)
    Env_w = reshape(Env_w,prod(sizeEnv_w[1:2]),prod(sizeEnv_w[1:2]))
    Rlt = svd(Env_w)        
    chitemp = sum(Rlt.S/maximum(Rlt.S) .> 1.0e-14)
    if rg == false    
        chikept = min(prod(sizeEnv_w[1:2]),chimax,chitemp)
    else
        chikept = min(prod(sizeEnv_w[1:2]),chimax)
    end
    w = reshape(Rlt.U[:,1:chikept],sizeEnv_w[1],sizeEnv_w[2],chikept)
    
    println("chikept: ",chikept)
    println(norm(tr(Env_w)-tr(Env_w*Rlt.U[:,1:chikept]*(Rlt.U[:,1:chikept])'))/norm(tr(Env_w)))
    @tensor Ta_new[:] := sT_right[6,13,1,2]*sT_left[14,12,7,8]*wRup[3,4,6]*wLup[3,5,14]*wRdn[9,10,13]*wLdn[9,11,12]*y[2,1,-2]*y[8,7,-4]*w[10,11,-3]*w[4,5,-1]
    
    #=
    @tensor sTpos_left[:] := Ta[2,1,5,-3]*Apos[5,3,4,-4]*sLup[1,2,-1]*sLdn[3,4,-2]
    @tensor Apos_new[:] := sT_right[6,13,1,2]*sTpos_left[14,12,7,8]*wRup[3,4,6]*wLup[3,5,14]*wRdn[9,10,13]*wLdn[9,11,12]*y[2,1,-2]*y[8,7,-4]*w[10,11,-3]*w[4,5,-1]
    return Ta_new,Apos_new,y,w
    =#

    #
    @tensor sTpos_left[:] := Apos[2,1,5,-3]*Ta[5,3,4,-4]*sLup[1,2,-1]*sLdn[3,4,-2]+Ta[2,1,5,-3]*Apos[5,3,4,-4]*sLup[1,2,-1]*sLdn[3,4,-2]
    @tensor sTpos_right[:] := Apos[1,-3,5,2]*Tb[5,-4,4,3]*sRup[2,1,-1]*sRdn[3,4,-2]+Tb[1,-3,5,2]*Apos[5,-4,4,3]*sRup[2,1,-1]*sRdn[3,4,-2]
    @tensor Apos_new[:] := sT_right[6,13,1,2]*sTpos_left[14,12,7,8]*wRup[3,4,6]*wLup[3,5,14]*wRdn[9,10,13]*wLdn[9,11,12]*y[2,1,-2]*y[8,7,-4]*w[10,11,-3]*w[4,5,-1]+
                     sTpos_right[6,13,1,2]*sT_left[14,12,7,8]*wRup[3,4,6]*wLup[3,5,14]*wRdn[9,10,13]*wLdn[9,11,12]*y[2,1,-2]*y[8,7,-4]*w[10,11,-3]*w[4,5,-1]
    return Ta_new,Apos_new/4,y,w
    #    


end





function renormalize_impurity(Ta,Tb,Apos,wLup,wRup,sLup,sRup,wLdn,wRdn,sLdn,sRdn,dis,
    wLup_imp,wRup_imp,sLup_imp,sRup_imp,wLdn_imp,wRdn_imp,sLdn_imp,sRdn_imp,dis_imp,chimax;rg=false)


    @tensor sT_left[:] := Ta[2,1,5,-3]*Ta[5,3,4,-4]*sLup[1,2,-1]*sLdn[3,4,-2]
    @tensor sT_right[:] := Tb[1,-3,5,2]*Tb[5,-4,4,3]*sRup[2,1,-1]*sRdn[3,4,-2]
    @tensor wLwR_up[:] := wLup[1,4,-2]*wRup[1,3,-1]*wLup[2,4,-4]*wRup[2,3,-3] 
    @tensor wLwR_dn[:] := wLdn[1,3,-2]*wRdn[1,4,-1]*wLdn[2,3,-4]*wRdn[2,4,-3]


    # compute isometry y
    @tensor Env_y[:] := sT_left[3,5,1,2]*sT_left[4,6,1,2]*sT_right[7,8,-2,-1]*sT_right[9,10,-4,-3]*wLwR_up[7,3,9,4]*wLwR_dn[8,5,10,6]
    sizeEnv_y = size(Env_y)
    Env_y = reshape(Env_y,prod(sizeEnv_y[1:2]),prod(sizeEnv_y[1:2]))
    Rlt = svd(Env_y)

    chitemp = sum(Rlt.S/maximum(Rlt.S) .> 1.0e-14)
    if rg == false    
        chikept = min(prod(sizeEnv_y[1:2]),chimax,chitemp)
    else
        chikept = min(prod(sizeEnv_y[1:2]),chimax)
    end
    y = reshape(Rlt.U[:,1:chikept],sizeEnv_y[1],sizeEnv_y[2],chikept)
    println("chikept: ",chikept)
    println(norm(tr(Env_y)-tr(Env_y*Rlt.U[:,1:chikept]*(Rlt.U[:,1:chikept])'))/norm(tr(Env_y)))


    # compute isometry w 
    @tensor Env_w[:] := sT_left[1,2,3,4]*sT_right[25,12,3,4]*sT_left[26,19,17,18]*sT_right[14,13,17,18]*wRup[24,-1,25]*wLup[24,-2,26]*wRdn[9,7,12]*wLdn[9,8,19]*
                        sT_left[1,2,6,5]*sT_right[22,11,6,5]*sT_left[23,20,16,15]*sT_right[14,13,16,15]*wRup[21,-3,22]*wLup[21,-4,23]*wRdn[10,7,11]*wLdn[10,8,20]
    sizeEnv_w = size(Env_w)
    Env_w = reshape(Env_w,prod(sizeEnv_w[1:2]),prod(sizeEnv_w[1:2]))
    Rlt = svd(Env_w)        
    chitemp = sum(Rlt.S/maximum(Rlt.S) .> 1.0e-14)
    if rg == false    
        chikept = min(prod(sizeEnv_w[1:2]),chimax,chitemp)
    else
        chikept = min(prod(sizeEnv_w[1:2]),chimax)
    end
    w = reshape(Rlt.U[:,1:chikept],sizeEnv_w[1],sizeEnv_w[2],chikept)
    
    println("chikept: ",chikept)
    println(norm(tr(Env_w)-tr(Env_w*Rlt.U[:,1:chikept]*(Rlt.U[:,1:chikept])'))/norm(tr(Env_w)))
    @tensor Ta_new[:] := sT_right[6,13,1,2]*sT_left[14,12,7,8]*wRup[3,4,6]*wLup[3,5,14]*wRdn[9,10,13]*wLdn[9,11,12]*y[2,1,-2]*y[8,7,-4]*w[10,11,-3]*w[4,5,-1]

    @tensor sTup_left_imp[:] := Apos[2,1,5,-3]*Ta[5,3,4,-4]*sLup_imp[1,2,-1]*sLdn[3,4,-2]
    @tensor sTdn_left_imp[:] := Ta[2,1,5,-3]*Apos[5,3,4,-4]*sLup[1,2,-1]*sLdn_imp[3,4,-2]
    @tensor sTup_right_imp[:] := Apos[1,-3,5,2]*Tb[5,-4,4,3]*sRup_imp[2,1,-1]*sRdn[3,4,-2]
    @tensor sTdn_right_imp[:] := Ta[1,-3,5,2]*Apos[5,-4,4,3]*sRup[2,1,-1]*sRdn_imp[3,4,-2]
    @tensor Apos_new[:] := #sT_right[6,13,1,2]*sTup_left_imp[14,12,7,8]*wRup[3,4,6]*wLup_imp[3,5,14]*wRdn[9,10,13]*wLdn[9,11,12]*y[2,1,-2]*y[8,7,-4]*w[10,11,-3]*w[4,5,-1]#+
                        #sT_right[6,13,1,2]*sTdn_left_imp[14,12,7,8]*wRup[3,4,6]*wLup[3,5,14]*wRdn[9,10,13]*wLdn_imp[9,11,12]*y[2,1,-2]*y[8,7,-4]*w[10,11,-3]*w[4,5,-1]+
                        sTup_right_imp[6,13,1,2]*sT_left[14,12,7,8]*wRup_imp[3,4,6]*wLup[3,5,14]*wRdn[9,10,13]*wLdn[9,11,12]*y[2,1,-2]*y[8,7,-4]*w[10,11,-3]*w[4,5,-1]#+
                        #sTdn_right_imp[6,13,1,2]*sT_left[14,12,7,8]*wRup[3,4,6]*wLup[3,5,14]*wRdn_imp[9,10,13]*wLdn[9,11,12]*y[2,1,-2]*y[8,7,-4]*w[10,11,-3]*w[4,5,-1]
    return Ta_new,Apos_new,y,w

    #=
    @tensor Ta_new[:] := sT_right[6,13,1,2]*sT_left[14,12,7,8]*wRup[3,4,6]*wLup[3,5,14]*wRdn[9,10,13]*wLdn[9,11,12]*y[2,1,-2]*y[8,7,-4]*w[10,11,-3]*w[4,5,-1]
    @tensor sTpos_left[:] := Ta[2,1,5,-3]*Apos[5,3,4,-4]*sLup[1,2,-1]*sLdn[3,4,-2]
    @tensor Apos_new[:] := sT_right[6,13,1,2]*sTpos_left[14,12,7,8]*wRup[3,4,6]*wLup[3,5,14]*wRdn[9,10,13]*wLdn[9,11,12]*y[2,1,-2]*y[8,7,-4]*w[10,11,-3]*w[4,5,-1]
    return Ta_new,Apos_new,y,w
    =#
end











function renormalize_unit_two(Ta,Tb,wLup,wRup,sLup,sRup,wLdn,wRdn,sLdn,sRdn,dis,chimax)

    @tensor Ta_new[:] := Ta[2,1,5,11]*Tb[6,11,10,7]*Tb[5,3,4,12]*Ta[10,12,9,8]*sLup[1,2,-1]*sRup[7,6,-4]*sLdn[3,4,-2]*sRdn[8,9,-3]
    @tensor Tb_new[:] := wLup[4,2,-3]*wRup[4,1,-2]*wLdn[3,2,-4]*wRdn[3,1,-1] 
    return Ta_new,Tb_new
end