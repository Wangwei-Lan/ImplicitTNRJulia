function compute_gradient_sLup(psipsi_wo_sLupsLup,psiphi_wo_sLup,sLupnew)

    @tensor psipsi_wo_sLup[:] := psipsi_wo_sLupsLup[-1,-2,1,2]*sLupnew[1,2,-3]

    grad_k = psipsi_wo_sLup-2*psipsi_wo_sLup

end

function compute_f_sLup(psipsi_wo_sLupsLup,psiphi_wo_sLup,sLuptemp)
    @tensor psipsi[:] := psipsi_wo_sLupsLup[3,4,1,2]*sLuptemp[3,4,5]*sLuptemp[1,2,5]
    @tensor psiphi[:] := psiphi_wo_sLup[1,2,3]*sLuptemp[1,2,3]
    F0 = psipsi[1] - 2*psiphi[1]
    return F0
end


function update_sLup_non_symmetric(Ta,Tb,wLup,wRup,sLup,sRup,dis)

    @tensor P[:] := Ta[-2,-1,12,13]*Ta[10,9,12,11]*Tb[1,13,4,2]*Tb[6,11,4,5]*
                    wLup[9,8,-3]*wRup[5,7,3]*sRup[2,1,3]*dis[10,6,8,7]
    @tensor B[:] := Ta[-2,-1,8,7]*Ta[-4,-3,8,9]*Tb[1,7,5,2]*Tb[4,9,5,3]*
                    sRup[2,1,6]*sRup[3,4,6]
    sLup = idr_update(P,B,2)
    return sLup
end


function update_sLup_non_symmetric_impurity(Ta,Tb,Apos,wLup,wRup,sLup,sRup,wLup_imp,wRup_imp,sLup_imp,sRup_imp,dis)


    @tensor P[:] := Ta[-2,-1,12,13]*Ta[10,9,12,11]*Tb[1,13,4,2]*Tb[6,11,4,5]*
                    wLup[9,8,-3]*wRup[5,7,3]*sRup[2,1,3]*dis[10,6,8,7]
    @tensor B[:] := Ta[-2,-1,8,7]*Ta[-4,-3,8,9]*Tb[1,7,5,2]*Tb[4,9,5,3]*
                    sRup[2,1,6]*sRup[3,4,6]
    @tensor P_imp[:] := Ta[-2,-1,12,13]*Ta[10,9,12,11]*Apos[1,13,4,2]*Apos[6,11,4,5]*
                    wLup[9,8,-3]*wRup_imp[5,7,3]*sRup_imp[2,1,3]*dis[10,6,8,7]
    @tensor B_imp[:] := Ta[-2,-1,8,7]*Ta[-4,-3,8,9]*Apos[1,7,5,2]*Apos[4,9,5,3]*
                    sRup_imp[2,1,6]*sRup_imp[3,4,6]
    sLup = idr_update(P+P_imp,B+B_imp,2)

    return sLup

end


function update_sLdn_non_symmetric(Ta,Tb,wLdn,wRdn,sLdn,sRdn,dis)

    @tensor P[:] := Tb[13,-1,-2,14]*Tb[13,11,10,9]*Ta[4,14,1,2]*Ta[4,9,7,5]*
                        wLdn[11,8,-3]*wRdn[5,6,3]*sRdn[2,1,3]*dis[10,7,8,6]
    @tensor B[:] := Tb[8,-1,-2,9]*Tb[8,-3,-4,7]*Ta[5,9,3,4]*Ta[5,7,1,2]*
                        sRdn[4,3,6]*sRdn[2,1,6]
    sLdn = idr_update(P,B,2)
    return sLdn
end





function update_sLup_non_symmetric_cg(Ta,Tb,wLup,wRup,sLup,sRup,dis)

    @tensor psiphi_wo_sLup[:] := Ta[-2,-1,12,13]*Ta[10,9,12,11]*Tb[1,13,4,2]*Tb[7,11,4,5]*sRup[2,1,3]*wLup[9,8,-3]*wRup[5,6,3]*dis[10,7,8,6]
    @tensor psipsi_wo_sLupsLup[:] := Ta[-2,-1,8,7]*Ta[-4,-3,8,9]*Tb[1,7,5,2]*Tb[3,9,5,4]*sRup[2,1,6]*sRup[4,3,6]
    @tensor psipsi_wo_sLup[:] := psipsi_wo_sLupsLup[-1,-2,1,2]*sLup[1,2,-3]

    # compute gradient    
    @tensor psipsi[:] := psipsi_wo_sLupsLup[3,4,1,2]*sLup[3,4,5]*sLup[1,2,5]
    @tensor psiphi[:] := psiphi_wo_sLup[1,2,3]*sLup[1,2,3]
    F0 = psipsi[1] - 2*psiphi[1]    
    grad_k = psipsi_wo_sLup-psiphi_wo_sLup

    d_km1 = zeros(size(sLup)...)
    grad_km1 = zeros(size(sLup)...)
    sLuptemp = deepcopy(sLup)

    for k in 1:100

        # compute new derivative 
        grad_k = compute_gradient_sLup(psipsi_wo_sLupsLup,psiphi_wo_sLup,sLuptemp)
        
        # compute k-th update gradient 
        if k == 1
            d_k = -1*grad_k
        else
            beta_k = dot(grad_k,grad_k)/dot(grad_km1,grad_km1)
            d_k = -1*grad_k+beta_k*d_km1
        end

        # update for next iteration
        grad_km1 = grad_k
        d_km1 = d_k
        
        mu = 0.01 
        alpha = 1.0
        # determine alpha
        for j in 1:50
            F_hat = F0 + mu*alpha*dot(grad_k,d_k)
            sLuptemp = sLup+alpha*d_k

            F_alpha = compute_f_sLup(psipsi_wo_sLupsLup,psiphi_wo_sLup,sLuptemp)

            if F_alpha >= F_hat
                alpha = alpha*0.5^j
            else F_alpha < F_hat
                sLuptemp = sLup+alpha*d_k
            end

        end

    end

    return sLuptemp
    #=
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
        =#
end

