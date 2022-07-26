function compute_isometry_up(A,chimax)

    @tensor EnvL[:] := A[-2,-1,5,4]*A[-4,-3,5,6]*A[3,4,1,2]*A[3,6,1,2]
    sizeEnvL = size(EnvL)
    EnvL = reshape(EnvL,prod(sizeEnvL[1:2]),prod(sizeEnvL[3:4]))
    RltL = svd(EnvL)
    chitemp = sum(RltL.S/maximum(RltL.S) .> 1.0e-14)
    chikept = min(chitemp,chimax,prod(sizeEnvL[1:2]))
    wLup = reshape(RltL.U[:,1:chikept],sizeEnvL[1],sizeEnvL[2],chikept)
 

    @tensor EnvR[:] := A[1,2,3,4]*A[1,2,3,5]*A[-2,4,6,-1]*A[-4,5,6,-3]
    sizeEnvR = size(EnvR)
    EnvR = reshape(EnvR,prod(sizeEnvR[1:2]),prod(sizeEnvR[3:4]))
    RltR = svd(EnvR)
    chitemp = sum(RltR.S/maximum(RltR.S) .> 1.0e-14)
    chikept = min(chitemp,chimax,prod(sizeEnvR[1:2]))
    wRup = reshape(RltR.U[:,1:chikept],sizeEnvR[1],sizeEnvR[2],chikept)
 
    return wLup,wRup
end


function compute_isometry_up_impurity(A,Aimp,chimax)

    @tensor EnvL[:] := Aimp[-2,-1,5,4]*Aimp[-4,-3,5,6]*A[3,4,1,2]*A[3,6,1,2]
    sizeEnvL = size(EnvL)
    EnvL = reshape(EnvL,prod(sizeEnvL[1:2]),prod(sizeEnvL[3:4]))
    RltL = svd(EnvL)
    chitemp = sum(RltL.S/maximum(RltL.S) .> 1.0e-14)
    chikept = min(chitemp,chimax,prod(sizeEnvL[1:2]))
    wLup = reshape(RltL.U[:,1:chikept],sizeEnvL[1],sizeEnvL[2],chikept)
 

    @tensor EnvR[:] := A[1,2,3,4]*A[1,2,3,5]*Aimp[-2,4,6,-1]*Aimp[-4,5,6,-3]
    sizeEnvR = size(EnvR)
    EnvR = reshape(EnvR,prod(sizeEnvR[1:2]),prod(sizeEnvR[3:4]))
    RltR = svd(EnvR)
    chitemp = sum(RltR.S/maximum(RltR.S) .> 1.0e-14)
    chikept = min(chitemp,chimax,prod(sizeEnvR[1:2]))
    wRup = reshape(RltR.U[:,1:chikept],sizeEnvR[1],sizeEnvR[2],chikept)
 
    return wLup,wRup
end


function compute_isometry_dn(A,chimax)

    @tensor EnvL[:] := A[5,-1,-2,6]*A[1,6,3,2]*A[5,-3,-4,4]*A[1,4,3,2]
    sizeEnvL = size(EnvL)
    EnvL = reshape(EnvL,prod(sizeEnvL[1:2]),prod(sizeEnvL[3:4]))
    RltL = svd(EnvL)
    chitemp = sum(RltL.S/maximum(RltL.S) .> 1.0e-14)
    chikept = min(chitemp,chimax,prod(sizeEnvL[1:2]))
    wLdn = reshape(RltL.U[:,1:chikept],sizeEnvL[1],sizeEnvL[2],chikept)
 



    @tensor EnvR[:] := A[3,2,1,5]*A[3,2,1,4]*A[6,5,-2,-1]*A[6,4,-4,-3]
    sizeEnvR = size(EnvR)
    EnvR = reshape(EnvR,prod(sizeEnvR[1:2]),prod(sizeEnvR[3:4]))
    sizeEnLR = size(EnvR)
    EnvR = reshape(EnvR,prod(sizeEnvR[1:2]),prod(sizeEnvR[3:4]))
    RltR = svd(EnvR)
    chitemp = sum(RltR.S/maximum(RltR.S) .> 1.0e-14)
    chikept = min(chitemp,chimax,prod(sizeEnvR[1:2]))
    wRdn = reshape(RltR.U[:,1:chikept],sizeEnvR[1],sizeEnvR[2],chikept)
    
    return wLdn,wRdn
end 

function compute_isometry_dn_impurity(A,Aimp,chimax)

    @tensor EnvL[:] := Aimp[5,-1,-2,6]*Aimp[1,6,3,2]*A[5,-3,-4,4]*A[1,4,3,2]
    sizeEnvL = size(EnvL)
    EnvL = reshape(EnvL,prod(sizeEnvL[1:2]),prod(sizeEnvL[3:4]))
    RltL = svd(EnvL)
    chitemp = sum(RltL.S/maximum(RltL.S) .> 1.0e-14)
    chikept = min(chitemp,chimax,prod(sizeEnvL[1:2]))
    wLdn = reshape(RltL.U[:,1:chikept],sizeEnvL[1],sizeEnvL[2],chikept)
 



    @tensor EnvR[:] := A[3,2,1,5]*A[3,2,1,4]*Aimp[6,5,-2,-1]*Aimp[6,4,-4,-3]
    sizeEnvR = size(EnvR)
    EnvR = reshape(EnvR,prod(sizeEnvR[1:2]),prod(sizeEnvR[3:4]))
    sizeEnLR = size(EnvR)
    EnvR = reshape(EnvR,prod(sizeEnvR[1:2]),prod(sizeEnvR[3:4]))
    RltR = svd(EnvR)
    chitemp = sum(RltR.S/maximum(RltR.S) .> 1.0e-14)
    chikept = min(chitemp,chimax,prod(sizeEnvR[1:2]))
    wRdn = reshape(RltR.U[:,1:chikept],sizeEnvR[1],sizeEnvR[2],chikept)
    
    return wLdn,wRdn
end 


function compute_projector(A,wLup,wRup,wLdn,wRdn,chimax)


    @tensor sT_left[:] := A[2,1,5,-3]*A[5,3,4,-4]*wLup[1,2,-1]*wLdn[3,4,-2]
    @tensor sT_right[:] := A[1,-3,5,2]*A[5,-4,4,3]*wRup[2,1,-1]*wRdn[3,4,-2]
    @tensor wLwR_up[:] := wLup[1,4,-2]*wRup[1,3,-1]*wLup[2,4,-4]*wRup[2,3,-3] 
    @tensor wLwR_dn[:] := wLdn[1,3,-2]*wRdn[1,4,-1]*wLdn[2,3,-4]*wRdn[2,4,-3]


    # compute isometry y
    @tensor Env_y[:] := sT_left[3,5,1,2]*sT_left[4,6,1,2]*sT_right[7,8,-2,-1]*sT_right[9,10,-4,-3]*wLwR_up[7,3,9,4]*wLwR_dn[8,5,10,6]
    sizeEnv_y = size(Env_y)
    Env_y = reshape(Env_y,prod(sizeEnv_y[1:2]),prod(sizeEnv_y[1:2]))
    Rlt = svd(Env_y)

    chitemp = sum(Rlt.S/maximum(Rlt.S) .> 1.0e-14)
    chikept = min(prod(sizeEnv_y[1:2]),chimax,chitemp)

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
    chikept = min(prod(sizeEnv_w[1:2]),chimax,chitemp)

    w = reshape(Rlt.U[:,1:chikept],sizeEnv_w[1],sizeEnv_w[2],chikept)
    


    println("chikept: ",chikept)
    println(norm(tr(Env_w)-tr(Env_w*Rlt.U[:,1:chikept]*(Rlt.U[:,1:chikept])'))/norm(tr(Env_w)))
    

    #=
    @tensor A_new[:] := sT_right[6,13,1,2]*sT_left[14,12,7,8]*wRup[3,4,6]*wLup[3,5,14]*wRdn[9,10,13]*wLdn[9,11,12]*y[2,1,-2]*y[8,7,-4]*w[10,11,-3]*w[4,5,-1]
    #
    @tensor sTpos_left[:] := Ta[2,1,5,-3]*Apos[5,3,4,-4]*sLup[1,2,-1]*sLdn[3,4,-2]
    @tensor Apos_new[:] := sT_right[6,13,1,2]*sTpos_left[14,12,7,8]*wRup[3,4,6]*wLup[3,5,14]*wRdn[9,10,13]*wLdn[9,11,12]*y[2,1,-2]*y[8,7,-4]*w[10,11,-3]*w[4,5,-1]
    return Ta_new,Apos_new,y,w
    =#

    #=
    @tensor sTpos_left[:] := Apos[2,1,5,-3]*Ta[5,3,4,-4]*sLup[1,2,-1]*sLdn[3,4,-2]+Ta[2,1,5,-3]*Apos[5,3,4,-4]*sLup[1,2,-1]*sLdn[3,4,-2]
    @tensor sTpos_right[:] := Apos[1,-3,5,2]*Tb[5,-4,4,3]*sRup[2,1,-1]*sRdn[3,4,-2]+Tb[1,-3,5,2]*Apos[5,-4,4,3]*sRup[2,1,-1]*sRdn[3,4,-2]
    @tensor Apos_new[:] := sT_right[6,13,1,2]*sTpos_left[14,12,7,8]*wRup[3,4,6]*wLup[3,5,14]*wRdn[9,10,13]*wLdn[9,11,12]*y[2,1,-2]*y[8,7,-4]*w[10,11,-3]*w[4,5,-1]+
                     sTpos_right[6,13,1,2]*sT_left[14,12,7,8]*wRup[3,4,6]*wLup[3,5,14]*wRdn[9,10,13]*wLdn[9,11,12]*y[2,1,-2]*y[8,7,-4]*w[10,11,-3]*w[4,5,-1]
    =#
    return y,w

end