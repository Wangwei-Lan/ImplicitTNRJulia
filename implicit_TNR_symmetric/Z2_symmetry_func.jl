#! Z2 symmetric svd, QR decomposition and compute isometries
#? 
#* Not quite sure how things work out 

"""
    svd_Z2(M,Qnum_in,Qnum_2)

take svd decomposition of input matrix M
"""
function svd_Z2(M,Qnum_in,Qnum_out)

    sizeM = size(M)
    Qnum_S = size(Qnum_in,1) < size(Qnum_out,1) ? Qnum_in : Qnum_out

    # odd index
    odd_in = findall(x->x==1,Qnum_in);odd_out = findall(x->x==1,Qnum_out)
    # even index
    even_in = findall(x->x==0,Qnum_in);even_out = findall(x->x==0,Qnum_out)
    # odd and even index for S
    odd_S = findall(x->x==1,Qnum_S); even_S = findall(x->x==0,Qnum_S);

    # initial results
    U = zeros(sizeM[1],min(sizeM...))
    S = zeros(min(sizeM...))
    V = zeros(sizeM[2],min(sizeM...))
    
    # odd part  
    M_odd = M[odd_in,odd_out];
    Rlt_odd = svd(M_odd)   

    U[odd_in,odd_S] = Rlt_odd.U
    S[odd_S] = Rlt_odd.S
    V[odd_out,odd_S] = Rlt_odd.V

    # even part
    M_even = M[even_in,even_out]; 
    Rlt_even = svd(M_even)

    U[even_in,even_S] = Rlt_even.U
    S[even_S] = Rlt_even.S
    V[even_out,even_S] = Rlt_even.V

    return U,S,V

end


"""
    QR_Z2(M,Qnum_in,Qnum_out)

QR decomposition for Z2 symmetric Matrix
"""
function QR_Z2(M,Qnum_in,Qnum_out)

    sizeM = size(M)
    Qnum_R = size(Qnum_in,1) < size(Qnum_out,1) ? Qnum_in : Qnum_out

    # odd index
    odd_in = findall(x->x==1,Qnum_in);odd_out = findall(x->x==1,Qnum_out)
    # even index
    even_in = findall(x->x==0,Qnum_in);even_out = findall(x->x==0,Qnum_out)
    # odd and even index for S
    odd_R = findall(x->x==1,Qnum_R); even_R = findall(x->x==0,Qnum_R);

    # create new 
    Q = zeros(size(Qnum_in,1),size(Qnum_R,1))    
    R = zeros(size(Qnum_R,1),size(Qnum_R,1))

    M_odd = M[odd_in,odd_out];
    Rlt_odd = qr(M_odd)
    
    M_even = M[even_in,even_out]; 
    Rlt_even = qr(M_even)

    Q[odd_in,odd_R] = Rlt_odd.Q[:,1:min(size(odd_R,1),size(odd_out,1))]
    Q[even_in,even_R] = Rlt_even.Q[:,1:min(size(even_R,1),size(even_out,1))]

    R[odd_R,odd_R] = Rlt_odd.R
    R[even_R,even_R] = Rlt_even.R

    return Q,R
end

"""
    compute_isometry_Z2(M,qnum_in,qnum_out)

# compute isometry to truncate bond dimension
# now, only works for symmetric matrix; Nov,19,2020
""" 
function compute_isometry_Z2(M,qnum_in,qnum_out,chimax;tol=1.0e-16)

    # odd index
    odd_in = findall(x->x==1,qnum_in);odd_out = findall(x->x==1,qnum_out)
    # even index
    even_in = findall(x->x==0,qnum_in);even_out = findall(x->x==0,qnum_out)

    # odd part
    M_odd = M[odd_in,odd_out];M_odd = M_odd#/maximum(M_odd)
    Rlt_odd = svd(M_odd)    
    chitemp = sum(Rlt_odd.S/maximum(Rlt_odd.S) .> 1.0e-14)  
    chi_odd = min(chitemp,chimax)
    U_odd_temp = Rlt_odd.U[:,1:chi_odd]

    # even part
    M_even = M[even_in,even_out]; M_even = M_even#/maximum(M_even)
    Rlt_even = svd(M_even)
    chitemp = sum(Rlt_even.S/maximum(Rlt_even.S) .> 1.0e-14)
    chi_even = min(chitemp,chimax)
    U_even_temp = Rlt_even.U[:,1:chi_even]    


    chikept = chi_odd+chi_even
    U = zeros(size(qnum_in,1),chikept)
    U[even_in,1:chi_even] = U_even_temp
    U[odd_in,chi_even+1:chi_even+chi_odd] = U_odd_temp

    qnum_new = zeros(chikept) 
    qnum_new[chi_even+1:chi_odd+chi_even] .= 1

    return U,qnum_new,Rlt_odd.S,Rlt_even.S 
end


"""
    compute_projector_Z2(A,v,chimax)

compute projector y and w to renormalize the network
"""
function compute_projector_Z2(A,Qnum,v,chimax)

    Qnum_new = Array{Array{Float64}}(undef,4)

    @tensor vAvA[:] := v[1,2,-1]*A[2,1,5,-3]*v[3,4,-2]*A[4,3,5,-4]
    @tensor v4[:] := v[1,4,-2]*v[1,3,-1]*v[2,4,-4]*v[2,3,-3]
    @tensor Envy[:] := vAvA[7,8,-2,-1]*vAvA[3,5,1,2]*vAvA[4,6,1,2]*vAvA[9,10,-4,-3]*v4[7,3,9,4]*v4[8,5,10,6]
    sizeEnvy = size(Envy)
    Envy = reshape(Envy,prod(sizeEnvy[1:2]),prod(sizeEnvy[1:2]))
    qnumtemp = Merge(Qnum[2],Qnum[2])
    y,qnum_new = compute_isometry_Z2(Envy,qnumtemp,qnumtemp,chimax)  
    Qnum_new[2] = qnum_new;Qnum_new[4] = qnum_new
    y = reshape(y,size(Qnum[2],1),size(Qnum[2],1),size(y,2))


    @tensor Envw[:] := vAvA[1,2,3,4]*vAvA[25,12,3,4]*vAvA[26,19,17,18]*vAvA[14,13,17,18]*
                    vAvA[1,2,6,5]*vAvA[22,11,6,5]*vAvA[23,20,16,15]*vAvA[14,13,16,15]*
                    v[24,-1,25]*v[24,-2,26]*v[9,7,12]*v[9,8,19]*v[10,7,11]*v[10,8,20]*v[21,-3,22]*v[21,-4,23]
    sizeEnvw = size(Envw)
    Envw = reshape(Envw,prod(sizeEnvw[1:2]),prod(sizeEnvw[1:2]))
    qnumtemp = Merge(Qnum[1],Qnum[1])
    w,qnum_new = compute_isometry_Z2(Envw,qnumtemp,qnumtemp,chimax)
    Qnum_new[1] = qnum_new;Qnum_new[3] = qnum_new
    w = reshape(w,size(Qnum[1],1),size(Qnum[1],1),size(w,2))

    return y,w,Qnum_new
end


function compute_projector_TNR_Z2(A,Qnum,vL,Qnum_vL,vR,Qnum_vR,sL,sR,chimax)

    Qnum_new = Array{Array{Float64,1},1}(undef,4)

    #
    #   compute y
    #
    @tensor sLAsLA[:] := sL[1,2,-1]*A[2,1,5,-3]*sL[3,4,-2]*A[4,3,5,-4]
    @tensor sRAsRA[:] := sR[2,1,-1]*A[1,-3,5,2]*sR[3,4,-2]*A[4,-4,5,3]
    @tensor v4[:] := vL[1,4,-2]*vR[1,3,-1]*vL[2,4,-4]*vR[2,3,-3]
    
    #@tensor Envy[:] := sRAsRA[7,8,-2,-1]*sLAsLA[3,5,1,2]*sLAsLA[4,6,1,2]*sRAsRA[9,10,-4,-3]*v4[7,3,9,4]*v4[8,5,10,6]
    #
    @tensor Envy[:] := sLAsLA[3,5,1,2]*sLAsLA[4,6,1,2]*sRAsRA[7,8,13,14]*sRAsRA[9,10,11,12]*
                        sLAsLA[15,17,13,14]*sLAsLA[16,18,11,12]*sRAsRA[21,22,-1,-2]*
                        sRAsRA[19,20,-3,-4]*v4[7,3,9,4]*v4[8,5,10,6]*v4[21,15,19,16]*v4[22,17,20,18]
    #
    sizeEnvy = size(Envy)
    Envy = reshape(Envy,prod(sizeEnvy[1:2]),prod(sizeEnvy[1:2]))
    qnum_y_in = Merge(Qnum[2],Qnum[2]);qnum_y_out = Merge(Qnum[2],Qnum[2])
    y,qnum_new,S1,S2 = compute_isometry_Z2(Envy,qnum_y_in,qnum_y_out,chimax)
    Qnum_new[2] = deepcopy(qnum_new);Qnum_new[4] = deepcopy(qnum_new)
    chitemp = size(y,2)
    println("Truncation error for y: ",(tr(Envy)-tr(Envy*y*y'))/tr(Envy))
    #
    #    compute w
    # 
    @tensor Envw[:] := sLAsLA[1,2,3,4]*sRAsRA[25,12,3,4]*sLAsLA[26,19,17,18]*sRAsRA[14,13,17,18]*
                    sLAsLA[1,2,6,5]*sRAsRA[22,11,6,5]*sLAsLA[23,20,16,15]*sRAsRA[14,13,16,15]*
                    vR[24,-1,25]*vL[24,-2,26]*vR[9,7,12]*vL[9,8,19]*vR[10,7,11]*vL[10,8,20]*vR[21,-3,22]*vL[21,-4,23]
    sizeEnvw = size(Envw)
    Envw = reshape(Envw,prod(sizeEnvw[1:2]),prod(sizeEnvw[1:2]))    
    qnum_w_in = Merge(Qnum_vR[2],Qnum_vL[2]);qnum_w_out = Merge(Qnum_vR[2],Qnum_vL[2])
    w,qnum_new,S3,S4 = compute_isometry_Z2(Envw,qnum_w_in,qnum_w_out,chimax)
    Qnum_new[1] = deepcopy(qnum_new);Qnum_new[3] = deepcopy(qnum_new)
    chitemp = size(w,2) 
    println("Truncation error for w: ",(tr(Envw)-tr(Envw*w*w'))/tr(Envw))
    return y,w,Qnum_new,S1,S2,S3,S4
end



function compute_projector_TNR_Z2_modified(A,Qnum,v,Qnum_v,s,chimax)


    Qnum_new = Array{Array{Float64,1},1}(undef,4)

    #
    #   compute y
    #
    @tensor sLAsLA[:] := s[1,2,-1]*A[2,1,5,-3]*s[3,4,-2]*A[4,3,5,-4]
    #@tensor sRAsRA[:] := sR[2,1,-1]*A[1,-3,5,2]*sR[3,4,-2]*A[4,-4,5,3]
    sRAsRA = deepcopy(sLAsLA)
    @tensor v4[:] := v[1,4,-2]*v[1,3,-1]*v[2,4,-4]*v[2,3,-3]
    
    #@tensor Envy[:] := sRAsRA[7,8,-2,-1]*sLAsLA[3,5,1,2]*sLAsLA[4,6,1,2]*sRAsRA[9,10,-4,-3]*v4[7,3,9,4]*v4[8,5,10,6]
    #
    @tensor Envy[:] := sLAsLA[3,5,1,2]*sLAsLA[4,6,1,2]*sRAsRA[7,8,13,14]*sRAsRA[9,10,11,12]*
                        sLAsLA[15,17,13,14]*sLAsLA[16,18,11,12]*sRAsRA[21,22,-1,-2]*
                        sRAsRA[19,20,-3,-4]*v4[7,3,9,4]*v4[8,5,10,6]*v4[21,15,19,16]*v4[22,17,20,18]
    #
    sizeEnvy = size(Envy)
    Envy = reshape(Envy,prod(sizeEnvy[1:2]),prod(sizeEnvy[1:2]))
    qnum_y_in = Merge(Qnum[2],Qnum[2]);qnum_y_out = Merge(Qnum[2],Qnum[2])
    y,qnum_new,S1,S2 = compute_isometry_Z2(Envy,qnum_y_in,qnum_y_out,chimax)
    Qnum_new[2] = deepcopy(qnum_new);Qnum_new[4] = deepcopy(qnum_new)
    chitemp = size(y,2)
    println("Truncation error for y: ",(tr(Envy)-tr(Envy*y*y'))/tr(Envy))
    println("Truncation error for y: ",(tr(Envy)-tr(Envy*y[:,1:chitemp-1]*y'[1:chitemp-1,:]))/tr(Envy))
    #
    #    compute w
    # 
    @tensor Envw[:] := sLAsLA[1,2,3,4]*sRAsRA[25,12,3,4]*sLAsLA[26,19,17,18]*sRAsRA[14,13,17,18]*
                    sLAsLA[1,2,6,5]*sRAsRA[22,11,6,5]*sLAsLA[23,20,16,15]*sRAsRA[14,13,16,15]*
                    v[24,-1,25]*v[24,-2,26]*v[9,7,12]*v[9,8,19]*v[10,7,11]*v[10,8,20]*v[21,-3,22]*v[21,-4,23]
    sizeEnvw = size(Envw)
    Envw = reshape(Envw,prod(sizeEnvw[1:2]),prod(sizeEnvw[1:2]))    
    qnum_w_in = Merge(Qnum_v[2],Qnum_v[2]);qnum_w_out = Merge(Qnum_v[2],Qnum_v[2])
    w,qnum_new,S3,S4 = compute_isometry_Z2(Envw,qnum_w_in,qnum_w_out,chimax)
    Qnum_new[1] = deepcopy(qnum_new);Qnum_new[3] = deepcopy(qnum_new)
    chitemp = size(w,2) 
    println("Truncation error for w: ",(tr(Envw)-tr(Envw*w*w'))/tr(Envw))
    println("Truncation error for w: ",(tr(Envw)-tr(Envw*w[:,1:chitemp-1]*w'[1:chitemp-1,:]))/tr(Envw))
    return y,w,Qnum_new,S1,S2,S3,S4


end


"""
    Merge(qnum1,qnum2)
    
Merge two quantum number together; loop over qnum2, append to new quantum number
"""
function Merge(qnum1,qnum2)

    size1 = size(qnum1,1)
    size2 = size(qnum2,1)
    qnum = zeros(size1*size2)
    # loop over second quanum number; 
    for i in 1:size2
        qnumtemp = (qnum2[i] .+ qnum1).%2
        qnum[(i-1)*size1+1:i*size1] = qnumtemp
    end

    return qnum
end





"""
    update_vM_Z2(M,qnum_in,qnum_out) 
   
    M = U*S*V' 
    update isometry through vM = U*V' 
"""
function update_vM_Z2(M,qnum_in,qnum_out)

    # odd quantum number
    odd_in = findall(x->x==1,qnum_in);odd_out = findall(x->x==1,qnum_out)

    # even quantum number 
    even_in = findall(x->x==0,qnum_in);even_out = findall(x->x==0,qnum_out)

    # odd matrix 
    M_odd = M[odd_in,odd_out]; M_odd = M_odd/maximum(M_odd)
    Rlt_odd = svd(M_odd)            # svd for odd part    
    #println((Rlt_odd.S/maximum(Rlt_odd.S))[end])

    P_odd = Rlt_odd.U*Rlt_odd.V'    # projector for odd part


    # even matrix
    M_even = M[even_in,even_out];M_even = M_even/maximum(M_even)
    Rlt_even = svd(M_even)
    P_even = Rlt_even.U*Rlt_even.V'

    P = zeros(size(qnum_in,1),size(qnum_out,1))
    P[odd_in,odd_out] = P_odd
    P[even_in,even_out] = P_even

    return P
end




function update_sM_Z2(PM,BM,PM_qnum_in,PM_qnum_out,BM_qnum_in,BM_qnum_out;tol=1.0e-14)

    # define size of PM
    sizePM = size(PM)

    # odd index position for P
    PM_odd_in = findall(x->x==1,PM_qnum_in);PM_odd_out = findall(x->x==1,PM_qnum_out)

    # even index position for P
    PM_even_in = findall(x->x==0,PM_qnum_in);PM_even_out = findall(x->x==0,PM_qnum_out)

    # odd index position for B
    BM_odd_in = findall(x->x==1,BM_qnum_in);BM_odd_out = findall(x->x==1,BM_qnum_out)

    # even index position for B
    BM_even_in = findall(x->x==0,BM_qnum_in);BM_even_out = findall(x->x==0,BM_qnum_out)

    # create Snew
    Snew = zeros(sizePM)
    #
    # update odd section
    PM_odd = PM[PM_odd_in,PM_odd_out]
    BM_odd = BM[BM_odd_in,BM_odd_out]
    #Snew_odd = BM_odd\PM_odd
    #Snew_odd = pinv(BM_odd,rtol=sqrt(eps(real(float(one(eltype(M)))))))*PM_odd
    Snew_odd = pinv(BM_odd,rtol=tol)*PM_odd
    #Snew_odd = pinv(BM_odd)*PM_odd

    # update even section
    PM_even = PM[PM_even_in,PM_even_out]
    BM_even = BM[BM_even_in,BM_even_out]
    #Snew_even = BM_even\PM_even    
    #Snew_even = pinv(BM_even,rtol=sqrt(eps(real(float(one(eltype(M)))))))*PM_even
    Snew_even = pinv(BM_even,rtol=tol)*PM_even
    #Snew_even = pinv(BM_even)*PM_even
    # update s
    Snew[PM_odd_in,PM_odd_out] = Snew_odd
    Snew[PM_even_in,PM_even_out] = Snew_even
    #
    #=
    Snew = pinv(BM,rtol=sqrt(eps(real(float(one(eltype(M)))))))*PM
    Snew = Snew .* (abs.(Snew) .>1.0e-14)
    =#

    return Snew
end



function compute_fidelity(A,vL,vR,sL,sR,dis;printdetail=false)

    @tensor phiphi[:] := A[1,2,3,7]*A[1,2,3,8]*A[6,7,4,5]*A[6,8,4,5]

    @tensor psiphi[:] :=  A[2,1,5,12]*A[10,12,13,11]*A[6,4,5,8]*A[9,8,13,15]*vL[4,7,3]*vR[15,14,16]*
    sL[1,2,3]*sR[11,10,16]*dis[6,9,7,14]

    @tensor psipsi[:] :=  A[4,3,6,13]*A[9,13,11,10]*A[2,1,6,14]*A[8,14,11,7]*sL[3,4,5]*sL[1,2,5]*sR[10,9,12]*sR[7,8,12]

    if printdetail ==true
        @printf " err: %.16e "  (psipsi[1]+phiphi[1]-2*psiphi[1])/phiphi[1]
    end
    return (psipsi[1]+phiphi[1]-2*psiphi[1])/phiphi[1]

end


function compute_fidelity_imp(A,vL,vR,sL,sR,dis;printdetail=false)

    @tensor phiphi[:] := A[1,2,3,7]*A[1,2,3,8]*A[6,7,4,5]*A[6,8,4,5]

    @tensor psiphi[:] :=  A[2,1,5,12]*A[10,12,13,11]*A[6,4,5,8]*A[9,8,13,15]*vL[4,7,3]*vR[15,14,16]*
    sL[1,2,3]*sR[11,10,16]*dis[6,9,7,14]

    @tensor psipsi[:] := A[14,13,18,19]*A[4,19,8,5]*A[16,15,18,20]*A[7,20,8,6]*sL[13,14,12]*sL[15,16,17]*
                vL[11,10,12]*vL[11,10,17]*sR[5,4,3]*sR[6,7,9]*vR[2,1,3]*vR[2,1,9]


    if printdetail ==true
        @printf " err: %.16e  %.16e %.16e %.16e" phiphi[1] psiphi[1] psipsi[1] (psipsi[1]+phiphi[1]-2*psiphi[1])/phiphi[1]
    end
    return (psipsi[1]+phiphi[1]-2*psiphi[1])/phiphi[1]




end





function compute_fidelity_modified(A,wL,wR,vL,vR,sL,sR,dis;printdetail=false)
    @tensor sL[:] := wL[-1,-2,1]*sL[1,-3]
    @tensor sR[:] := wR[-1,-2,1]*sR[1,-3]

    @tensor phiphi[:] := A[1,2,3,7]*A[1,2,3,8]*A[6,7,4,5]*A[6,8,4,5]
    @tensor psiphi[:] := A[2,1,5,12]*A[10,12,13,11]*A[6,4,5,8]*A[9,8,13,15]*vL[4,7,3]*vR[15,14,16]*
                        sL[1,2,3]*sR[11,10,16]*dis[6,9,7,14]
    @tensor psipsi[:] := A[4,3,6,13]*A[9,13,11,10]*A[2,1,6,14]*A[8,14,11,7]*sL[3,4,5]*sL[1,2,5]*sR[10,9,12]*sR[7,8,12]

    if printdetail ==true
        @printf " err: %.16e  " (psipsi[1]+phiphi[1]-2*psiphi[1])/phiphi[1]
    end
    return (psipsi[1]+phiphi[1]-2*psiphi[1])/phiphi[1]
end


function compute_fidelity_modified_large_env(A,wL,wR,vL,vR,sL,sR,dis;printdetail=false)

    @tensor sL[:] := wL[-1,-2,1]*sL[1,-3]
    @tensor sR[:] := wR[-1,-2,1]*sR[1,-3]

    @tensor phiphi[:] := A[8,7,12,11]*A[9,11,13,10]*A[2,1,12,5]*A[3,5,13,4]*A[15,14,20,18]*A[16,18,21,17]*
                            A[2,1,20,6]*A[3,6,21,4]*wL[7,8,19]*wL[14,15,19]*wR[10,9,22]*wR[17,16,22]
    @tensor psipsi[:] :=  A[8,7,12,11]*A[9,11,13,10]*A[2,1,12,5]*A[3,5,13,4]*A[15,14,20,18]*A[16,18,21,17]*
                    A[2,1,20,6]*A[3,6,21,4]*sL[7,8,19]*sL[14,15,19]*sR[10,9,22]*sR[17,16,22]
    @tensor psiphi[:] :=  A[23,22,27,26]*A[24,26,28,25]*A[15,14,27,18]*A[16,18,28,17]*A[15,14,20,19]*A[16,19,21,17]*
                    A[8,7,20,11]*A[9,11,21,10]*wL[7,8,12]*wL[1,2,12]*wR[10,9,13]*wR[4,6,13]*vL[1,3,29]*vR[4,5,30]*
                    sL[22,23,29]*sR[25,24,30]*dis[2,6,3,5]

    if printdetail ==true
        @printf " err: %.16e  " (psipsi[1]+phiphi[1]-2*psiphi[1])/phiphi[1]
    end
    return (psipsi[1]+phiphi[1]-2*psiphi[1])/phiphi[1]

end








function compute_fidelity_impurity_modified(A,A_single,wL,wR,wL_single,wR_single,vL,vR,vL_single,vR_single,sL,sR,sL_single,sR_single,dis;printdetail=false)


    @tensor sL[:] := wL[-1,-2,1]*sL[1,-3]
    @tensor sR[:] := wR[-1,-2,1]*sR[1,-3]
    @tensor sL_single[:] := wL_single[-1,-2,1]*sL_single[1,-3]
    @tensor sR_single[:] := wR_single[-1,-2,1]*sR_single[1,-3]

    @tensor phiphi_L[:] := A_single[1,2,3,7]*A_single[1,2,3,8]*A[6,7,4,5]*A[6,8,4,5]
    @tensor psiphi_L[:] := A_single[2,1,5,12]*A[10,12,13,11]*A_single[6,4,5,8]*A[9,8,13,15]*vL_single[4,7,3]*vR[15,14,16]*
                        sL_single[1,2,3]*sR[11,10,16]*dis[6,9,7,14]
    @tensor psipsi_L[:] := A_single[4,3,6,13]*A[9,13,11,10]*A_single[2,1,6,14]*A[8,14,11,7]*sL_single[3,4,5]*sL_single[1,2,5]*sR[10,9,12]*sR[7,8,12]



    @tensor phiphi_R[:] := A[1,2,3,7]*A[1,2,3,8]*A_single[6,7,4,5]*A_single[6,8,4,5]
    @tensor psiphi_R[:] := A[2,1,5,12]*A_single[10,12,13,11]*A[6,4,5,8]*A_single[9,8,13,15]*vL[4,7,3]*vR_single[15,14,16]*
                        sL[1,2,3]*sR_single[11,10,16]*dis[6,9,7,14]
    @tensor psipsi_R[:] := A[4,3,6,13]*A_single[9,13,11,10]*A[2,1,6,14]*A_single[8,14,11,7]*sL[3,4,5]*sL[1,2,5]*sR_single[10,9,12]*sR_single[7,8,12]


    if printdetail ==true
        @printf " err: %.16e  ;    %.16e \n" (psipsi_L[1]+phiphi_L[1]-2*psiphi_L[1])/phiphi_L[1]   (psipsi_R[1]+phiphi_R[1]-2*psiphi_R[1])/phiphi_R[1]
    end
    return (psipsi_L[1]+phiphi_L[1]-2*psiphi_L[1])/phiphi_L[1],(psipsi_R[1]+phiphi_R[1]-2*psiphi_R[1])/phiphi_R[1]
end




function check_gauge(A,Qnum)
    sizeA = size(A);
    x = Matrix(1.0I,sizeA[2],sizeA[2])
    qnum = Qnum[2]
    @tensor phiphi[:] := A[1,2,3,4]*A[1,2,3,4]
    @tensor psiphi[:] := A[6,1,4,2]*A[4,3,6,5]*x[3,1]*x[5,2]
    println("iteration 0: ",(phiphi[1]-psiphi[1])/phiphi[1])
    for j in 1:100
        @tensor Env[:] := A[4,-2,2,1]*A[2,-1,4,3]*x[3,1]
        x = update_vM_Z2(Env,qnum,qnum)
        
        if j%100 ==0
            @tensor phiphi[:] := A[1,2,3,4]*A[1,2,3,4]
            @tensor psiphi[:] := A[6,1,4,2]*A[4,3,6,5]*x[3,1]*x[5,2]
            println("iteration $j: ",(phiphi[1]-psiphi[1])/phiphi[1])
        end
    
    end
    

end




"""
    without single impurity
"""
function check_fidelity(A,A_single,vL,vR,wL,wR,sL,sR,dis;printdetail=false)

    @tensor sL[:] := wL[-1,-2,1]*sL[1,-3]
    @tensor sR[:] := wR[-1,-2,1]*sR[1,-3]
 
    @tensor phiphi[:] := A[1,2,3,7]*A[1,2,3,8]*A_single[6,7,4,5]*A_single[6,8,4,5]
    @tensor psiphi[:] := A[2,1,6,10]*A[4,3,6,11]*A_single[9,10,7,8]*A_single[9,11,7,8]*sL[1,2,5]*sL[3,4,5]
    @tensor psipsi[:] := A[6,5,12,7]*A[11,10,12,13]*A_single[3,7,1,2]*A_single[4,13,1,2]*sL[5,6,8]*vL[10,9,8]*dis[11,4,9,3]

    if printdetail ==true
        @printf " err: %.16e  \n" (psipsi[1]+phiphi[1]-2*psiphi[1])/phiphi[1]
    end
    return (psipsi[1]+phiphi[1]-2*psiphi[1])/phiphi[1]
end




function check_replace(A,A_single,vL,vR,wL,wR,sL,sR,vL_single,vR_single,wL_single,wR_single,sL_single,sR_single,dis;printdetail=false)


    @tensor sL[:] := wL[-1,-2,1]*sL[1,-3]
    @tensor sR[:] := wR[-1,-2,1]*sR[1,-3]
    @tensor sL_single[:] := wL_single[-1,-2,1]*sL_single[1,-3]
    @tensor sR_single[:] := wR_single[-1,-2,1]*sR_single[1,-3]


    @tensor phiphi[:] := A[9,1,2,5]*A[9,7,8,13]*A[14,13,12,11]*A_single[14,5,3,4]*A[10,1,2,6]*A[10,7,8,15]*A[16,15,12,11]*A_single[16,6,3,4]
    @tensor psiphi[:] := A[22,16,17,21]*A[22,5,4,11]*A[23,11,10,9]*A_single[23,21,18,19]*A[14,13,17,20]*A_single[25,20,18,19]*A[2,1,4,12]*A[6,12,10,7]*
                        sL[13,14,15]*vL[16,24,15]*sL[1,2,3]*vL[5,24,3]*sR[7,6,8]*vR[9,25,8]
    @tensor psipsi[:] := A[6,5,9,25]*A_single[23,25,20,19]*A[13,12,16,28]*A[27,28,33,26]*A[8,7,9,24]*A_single[22,24,20,19]*A[15,14,16,34]*A[31,34,33,30]*
                        sL[14,15,17]*vL[3,2,17]*sL[7,8,11]*vL[4,2,11]*sL[5,6,10]*vL[4,1,10]*sL[12,13,18]*vL[3,1,18]*sR[30,31,32]*vR[21,22,32]*sR[26,27,29]*vR[21,23,29]


    @tensor psiphi_1[:] := A[27,16,17,23]*A_single[28,23,21,22]*A[27,5,4,12]*A[28,12,10,9]*A[14,13,17,24]*A[2,1,4,11]*A[6,11,10,7]*A_single[18,24,21,19]*
                        sL[13,14,15]*vL[16,25,15]*sL[1,2,3]*vL[5,25,3]*sR[7,6,8]*vR[9,26,8]*sR_single[19,18,20]*vR_single[22,26,20]
                        #
    @tensor psipsi_1[:] := A[6,5,9,40]*A[15,14,16,39]*A[31,39,37,30]*A_single[19,40,23,20]*A[8,7,9,41]*A[13,12,16,42]*A[34,42,37,35]*A_single[21,41,23,22]*
                        sL[7,8,10]*vL[4,2,10]*sL[12,13,18]*vL[3,2,18]*sR[35,34,38]*vR[27,25,38]*sR_single[22,21,29]*vR_single[26,25,29]*sR[30,31,33]*vR[27,24,33]*
                        sR_single[20,19,28]*vR_single[26,24,28]*sL[5,6,11]*vL[4,1,11]*sL[14,15,17]*vL[3,1,17]
                        
    if printdetail == true
        @printf " err: %.16e  %.16e  %.16e %.16e %.16e \n" phiphi[1] psiphi[1] psipsi[1]  1-psiphi[1]^2/psipsi[1]/phiphi[1]  (phiphi[1]+psipsi[1]-2*psiphi[1])/phiphi[1]
        @printf " err: %.16e  %.16e  %.16e %.16e %.16e " phiphi[1] psiphi_1[1] psipsi_1[1]  1-psiphi_1[1]^2/psipsi_1[1]/phiphi[1]  (phiphi[1]+psipsi_1[1]-2*psiphi_1[1])/phiphi[1]
    end

end


function linear_update(PM,BM,S,PM_qnum_in,PM_qnum_out,BM_qnum_in,BM_qnum_out;type="none")

    # define size of PM
    sizePM = size(PM)

    BM = 0.5*(BM+BM');
    # odd index position for P
    PM_odd_in = findall(x->x==1,PM_qnum_in);PM_odd_out = findall(x->x==1,PM_qnum_out)
    # even index position for P
    PM_even_in = findall(x->x==0,PM_qnum_in);PM_even_out = findall(x->x==0,PM_qnum_out)
    # odd index position for B
    BM_odd_in = findall(x->x==1,BM_qnum_in);BM_odd_out = findall(x->x==1,BM_qnum_out)
    # even index position for B
    BM_even_in = findall(x->x==0,BM_qnum_in);BM_even_out = findall(x->x==0,BM_qnum_out)


    # create Snew
    Snew = zeros(sizePM)
    #
    # update odd section
    S_odd = S[PM_odd_in,PM_odd_out]
    PM_odd = PM[PM_odd_in,PM_odd_out]
    BM_odd = BM[BM_odd_in,BM_odd_out]
    
    
    # update even section
    S_even = S[PM_even_in,PM_even_out]
    PM_even = PM[PM_even_in,PM_even_out]
    BM_even = BM[BM_even_in,BM_even_out]
    
    sizeS_odd = size(S_odd)
    sizeS_even = size(S_even)
    if type =="v"
        Snew_odd_temp,info1 = linsolve(x-> reshape(reshape(x,sizeS_odd)*BM_odd,prod(sizeS_odd)),reshape(PM_odd,prod(sizeS_odd)),reshape(S_odd,prod(sizeS_odd)),
                         maxiter=1,issymmetric=true,isposdef=true,ishermitian=true)
        Snew_even_temp,info2 = linsolve(x->reshape(reshape(x,sizeS_even)*BM_even,prod(sizeS_even)),reshape(PM_even,prod(sizeS_even)),reshape(S_even,prod(sizeS_even)),
                         maxiter=1,issymmetric=true,isposdef=true,ishermitian=true)
    elseif type =="s"
        Snew_odd_temp,info1 = linsolve(x-> reshape(BM_odd*reshape(x,sizeS_odd),prod(sizeS_odd)),reshape(PM_odd,prod(sizeS_odd)),reshape(S_odd,prod(sizeS_odd)),
                            maxiter=1,issymmetric=true,isposdef=true,ishermitian=true)
        Snew_even_temp,info2 = linsolve(x->reshape(BM_even*reshape(x,sizeS_even),prod(sizeS_even)),reshape(PM_even,prod(sizeS_even)),reshape(S_even,prod(sizeS_even)),
                            maxiter=1,issymmetric=true,isposdef=true,ishermitian=true)
    elseif type== "none"
        Snew_odd_temp,info1 = linsolve(x-> reshape(BM_odd*reshape(x,sizeS_odd),prod(sizeS_odd)),reshape(PM_odd,prod(sizeS_odd)),reshape(S_odd,prod(sizeS_odd)),
                            maxiter=1,issymmetric=true,isposdef=true,ishermitian=true)
        Snew_even_temp,info2 = linsolve(x->reshape(BM_even*reshape(x,sizeS_even),prod(sizeS_even)),reshape(PM_even,prod(sizeS_even)),reshape(S_even,prod(sizeS_even)),
                            maxiter=1,issymmetric=true,isposdef=true,ishermitian=true)
    end
                        
    #update s
    Snew_odd = reshape(Snew_odd_temp,sizeS_odd)
    Snew_even = reshape(Snew_even_temp,sizeS_even)
    Snew[PM_odd_in,PM_odd_out] = Snew_odd
    Snew[PM_even_in,PM_even_out] = Snew_even
    return Snew,info1,info2
end




function linear_update(PM,BM1,BM2,S,PM_qnum_in,PM_qnum_out,BM1_qnum_in,BM1_qnum_out,BM2_qnum_in,BM2_qnum_out)

        # define size of PM
        sizePM = size(PM)

        BM1 = 0.5*(BM1+BM1');
        BM2 = 0.5*(BM2+BM2');
        # odd index position for P
        PM_odd_in = findall(x->x==1,PM_qnum_in);PM_odd_out = findall(x->x==1,PM_qnum_out)
        # even index position for P
        PM_even_in = findall(x->x==0,PM_qnum_in);PM_even_out = findall(x->x==0,PM_qnum_out)

        # odd index position for B
        BM1_odd_in = findall(x->x==1,BM1_qnum_in);BM1_odd_out = findall(x->x==1,BM1_qnum_out)
        # even index position for B
        BM1_even_in = findall(x->x==0,BM1_qnum_in);BM1_even_out = findall(x->x==0,BM1_qnum_out)
    
        # odd index position for B
        BM2_odd_in = findall(x->x==1,BM2_qnum_in);BM2_odd_out = findall(x->x==1,BM2_qnum_out)
        # even index position for B
        BM2_even_in = findall(x->x==0,BM2_qnum_in);BM2_even_out = findall(x->x==0,BM2_qnum_out)
   

    Snew = zeros(size(PM)...)
    # update odd section 
    S_odd = S[PM_odd_in,PM_odd_out]
    PM_odd = PM[PM_odd_in,PM_odd_out]
    BM1_odd = BM1[BM1_odd_in,BM1_odd_out]
    BM2_odd = BM2[BM2_odd_in,BM2_odd_out]
    sizeS_odd = size(S_odd)
    Snew_odd_temp,info1 = linsolve(x-> reshape(BM2_odd*reshape(x,sizeS_odd)*BM1_odd,prod(sizeS_odd)),reshape(PM_odd,prod(sizeS_odd)),reshape(S_odd,prod(sizeS_odd)),
                                maxiter=1,issymmetric=true,isposdef=true,ishermitian=true)
    Snew_odd = reshape(Snew_odd_temp,sizeS_odd)



    # update even section
    S_even = S[PM_even_in,PM_even_out]
    PM_even = PM[PM_even_in,PM_even_out]
    BM1_even = BM1[BM1_even_in,BM1_even_out]
    BM2_even = BM2[BM2_even_in,BM2_even_out]
    sizeS_even = size(S_even)

    Snew_even_temp,info2 = linsolve(x->reshape(BM2_even*reshape(x,sizeS_even)*BM1_even,prod(sizeS_even)),reshape(PM_even,prod(sizeS_even)),reshape(S_even,prod(sizeS_even)),
                            maxiter=1,issymmetric=true,isposdef=true,ishermitian=true)
    Snew_even = reshape(Snew_even_temp,sizeS_even)

    # update s
    Snew[PM_odd_in,PM_odd_out] = Snew_odd
    Snew[PM_even_in,PM_even_out] = Snew_even


    return Snew,info1,info2

end
