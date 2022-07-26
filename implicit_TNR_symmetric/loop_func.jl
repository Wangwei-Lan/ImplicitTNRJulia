"""
    compute_isometry(M,chimax)
compute isometry to truncate matrix M
"""
function compute_isometry(M,chimax)
    Rlt = svd(M)
    chitemp = sum(Rlt.S/maximum(Rlt.S) .> 1.0e-14)
    chikept = min(chimax,chitemp)
    return Rlt.U[:,1:chikept]
end



function f(B1,B2,A,x)
    #@tensor y[:] := B1[4,3,6,-3]*A[1,5,4,2]*A[-2,5,6,-1]*x[2,1,3]+B2[6,-3,3,4]*A[-2,6,5,-1]*A[2,3,5,1]*x[1,2,4]
    @tensor y[:] := B2[6,-3,3,4]*A[-2,6,5,-1]*A[2,3,5,1]*x[1,2,4]
    #@tensor y[:] := B1[4,3,6,-3]*A[1,5,4,2]*A[-2,5,6,-1]*x[2,1,3]
    return y 
end

function update_s(P1,B1,P2,B2,A,S,chimax,iter)

    sizeP1 = size(P1); sizeB1 = size(B1) 
    sizeP2 = size(P2); sizeB2 = size(B2)
    B1 = (B1 + permutedims(B1,[3,4,1,2]))*0.5
    B2 = (B2 + permutedims(B2,[3,4,1,2]))*0.5
    iter <= 5 ? maxiteration = 300 : maxiteration = 10
    #snew,info = linsolve(x->f(B1,B2,A,x),P1+P2,S,issymmetric=true,ishermitian=true,isposdef=true,maxiter=maxiteration)
    snew,info = linsolve(x->f(B1,B2,A,x),P2,S,issymmetric=true,ishermitian=true,isposdef=true,maxiter=maxiteration)
    #snew,info = linsolve(x->f(B1,B2,A,x),P1,S,issymmetric=true,ishermitian=true,isposdef=true,maxiter=maxiteration)
    return snew
end




function g(B1,B2,x)
    #@tensor y[:] := B1[-1,-3,2,1]*x[2,-2,1]+ B2[1,2,-2,-3]*x[-1,1,2]
    @tensor y[:] := B2[1,2,-2,-3]*x[-1,1,2]
    #@tensor y[:] :=  B1[-1,-3,2,1]*x[2,-2,1]
    return y  
end

function update_v(P1,B1,P2,B2,V,chimax,iter)
    sizeP1 = size(P1) ; sizeB1 = size(B1)
    sizeP2 = size(P2) ; sizeB2 = size(B2)

    B1 = 0.5*(B1 + permutedims(B1,[3,4,1,2]))
    B2 = 0.5*(B2 + permutedims(B2,[3,4,1,2]))
    iter <= 5 ? maxiteration = 300 : maxiteration = 10
    #vnew,info = linsolve(x->g(B1,B2,x),P1+P2,V,issymmetric=true,ishermitian=true,isposdef=true,maxiter=maxiteration)
    vnew,info = linsolve(x->g(B1,B2,x),P2,V,issymmetric=true,ishermitian=true,isposdef=true,maxiter=maxiteration)
    #vnew,info = linsolve(x->g(B1,B2,x),P1,V,issymmetric=true,ishermitian=true,isposdef=true,maxiter=maxiteration)
    return vnew
end



function compute_fidelity(A,s,v,phiphi_1,phiphi_2;printdetail=false)

    #@tensor phiphi_1[:] := A[3,4,5,13]*A[10,9,11,13]*A[2,1,5,14]*A[7,8,11,14]*A[2,1,6,15]*
    #            A[7,8,12,15]*A[3,4,6,16]*A[10,9,12,16]
    @tensor psiphi_1[:] :=  A[16,17,23,25]*A[10,9,11,25]*A[22,21,23,26]*A[4,5,11,26]*
                A[18,21,24,19]*A[2,5,12,1]*A[13,17,24,14]*A[7,9,12,6]*s[19,18,20]*s[1,2,3]*
                s[6,7,8]*s[14,13,15]*v[27,22,20]*v[27,4,3]*v[28,10,8]*v[28,16,15]
    @tensor psipsi_1[:] :=  A[9,16,17,10]*A[20,25,35,19]*A[1,7,17,2]*A[28,34,35,27]*
                A[4,7,18,5]*A[31,34,36,30]*A[12,16,18,13]*A[23,25,36,22]*s[10,9,11]*
                s[19,20,21]*s[2,1,3]*s[27,28,29]*v[37,15,11]*v[37,26,21]*v[38,8,3]*v[38,33,29]*
                s[5,4,6]*s[30,31,32]*s[22,23,24]*s[13,12,14]*v[39,8,6]*v[39,33,32]*v[40,26,24]*v[40,15,14]

    #@tensor phiphi_2[:] := A[9,13,7,8]*A[14,13,12,11]*A[9,5,2,1]*A[14,5,3,4]*
    #            A[10,6,2,1]*A[16,6,3,4]*A[10,15,7,8]*A[16,15,12,11]
    @tensor psiphi_2[:] := A[20,19,16,17]*A[11,19,10,9]*A[20,21,27,26]*A[11,21,4,5]*
                A[22,28,27,23]*A[1,28,4,2]*A[14,18,16,13]*A[6,18,10,7]*s[23,22,24]*
                s[2,1,3]*s[7,6,8]*s[13,14,15]*v[26,25,24]*v[5,12,3]*v[9,12,8]*v[17,25,15]
    @tensor psipsi_2[:] := A[10,37,13,9]*A[30,37,34,31]*A[2,38,5,1]*A[19,38,23,20]*
                A[4,39,5,3]*A[22,39,23,21]*A[32,40,34,33]*A[12,40,13,11]*s[9,10,15]*s[31,30,36]*
                s[1,2,7]*s[20,19,25]*v[14,17,15]*v[6,17,7]*v[27,28,36]*v[24,28,25]*s[3,4,8]*
                s[21,22,26]*s[33,32,35]*s[11,12,16]*v[6,18,8]*v[24,29,26]*v[27,29,35]*v[14,18,16]

    if printdetail == true
        #@printf " err: %.16e  %.16e  %.16e %.16e %.16e \n" phiphi_1[1] psiphi_1[1] psipsi_1[1]  1-psiphi_1[1]^2/psipsi_1[1]/phiphi_1[1]  (phiphi_1[1]+psipsi_1[1]-2*psiphi_1[1])/phiphi_1[1]
        #@printf " err: %.16e  %.16e  %.16e %.16e %.16e " phiphi_2[1] psiphi_2[1] psipsi_2[1]  1-psiphi_2[1]^2/psipsi_2[1]/phiphi_2[1]  (phiphi_2[1]+psipsi_2[1]-2*psiphi_2[1])/phiphi_2[1]
        @printf "Fidelity: %.16e " (phiphi_1[1]+psipsi_1[1]-2*psiphi_1[1])/phiphi_1[1]+(phiphi_2[1]+psipsi_2[1]-2*psiphi_2[1])/phiphi_2[1]
    end
    
    return (phiphi_2[1]+psipsi_2[1]-2*psiphi_2[1])/phiphi_2[1]+(phiphi_1[1]+psipsi_1[1]-2*psiphi_1[1])/phiphi_1[1]

end


function compute_fidelity_1(A,s,v,phiphi_1,phiphi_2,sA,sAv,sAvA_1,sAvsAv_1,sAvA_2,sAvsAv_2;printdetail=false)

    #! sA = s*A; sAv = sA*v ; sAvA = sAv*A; sAvsAv = sAv*sAv 
    @tensor sA[:] = A[2,-2,-1,1]*s[1,2,-3];
    @tensor sAv[:] = sA[-1,-2,1]*v[-3,-4,1];
    @tensor sAvA_1[:] = sAv[-4,2,-3,1]*A[1,2,-1,-2];
    @tensor sAvsAv_1[:] = sAv[-3,2,-4,1]*sAv[-1,2,-2,1]
    @tensor sAvA_2[:] = sAv[1,-3,2,-4]*A[-1,-2,1,2] 
    @tensor sAvsAv_2[:] = sAv[1,-2,2,-1]*sAv[1,-4,2,-3]

    @tensor psiphi_1_half[:] := sAvA_1[-4,1,2,-2]*sAvA_1[-3,1,2,-1]
    @tensor psipsi_1_half[:] := sAvsAv_1[-2,2,-4,1]*sAvsAv_1[-1,2,-3,1]
    @tensor psiphi_1[:] := psiphi_1_half[1,2,3,4]*psiphi_1_half[1,2,3,4]
    @tensor psipsi_1[:] := psipsi_1_half[1,2,3,4]*psipsi_1_half[1,2,3,4] 

    @tensor psiphi_2_half[:] := sAvA_2[-4,1,2,-2]*sAvA_2[-3,1,2,-1]
    @tensor psipsi_2_half[:] := sAvsAv_2[-3,2,-1,1]*sAvsAv_2[-4,2,-2,1]
    @tensor psiphi_2[:] := psiphi_2_half[1,2,3,4]*psiphi_2_half[1,2,3,4] 
    @tensor psipsi_2[:] := psipsi_2_half[1,2,3,4]*psipsi_2_half[1,2,3,4]


    if printdetail == true
        @printf "Fidelity: %.16e " (phiphi_1[1]+psipsi_1[1]-2*psiphi_1[1])/phiphi_1[1]
        @printf "   %.16e " (phiphi_2[1]+psipsi_2[1]-2*psiphi_2[1])/phiphi_2[1]
        @printf "   %.16e " (phiphi_1[1]+psipsi_1[1]-2*psiphi_1[1])/phiphi_1[1]+(phiphi_2[1]+psipsi_2[1]-2*psiphi_2[1])/phiphi_2[1]
    end
    
    #return (phiphi_1[1]+psipsi_1[1]-2*psiphi_1[1])/phiphi_1[1]
    return (phiphi_2[1]+psipsi_2[1]-2*psiphi_2[1])/phiphi_2[1]
    #return (phiphi_2[1]+psipsi_2[1]-2*psiphi_2[1])/phiphi_2[1]+(phiphi_1[1]+psipsi_1[1]-2*psiphi_1[1])/phiphi_1[1]
end



function  compute_projectors(A,s,v,chimax)

    #
    #   compute y
    #
    @tensor sA[:] := A[2,-3,5,1]*A[4,-4,5,3]*s[1,2,-1]*s[3,4,-2]
    @tensor v4[:] := v[1,3,-1]*v[1,4,-2]*v[2,4,-4]*v[2,3,-3] 
    
    #
    @tensor Envy[:] := sA[3,5,1,2]*sA[4,6,1,2]*sA[7,8,13,14]*sA[9,10,11,12]*sA[15,17,13,14]*sA[16,18,11,12]*
                    sA[21,22,-1,-2]*sA[19,20,-3,-4]*v4[7,3,9,4]*v4[8,5,10,6]*v4[21,15,19,16]*v4[22,17,20,18]
                    println(norm(Envy-permutedims(Envy,[2,1,4,3]))/norm(Envy))
    #
    sizeEnvy = size(Envy)
    Envy = reshape(Envy,prod(sizeEnvy[1:2]),prod(sizeEnvy[1:2]))
    y = compute_isometry(Envy,chimax)
    println("Truncation error for y: ",(tr(Envy)-tr(Envy*y*y'))/tr(Envy))
    y = reshape(y,sizeEnvy[1],sizeEnvy[2],size(y,2))
    #
    #    compute w
    # 
    @tensor Envw[:] := sA[1,2,3,4]*sA[25,12,3,4]*sA[26,19,17,18]*sA[14,13,17,18]*sA[1,2,6,5]*sA[22,11,6,5]*
                    sA[23,20,16,15]*sA[14,13,16,15]*v[24,-1,25]*v[24,-2,26]*v[9,7,12]*v[9,8,19]*v[10,7,11]*
                    v[10,8,20]*v[21,-3,22]*v[21,-4,23]
                    println(norm(Envw-permutedims(Envw,[2,1,4,3]))/norm(Envw))
    sizeEnvw = size(Envw)
    Envw = reshape(Envw,prod(sizeEnvw[1:2]),prod(sizeEnvw[1:2]))    
    w = compute_isometry(Envw,chimax)
    println("Truncation error for w: ",(tr(Envw)-tr(Envw*w*w'))/tr(Envw))
    w = reshape(w,sizeEnvw[1],sizeEnvw[1],size(w,2))
    return y,w


end


function compute_projector_vertical(A,s,v,chimax)

    @tensor sA[:] := v[-3,1,-1]*v[-4,1,-2] 
    @tensor v4[:] := A[2,5,12,1]*A[3,5,11,4]*A[7,10,11,6]*
                    A[8,10,12,9]*s[1,2,-1]*s[4,3,-2]*s[6,7,-4]*s[9,8,-3]
    #
    @tensor Envy[:] := sA[3,5,1,2]*sA[4,6,1,2]*sA[7,8,13,14]*sA[9,10,11,12]*sA[15,17,13,14]*sA[16,18,11,12]*
                    sA[21,22,-1,-2]*sA[19,20,-3,-4]*v4[7,3,9,4]*v4[8,5,10,6]*v4[21,15,19,16]*v4[22,17,20,18]
    println(norm(Envy-permutedims(Envy,[2,1,4,3]))/norm(Envy))
    #
    sizeEnvy = size(Envy)
    Envy = reshape(Envy,prod(sizeEnvy[1:2]),prod(sizeEnvy[1:2]))
    y = compute_isometry(Envy,chimax)
    println("Truncation error for y: ",(tr(Envy)-tr(Envy*y*y'))/tr(Envy))
    y = reshape(y,sizeEnvy[1],sizeEnvy[2],size(y,2))



    #
    #    compute w
    # 
    @tensor v[:] := A[2,-1,-2,1]*s[1,2,-3] 
    @tensor Envw[:] := sA[1,2,3,4]*sA[25,12,3,4]*sA[26,19,17,18]*sA[14,13,17,18]*sA[1,2,6,5]*sA[22,11,6,5]*
                    sA[23,20,16,15]*sA[14,13,16,15]*v[24,-1,25]*v[24,-2,26]*v[9,7,12]*v[9,8,19]*v[10,7,11]*
                    v[10,8,20]*v[21,-3,22]*v[21,-4,23]
                    println(norm(Envw-permutedims(Envw,[2,1,4,3]))/norm(Envw))
    sizeEnvw = size(Envw)
    Envw = reshape(Envw,prod(sizeEnvw[1:2]),prod(sizeEnvw[1:2]))    
    w = compute_isometry(Envw,chimax)
    println("Truncation error for w: ",(tr(Envw)-tr(Envw*w*w'))/tr(Envw))
    w = reshape(w,sizeEnvw[1],sizeEnvw[1],size(w,2))
    return y,w


end

function fidelity_check(A,s,v,chimax;printdetail=false)

    @tensor phiphi[:] := A[4,11,5,6]*A[8,11,10,9]*A[4,3,2,1]*A[7,3,2,1]*A[7,12,5,6]*A[8,12,10,9]
    @tensor psiphi[:] := A[13,20,11,12]*A[17,20,19,18]*A[13,6,5,4]*A[2,6,5,1]*A[9,21,11,8]*A[14,21,19,15]*
                        s[1,2,3]*s[8,9,10]*s[15,14,16]*v[4,7,3]*v[12,7,10]*v[18,17,16]
    @tensor psipsi[:] := A[16,25,18,15]*A[20,25,29,21]*A[2,6,5,1]*A[4,6,5,3]*A[14,28,18,13]*A[26,28,29,27]*
                        s[15,16,17]*s[21,20,24]*s[1,2,8]*v[12,11,17]*v[23,22,24]*v[9,11,8]*s[3,4,7]*s[13,14,19]*
                        s[27,26,30]*v[9,10,7]*v[12,10,19]*v[23,22,30]
    #
    if printdetail == true
        @printf "Fidelity: phiphi %.16e  psiphi %.16e  psipsi %.16e " phiphi[1] psiphi[1] psipsi[1]
        @printf "   %.16e " (phiphi[1]+psipsi[1]-2*psiphi[1])/phiphi[1]
    end
    #    
end