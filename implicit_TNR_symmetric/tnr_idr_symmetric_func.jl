function compute_isometry_symmetric(A,chimax;Aimp=0)

    if Aimp ==0
        #@tensor Env[:] := A[-2,-1,13,14]*A[8,7,9,14]*A[2,1,13,5]*A[3,4,9,5]*A[2,1,11,6]*A[3,4,10,6]*A[-4,-3,11,12]*A[8,7,10,12]
        @tensor Env[:] := A[1,2,3,4]*A[-2,-1,6,4]*A[1,2,3,5]*A[-4,-3,6,5]
    else
        @tensor Env[:] := Aimp[-2,-1,13,14]*A[8,7,9,14]*A[2,1,13,5]*A[3,4,9,5]*A[2,1,11,6]*A[3,4,10,6]*Aimp[-4,-3,11,12]*A[8,7,10,12]
    end
    
    sizeA = size(A)
    Env = reshape(Env,prod(sizeA[1:2]),prod(sizeA[1:2]))
    Rlt = svd(Env)
    
    chitemp = sum(Rlt.S/maximum(Rlt.S) .> 1.0e-14)
    #println(Rlt.S/maximum(Rlt.S))
    chikept = min(chitemp,chimax,prod(sizeA[1:2]))
    v = reshape(Rlt.U[:,1:chikept],sizeA[2],sizeA[1],chikept)
    return v    
end


function compute_isometry_symmetric_single_tensor(A,chimax)

    sizeA = size(A)
    A = permutedims(A,[2,1,3,4])
    Env = reshape(A,prod(sizeA[1:2]),prod(sizeA[3:4]))

    Rlt = svd(Env)
    chitemp = sum(Rlt.S/maximum(Rlt.S) .> 1.0e-14)
    #println(Rlt.S/maximum(Rlt.S))
    chikept = min(chitemp,chimax,prod(sizeA[1:2]))
    v = reshape(Rlt.U[:,1:chikept],sizeA[2],sizeA[1],chikept)
    return v 
end

function compute_isometry_symmetric_two_tensor(A,chimax;Aimp =0)

    if Aimp ==0
        #@tensor Env[:] := A[-2,-1,13,14]*A[8,7,9,14]*A[2,1,13,5]*A[3,4,9,5]*A[2,1,11,6]*A[3,4,10,6]*A[-4,-3,11,12]*A[8,7,10,12]
        @tensor Env[:] := A[-2,-1,5,4]*A[3,2,1,4]*A[-4,-3,5,6]*A[3,2,1,6]
    else
        #@tensor Env[:] := Aimp[-2,-1,13,14]*A[8,7,9,14]*A[2,1,13,5]*A[3,4,9,5]*A[2,1,11,6]*A[3,4,10,6]*Aimp[-4,-3,11,12]*A[8,7,10,12]
        @tensor Env[:] := Aimp[-2,-1,5,4]*Aimp[-4,-3,5,6]*A[3,2,1,4]*A[3,2,1,6]
    end
    
    sizeA = size(A)
    Env = reshape(Env,prod(sizeA[1:2]),prod(sizeA[1:2]))
    Rlt = svd(Env)
    
    chitemp = sum(Rlt.S/maximum(Rlt.S) .> 1.0e-14)
    #println(Rlt.S/maximum(Rlt.S))
    chikept = min(chitemp,chimax,prod(sizeA[1:2]))
    v = reshape(Rlt.U[:,1:chikept],sizeA[2],sizeA[1],chikept)
    return v    
end


function compute_isometry_symmetric_two_tensor_impurity(A,Aimp;position="leftup")


    if position == "leftup" 
        @tensor Env[:] :=  A[1,2,3,4]*A[1,2,3,5]*Aimp[-2,4,6,-1]*Aimp[-4,5,6,-3]
    elseif position == "leftdown"
        @tensor Env[:] := A[1,2,3,5]*A[1,2,3,4]*Aimp[6,5,-2,-1]*Aimp[6,4,-4,-3]
    elseif position =="rightup"
        @tensor Env[:] := Aimp[-2,-1,5,4]*Aimp[-4,-3,5,6]*A[3,2,1,4]*A[3,2,1,6]
    elseif position == "rightdown"
        @tensor Env[:] := A[1,2,3,6]*A[1,2,3,4]*Aimp[5,-3,-4,4]*Aimp[5,-1,-2,6]

    end

    sizeEnv = size(Env)
    Env = reshape(Env,prod(sizeEnv[1:2]),prod(sizeEnv[3:4]))
    Rlt = svd(Env)

    chitemp = sum(Rlt.S/maximum(Rlt.S) .> 1.0e-14)
    #println(Rlt.S/maximum(Rlt.S))
    chikept = min(chitemp,chimax,prod(sizeEnv[1:2]))
    v_impurity = reshape(Rlt.U[:,1:chikept],sizeEnv[1],sizeEnv[2],chikept)
    return v_impurity    
end



function compute_projector_symmetric(A,v,chimax)

    @tensor vAvA[:] := v[1,2,-1]*A[2,1,5,-3]*v[3,4,-2]*A[4,3,5,-4]
    @tensor v4[:] := v[1,4,-2]*v[1,3,-1]*v[2,4,-4]*v[2,3,-3]
    @tensor Envy[:] := vAvA[7,8,-2,-1]*vAvA[3,5,1,2]*vAvA[4,6,1,2]*vAvA[9,10,-4,-3]*v4[7,3,9,4]*v4[8,5,10,6]
    sizeEnvy = size(Envy)
    Envy = reshape(Envy,prod(sizeEnvy[1:2]),prod(sizeEnvy[1:2]))
    Rlt = svd(Envy)

    chitemp = sum(Rlt.S/maximum(Rlt.S) .> 1.0e-14)
    chikept = min(chitemp,chimax,prod(sizeEnvy[1:2]))
    y = reshape(Rlt.U[:,1:chikept],sizeEnvy[1],sizeEnvy[2],chikept)

    @tensor Envw[:] := vAvA[1,2,3,4]*vAvA[25,12,3,4]*vAvA[26,19,17,18]*vAvA[14,13,17,18]*
                    vAvA[1,2,6,5]*vAvA[22,11,6,5]*vAvA[23,20,16,15]*vAvA[14,13,16,15]*
                    v[24,-1,25]*v[24,-2,26]*v[9,7,12]*v[9,8,19]*v[10,7,11]*v[10,8,20]*v[21,-3,22]*v[21,-4,23]
    sizeEnvw = size(Envw)
    #println(size(Envw))
    Envw = reshape(Envw,prod(sizeEnvw[1:2]),prod(sizeEnvw[1:2]))
    Rlt = svd(Envw)
    chitemp = sum(Rlt.S/maximum(Rlt.S) .> 1.0e-14)
    chikept = min(chitemp,chimax,prod(sizeEnvw[1:2]))
    w = reshape(Rlt.U[:,1:chikept],sizeEnvw[1],sizeEnvw[2],chikept)

    return y,w,vAvA
end


function compute_projector_tnr_symmetric(A,v,s,dis,chimax)

    @tensor sAsA[:] := s[1,2,-1]*A[2,1,5,-3]*s[3,4,-2]*A[4,3,5,-4]
    @tensor v4[:] := v[1,4,-2]*v[1,3,-1]*v[2,4,-4]*v[2,3,-3]
    @tensor Envy[:] := sAsA[7,8,-2,-1]*sAsA[3,5,1,2]*sAsA[4,6,1,2]*sAsA[9,10,-4,-3]*v4[7,3,9,4]*v4[8,5,10,6]
    sizeEnvy = size(Envy)
    Envy = reshape(Envy,prod(sizeEnvy[1:2]),prod(sizeEnvy[1:2]))
    Rlt = svd(Envy)

    chitemp = sum(Rlt.S/maximum(Rlt.S) .> 1.0e-14)
    chikept = min(chitemp,chimax,prod(sizeEnvy[1:2]))
    y = reshape(Rlt.U[:,1:chikept],sizeEnvy[1],sizeEnvy[2],chikept)

    @tensor Envw[:] := sAsA[1,2,3,4]*sAsA[25,12,3,4]*sAsA[26,19,17,18]*sAsA[14,13,17,18]*
                    sAsA[1,2,6,5]*sAsA[22,11,6,5]*sAsA[23,20,16,15]*sAsA[14,13,16,15]*
                    v[24,-1,25]*v[24,-2,26]*v[9,7,12]*v[9,8,19]*v[10,7,11]*v[10,8,20]*v[21,-3,22]*v[21,-4,23]
    sizeEnvw = size(Envw)
    #println(size(Envw))
    Envw = reshape(Envw,prod(sizeEnvw[1:2]),prod(sizeEnvw[1:2]))
    Rlt = svd(Envw)
    chitemp = sum(Rlt.S/maximum(Rlt.S) .> 1.0e-14)
    chikept = min(chitemp,chimax,prod(sizeEnvw[1:2]))
    w = reshape(Rlt.U[:,1:chikept],sizeEnvw[1],sizeEnvw[2],chikept)

    return y,w,sAsA
end


function compute_fidelity(A,v,s,dis,phiphi;printdetail=false)

    #@tensor phiphi[:] := A[1,2,3,7]*A[1,2,3,8]*A[6,5,4,7]*A[6,5,4,8]
    @tensor psiphi[:] := A[2,1,5,12]*A[10,11,13,12]*A[6,4,5,8]*A[9,15,13,8]*v[4,7,3]*v[15,14,16]*
                            s[1,2,3]*s[11,10,16]*dis[6,9,7,14]
    @tensor psipsi[:] := A[4,3,6,13]*A[9,10,11,13]*A[2,1,6,14]*A[8,7,11,14]*s[3,4,5]*s[1,2,5]*s[10,9,12]*s[7,8,12]

    if printdetail ==true
        @printf "phiphi: %.16e   psipsi: %.16e   psiphi: %.16e: err: %.16e  \n" phiphi[1] psipsi[1] psiphi[1] 1-psiphi[1]^2/psipsi[1]/phiphi[1] 
    end
    return (phiphi[1]+psipsi[1]-2*psiphi[1])/phiphi[1]    #1-psiphi[1]^2/psipsi[1]/phiphi[1] 
end






function renormalize_fragment_one_impurity(A,A_imp,v,v_imp,y,w)

 
    @tensor A_new[:] :=     A_imp[13,12,16,17]*A[15,14,16,18]*A[1,2,5,7]*A[3,4,5,6]*v_imp[12,13,24]*v_imp[8,10,24]*v[14,15,22]*v[19,21,22]*v[2,1,11]*v[8,9,11]*
                            v[4,3,23]*v[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3]+
                            A[13,12,16,17]*A_imp[15,14,16,18]*A[1,2,5,7]*A[3,4,5,6]*v[12,13,24]*v[8,10,24]*v_imp[14,15,22]*v_imp[19,21,22]*v[2,1,11]*v[8,9,11]*
                            v[4,3,23]*v[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3]+
                            A[13,12,16,17]*A[15,14,16,18]*A_imp[1,2,5,7]*A[3,4,5,6]*v[12,13,24]*v[8,10,24]*v[14,15,22]*v[19,21,22]*v_imp[2,1,11]*v_imp[8,9,11]*
                            v[4,3,23]*v[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3]+
                            A[13,12,16,17]*A[15,14,16,18]*A[1,2,5,7]*A_imp[3,4,5,6]*v[12,13,24]*v[8,10,24]*v[14,15,22]*v[19,21,22]*v[2,1,11]*v[8,9,11]*
                            v_imp[4,3,23]*v_imp[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3]
    return A_new
end



function renormalize_fragment_two_impurity(A,A_imp1,A_imp2,v,v_imp1,v_imp2,y,w)

    @tensor A_new[:] := A_imp1[13,12,16,17]*A_imp2[15,14,16,18]*A[1,2,5,7]*A[3,4,5,6]*v_imp1[12,13,24]*v_imp1[8,10,24]*v_imp2[14,15,22]*v_imp2[19,21,22]*v[2,1,11]*v[8,9,11]*
                        v[4,3,23]*v[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3]+
                        A_imp1[13,12,16,17]*A[15,14,16,18]*A_imp2[1,2,5,7]*A[3,4,5,6]*v_imp1[12,13,24]*v_imp1[8,10,24]*v[14,15,22]*v[19,21,22]*v_imp2[2,1,11]*v_imp2[8,9,11]*
                        v[4,3,23]*v[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3]+
                        A_imp1[13,12,16,17]*A[15,14,16,18]*A[1,2,5,7]*A_imp2[3,4,5,6]*v_imp1[12,13,24]*v_imp1[8,10,24]*v[14,15,22]*v[19,21,22]*v[2,1,11]*v[8,9,11]*
                        v_imp2[4,3,23]*v_imp2[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3]+
                        A[13,12,16,17]*A_imp1[15,14,16,18]*A_imp2[1,2,5,7]*A[3,4,5,6]*v[12,13,24]*v[8,10,24]*v_imp1[14,15,22]*v_imp1[19,21,22]*v_imp2[2,1,11]*v_imp2[8,9,11]*
                        v[4,3,23]*v[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3]+
                        A[13,12,16,17]*A_imp1[15,14,16,18]*A[1,2,5,7]*A_imp2[3,4,5,6]*v[12,13,24]*v[8,10,24]*v_imp1[14,15,22]*v_imp1[19,21,22]*v[2,1,11]*v[8,9,11]*
                        v_imp2[4,3,23]*v_imp2[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3]+
                        A[13,12,16,17]*A[15,14,16,18]*A_imp1[1,2,5,7]*A_imp2[3,4,5,6]*v[12,13,24]*v[8,10,24]*v[14,15,22]*v[19,21,22]*v_imp1[2,1,11]*v_imp1[8,9,11]*
                        v_imp2[4,3,23]*v_imp2[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3]

    return A_new
end



function renormalize_fragment_three_impurity(A,A_imp1,A_imp2,A_imp3,v,v_imp1,v_imp2,v_imp3,y,w)

    @tensor A_new[:] := A_imp1[13,12,16,17]*A_imp2[15,14,16,18]*A_imp3[1,2,5,7]*A[3,4,5,6]*v_imp1[12,13,24]*v_imp1[8,10,24]*v_imp2[14,15,22]*v_imp2[19,21,22]*v_imp3[2,1,11]*v_imp3[8,9,11]*
                        v[4,3,23]*v[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3]+
                        A_imp1[13,12,16,17]*A[15,14,16,18]*A_imp2[1,2,5,7]*A_imp3[3,4,5,6]*v_imp1[12,13,24]*v_imp1[8,10,24]*v[14,15,22]*v[19,21,22]*v_imp2[2,1,11]*v_imp2[8,9,11]*
                        v_imp3[4,3,23]*v_imp3[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3]+
                        A_imp1[13,12,16,17]*A_imp2[15,14,16,18]*A[1,2,5,7]*A_imp3[3,4,5,6]*v_imp1[12,13,24]*v_imp1[8,10,24]*v_imp2[14,15,22]*v_imp2[19,21,22]*v[2,1,11]*v[8,9,11]*
                        v_imp3[4,3,23]*v_imp3[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3]+
                        A[13,12,16,17]*A_imp1[15,14,16,18]*A_imp2[1,2,5,7]*A_imp3[3,4,5,6]*v[12,13,24]*v[8,10,24]*v_imp1[14,15,22]*v_imp1[19,21,22]*v_imp2[2,1,11]*v_imp2[8,9,11]*
                        v_imp3[4,3,23]*v_imp3[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3]
    return A_new
end


function renormalize_fragment_four_impurity(A,A_imp1,A_imp2,A_imp3,A_imp4,v,v_imp1,v_imp2,v_imp3,v_imp4,y,w)

    @tensor A_new[:] := A_imp1[13,12,16,17]*A_imp2[15,14,16,18]*A_imp3[1,2,5,7]*A_imp4[3,4,5,6]*v_imp1[12,13,24]*v_imp1[8,10,24]*v_imp2[14,15,22]*v_imp2[19,21,22]*v_imp3[2,1,11]*v_imp3[8,9,11]*
                        v_imp4[4,3,23]*v_imp4[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3]

    return A_new
end






function renormalize_fragment_tnr_one_impurity(A,A_imp,v,v_imp,s,s_imp,y,w)

    #
    @tensor A_single_new[:] := A_imp[13,12,16,17]*A[15,14,16,18]*A[1,2,5,7]*A[3,4,5,6]*s_imp[12,13,24]*v_imp[8,10,24]*s[14,15,22]*v[19,21,22]*s[2,1,11]*v[8,9,11]*
                            s[4,3,23]*v[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3]+
                            A[13,12,16,17]*A_imp[15,14,16,18]*A[1,2,5,7]*A[3,4,5,6]*s[12,13,24]*v[8,10,24]*s_imp[14,15,22]*v_imp[19,21,22]*s[2,1,11]*v[8,9,11]*
                            s[4,3,23]*v[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3]+
                            A[13,12,16,17]*A[15,14,16,18]*A_imp[1,2,5,7]*A[3,4,5,6]*s[12,13,24]*v[8,10,24]*s[14,15,22]*v[19,21,22]*s_imp[2,1,11]*v_imp[8,9,11]*
                            s[4,3,23]*v[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3]+
                            A[13,12,16,17]*A[15,14,16,18]*A[1,2,5,7]*A_imp[3,4,5,6]*s[12,13,24]*v[8,10,24]*s[14,15,22]*v[19,21,22]*s[2,1,11]*v[8,9,11]*
                            s_imp[4,3,23]*v_imp[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3]
    #
    return A_single_new
end

function renormalize_fragment_tnr_two_impurity(A,A_imp1,A_imp2,v,v_imp1,v_imp2,s,s_imp1,s_imp2,y,w)

    @tensor A_new[:] := A_imp1[13,12,16,17]*A_imp2[15,14,16,18]*A[1,2,5,7]*A[3,4,5,6]*s_imp1[12,13,24]*v_imp1[8,10,24]*s_imp2[14,15,22]*v_imp2[19,21,22]*v[2,1,11]*v[8,9,11]*
                                v[4,3,23]*v[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3]+
                                A_imp1[13,12,16,17]*A[15,14,16,18]*A_imp2[1,2,5,7]*A[3,4,5,6]*s_imp1[12,13,24]*v_imp1[8,10,24]*v[14,15,22]*v[19,21,22]*s_imp2[2,1,11]*v_imp2[8,9,11]*
                                v[4,3,23]*v[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3]+
                                A_imp1[13,12,16,17]*A[15,14,16,18]*A[1,2,5,7]*A_imp2[3,4,5,6]*s_imp1[12,13,24]*v_imp1[8,10,24]*v[14,15,22]*v[19,21,22]*v[2,1,11]*v[8,9,11]*
                                s_imp2[4,3,23]*v_imp2[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3]+
                                A[13,12,16,17]*A_imp1[15,14,16,18]*A_imp2[1,2,5,7]*A[3,4,5,6]*v[12,13,24]*v[8,10,24]*s_imp1[14,15,22]*v_imp1[19,21,22]*s_imp2[2,1,11]*v_imp2[8,9,11]*
                                v[4,3,23]*v[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3]+
                                A[13,12,16,17]*A_imp1[15,14,16,18]*A[1,2,5,7]*A_imp2[3,4,5,6]*v[12,13,24]*v[8,10,24]*s_imp1[14,15,22]*v_imp1[19,21,22]*v[2,1,11]*v[8,9,11]*
                                s_imp2[4,3,23]*v_imp2[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3]+
                                A[13,12,16,17]*A[15,14,16,18]*A_imp1[1,2,5,7]*A_imp2[3,4,5,6]*v[12,13,24]*v[8,10,24]*v[14,15,22]*v[19,21,22]*s_imp1[2,1,11]*v_imp1[8,9,11]*
                                s_imp2[4,3,23]*v_imp2[19,20,23]*y[6,7,-2]*y[18,17,-4]*w[9,10,-1]*w[20,21,-3]
    return A_new
end




function  HOTRG_initial(A,A_single,A_double,A_triple,A_quadruple)

    sizeA = size(A)

    #
    @tensor A_quadruple[:] := (A_quadruple[-1,-2,1,-5]*A[-4,-3,1,-6]+A[-1,-2,1,-5]*A_quadruple[-4,-3,1,-6]+
                                4*A_triple[-1,-2,1,-5]*A_single[-4,-3,1,-6]+4*A_single[-1,-2,1,-5]*A_triple[-4,-3,1,-6]+
                                6*A_double[-1,-2,1,-5]*A_double[-4,-3,1,-6])*0.0625
    @tensor A_triple[:] := (A_triple[-1,-2,1,-5]*A[-4,-3,1,-6]+A[-1,-2,1,-5]*A_triple[-4,-3,1,-6]+
                                3*A_double[-1,-2,1,-5]*A_single[-4,-3,1,-6]+3*A_single[-1,-2,1,-5]*A_double[-4,-3,1,-6])*0.125
    @tensor A_double[:] := (A_double[-1,-2,1,-5]*A[-4,-3,1,-6]+A[-1,-2,1,-5]*A_double[-4,-3,1,-6]+
                                2*A_single[-1,-2,1,-5]*A_single[-4,-3,1,-6])*0.25
    @tensor A_single[:] := (A_single[-1,-2,1,-5]*A[-4,-3,1,-6] + A[-1,-2,1,-5]*A_single[-4,-3,1,-6])*0.5
    #@tensor A_single[:] := A_single[-1,-2,1,-5]*A[-4,-3,1,-6]
    @tensor A[:] := A[-1,-2,1,-5]*A[-4,-3,1,-6]

    A_quadruple = reshape(A_quadruple,sizeA[1],sizeA[2]^2,sizeA[1],sizeA[2]^2)
    A_triple= reshape(A_triple,sizeA[1],sizeA[2]^2,sizeA[1],sizeA[2]^2)
    A_double = reshape(A_double,sizeA[1],sizeA[2]^2,sizeA[1],sizeA[2]^2)
    A_single = reshape(A_single,sizeA[1],sizeA[2]^2,sizeA[1],sizeA[2]^2)
    #
    A = reshape(A,sizeA[1],sizeA[2]^2,sizeA[1],sizeA[2]^2)

    return A,A_single,A_double,A_triple,A_quadruple

end





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
    tensor = reshape(pinv(B,rtol=1.0e-12)*P,sizeP)
    #tensor = reshape(pinv(B)*P,sizeP)
    return tensor
end


function f(B,x)
    @tensor y[:] := B[-1,-2,1,2]*x[1,2,-3]
    return y
end

function  idr_update_iterative(P,B,s,iter)
    
    sizeP = size(P);sizeB = size(B)
    B = (B + permutedims(B,[3,4,1,2]))*0.5
    iter <= 100 ? maxiteration = 100 : maxiteration = 50
    tensor,info = linsolve(x->f(B,x),P,s,isposdef=true,ishermitian=true,issymmetric=true,maxiter=maxiteration)
    return tensor
end
function update_s(A,v,s,dis,iter)

    @tensor P[:] := A[-2,-1,12,13]*A[1,2,4,13]*A[10,9,12,11]*A[6,5,4,11]*s[2,1,3]*v[5,7,3]*v[9,8,-3]*dis[10,6,8,7]
    @tensor B[:] := A[-2,-1,8,7]*A[1,2,5,7]*A[-4,-3,8,9]*A[4,3,5,9]*s[2,1,6]*s[3,4,6]
    snew = idr_update_iterative(P,B,s,iter)
    return snew
end


function update_s(A,v,s,dis)

    @tensor P[:] := A[-2,-1,12,13]*A[1,2,4,13]*A[10,9,12,11]*A[6,5,4,11]*s[2,1,3]*v[5,7,3]*v[9,8,-3]*dis[10,6,8,7]
    @tensor B[:] := A[-2,-1,8,7]*A[1,2,5,7]*A[-4,-3,8,9]*A[4,3,5,9]*s[2,1,6]*s[3,4,6]
    snew = idr_update(P,B,2)
    return snew
end


function update_s_large(A,v,s,dis,w,A4,A4v,iter)

    @tensor P[:] := A4v[13,12,10,11]*A[-2,-1,13,14]*A[7,8,12,14]*w[1,2,10]*w[4,6,11]*dis[2,6,3,5]*v[1,3,-3]*v[4,5,9]*s[8,7,9]
    @tensor B[:] := A4[10,6,8,7]*A[-2,-1,10,11]*A[1,2,6,11]*A[-4,-3,8,9]*A[4,3,7,9]*s[2,1,5]*s[3,4,5]
    snew = idr_update_iterative(P,B,s,iter)
    return snew
end



function update_dis(A,v,s,dis)
    sizedis = size(dis)
    @tensor Env_dis[:] := A[1,2,5,11]*A[6,7,9,11]*A[-1,4,5,12]*A[-2,10,9,12]*s[2,1,3]*s[7,6,8]*v[4,-3,3]*v[10,-4,8]
    #println("dis symmetric: ",norm(Env_dis-permutedims(Env_dis,[2,1,4,3]))/norm(Env_dis))
    dis,S = update_tensor(Env_dis+permutedims(Env_dis,[2,1,4,3]),2)
    return dis
end




function update_v(A,v,s,dis)
    sizev = size(v)
    @tensor Env_v[:] := A[8,9,11,10]*A[1,2,4,10]*A[13,-1,11,12]*A[6,5,4,12]*s[9,8,-3]*s[2,1,3]*v[5,7,3]*dis[13,6,-2,7]
    v,S = update_tensor(Env_v,2)
    return v
end

function update_v_impurity(A,Aimp,v,vimp,s,simp,dis)
    sizev = size(v)
    @tensor Env_vimp[:] := Aimp[8,9,11,10]*A[1,2,4,10]*Aimp[13,-1,11,12]*A[6,5,4,12]*simp[9,8,-3]*s[2,1,3]*v[5,7,3]*dis[13,6,-2,7]
    vimp,S = update_tensor(Env_vimp,2)
    return vimp
end



function update_s_impurity(A,Aimp,v,vimp,s,simp,dis)

    @tensor Pimp[:] := Aimp[-2,-1,12,13]*A[1,2,4,13]*Aimp[10,9,12,11]*A[6,5,4,11]*s[2,1,3]*vimp[9,8,-3]*v[5,7,3]*dis[10,6,8,7]
    @tensor Bimp[:] := Aimp[-2,-1,8,7]*Aimp[-4,-3,8,9]*A[1,2,5,7]*A[4,3,5,9]*s[2,1,6]*s[3,4,6]

    simp = idr_update(Pimp,Bimp,2) 
    return simp
end


function update_dis_impurity(A,Aimp,v,vimp,s,simp,dis)


    sizedis = size(dis)
    @tensor Env_dis_1[:] := A[1,2,5,11]*A[6,7,9,11]*A[-1,4,5,12]*A[-2,10,9,12]*s[2,1,3]*s[7,6,8]*v[4,-3,3]*v[10,-4,8]
    @tensor Env_dis_2[:] := Aimp[1,2,5,11]*A[6,7,9,11]*Aimp[-1,4,5,12]*A[-2,10,9,12]*simp[2,1,3]*s[7,6,8]*vimp[4,-3,3]*v[10,-4,8]
    @tensor Env_dis_3[:] := A[1,2,5,11]*Aimp[6,7,9,11]*A[-1,4,5,12]*Aimp[-2,10,9,12]*s[2,1,3]*simp[7,6,8]*v[4,-3,3]*vimp[10,-4,8]
    
    Env_dis = Env_dis_1+Env_dis_2+Env_dis_3
    #println("is symmetric? :",norm(Env_dis-permutedims(Env_dis,[2,1,4,3]))/norm(Env_dis))
    dis,S = update_tensor(Env_dis+permutedims(Env_dis,[2,1,4,3]),2)
    return dis

end


function update_s_double_implicit(A,v,s,dis,chimax)


    #@tensor 


end




function compute_gauge(A;printdetail=false)

    sizeA = size(A);
    x = Matrix(1.0I,sizeA[2],sizeA[2])
    #x = rand(sizeA[2],sizeA[2])
    @tensor psiphi[:] := A[4,1,5,2]*A[5,3,4,6]*x[3,1]*x[6,2]
    @tensor psipsi[:] := A[1,2,3,4]*A[1,2,3,4]
    @printf "psipsi: %.16e   psiphi: %.16e: err: %.16e  \n" psipsi[1] psiphi[1] (psipsi[1]+psipsi[1]-2*psiphi[1])/psipsi[1] 
    UpdateLoop =5
    for j in 1:UpdateLoop
        #=
        @tensor Env_1[:] := A[2,1,4,-2]*A[4,3,2,-1]*x[3,1]
        @tensor Env_2[:] := A[2,-2,4,1]*A[4,-1,2,3]*x[3,1]
        =#
        @tensor Env_1[:] := A[2,-2,3,1]*A[3,-1,2,4]*x[4,1]
        @tensor Env_2[:] := A[3,1,4,-2]*A[4,2,3,-1]*x[2,1]
        x,S = update_tensor(Env_1+Env_2,1)

        if (j%100 ==0 || j==UpdateLoop)&& printdetail == true
            @tensor psiphi[:] := A[4,1,5,2]*A[5,3,4,6]*x[3,1]*x[6,2]
            @tensor psipsi[:] := A[1,2,3,4]*A[1,2,3,4]
            @printf "psipsi: %.16e   psiphi: %.16e: err: %.16e  \n" psipsi[1] psiphi[1] (psipsi[1]+psipsi[1]-2*psiphi[1])/psipsi[1] 
        end
    end

    return x

end


function compute_gradient(A,v,s,dis)

    @tensor B[:] := A[-2,-1,8,7]*A[1,2,5,7]*A[-4,-3,8,9]*A[4,3,5,9]*s[2,1,6]*s[3,4,6]
    @tensor g1[:] := B[-1,-2,1,2]*s[1,2,-3]
    @tensor g2[:] := A[-2,-1,12,13]*A[1,2,4,13]*A[10,9,12,11]*A[6,5,4,11]*s[2,1,3]*v[5,7,3]*v[9,8,-3]*dis[10,6,8,7]
    grad = 4*(g1-g2)
    return grad
end


function compute_gradient(A,v,s,dis,AAvvdis)

    #@tensor B[:] := A[-2,-1,8,7]*A[1,2,5,7]*A[-4,-3,8,9]*A[4,3,5,9]*s[2,1,6]*s[3,4,6]
    #@tensor g1[:] := B[-1,-2,1,2]*s[1,2,-3]
    @tensor As[:] := A[1,2,-1,-2]*s[2,1,-3]
    @tensor g1[:] := As[1,5,2]*As[1,3,2]*As[4,3,-3]*A[-2,-1,4,5]
    @tensor g2[:] := AAvvdis[6,3,-3,4]*A[-2,-1,6,5]*A[1,2,3,5]*s[2,1,4]
    grad = 4*(g1-g2)
    return grad
end




function apply_Hessian(A,s,v,dis,AAs,AAss,AAvvdis,x)

    @tensor AAvx[:] := A[2,1,5,-1]*A[3,4,5,-2]*v[1,2,6]*x[4,3,6]
    @tensor h1[:] := AAs[-1,-2,-3,1,2]*AAvx[1,2] 
    @tensor h2[:] := AAs[-1,-2,-3,1,2]*AAvx[2,1]
    @tensor h3[:] := A[-2,-1,4,5]*x[1,2,-3]*AAss[5,3]*A[2,1,4,3]
    @tensor h4[:] := AAvvdis[6,3,-3,4]*A[-2,-1,6,5]*A[1,2,3,5]*x[2,1,4]
    return 4*h1+4*h2+4*h3-4*h4
end

"""
    compute_direction

    compute directions for updating s; Use Hessian 
"""
function compute_direction(A,v,s,dis,grad)


    @tensor AAs[:] := A[-2,-1,3,-4]*A[2,1,3,-5]*s[1,2,-3]
    @tensor AAss[:] := A[2,1,5,-1]*A[3,4,5,-2]*s[1,2,6]*s[4,3,6]
    @tensor AAvvdis[:] := A[5,4,-1,7]*A[3,2,-2,7]*v[4,6,-3]*v[2,1,-4]*dis[5,3,6,1]

    sdirt,info = linsolve(x->apply_Hessian(A,s,v,dis,AAs,AAss,AAvvdis,x),-1*grad,s,maxiter=100)
    
    return sdirt
end



function alpha_denominator(A,v,s,dis,d_k)



    @tensor AAs[:] := A[-2,-1,3,-4]*A[2,1,3,-5]*s[1,2,-3]
    @tensor AAss[:] := A[2,1,5,-1]*A[3,4,5,-2]*s[1,2,6]*s[4,3,6]
    @tensor AAvvdis[:] := A[5,4,-1,7]*A[3,2,-2,7]*v[4,6,-3]*v[2,1,-4]*dis[5,3,6,1]

    @tensor AAvx[:] := A[2,1,5,-1]*A[3,4,5,-2]*v[1,2,6]*x[4,3,6]
    @tensor h1[:] := AAs[-1,-2,-3,1,2]*AAvx[1,2] 
    @tensor h2[:] := AAs[-1,-2,-3,1,2]*AAvx[2,1]
    @tensor h3[:] := A[-2,-1,4,5]*x[1,2,-3]*AAss[5,3]*A[2,1,4,3]
    @tensor h4[:] := AAvvdis[6,3,-3,4]*A[-2,-1,6,5]*A[1,2,3,5]*x[2,1,4]
    

    



end