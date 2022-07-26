function initial_S(M,qnum_1,qnum_2,chis_max)
    sizeM = size(M)
    M = reshape(M,prod(sizeM[1:2]),prod(sizeM[3:4])) 
    qnum_temp = Merge(qnum_1,qnum_2) 
    Snew,qnum_S = compute_isometry_Z2(M,qnum_temp,qnum_temp,chis_max)

    return Snew,qnum_S
end



function update_S(P,B,S)
    
    #@tensor 

end