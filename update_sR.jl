function compute_gradient_sRup(Ta,Tb,wLup,wRup,sLup,sRup,dis)

    #@tensor 

end

function compute_f_sRup(Ta,Tb,wLup,wRup,sLup,sRup,dis)


end

function update_sRup_non_symmetric(Ta,Tb,wLup,wRup,sLup,sRup,dis)

    @tensor P[:] := Ta[2,1,5,12]*Ta[6,4,5,11]*Tb[-2,12,13,-1]*Tb[10,11,13,9]*
                    wLup[4,7,3]*wRup[9,8,-3]*sLup[1,2,3]*dis[6,10,7,8]
    @tensor B[:] := Ta[2,1,6,7]*Ta[4,3,6,8]*Tb[-2,7,9,-1]*Tb[-4,8,9,-3]*
                    sLup[1,2,5]*sLup[3,4,5]
    sRup = idr_update(P,B,2)
    return sRup
end


function update_sRup_non_symmetric_impurity(Ta,Tb,Apos,wLup,wRup,sLup,sRup,wLup_imp,wRup_imp,sLup_imp,sRup_imp,dis)

    @tensor P[:] := Ta[2,1,5,12]*Ta[6,4,5,11]*Tb[-2,12,13,-1]*Tb[10,11,13,9]*
                    wLup[4,7,3]*wRup[9,8,-3]*sLup[1,2,3]*dis[6,10,7,8]
    @tensor B[:] := Ta[2,1,6,7]*Ta[4,3,6,8]*Tb[-2,7,9,-1]*Tb[-4,8,9,-3]*
                    sLup[1,2,5]*sLup[3,4,5]
    sRup = idr_update(P,B,2)

    @tensor P_imp[:] := Ta[2,1,5,12]*Ta[6,4,5,11]*Apos[-2,12,13,-1]*Apos[10,11,13,9]*
                    wLup[4,7,3]*wRup_imp[9,8,-3]*sLup[1,2,3]*dis[6,10,7,8]
    @tensor B_imp[:] := Ta[2,1,6,7]*Ta[4,3,6,8]*Apos[-2,7,9,-1]*Apos[-4,8,9,-3]*
                    sLup[1,2,5]*sLup[3,4,5]
    sRup_imp = idr_update(P_imp,B_imp,2)

    return sRup,sRup_imp
end





function update_sRdn_non_symmetric(Ta,Tb,wLdn,wRdn,sLdn,sRdn,dis)
    @tensor P[:] := Tb[5,1,2,12]*Tb[5,4,7,9]*Ta[13,12,-2,-1]*Ta[13,9,10,11]*
                        wLdn[4,6,3]*wRdn[11,8,-3]*sLdn[1,2,3]*dis[7,10,6,8]
    @tensor B[:] := Tb[6,3,4,8]*Tb[6,1,2,7]*Ta[9,8,-2,-1]*Ta[9,7,-4,-3]*
                        sLdn[3,4,5]*sLdn[1,2,5]
    sRdn = idr_update(P,B,2)
    return sRdn
end





function update_sRup_non_symmetric_cg(Ta,Tb,wLup,wRup,sLup,sRup,dis)


    #@tensor psipsi_wo_sRupsRup[:] := Ta*Ta*Tb*Tb*wLup*wRup*sLup*sRup*dis
    #@tensor psiphi_wo_sRup[:] := Ta*Ta*Tb*Tb*wLup*wRup*sLup*sRup*dis


end