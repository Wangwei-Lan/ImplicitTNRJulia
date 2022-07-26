function update_wLdn_non_symmetric(Ta,Tb,wLdn,wRdn,sLdn,sRdn,dis)
    @tensor Env_wLdn[:] := Tb[11,8,9,10]*Tb[11,-1,13,12]*Ta[4,10,1,2]*Ta[4,12,7,5]*
                    wRdn[5,6,3]*sLdn[8,9,-3]*sRdn[2,1,3]*dis[13,7,-2,6]
    wLdn,S = update_tensor(Env_wLdn,2)
    return wLdn,S
end

function update_wLup_non_symmetric(Ta,Tb,wLup,wRup,sLup,sRup,dis)
    @tensor Env_wLup[:] := Ta[8,9,11,10]*Ta[13,-1,11,12]*Tb[1,10,4,2]*Tb[6,12,4,5]*
                    wRup[5,7,3]*sLup[9,8,-3]*sRup[2,1,3]*dis[13,6,-2,7]
    wLup,S = update_tensor(Env_wLup,2)
    return wLup,S
end



function update_wLup_non_symmetric_impurity(Ta,Tb,Apos,wLup,wRup,sLup,sRup,wLup_imp,wRup_imp,sLup_imp,sRup_imp,dis)

    @tensor Env_wLup[:] := Ta[8,9,11,10]*Ta[13,-1,11,12]*Tb[1,10,4,2]*Tb[6,12,4,5]*
                    wRup[5,7,3]*sLup[9,8,-3]*sRup[2,1,3]*dis[13,6,-2,7]

    @tensor Env_wLup_impurity[:] := Ta[8,9,11,10]*Ta[13,-1,11,12]*Apos[1,10,4,2]*Apos[6,12,4,5]*
                    wRup_imp[5,7,3]*sLup[9,8,-3]*sRup_imp[2,1,3]*dis[13,6,-2,7]
    
    wLup,S = update_tensor(Env_wLup+Env_wLup_impurity,2)


    return wLup,S
end