function update_wRdn_non_symmetric(Ta,Tb,wLdn,wRdn,sLdn,sRdn,dis)
    @tensor Env_wRdn[:] := Tb[5,1,2,10]*Tb[5,4,7,11]*Ta[12,10,8,9]*Ta[12,11,13,-1]*
                    wLdn[4,6,3]*sLdn[1,2,3]*sRdn[9,8,-3]*dis[7,13,6,-2]
    wRdn,S = update_tensor(Env_wRdn,2)
    return wRdn,S
end




function update_wRup_non_symmetric_impurity(Ta,Tb,Apos,wLup,wRup,sLup,sRup,wLup_imp,wRup_imp,sLup_imp,sRup_imp,dis)

    @tensor Env_wRup[:] := Ta[2,1,5,10]*Ta[7,4,5,11]*Tb[8,10,12,9]*Tb[13,11,12,-1]*
                        wLup[4,6,3]*sLup[1,2,3]*sRup[9,8,-3]*dis[7,13,6,-2]
    wRup,S = update_tensor(Env_wRup,2)
    
    
    @tensor Env_wRup_imp[:] := Ta[2,1,5,10]*Ta[7,4,5,11]*Apos[8,10,12,9]*Apos[13,11,12,-1]*
                        wLup[4,6,3]*sLup[1,2,3]*sRup_imp[9,8,-3]*dis[7,13,6,-2]
    wRup_imp,S = update_tensor(Env_wRup_imp,2)
    
    
    return wRup,wRup_imp
end


function update_wRup_non_symmetric(Ta,Tb,wLup,wRup,sLup,sRup,dis)

    @tensor Env_wRup[:] := Ta[2,1,5,10]*Ta[7,4,5,11]*Tb[8,10,12,9]*Tb[13,11,12,-1]*
                    wLup[4,6,3]*sLup[1,2,3]*sRup[9,8,-3]*dis[7,13,6,-2]
    wRup,S = update_tensor(Env_wRup,2)
    return wRup,S
end



