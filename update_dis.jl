function update_dis_non_symmetric(Ta,Tb,wLup,wRup,sLup,sRup,wLdn,wRdn,sLdn,sRdn,dis)

    @tensor Env_dis_dn[:] := Tb[5,1,2,12]*Tb[5,4,-1,11]*Ta[9,12,6,7]*Ta[9,11,-2,10]*
                wLdn[4,-3,3]*wRdn[10,-4,8]*sLdn[1,2,3]*sRdn[7,6,8]
    @tensor Env_dis_up[:] := Ta[1,2,5,11]*Ta[-1,4,5,12]*Tb[6,11,9,7]*Tb[-2,12,9,10]*
                wLup[4,-3,3]*wRup[10,-4,8]*sLup[2,1,3]*sRup[7,6,8]
    dis,S = update_tensor(Env_dis_up+Env_dis_dn,2)
    #dis,S = update_tensor(Env_dis_up,2)
    #dis,S = update_tensor(Env_dis_dn,2)
    return dis,S    
end


function update_dis_non_symmetric_impurity(Ta,Tb,Apos,wLup,wRup,sLup,sRup,wLdn,wRdn,sLdn,sRdn,
                    wLup_imp,wRup_imp,sLup_imp,sRup_imp,wLdn_imp,wRdn_imp,sLdn_imp,sRdn_imp,dis)

    @tensor Env_dis_dn[:] := Tb[5,1,2,12]*Tb[5,4,-1,11]*Ta[9,12,6,7]*Ta[9,11,-2,10]*
                wLdn[4,-3,3]*wRdn[10,-4,8]*sLdn[1,2,3]*sRdn[7,6,8]
    @tensor Env_dis_up[:] := Ta[1,2,5,11]*Ta[-1,4,5,12]*Tb[6,11,9,7]*Tb[-2,12,9,10]*
                wLup[4,-3,3]*wRup[10,-4,8]*sLup[2,1,3]*sRup[7,6,8]
    #@tensor Env_dis_dn_imp[:] := Tb[5,1,2,12]*Tb[5,4,-1,11]*Ta[9,12,6,7]*Ta[9,11,-2,10]*
    #            wLdn[4,-3,3]*wRdn[10,-4,8]*sLdn[1,2,3]*sRdn[7,6,8]
    @tensor Env_dis_up_imp[:] := Ta[1,2,5,11]*Ta[-1,4,5,12]*Apos[6,11,9,7]*Apos[-2,12,9,10]*
                wLup[4,-3,3]*wRup_imp[10,-4,8]*sLup[2,1,3]*sRup_imp[7,6,8]
    
    #dis,S = update_tensor(Env_dis_up+Env_dis_dn,2)
    dis,S = update_tensor(Env_dis_up+Env_dis_dn+Env_dis_up_imp,2)
    #dis,S = update_tensor(Env_dis_up,2)
    #dis,S = update_tensor(Env_dis_dn,2)
    return dis,S    
end


