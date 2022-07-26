function construct_weight(q,Temperature)
    beta = 1/Temperature
    W = zeros(q,q)

    for i in 1:q
        for j in 1:q
            if i ==j
                W[i,j] = exp(beta*1)
            else
                W[i,j] = exp(beta*0)
            end
        end
    end

    lambda = zeros(q)
    


    return W
end