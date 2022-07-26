using FileIO
FEerror = []
for j in 2:2:26
    println("TNR_IDR_pinv_chimax_$j.jld2")
    #FE= load("TNR_IDR_pinv_cg_chimax_$j.jld2")
    FE= load("TNR_IDR_pinv_chimax_$j.jld2")
    append!(FEerror,FE["Result"]["FreeEnergyErr"][end])
end