using Plots
append!(MagError,((abs.(Result["Magnetization"]) .- M)./M)[end])
append!(FEError,abs.((Result["FreeEnergy"] .-feexact)./feexact)[25])
append!(pinvError,1.0e-9)
#=
gr(size=(600,600), legend=false)
#
#p1 = plot(abs.(Result["FreeEnergy"] .-feexact),markershape=:circle,yscale=:log10)
#p2 = plot(abs.(abs.(Result["Magnetization"]).-M),marker=:circle,yscale=:log10)
#
#
plot!(p1,abs.(Result["FreeEnergy"] .-feexact),markershape=:square,yscale=:log10)
plot!(p2,abs.(abs.(Result["Magnetization"]).-M),marker=:square,yscale=:log10)
p3 = plot(p1, p2, layout=(1,2))
#plot(p3,[p1,p2], layout=(1,2))
#
=#