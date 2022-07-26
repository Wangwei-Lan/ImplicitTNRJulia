beta = 1/Temperature
Jtemp = zeros(2,2,2,2); Jtemp[1,1,1,1]=1.0; Jtemp[2,2,2,2]=1.0;
Jtemp_impurity = zeros(2,2,2,2); Jtemp_impurity[1,1,1,1] = 1.0; Jtemp_impurity[2,2,2,2] = 0.0

Ltemp = [exp(J1*beta) exp(-J1*beta); exp(-J1*beta) exp(J1*beta)];
Etemp = sqrt(Ltemp)





#
# Binder ration for magnetization
@tensor A[:] := Jtemp[1,2,3,4]*Etemp[1,-1]*Etemp[2,-2]*Etemp[3,-3]*Etemp[4,-4]  
@tensor A_single[:] := Jtemp_impurity[1,2,3,4]*Etemp[1,-1]*Etemp[2,-2]*Etemp[3,-3]*Etemp[4,-4];  
@tensor A_double[:] := Jtemp[1,2,3,4]*Etemp[1,-1]*Etemp[2,-2]*Etemp[3,-3]*Etemp[4,-4]
@tensor A_triple[:] := Jtemp_impurity[1,2,3,4]*Etemp[1,-1]*Etemp[2,-2]*Etemp[3,-3]*Etemp[4,-4]  
@tensor A_quadruple[:] := Jtemp[1,2,3,4]*Etemp[1,-1]*Etemp[2,-2]*Etemp[3,-3]*Etemp[4,-4]

@tensor A_single_1[:] := A_single[-1,-3,2,1]*A[-2,-7,4,1]*A[-5,-4,2,3]*A[-6,-8,4,3]
@tensor A[:] := A[-1,-3,2,1]*A[-2,-7,4,1]*A[-5,-4,2,3]*A[-6,-8,4,3]
A_single_1 = reshape(A_single_1,4,4,4,4)
A = reshape(A,4,4,4,4)

Atemp = reshape(A,16,16)
Atemp_single = reshape(A_single_1,16,16)

R = svd(Atemp);

x = 1