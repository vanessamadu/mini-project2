from numerical_M import *
import matplotlib.pyplot as plt

''' B,S = params'''


# params
B = 1
# Specifying L
phi = 0.5
L = geom_L(phi,B)
S = 200 
coeff = geom_coeff
coeff_param = phi
params = [B,S,coeff,coeff_param,L]
# estimator parameters
tol = 10E-6
p = 10

vals = approx_alternating_series(params,tol,p,uniform_char_func)
print(vals[2])
#plt.plot(vals[2])
#plt.show()