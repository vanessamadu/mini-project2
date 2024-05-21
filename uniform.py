from numerical_M import *
import matplotlib.pyplot as plt

''' B,S = params'''

# Specifying L
phi = 0.5
L = geom_L(phi)
# params
B = 1
S = 100 
params = [B,S]
# estimator parameters
tol = 10E-6
p = 20

vals = approx_alternating_series(params,tol,p,uniform_char_func)
print(vals[3])
plt.plot(vals[2])
plt.show()