from numerical_M import *
import matplotlib.pyplot as plt
import pandas as pd

# params
alpha = 0.5
# Specifying L
phi = 0.2
L = geom_L(phi,0.5)
S = 200 
coeff = geom_coeff
coeff_param = phi
params = [alpha, S, coeff,coeff_param,L]
# estimator parameters
tol = 10E-6
p = 10
'''
# find M as a function of phi
alpha = 2
num=10
start = 0.0
stop = 0.999

df = pd.DataFrame(columns=["phi","k","remainders","partial sums","2M/π"])
for phi in np.linspace(start,stop,num):
    L = geom_L(phi,0.5)
    coeff_param = phi
    params = [alpha, S, coeff,coeff_param,L]
    vals = 
'''


print(M(params,tol,p,beta_char_func))
vals = approx_alternating_series(params,tol,p,beta_char_func)
print(vals[3])
