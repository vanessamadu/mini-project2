from numerical_M import *
import matplotlib.pyplot as plt

phi = 0.1
S = 200
tol = 10E-6
p = 20
alpha = 0.5

params = [geom_L(phi),S,phi,alpha]

vals = approx_alternating_series(params,tol,p,beta_char_func)
plt.plot(vals[2])
plt.show()