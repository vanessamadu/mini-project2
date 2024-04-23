from numerical_M import *
import matplotlib.pyplot as plt

phi = 0.5
S = 200 
tol = 10E-4
p = 20 
alpha = 0.25

params = [geom_L(phi),S,phi,alpha]

vals = approx_alternating_series(params,tol,p,beta_char_func)
print(np.real(vals[3]))
plt.plot([np.real(x) for x in vals[2]])
plt.show()