from numerical_M import *
import matplotlib.pyplot as plt

phi = 0.9
S = 200
tol = 10E-6
p = 50
params = [geom_L(phi),S,phi]
vals = approx_alternating_series(params,tol,p,uniform_char_func)
plt.plot(vals[2])
plt.show()