from numerical_M import *
import matplotlib.pyplot as plt

phi = 0.9
S = 200
tol = 10E-6
p = 50

vals = approx_alternating_series(phi,S,tol,p,uniform_char_func,geom_L)
plt.plot(vals[2])
plt.show()