import matplotlib.pyplot as plt
import numpy as np
from numerical_M import *

def PDF(charfunc,cf_params,tol,p,x,N):
    L = cf_params[-1] #always put L last
    M_val = M(cf_params[:-1],tol,p,charfunc)[1]
    pdf = (1/2 + (M_val/np.pi)*np.sum([np.real_if_close(charfunc(n*np.pi/L,cf_params[:-1]))*np.cos(n*np.pi*x/L) for n in range(1,N+1)]))/L 
    return pdf
#------------------------ plotting --------------------------#
toggle = 0
if toggle == 0:
    B = 0.5
    S = 50
    T = np.linspace(-5,5,1000)
    alpha = 2
    coeff = geom_coeff
    phi = 0.1
    uniform_params = [B,S,coeff,phi]
    beta_params = [B,alpha,S,coeff,phi]

    uniform_vals = [gen_uniform_char_func(t,uniform_params) for t in T]
    beta_vals = [np.real_if_close(gen_beta_char_func(t,beta_params)) for t in T]
    print(np.real_if_close(gen_beta_char_func(0,beta_params)))
    plt.plot(T,uniform_vals)
    plt.show()
elif toggle == 1:
    # parameter values
    phi = 0.9
    B = 1
    alpha = 0.5
    # maximum values
    S = 40
    N = 40

    coeff = geom_coeff
    coeff_param = phi
    L = geom_L(phi,B)
    tol = 10E-4
    p = 10
    # param arrays
    uniform_params = [B,S,coeff,coeff_param,L]
    beta_params = [B,alpha, S, coeff,coeff_param,L]

    X = np.linspace(-L,L,50)
    #uniform_vals = [PDF(uniform_char_func,uniform_params,tol,p,x,N) for x in X]
    beta_vals = [PDF(gen_beta_char_func,beta_params,tol,p,x,N) for x in X]
    plt.plot(X,beta_vals)
    #plt.ylim(bottom=0)
    plt.show()
