import matplotlib.pyplot as plt
import numpy as np
from function_approximations import *

#------------------------ plotting --------------------------#
toggle = 1
if toggle == 0:
    '''
    B = 2
    S = 50
    T = np.linspace(-5,5,1000)
    alpha =0.5
    coeff = geom_coeff
    phi = 0.9
    uniform_params = [B,S,coeff,phi]
    beta_params = [B,alpha,S,coeff,phi]

    uniform_vals = [gen_uniform_char_func(t,uniform_params) for t in T]
    beta_vals = [np.real_if_close(gen_beta_char_func(t,beta_params)) for t in T]
    print(np.real_if_close(gen_beta_char_func(0,beta_params)))
    plt.plot(T,beta_vals)
    plt.show()
    '''
    pass
elif toggle == 1:
    # parameter values
    phi = 0.6
    B = 0.5
    alpha = 3
    # maximum values
    S = 20
    N = 20

    c_s_func = c_s_geom
    c_s_params = phi
    L_func = L_geom
    L_params = [B,phi]
    L = L_func(L_params)
    tol = 10E-4
    p = 10
    # param arrays
    uniform_params = B
    beta_params = [alpha,B]

    X = np.linspace(-L,L,100)
    #M_val_beta = M(L_func,L_params,S,c_s_func,c_s_params,beta_params,beta_char_func,tol,p)
    
    M_val_uniform = M(L_func,L_params,S,c_s_func,c_s_params,uniform_params,uniform_char_func,tol,p)
    print(M_val_uniform)
    uniform_vals = [PDF(x,M_val_uniform,N,L_func,L_params,S,c_s_func,c_s_params,uniform_params,
                        uniform_char_func) for x in X]
    #beta_vals = [PDF(x,M_val_beta,N,L_func,L_params,S,c_s_func,c_s_params,beta_params,beta_char_func) for x in X]
    plt.plot(X,uniform_vals)
    plt.ylim(bottom=0)
    plt.show()
