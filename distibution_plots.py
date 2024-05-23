import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gamma,iv
from numerical_M import *


def gen_uniform_char_func(t,params):
    '''
    description:    finite approximation of the characteristic function of an infinite 
                    sum of uniformly distributed random variables on the interval [-B,B].
    params:     
    B:              real number > 0
    S:              integer > 0 (upper value for infinite product estimation)
    t:              real number = pi/L > 0
    returns:        real number (approximation of infinite product of sinc(n phi^s/L) wrt s)  
    '''
    B,S = params
    return np.prod([np.sinc(B*t) for s in range(S)])

def gen_beta_char_func(t,params):
    '''
    description:    finite approximation of the characteristic function of an infinite
                    sum of symmetric beta distributed random variables on the interval [-1/2,1/2].
    params:     
    alpha:          real number > 0
    S:              integer > 0 (upper value for infinite product estimation)
    t:              real number = pi/L > 0
    returns:        real number (approximation of infinite product of sinc(n phi^s/L) wrt s)  
    '''
    B,alpha, S = params
    const = (2*B)**(2*alpha-1)*gamma(alpha+0.5)
    return np.prod([const*iv(alpha-0.5,1j*t*B)*(1j*t*B/2)**(0.5-alpha) for s in range(S)])

def PDF(charfunc,cf_params,tol,p,x,N):
    L = cf_params[-1] #always put L last
    M_val = M(cf_params,tol,p,charfunc)[1]
    pdf = (1/2 + (M_val/np.pi)*np.sum([charfunc(n,cf_params)*np.cos(n*np.pi*x/L) for n in range(1,N+1)]))/L 
    return pdf
#------------------------ plotting --------------------------#
toggle = 0
if toggle == 0:
    B = 0.5
    S = 50
    T = np.linspace(-1,1,1000)
    alpha = 0.1
    uniform_params = [B,S]
    beta_params = [B,alpha,S]

    uniform_vals = [gen_uniform_char_func(t,uniform_params) for t in T]
    beta_vals = [np.real_if_close(gen_beta_char_func(t,beta_params)) for t in T]
    print(np.real_if_close(gen_beta_char_func(0,beta_params)))
    plt.plot(T,beta_vals)
    plt.ylim(bottom=0)
    plt.show()
elif toggle == 1:
    # parameter values
    phi = 0.2
    B = 5
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
    beta_vals = [PDF(beta_char_func,beta_params,tol,p,x,N) for x in X]
    plt.plot(X,beta_vals)
    plt.ylim(bottom=0)
    plt.show()
