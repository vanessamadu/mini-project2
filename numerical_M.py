import numpy as np
from scipy.special import gamma,iv

def uniform_char_func(n,params):
    '''
    description:    finite approximation of the characteristic function of an infinite 
                    sum of uniformly distributed random variables on the interval [-B,B].
    params:     
    B:              real number > 0
    S:              integer > 0 (upper value for infinite product estimation)
    t:              real number = pi/L > 0
    coeff:          function (defining coefficients c_s)
    returns:        real number (approximation of infinite product of sinc(n phi^s/L) wrt s)  
    '''
    B,S,coeff,coeff_param,L = params
    return np.prod([np.sinc(coeff(coeff_param,s)*B*n/L) for s in range(S)])

def beta_char_func(n,params):
    '''
    description:    finite approximation of the characteristic function of an infinite
                    sum of symmetric beta distributed random variables on the interval [-1/2,1/2].
    params:     
    alpha:          real number > 0
    S:              integer > 0 (upper value for infinite product estimation)
    t:              real number = pi/L > 0
    coeff:          function (defining coefficients c_s)
    returns:        real number (approximation of infinite product of sinc(n phi^s/L) wrt s)  
    '''
    alpha, S, coeff,coeff_param,L = params
    const = gamma(alpha+0.5)
    return np.prod([const*iv(alpha-0.5,1j*coeff(coeff_param,s)*n*np.pi/L/2)*(1j*coeff(coeff_param,s)*n*np.pi/L/4)**(0.5-alpha) for s in range(S)])

def partial_alternating_sum(k,params,f):
    '''
    description:    finite approximation of the alternating series sum((-1)^{n-1}char_func(n))

    params:
    k:              integer > 0 (number of terms included in the series)
    t:              real number = pi/L > 0
    f:              characteristic function

    returns:        real number (approximation of the alternating series for n = 1,...,k+1)
    '''

    return np.sum([(-1)**(n-1)*f(n,params) for n in range(1,k+1)])

def remainder(k,params,f):
    '''
    description:    finds the difference between the kth and (k-1)th partial sums

    params:
    k:              integer > 0 (number of terms included in the series)
    t:              real number = pi/L > 0
    f:              characteristic function

    returns:        real number (see description)
    '''
    return partial_alternating_sum(k,params,f)-partial_alternating_sum(k-1,params,f)

def geom_L(phi,B):
    '''
    infinite sum of geometric series. 
    phi: real number: [0,1)
    '''
    return B/(1-np.abs(phi))

def geom_coeff(phi,s):
    return phi**s

def approx_alternating_series(params,tol,p,f):
    '''
    description:    approximates the alternating series sum((-1)^{n-1}char_func(n))

    S:              integer > 0 (upper value for infinite product estimation)
    tol:            real number (tolerance for convergence)
    phi:            real number: [0,1)
    p:              integer > 1 (required number of consecutive remainders less than tol)
    f:              characteristic function
    L:              real number > 0

    returns:        the value of k that exited the while loop, remainders, partial sum for k value
    '''
    k = 1
    remainders = []
    partial_sums = []

    for k in range(1,p):
        remainders.append(remainder(k,params,f))
        partial_sums.append(partial_alternating_sum(k,params,f))
    while (np.abs(remainders[::-1][:p]) > tol).any():
        remainders.append(remainder(k,params,f))
        partial_sums.append(partial_alternating_sum(k,params,f))
        k +=1
    return k, remainders, partial_sums, partial_alternating_sum(k,params,f)

def M(params,tol,p,f):
    val = approx_alternating_series(params,tol,p,f)[3]
    return f'{round(approx_alternating_series(params,tol,p,f)[3],2)*2}*π', np.pi/(2*val)