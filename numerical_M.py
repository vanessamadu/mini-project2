import numpy as np
from scipy.special import gamma,iv

def uniform_char_func(n,L,S,phi):
    '''
    description:    finite approximation of the characteristic function of an infinite geometrically weighted
                    sum of uniformly distributed random variables on the interval [-1,1].
    params:     
    n:              integer > 0
    L:              real number > 0
    S:              integer > 0 (upper value for infinite product estimation)
    phi:            real number [0,1)     

    returns:        real number (approximation of infinite product of sinc(n phi^s/L) wrt s)  
    '''
    return np.prod([np.sinc((phi**s)*n/L) for s in range(S)])

def beta_char_func(n,L,S,phi,alpha):
    const = beta_normalising_constant(alpha)*np.sqrt(np.pi)*gamma(alpha)
    t = n*np.pi/L
    return np.prod([const*iv(alpha-0.5,np.j*t*(phi**s)/2)*(np.j*t*(phi**s))**(0.5-alpha) for s in range(S)])

def beta_normalising_constant(alpha):
    return gamma(2*alpha)/gamma(alpha)**2

def partial_alternating_sum(k,L,S,phi,f):
    '''
    description:    finite approximation of the alternating series sum((-1)^{n-1}char_func(n))

    params:
    k:              integer > 0 (number of terms included in the series)
    L:              real number > 0
    S:              integer > 0 (upper value for infinite product estimation)
    phi:            real number: [0,1)
    f:              characteristic function

    returns:        real number (approximation of the alternating series for n = 1,...,k+1)
    '''
    return np.sum([(-1)**(n-1) *f(n,L,S,phi) for n in range(1,k+1)])

def remainder(k,L,S,phi,f):
    '''
    description:    finds the difference between the kth and (k-1)th partial sums

    params:
    k:              integer > 1 (number of terms included in the series)
    L:              real number > 0
    S:              integer > 0 (upper value for infinite product estimation)
    phi:            real number: [0,1)
    f:              characteristic function

    returns:        real number (see description)
    '''
    return partial_alternating_sum(k,L,S,phi,f)-partial_alternating_sum(k-1,L,S,phi,f)

def geom_L(phi):
    '''
    infinite sum of geometric series. 
    phi: real number: [0,1)
    '''
    return 1/(1-np.abs(phi))

def approx_alternating_series(phi,S,tol,p,f,L):
    '''
    description:    approximates the alternating series sum((-1)^{n-1}char_func(n))

    S:              integer > 0 (upper value for infinite product estimation)
    tol:            real number (tolerance for convergence)
    phi:            real number: [0,1)
    p:              integer > 1 (required number of consecutive remainders less than tol)
    f:              characteristic function
    L:              function (defining L)

    returns:        the value of k that exited the while loop, remainders, partial sum for k value
    '''
    k = 1
    remainders = []
    partial_sums = []

    for k in range(1,p):
        remainders.append(remainder(k,L(phi),S,phi,f))
        partial_sums.append(partial_alternating_sum(k,L(phi),S,phi,f))
    while (np.abs(remainders[::-1][:p]) > tol).any():
        remainders.append(remainder(k,L(phi),S,phi,f))
        partial_sums.append(partial_alternating_sum(k,L(phi),S,phi,f))
        k +=1
    return k, remainders, partial_sums, partial_alternating_sum(k,L(phi),S,phi,f)
