import numpy as np

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

def partial_alternating_sum(k,L,S,phi,f):
    '''
    description:    finite approximation of the alternating series sum(char_func(n))

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
    k:              integer > 0 (number of terms included in the series)
    L:              real number > 0
    S:              integer > 0 (upper value for infinite product estimation)
    phi:            real number: [0,1)
    f:              characteristic function

    returns:        real number (see description)
    '''
    return partial_alternating_sum(k,L,S,phi,f)-partial_alternating_sum(k-1,L,S,phi,f)