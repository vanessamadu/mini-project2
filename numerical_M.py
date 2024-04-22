import numpy as np

def uniform_char_func(n,L,S,phi):
    '''
    description:    finite approximation of the characteristic function of an infinite geometrically weighted
                    sum of uniformly distributed random variables on the interval [-1,1].
    params:     
    n:              integer
    L:              real number
    S:              integer (upper value for infinite product estimation)
    phi:            parameter [0,1)     

    returns: Approximation of infinite product of sinc(n phi^s/L) wrt s     
    '''
    return np.prod([np.sinc((phi**s)*n/L) for s in range(S)])