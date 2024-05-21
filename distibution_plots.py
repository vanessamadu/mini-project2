import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gamma,iv


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
    alpha, S = params
    const = gamma(alpha+0.5)
    return np.prod([const*iv(alpha-0.5,1j*t/2)*(1j*t/4)**(0.5-alpha) for s in range(S)])