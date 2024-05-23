import numpy as np
from scipy.special import hyp1f1

#--------------- STANDARD SINGLE RV CHARACTERISTIC FUNCTIONS --------------#
def beta_char_func(t,beta_params):
    '''
    description:    characteristic function of a symmetric beta distributed random
                    variable on the interval [-B,B] with parameter alpha.
    params:
    t:              real number > 0
    beta_params:    [alpha,B], real numbers > 0       
    '''
    # unpack
    alpha, B = beta_params

    return np.real_if_close((2*B)**(2*alpha-1)*np.exp(1j*B*t)*hyp1f1(alpha,2*alpha,2j*B*t))

def uniform_char_func(t,uniform_params):
    '''
    description:    characteristic function of a uniform distributed random variable
                    on the interval [-B,B]
    params:
    t:              real number > 0
    uniform_params: B, real number > 0
    '''
    # unpack
    B = uniform_params

    return np.sinc(B*t/np.pi)

#-------------------- INFINITE WEIGHTED SUM CHARACTERISTIC FUNCTIONS ----------------#

def infinite_weighted_sum_beta(t,S,c_s_func,c_s_params,beta_params):
    '''
    description:    finite approximation of the characteristic function of the infinite 
                    weighted sum of symmetric beta distributed random variables on the 
                    interval [-B,B] with parameter alpha
    params:
    t:              real number > 0
    S:              upper limit of product approximation. integer > 0
    c_s_func:       defines coefficients as a function of s, real function
    c_s_params:     real number(s)

    beta_params:    [alpha,B], real numbers > 0     
    '''
    return np.prod([beta_char_func(c_s_func(s,c_s_params)*t,beta_params) for s in range(S)])

'''
    L_func:         defines bound of infinite weighted sum, real function
    L_params:       real number(s)
'''