import numpy as np
from scipy.special import hyp1f1
from abc import ABC,abstractmethod

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

def infinite_weighted_sum_RV(t,S,c_s_func,c_s_params,cf_params,cf):
    '''
    description:    finite approximation of the characteristic function of the infinite 
                    weighted sum of IID random variables
    params:
    t:              real number > 0
    S:              upper limit of product approximation. integer > 0
    c_s_func:       defines coefficients as a function of s, real function
    c_s_params:     real number(s)
    cf_params:      real numbers > 0     
    '''
    return np.prod([cf(c_s_func(s,c_s_params)*t,cf_params) for s in range(S)])

#------------------- WEIGHTING SCHEMES ------------------#
# geometric

def c_s_geom(theta,s):
    # -1 < theta < 1
    return theta**s
def L_geom(theta):
    # -1 < theta < 1
    return 1/(1-np.abs(theta))

