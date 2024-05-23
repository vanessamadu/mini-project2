import numpy as np
from scipy.special import hyp1f1

def beta_char_func(t,beta_params):
    '''
    description:    characteristic function of a symmetric beta distributed random
                    variable on the interval [-B,B] with parameter alpha.
    params:
    t:              real number > 0
    beta_params:    [B,alpha], real numbers > 0       
    '''
    # alphabetically unpack
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
    # alphabetically unpack
    B = uniform_params
    return np.sinc(B*t/np.pi)