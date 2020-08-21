# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 18:23:57 2020

@author: ielham
"""

"""

Code to generate physically relevant spectra

"""

import numpy as np
from numpy import linalg as LA
from scipy import fftpack as ft
from scipy.fftpack import idct,dct
import FrechetMean as fm
from scipy import stats

# Some basic function

def power_law(m=136,index=-2.4):
    """
    Power law spectrum

    opt: m - array dimension
         index - spectral index of the PL
    """

    temp = np.exp(index*np.linspace(1,m,m)/m)

    return temp/np.linalg.norm(temp)

def Emission_law(m=136,pos=0.5,width=0.2):
    """
    Gaussian-shapes emission law

    opt: m - array dimension
         pos - position of the emission peak (between 0 and 1)
         width - width of the peak
    """

    temp = np.linspace(0,m-1,m)/m - pos
    temp = np.exp(-temp**2./width)

    return temp/np.linalg.norm(temp)



def VS_give_2D(x,t, higher_freq, activ_param, lower_freq=1,Energy=1):
    
    x[0,0]=0
    x=Energy*x/(LA.norm(x))#normalization
    
        
    res=ft.idct(ft.idct(x,axis=0, norm='ortho'), axis=1, norm='ortho')
    
    res+=abs(np.min(res))
    a=np.max(res)
    #res=res/np.max(res)#x non-negative
    return((res)/(1.*a))


def gen_VS_2D(rho, freq_cent, t, sigma):
    import scipy as sp
    
    wi=6*sigma #beyond this value, the gaussian will be almost null anyway
    
    e=gaussian_filter_2D(wi, sigma)
     
    d1=np.zeros((t, t))

    x=np.array(sp.stats.bernoulli.rvs(rho,size=(t, t)))
    
    if freq_cent>wi/2.:
        
        d1[freq_cent-wi/2:freq_cent+wi/2,freq_cent-wi/2:freq_cent+wi/2]=e*x[freq_cent-wi/2:freq_cent+wi/2,freq_cent-wi/2:freq_cent+wi/2]
    else:
        len_e=freq_cent+wi/2
        
        d1[:len_e,:len_e]=e[3*sigma-freq_cent:3*sigma-freq_cent+len_e, 3*sigma-freq_cent:3*sigma-freq_cent+len_e]*x[:len_e,:len_e]
    return(d1)



# Creating spectral variabilities


def Get_SV_PowerLaw(m=136,t=230,minmax_index=[2.4,3.4],var_vec=None):

    """
    Creating a single sv matrix with power law spectra

    opt: m - array dimension
         t - number of samples
         minmax_index - spectral index of the PL (min and max values)
         var_vec - if not None, provides the vector of variabilities for the power law (default is a simple cosine variation)
                   must be normalized to lie within the [0,1] interval

    output: SV - m x t matrix (spectral variabilities for a single source)
            Param - vector for the spectral index variabilities
    """

    if var_vec is None:
        var_vec = np.cos(np.pi*np.linspace(0,t-1,t)/t)

    Param = (minmax_index[1]-minmax_index[0])*var_vec+minmax_index[0]
    SV = np.zeros((m,t, t))

    for r1 in range(t):
        for r2 in range(t):
            SV[:,r1, r2] = power_law(m,index=Param[r1, r2])

    return SV,Param

# Creating spectral variabilities

def Get_SV_EmissionLaw(m=136,t=230,minmax_pos=[0.2,0.3],var_vec=None,width=0.2,freq=1): # we could actually the width as well if needed

    """
    Creating a single sv matrix with emission law spectra

    opt: m - array dimension
         t - number of samples
         minmax_index - spectral index of the PL (min and max values)
         var_vec - if not None, provides the vector of variabilities for the power law (default is a simple cosine variation)
                   must be normalized to lie within the [0,1] interval
         width - width of the Gaussian shape

    output: SV - m x t matrix (spectral variabilities for a single source)
            Param - vector for the emission peak position variabilities
    """

    if var_vec is None:
        var_vec = (np.cos(freq*np.pi*np.linspace(0,t-1,t)/t)+1)/2

    Param = (minmax_pos[1]-minmax_pos[0])*var_vec+minmax_pos[0]
    SV = np.zeros((m,t, t))

    for r1 in range(t):
        for r2 in range(t):
            SV[:,r1, r2] = Emission_law(m,pos=Param[r1, r2],width=width)

    return SV,Param


    
def gaussian_filter_2D(t_samp, sigma):
    x = np.linspace(1,t_samp,t_samp)-t_samp/2
    y = x[:,np.newaxis]

    kern = 1 / (sigma * np.sqrt(2*np.pi))*np.exp(-((x)**2+y**2)/(2*sigma**2))
 
    return (kern)
    

    
