# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 19:39:25 2020

@author: ielham
"""

import scipy.io as sio

from astropy.io import fits
from astropy.visualization import simple_norm

import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm,LogNorm
from copy import deepcopy as dp 
import GMCA_star as gStar

import Chandra_init as Cinit
import FMean as FMean
import defLowFlux as LowF
import generation_physpecChandra as gen
#%%
def LGMCA(X, nb_sources, nb_obs, nb_patch=2, J=1, thres=500,pvalue=0,kend=3):
    
    Aref, Sref=Cinit.GMCA_global(kend,X, nb_sources, J)
    
    if pvalue>0:

        Xc, nc=LowF.defXcut(X, pvalue, nb_obs, most_rest=0)
    else:
        Xc, nc=LowF.defXcut_thres(X, thres, nb_obs, most_rest=1)
    
    print('size of Xc:',np.shape(Xc)[1])
    
    if np.shape(Xc)[1]>nb_patch:
    
        Patch=np.shape(Xc)[1]/nb_patch
    
        A, S, a, b=Cinit.Init(kend,Xc, Patch, nb_sources, J)
   
        Afref,Sfref=LowF.permu_gl_deux(Aref, Sref, a )
    
        Af=dp(Afref)
        Af[:,nc:-nc,nc:-nc,:]=dp(A)
        
        Sf=dp(Sfref)
        Sf[:,nc:-nc,nc:-nc]=dp(S)
    
    else:
        
        Patch=np.shape(X)[1]/nb_patch
        
        Af, Sf, a, b=Cinit.Init(kend,X, Patch, nb_sources, J)
        
        Afref,Sfref=LowF.permu_gl_deux(Aref, Sref, a )
    
    A0=dp(Aref)
    
    return(A0, Af, Sf, Patch)